import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from .lora import wrap_with_lora


class ResBlock(nn.Module):
    """
    Residual Block (min 3 layers) for deep nets with modular compression
    """
    def __init__(
        self,
        in_dim: int,
        span: int = 3,
        decay: float = 0.5,
        act: nn.Module = nn.GELU,
        min_dim: int = 512
    ):
        super().__init__()
        assert span >= 3, "span must be bigger or equal to 3"
        dims = [in_dim]
        # build dims, clamping at min_dim each time
        for _ in range(span):
            raw = int(dims[-1] * decay)
            dims.append(max(min_dim, raw))
        # layers: Linear → Act → LayerNorm
        layers = []
        for i in range(span):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                act(),
                nn.LayerNorm(dims[i+1])
            ]
        self.net = nn.Sequential(*layers)
        final_dim = dims[-1]
        self.skip_proj = nn.Linear(in_dim, final_dim) \
                         if in_dim != final_dim else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip_proj(x)
        out = self.net(x)
        return self.alpha * res + (1 - self.alpha) * out



class Projector(nn.Module):
    """
    Embedding projector with modular blocks for early, middle or late compression of dimensionality
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        decay: float = 0.5,
        span: int = 3,
        n_blocks: int = 3,
        compression: str = "late"
    ):
        super().__init__()
        assert compression in ("simple", "early", "middle", "late"), \
            f"compression must be one of early/middle/late, got {compression}"
        assert 0 < decay < 1, f"decay must be in (0,1), got {decay}"
        assert n_blocks >= 2, f"n_blocks must be ≥2, got {n_blocks}"

        self.blocks = nn.ModuleList()
        current_dim = in_dim

        # compute target compressed dim, but floor at out_dim
        raw_c = int(in_dim * decay**span)
        compressed_dim = max(out_dim, raw_c)

        mid = n_blocks // 2
        for i in range(n_blocks):
            do_compress = (
                compression == "early"  and i == 0 or
                compression == "middle" and i == mid or
                compression == "late"   and i == n_blocks - 1
            )
            if do_compress:
                # compress this block, clamped at out_dim
                self.blocks.append(
                    ResBlock(current_dim, span, decay, min_dim=out_dim)
                )
                current_dim = compressed_dim
            else:
                # identity (decay=1.0), still respect min_dim
                self.blocks.append(
                    ResBlock(current_dim, span, decay=1.0, min_dim=out_dim)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ContrastiveHead(nn.Module):
    def __init__(self, logit_scale_init: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1/logit_scale_init).log())

    def forward(self, img_proj: torch.Tensor, txt_proj: torch.Tensor) -> torch.Tensor:

        # With this clamping, temperature is always ~14.5
        # scale = self.logit_scale.exp().clamp(1e-4, 100.0)
        # This new scaling enforces temperature in range [0.05,1.0]
        scale = self.logit_scale.exp().clamp(1.0, 20.0)
        logits = img_proj @ txt_proj.T
        return logits * scale


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        heads: int = 8,
        attn_dropout: float = 0.1,
        out_dropout: float = 0.2
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attn: bool = False
    ) -> Any:
        B, Lq, D = query.shape
        _, Lk, _ = key.shape
        _, Lv, _ = value.shape

        # 1) project
        q = self.to_q(query).view(B, self.heads, Lq, self.head_dim)
        k = self.to_k(key)  .view(B, self.heads, Lk, self.head_dim)
        v = self.to_v(value).view(B, self.heads, Lv, self.head_dim)

        # 2) scaled dot-attention
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v

        # 3) combine
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.to_out(out)
        out = self.out_dropout(out)

        # 4) residual + norm
        out = self.layer_norm(out + query)

        return (out, attn) if return_attn else out

def get_layers(
    in_dim: int,
    out_dim: int = 512,
    decay: float = 0.5,
    act_func: nn.Module = nn.GELU(),
    dropout: float = 0.2,
) -> nn.Sequential:
    assert 0 < decay < 1, f"decay must be in [0, 1], got {decay}"

    dims = [in_dim]
    first_decay = int(in_dim * decay)

    if first_decay <= out_dim:
        hidden = (in_dim + out_dim) // 2
        dims += [hidden, out_dim]
    else:
        while True:
            nxt = int(dims[-1] * decay)
            if nxt <= out_dim:
                break
            dims.append(nxt)
        dims.append(out_dim)

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.BatchNorm1d(dims[i+1]))
        if i < len(dims) - 2:
            layers.append(act_func)
            layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)

def get_hidden(in_dim: int, out_dim: int, hidden_layers: int) -> list:
    return [round(in_dim - i * (in_dim - out_dim) / hidden_layers) for i in range(hidden_layers + 1)]

def simple_layers(in_dim: int, out_dim: int, hidden_layers: int,
                  gated: bool = False,
                  pre_gating: bool = True,
                  post_gating: bool = False,
                  use_residuals: bool = False,
                  gate_at: int = 1  # zero-based index
                  ) -> nn.Sequential:
    # 1) build int dims
    dims = [
        int(round(in_dim + i*(out_dim-in_dim)/hidden_layers))
        for i in range(hidden_layers+1)
    ]
    layers = []

    for i in range(hidden_layers):
        d_in, d_out = dims[i], dims[i+1]

        # pre-gating
        if gated and pre_gating and i == gate_at:
            layers += [nn.LayerNorm(d_in),
                       GateUnit(d_in, use_residuals)]

        layers += [nn.Linear(d_in, d_out), nn.GELU()]

        # post-gating
        if gated and post_gating and i == gate_at:
            layers += [nn.LayerNorm(d_out),
                       GateUnit(d_out, use_residuals)]

    # final projection
    layers += [nn.Linear(dims[-1], out_dim), nn.LayerNorm(out_dim)]
    return nn.Sequential(*layers)

def get_direct_head(in_dim:int, out_dim:int = 512):
    projection = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim)
    )
    
    return projection
    

# class GateUnit(nn.Module):
#     def __init__(self, dim: int, use_residuals: bool = False):
#         super().__init__()
#         self.use_residuals = use_residuals
#         self.gate = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         g = self.gate(x)       # (…, dim)
#         out = x * g            # elementwise
#         if self.use_residuals:
#             out = out + x
#         return out

class GateUnit(nn.Module):
    def __init__(self, dim: int, use_residuals: bool = False,
                 alpha_init: float = 0.5):
        super().__init__()
        self.use_residuals = use_residuals
        # learnable scalar for residual interpolation
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        # gating linear with bias
        self.linear = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute gate with sigmoid activation
        g = torch.sigmoid(self.linear(x))    # (batch, dim)
        if self.use_residuals:
            # fused interpolation: m = α + (1-α)*g
            m = self.alpha + (1.0 - self.alpha) * g
            return x * m
        # without residuals: simple gating
        return x * g


class Model(nn.Module):
    def __init__(
        self,
        txt_encoder: nn.Module,
        img_encoder: nn.Module,
        text_backbone: str,
        vision_backbone: str,
        proj_dim: int = 512,
        decay: float = 0.5,
        span: int = 3,
        n_blocks: int = 3,
        img_comp: str = "linear",
        txt_comp: str = "linear",
        img_hidden: int = 5,
        txt_hidden: int = 3,
        gated: bool = False,
        pre_gating: bool = False,
        post_gating: bool = True,
        use_residuals: bool = False,
        attention: bool = False,
        lora:bool = False,
        lora_mode:str = 'none',
        r:int = 4,
        lora_dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        assert img_comp in ("simple", "linear", "direct", "gated", "early", "middle", "late"), \
            f"compression must be one of 'linear', 'early', 'middle', 'late', 'direct', got {img_comp}"
        assert txt_comp in ("simple", "linear", "direct", "gated", "early", "middle", "late"), \
            f"compression must be one of 'linear', 'early', 'middle', 'late', 'direct', got {txt_comp}"

        self.txt_encoder = txt_encoder
        self.img_encoder = img_encoder
        self.use_attn = attention
        self.text_backbone = text_backbone
        self.vision_backbone = vision_backbone
        self.lora_mode = lora_mode
        self.r = r
        self.lora_dropout=lora_dropout

        # infer text hidden dim
        if hasattr(txt_encoder, "config") and hasattr(txt_encoder.config, "hidden_size"):
            txt_hid = txt_encoder.config.hidden_size
        else:
            raise ValueError("Cannot infer txt_encoder hidden_size")

        # infer image hidden dim & forward fn
        if hasattr(img_encoder, "config") and hasattr(img_encoder.config, "hidden_size"):
            img_hid = img_encoder.config.hidden_size
            self._img_forward = lambda x: img_encoder(x).last_hidden_state
        elif hasattr(img_encoder, "embed_dim") and hasattr(img_encoder, "forward_features"):
            img_hid = img_encoder.embed_dim
            self._img_forward = img_encoder.forward_features
        else:
            raise ValueError("Cannot infer img_encoder hidden size")

        self.img_in_dim = img_hid * 2
        self.txt_in_dim = txt_hid

        # print(f"Image compression found: {img_comp}")
        # print(f"Text compression found: {txt_comp}")

        # choose projection nets (contrastive adaptors)
        if img_comp == "linear":
            self.img_proj_net = get_layers(
                in_dim=self.img_in_dim,
                out_dim=proj_dim,
                decay=decay
            )

        elif img_comp == "simple":
            self.img_proj_net = simple_layers(in_dim=self.img_in_dim, out_dim=proj_dim, hidden_layers=img_hidden,
                                             gated=gated, pre_gating=pre_gating, post_gating=post_gating, use_residuals=use_residuals)
        
        elif img_comp == 'direct':
            self.img_proj_net = get_direct_head(in_dim=self.img_in_dim, out_dim=proj_dim)

        else:
            self.img_proj_net = Projector(
                in_dim=self.img_in_dim,
                out_dim=proj_dim,
                decay=decay,
                span=span,
                n_blocks=n_blocks,
                compression=img_comp
            )


        if txt_comp == "linear":
            self.txt_proj_net = get_layers(
                in_dim=self.txt_in_dim,
                out_dim=proj_dim,
                decay=decay
            )

        elif txt_comp == "simple":
            self.txt_proj_net = simple_layers(in_dim=self.txt_in_dim, out_dim=proj_dim, hidden_layers=txt_hidden,
                                             gated=gated, pre_gating=pre_gating, post_gating=post_gating, use_residuals=use_residuals)
        
        elif txt_comp == 'direct':
            self.txt_proj_net = get_direct_head(in_dim=self.txt_in_dim, out_dim=proj_dim)    
        
        else:
            self.txt_proj_net = Projector(
                in_dim=self.txt_in_dim,
                out_dim=proj_dim,
                decay=decay,
                span=span,
                n_blocks=n_blocks,
                compression=txt_comp
            )

        # optional cross‐attention
        if attention:
            self.cross_attn = CrossAttention(dim=proj_dim, heads=num_heads)
            
        if lora:
            self.inject_lora()

        # contrastive head
        self.contrastive_head = ContrastiveHead()

    def forward(
        self,
        images: torch.Tensor,
        tokens: Dict[str, torch.Tensor],
        attention: bool = False
    ) -> Dict[str, Any]:
        B = images.size(0)

        txt_feats = self.txt_encoder(**tokens).last_hidden_state
        img_feats = self._img_forward(images)

        img_cls = torch.cat([img_feats[:,0], img_feats[:,1:].mean(dim=1)], dim=-1)
        txt_cls = txt_feats[:,0]

        img_proj = self.img_proj_net(img_cls)
        txt_proj = self.txt_proj_net(txt_cls)

        if not attention:
            img_norm = F.normalize(img_proj, dim=-1)
            txt_norm = F.normalize(txt_proj, dim=-1)
            return {"logits": self.contrastive_head(img_norm, txt_norm),
                    "img_proj": img_proj, "txt_proj": txt_proj}

        i2t, attn_i2t = self.cross_attn(
            query=img_proj.unsqueeze(1),
            key=txt_proj.unsqueeze(1),
            value=txt_proj.unsqueeze(1),
            return_attn=True
        )
        t2i, attn_t2i = self.cross_attn(
            query=txt_proj.unsqueeze(1),
            key=img_proj.unsqueeze(1),
            value=img_proj.unsqueeze(1),
            return_attn=True
        )
        i2t, t2i = i2t.squeeze(1), t2i.squeeze(1)

        i2t_norm = F.normalize(i2t, dim=-1)
        t2i_norm = F.normalize(t2i, dim=-1)
        txt_norm = F.normalize(txt_proj, dim=-1)
        img_norm = F.normalize(img_proj, dim=-1)

        return {
            "logits_img": self.contrastive_head(i2t_norm, txt_norm),
            "logits_txt": self.contrastive_head(img_norm, t2i_norm),
            "attn_i2t": attn_i2t,
            "attn_t2i": attn_t2i,
            "img_proj": img_proj,
            "txt_proj": txt_proj,
        }
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        img_feats = self._img_forward(images)
        img_cls = torch.cat([img_feats[:, 0], img_feats[:, 1:].mean(dim=1)], dim=-1)
        return self.img_proj_net(img_cls)
    
    def encode_text(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode and project a batch of tokenized text inputs.

        Parameters
        ----------
        tokens : Dict[str, torch.Tensor]
            A dict of token-ids, attention masks, etc., as returned by a HuggingFace tokenizer.

        Returns
        -------
        torch.Tensor
            The projected text embeddings (shape [batch, proj_dim]).
        """
        # 1) raw encoder outputs: [batch, seq_len, hidden_dim]
        txt_feats = self.txt_encoder(**tokens).last_hidden_state

        # 2) take the [CLS] token representation
        txt_cls = txt_feats[:, 0]  # shape [batch, hidden_dim]

        return self.txt_proj_net(txt_cls)
    
    def get_img_attention(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute and return image encoder attention maps.

        Parameters
        ----------
        images : torch.Tensor
            Input image batch tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Attention maps from each encoder layer: each tensor has shape
            (batch_size, num_heads, seq_len, seq_len).
        """
        try:
            outputs = self.img_encoder(images, output_attentions=True)
        except TypeError:
            raise NotImplementedError("Image encoder does not support output_attentions flag.")

        if not hasattr(outputs, "attentions"):
            raise NotImplementedError("Image encoder did not return attentions.")

        return outputs.attentions  # tuple of (batch, heads, seq_len, seq_len)

    def get_text_attention(self, tokens: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Compute and return text encoder attention maps.

        Parameters
        ----------
        tokens : Dict[str, torch.Tensor]
            Tokenized input dict for HuggingFace text encoder.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Attention maps from each encoder layer: shape (batch, heads, seq_len, seq_len).
        """
        # call encoder with attentions
        outputs = self.txt_encoder(**tokens, output_attentions=True)
        if not hasattr(outputs, "attentions"):
            raise NotImplementedError("Text encoder did not return attentions. Ensure output_attentions=True.")
        return outputs.attentions
    
    def inject_lora(self):
        if self.lora_mode == 'text_encoder' or self.lora_mode == 'both_encoders':
            self.txt_encoder = wrap_with_lora(self.txt_encoder,
                                               self.r,
                                               self.lora_dropout,
                                               self.text_backbone,
                                               enc_type='text')
        
        if self.lora_mode == 'image_encoder' or self.lora_mode == 'both_encoders':
            self.img_encoder = wrap_with_lora(self.img_encoder,
                                               self.r,
                                               self.lora_dropout,
                                               self.vision_backbone,
                                               enc_type='image')


def compute_temperature(head: nn.Module) -> float:
    return head.logit_scale.exp().item()


#: Parameters in the released alignment: rank-4 LoRA on both encoders, the two
#: projection heads and the learned logit scale. A checkpoint that carries only
#: these is publishable; a full training checkpoint is ~2.8 GB because it also
#: holds the frozen Virchow2 and BioMedBERT weights, which are not ours to
#: redistribute.
ALIGNMENT_PARAMS = 2_984_961

#: Tensor-name fragments that belong to the alignment rather than a backbone.
_ALIGNMENT_TAGS = ("lora_", "img_proj", "txt_proj", "logit_scale")


def _read_state_dict(path: str, device) -> Dict[str, "torch.Tensor"]:
    """Read a ``.safetensors`` alignment or a ``.pt`` training checkpoint."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(path), device=str(device))
    blob = torch.load(path, map_location=device)
    return blob["model"] if isinstance(blob, dict) and "model" in blob else blob


def _load_alignment(model: nn.Module, state_dict: Dict[str, Any], src: str) -> None:
    """Load a full checkpoint, or an alignment-only one onto fresh backbones.

    The released adapter omits every backbone tensor, so ``strict=True`` would
    reject it. Loading non-strictly instead would accept a file that is missing
    half the alignment and score plausibly, so the missing keys are required to
    be backbone tensors only and the loaded alignment is counted.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"{src}: {len(unexpected)} tensors do not belong to this "
            f"architecture, first few: {list(unexpected)[:5]}")
    if not missing:
        return

    stranded = [k for k in missing if any(t in k for t in _ALIGNMENT_TAGS)]
    if stranded:
        raise RuntimeError(
            f"{src}: {len(stranded)} alignment tensors are absent, first few: "
            f"{stranded[:5]}. This is not a complete alignment.")

    loaded = sum(v.numel() for k, v in state_dict.items()
                 if any(t in k for t in _ALIGNMENT_TAGS))
    if loaded != ALIGNMENT_PARAMS:
        raise RuntimeError(
            f"{src}: loaded {loaded:,} alignment parameters, expected "
            f"{ALIGNMENT_PARAMS:,}. The backbones are untouched by training, so "
            f"a mismatch means this is not the released alignment.")


def load_model(state_dict_path: str, cfg_path: str, device) -> Model:
    """
    Load a model from a state dict and configuration file.
    """
    import json
    from pathlib import Path
    from .backbones import load_text_backbone, load_vision_backbone
    print("[MODEL] Loading model from config file")
    cfg: Dict = json.loads(Path(cfg_path).read_text())

    txt_name = cfg.pop("text_backbone")
    vsn_name = cfg.pop("vision_backbone")
    txt_enc, tokenizer = load_text_backbone(txt_name)
    vsn_enc, _, img_tfms, _, _ = load_vision_backbone(vsn_name)

    remap = {
        "image_hidden":      "img_hidden",
        "text_hidden":       "txt_hidden",
        "image_compression": "img_comp",
        "text_compression":  "txt_comp",
        "number_blocks":     "n_blocks",
        "proj_dimensions":   "proj_dim"
    }
    cfg = {remap.get(k,k): v for k,v in cfg.items() if k!="architecture"}
    cfg.update(
        txt_encoder=txt_enc.eval().to(device),
        img_encoder=vsn_enc.eval().to(device),
        text_backbone=txt_name,
        vision_backbone=vsn_name,
    )
    model = Model(**cfg)

    state_dict = _read_state_dict(state_dict_path, device)
    _load_alignment(model, state_dict, state_dict_path)
    
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    return model, tokenizer, img_tfms