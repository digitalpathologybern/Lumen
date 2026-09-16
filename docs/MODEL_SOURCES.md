# Model sources

Every checkpoint this work evaluates, with the Hugging Face repository and the
exact commit it was taken at. **No weights are redistributed here.** They are
downloaded into `assets/models/`, which is gitignored, so this file is how the
same weights are obtained. Populating `assets/models/` from the two tables below
is enough to run every stage in `REPRODUCE.md` except training.

Third-party checkpoints downloaded 2026-07-02. Example images from model cards
were skipped.

## Lumen

Lumen is `lumen_retrain_quilt`, trained on QUILT-1M. It ships in
`assets/models/lumen_quilt_r4/` and is also published at
<https://huggingface.co/digitalpathologybern/Lumen>; the copy here is
byte-identical, and the digest below is what that repo records in its own
`SHA256SUMS`. These weights are CC BY-NC 4.0, the same as the rest of the
repository.

What ships is the **alignment only**: rank-4 LoRA adapters on both encoders, the
two projection heads and the learned logit scale, 2,984,961 parameters, about
12 MB. `lumen/utils/model.py` assembles it onto the backbones at load time and
refuses a partial alignment. The backbones are not here; see "Backbones" below.

The corpus ablation ships alongside it, in the same layout: `model.safetensors`
and `checkpoint_best_model_cfg.json` per directory.

| Asset directory | Registry key | sha256 of `model.safetensors` |
| --- | --- | --- |
| `lumen_quilt_r4/` | `lumen_retrain_quilt` (= `PAPER_CHECKPOINT`) | `3559c771a769286939a0916040e62a48cc25afe7c8bcd887ab3bebd5914c3c89` |
| `lumen_pathgen_r4/` | `lumen_retrain_pathgen` | `4dd7f87226988273c485aedffd5d97ec82749482a68e749a17113cb93378d893` |

Training writes a full ~3 GB checkpoint under `outputs/training/` instead, which
embeds the frozen backbones and therefore cannot be published. To evaluate your
own run, see `REPRODUCE.md`.

**The adaptation ablation is not distributable.** The full-finetuning control
trains with LoRA off and both backbones unfrozen, so what it learns is the
modified Virchow2 and BioMedBERT weights: 8.3 GB, no adapter tensors to extract,
and a derivative of a model whose licence forbids derivatives. It is not in the
registry and its numbers are reported in the paper only.

## Third-party baselines

| Asset directory | Source | Status |
| --- | --- | --- |
| `clip_vit_b16/` | `openai/clip-vit-base-patch16` @ `57c216476eefef5ab752ec549e440a49ae4ae5f3` | Downloaded |
| `clip_vit_b32/` | `openai/clip-vit-base-patch32` @ `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | Downloaded |
| `clip_vit_l14/` | `openai/clip-vit-large-patch14` @ `32bd64288804d66eefd0ccbe215aa642df71cc41` | Downloaded |
| `plip/` | `vinid/plip` @ `67ade53ddd32195868f422585f72698ef5d15094` | Downloaded |
| `pathclip_base/` | `jamessyx/pathclip` @ `1b5a57a5126d134b936dcc0b3b288bdcd630ab8c` | Downloaded |
| `quiltnet_b16/` | `wisdomik/QuiltNet-B-16` @ `a5a588b6cec62d610bfc25f64dfe1df30fa6af98` | Downloaded |
| `quiltnet_b32/` | `wisdomik/QuiltNet-B-32` @ `8ce77289ce35a90b2f1db1137dfa4bc2df175e33` | Downloaded |
| `pathgen_l14_hf/` | `jamessyx/pathgenclip-vit-large-patch14-hf` @ `33af8295d6e42422f027a61bd1d960ab258488f7` | Downloaded |
| `conch/` | `MahmoodLab/conch` @ `f9ca9f877171a28ade80228fb195ac5d79003357` | Downloaded; gated HF repo accessible to this account |
| `biomedclip_pubmedbert/` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` @ `9f341de24bfb00180f1b847274256e9b65a3a32e` | Downloaded |
| `biomedbert_base/` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` @ `e1354b7a3a09615f6aba48dfad4b7a613eef7062` | Downloaded; KEEP's text skeleton is built from this snapshot |
| `pathgen_b16/` | `jamessyx/PathGen-CLIP` @ `fb80c40436e3d8ba7f624ae17117fce3a2d63471` | Downloaded; gated HF repo accessible to this account |

## Backbones

Lumen's two backbones are **not** loaded from `assets/models/`. They are reached
by Hub id and resolve through the Hugging Face cache:

| Backbone | Hub id | Reached from |
| --- | --- | --- |
| Virchow2 | `paige-ai/Virchow2` @ `3158645804b69e3f3bc4439d4116edddf0840a72` | `timm.create_model("hf-hub:paige-ai/Virchow2")`, `lumen/utils/backbones.py` |
| BioMedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` @ `e1354b7a3a09615f6aba48dfad4b7a613eef7062` | `AutoModel.from_pretrained(...)`, `lumen/utils/backbones.py` |

Putting them under `assets/models/` does nothing; nothing reads that path for
these two. Virchow2 is gated, so accept its licence on the Hub and
`huggingface-cli login` first. Its terms require each user of an organisation to
register individually, which is part of why the weights are not vendored here.

Both are pinned to the commits above in `lumen/utils/backbones.py`, so a run
fetches those revisions rather than whatever `main` points at, and
`tests/test_backbone_pins.py` fails if the code and this file disagree. Nothing
else in the pipeline reaches the Hub: the baselines load from `assets/models/`
and Lumen is in the repository.

Virchow2's licence also requires attribution, so cite it if you use this work:

```bibtex
@article{zimmermann2024virchow2,
  title={Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology},
  author={Eric Zimmermann and Eugene Vorontsov and Julian Viret and Adam Casson
          and Michal Zelechowski and George Shaikovski and Neil Tenenholtz and
          James Hall and Thomas Fuchs and Nicolo Fusi and Siqi Liu and
          Kristen Severson},
  journal={arXiv preprint arXiv:2408.00738},
  year={2024},
}
``` Then prime the cache **once with
`HF_HUB_OFFLINE` unset**, because the reproduction runs set it and offline mode
only reads what is already cached:

```bash
huggingface-cli download paige-ai/Virchow2
huggingface-cli download microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

Adjacent baseline checked but not downloaded:

- `Iyyakutti/CPLIP` @ `7c07c0fc98e42c90f3741b04f3bc8c8272294f42`:
  the Hugging Face repo currently contains config/tokenizer files but no weight
  artifact.
