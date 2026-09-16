import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.training.grad_cache import cached_contrastive_backward
from lumen.training.loss import clip_contrastive_loss


class _IdentityScaler:
    def scale(self, loss):
        return loss


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.3))

    def forward(self, image, text):
        return image @ text.T * self.logit_scale.exp().clamp(1.0, 20.0)


class _ToyModel(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(4, 7), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(7, 3))
        self.text = nn.Sequential(nn.Linear(5, 7), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(7, 3))
        self.contrastive_head = _Head()
        self.seen = []

    def forward(self, images, tokens):
        image = self.image(images)
        text = self.text(tokens["features"])
        self.seen.append((image.detach().clone(), text.detach().clone()))
        logits = self.contrastive_head(
            F.normalize(image, dim=-1), F.normalize(text, dim=-1))
        return {"logits": logits, "img_proj": image, "txt_proj": text}


def _batch():
    generator = torch.Generator().manual_seed(11)
    images = torch.randn(8, 4, generator=generator)
    text = torch.randn(8, 5, generator=generator)
    return images, {"features": text}


def _split(batch, size=2):
    images, tokens = batch
    return [(images[i:i + size],
             {name: value[i:i + size] for name, value in tokens.items()})
            for i in range(0, len(images), size)]


def test_gradient_cache_matches_full_batch_gradients():
    base = _ToyModel()
    full = copy.deepcopy(base)
    cached = copy.deepcopy(base)
    batch = _batch()

    out = full(*batch)
    full_loss = clip_contrastive_loss(out["logits"])
    full_loss.backward()

    result = cached_contrastive_backward(
        cached, _split(batch), torch.device("cpu"), False, _IdentityScaler())

    assert result.pairs == 8
    assert abs(result.loss - full_loss.item()) < 1e-7
    for (full_name, full_param), (cache_name, cache_param) in zip(
            full.named_parameters(), cached.named_parameters()):
        assert full_name == cache_name
        torch.testing.assert_close(cache_param.grad, full_param.grad,
                                   rtol=2e-5, atol=2e-6)


def test_gradient_cache_replays_identical_dropout_masks():
    model = _ToyModel(dropout=0.35).train()
    microbatches = _split(_batch(), size=4)
    torch.manual_seed(123)

    cached_contrastive_backward(
        model, microbatches, torch.device("cpu"), False, _IdentityScaler())

    assert len(model.seen) == 2 * len(microbatches)
    midpoint = len(microbatches)
    for cached, replayed in zip(model.seen[:midpoint], model.seen[midpoint:]):
        torch.testing.assert_close(cached[0], replayed[0], rtol=0, atol=0)
        torch.testing.assert_close(cached[1], replayed[1], rtol=0, atol=0)
