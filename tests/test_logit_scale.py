"""Every adapter must expose the temperature its checkpoint was trained with.

CONCH and Lumen were both constructed with ``logit_scale=None``, so
``resolve_scale`` silently fell back to CLIP's 1/0.07 = 14.286 -- while the
CLIP-family models used their learned ~100. Probabilities, and therefore any
threshold on them, were not comparable across the benchmark.

Argmax metrics are invariant to the scale, so the reported patch and WSI tables
never moved (they do not use it at all). What it corrupted was the probability
maps, the QuPath overlays, and any threshold-tuning protocol.
"""

import unittest

import torch

from lumen.zeroshot import DEFAULT_TEMPERATURE, resolve_scale


class ResolveScaleTests(unittest.TestCase):
    def test_none_falls_back_to_clip_default(self):
        self.assertAlmostEqual(resolve_scale(None, None), DEFAULT_TEMPERATURE)

    def test_learned_scale_is_used_when_present(self):
        self.assertAlmostEqual(resolve_scale(56.35, None), 56.35)

    def test_explicit_temperature_beats_the_learned_scale(self):
        self.assertAlmostEqual(resolve_scale(56.35, 1.0), 1.0)

    def test_a_zero_scale_falls_back_rather_than_collapsing(self):
        # `logit_scale or DEFAULT`: 0.0 is falsy, and a 0 scale would make every
        # class equiprobable. Falling back is the safe reading.
        self.assertAlmostEqual(resolve_scale(0.0, None), DEFAULT_TEMPERATURE)


class LumenHeadScaleTests(unittest.TestCase):
    """The ContrastiveHead clamps to [1, 20] on every forward, so the clamp is part
    of the model. The loader must read the *effective* value, not the raw exp."""

    def test_reader_applies_the_heads_own_clamp(self):
        from lumen.models.loaders import _lumen_logit_scale
        from lumen.utils.model import ContrastiveHead

        class Stub:
            pass

        head = ContrastiveHead()
        # raw exp would be e^5 = 148, but the head clamps to 20
        with torch.no_grad():
            head.logit_scale.copy_(torch.tensor(5.0))
        m = Stub()
        m.contrastive_head = head
        self.assertAlmostEqual(_lumen_logit_scale(m), 20.0, places=4)

    def test_default_init_is_the_clip_temperature(self):
        from lumen.models.loaders import _lumen_logit_scale
        from lumen.utils.model import ContrastiveHead

        class Stub:
            pass

        m = Stub()
        m.contrastive_head = ContrastiveHead()      # init 1/0.07
        self.assertAlmostEqual(_lumen_logit_scale(m), DEFAULT_TEMPERATURE, places=3)

    def test_missing_head_returns_none_rather_than_raising(self):
        # The HF export drops the ContrastiveHead; it must fall back, not crash.
        from lumen.models.loaders import _lumen_logit_scale

        class Stub:
            pass

        self.assertIsNone(_lumen_logit_scale(Stub()))


if __name__ == "__main__":
    unittest.main()
