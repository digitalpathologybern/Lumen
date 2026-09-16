import unittest

import numpy as np

from lumen.zeroshot import zero_shot_probs


class ZeroShotProbabilityTests(unittest.TestCase):
    def test_single_prompt_uses_sigmoid_not_trivial_softmax(self):
        image_emb = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)

        def encode_text(texts):
            self.assertEqual(texts, ["tumor"])
            return np.array([[1.0, 0.0]], dtype=np.float32)

        classes, probs = zero_shot_probs(
            image_emb,
            encode_text,
            {"tumor": ["tumor"]},
            temperature=1.0,
        )

        self.assertEqual(classes, ["tumor"])
        self.assertEqual(probs.shape, (2, 1))
        self.assertGreater(probs[0, 0], 0.5)
        self.assertLess(probs[1, 0], 0.5)

    def test_multi_prompt_probabilities_still_sum_to_one(self):
        image_emb = np.array([[1.0, 0.0]], dtype=np.float32)

        def encode_text(texts):
            lookup = {
                "tumor": np.array([1.0, 0.0], dtype=np.float32),
                "benign": np.array([0.0, 1.0], dtype=np.float32),
            }
            return np.stack([lookup[text] for text in texts])

        classes, probs = zero_shot_probs(
            image_emb,
            encode_text,
            {"tumor": ["tumor"], "benign": ["benign"]},
            temperature=1.0,
        )

        self.assertEqual(classes, ["tumor", "benign"])
        self.assertAlmostEqual(float(probs.sum(axis=1)[0]), 1.0)
        self.assertGreater(probs[0, 0], probs[0, 1])


if __name__ == "__main__":
    unittest.main()
