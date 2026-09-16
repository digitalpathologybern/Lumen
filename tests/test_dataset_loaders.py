import unittest

import numpy as np

from lumen.data.loaders import load_eval_set
from lumen.data.registry import DATASETS, dataset_dir


class DatasetLoaderTests(unittest.TestCase):
    def test_bach_loads_labeled_microscopy_images_when_available(self):
        spec = DATASETS["bach"]
        root = dataset_dir(spec)
        if not root.exists():
            self.skipTest("BACH assets are not available")

        ds = load_eval_set(spec)

        self.assertEqual(len(ds), 400)
        self.assertEqual(ds.class_names, spec.class_names)
        np.testing.assert_array_equal(np.unique(ds.labels), [0, 1, 2, 3])

    def test_wsss4luad_filters_to_pure_classification_patches(self):
        spec = DATASETS["wsss4luad"]
        if not (dataset_dir(spec) / "train.parquet").exists():
            self.skipTest("WSSS4LUAD assets are not available")

        ds = load_eval_set(spec)

        self.assertEqual(len(ds), 4693)
        self.assertEqual(ds.class_names, spec.class_names)
        np.testing.assert_array_equal(np.unique(ds.labels), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
