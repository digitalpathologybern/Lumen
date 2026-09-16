from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from lumen.inference.config import apply_inference_overrides, resolve_table_paths
from lumen.inference.records import (
    combined_parent_tag,
    group_records,
    load_slide_records,
    normalize_slide_record,
)


class RecordHelpersTest(unittest.TestCase):
    def test_normalize_slide_record_coerces_common_numeric_fields(self):
        row = normalize_slide_record({
            "slide": "s1",
            "group": "crc",
            "gt": "1.0",
            "pred": "0",
            "max_prob": "0.75",
            "avg_prob": "0.25",
            "n_tiles": "42",
        })

        self.assertEqual(row["gt"], 1)
        self.assertEqual(row["pred"], 0)
        self.assertEqual(row["n_tiles"], 42)
        self.assertAlmostEqual(row["max_prob"], 0.75)
        self.assertAlmostEqual(row["avg_prob"], 0.25)

    def test_load_group_and_tag_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "internal" / "internal_checkpoint.csv"
            path.parent.mkdir()
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "slide", "filepath", "group", "gt", "pred",
                        "max_prob", "avg_prob", "n_tiles",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "slide": "a", "filepath": "/a.svs", "group": "crc",
                    "gt": "1", "pred": "1", "max_prob": "0.9",
                    "avg_prob": "0.6", "n_tiles": "7",
                })
                writer.writerow({
                    "slide": "b", "filepath": "/b.svs", "group": "breast",
                    "gt": "0", "pred": "0", "max_prob": "0.2",
                    "avg_prob": "0.1", "n_tiles": "5",
                })

            rows = load_slide_records([path])
            grouped = group_records(rows)

            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(grouped), ["breast", "crc"])
            self.assertEqual(combined_parent_tag([path]), "internal")


class ConfigHelpersTest(unittest.TestCase):
    def test_apply_overrides_returns_copy(self):
        cfg = {
            "inference": {"resolution": 4.0, "tile_size": 224},
            "tables": ["a.xlsx"],
        }

        updated = apply_inference_overrides(cfg, resolution=2.0)

        self.assertEqual(cfg["inference"]["resolution"], 4.0)
        self.assertEqual(updated["inference"]["resolution"], 2.0)
        self.assertEqual(updated["inference"]["tile_size"], 224)

    def test_resolve_table_paths_prefers_cli_tables(self):
        cfg = {"tables": ["from-config.xlsx"]}
        self.assertEqual(resolve_table_paths(cfg), [Path("from-config.xlsx")])
        self.assertEqual(resolve_table_paths(cfg, ["from-cli.xlsx"]), [Path("from-cli.xlsx")])


if __name__ == "__main__":
    unittest.main()
