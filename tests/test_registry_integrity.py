"""Model/dataset registry integrity.

Encodes two invariants that previously broke results: Lumen must be keyed
``lumen`` everywhere, and the patch benchmark is exactly the reported datasets.

The model has been renamed twice, ``vislang`` then ``odyssey``. Each time an
alias lingered somewhere and silently dropped the model from merged summaries,
so both dead keys are asserted absent rather than only the most recent one.
"""

import unittest

from lumen.benchmark import prompts as pb
from lumen.data.registry import DATASETS
from lumen.models.registry import MODELS, PAPER_CHECKPOINT, RETRAIN_MODELS

BENCHMARK_DATASETS = {"lc25000", "osteo", "pcam", "sicap", "mhist",
                      "databiox", "bach", "nct_crc", "wsss4luad"}
BENCHMARK_MODELS = {
    "clip_b16", "clip_b32", "clip_l14", "plip", "quilt_b16", "quilt_b32",
    "pathgen_l14", "pathgen_b16", "pathclip", "biomedclip", "conch", "keep",
}
# Architecture variants trained on a stated corpus or schedule, registered so the
# benchmark can run them. Note that ``lumen_retrain_quilt`` is the paper's
# reported checkpoint despite the ``retrain`` prefix: an earlier comment here
# claimed the opposite, which is a costly thing to believe, because ``lumen``
# scores close enough to it that reporting the wrong one is not obvious.
RETRAIN_KEYS = {"lumen_retrain_quilt", "lumen_retrain_pathgen"}


class ModelRegistryTests(unittest.TestCase):
    def test_no_superseded_alias_or_unobtainable_run(self):
        self.assertIn(PAPER_CHECKPOINT, MODELS)
        # The bare "lumen" key was an earlier run of the same configuration,
        # loaded from a shared folder no reproducer can obtain. It is not an
        # alias that can come back harmlessly: it scored close enough to the
        # reported run to be mistaken for it.
        self.assertNotIn("lumen", MODELS)
        for dead in ("vislang", "odyssey"):
            self.assertNotIn(dead, MODELS)
            self.assertFalse([k for k in MODELS if k.startswith(dead)],
                             f"registry still carries a {dead!r}-prefixed key")

    def test_every_model_is_obtainable_without_a_retrain(self):
        """Every checkpoint is either a local asset or a named Hub repo.

        The Lumen specs used to carry an absolute path into outputs/training/,
        so they resolved to a directory that exists only after a retrain and is
        named nowhere in the docs. Nothing reported was loadable from a clone.
        """
        from lumen.models.registry import model_dir
        from lumen.paths import models_root

        root = models_root()
        for key, spec in MODELS.items():
            self.assertIn(root, model_dir(spec).parents,
                          f"{key} resolves outside assets/models/")

    def test_dict_keys_match_spec_keys_and_are_unique(self):
        for key, spec in MODELS.items():
            self.assertEqual(key, spec.key, f"registry key {key!r} != spec.key {spec.key!r}")
        self.assertEqual(len(MODELS), len(set(MODELS)))

    def test_exactly_the_benchmark_models_plus_named_retrains(self):
        self.assertEqual(set(MODELS), BENCHMARK_MODELS | RETRAIN_KEYS)

    def test_retrains_are_declared_and_disjoint_from_reported_models(self):
        self.assertEqual(set(RETRAIN_MODELS), RETRAIN_KEYS)
        self.assertFalse(set(RETRAIN_MODELS) & BENCHMARK_MODELS)
        for key in RETRAIN_MODELS:
            self.assertIn(key, MODELS)

    def test_paper_checkpoint_is_pinned(self):
        """Pin which key the paper reports.

        Every headline number comes from this checkpoint. ``lumen`` is an
        earlier run of the same configuration whose scores are close but not
        equal (external AUROC 0.951 against 0.955, PatchCamelyon 0.626 against
        0.722 chance-corrected), so pointing reporting code at the wrong key
        produces plausible numbers rather than an error.
        """
        self.assertEqual(PAPER_CHECKPOINT, "lumen_retrain_quilt")
        self.assertIn(PAPER_CHECKPOINT, MODELS)


class DatasetRegistryTests(unittest.TestCase):
    def test_exactly_the_expected_patch_datasets(self):
        self.assertEqual(set(DATASETS), BENCHMARK_DATASETS)

    def test_num_classes_matches_class_names(self):
        for key, spec in DATASETS.items():
            self.assertEqual(key, spec.key)
            self.assertEqual(spec.num_classes, len(spec.class_names))
            self.assertGreaterEqual(spec.num_classes, 2)
            self.assertTrue(all(isinstance(n, str) and n for n in spec.class_names))

    def test_dataset_keys_align_with_prompt_banks(self):
        self.assertEqual(set(DATASETS), set(pb.CLASS_NAME_VARIANTS))



class ReferenceCheckpointProvenanceTests(unittest.TestCase):
    """No analysis may hardcode which checkpoint it reports.

    Several reported numbers were produced against a superseded checkpoint
    rather than the paper's ``lumen_retrain_quilt``: the paired lymph-node
    comparisons, the melanoma prompt-sensitivity AUROCs, the corpus-ablation
    lymph-node row and the top-five pooling check. None
    raised an error, because the two checkpoints are the same architecture
    trained the same way and score close enough that the wrong one looks right.
    The only durable fix is to forbid the literal.
    """

    # Files that legitimately mention the key: the registry defines it, and the
    # loader dispatch table is keyed by it.
    ALLOWED = {"lumen/models/registry.py", "lumen/models/loaders.py"}

    # The package is also named ``lumen``, so the literal no longer implies a
    # model key: a path segment or an HF config field reads identically. Rather
    # than exempt whole files and lose the guard over them, a line may opt out
    # with this marker, which has to be justified where it is written.
    EXEMPT_MARKER = "not-a-model-key"

    def test_no_analysis_module_hardcodes_the_lumen_key(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        offenders = []
        for sub in ("lumen/benchmark", "lumen/wsi", "cli"):
            for path in (root / sub).rglob("*.py"):
                rel = path.relative_to(root).as_posix()
                if rel in self.ALLOWED:
                    continue
                for i, line in enumerate(path.read_text().splitlines(), 1):
                    if self.EXEMPT_MARKER in line:
                        continue
                    if f'"{PAPER_CHECKPOINT}"' in line and not line.lstrip().startswith("#"):
                        offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(
            offenders, [],
            f"hardcoded {PAPER_CHECKPOINT!r}; use PAPER_CHECKPOINT so the "
            "reported checkpoint is chosen in one place:\n" + "\n".join(offenders))

if __name__ == "__main__":
    unittest.main()
