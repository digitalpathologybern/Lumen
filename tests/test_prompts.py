"""Prompt-bank structure and dataset<->prompt consistency.

The alignment test (``canonical`` wording == registry ``class_names``) guards the
exact prompt text behind the reported benchmark numbers: a silent edit to one
but not the other previously shifted SICAPv2 from 0.452 to 0.424.
"""

import unittest

from lumen.benchmark import prompts as pb
from lumen.data.registry import DATASETS


class PromptBankTests(unittest.TestCase):
    def test_canonical_prompts_expand_over_every_template(self):
        n_templates = sum(len(frames) for frames in pb.TEMPLATES.values())
        for dkey in DATASETS:
            ens = pb.canonical_prompts(dkey)
            classes = pb.class_names_for(dkey, "canonical")
            self.assertEqual(list(ens.keys()), list(classes))
            for cls, prompts in ens.items():
                self.assertEqual(len(prompts), n_templates)
                # the bare "{}" template reproduces the class name verbatim
                self.assertIn(cls, prompts)

    def test_build_prompt_set_fills_the_named_template(self):
        ps = pb.build_prompt_set("pcam", "he_image", "canonical")
        names = pb.class_names_for("pcam", "canonical")
        self.assertEqual(list(ps.keys()), list(names))
        for cls, prompts in ps.items():
            self.assertEqual(prompts, [f"an H&E stained image of {cls}."])

    def test_every_variant_has_the_dataset_class_count(self):
        for dkey, spec in DATASETS.items():
            self.assertIn("canonical", pb.variant_names(dkey))
            for variant in pb.variant_names(dkey):
                names = pb.class_names_for(dkey, variant)
                self.assertEqual(len(names), spec.num_classes,
                                 f"{dkey}/{variant} class count mismatch")

    def test_registry_and_prompt_banks_cover_the_same_datasets(self):
        self.assertEqual(set(DATASETS), set(pb.CLASS_NAME_VARIANTS))
        self.assertNotIn("camelyon17", DATASETS)  # patch cam17 retired

    def test_canonical_wording_matches_registry_class_names(self):
        for dkey, spec in DATASETS.items():
            self.assertEqual(
                list(pb.class_names_for(dkey, "canonical")),
                list(spec.class_names),
                f"{dkey}: canonical prompt wording drifted from class_names")

    def test_iter_prompt_sets_covers_template_by_variant_grid(self):
        sets = list(pb.iter_prompt_sets("sicap"))
        expected = len(pb.DEFAULT_TEMPLATE_KEYS) * len(pb.variant_names("sicap"))
        self.assertEqual(len(sets), expected)
        # each yielded set is (template_key, variant, {class: [prompts]})
        for tkey, variant, ps in sets:
            self.assertIn(tkey, pb.TEMPLATES)
            self.assertIn(variant, pb.variant_names("sicap"))
            self.assertEqual(list(ps), list(pb.class_names_for("sicap", variant)))


if __name__ == "__main__":
    unittest.main()


# ── Richer prompt bank ───────────────────────────────────────────────────────

def test_rich_templates_are_the_twenty_two_conch_frames():
    from lumen.benchmark import prompts as pb
    assert len(pb.RICH_TEMPLATES) == 22
    assert all("CLASSNAME" in t for t in pb.RICH_TEMPLATES)


def test_rich_bank_is_synonyms_times_templates():
    """Prompt count per class must be variants x 22, with duplicates dropped."""
    from lumen.benchmark import prompts as pb
    for key in pb.CLASS_NAME_VARIANTS:
        syns = pb.synonyms_for(key)
        rich = pb.rich_prompts(key)
        assert set(syns) == set(rich)
        for name, prompts in rich.items():
            assert len(prompts) == len(syns[name]) * 22
            assert len(set(prompts)) == len(prompts), f"{key}/{name} repeats a prompt"


def test_rich_bank_keys_are_the_canonical_class_names():
    """The read-out maps scores back by canonical name, so keys must not drift."""
    from lumen.benchmark import prompts as pb
    for key in pb.CLASS_NAME_VARIANTS:
        assert list(pb.rich_prompts(key)) == pb.class_names_for(key, "canonical")
