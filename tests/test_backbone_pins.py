"""The backbones are the only weights pulled from the Hub, and they are pinned.

An unpinned repository resolves to whatever `main` points at today, so a rerun
can quietly get different weights than the reported numbers came from. The pins
live in the code; docs/MODEL_SOURCES.md is what a reader checks them against,
and the two drifting apart is the failure this catches.
"""

import pathlib
import re

from lumen.utils.backbones import (
    BACKBONE_REVISION,
    accepted_text_backbones,
    accepted_vision_backbones,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "docs" / "MODEL_SOURCES.md").read_text()


def test_both_lumen_backbones_are_pinned():
    assert set(BACKBONE_REVISION) == {"Virchow2", "biomedBERT"}
    for key, rev in BACKBONE_REVISION.items():
        assert re.fullmatch(r"[0-9a-f]{40}", rev), f"{key} pin is not a commit sha"


def test_pins_match_the_documented_commits():
    for key, rev in BACKBONE_REVISION.items():
        repo = (accepted_vision_backbones.get(key)
                or accepted_text_backbones[key]).removeprefix("hf-hub:")
        assert repo in SOURCES, f"{repo} is not in docs/MODEL_SOURCES.md"
        assert rev in SOURCES, (
            f"{key} is pinned to {rev}, which docs/MODEL_SOURCES.md does not list")
