"""The level-0 resolution guard, and that every consumer goes through it.

A slide's declared resolution is not trustworthy. One cohort here declares 1000
microns per pixel because its TIFF records 10 pixels per centimetre, and a
pipeline that believes it tries to upsample a thousandfold. The guard rejects any
value outside a plausible range, whatever its source.

These tests exist because the guard was originally written inside the tiling
function only, so tiling rejected the bad value while the mapping and figure
paths accepted it, and the same slide was readable by one part of the pipeline
and fatal to another. The consumer tests below are what stops that recurring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import openslide
import pytest

from lumen.utils.annotations import (
    DEFAULT_BASE_MPP,
    PLAUSIBLE_MPP,
    resolve_base_mpp,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeSlide:
    """Just enough OpenSlide surface for resolution questions."""

    def __init__(self, mpp=None, objective=None):
        self.properties = {}
        if mpp is not None:
            self.properties[openslide.PROPERTY_NAME_MPP_X] = mpp
        if objective is not None:
            self.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] = objective


@pytest.mark.parametrize("declared", ["0.25", "0.5", 0.2305, "0.251266", 1.0, 2.0])
def test_plausible_resolutions_pass_through_untouched(declared):
    assert resolve_base_mpp(FakeSlide(mpp=declared)) == pytest.approx(float(declared))


@pytest.mark.parametrize("declared", [
    "1000",       # the real fault: 10 pixels per centimetre read as microns
    1000.0,
    "0",          # a zero would make the downsample a division by zero
    "-0.5",
    "1e6",
    "0.001",      # finer than light microscopy resolves
    "500",
])
def test_implausible_resolutions_are_rejected(declared):
    assert resolve_base_mpp(FakeSlide(mpp=declared)) == DEFAULT_BASE_MPP


@pytest.mark.parametrize("declared", [None, "", "unknown", "N/A", float("nan")])
def test_missing_or_unparseable_resolution_falls_back(declared):
    assert resolve_base_mpp(FakeSlide(mpp=declared)) == DEFAULT_BASE_MPP


def test_range_boundaries_are_inclusive():
    lo, hi = PLAUSIBLE_MPP
    assert resolve_base_mpp(FakeSlide(mpp=lo)) == pytest.approx(lo)
    assert resolve_base_mpp(FakeSlide(mpp=hi)) == pytest.approx(hi)


@pytest.mark.parametrize("objective,expected", [(40, 0.25), ("40", 0.25),
                                               (20, 0.5), (10, 1.0), (5, 2.0)])
def test_objective_power_is_used_when_the_resolution_is_unusable(objective, expected):
    """A slide that states its magnification but not its resolution is not lost."""
    assert resolve_base_mpp(FakeSlide(mpp="1000", objective=objective)) == expected
    assert resolve_base_mpp(FakeSlide(mpp=None, objective=objective)) == expected


def test_a_believable_resolution_beats_the_objective_power():
    """The declared resolution wins when it is usable, even against an objective."""
    assert resolve_base_mpp(FakeSlide(mpp="0.25", objective=20)) == pytest.approx(0.25)


def test_unknown_objective_power_falls_back():
    assert resolve_base_mpp(FakeSlide(mpp="1000", objective=63)) == DEFAULT_BASE_MPP
    assert resolve_base_mpp(FakeSlide(mpp="1000", objective="?")) == DEFAULT_BASE_MPP


def test_the_guard_bounds_the_work_a_bad_tag_can_cause():
    """The point of the guard, stated as the quantity that actually broke.

    At the declared 1000 MPP, resampling a 56,244 px slide to 1 micron per pixel
    asks for a 56-million-pixel width. Under the guard it asks for 28,122, which
    is what the sibling cohort with correct metadata needs.
    """
    width0 = 56244
    naive = width0 / (1.0 / 1000.0)
    guarded = width0 / (1.0 / resolve_base_mpp(FakeSlide(mpp="1000")))
    assert naive > 5e7
    assert guarded == pytest.approx(28122)


# Modules that resolve a slide's resolution to read pixels at a physical scale.
# A new one added here without going through the guard is the regression this
# catches, so the list is deliberately explicit rather than globbed.
_RESOLUTION_CONSUMERS = [
    "lumen/wsi/segment_classifier.py",
]


@pytest.mark.parametrize("rel", _RESOLUTION_CONSUMERS)
def test_no_consumer_reads_the_resolution_property_directly(rel):
    """Nobody may reach for ``openslide.mpp-x`` outside the guard itself.

    Reading it directly is how a module ends up with its own fallback. Before
    this test there were three answers in the tree to what a slide's resolution
    is, with fallbacks of 0.5, 0.5 and 0.25.
    """
    src = (ROOT / rel).read_text()
    assert "mpp-x" not in src, (
        f"{rel} reads openslide.mpp-x directly; call "
        f"lumen.utils.annotations.resolve_base_mpp instead"
    )
    assert "PROPERTY_NAME_MPP_X" not in src, (
        f"{rel} reads PROPERTY_NAME_MPP_X directly; call resolve_base_mpp instead"
    )


def test_the_guard_is_general_not_cohort_specific():
    """The check is on the number, so it catches any vendor's bad tag.

    A guard written as "if this is the endometrial cohort" would pass the unit
    tests above and still fail on the next scanner that mislabels its units.
    """
    src = (ROOT / "lumen/utils/annotations.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "resolve_base_mpp")
    body = ast.get_source_segment(src, fn) or ""
    for token in ("histai", "endometrial", "HISTAI", "case_"):
        assert token not in body, f"resolve_base_mpp branches on {token!r}"
