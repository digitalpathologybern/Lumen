"""Annotations must land in the frame the maps are drawn in.

A MIRAX slide records its scan inside a larger canvas and declares where that
scan starts. A tool that reads the slide from the bounds origin, as MetAssist-2
does, writes coordinates relative to that origin, while a map built from
OpenSlide's ``read_region`` is in canvas coordinates. Overlaying one on the other
without the offset moves every polygon by the bounds origin, which on the slides
here is up to 339,456 px, or 82 mm.

The failure is quiet and expensive: the polygons still land inside the canvas, so
nothing errors, and the figure simply shows two models disagreeing about where a
metastasis is. That misreading survived a round of review before the offset was
found, which is why these tests exist.
"""

from __future__ import annotations

import json

import numpy as np
import openslide
import pytest

from lumen.utils.annotations import (
    load_geojson_polygons,
    load_metassist_polygons,
    slide_origin,
)


class FakeSlide:
    """Only the properties the frame calculation reads."""

    def __init__(self, bx=None, by=None):
        self.properties = {}
        if bx is not None:
            self.properties[openslide.PROPERTY_NAME_BOUNDS_X] = bx
        if by is not None:
            self.properties[openslide.PROPERTY_NAME_BOUNDS_Y] = by


def _write(tmp_path, rings, name="s_metastasis.geojson", classification="Tumor"):
    feats = [{"geometry": {"type": "Polygon", "coordinates": [r]},
              "properties": {"classification": {"name": classification}}}
             for r in rings]
    p = tmp_path / name
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return p


SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def test_slide_without_bounds_reads_as_no_offset():
    assert slide_origin(FakeSlide()) == (0, 0)


def test_bounds_are_read_as_the_origin():
    assert slide_origin(FakeSlide(bx=339456, by=92928)) == (339456, 92928)


@pytest.mark.parametrize("raw", ["5376", 5376])
def test_bounds_are_accepted_as_text_or_number(raw):
    """OpenSlide hands these back as strings; a fake or a cache may not."""
    assert slide_origin(FakeSlide(bx=raw, by=raw)) == (5376, 5376)


def test_polygons_are_shifted_into_the_canvas_frame(tmp_path):
    path = _write(tmp_path, [SQUARE])
    shifted = load_geojson_polygons(path, FakeSlide(bx=5376, by=28928))
    assert len(shifted) == 1
    assert shifted[0][0].tolist() == [5376.0, 28928.0]
    assert shifted[0][2].tolist() == [5386.0, 28938.0]


def test_a_slide_without_bounds_leaves_coordinates_alone(tmp_path):
    """The correction has to be safe to apply unconditionally."""
    path = _write(tmp_path, [SQUARE])
    plain = load_geojson_polygons(path, FakeSlide())
    assert np.allclose(plain[0], np.array(SQUARE))
    assert np.allclose(load_geojson_polygons(path)[0], np.array(SQUARE))


def test_the_offset_is_the_whole_difference(tmp_path):
    """Shape and vertex order must survive the shift untouched."""
    path = _write(tmp_path, [SQUARE])
    a = load_geojson_polygons(path)[0]
    b = load_geojson_polygons(path, FakeSlide(bx=100, by=200))[0]
    assert np.allclose(b - a, np.array([100.0, 200.0]))


def test_non_tumour_objects_are_filtered_out(tmp_path):
    """The lymph-node outline is written beside the tumour and is not tumour."""
    path = _write(tmp_path, [SQUARE], classification="Lymph node")
    assert load_geojson_polygons(path, keep="tumor") == []
    assert len(load_geojson_polygons(path)) == 1


def test_degenerate_rings_are_dropped(tmp_path):
    path = _write(tmp_path, [SQUARE, [[0.0, 0.0], [1.0, 1.0]]])
    assert len(load_geojson_polygons(path)) == 1


def test_a_run_with_no_metastasis_file_is_not_an_error(tmp_path):
    """Distinct from a missing run, and a valid result to draw."""
    assert load_metassist_polygons(tmp_path) == []


def test_metassist_loader_applies_the_offset(tmp_path):
    _write(tmp_path, [SQUARE])
    polys = load_metassist_polygons(tmp_path, FakeSlide(bx=2560, by=24832))
    assert polys[0][0].tolist() == [2560.0, 24832.0]


def test_no_module_parses_the_geojson_on_its_own():
    """Only ``lumen.utils.annotations`` may read the metastasis GeoJSON.

    Several consumers once had their own parser and none applied the canvas
    offset, so a new private copy reintroduces a silent misregistration in one
    consumer while the others stay correct.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in sorted(
        p.relative_to(root).as_posix()
        for d in ("cli", "lumen")
        for p in (root / d).rglob("*.py")
        if p.name != "annotations.py"
    ):
        src = (root / rel).read_text(encoding="utf-8")
        assert "_metastasis.geojson" not in src, (
            f"{rel} globs the metastasis GeoJSON itself; call "
            f"lumen.utils.annotations.load_metassist_polygons instead"
        )
