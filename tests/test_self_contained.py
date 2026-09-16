"""The repository must run from a clone, on a machine that is not ours.

This is a release requirement, not a style preference. Reviewers get the code
without our cluster, without our home directory and without the sibling
projects that happen to sit beside it here.
"""

from __future__ import annotations

import ast
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIRS = ("lumen", "cli", "tests")

# This file necessarily contains the patterns it searches for.
ABSOLUTE_PATH_ALLOWED = {"tests/test_self_contained.py"}


def _python_files():
    for d in CODE_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if "__pycache__" not in p.parts:
                yield p


def test_no_machine_specific_absolute_paths_in_code():
    """A literal under /storage or /homefs makes the clone unusable elsewhere.

    The dataset registry once held such a literal, pointing into one user's
    directory, so it resolved to nothing on any other machine and said nothing
    about why.
    """
    offenders = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in ABSOLUTE_PATH_ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith(("*", '"""')):
                continue
            for needle in ("/storage/", "/homefs/", "/home/"):
                if needle in line and "LUMEN_" not in line:
                    offenders.append(f"{rel}:{i}: {stripped[:90]}")
    assert not offenders, (
        "machine-specific absolute paths; route through lumen.paths:\n"
        + "\n".join(offenders))


def test_no_imports_from_sibling_projects():
    """Nothing may import a neighbouring project.

    A sibling's `src/` was on `sys.path` through its editable install, which put
    bare `models`, `training` and `inference` on the path. Those names collide
    with this package's own subpackages, so an import could silently resolve
    into the wrong project.
    """
    forbidden = {"metassist", "MetAssist"}
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in forbidden:
                    offenders.append(f"{path.relative_to(ROOT)}: import {n}")
    assert not offenders, offenders


def test_every_path_helper_is_repo_relative():
    """Bundled assets must resolve under the repo, wherever it is cloned."""
    from lumen.paths import assets_root, datasets_root, models_root, project_root
    for helper in (assets_root, models_root, datasets_root):
        assert project_root() in helper().parents or helper() == project_root()
