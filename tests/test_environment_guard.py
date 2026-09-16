"""A run must not quietly use packages the environment did not provide.

PYTHONNOUSERSITE has to be set before the interpreter starts, so nothing here
can set it. What it can do is refuse to load a model once a dependency has
demonstrably come from ~/.local instead, which is the point where the numbers
stop being comparable with the reported ones.
"""

import sys
import types

import pytest

from lumen import require_environment


class _Flags:
    """sys.flags is a read-only struct sequence, so stand in for it.

    Everything except ``no_user_site`` defers to the real flags; the interpreter
    reads others (``utf8_mode``) during ordinary file access.
    """

    def __init__(self, real, no_user_site):
        self._real, self.no_user_site = real, no_user_site

    def __getattr__(self, name):
        return getattr(self._real, name)


def _fake_user_site(monkeypatch, tmp_path, module_name):
    """Make ``module_name`` look as if it were imported from the user site."""
    user_site = tmp_path / "site-packages"
    (user_site / module_name).mkdir(parents=True)
    origin = user_site / module_name / "__init__.py"
    origin.write_text("")                      # before sys.flags is stood in for

    monkeypatch.setattr("site.getusersitepackages", lambda: str(user_site))
    module = types.ModuleType(module_name)
    module.__file__ = str(origin)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(sys, "flags", _Flags(sys.flags, 0))
    return origin


def test_silent_when_user_site_is_disabled(monkeypatch):
    monkeypatch.setattr(sys, "flags", _Flags(sys.flags, 1))
    require_environment()


def test_refuses_when_a_dependency_came_from_the_user_site(monkeypatch, tmp_path):
    origin = _fake_user_site(monkeypatch, tmp_path, "torch")
    with pytest.raises(RuntimeError) as exc:
        require_environment()
    assert "torch" in str(exc.value)
    assert str(origin) in str(exc.value)
    assert "PYTHONNOUSERSITE=1" in str(exc.value)


def test_ignores_packages_that_do_not_decide_the_numbers(monkeypatch, tmp_path):
    _fake_user_site(monkeypatch, tmp_path, "some_unrelated_package")
    require_environment()


def test_load_adapter_calls_it(monkeypatch):
    """The guard is on the path every evaluation stage takes."""
    import lumen.models.loaders as loaders

    called = []
    monkeypatch.setattr("lumen.require_environment",
                        lambda: called.append(True) or (_ for _ in ()).throw(
                            RuntimeError("guard reached")))
    with pytest.raises(RuntimeError, match="guard reached"):
        loaders.load_adapter(object())
    assert called
