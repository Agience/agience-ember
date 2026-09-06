"""Shared test wiring for the ember suite.

The BLAS thread pin is the first statement below this docstring, and it has to be. `ember/__init__.py`
pins OpenBLAS to one thread, but of the 23 test modules here that import numpy, 22 import it before
they import ember. OpenBLAS sizes its worker pool when the library loads, so a pin applied by the
package arrives too late to bind. `conftest.py` is imported before the test modules in its directory,
which makes this the first in-process opportunity — see the note in `ember/__init__.py` for the
measurement and for why it is `setdefault` rather than an assignment.

Fixtures shared across modules live in `_fakes.py` and `_paths.py` beside this file, rather than in a
test module that others import. `tests/` has no `__init__.py`, so it is not a package and one test
module cannot import another. The `sys.path` line below makes the helpers importable as plain siblings
(`from _fakes import _FakeStore`) under any pytest import mode, including `importlib`, where the
default directory insertion does not happen.
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # must precede numpy; see the docstring
del _os

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ── the operator bundles ────────────────────────────────────────────────────────────────────────
#
# `prism.runner` resolves an operator group to a sha-verified payload and there is no in-package
# copy to fall back to — deliberately: a second copy of a content-addressed payload can drift from
# the one the mesh carries, and the sha gate would then verify the wrong bytes faithfully. So the
# payloads live in `agience-observe/bundles/`.
#
# This finds a sibling checkout so nobody has to set the variable by hand. It does NOT make the
# suite runnable without one, and that was measured rather than assumed: fifteen modules fail at
# IMPORT without the payloads, and a further hundred-odd tests fail at RUN time when they call into
# the runner. 140 failures in total. Ignoring the import-time fifteen converts collection errors
# into test failures and buys nothing, so it is not attempted — the suite needs the bundles, and the
# header below says so rather than letting a reader discover it one failure at a time.
_BUNDLES = Path(__import__("os").environ.get("AGIENCE_BUNDLE_ROOT")
                or (Path(__file__).resolve().parents[2] / "agience-observe" / "bundles"))
BUNDLES_PRESENT = _BUNDLES.is_dir() and any(_BUNDLES.glob("*.json"))

if BUNDLES_PRESENT:
    import os as _o
    _o.environ.setdefault("AGIENCE_BUNDLE_ROOT", str(_BUNDLES))
    del _o


def pytest_report_header(config):
    """Say where the payloads came from, or that they are missing, before anything runs."""
    if BUNDLES_PRESENT:
        return f"operator bundles: {_BUNDLES}"
    return (
        f"operator bundles: NOT FOUND at {_BUNDLES}. ~140 tests need them and will fail. "
        f"Set AGIENCE_BUNDLE_ROOT, or check out `agience-observe` beside this repository."
    )
