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


# ── no operator payloads ────────────────────────────────────────────────────────────────────────
#
# This suite needs none, and that is deliberate. Ember EXECUTES operators at runtime — the engine
# resolves a group through `prism.runner` and runs a sha-verified payload — but the payloads are
# chorus's, built from chorus source, and a test suite that needed them made ember's CI depend on a
# repository above it in the DAG.
#
# The 117 tests that exercised operators through the runner now live in `agience-chorus`, beside the
# operators they are about. What is left here is ember: the cache, the read path, the mesh, the
# relay, identity, the ontology coordinate. It runs with no sibling checkout and no environment
# variable — `python -m pytest -q` is the whole command.
