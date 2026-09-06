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
