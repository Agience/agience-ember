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


# ── the store environment this suite runs in ────────────────────────────────────────────────────
#
# ⛔ THE LINE ABOVE WAS NOT TRUE, AND ONLY CI COULD SEE IT. "No environment variable" described the
# intent; 10 tests actually required two variables that happened to be exported in the developer's
# shell and nowhere else. Measured 2026-09-16 on a clean runner: `test_console` (6),
# `test_invoke_remote_dispatch` (1) and `test_compose_answer` (3) all failed at store construction —
# `EMBER_STORE_BACKEND=sqlite requires EMBER_NODE_ID`, and a keys directory the store refuses to
# create. Every one of them passed locally. A suite that passes only on the machine that wrote it is
# not a suite, and the claim above is the thing this fixture makes honest.
#
# ⚠ SET UNCONDITIONALLY, NOT `setdefault`. Inheriting the developer's real values is the failure
# this is about: `EMBER_STORE_KEYS_DIR` would then point at a REAL keys directory holding a real
# `content.key`, and a test that writes would write there. Deterministic values also mean a test
# asserting on the origin gets the same answer on every machine.
#
# ⚠ SESSION-SCOPED, so a per-test `monkeypatch.setenv` still wins — `test_console.py`'s own
# `_clean_env` fixture is function-scoped and continues to override all of this, which is correct:
# that fixture isolates per-test state, this one only establishes a floor so nothing falls through
# to the developer's shell.
#
# ⭐ NOT A CI FIX. Setting these in `.github/workflows/ci.yml` would have turned the runner green
# and left the suite just as dependent on ambient state — the next person to clone it would see the
# same 10 failures, and the workflow would be carrying a fact about the tests that the tests do not
# state themselves.
import pytest as _pytest


@_pytest.fixture(autouse=True, scope="session")
def _suite_store_environment(tmp_path_factory):
    """Give the whole suite a store identity and a keys directory of its own."""
    mp = _pytest.MonkeyPatch()
    keys = tmp_path_factory.mktemp("suite-keys")
    # `(_origin, _seq)` is the store's only version identity, so the origin has to be stable across
    # the session rather than per-test: a value that changed between tests would fork this
    # observer's proper time inside a single run.
    mp.setenv("EMBER_NODE_ID", "test-node")
    mp.setenv("EMBER_STORE_KEYS_DIR", str(keys))
    yield
    mp.undo()
