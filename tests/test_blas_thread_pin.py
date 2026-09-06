"""The BLAS thread pin travels with the package, and is measured by its effect.

`ember/__init__.py` runs `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")` at package scope,
above the imports that reach `mantle.shard.cache`. Ember is the runner — the package that spreads
onto every machine — and two threads inside `numpy.linalg.eigh` fault the reference box's OpenBLAS
3/3 (exit 139) and can hang instead, so one green run proves little.

Effect, not string: the pool assertions read the thread count back through `threadpoolctl`. A pin
sitting below an import that pulls numpy still writes "1" into the environment while OpenBLAS has
already sized its pool, and only the pool size tells the two apart.
`test_late_pin_is_inert_negative_control` sets the variable after `import numpy` and requires the
pool to stay wide, which is what gives the pool assertions their meaning.

The parametrised cases name the modules AST-measured to call `numpy.linalg.*` and import each one
directly, so coverage is measured per module rather than inherited from the package path.
`test_operator_value_is_not_overridden` pins the pin as a default: a value an operator exported
survives import.

Everything runs in a subprocess: this process has already imported numpy, so its own pool was
sized long before any assertion here could run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

PACKAGE = "ember"
PIN = "OPENBLAS_NUM_THREADS"

# Modules that import numpy at module level and call `numpy.linalg.*`, measured from the AST.
# Only the ones that import standalone without a live store are listed; the pin is a package-scope
# line, so these are a representative probe of the path rather than a hand-maintained roster.
BLAS_MODULES = [
    "ember.signal.projection",   # numpy.linalg.svd
    "ember.signal.forgetting",   # numpy.linalg.norm
]


def _run(body: str, env_pin: str | None) -> dict:
    """Run `body` in a clean interpreter and return its JSON verdict.

    `env_pin=None` removes the variable from the child's environment — the unset case the pin
    exists to cover. The parent's `sys.path` is handed over in-band rather than through
    PYTHONPATH, so this works from a bare `pytest` with nothing exported.
    """
    prelude = f"import sys, os, json\nsys.path[:0] = {json.dumps(sys.path)}\n"
    env = dict(os.environ)
    env.pop(PIN, None)
    if env_pin is not None:
        env[PIN] = env_pin
    proc = subprocess.run(
        [sys.executable, "-c", prelude + body],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode == 0, f"child failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_POOL_READER = (
    "import threadpoolctl\n"
    "pools = [d['num_threads'] for d in threadpoolctl.threadpool_info()\n"
    "         if d.get('internal_api') == 'openblas']\n"
)


def test_threadpoolctl_is_available() -> None:
    """threadpoolctl is a declared dev dependency, so the pool can always be read.

    A missing measurement tool fails here rather than turning the pool assertions into skips, so
    the suite cannot be green with nothing measured."""
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:  # pragma: no cover - the point is that this is loud
        pytest.fail(
            "threadpoolctl is not installed, so the BLAS pin can only be checked as a string and "
            "not as an effect. Install it (`pip install threadpoolctl`). Deliberately a failure, "
            "not a skip."
        )


def test_pin_is_set_by_importing_the_package() -> None:
    """Importing `ember` with the variable unset leaves it pinned to "1".

    That is the `os.environ.setdefault` block in `ember/__init__.py` doing its work."""
    v = _run(f"import {PACKAGE}\nprint(json.dumps({{'val': os.environ.get({PIN!r})}}))", None)
    assert v["val"] == "1", (
        f"{PACKAGE} did not pin {PIN} on import (got {v['val']!r}). The pin must be set by the "
        f"package, not by whoever remembers to export it."
    )


@pytest.mark.parametrize("mod", BLAS_MODULES)
def test_pin_covers_each_measured_blas_module(mod: str) -> None:
    """A measured LAPACK caller imported directly is covered too.

    Python initialises parent packages before submodules, so `import ember.signal.projection`
    runs `ember/__init__.py` first. The pin therefore belongs at the package root, where every
    such import passes through it."""
    v = _run(f"import {mod}\nprint(json.dumps({{'val': os.environ.get({PIN!r})}}))", None)
    assert v["val"] == "1", f"importing {mod} left {PIN}={v['val']!r}"


@pytest.mark.parametrize("mod", BLAS_MODULES)
def test_pin_actually_sizes_the_openblas_pool(mod: str) -> None:
    """After importing the module, OpenBLAS reports one thread.

    The pool size is what shows the pin took effect: a present-but-late pin still writes "1" into
    the environment."""
    v = _run(f"import {mod}\n" + _POOL_READER + "print(json.dumps({'pools': pools}))", None)
    assert v["pools"], "no OpenBLAS pool reported; threadpoolctl saw no BLAS backend to measure"
    assert all(p == 1 for p in v["pools"]), (
        f"OpenBLAS pool is {v['pools']} after importing {mod}. The variable may well be set — a "
        f"late set is inert, because the pool is sized when the library loads."
    )


def test_late_pin_is_inert_negative_control() -> None:
    """Negative control: it shows the pool measurement above can fail.

    Import numpy first, then set the variable, then force a LAPACK call. The pool stays wider than
    one thread, because OpenBLAS sized it when the library loaded. A pool of 1 here would mean
    `test_pin_actually_sizes_the_openblas_pool` passes for free.

    A machine that reports one thread unpinned — a single-core box, or an OpenBLAS built without
    threading — cannot discriminate between pinned and unpinned. That is a failure here rather
    than a skip, so a run that measured nothing is visible.
    """
    v = _run(
        "import numpy\n"
        f"os.environ[{PIN!r}] = '1'\n"
        "numpy.linalg.eigh(numpy.eye(64))\n" + _POOL_READER + "print(json.dumps({'pools': pools}))",
        None,
    )
    assert v["pools"], "no OpenBLAS pool reported; the control cannot run"
    assert all(p > 1 for p in v["pools"]), (
        f"unpinned OpenBLAS pool is {v['pools']}, not >1. Either this machine has one usable core "
        f"or OpenBLAS is single-threaded here; either way the pin cannot be distinguished from a "
        f"no-op on this hardware, so test_pin_actually_sizes_the_openblas_pool proves nothing. "
        f"cpu_count={os.cpu_count()}"
    )


def test_operator_value_is_not_overridden() -> None:
    """An operator who exported a value keeps it: the pin is a default rather than a policy.

    This bounds what the pin covers. A deployment that exports `OPENBLAS_NUM_THREADS=8` reinstates
    the fault the pin avoids; what the package owes is the unset case."""
    v = _run(f"import {PACKAGE}\nprint(json.dumps({{'val': os.environ.get({PIN!r})}}))", "3")
    assert v["val"] == "3", (
        f"{PACKAGE} overwrote an operator-set {PIN} (got {v['val']!r}, expected '3'). Use "
        f"os.environ.setdefault, not assignment."
    )
