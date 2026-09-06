"""Ember's host seam for the pattern runner — the loader itself is `prism.runner`.

Everything that verifies and execs a bundle lives in `prism/runner.py`; this module re-exports it
unchanged so every `from ember.runtime.runner import …` call site keeps working, and it does the
one thing that is genuinely ember's: it fills ember's host seam.

The loader lives in prism rather than in ember or chorus because its dependencies (`canonical`,
`crystal_model`, `mass`, `trust.opsign`) are prism's, and prism is a pure leaf that both the runner
and the personas reach downward — the only home below both.

A bundle declares the host modules it may reach for (`host_seams`); which module fills a seam is
the host's answer, not the loader's. It is registered here, by the host that owns the module —
which is what a seam means.

The seam is registered at import, before the first `load()` — the same ordering `attach(store)`
follows. Registering at import means any path that reaches a bundle through ember has already
bound it. A bundle loaded through `prism.runner` directly (chorus's path) does not resolve `match`
and falls back to the bundle's own default (`operators.select_for` answers `basis="generic"`) — the
documented behavior of an unfilled seam.
"""
from __future__ import annotations

from prism import runner as _runner
from prism.runner import (  # noqa: F401  — re-exported for ember's existing call sites
    BUNDLE_ARTIFACT_PREFIX,
    BUNDLE_CONTENT_TYPE,
    GROUPS,
    BundleIntegrityError,
    BundleTrustError,
    attach,
    load,
    loaded,
    register_fns,
    register_seam,
    registered_seams,
    verify_provenance,
)

# Ember's host seams — the `operators` group's optional geometric matcher and the three measurements
# the personas declare. See the module docstring for why they are bound here and not in the loader,
# and `ember/runtime/seams.py` for the table itself.
#
# The table is four entries; `ember/__init__.py` binds them too, so a process that has ember at all
# has them bound, not only one that reached the loader. This call keeps the bundle path's ordering
# unchanged.
from ember.runtime.seams import register_host_seams

register_host_seams()

__all__ = ["BUNDLE_CONTENT_TYPE", "BUNDLE_ARTIFACT_PREFIX", "GROUPS", "BundleIntegrityError",
           "BundleTrustError", "attach", "load", "loaded", "register_fns", "register_seam",
           "registered_seams", "verify_provenance"]


def __getattr__(name: str):
    """PEP 562: `from ember.runtime.runner import arithmetic` / `… import evolution` still reads as it
    always did, and resolves through the single distribution path in prism."""
    return getattr(_runner, name)
