"""The cross-source colimit sweep must not be gated on remote source reachability.

Measured on 71/home, 2026-08-25: `consolidates` edges frozen at **11,231** — the same count
recorded on 2026-08-24 — and the sweep had not run since `stage.1.grammar` promoted on
**2026-07-25**. The cause was a coupling, not a defect in the sweep:

    consolidate_crosswalk ran only when advance_curriculum reported `promoted` / `complete`
      -> a stage promotes only when EVERY source in it is drained
        -> a source that cannot be fetched (OEWN 503, an OMW licence clearing) DEFERS
          -> so one unreachable remote host silently withheld a sweep over local data

The sweep reads only what is already in the store. These tests hold the two halves of the fix:
a promotion still triggers it, and a promotion is no longer required.
"""
from __future__ import annotations

import _paths  # noqa: F401  (puts src/ on the path the way the rest of the suite does)

from ember.runtime import worker


class _Genesis:
    """Stands in for `ember.genesis`. Records what the tick asked it to do."""

    def __init__(self, advance=None):
        self.swept = 0
        self.applied = []
        self._advance = advance if advance is not None else {}

    def advance_curriculum(self, bundle):
        return dict(self._advance)

    def consolidate_crosswalk(self, bundle, *, apply=False, limit=None, **kw):
        self.swept += 1
        self.applied.append((apply, limit))
        return {"diagrams": 7, "applied": apply}


class _Improve:
    @staticmethod
    def improve_cycle(bundle):
        return {"rho": 1.0}


def _install(monkeypatch, genesis_mod):
    """Put a stand-in where `_tick` will actually look for it.

    `_tick` does `from ember import genesis` and `from ember.runtime import improve`, and once a
    package is imported those resolve the ATTRIBUTE on the package, not `sys.modules`. Patching
    only `sys.modules` therefore works in isolation and silently does nothing as soon as any other
    test in the run has already imported ember.genesis — which is exactly how these tests first
    passed alone and failed in the suite. Both are patched, so import order stops mattering.
    """
    import sys

    import ember
    import ember.runtime
    monkeypatch.setitem(sys.modules, "ember.genesis", genesis_mod)
    monkeypatch.setattr(ember, "genesis", genesis_mod, raising=False)
    monkeypatch.setitem(sys.modules, "ember.runtime.improve", _Improve)
    monkeypatch.setattr(ember.runtime, "improve", _Improve, raising=False)


def _run_tick(monkeypatch, *, ingest=False, consolidate=False, advance=None):
    import types
    g = _Genesis(advance)
    mod = types.ModuleType("ember.genesis")
    mod.advance_curriculum = g.advance_curriculum
    mod.consolidate_crosswalk = g.consolidate_crosswalk
    _install(monkeypatch, mod)
    rec = worker._tick(object(), ingest=ingest, consolidate=consolidate)
    return g, rec


# ── the regression this exists for ────────────────────────────────────────────────────────────
def test_a_deferring_remote_source_no_longer_withholds_the_sweep(monkeypatch):
    """ingest ran, NOTHING promoted (a source deferred) — the sweep must still run when due."""
    g, rec = _run_tick(monkeypatch, ingest=True, consolidate=True,
                       advance={"stage": "stage.0.lexicon", "ingested": 0})
    assert g.swept == 1, "an unreachable source still held the sweep"
    assert rec.get("consolidated") is not None


def test_the_sweep_runs_with_no_ingest_at_all(monkeypatch):
    """It reads what is already in the store, so it does not need an ingest to have happened."""
    g, rec = _run_tick(monkeypatch, ingest=False, consolidate=True)
    assert g.swept == 1
    assert rec.get("consolidated") is not None


# ── what must NOT change ──────────────────────────────────────────────────────────────────────
def test_a_promotion_still_triggers_the_sweep(monkeypatch):
    g, _ = _run_tick(monkeypatch, ingest=True, consolidate=False, advance={"promoted": True})
    assert g.swept == 1, "the promotion trigger was dropped instead of joined"


def test_a_complete_curriculum_still_triggers_the_sweep(monkeypatch):
    g, _ = _run_tick(monkeypatch, ingest=True, consolidate=False, advance={"curriculum": "complete"})
    assert g.swept == 1


def test_an_ordinary_tick_does_not_sweep(monkeypatch):
    """It is a full-corpus scan. Running it every tick is the waste the gate exists to prevent."""
    g, rec = _run_tick(monkeypatch, ingest=True, consolidate=False, advance={"ingested": 500})
    assert g.swept == 0
    assert "consolidated" not in rec


def test_the_sweep_stays_bounded_and_applies(monkeypatch):
    """`limit` is what keeps one tick from scanning the whole corpus unbounded."""
    g, _ = _run_tick(monkeypatch, consolidate=True)
    assert g.applied == [(True, 3000)]


def test_a_failing_sweep_does_not_kill_the_tick(monkeypatch):
    """A bad sweep must be recorded, not raised — the tick still has to measure and heartbeat."""
    import types
    mod = types.ModuleType("ember.genesis")

    def boom(bundle, **kw):
        raise RuntimeError("store unavailable")

    mod.consolidate_crosswalk = boom
    mod.advance_curriculum = lambda bundle: {}
    _install(monkeypatch, mod)
    rec = worker._tick(object(), ingest=False, consolidate=True)
    assert rec["ok"] is True
    assert "error" in rec["consolidated"]


# ── the cadence ───────────────────────────────────────────────────────────────────────────────
def test_zero_disables_the_cadence_which_is_the_old_behaviour():
    assert [t for t in range(1, 25) if worker.consolidation_due(t, 0)] == []


def test_the_cadence_fires_every_nth_tick_and_not_every_tick():
    assert [t for t in range(1, 13) if worker.consolidation_due(t, 4)] == [4, 8, 12]
    assert [t for t in range(1, 6) if worker.consolidation_due(t, 1)] == [1, 2, 3, 4, 5]
