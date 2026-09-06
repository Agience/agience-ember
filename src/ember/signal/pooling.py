"""The pooling Screen — one accumulating Screen that says what basis it holds.

A Screen takes one `(T_p, F)` plane per placement and reads one spectrum over everything pooled.
`F` and the coordinate token are pinned by the first plane that pools; a plane that disagrees with
either is a counted, logged drop rather than a merge.

**A dropped plane is data.** It is counted by reason, logged once per reason, and reported on
`accumulated()` beside the read it bears on, so a caller can see that a pooled read stopped growing
and why ([[absence-is-not-an-affirmative-claim]]).

**Two owners, one mechanism.** `ember.optics.accumulator` is owned by `Delegate._acc` (artifact
`delegate.<id>.screen`, fed from `ontology/activation.py`) for one agent's cognition, and by
`node_screen()` (`node.<id>.screen`) for the node's own state. This module is that accumulating
Screen as its own object — the same accumulator, the same lazy `F` pin, the same never-raises
`place()` — so a second owner is a second instance rather than a second mechanism.

**The coordinate includes the zoom.** `projection._zoom` takes each read at the `k` its own frame
resolves, clamped into the basis generation's certified band, so one generation serves every width
in that band. A generation id alone therefore does not identify a coordinate: see `coordinate_id`,
which carries the `k`, and `_at_zoom`, which completes it from a plane that did not state one.

**Why the basis token carries weight `F` cannot.** Two planes can agree on every shape while the
numbers mean different things — a basis rebuilt to the same `k`, or a re-enrichment that re-scales
every raw `dense_vec` at a constant `D=2048`. Pooling across that yields a confident spectrum of
nothing. So a placer states the coordinate it built in, the Screen pins it alongside `F`, and a
disagreement is a counted drop.

**A placer that states no basis is recorded as having stated none.** `basis=None` means unrecorded.
Such a plane pools — an unlabelled plane still carries real data — and it is counted in
`unrecorded_planes` and surfaced on `accumulated()`, so "this Screen cannot say what basis it holds"
is a reading a caller can take rather than an absence to infer. Treating `None` as agreement would
make the check unable to fail ([[verification-that-cannot-fail]]).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

_log = logging.getLogger(__name__)

#: Why a plane did not pool. Each is counted separately because they are different faults: `width`
#: is a coordinate change visible in `F`, `basis` is a coordinate change the width cannot see,
#: `malformed` is a caller handing over something that is not a plane, and `error` is the instrument
#: itself having no reading to give.
DROP_REASONS: Tuple[str, ...] = ("width", "basis", "malformed", "error")


def coordinate_id(store, *, corpus_basis: bool, k: Optional[int] = None) -> Optional[Tuple]:
    """Which coordinate a frame built by `projection.frame(..., corpus_basis=…)` came back in.

    A token to compare, not a value to interpret: two frames pool iff their tokens are equal.

    · `corpus_basis=True` — the frame was projected onto the live `geom.corpus-basis` generation, so
      the identity is that generation's artifact id together with the zoom `k` the frame was taken
      at. `None` when the store holds no basis or cannot resolve which generation is live.
    · `corpus_basis=False` — the raw dense coordinate the coupling bases live in, whose identity is
      the geometry that emits it. It has no zoom, so `k` is ignored there.

    **The zoom is part of the coordinate.** `projection._zoom` takes each read at the `k` its own
    frame resolves, so one generation serves every width in its certified band. Two frames projected
    onto `B[:123]` and `B[:280]` of the same generation are in different coordinates: `B[:123]` is a
    different feature axis with different columns, not a sub-coordinate of `B[:280]` that anything
    may compare across. PAPER §5.5 states that coupling *"raises without"* a shared basis and needs
    shared `T` and `D`; REASONING-PORT R3.2 names this as the price of a per-read `k`. A token
    naming only the generation would call two frames that share no axis the same coordinate.

    **`k=None` means unstated.** It is carried as `None` in the token, which is unequal to every
    stated `k`, so an unstated frame compares equal only to another unstated one.
    `PooledScreen.place` completes it from the plane it was handed — `F` is the zoom on a
    corpus-basis frame — which is a measurement of the plane rather than an assumption about it, and
    it is what lets a caller that did not thread the width through (`activation.compose`) still pin
    an exact coordinate. A caller holding the frame can pass `k=W.shape[1]`.

    **The generation id is the identity.** Each generation is its own content-addressed artifact
    (`projection._write_generation`), so every observer computes the same token for the same
    generation and a replicated plane pools where it should. The identity travels with the artifact,
    with no separate stamp to keep in step.

    **`None` means unrecorded.** A store that cannot name its live generation cannot certify that
    its basis is the one a Screen already pooled. `basis_head` returns `None` there, so an
    unverifiable store compares equal to no other ([[verification-that-cannot-fail]]).

    **What the raw-dense token covers.** It carries `GEOMETRY_VERSION` and, via `F`, `D`. The IC
    values `dense_vec` is scaled by lie outside it, so a re-enrichment of the IC values moves the
    raw coordinate while this token holds still. `crystal.ontology.geometry.basis_fingerprint()` is
    the exact identity; it walks the whole synset vocabulary, and its `_IC_REV_CACHE` is keyed on
    `(count, first name, last name)`, which does not move when the IC values change. So this
    function is the cheap default and states only what it verified; a caller that knows its
    coordinate exactly can pass its own token to `PooledScreen.place`. The corpus token carries
    `GEOMETRY_VERSION` for the same reason — the projection is `dense_vec @ B.T`, so it inherits
    what the dense coordinate means as well as which basis it was projected onto."""
    from crystal.ontology import geometry as _g
    if not corpus_basis:
        return ("geometry", _g.GEOMETRY_VERSION, "dense")
    from ember.signal import projection as _projection
    gen = _projection.basis_head(store)
    if gen is None:
        return None
    return ("geometry", _g.GEOMETRY_VERSION, "corpus", gen, None if k is None else int(k))


def _at_zoom(token: Optional[Tuple], width: int) -> Optional[Tuple]:
    """A corpus token whose zoom was left unstated, completed from the plane's own width.

    This is a measurement of the plane. On the corpus arm the frame is `W @ B[:k].T`, so `F` and `k`
    are the same integer by construction, and reading it off the plane agrees with the projection
    that produced it. That is what makes it available here and unavailable in `coordinate_id`, where
    there is no plane to read.

    Everything else passes through untouched: a token that already states a zoom keeps it (the
    caller measured it), `None` stays `None` — an unlabelled plane is still unlabelled
    ([[absence-is-not-an-affirmative-claim]]) — and the raw dense token has no zoom slot to fill."""
    if (isinstance(token, tuple) and len(token) == 5 and token[2] == "corpus"
            and token[4] is None):
        return token[:4] + (int(width),)
    return token


class PooledScreen:
    """An accumulating Screen: feed it one plane per placement, read one spectrum over everything.

    `F` and the basis token are pinned by the first plane that pools and hold from then on.
    `place()` never raises: pooling is bookkeeping alongside whatever the caller was doing, so the
    outcome is returned and counted, and the caller's own work carries on.
    """

    def __init__(self, name: str, *, whiten: bool = False):
        self.name = str(name)
        self._whiten = bool(whiten)
        self._lock = threading.Lock()
        self._acc = None                                  # ember.optics.accumulator; F pinned lazily
        self._planes = 0                                  # planes POOLED
        self._unrecorded = 0                              # pooled planes that stated no basis
        self._basis: Optional[Tuple] = None               # the pinned coordinate token
        self._basis_pinned = False                        # distinct from `_basis is None`
        self._drops: Dict[str, int] = {r: 0 for r in DROP_REASONS}
        self._logged: set = set()                         # log each reason ONCE — loud, not a flood

    # ── identity ─────────────────────────────────────────────────────────────────────────────
    @property
    def artifact_id(self) -> str:
        return "%s.screen" % self.name

    # ── placing ──────────────────────────────────────────────────────────────────────────────
    def place(self, plane, *, basis: Optional[Tuple] = None) -> str:
        """Pool one `(T_p, F)` plane. Returns `"pooled"` or `"dropped:<reason>"`. Never raises.

        `basis` is the coordinate the caller built the plane in (see `coordinate_id`). `None` means
        the caller did not say, which is recorded as such and read as unstated rather than as
        agreement.

        A corpus token that did not state its zoom is completed from the plane (`_at_zoom`), because
        on that arm `F` is the zoom. That is what keeps two frames in genuinely different
        coordinates from minting the same token under a per-read `k`, and the token is what a peer
        or a later turn compares against. The width pin below catches the same disagreement, but
        only within this Screen.
        """
        try:
            import numpy as np
            W = np.asarray(plane, dtype=float)
            if W.ndim != 2 or W.shape[0] < 1 or W.shape[1] < 2 or not np.isfinite(W).all():
                return self._drop("malformed", "shape=%r" % (getattr(W, "shape", None),))
            basis = _at_zoom(basis, int(W.shape[1]))
            with self._lock:
                if self._acc is None:
                    from ember.optics import accumulator
                    self._acc = accumulator(int(W.shape[1]), whiten=self._whiten)
                    self._basis, self._basis_pinned = basis, True
                if int(W.shape[1]) != int(self._acc.F):
                    # A different feature axis: the plane is counted as dropped and left the shape
                    # it arrived in.
                    return self._drop("width", "F=%d pooled=%d" % (int(W.shape[1]), int(self._acc.F)))
                if basis is not None and self._basis is not None and basis != self._basis:
                    # Same width, different meaning — the drop `F` alone cannot see.
                    return self._drop("basis", "%r vs pooled %r" % (basis, self._basis))
                self._acc.add(W)
                self._planes += 1
                if basis is None:
                    self._unrecorded += 1
                return "pooled"
        except Exception as e:                                        # noqa: BLE001 — see class doc
            return self._drop("error", repr(e))

    def _drop(self, reason: str, detail: str) -> str:
        """Count it, and log it once per reason. Called with `_lock` held or not, so it takes no
        lock itself: taking one here would re-enter, and `+=` on an int under the GIL is outside
        what the lock protects."""
        self._drops[reason] = self._drops.get(reason, 0) + 1
        if reason not in self._logged:
            self._logged.add(reason)
            _log.warning("PooledScreen %s DROPPED a plane (%s): %s — pooled=%d; "
                         "further %s drops are counted, not logged",
                         self.artifact_id, reason, detail, self._planes, reason)
        return "dropped:%s" % reason

    # ── reading ──────────────────────────────────────────────────────────────────────────────
    def summary(self) -> Optional[Dict[str, Any]]:
        """What this Screen holds, without asking the instrument to resolve it. `None` only when
        nothing has ever been placed.

        The split from `accumulated()` is about cost. `accumulated()` reads a pooled `(F, F)`
        covariance — at the raw dense coordinate that is `(2048, 2048)`, an eigendecomposition
        measured in seconds. `planes`, `T`, `F`, the drops and the basis are bookkeeping the
        accumulator already keeps, so this answers "did it pool, and is the pool growing?" for the
        price of reading fields. A placer reporting what it just did uses this; a caller asking
        whether the read is certified yet uses `accumulated()`."""
        dropped = sum(self._drops.values())
        if self._acc is None and self._planes == 0 and dropped == 0:
            return None
        return {
            "screen": self.artifact_id,
            "planes": self._planes,
            "T": int(getattr(self._acc, "T", 0) or 0),
            "F": int(self._acc.F) if self._acc is not None else None,
            "dropped": dropped,
            "drops": {k: v for k, v in self._drops.items() if v},
            "basis": self._basis,
            "basis_recorded": bool(self._basis_pinned and self._basis is not None),
            "unrecorded_planes": self._unrecorded,
        }

    def accumulated(self) -> Optional[Dict[str, Any]]:
        """What this Screen currently holds — the pooled read, what it dropped, and which basis.

        `None` only when nothing has ever been placed. If planes were placed and every one was
        dropped, that is a reading and it is returned: `planes=0, dropped=N`. An all-dropped Screen
        and an untouched Screen are therefore distinguishable.

        `read` is False when the instrument had nothing to resolve; the read keys (`T`, `F`, `band`,
        `k_signal`, `interval`, `certified`) are then absent rather than filled with zeros.

        The read itself happens in `ember.optics`, the one door onto the instrument
        ([[one-instrument-enforced]]) — the other entry points apply an entropy fold guard that
        destroys a sparse carrier. This asks the instrument and reports what it says."""
        out = self.summary()
        if out is None:
            return None
        out["read"] = False
        if self._acc is None or self._planes == 0:
            return out
        from ember.optics import accumulated_read
        read = accumulated_read(self._acc)
        if read is None:
            return out
        out.update(read)
        out["read"] = True
        return out


# ── the node's own Screen ────────────────────────────────────────────────────────────────────────
#: One per node id per process, exactly as `Delegate._CACHE` is one per delegate id per process.
_NODE_SCREENS: Dict[str, PooledScreen] = {}
_NODE_LOCK = threading.Lock()


def node_id() -> str:
    """Whose headroom this is. `EMBER_NODE_ID` is the identity the mesh plane keys on and the one
    `worker.py` requires to start; `EMBER_HOST_ID` and then `local` are the same fallback chain
    `delegate._host_id` uses, so an unidentified process still has a Screen to place on and the
    reading is kept."""
    return (os.getenv("EMBER_NODE_ID") or os.getenv("EMBER_HOST_ID") or "local").strip() or "local"


def node_screen(node: Optional[str] = None) -> PooledScreen:
    """The node's accumulating Screen — `node.<EMBER_NODE_ID>.screen`.

    This is the node's own state — its headroom, its occupancy — which is one fact about one box
    rather than one observer's view of the world, so there is one Screen per node because there is
    one node. `delegate.py` holds the complementary rule for cognition: one Screen per agent, since
    pooled attention turns first-hand memory into hearsay and a busy neighbour would displace a
    quiet agent's memory.

    In-process only, as the delegate's accumulator is. `Delegate.save_screen` persists the
    forgetting screen (traces + tick); the accumulator behind `Delegate._acc` lives in memory, and
    so does this one. A pooled `(2048, 2048)` covariance is 32 MB, and a small summary stored beside
    the real thing would be the side-car [[everything-is-an-artifact]] rules out. `artifact_id`
    names where it would live."""
    nid = (node or node_id()).strip() or "local"
    with _NODE_LOCK:
        s = _NODE_SCREENS.get(nid)
        if s is None:
            s = PooledScreen("node.%s" % nid)
            _NODE_SCREENS[nid] = s
        return s


def reset_node_screens() -> None:
    """Test seam — drop every node Screen. Its callers are tests; production code holds its
    Screens for the life of the process."""
    with _NODE_LOCK:
        _NODE_SCREENS.clear()


__all__ = ["DROP_REASONS", "PooledScreen", "coordinate_id", "node_id", "node_screen",
           "reset_node_screens"]
