"""`op.measure` — the node's state, placed as a signal.

This module measures and places. It runs no loop, holds no clock, and calls no reclamation
operator; what fires on the reading is decided by coupling, at the offer. A caller invokes it
through activation, as it would any other tekton.

## The mechanism

`ember/signal/projection.py::frame(store, names, energy=…)` takes concepts and per-row activation
energy and scales each row by `sqrt(energy)`: a beam's amplitude is the square root of its energy,
and `‖row‖²` is that row's energy. That is what a state reading needs —

    the concept names what is measured; the energy is the measurement.

So a reading is the same call `seeds_from_text` → `compose` makes for words, with a byte count where
a word's information content would be. A state reading and a question are then the same kind of
thing in the same basis, they accumulate on one Screen, and one instrument reads both.

## Scarcity is amplitude

Ample headroom is a shortfall of zero, so the `headroom` row is scaled by `sqrt(0)` and carries no
energy: the row is present, it stays below the instrument's noise floor, and nothing couples to it. A
deficit twice as large is a signal with twice the energy that couples twice as hard, so urgency is
physical rather than an enum, and the module holds no comparison of its own
([[one-resolution-not-thresholds]], [[no-arbitrary-caps]],
[[never-impose-knowledge-derive-it]]).

## Every energy is a live measurement in bytes

Two readings in one frame are comparable when they are the same quantity, so everything here is
bytes, and every one is read rather than typed:

  · free / used / total       `shutil.disk_usage` on the store volume, via
                              `prism.envelope.disk_free_bytes` — the same reader
                              `runtime/breeding.py` uses. `None` means not measured, and stays
                              distinct from a measured 0.
  · the free-space floor      `mantle.db.content_cache._min_free_bytes()` — the store's own
                              reader of the declared policy (`EMBER_CACHE_MIN_FREE_GB=20` in
                              `_fleet/peers/71/ember/serve.env`: *"evict LRU content to keep >=20GB
                              free on D:; content is durable in S3 / re-ingestable"*). An unset
                              policy yields 0, so no declared floor means no shortfall and the
                              headroom row stays silent.
  · lattice / cache bytes     `os.stat` on the store's own files and an `os.scandir` walk of the CAS
                              tree — the same walk `content_cache._ensure_free` performs for this
                              question. Its cost is reported as `walk_seconds` and the file counts.

## What is placed

Demand and occupancy. The volume's total capacity is a declaration about the box that holds still as
the node works, so placing it would add a large constant to every frame and drown the readings that
move. `total_bytes` and `used_bytes` are reported in the envelope for a facet to present, and are
not rows.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

# ── the roster: which concept names which reading ────────────────────────────────────────────────
# Order is data: axis 0 of a screen is ordered and coherence is a lag-1 statistic, so this sequence
# is part of the measurement. Every entry is (synset, envelope key). The synsets are `*.n.01` names
# that carry a real coordinate in the store; an `oewn-*` synset returns an all-zero `dense_vec`,
# which `projection.frame` drops, so a reading named with one would contribute nothing.
READINGS: Tuple[Tuple[str, str], ...] = (
    ("headroom.n.01",  "shortfall_bytes"),   # free space missing against the declared floor
    ("occupancy.n.01", "store_bytes"),       # everything this node has resident under its store root
    ("lattice.n.01",   "lattice_bytes"),     # the index footprint — durable, not evictable
    ("cache.n.01",     "evictable_bytes"),   # the local CAS/content mass — the evictable part
    ("eviction.n.01",  "eviction_bytes"),    # the shortfall eviction could meet
)


def _dir_bytes(path: str) -> Tuple[int, int]:
    """`(bytes, files)` under `path`, by `os.scandir` walk. `(0, 0)` for an absent tree.

    The same walk `content_cache._ensure_free` performs over the CAS to answer this question, so
    there is one mechanism for it rather than two. Its cost is returned to the caller as a file
    count and timed by `envelope`, so an unbounded probe carries its own price tag."""
    total = files = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue                 # a file that vanished mid-walk carries no reading
        except OSError:
            continue
    return total, files


def _floor_bytes() -> int:
    """The declared free-space floor, in bytes, read from the store's own policy reader.

    `content_cache._min_free_bytes()` is the single definition of this policy
    (`MANTLE_CACHE_MIN_FREE_GB`, falling back to the deployed `EMBER_CACHE_MIN_FREE_GB`), and it is
    what the cache's own eviction path obeys. Reading it here means the shortfall this operator
    reports is measured against the floor that is actually enforced.

    Unset ⇒ 0 ⇒ no declared floor ⇒ no shortfall, so an absent policy produces no demand."""
    from mantle.db.content_cache import _min_free_bytes
    return int(_min_free_bytes())


def envelope(store=None, *, root: Optional[str] = None,
             free_bytes: Optional[int] = None) -> Dict[str, Any]:
    """The live envelope of this node, in bytes. Pure measurement — decides nothing.

    `root` defaults to `EMBER_SQLITE_DIR` (what `sqlite_store.open_sqlite_store` itself resolves the
    store root from), so this measures the volume and the tree the store actually occupies.

    `free_bytes` substitutes the free-space reading and nothing else, so the coupling can be
    exercised across the range of envelopes this node can be in. The floor, the footprint and the
    arithmetic all run as they normally do. It is named for the reading it supplies rather than for
    a scenario, because that is what a caller is stating when it passes one.

    Every key that could not be measured is `None` rather than 0, so a path that does not exist and
    a full disk produce different output ([[absence-is-not-an-affirmative-claim]]). The same rule is
    held by `prism/envelope.py::disk_free_bytes` and `runtime/breeding.py`."""
    from prism.envelope import disk_free_bytes
    root = root or os.getenv("EMBER_SQLITE_DIR") or ""
    db_name = os.getenv("EMBER_SQLITE_DB") or "lattice.db"

    free = int(free_bytes) if free_bytes is not None else disk_free_bytes(root)
    total = used = None
    try:
        import shutil
        u = shutil.disk_usage(root or ".")
        total, used = int(u.total), int(u.used)
    except OSError:
        pass

    # The lattice index: the db and its write-ahead log siblings. O(1) — three stats.
    lattice_bytes = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            lattice_bytes += os.stat(os.path.join(root, db_name + suffix)).st_size
        except OSError:
            continue

    t0 = time.time()
    cas_bytes, cas_files = _dir_bytes(os.path.join(root, "cas"))
    content_bytes, content_files = _dir_bytes(os.path.join(root, "content"))
    walk_s = time.time() - t0

    floor = _floor_bytes()
    # The shortfall is a subtraction. `max(0, …)` says a surplus is zero demand rather than negative
    # demand, since negative energy is not a reading; it is not a branch on a threshold.
    shortfall = None if free is None else max(0, floor - int(free))
    evictable = cas_bytes + content_bytes
    # What eviction could release: the demand, bounded by the mass that exists to release. A `min`
    # of two measured quantities, derived rather than chosen.
    eviction = None if shortfall is None else min(shortfall, evictable)

    return {
        "root": root, "node": os.getenv("EMBER_NODE_ID"),
        "free_bytes": free, "free_measured": free is not None,
        "total_bytes": total, "used_bytes": used,
        "floor_bytes": floor, "floor_declared": floor > 0,
        "shortfall_bytes": shortfall,
        "lattice_bytes": lattice_bytes,
        "evictable_bytes": evictable,
        "store_bytes": lattice_bytes + evictable,
        "eviction_bytes": eviction,
        "cas_files": cas_files, "content_files": content_files,
        "walk_seconds": round(walk_s, 3),
        "measured_at": time.time(),
    }


def energies(env: Dict[str, Any]) -> List[float]:
    """The per-row activation energies for `READINGS`, in order, in bytes.

    An unmeasured reading contributes 0.0: the row carries no energy and therefore couples to
    nothing, which is what an absent measurement means physically. The envelope carries
    `free_measured: False` beside it, so absence stays distinguishable from a measured zero."""
    out: List[float] = []
    for _syn, key in READINGS:
        v = env.get(key)
        out.append(0.0 if v is None else float(v))
    return out


def place(store, env: Dict[str, Any]):
    """The reading as a beam: the `(T, F)` frame whose rows are the `READINGS` concepts and whose
    row energies are the measured bytes. `None` when nothing was measured.

    `corpus_basis=False` is load-bearing. A tekton's coupling basis (`match.tekton_basis_for`) is
    built from unprojected `geometry.dense_vec`, i.e. `(2048, k)`, and `absorb_transmit` couples
    when the basis's first axis equals the frame's feature axis. A corpus-projected frame has the
    basis generation's zoom as its feature axis, so the split over it yields no value. The two
    halves of the activation substrate live in different coordinate systems; see
    `projection.frame`.

    This function is pure given `env`: same envelope, same frame. That is what makes "would this
    have coupled?" answerable without touching the disk, and it is the split the tekton/organon rule
    asks for — the deciding is testable on its own."""
    from ember.signal import projection
    return projection.frame(store, [s for s, _ in READINGS],
                            energy=energies(env), corpus_basis=False)


def measure(store, *, root: Optional[str] = None, free_bytes: Optional[int] = None,
            screen=None) -> Dict[str, Any]:
    """`op.measure` — read the envelope and place the reading on the node's Screen.

    The placement is the act. The frame goes onto `pooling.node_screen()` —
    `node.<EMBER_NODE_ID>.screen` — which accumulates across readings as a delegate's screen
    accumulates across turns, so `T` grows and the certified band tightens on the node's own state
    rather than every reading starting from zero samples.

    Placing is not invoking. There is no loop here, no clock, and no caller for any reclamation
    operator: what fires on this signal is decided by coupling, at the offer. A scheduler calls a
    named function; an originator places a signal, and this is the placement.

    The basis is stated. `place()` builds in the raw dense coupling coordinate
    (`corpus_basis=False`), so that is the token handed to the Screen, and a plane arriving in a
    different coordinate is a counted, logged drop.

    The frame is also returned encoded (`prism.frames.encode_frame`) so it can ride inside a reach
    payload as the signal itself rather than as a stringified summary, as `prism/frames.py`
    specifies, and so the placement is testable. `energy` is reported per row alongside the concept
    that names it, so a facet such as `mesh.health` can present the reading without re-measuring
    anything ([[never-handroll-probes]]).

    `screen` substitutes the Screen placed on — a test seam, the same input-substitution shape as
    `free_bytes`. It states which Screen you are placing on."""
    from prism.conservation import energy as _energy
    from prism.frames import FRAME_KEY, encode_frame
    from ember.signal import pooling
    env = envelope(store, root=root, free_bytes=free_bytes)
    W = place(store, env)
    rows = [{"concept": s, "reading": k, "energy_bytes": e}
            for (s, k), e in zip(READINGS, energies(env))]
    out: Dict[str, Any] = {"envelope": env, "rows": rows, "placed": W is not None}
    if W is None:
        # Every reading was zero or unmeasured, so there is no beam. A frame of zeros would be a
        # signal nobody measured.
        out["reason"] = ("no reading carried energy — nothing to place "
                         "(floor_declared=%s, free_measured=%s)"
                         % (env["floor_declared"], env["free_measured"]))
        return out
    out["shape"] = [int(W.shape[0]), int(W.shape[1])]
    out["incident_energy"] = _energy(W)
    out[FRAME_KEY] = encode_frame(W)

    sc = pooling.node_screen() if screen is None else screen
    outcome = sc.place(W, basis=pooling.coordinate_id(store, corpus_basis=False))
    # `summary`, not `accumulated`: the certified read is an eigendecomposition of a (2048, 2048)
    # pooled covariance, measured in seconds. A placer reports what it placed; a caller that wants
    # the instrument's read asks the Screen for it, so neither pays the other's cost.
    out["screen"] = {"id": sc.artifact_id, "outcome": outcome, "pooled": sc.summary()}
    return out


__all__ = ["READINGS", "envelope", "energies", "place", "measure"]
