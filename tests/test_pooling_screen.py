"""The node's own Screen — a state reading always has somewhere to land, and a drop is never silent.

Each test pins one property of the node-level pooling Screen:

  · a `(5, 2048)` state frame handed to the node's accumulating Screen is accepted and reported;
  · a mismatched frame width is dropped, logged, and counted, distinct from a silent drop folded
    into the accumulated pool;
  · a Screen that has dropped everything it was ever handed reports that state, distinguishable
    from a Screen nobody has touched;
  · pooled `T` grows monotonically with each reading, and the certified band tightens as it grows;
  · the basis a Screen pooled under is recorded, so a basis rebuilt to the same width is still
    detectable, which `F` alone cannot show;
  · `op.measure` places its reading on the node's Screen rather than only returning it to a caller.

Hermetic: the ontology coordinate is stubbed exactly as `test_state_signal.py` stubs it. Nothing
here reads the live corpus and nothing here touches a running service.
"""
import logging

import numpy as np
import pytest

from ember.signal import pooling
from ember.signal import state as S


class _Syn:
    def __init__(self, name, vec):
        self._n, self.vec = name, vec

    def name(self):
        return self._n


D = 2048
GB = 2 ** 30
FLOOR = 20 * GB


@pytest.fixture()
def coords(monkeypatch):
    """A sparse 2048-wide coordinate per READING concept — the real one's shape (~6 nnz)."""
    rng = np.random.default_rng(11)
    vecs = {}
    for syn, _key in S.READINGS:
        v = np.zeros(D)
        v[rng.choice(D, 6, replace=False)] = rng.normal(size=6)
        vecs[syn] = v / np.linalg.norm(v)
    from crystal.ontology import geometry as g
    from crystal.ontology import driver as wn
    monkeypatch.setattr(wn, "synset", lambda n: (_Syn(n, vecs[n]) if n in vecs else None))
    monkeypatch.setattr(g, "load_ic", lambda: None)
    monkeypatch.setattr(g, "dense_vec", lambda s, ic, center=None: s.vec)
    return vecs


@pytest.fixture()
def node(tmp_path, monkeypatch):
    (tmp_path / "lattice.db").write_bytes(b"x" * 4096)
    (tmp_path / "cas").mkdir()
    (tmp_path / "cas" / "aa").mkdir()
    (tmp_path / "cas" / "aa" / "blob").write_bytes(b"y" * 2048)
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(tmp_path))
    monkeypatch.setenv("EMBER_SQLITE_DB", "lattice.db")
    monkeypatch.setenv("MANTLE_CACHE_MIN_FREE_GB", "20")
    monkeypatch.delenv("EMBER_CACHE_MIN_FREE_GB", raising=False)
    monkeypatch.setenv("EMBER_NODE_ID", "test-node")
    pooling.reset_node_screens()
    yield tmp_path
    pooling.reset_node_screens()


# ═════ Measurement 1 — a (5, 2048) state frame is accepted, not silently dropped ══════════════════

def test_a_state_frame_is_accepted_by_the_node_screen(node, coords):
    """The node's Screen exists so a reading that belongs to no conversation still has an owner, and
    its acceptance is observable on the Screen's own report rather than inferred from silence."""
    W = S.place(None, S.envelope(None, free_bytes=0))
    assert W.shape == (len(S.READINGS), D)

    sc = pooling.node_screen()
    assert sc.artifact_id == "node.test-node.screen"
    assert sc.summary() is None, "an untouched Screen must not claim to hold anything"

    assert sc.place(W, basis=pooling.coordinate_id(None, corpus_basis=False)) == "pooled"
    s = sc.summary()
    assert s["planes"] == 1 and s["F"] == D and s["T"] == W.shape[0]
    assert s["dropped"] == 0 and s["drops"] == {}


# ═════ Measurement 2 — a mismatched F is dropped loudly: the count is visible ══════════════════════

def test_a_mismatched_width_is_dropped_LOUDLY_and_counted(node, coords, caplog):
    """A width mismatch is dropped rather than silently absorbed into the pool: the drop is counted,
    logged, and legible on the Screen's own report, so a stale spectrum never reports as current."""
    sc = pooling.PooledScreen("t.width")
    tok = pooling.coordinate_id(None, corpus_basis=False)
    assert sc.place(np.random.default_rng(1).normal(size=(6, D)), basis=tok) == "pooled"

    with caplog.at_level(logging.WARNING, logger="ember.signal.pooling"):
        narrow = np.random.default_rng(2).normal(size=(6, 195))     # the pre-rebuild width, exactly
        assert sc.place(narrow, basis=tok) == "dropped:width"
        assert sc.place(narrow, basis=tok) == "dropped:width"

    assert any("DROPPED" in r.getMessage() for r in caplog.records), \
        "the drop was silent — this is the original defect, restated"

    s = sc.accumulated()
    assert s["planes"] == 1, "a 195-wide plane must not be reshaped into a 2048-wide pool"
    assert s["dropped"] == 2 and s["drops"] == {"width": 2}
    assert s["F"] == D


def test_an_all_dropped_screen_does_not_report_as_an_untouched_one(node):
    """A Screen that dropped everything it was ever handed reports that state distinctly from one
    nobody has touched: `accumulated()` is `None` until a placement is attempted, then reports
    `planes == 0` and the drop count rather than reverting to `None`
    ([[absence-is-not-an-affirmative-claim]])."""
    sc = pooling.PooledScreen("t.all-dropped")
    assert sc.accumulated() is None
    assert sc.place(np.zeros((3,)), basis=None) == "dropped:malformed"   # 1-D: not a plane
    got = sc.accumulated()
    assert got is not None and got["planes"] == 0 and got["drops"] == {"malformed": 1}
    assert got["read"] is False


# ═════ Measurement 3 — the Screen accumulates: pooled T grows and is reported ══════════════════════

def test_the_screen_accumulates_and_says_so(node, coords):
    """Pooled `T` grows monotonically with each reading, and `accumulated()` reports the accumulated
    growth rather than the last frame placed — this is what lets the certified band tighten as more
    readings pool, rather than resetting to `sqrt(F/T) + F/T` on every read."""
    sc = pooling.node_screen()
    tok = pooling.coordinate_id(None, corpus_basis=False)
    ts, bands = [], []
    for free in (FLOOR + GB, FLOOR, FLOOR - GB, FLOOR - 5 * GB, 0):
        W = S.place(None, S.envelope(None, free_bytes=free))
        assert sc.place(W, basis=tok) == "pooled"
        s = sc.summary()
        ts.append(s["T"])
        bands.append(sc._acc.band())
    assert ts == [5, 10, 15, 20, 25], "pooled T did not grow with the readings: %r" % ts
    assert sc.summary()["planes"] == 5
    assert bands == sorted(bands, reverse=True), \
        "the certified band must tighten as T grows: %r" % bands


# ═════ Measurement 4 — the basis is recorded, and a different one is detectable ════════════════════

def test_the_screen_records_which_basis_it_pooled_under(node, coords):
    """A Screen records which basis it pooled under, because `F` alone cannot answer that question: a
    basis rebuilt to the same width, or a re-enrichment that re-scales every `dense_vec` at a
    constant D=2048, changes what the coordinate means while every shape still agrees."""
    sc = pooling.node_screen()
    dense = pooling.coordinate_id(None, corpus_basis=False)
    assert dense is not None
    W = S.place(None, S.envelope(None, free_bytes=0))
    assert sc.place(W, basis=dense) == "pooled"

    s = sc.summary()
    assert s["basis"] == dense and s["basis_recorded"] is True
    assert s["unrecorded_planes"] == 0

    # Same width, different coordinate — the drop `F` cannot see.
    other = ("geometry", "geom-vNEXT", "dense")
    assert other != dense and sc.place(W, basis=other) == "dropped:basis"
    s = sc.summary()
    assert s["planes"] == 1 and s["drops"] == {"basis": 1}


def test_an_unstated_basis_is_recorded_as_unstated_never_as_agreement(node, coords):
    """[[verification-that-cannot-fail]]. If an omitted basis defaulted to "matches whatever is
    pinned", the check could never fire and would prove nothing, so an unstated basis still pools —
    discarding it would drop real data from the one existing caller — but the pool is counted, and a
    Screen whose first plane was unstated reports `basis_recorded=False` rather than inventing an
    identity for it."""
    sc = pooling.PooledScreen("t.unstated")
    W = np.random.default_rng(3).normal(size=(4, 64))
    assert sc.place(W, basis=None) == "pooled"
    s = sc.summary()
    assert s["basis"] is None and s["basis_recorded"] is False and s["unrecorded_planes"] == 1
    # A later plane that does state a basis is not retro-applied to the unlabelled pool.
    assert sc.place(W, basis=("geometry", "geom-v1", "dense")) == "pooled"
    assert sc.summary()["basis_recorded"] is False


def test_an_unverifiable_store_yields_no_corpus_basis_token(node):
    """`freshness`'s own rule: a store that cannot report its freshness is not cached from, and is
    not treated as current. A store that cannot resolve the `geom.corpus-basis` lineage gives
    `None` — which means unrecorded, and `None` is never equal to a real token, so it can never be
    read as agreement with a pooled basis.

    The token is the generation's own artifact id; see `test_basis_generations.py` for why a
    freshness stamp beside it cannot serve as the token."""
    class _Mute:
        pass
    assert pooling.coordinate_id(_Mute(), corpus_basis=True) is None
    assert pooling.coordinate_id(_Mute(), corpus_basis=False) is not None, \
        "the raw dense coordinate does not come from an artifact and is always nameable"


# ═════ Measurement 5 — op.measure places: twice leaves more pooled planes than once ════════════════

def test_op_measure_places_its_reading_it_does_not_only_return_it(node, coords):
    """`op.measure` places its reading on the node's Screen rather than only returning the encoded
    frame to its caller — a reading returned to nobody has not been placed. Invoking it twice leaves
    the Screen holding strictly more than invoking it once, and that difference is the only evidence
    that distinguishes placing from returning."""
    sc = pooling.node_screen()
    assert sc.summary() is None

    first = S.measure(None, free_bytes=0)
    assert first["placed"] is True
    assert first["screen"]["id"] == "node.test-node.screen"
    assert first["screen"]["outcome"] == "pooled"
    p1, t1 = first["screen"]["pooled"]["planes"], first["screen"]["pooled"]["T"]

    second = S.measure(None, free_bytes=0)
    p2, t2 = second["screen"]["pooled"]["planes"], second["screen"]["pooled"]["T"]
    assert (p2, t2) == (p1 + 1, t1 + len(S.READINGS)), \
        "the second reading did not accumulate: %r -> %r" % ((p1, t1), (p2, t2))

    # And the Screen the operator reported is the node's own, still holding both readings.
    assert pooling.node_screen().summary()["planes"] == 2
    assert first["screen"]["pooled"]["basis"] == pooling.coordinate_id(None, corpus_basis=False)


def test_op_measure_still_decides_nothing_and_calls_nothing(node, coords):
    """The distinction the design draws: a scheduler calls a named function; an originator places a
    signal. Placement does not smuggle in a decision, a loop, or an invocation."""
    import inspect
    out = S.measure(None, free_bytes=0)
    assert not any(k in out for k in ("verdict", "severity", "should_evict", "action"))
    for mod in (S, pooling):
        src = inspect.getsource(mod)
        assert "while True" not in src and "time.sleep" not in src, \
            "%s grew a loop — that is the scheduler this design forbids" % mod.__name__
    src = inspect.getsource(S)
    assert "op.reclaim" not in src.replace("`op.reclaim`", "")
