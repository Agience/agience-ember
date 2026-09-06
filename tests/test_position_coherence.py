"""`optics.position_coherence` — the per-position read, and the resolver built on it.

The read answers a one-row question: does the row at `position` belong where it sits? It is taken at
a position rather than over a frame because `projection.coherence` is a mean over ~w ordered pairs,
so replacing one row moves two of them and dilutes a one-row question by w/2 at whatever window it
is given. On the canon at windows 6..48 the frame read separates by 0.1-0.36 sd at a chance
win-rate; the same frames read at the position separate by 0.53-0.72 sd, 72-82%.

Rows are unit-normalised inside the read. `R = Re(G)**2` on raw rows is dominated by ||row||, and
canon sections span 90-100x in norm, so without that normalisation the read measures section length
rather than belonging (the underlying signal is neighbour cosine 0.245 against foreign 0.094,
Cohen d=0.78).

`revision_resolver()` is the read-time head resolver `mantle.shard.cache` takes as its `resolve`
seam: head is resolved against the reader's own recall frame rather than decided at write time.
"""
from __future__ import annotations

import numpy as np
import pytest

from ember import optics


def _frame(seed=0, n=12, f=64):
    """A frame with real ordered structure: neighbours share a drifting component."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, f))
    for i in range(1, n):
        base[i] = 0.6 * base[i - 1] + 0.4 * base[i]      # ordered drift => neighbours are alike
    return base


def _alien(seed=99, f=64):
    return np.random.default_rng(seed).normal(size=f)


# ── the read ─────────────────────────────────────────────────────────────────────────────────
def test_a_row_that_BELONGS_reads_above_the_frames_null_and_an_alien_below():
    """The zero is computed — it is the frame's own off-diagonal mean, not a chosen cut — so the
    sign carries the verdict. A comparison against a typed constant would not track the frame.
    """
    W = _frame()
    p = 6
    z_true = optics.position_coherence(W, p)
    alt = W.copy()
    alt[p] = _alien()
    z_alien = optics.position_coherence(alt, p)
    assert z_true is not None and z_alien is not None
    assert z_true > 0 > z_alien, (z_true, z_alien)


def test_the_verdict_does_NOT_depend_on_the_WINDOW_SIZE():
    """The scale property. Measured on the canon the separation is flat across w = 6..48, so there
    is no window to choose — a bigger frame sharpens the null, it does not change the answer.

    A read that worked at only one width would be a hidden constant, and a caller would have to pick
    one to get an answer.
    """
    verdicts = []
    for n in (5, 8, 12, 20, 32):
        W = _frame(n=n)
        p = n // 2
        alt = W.copy(); alt[p] = _alien()
        verdicts.append((optics.position_coherence(W, p) > 0,
                         optics.position_coherence(alt, p) < 0))
    assert all(t and a for t, a in verdicts), verdicts


def test_UNNORMALISED_rows_cannot_break_it():
    """Scaling rows by wildly different factors leaves the verdict unchanged.

    `Re(G)**2` on raw rows is dominated by ||row||, so without the internal unit-normalisation the
    read measures row length, and on the canon (90-100x norm spread) it returns chance.
    """
    W = _frame()
    p = 6
    before = optics.position_coherence(W, p)
    scaled = W * np.linspace(0.01, 100.0, len(W))[:, None]      # 10,000x spread
    after = optics.position_coherence(scaled, p)
    assert before is not None and after is not None
    assert abs(before - after) < 1e-9, (before, after)


def test_it_GROUNDS_OUT_rather_than_returning_a_number_it_cannot_support():
    """Three computed groundings-out: a frame too short to have neighbours, a position off the end,
    and a frame with no off-diagonal spread to standardise against. Each is an absence of
    measurement — the read has nowhere to stand, so there is no reading to give.

    Each returns None. Returning 0.0 would be indistinguishable from "sits exactly at the null".
    """
    assert optics.position_coherence(np.zeros((1, 8)), 0) is None       # no neighbour
    assert optics.position_coherence(_frame(), 999) is None             # off the end
    identical = np.tile(np.ones(8), (6, 1))                             # no spread
    assert optics.position_coherence(identical, 3) is None


# ── the resolver ─────────────────────────────────────────────────────────────────────────────
def _rev(rid, vec):
    return {"id": rid, "vector": vec}


def _cluster(n=9, f=64, seed=2):
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=f)
    return [centre + 0.35 * rng.normal(size=f) for _ in range(n)], centre, rng


def test_the_resolver_PICKS_the_revision_that_reads_above_the_null():
    """The resolver names the revision that reads above the frame's own null. Returning every
    revision regardless is indistinguishable from mantle's no-resolver default, so the seam would
    look wired while doing nothing."""
    rows, centre, rng = _cluster()
    frame = {"n%d" % i: v for i, v in enumerate(rows)}
    frame["good"] = centre + 0.3 * rng.normal(size=64)
    frame["bad"] = rng.normal(size=64) * 3.0
    resolve = optics.revision_resolver()
    got = resolve("root", [_rev("good", None), _rev("bad", None)], {}, frame)
    assert got == ["good"], got


def test_the_resolver_lets_EVERYTHING_STAND_when_nothing_reads_above_the_null():
    """When no candidate beats the frame's own null, every revision stands — the same default
    mantle has with no resolver at all. There is nothing measured to separate them, so nothing
    moves.

    `max()` over the candidates unconditionally always returns someone, which is a headcount in a
    new coat.
    """
    rows, centre, rng = _cluster(seed=5)
    frame = {"n%d" % i: v for i, v in enumerate(rows)}
    frame["a"] = rng.normal(size=64) * 3.0
    frame["b"] = rng.normal(size=64) * 3.0
    resolve = optics.revision_resolver()
    revs = [_rev("a", None), _rev("b", None)]
    assert sorted(resolve("root", revs, {}, frame)) == ["a", "b"]


def test_an_ABSENT_or_TINY_frame_hides_nothing():
    """A resolver with nothing to measure against lets every revision stand. Returning [] when there
    is no frame would make every revision vanish from reads the moment the recall set was small."""
    resolve = optics.revision_resolver()
    revs = [_rev("a", None), _rev("b", None)]
    assert sorted(resolve("root", revs, {}, {})) == ["a", "b"]          # no frame at all
    assert sorted(resolve("root", revs, {}, {"a": np.ones(8), "b": np.ones(8)})) == ["a", "b"]


def test_a_revision_with_NO_VECTOR_costs_the_OTHERS_NOTHING():
    """An unreadable candidate leaves the others intact. Skipping it would let a revision disappear
    from every read because of a missing embedding — an absence of measurement becoming a verdict."""
    rows, centre, rng = _cluster(seed=7)
    frame = {"n%d" % i: v for i, v in enumerate(rows)}
    frame["has"] = centre + 0.3 * rng.normal(size=64)
    resolve = optics.revision_resolver()
    revs = [_rev("has", None), {"id": "none"}]          # "none" has no vector in the frame
    assert sorted(resolve("root", revs, {}, frame)) == ["has", "none"]


def test_it_gives_NO_VERDICT_on_a_frame_with_no_ordered_structure():
    """The universality control. `_frame` is synthetic AR(1) drift, a construction that guarantees
    neighbours are alike, and `optics.resolvable` records a synthetic frame whose separation did not
    survive on real data. A read validated only on a fixture built to satisfy it has proven nothing.

    So on rows with no ordered relationship a true row reads no better than an alien one. Reliable
    separation here would mean the read responds to something other than order, and every positive
    result above would be suspect — a statistic that scores any row against a frame would "win"
    here too.
    """
    rng = np.random.default_rng(4)
    wins = 0
    trials = 60
    for t in range(trials):
        W = rng.normal(size=(16, 64))          # i.i.d. rows: no ordered structure to find
        p = 8
        alt = W.copy(); alt[p] = rng.normal(size=64)
        a = optics.position_coherence(W, p)
        b = optics.position_coherence(alt, p)
        if a is not None and b is not None and a > b:
            wins += 1
    rate = wins / trials
    assert 0.35 < rate < 0.65, (
        "on a frame with no ordered structure the read picked a winner %.0f%% of the time — it is "
        "responding to something other than order" % (100 * rate))


def test_the_ORDERED_case_is_measurably_different_from_the_unordered_one():
    """The other half of the control above, which an inert read would also pass. On a frame that
    does carry ordered structure the same procedure separates.
    """
    rng = np.random.default_rng(4)
    wins = 0
    trials = 60
    for t in range(trials):
        W = _frame(seed=t, n=16)
        p = 8
        alt = W.copy(); alt[p] = rng.normal(size=W.shape[1])
        a = optics.position_coherence(W, p)
        b = optics.position_coherence(alt, p)
        if a is not None and b is not None and a > b:
            wins += 1
    assert wins / trials > 0.8, (
        "the read does not separate even on a frame built with ordered structure: %d/%d"
        % (wins, trials))


# ── belonging(): the unordered read the cache uses ───────────────────────────────────────────
def test_belonging_separates_WITHOUT_any_ordered_axis():
    """`mantle.shard.cache` has no sequence and no adjacency, so a lag-1 ordered read there would
    require an order to be imposed. `belonging` reads the candidate's whole off-diagonal row against
    the same null and needs none.

    The frame here is a cluster plus an alien, deliberately unordered, so the test cannot be
    satisfied by adjacency: reusing the ordered terms would read whichever two rows happen to sit
    beside the candidate in the array.
    """
    rng = np.random.default_rng(2)
    centre = rng.normal(size=64)
    W = np.vstack([centre + 0.35 * rng.normal(size=64) for _ in range(9)])   # one cluster
    p = 4
    z_member = optics.belonging(W, p)
    alt = W.copy(); alt[p] = rng.normal(size=64) * 3.0                        # an alien
    z_alien = optics.belonging(alt, p)
    assert z_member is not None and z_alien is not None
    assert z_member > 0 > z_alien, (z_member, z_alien)


def test_belonging_gives_NO_VERDICT_when_the_frame_has_no_cluster():
    """The universality control for the unordered read: on i.i.d. rows there is no neighbourhood to
    belong to, so a member reads no better than an alien. A statistic that scores any row against
    any frame would separate here.
    """
    rng = np.random.default_rng(9)
    wins = 0
    for _ in range(60):
        W = rng.normal(size=(12, 64))
        p = 5
        alt = W.copy(); alt[p] = rng.normal(size=64)
        a, b = optics.belonging(W, p), optics.belonging(alt, p)
        if a is not None and b is not None and a > b:
            wins += 1
    assert 0.35 < wins / 60 < 0.65, wins / 60


def test_belonging_is_SCALE_INVARIANT_and_GROUNDS_OUT_on_a_degenerate_frame():
    """Same two guarantees as the ordered read: ||row|| does not decide, and a frame with no spread
    grounds out rather than answering 0.0.

    Dropping the internal unit-normalisation makes the read measure row length, which on the canon
    returns chance; widening the degeneracy bound past float resolution makes an identical-row frame
    return a confident 0.0 where nothing was measured.
    """
    rng = np.random.default_rng(3)
    centre = rng.normal(size=64)
    W = np.vstack([centre + 0.35 * rng.normal(size=64) for _ in range(9)])
    before = optics.belonging(W, 4)
    after = optics.belonging(W * np.linspace(0.01, 100.0, len(W))[:, None], 4)
    assert abs(before - after) < 1e-9, (before, after)
    assert optics.belonging(np.tile(np.ones(8), (6, 1)), 3) is None
    assert optics.belonging(np.zeros((2, 8)), 0) is None

