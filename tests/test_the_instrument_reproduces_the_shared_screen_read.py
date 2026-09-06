"""The instrument half of the screen-read gate — `prism.vectors/screen_read_vectors.json`, asserted.

## Why both halves have to exist

One question — how many independent directions a frame has — has two implementations, and they
cannot import each other. `ember.optics` reads it through the full entroptics instrument;
`mantle/search/beacon` reads it with a reduced, dependency-free engine that ships to installs which
never get entroptics. A shared vector file is what holds them to the same answer, and it only does
that while both halves read it. Read by one, it is a private fixture wearing a shared name.

The beacon half is `agience-mantle/tests/test_beacon_conformance.py`. This is the instrument half.
ember is where it belongs: `ember.optics` is the single sanctioned entroptics door
(`tests/test_one_instrument.py`), so this is the only package that can take an instrument read at all.

Each case rebuilds its frame from `(seed, N, F, planted, snr, dead_cols)` per the file's
`_how_to_build`, so no matrix is stored and both engines construct byte-identical input from the
same recipe.

The vectors are also the fine half of the entroptics binding. `ember.optics` pins the version, but a
release can change values without changing the API, and a matching version does not by itself mean
matching numbers. These cases are what would catch that.

## The oracle is the construction, not the history

`test_the_planted_rank_is_what_the_instrument_recovers` checks `k_signal` against `planted` — a fact
about how the frame was built rather than a number copied from a previous run. Without it the pins
would only say the instrument still agrees with itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from prism.vectors import load_vectors, vector_path

VECTORS = "screen_read_vectors"

# No `or {}`, no default, no `skipif`. Absent vectors raise out of collection and this module ERRORS
# — the only outcome that cannot be mistaken for the two engines agreeing.
_DOC = load_vectors(VECTORS)
_VECTORS = _DOC["vectors"]
assert _VECTORS, (
    "%s carries no cases — every parametrised test below would collect zero cases and this gate "
    "would report success while comparing the two engines on nothing" % VECTORS)

_IDS = [v["name"] for v in _VECTORS]


def _build(v: dict) -> np.ndarray:
    """One case's frame, per `_how_to_build`. Draw order is part of the contract: the base matrix
    first, then each planted mode's `u` before its `w`, and the dead columns killed last."""
    rng = np.random.default_rng(v["seed"])
    if v.get("collapse"):
        m = np.zeros((v["N"], v["F"]))
        m[:, 0] = rng.normal(size=v["N"])
        return m
    m = rng.normal(size=(v["N"], v["F"]))
    for _ in range(v["planted"]):
        u = rng.normal(size=v["N"])
        w = rng.normal(size=v["F"])
        m += v["snr"] * np.outer(u / np.linalg.norm(u), w / np.linalg.norm(w))
    dead = int(v.get("dead_cols", 0))
    if dead:
        m[:, v["F"] - dead:] = 0.0
    return m


def _read(v):
    from ember.optics import read_ordered
    return read_ordered(_build(v), seed=0)


def test_the_vectors_file_is_present_and_populated():
    """This gate's own precondition. A parametrised test over zero cases passes, so the count is
    asserted rather than assumed."""
    assert vector_path(VECTORS).is_file(), (
        "%s is missing from the installed prism package; the beacon half reads the same set"
        % VECTORS)
    assert len(_VECTORS) >= 7, (
        "%s carries %d cases; the set spans noise, one and three modes, a wide frame, a collapsed "
        "axis and a dead channel" % (VECTORS, len(_VECTORS)))
    for v in _VECTORS:
        assert "beam" in v and "beacon" in v, \
            "%s pins only one engine, so it compares nothing" % v["name"]


def test_the_recipe_is_deterministic():
    """Two builds of one case must be identical bytes. The shared file stores a recipe rather than a
    matrix, so determinism is what makes the pins mean anything on the other engine's machine."""
    for v in _VECTORS:
        assert np.array_equal(_build(v), _build(v)), \
            "%s rebuilt differently from the same seed" % v["name"]


@pytest.mark.parametrize("v", _VECTORS, ids=_IDS)
def test_the_instrument_reproduces_the_pinned_read(v):
    """`k_signal` and `scale_hazard` on each pinned frame.

    A failure here is one of two things, worth telling apart: entroptics moved underneath the
    instrument, or the instrument's own read changed. Either way the fix is deliberate — move the
    vectors in the same commit as the change.
    """
    expected = v["beam"]
    read = _read(v)
    assert read.k_signal == expected["k_signal"], (
        "%s: k_signal pinned at %d, measured %d. This is the count everything downstream bands on."
        % (v["name"], expected["k_signal"], read.k_signal))
    assert bool(read.scale_hazard) == bool(expected["scale_hazard"]), (
        "%s: scale_hazard pinned at %r, measured %r"
        % (v["name"], expected["scale_hazard"], read.scale_hazard))


def test_the_planted_rank_is_what_the_instrument_recovers():
    """The independent oracle. Every non-collapsed case plants a known number of modes well clear of
    the noise edge, and `k_signal` must be that number. `noise_only` plants nothing and must resolve
    nothing."""
    wrong = []
    for v in _VECTORS:
        if v.get("collapse"):
            continue                    # a destroyed feature axis has no planted rank to recover
        read = _read(v)
        if read.k_signal != v["planted"]:
            wrong.append("%s: planted %d, resolved %d" % (v["name"], v["planted"], read.k_signal))
    assert not wrong, (
        "the instrument did not recover the rank that was planted: %s. The vectors agree with the "
        "instrument's history but not with the construction, which means the history is what is "
        "wrong." % "; ".join(wrong))


def test_a_dead_channel_does_not_change_the_instrument_read():
    """The other side of the invariance the beacon half pins.

    `dead_channels` is a frame measured at 10 channels and stored at 16. The instrument's `phi`
    restricts to its own live view before decomposing, so the stored frame and the tight frame are
    one measurement and must read alike. Beacon asserts the same equality on `occupancy_fraction`;
    between them the two engines cannot drift apart on how a frame was laid out.
    """
    from ember.optics import read_ordered

    case = next((v for v in _VECTORS if v.get("dead_cols")), None)
    assert case is not None, "the dead-channel vector must exist; the invariance is unpinned without it"

    stored = _build(case)
    tight = stored[:, :case["F"] - case["dead_cols"]]
    assert np.count_nonzero(stored[:, case["F"] - case["dead_cols"]:]) == 0, \
        "the case did not actually build dead columns, so it proves nothing"

    wide, narrow = read_ordered(stored, seed=0), read_ordered(tight, seed=0)
    assert wide.k_signal == narrow.k_signal, (
        "k_signal moved %d -> %d between the same frame stored wide and stored tight"
        % (wide.k_signal, narrow.k_signal))
    assert bool(wide.scale_hazard) == bool(narrow.scale_hazard), \
        "the storage stride changed whether the frame reports a scale hazard"


def test_the_two_engines_are_pinned_to_the_same_rank():
    """What the shared file is for. Where both engines answer the rank question they must answer it
    with the same number, and the pins are where that is written down.

    `collapsed_axis` is excluded and the file's `_hazard` note says why: beacon reports the frame
    degraded, the instrument reports no scale hazard, and both are correct — beacon's domain is the
    corpus, where a dead channel is anomalous, and the instrument's is the signal, where a zero-MAD
    channel is ordinary. That divergence is the embodiment split, recorded rather than repaired.
    """
    for v in _VECTORS:
        if v.get("collapse"):
            continue
        assert v["beacon"]["k"] == v["beam"]["k_signal"] or v["planted"] == 0, (
            "%s pins beacon k=%d against instrument k_signal=%d; a shared vector file that records "
            "two different answers to the rank question has stopped being a gate"
            % (v["name"], v["beacon"]["k"], v["beam"]["k_signal"]))
