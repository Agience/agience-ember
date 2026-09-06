"""The proximity path, driven by the real instrument rather than a stub.

## What this covers

`collection_frame`, `digest_refresh` and `collection_proximity` each carry their own tests against
mantle's own stubs (`_BruteForceProbe`, a local `_spectral_distance`) injected into the `read` /
`engine_id` / `probe_factory` seams. A stub establishes that a seam has the right shape; it cannot
establish that the two halves fit — that mantle's frame is something `mp_deviation` can read, that
the digests it returns are something `SpectrumProbe` can index, or that the whole survives a round
trip through mantle's serialisation.

This file is the join. ember is the only package that may hold both sides: it imports mantle
directly, and it is the sole sanctioned entroptics door (`tests/test_one_instrument.py`).

## What is asserted

The properties that make the capability worth wiring, each measured rather than assumed:

  * a mantle collection frame digests through the real read;
  * the digest survives mantle's own serialisation byte-for-byte;
  * distance is a real metric on it — a collection is nearer itself than another;
  * the exact probe returns exactly what a full scan returns, which is the property that lets it
    prune work without pruning candidates;
  * the engine id travels, so a cross-engine comparison is detectable rather than silent.

## What is NOT asserted

Ranking quality. Proximity's own header is explicit that it is "not a recall path, not a ranking,
not a retrieval policy — a capability", and the module records that a 4-artifact collection beats a
1,621-artifact one at 2.05x chance, which is why `same_rows` exact-equality is the only size
comparison offered. Nothing here compares collections of different sizes.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from ember import optics
from mantle.search.ingest.collection_frame import (
    FrameNotDigestible, build_frame, digest_collection, digest_is_spent, dumps_digest,
    loads_digest,
)


# Three collections. Two are about the same thing in different words; the third is not, and all
# three carry the same number of rows so nothing here leans on the size attractor the module warns
# about.
_ANIMALS = [
    ("a1", {"title": "the dog barked", "content": "a dog is a domestic animal that barks"}),
    ("a2", {"title": "cat and dog", "content": "the cat and the dog are domestic animals"}),
    ("a3", {"title": "wolf pack", "content": "the wolf is a wild animal related to the dog"}),
]
_PETS = [
    ("p1", {"title": "domestic animals", "content": "a dog barks and a cat is a domestic animal"}),
    ("p2", {"title": "the barking dog", "content": "dogs are animals that bark at wolves"}),
    ("p3", {"title": "wild and tame", "content": "a wolf is wild, a dog is domestic"}),
]
_FINANCE = [
    ("f1", {"title": "quarterly revenue", "content": "revenue grew against forecast this quarter"}),
    ("f2", {"title": "balance sheet", "content": "assets and liabilities on the balance sheet"}),
    ("f3", {"title": "cash flow", "content": "operating cash flow and capital expenditure"}),
]


@pytest.fixture(scope="module")
def instrument():
    """The real seam fill — exactly what a host passes mantle."""
    return {"read": optics.proximity_read(), "engine_id": optics.proximity_engine_id()}


def _digest(members, instrument, cid):
    return digest_collection(members, exhaustive=True, collection_id=cid, **instrument)


def test_a_mantle_frame_digests_through_the_real_read(instrument):
    """The join itself: mantle builds the frame, entroptics reads it, and neither had to change."""
    frame = build_frame(_ANIMALS, collection_id="animals")
    assert frame.rows == 3, "the frame did not take every member"

    d = _digest(_ANIMALS, instrument, "animals")
    assert d.engine_id == "entroptics.mp.dev", (
        "the digest does not carry the instrument that took it, so a cross-engine comparison "
        "could not be refused: %r" % (d.engine_id,))
    assert d.rows == 3
    assert len(d.read) > 0 and all(np.isfinite(v) for v in d.read), (
        "the read produced no finite spectrum, so the frame and the instrument do not actually "
        "fit — which a shape-only stub could never have detected: %r" % (d.read,))


def test_the_digest_is_deterministic_and_survives_serialisation(instrument):
    """Bit-identical on repeat, and byte-identical through mantle's own codec.

    A digest that changed between two reads of one collection would make every comparison below a
    coin toss, and one that changed through storage would make a stored digest incomparable with a
    freshly-taken one — which is exactly the query-time comparison.
    """
    a = _digest(_ANIMALS, instrument, "animals")
    b = _digest(_ANIMALS, instrument, "animals")
    assert np.array_equal(a.read, b.read), "the read is not deterministic"

    round_tripped = loads_digest(dumps_digest(a))
    assert np.array_equal(round_tripped.read, a.read), "serialisation moved the spectrum"
    assert round_tripped.engine_id == a.engine_id and round_tripped.rows == a.rows


def test_distance_is_a_metric_on_these_digests(instrument):
    """The metric properties — identity, symmetry, distinctness. Not semantic ordering.

    At this collection size the digest reproduces its own documented failure mode, which is worth
    recording rather than hiding behind a weaker assertion. The intuitive property — a collection
    about dogs and cats digesting nearer a paraphrase of itself than a collection about cash flow —
    does not hold:

        animals <-> pets     0.684990
        animals <-> finance  0.259987

    The unrelated collection is nearer, by a factor of 2.6.

    That is the short-record attractor `collection_proximity` documents: with few rows, 74.8% of a
    read's L2 energy sits in its first four modes, so small collections are compared on the shape of
    their count distribution rather than on their vocabulary. Three-row collections are squarely in
    that regime. The module's measured example is the same effect at a different scale — a
    4-artifact collection beating a 1,621-artifact one at 2.05x chance — and it is why `same_rows`
    exact equality is the only size comparison offered and why every proposed repair was rejected as
    "a number somebody picks".

    Asserting semantic ordering here would assert something the capability does not claim at this
    size: the test would pass by luck or fail for the right reason. What is claimed, and what a
    narrowing depends on, is that the distance is a metric.
    """
    animals = _digest(_ANIMALS, instrument, "animals")
    pets = _digest(_PETS, instrument, "pets")
    finance = _digest(_FINANCE, instrument, "finance")
    reads = [animals.read, pets.read, finance.read]

    for r in reads:                                   # identity
        assert optics.spectral_distance(r, r) == 0.0, "a digest is not at distance 0 from itself"

    for i, a in enumerate(reads):                     # symmetry and distinctness
        for j, b in enumerate(reads):
            d_ab = optics.spectral_distance(a, b)
            assert d_ab == optics.spectral_distance(b, a), "the distance is not symmetric"
            assert d_ab >= 0.0
            if i != j:
                assert d_ab > 0.0, (
                    "two different collections digested to the same point, so the digest cannot "
                    "separate them at all")


def test_the_probe_returns_exactly_what_a_full_scan_returns(instrument):
    """The property that lets the probe prune work without pruning candidates.

    `SpectrumProbe` is EXACT, not approximate — its range scan is lossless by the componentwise
    bound `|x_j - y_j| <= ||x - y||`. If that ever became an approximation, a narrowing built on it
    would silently drop authorized answers, which is the one failure a narrowing must not have.
    """
    SpectrumProbe = optics.proximity_probe_factory()
    digests = [_digest(m, instrument, c) for m, c in
               ((_ANIMALS, "animals"), (_PETS, "pets"), (_FINANCE, "finance"))]
    spectra = [d.read for d in digests]

    probe = SpectrumProbe(spectra)
    query = spectra[0]

    for radius in (0.0, 0.05, 0.2, 1.0, 10.0):
        got = sorted(h.index for h in probe.within(query, radius))
        want = sorted(i for i, s in enumerate(spectra)
                      if optics.spectral_distance(digests[0].read, digests[i].read) <= radius)
        assert got == want, (
            "the probe disagreed with a full scan at radius %.3f: probe=%r scan=%r. It is "
            "documented as exact; an approximation here drops answers a caller was authorized to "
            "see." % (radius, got, want))

    for k in (1, 2, 3):
        got = [h.index for h in probe.nearest(query, k)]
        want = sorted(range(len(spectra)),
                      key=lambda i: optics.spectral_distance(digests[0].read, digests[i].read))[:k]
        assert sorted(got) == sorted(want), (
            "probe.nearest(k=%d) returned %r, a full scan returns %r" % (k, got, want))


def test_a_truncated_enumeration_refuses_rather_than_digesting_a_prefix(instrument):
    """`exhaustive=False` is a refusal, not a smaller digest.

    Above `edges_of`'s limit the store genuinely does not have the collection, and a digest of a
    prefix would be a fabricated measurement OF THE WHOLE — comparable, plausible, and wrong.
    """
    with pytest.raises(FrameNotDigestible) as exc:
        digest_collection(_ANIMALS, exhaustive=False, collection_id="animals", **instrument)
    assert exc.value.reason == FrameNotDigestible.ENUMERATION_TRUNCATED


def test_the_staleness_rule_is_derived_and_needs_no_schedule(instrument):
    """`digest_is_spent(rows, events_since) := events_since >= rows`, including the base case.

    A never-digested collection has 0 rows and is therefore spent at 0 events, so "digest it the
    first time" falls out of the rule instead of being a special case somebody has to remember.
    """
    assert digest_is_spent(0, 0) is True, "a never-digested collection must be due immediately"
    assert digest_is_spent(3, 0) is False
    assert digest_is_spent(3, 2) is False
    assert digest_is_spent(3, 3) is True, "after `rows` events every original row may be replaced"
