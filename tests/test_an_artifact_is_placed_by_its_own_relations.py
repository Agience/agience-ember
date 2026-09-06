"""An artifact the English lexicon never named is placed by its own relations, or honestly refused.

## The gap this closes

> John: *"Who cares if wordnet doesn't have it??? Of course it wouldn't. Every artifact should be
> independent."*

`ANCHOR_LABEL` is `"lex:en"` and `candidates()` walked `("lex:en", "assoc:en")` — **both lemma-hub
edges**. So an artifact the English lexicon never named had no anchors, no candidates and no
evidence frame, however many relations it carried. Measured 2026-08-25 over 40 such ConceptNet
terms on 71/home:

    with NO relation at all                        0 of 40
    non-containment degree     1-4: 27 · 5-19: 10 · 20+: 3
    `anchors()`    -> 0 anchors                   38 of 40
    `candidates()` -> 0 candidates                38 of 40

📄 `candidates()` justified that as a property of the graph — *"there is no path in the lattice along
which evidence about it could reach `aid`"*. `cn-nosode --related_to--> cn-homeopathy` is such a
path. It was a label whitelist wearing a graph-theoretic argument.

After: `cn-nosode` lands on `cn-autonosodes` at residual 0.6667, `cn-singlish` on `cn-yinglish` at
0.8000 — both English-contact varieties — and `cn-cow` still lands on `wn-cow.n.01` at 0.9703,
unchanged to four decimals.

## The three things this pins, each of which was measured going wrong

1. **A lemma is not a candidate.** The first version yielded `lemma:physicist` as something to
   merge `pwn-e` with. A surface form is travelled THROUGH, not merged with, and it has no evidence
   frame, so it came back unmeasurable. `test_consolidate_colimit` caught it.
2. **The widened reach must not run where the lexicon already spoke.** Unguarded — even with the
   horizon prune — `superposition(cn-cow)` went from **16.1 s to past ten minutes**: 338 edges, each
   neighbour's whole neighbourhood expanded, ~34,000 counting queries. Guarding it also makes the
   change strictly additive, so no merge already certified on the WordNet side can move.
3. **An empty frame is not a refusal about the artifact.** Widening the reach alone left `cn-nosode`
   with 3 candidates and still refusing, because `evidence()` features a row by the NEIGHBOUR'S
   anchors and its neighbours had none either.

And what must NOT move: a refusal is still available. `separated` decides, and a source whose
candidates do not separate is reported ambiguous rather than placed on the nearest thing.
"""
from __future__ import annotations

import pytest

from ember.consolidate import diagram as D
from mantle.db import open_lattice


def _mk(store, aid, **doc):
    store.artifacts.put_artifact(dict(
        {"id": aid, "content_type": "text/x-wordnet", "state": "committed"}, **doc))


@pytest.fixture()
def unlexicalised(store_path):
    """A concept the lexicon never named, related to two others that share its neighbourhood.

    No `lemma:` vertex points at `x-nosode` at all — which is the whole point. `x-homeopathy` is the
    shared relation, and `x-autonosode` is the artifact that shares it."""
    s = open_lattice(store_path, origin="test-node")
    for a in ("x-nosode", "x-autonosode", "x-homeopathy", "x-unrelated"):
        _mk(s, a, title=a[2:], lemmas=[a[2:]])
    s.graph.add_edges([("x-nosode", "x-homeopathy", "related_to", {}),
                       ("x-autonosode", "x-homeopathy", "related_to", {}),
                       ("x-autonosode", "x-nosode", "form_of", {})], batch=8)
    return s


@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path / "relations.db")


def test_an_artifact_with_no_lemma_anchor_still_has_candidates(unlexicalised):
    """The defect, stated directly: `anchors()` is empty and `candidates()` used to be too."""
    assert D.anchors(unlexicalised, "x-nosode")[0] == [], "the fixture gave it a lemma anchor"
    got = set(D.candidates(unlexicalised, "x-nosode"))
    assert "x-autonosode" in got, (
        "an artifact sharing a relation is not reachable: %r" % (sorted(got),))


def test_a_lemma_is_travelled_through_and_never_offered_as_a_candidate(unlexicalised):
    """Hazard 1. A surface form is not a thing to merge with."""
    unlexicalised.graph.add_edges([("lemma:nosode", "x-nosode", "lex:en", {}),
                                   ("lemma:nosode", "x-autonosode", "lex:en", {})], batch=8)
    got = set(D.candidates(unlexicalised, "x-nosode"))
    assert "x-autonosode" in got, "the lemma hub stopped reaching what it always reached"
    assert not [g for g in got if str(g).startswith("lemma:")], (
        "a lemma was offered as a merge candidate: %r" % (sorted(got),))


def test_the_widened_reach_does_not_run_where_the_lexicon_already_spoke(unlexicalised):
    """Hazard 2, and the guarantee that makes this change safe to land: an artifact that HAS a lemma
    anchor keeps exactly the reach it had, so no certified merge can move — and it does not pay the
    cost that took `superposition(cn-cow)` past ten minutes."""
    unlexicalised.graph.add_edges([("lemma:nosode", "x-nosode", "lex:en", {})], batch=8)
    assert D.anchors(unlexicalised, "x-nosode")[0] == ["lemma:nosode"]
    got = set(D.candidates(unlexicalised, "x-nosode"))
    # Only what the lemma hub reaches — `x-homeopathy`'s other neighbour is NOT pulled in.
    assert "x-autonosode" not in got, (
        "the widened arm ran for an artifact the lexicon named: %r" % (sorted(got),))


def test_relates_to_excludes_containment(unlexicalised):
    """A collection is not a relation. `mantle/db/vertex.py::_place` writes
    `(collection, artifact, "contains")`, so counting it reaches every member — 1.16M of them for
    `stage.0.conceptnet`, which is how a probe written for this measurement hung."""
    unlexicalised.graph.add_edges([("a-collection", "x-nosode", "contains", {})], batch=8)
    assert "a-collection" not in D.relates_to(unlexicalised, "x-nosode")


def test_a_shared_hub_is_worth_less_than_a_shared_rarity(unlexicalised):
    """The weight is `1/(1+log(in-degree))` — `lookup.hop_cost`'s own measure, and nothing chosen.
    Sharing something hundreds of things point at says almost nothing."""
    for i in range(30):
        _mk(unlexicalised, "x-many-%d" % i, title="m%d" % i)
    unlexicalised.graph.add_edges(
        [("x-many-%d" % i, "x-homeopathy", "related_to", {}) for i in range(30)], batch=64)
    w = D.relates_to(unlexicalised, "x-nosode")
    assert w["x-homeopathy"] < 0.5, w
    assert D.relates_to(unlexicalised, "x-autonosode")["x-nosode"] > w["x-homeopathy"], (
        "a rarely-pointed-at vertex must be worth more than a hub")


def test_an_artifact_nothing_relates_to_is_refused_not_placed(unlexicalised):
    """The refusal must survive. Placing something on the nearest available thing is what the whole
    measurement exists to avoid."""
    _mk(unlexicalised, "x-alone", title="alone")
    assert list(D.candidates(unlexicalised, "x-alone")) == []


# ── the batched anchor prefetch: same answer, fewer reads ────────────────────────────────────────
# `evidence()` asks `anchors()` for every neighbour, and `anchors()` is a DOUBLING PROBE — right for
# one unknown node, wrong for four thousand known ones. Measured 2026-08-25:
# `superposition(cn-singlish)` spent 80.4 s of 111.4 s inside `cn-music` alone (4,164 edges), and
# three candidates were 93% of the call.
#
# A performance change to a measurement is only allowed if the measurement does not move, so that
# is what these assert — not that it is faster, which the store already said, but that it is the
# SAME.
def test_the_prefetch_changes_nothing_about_the_frame(unlexicalised):
    """Bit-identical rows, weights and exhaustiveness, prefetched or not.

    The prefetch fills `anchor_cache`; handing it a cache already full of the same answers is the
    control, since `_prefetch` skips what is cached. If the two frames differ, the batch read and
    the probe disagree about what an anchor is."""
    unlexicalised.graph.add_edges([("lemma:nosode", "x-nosode", "lex:en", {}),
                                   ("lemma:nosode", "x-autonosode", "lex:en", {}),
                                   ("lemma:homeo", "x-homeopathy", "assoc:en", {})], batch=8)
    for arm in ({"label_keyed": False, "include_unanchored": True},
                {"label_keyed": True, "include_unanchored": True},
                {"label_keyed": False, "include_unanchored": False}):
        warm = {}
        for nid in ("x-nosode", "x-autonosode", "x-homeopathy", "lemma:nosode"):
            warm[nid] = D.anchors(unlexicalised, nid)          # per-id probe, the reference
        a = D.evidence(unlexicalised, "x-nosode", anchor_cache=dict(warm), **arm)
        b = D.evidence(unlexicalised, "x-nosode", anchor_cache={}, **arm)   # prefetched
        assert a.rows == b.rows, (arm, a.rows, b.rows)
        assert a.weights == b.weights, arm
        assert a.exhaustive == b.exhaustive, arm
        assert a.unanchored_rows == b.unanchored_rows, arm


def test_the_prefetch_and_the_probe_agree_on_every_neighbour(unlexicalised):
    """The prefetch writes into the same cache `anchors()` would fill, so the two must be equal
    entry for entry. A batch that quietly dropped a label would show up here and nowhere else."""
    unlexicalised.graph.add_edges([("lemma:nosode", "x-nosode", "lex:en", {}),
                                   ("lemma:x", "x-homeopathy", "assoc:en", {})], batch=8)
    cache = {}
    D.evidence(unlexicalised, "x-nosode", label_keyed=False, include_unanchored=True,
               anchor_cache=cache)
    assert cache, "the prefetch filled nothing — it is not running"
    for nid, got in cache.items():
        assert got == D.anchors(unlexicalised, nid), (
            "prefetch and probe disagree on %r: %r vs %r" % (nid, got, D.anchors(unlexicalised, nid)))
