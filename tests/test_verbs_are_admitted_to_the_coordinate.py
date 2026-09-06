"""`wn_synsets_for` places verbs as well as nouns, and a cross-part-of-speech pair stays disjoint.

## Why this is pinned

A question is mostly verbs. "when did humans learn to write" carries two content words and both are
verbs; with nouns alone the ontology places `humans` and nothing else, and a need grounded on one
token is grounded on almost nothing. Admitting verbs raised token coverage over 15 natural questions
from 43/66 to 52/66.

The risk admission carries is not that a verb has no position — it has one — but that the verb
hierarchy has no single root. `entity.n.01` tops every noun at IC 0, so no noun pair is ever
disjoint; verbs have 559 top nodes, so two verbs under different tops share no ancestor at all and
`jc_tree` returns `IC(a) + IC(b)` rather than a path length. A noun and a verb are always in that
case. What keeps such a pair out of a field is the horizon: the disjoint sum exceeds it for all but
the most generic synsets, and the propagation floor refuses the remainder.

## What is asserted, and what deliberately is not

The admission and the disjointness are properties of the lookup and of the tree, so they are
asserted directly. The horizon comparison is not: the horizon is derived from whichever IC basis
the corpus carries, and this fixture carries frequency-based Resnik IC from `ic-brown` while the
live lattice carries intrinsic IC. Both are valid bases and they place the horizon in different
places, so a threshold pinned here would pin the fixture rather than the system. What is
basis-independent — and is the whole safety argument — is that a cross-part-of-speech pair is
*exactly* the disjoint sum, never a path length that could pass for nearness.
"""
from __future__ import annotations

import pytest

from crystal.ontology import driver as wn
from crystal.ontology import geometry as g
from crystal.ontology.lookup import wn_synsets_for
from ember.ontology import match

from _fakes import _FakeStore, _install_offline_wordnet


@pytest.fixture(scope="module")
def wn_ready():
    if not _install_offline_wordnet():
        pytest.skip("WordNet not available in this environment")
    return True


@pytest.fixture()
def store(wn_ready):
    s = _FakeStore()
    wn.bind(s)
    match.invalidate(s)
    return s


def _pos_of(names):
    return {n.split(".")[1] for n in names}


def test_a_verb_only_word_resolves_to_verb_senses(store):
    """`learn` and `write` have no noun sense at all. Under nouns alone they place nowhere, which is
    silence rather than a measurement — the field simply never hears the word."""
    for word in ("learn", "write"):
        names = wn_synsets_for(word)
        assert names, "%r placed nowhere, so verb admission is not in effect" % word
        assert "v" in _pos_of(names), "%r resolved to %s, none of them a verb" % (word, names[:4])


def test_nouns_come_first_for_a_word_that_is_both(store):
    """`run` is both. Order is the sense prior's input, so admitting verbs must not reorder a word
    that already worked — the noun senses stay ahead of the verb senses."""
    names = wn_synsets_for("run")
    kinds = [n.split(".")[1] for n in names]
    assert "n" in kinds and "v" in kinds, "the fixture has no both-part-of-speech `run` to order"
    assert kinds.index("v") > max(i for i, k in enumerate(kinds) if k == "n"), \
        "a verb sense sorted ahead of a noun sense: %s" % names[:8]


def test_a_noun_only_word_is_unchanged(store):
    """The control. If admission were leaking senses in, it would show here first."""
    names = wn_synsets_for("glacier")
    assert names, "the fixture has no `glacier`"
    assert _pos_of(names) == {"n"}, "a non-noun sense reached a noun-only word: %s" % names


def test_a_verb_pair_in_one_tree_measures_a_real_distance(store):
    """Placed means a distance shorter than the disjoint sum — the two walked to a shared ancestor
    rather than falling through to `IC(a) + IC(b)`. This is what makes a verb reachable."""
    for a, b in (("run.v.01", "walk.v.01"), ("write.v.01", "compose.v.02")):
        s1, s2 = wn.synset(a), wn.synset(b)
        d = g.jc_tree(s1, s2, None)
        disjoint = g.ic_of(s1, None) + g.ic_of(s2, None)
        assert d < disjoint, \
            "%s <-> %s returned the disjoint sum (%.4f), so no shared ancestor was found" % (a, b, d)


def test_a_cross_part_of_speech_pair_is_exactly_the_disjoint_sum(store):
    """The safety property, and it holds by construction rather than by measurement: a noun and a
    verb share no ancestor, so `jc_tree` telescopes each path down to its own root and returns
    `IC(a) + IC(b)`. That sum is what the horizon and the propagation floor then refuse. Equality is asserted
    exactly — an inequality would also pass on a genuine path length, which is the thing this rules
    out."""
    for a, b in (("learn.v.01", "glacier.n.01"), ("dog.n.01", "run.v.01")):
        s1, s2 = wn.synset(a), wn.synset(b)
        assert g.jc_tree(s1, s2, None) == g.ic_of(s1, None) + g.ic_of(s2, None), \
            "%s <-> %s found a shared ancestor across parts of speech" % (a, b)


def test_coverage_rises_on_a_verb_bearing_question(store):
    """The end-to-end read. `coordinate_coverage` is a second pass over `fired_field`'s own tokens,
    so this measures what the field can hear rather than a lookup in isolation."""
    c = match.coordinate_coverage("when did humans learn to write", store)
    assert "learn" in c["placed"] and "write" in c["placed"], \
        "the verbs are unplaced: %s" % c["unplaced"]
    nouns_only = [t for t in c["placed"] if "v" not in _pos_of(wn_synsets_for(t))]
    assert len(c["placed"]) > len(nouns_only), \
        "admission placed no token that nouns alone would have missed"
