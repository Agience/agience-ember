"""An artifact is placed by what it is ABOUT, and what a thing is about can be a verb.

## The gap this closes

`crystal.ontology.lookup` holds two word→concept entry points twenty lines apart, and until now the
ranking used the wrong one for candidates:

    offer_synsets(text)     `held_senses(tok, wn.NOUN)`      NOUNS ONLY
    wn_synsets_for(word)    "nouns first, then verbs"        NOUNS AND VERBS

Measured 2026-08-25 on 71/home, 2,500 ConceptNet terms drawn uniformly by rowid:

    placed by the noun-only arm            49.0%
    placed ONLY once verbs are asked        2.8%     <- ~33,000 artifacts (+/- 0.7pp)
    placed by neither                      48.2%

`cn-frighten` → `frighten.v.01`, `cn-embrittle` → `embrittle.v.01`, `cn-mislead...` →
`mislead.v.01`. Every one of those has a hypernym tree, an information content and a position, and
the placement arm was not asking.

## Why this is a NEW function and not a change to `offer_synsets`

`offer_synsets` is noun-only **deliberately**, and its docstring states the property: *"describes
markdown documents contributes `markdown`/`document` and nothing from 'describes'."* In a
description the verb is the frame, not the subject; firing it smears the offer across the ontology.
Widening it would break that on purpose. The two questions share an input and are not the same
question, so they get two functions — and `test_an_offer_is_still_placed_by_its_nouns_alone` is what
stops the next pass from "simplifying" them back into one.

## What is deliberately still excluded

Adjectives and adverbs, by both. They carry an information content and NO hypernym parent, so they
have no least common subsumer with anything, `jc_tree` has nothing to measure, and a synset admitted
without a position would score a distance that came from nowhere.

The noun/verb cross-taxonomy pair needs nothing here: 📄 `wn_synsets_for` records that such a pair
has no subsumer, that the propagation floor already refuses it, and that the verb taxonomy's own diameter
(1.109) sits inside the horizon (1.691) — so no within-tree verb pair is refused for being a verb.

Lives in ember, not crystal, although the function is crystal's: this is a LEXICON question, the
lexicon is read out of the store, and `_fakes._install_offline_wordnet` is the only fixture in the
workspace that builds one. A crystal-side copy would assert against an empty index and pass on
every branch — the failure mode a sibling probe hit the same afternoon, reading 0.0% placement
against a corpus measured at 46.8% one command earlier.
"""
from __future__ import annotations

import pytest

from crystal.ontology import lookup as L
from crystal.ontology import driver as wn
from _fakes import _install_offline_wordnet, _FakeStore


@pytest.fixture(scope="module", autouse=True)
def _wn():
    """SKIP rather than run. Without the lexicon every lookup answers `[]`, and three of the
    assertions below are "this places nothing" — they would pass on an empty index and report a
    property nobody measured. `test_the_noun_only_arm_is_what_was_missing_it` is the live control
    on the same hazard: it requires the gap to actually exist before anything claims to close it."""
    if not _install_offline_wordnet():
        pytest.skip("WordNet not available in this environment")
    # A STORE as well as an index. `held_senses` asks the store whether the lexicon holds an
    # entry, and with none bound every lookup here raises `OntologyStoreRequired` inside
    # `offer_synsets`, which swallows it and answers `[]` — indistinguishable from "the word is not
    # a noun", which is the very distinction this file exists to make.
    wn.bind(_FakeStore())
    assert L.offer_synsets("document"), "the lexicon answers nothing at all — the fixture is dead"
    yield
    wn.bind(None)


def _pos(name):
    """The part of speech in a synset name — `frighten.v.01` -> `v`, `oewn-01315031-v` -> `v`."""
    if name.count(".") >= 2:
        return name.rsplit(".", 2)[-2]
    return name.rsplit("-", 1)[-1] if name.startswith("oewn-") else "?"


def test_a_verb_titled_artifact_is_placed():
    got = L.subject_synsets("frighten")
    assert got, "a verb the lexicon holds placed nowhere"
    assert _pos(got[0]) == "v", got


def test_the_noun_only_arm_is_what_was_missing_it():
    """The positive control on the gap itself. If `offer_synsets` placed this too, there was never
    a gap and every number in the docstring above is measuring nothing."""
    assert not L.offer_synsets("frighten"), (
        "offer_synsets already places a verb — the asymmetry this file documents does not exist")
    assert L.subject_synsets("frighten")


def test_an_offer_is_still_placed_by_its_nouns_alone():
    """The property that must NOT move. An offer's verb is its frame, not its subject."""
    got = L.offer_synsets("describes document")
    assert got, "the fixture holds none of these nouns — nothing is being proven"
    assert all(_pos(n) != "v" for n in got), got
    assert L.subject_synsets("describes"), (
        "`describes` places nowhere even as a subject — this pair proves nothing about the split")


def test_a_noun_is_placed_the_same_way_by_both():
    """Widening the part of speech must not move where an ordinary noun lands."""
    assert L.subject_synsets("dog") == L.offer_synsets("dog")


def test_a_modifier_is_still_placed_nowhere():
    """An adjective has an information content and no hypernym parent, so it has no position and
    admitting one would score a distance from nowhere. Both entry points must keep refusing."""
    for word in ("able", "quickly"):
        assert not [n for n in L.subject_synsets(word) if _pos(n) in ("a", "s", "r")], word


def test_the_shape_is_the_shape_the_caller_already_handles():
    """One entry per synset, most-frequent sense, in token order — a caller swapping between the
    two entry points must get no new shape."""
    got = L.subject_synsets("dog dog cat")
    assert got == list(dict.fromkeys(got)), "a synset was emitted twice: %r" % (got,)
