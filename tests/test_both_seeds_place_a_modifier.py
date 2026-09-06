"""Both lexicon seeds emit the labels a modifier is placed by, or one of them silently cannot.

## Two seeds, one corpus shape

A node's Stage-0 lexicon arrives by one of two operators, and they are both live:

    op.source.wordnet   genesis.ingest_stage0_wordnet          Princeton WordNet 3.0, via nltk
    op.source.oewn      stage0_sources.ingest_stage0_oewn      Open English WordNet 2024, LMF XML

Neither an adjective nor an adverb carries a hypernym parent, so neither has a position `jc_tree` can
measure. What each has is a noun it is about, and `crystal.ontology.lookup.projected_nouns_for`
reaches it along four labels: `derivation` or `attribute` directly, `similar` to a satellite's head
adjective, `pertainym` from an adverb to its adjective.

If one seed emits those and the other does not, a node seeded from nltk places no modifiers while a
node seeded from OEWN places 65.7% of them — same code, same query, different answer, and nothing
anywhere to say why. The OEWN path takes every relation its source names, so it cannot fall behind;
the nltk path names each relation explicitly, so it can, and it did.

## Why this is a source scan

The two emitters are generators inside long ingest functions that need a store, a corpus download and
several minutes. Running them here would test the corpus fetch, not the agreement. What can drift is
which labels the code names, and that is what this reads.
"""
from __future__ import annotations

import ast
import inspect

import pytest

#: The labels a modifier's projection walks. Imported rather than retyped so this file cannot claim
#: agreement with a set the projection has moved on from.
from crystal.ontology import lookup as _lookup

PLACEMENT_LABELS = frozenset(_lookup._TO_NOUN) | frozenset(_lookup._ADVERB_TO_ADJECTIVE) | \
    frozenset(_lookup._TO_HEAD_ADJECTIVE)


def _string_literals(fn) -> set:
    """Every string constant in a function's source, including nested definitions.

    An AST walk rather than a substring search: the labels appear as edge tuples inside a nested
    generator, and a prose mention in a docstring is a string constant too — which is why the
    assertions below check the labels are ALSO reachable as edge labels, not merely present.
    """
    tree = ast.parse(inspect.getsource(fn).lstrip())
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _edge_labels(fn) -> set:
    """The third element of every 4-tuple yielded in `fn` — i.e. the edge labels it emits.

    Both emitters yield `(src, dst, label, meta)`, so the label is positional and readable without
    running anything. A tuple whose third element is not a literal is skipped rather than guessed at.
    """
    tree = ast.parse(inspect.getsource(fn).lstrip())
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Yield) or not isinstance(node.value, ast.Tuple):
            continue
        elts = node.value.elts
        if len(elts) >= 3 and isinstance(elts[2], ast.Constant) and isinstance(elts[2].value, str):
            out.add(elts[2].value)
    return out


def test_the_nltk_seed_emits_every_placement_label():
    """`genesis.ingest_stage0_wordnet` names each relation explicitly, so it is the one that can
    fall behind — and it had. A missing label here means a node seeded from nltk places no
    adjectives at all, which reads as a corpus with none rather than as an ingest that skipped them.
    """
    from ember import genesis

    labels = _edge_labels(genesis.ingest_stage0_wordnet)
    missing = PLACEMENT_LABELS - labels
    assert not missing, (
        "the nltk seed does not emit %s, so `projected_nouns_for` has nothing to walk on a node "
        "seeded this way. It emits: %s" % (sorted(missing), sorted(labels)))


def test_the_oewn_seed_keeps_every_placement_label_it_can():
    """The OEWN path takes synset-level relations whole, so only the SENSE-level ones need naming —
    and `_SENSE_RELATIONS_KEPT` is where they are named. `attribute` and `similar` are synset-level
    and arrive without being listed anywhere, which is why they are not asserted against that set.
    """
    from ember.corpus.stage0_sources import _SENSE_RELATIONS_KEPT

    sense_level = {"derivation", "pertainym"}
    assert sense_level <= _SENSE_RELATIONS_KEPT, (
        "the OEWN parser drops %s. Sense-level relations are filtered by that set, so a label "
        "absent from it never reaches the store."
        % sorted(sense_level - _SENSE_RELATIONS_KEPT))
    assert sense_level <= PLACEMENT_LABELS, \
        "the projection stopped walking a label this seed goes out of its way to keep"


def test_the_two_seeds_agree_on_the_placement_labels():
    """The property that matters to a caller: which seed a node used must not change whether a
    modifier can be placed."""
    from ember import genesis
    from ember.corpus.stage0_sources import _SENSE_RELATIONS_KEPT

    nltk_labels = _edge_labels(genesis.ingest_stage0_wordnet)
    # OEWN: sense-level from the kept set, synset-level unconditionally (the source names them).
    oewn_labels = set(_SENSE_RELATIONS_KEPT) | {"attribute", "similar"}
    for label in sorted(PLACEMENT_LABELS):
        assert label in nltk_labels, "nltk seed missing %r" % label
        assert label in oewn_labels, "OEWN seed missing %r" % label


def test_the_scan_would_notice_a_missing_label():
    """Negative control. If `_edge_labels` returned everything, or nothing, the assertions above
    would be vacuous either way."""
    from ember import genesis

    labels = _edge_labels(genesis.ingest_stage0_wordnet)
    assert "hypernym" in labels, "the scan cannot see the backbone label, so it sees nothing"
    assert "not_a_real_relation" not in labels, "the scan reports labels that are not there"
    assert len(labels) < 30, "the scan is returning every string in the function, not edge labels"


def test_the_stale_third_emitter_is_gone():
    """`corpus/ingest.py` carried a third copy of this emitter — nltk-based, zero callers, and
    emitting only `hypernym` and `part_of`. Dead code that contradicts the live paths is worse than
    dead code: it reads as documentation of what the corpus contains.
    """
    from ember.corpus import ingest

    assert not hasattr(ingest, "wordnet_relation_edges"), (
        "the stale emitter is back. If a third seed path is wanted, it has to emit the same "
        "placement labels as the other two and be covered by the assertions above.")
