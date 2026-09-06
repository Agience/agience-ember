"""The labels crystal's projection walks are the ones ember's ingest actually writes.

Two repositories have to agree on one vocabulary: `ember.corpus.stage0_sources` decides which sense
relations survive parsing, and `crystal.ontology.lookup` walks a set of labels expecting to find
them. If those drift apart, the walk comes back empty on a correctly-backfilled store and nothing
raises — the projection just quietly stops projecting.

It lived in crystal's suite and could not stay there: it reads
`ember.corpus`, so making the claim from crystal meant a checkout of the repository ABOVE crystal.
Ember declares and imports `agience-crystal`, so ember can read both sides with the arrow pointing
the way the packages already point. This was also the only test in crystal's suite that needed
`AGIENCE_BUNDLE_ROOT`, so moving it took a second sibling dependency with it.
"""

from __future__ import annotations


def test_the_backfilled_labels_are_the_ones_the_parser_keeps():
    """The projection reads labels the ingest must actually write. These two sets drifting apart is
    how the walk would come back empty on a correctly-backfilled store, silently."""
    from crystal.ontology import lookup

    from ember.corpus.stage0_sources import _SENSE_RELATIONS_KEPT

    sense_level = {"derivation", "pertainym"}
    assert sense_level <= _SENSE_RELATIONS_KEPT, (
        "the parser drops %s, so the projection has no edges to walk"
        % (sense_level - _SENSE_RELATIONS_KEPT))
    walked = set(lookup._TO_NOUN) | set(lookup._ADVERB_TO_ADJECTIVE) | set(lookup._TO_HEAD_ADJECTIVE)
    assert sense_level <= walked, "the projection stopped reading a label the parser writes"
