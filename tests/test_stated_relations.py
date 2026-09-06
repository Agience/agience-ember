"""A2 — the edge graph in the answer, and the four properties that make the read a measurement.

The core property is a pair, asserted on one field that contains both cases:

  1. an edge the store holds between two fired concepts is stated, and
  2. a pair of fired concepts the store holds no edge between is never stated.

Each half is vacuous alone. (2) is satisfied by a renderer that says nothing at all, and (1) by one
that dumps co-activations — the concepts a query happened to light, which is how `wolf` gets
answered with `department of energy`. Together they say the output tracks the graph.

The read is a coupling, not an admissibility predicate. Incident edges go onto an ordered frame at
their far endpoints' measured energy, and the leads' own band absorbs what couples
(`ember.optics.absorb_transmit`). That gives two further properties, which belong to the geometry
and which a predicate could not state:

  3. a concept the propagation never fired has no amplitude, so there is nothing of it to absorb —
     the second clause of the predicate is reproduced by the geometry itself; and
  4. a held edge whose far endpoint lies outside the answer's band is not stated — the band decides
     what is absorbed, so the read selects rather than passing on whatever it is handed.

What is read is an edge rather than a list of label names, because label populations are a fact
about today's ingest and not about meaning: of the four relation labels the plan names, `hypernym`
(182,535) is the only one incident on any `wn-` artifact, while `related_to` (1,674,562), `is_a`
(221,203) and `synonym` (171,552) are entirely ConceptNet-internal (`cn-` -> `cn-`), a component
with no edges into the answered graph. Reading the edge lets `is_a` state itself unchanged on the
day a bridge exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import _paths

sys.path.insert(0, str(_paths.repo("agience-mantle") / "src"))

from mantle.db import open_lattice  # noqa: E402

WN = "text/x-wordnet"


@pytest.fixture(autouse=True)
def _clean_module_state():
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    wn.bind(None)
    g._DENSE_CACHE.clear()
    yield
    wn.bind(None)
    g._DENSE_CACHE.clear()


def _lattice(tmp_path):
    L = open_lattice(str(tmp_path / "rel.db"), origin="test-node")
    L.artifacts.ensure_schema()
    L.graph.ensure_schema()
    return L


def _world(tmp_path):
    """Two fired concepts joined by a real edge, and a third fired concept joined to nothing.

    `wn-department_of_energy.n.01` is the third: it is in the field, it is a perfectly good
    artifact, and the corpus holds no edge to it. A co-activation dump would name it."""
    L = _lattice(tmp_path)
    L.artifacts.put_many([
        {"id": "wn-wolf.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["wolf"], "hypernyms": ["canine.n.02"], "ic": 9.0,
         "content": "any of various predatory carnivorous canine mammals",
         "cited_from": "cite.wordnet"},
        {"id": "wn-canine.n.02", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["canine"], "hypernyms": [], "ic": 5.0,
         "content": "any of various fissiped mammals with nonretractile claws",
         "cited_from": "cite.wordnet"},
        {"id": "wn-department_of_energy.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["department_of_energy"], "hypernyms": [], "ic": 5.0,
         "content": "the federal department responsible for maintaining a national energy policy",
         "cited_from": "cite.wordnet"},
    ])
    L.graph.add_edge("wn-wolf.n.01", "wn-canine.n.02", "hypernym", {})
    return L


def _field():
    return {"wolf.n.01": 9.0, "canine.n.02": 5.0, "department_of_energy.n.01": 4.0}


def test_a_held_edge_between_fired_concepts_is_stated(tmp_path):
    """(1) The positive half: a held edge between two fired concepts is stated, with a citation.
    The negative half below is vacuous without it."""
    from ember.ontology import activation
    L = _world(tmp_path)
    triples, cites = activation._stated_relations(
        L, ["wolf.n.01", "canine.n.02", "department_of_energy.n.01"], _field())
    assert ("wolf", "hypernym", "canine") in triples, triples
    assert cites, "a stated relation carried no citation"


def test_a_co_activation_with_no_edge_is_never_stated(tmp_path):
    """(2) `department_of_energy` is in the field and joined to nothing, so it appears on neither
    side of any stated relation, under any label."""
    from ember.ontology import activation
    L = _world(tmp_path)
    triples, _ = activation._stated_relations(
        L, ["wolf.n.01", "canine.n.02", "department_of_energy.n.01"], _field())
    for s, label, d in triples:
        assert "department" not in s and "department" not in d, (s, label, d)


def test_an_edge_to_a_concept_that_did_not_fire_carries_no_amplitude(tmp_path):
    """(3) The geometry reproduces the predicate's second clause, with nothing checking it.

    `canine` is removed from the field, so it has no measured energy; its row in the incident frame
    is the zero vector, and the zero vector offers the band nothing to absorb. The wolf->canine edge
    is still in the store and still read — it carries no signal this turn. `field` is an energy
    here, not a membership set."""
    from ember.ontology import activation
    L = _world(tmp_path)
    triples, _ = activation._stated_relations(
        L, ["wolf.n.01"], {"wolf.n.01": 9.0, "department_of_energy.n.01": 4.0})
    assert triples == [], triples


def test_a_held_edge_outside_the_answers_band_is_not_stated(tmp_path):
    """(4) The band decides, which is the property a predicate could not carry.

    Here the store does hold `wolf -> department_of_energy` and `department_of_energy` did fire, so
    an admissibility rule reading "held, and its far endpoint fired" would state it. The band does
    not: `department_of_energy` shares no ancestry with the leads, so its coordinate lies wholly in
    the residual of the band `{wolf, canine}` and its absorbed energy is exactly zero. Measured on
    the live corpus, "what does a cat say" fires `department_of_energy.n.01` at salience 11.464 and
    absorbs 0.0000 against the cat/feline/carnivore band. The wolf->canine edge in the same call is
    still stated, which keeps this from passing vacuously."""
    from ember.ontology import activation
    L = _world(tmp_path)
    L.graph.add_edge("wn-wolf.n.01", "wn-department_of_energy.n.01", "hypernym", {})
    triples, _ = activation._stated_relations(L, ["wolf.n.01", "canine.n.02"], _field())
    assert ("wolf", "hypernym", "canine") in triples, triples
    for s, label, d in triples:
        assert "department" not in s and "department" not in d, (s, label, d)


def test_every_stated_relation_cites_a_resolvable_artifact(tmp_path):
    """A1. Every stated relation carries a citation that resolves to an artifact in the store, so a
    reader can go and look. A relation without one is indistinguishable from an invented one."""
    from ember.ontology import activation
    L = _world(tmp_path)
    triples, cites = activation._stated_relations(
        L, ["wolf.n.01", "canine.n.02"], _field())
    assert triples
    assert cites
    for c in cites:
        assert L.artifacts.get_artifact(c) is not None, f"citation {c!r} resolves to nothing"


def test_one_fact_held_in_both_directions_is_stated_once(tmp_path):
    """OEWN stores `hypernym` and its inverse `hyponym` as two rows. They are one fact, and the
    answer states it once: a second copy adds a sentence with no second measurement behind it."""
    from ember.ontology import activation
    L = _world(tmp_path)
    L.graph.add_edge("wn-canine.n.02", "wn-wolf.n.01", "hyponym", {})
    triples, _ = activation._stated_relations(L, ["wolf.n.01", "canine.n.02"], _field())
    assert len(triples) == 1, triples


def test_an_empty_field_states_nothing_and_composes_to_the_computed_null(tmp_path):
    """A3. An empty field states no relation and composes to the empty output. Nothing is said,
    including anything about the absence ([[state-what-it-is]])."""
    from ember.ontology import activation
    L = _world(tmp_path)
    assert activation._stated_relations(L, [], {}) == ([], set())
    assert activation.compose(L, "anything", []) == ("", [])
