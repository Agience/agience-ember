"""The answer's shape — what `compose` states, and how often.

The subject here is duplication. The corpus carries two id families over one concept (PWN
`einstein.n.01`, OEWN `oewn-10974490-n`) and both resolve to the same gloss text, so a renderer that
states each lead states one sentence twice. Two ids over one concept are not two readings; a second
occurrence adds no information, and repeating it asserts a corroboration nothing measured.

The neighbouring property — how many leads the cut keeps — belongs to the frame rather than to the
renderer, and is pinned in `test_projection.py` against a constructed oracle.

Every test below asserts what is still said beside what is not repeated. A renderer that emitted
only its first lead would satisfy "said once" while being a `[:1]` cap, so the two halves are
measured together.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import _paths

sys.path.insert(0, str(_paths.repo("agience-mantle") / "src"))

from mantle.db import open_lattice  # noqa: E402

WN = "text/x-wordnet"

#: One concept, two id families — the exact shape the live corpus holds for Einstein.
SHARED = "physicist born in germany who formulated the special theory of relativity"
OTHER = "a scientist trained in physics"


@pytest.fixture(autouse=True)
def _clean_module_state():
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    wn.bind(None)
    g._DENSE_CACHE.clear()
    yield
    wn.bind(None)
    g._DENSE_CACHE.clear()


@pytest.fixture()
def world(tmp_path):
    L = open_lattice(str(tmp_path / "compose.db"), origin="test-node")
    L.artifacts.ensure_schema()
    L.graph.ensure_schema()
    L.artifacts.put_many([
        {"id": "wn-einstein.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["einstein"], "hypernyms": [], "ic": 9.0,
         "content": SHARED, "cited_from": "cite.wordnet"},
        # the OEWN twin: a different artifact id, byte-identical gloss
        {"id": "wn-oewn-10974490-n", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["albert_einstein"], "hypernyms": [], "ic": 9.0,
         "content": SHARED, "cited_from": "cite.oewn"},
        {"id": "wn-physicist.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["physicist"], "hypernyms": [], "ic": 5.0,
         "content": OTHER, "cited_from": "cite.wordnet"},
    ])
    return L


def _acts():
    """Three readings the cut keeps together: a smooth ramp gives η² equal to its own null, so all
    three are leads and the renderer is asked to state all three."""
    return [{"concept": "einstein.n.01", "salience": 3.0, "cited_from": "cite.wordnet"},
            {"concept": "oewn-10974490-n", "salience": 2.0, "cited_from": "cite.oewn"},
            {"concept": "physicist.n.01", "salience": 1.0, "cited_from": "cite.wordnet"}]


def test_the_SAME_STATEMENT_reached_twice_is_stated_once(world):
    """Two ids resolving to one gloss produce one statement of it.

    The negative control sits in the same assertion: `physicist`'s different gloss is still stated.
    A renderer that dropped every lead after the first would satisfy "said once" and fail here,
    which is what separates the rule from a cap."""
    from ember.ontology import activation
    answer, cites = activation.compose(world, "who was albert einstein", _acts())
    body = answer.split("\n")[0]
    assert body.lower().count(SHARED) == 1, "the identical gloss was stated twice: %r" % (body,)
    assert OTHER in body.lower(), "a DIFFERENT gloss was dropped — this is a cap, not a dedupe"
    assert "cite.wordnet" in cites


def test_two_different_glosses_are_both_stated(world):
    """The rule as its own test, so it stands apart from the one above: the dedupe is on the
    statement rather than on the count. Two leads carrying two different sentences say both."""
    from ember.ontology import activation
    acts = [a for a in _acts() if a["concept"] != "oewn-10974490-n"]
    answer, _ = activation.compose(world, "who was albert einstein", acts)
    body = answer.split("\n")[0].lower()
    assert SHARED in body and OTHER in body


def test_an_empty_field_is_the_COMPUTED_NULL_not_a_sentence(world):
    """An empty field composes to an empty output carrying no citation, so a caller routes on the
    null itself rather than on a scripted sentence about it."""
    from ember.ontology import activation
    assert activation.compose(world, "zzqxwv plorbnak", []) == ("", [])


def test_a_lead_with_no_gloss_grounds_nothing(world):
    """An artifact with no definition condenses to no sentence, so a field of only such leads
    composes to the same null as an empty field."""
    from ember.ontology import activation
    world.artifacts.put_artifact({"id": "wn-blank.n.01", "content_type": WN, "state": "committed",
                                  "pos": "n", "lemmas": ["blank"], "content": "",
                                  "cited_from": "cite.wordnet"})
    answer, cites = activation.compose(world, "blank", [{"concept": "blank.n.01", "salience": 1.0}])
    assert (answer, cites) == ("", [])
