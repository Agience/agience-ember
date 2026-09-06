"""The reader addresses artifacts rather than WordNet names, across both id spaces the corpus holds.

`driver._n` strips the ontology's id prefix when it is there and returns the id unchanged when it is
not, so its output is a union of two spaces: bare WordNet names (`cow.n.01`) and foreign ids that
were never prefixed (`cn-cow`, `concept-952227…`). Addressing a concept by typing `"wn-" + name`
reaches `wn-cn-cow`, an id nothing holds, and the caller reads the empty result as an absence in the
corpus rather than as a wrong question. Measured on the live corpus after the colimit's 5,473
certified merges:

    cn-cow                        637 incident edges, of which the answer path saw 0
    edges pointing at concept-*   26,158 (8,392 hypernym, 9,868 lex:en) — every one invisible
    synsets("dog")                17 lemma entries, 16 resolved; `concept-59eae33e…` dropped
    conceptnet vertices           1,165,110, against WordNet's 676,225 — the larger half

So the assertions come in a pair, on one store holding both kinds:

  1. a foreign-id concept resolves to its own artifact — its lemmas, its IC, its edges, its
     `cited_from` — rather than to a WordNet artifact or an empty shell; and
  2. a name the store holds under neither candidate comes back absent.

(2) alone passes for a reader that resolves nothing at all; (1) alone passes for a reader that has
merely been made permissive, which is what offering a second id space could introduce — a candidate
tried second turning a genuine miss into a hit.

The design rests on the two spaces being disjoint, so that is measured rather than assumed
(`test_a_bare_synset_name_is_never_itself_an_artifact_id`). A corpus that minted an artifact under a
bare synset name would give `concept_ids` two live candidates for one name, and candidate order
would then decide which concept an answer was about.
"""
from __future__ import annotations

import sys

import pytest

import _paths

sys.path.insert(0, str(_paths.repo("agience-mantle") / "src"))

from mantle.db import open_lattice  # noqa: E402
from prism.grounding import TRANSDUCER_OP  # noqa: E402

WN = "text/x-wordnet"
CN = "text/x-conceptnet"
# Written out rather than imported from the module under test: a fixture that derived this would
# agree with whatever the module holds, including a wrong value. Same reasoning as
# `test_wn_store_parameterised`. Its presence puts the reader on the keyed path, which is the path
# the live corpus uses and the one these tests are about; without it the full-index load answers
# instead and derives its own IC, so every stored value below would read back as 1.0.
TRANSDUCER_ID = TRANSDUCER_OP + "language.en"


@pytest.fixture(autouse=True)
def _clean_module_state():
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    wn.bind(None)
    g._DENSE_CACHE.clear()
    yield
    wn.bind(None)
    g._DENSE_CACHE.clear()


def _world(tmp_path):
    """One concept in each id space the live corpus carries, plus a lemma vertex.

    `wn-wolf.n.01`   the ontology's own prefix
    `cn-wolf`        ConceptNet — a whole id, never prefixed, the larger half of the corpus
    `concept-c0ffee` the colimit's canonical id — what 8,392 hypernym edges point at
    `lemma:wolf`     a surface form: a vertex in the graph, and not an artifact
    """
    L = open_lattice(str(tmp_path / "ids.db"), origin="test-node")
    L.artifacts.ensure_schema()
    L.graph.ensure_schema()
    L.artifacts.put_many([
        {"id": TRANSDUCER_ID, "content_type": "application/x-transducer", "state": "committed",
         "kind": "transducer", "spec": {"lang": "en", "entry_label": "lex:en",
                                        "lemma_prefix": "lemma:", "exceptions": {}}},
        {"id": "wn-wolf.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["wolf"], "hypernyms": ["canine.n.02"], "ic": 9.0,
         "content": "any of various predatory carnivorous canine mammals",
         "cited_from": "cite.wordnet"},
        {"id": "wn-canine.n.02", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["canine"], "hypernyms": [], "ic": 5.0,
         "content": "any of various fissiped mammals with nonretractile claws",
         "cited_from": "cite.wordnet"},
        {"id": "cn-wolf", "content_type": CN, "state": "committed",
         "lemmas": ["wolf"], "ic": 6.5,
         "content": "wolf - ConceptNet 5.7 concept /c/en/wolf",
         "cited_from": "cite.conceptnet"},
        # No `cited_from` — the colimit does not mint one, and that absence is load-bearing below:
        # it is the row a `or "cite.wordnet"` fallback would fabricate a source for.
        {"id": "concept-c0ffee", "content_type": WN, "state": "committed",
         "lemmas": ["wolf", "grey_wolf"], "ic": 7.0,
         "content": "colimit of 2 records of one concept"},
    ])
    L.graph.add_edge("wn-wolf.n.01", "wn-canine.n.02", "hypernym", {})
    L.graph.add_edge("cn-wolf", "wn-canine.n.02", "related_to", {})
    L.graph.add_edge("concept-c0ffee", "wn-canine.n.02", "hypernym", {})
    L.graph.add_edge("lemma:wolf", "wn-wolf.n.01", "lex:en", {})
    return L


# ── (1) the positive half ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, lemma, ic, cite", [
    ("wolf.n.01", "wolf", 9.0, "cite.wordnet"),          # the ontology's own prefix
    ("cn-wolf", "wolf", 6.5, "cite.conceptnet"),         # ConceptNet
    ("concept-c0ffee", "wolf", 7.0, None),               # the colimit's canonical id
])
def test_a_concept_resolves_to_its_own_artifact(tmp_path, name, lemma, ic, cite):
    """Each name resolves to its own artifact, rather than to a WordNet one or an empty shell that
    merely stops raising.

    The IC and the citation are asserted because `cn-wolf` and `wn-wolf.n.01` share a lemma: a
    reader that fell through to the WordNet artifact would still hand back something called `wolf`,
    and only the values it carries say which record was read."""
    from crystal.ontology import driver as wn
    L = _world(tmp_path)
    wn.bind(L)
    s = wn.synset(name, store=L)
    assert s is not None
    assert lemma in [l.name() for l in s.lemmas()], s.lemmas()
    assert s.ic() == pytest.approx(ic), (name, s.ic())
    assert (wn.concept_artifact(name, L).get("cited_from")) == cite


def test_a_foreign_concepts_taxonomy_is_read_from_its_own_edges(tmp_path):
    """The 8,392 hypernym edges. `concept-c0ffee` carries no `hypernyms` field, exactly as
    `wn-dog.n.01` does not on the live corpus, so its parents come from `edge WHERE src=?`, keyed on
    the id that resolved. A concept whose parents cannot be found does not raise; it reads as a
    root, which is a wrong answer wearing a valid shape."""
    from crystal.ontology import driver as wn
    L = _world(tmp_path)
    wn.bind(L)
    assert [h.name() for h in wn.synset("concept-c0ffee", store=L).hypernyms()] == ["canine.n.02"]


def test_an_associative_hop_reads_a_foreign_ids_edges(tmp_path):
    """An associative hop reads the same resolved id. `cn-wolf` holds one `related_to` edge; keyed
    on `wn-cn-wolf` it holds none, and the caller cannot tell "no edges" from "wrong question"."""
    from ember.ontology import match
    L = _world(tmp_path)
    assert [t for t, _ in match.related(L, "cn-wolf")] == ["canine.n.02"]
    # the negative control on the same call: the prefixed case still reads, and the is-a exclusion
    # stays as narrow as it was
    assert match.related(L, "wolf.n.01") == []


# ── (2) the negative half — a second candidate does not manufacture a resolution ───────────────

def test_a_concept_the_store_does_not_hold_is_still_absent(tmp_path):
    """The control for the second candidate itself. Offering another id space is the kind of change
    that turns a miss into a hit, and if it did every test above would pass for the wrong reason."""
    from crystal.ontology import driver as wn
    L = _world(tmp_path)
    wn.bind(L)
    with pytest.raises(wn.WordNetError):
        wn.synset("nosuch.n.99", store=L)
    with pytest.raises(wn.WordNetError):
        wn.synset("cn-nosuch", store=L)
    assert wn.concept_artifact("cn-nosuch", L) == {}
    assert wn._get_synset("concept-deadbeef", L) is None


def test_a_lemma_vertex_is_not_a_concept(tmp_path):
    """`lemma:wolf` is a vertex in the graph rather than an artifact, so it resolves to no concept,
    and `is_lemma_id` says so from the ontology's own spec rather than from a typed id prefix."""
    from crystal.ontology import driver as wn
    L = _world(tmp_path)
    wn.bind(L)
    assert wn.is_lemma_id("lemma:wolf", L) is True
    assert wn.is_lemma_id("cn-wolf", L) is False
    assert wn.is_lemma_id("wolf.n.01", L) is False
    assert wn._get_synset("lemma:wolf", L) is None


# ── the assumption the design rests on ────────────────────────────────────────────────────────

def test_a_bare_synset_name_is_never_itself_an_artifact_id(tmp_path):
    """`concept_ids` offers two candidates and lets the store pick, which is unambiguous exactly
    while the two spaces are disjoint — while no artifact is stored under a bare synset name.

    Measured on the live corpus: `cow.n.01`, `dog.n.01`, `water.n.01` and `cattle.n.01` are all
    absent as artifact ids, and every synset is stored under `wn-<name>`. This test holds that
    shape, so a writer that stored one goes red here, where the cause is named, rather than
    deciding silently which of two concepts an answer was about."""
    from crystal.ontology import driver as wn
    L = _world(tmp_path)
    for name in ("wolf.n.01", "canine.n.02"):
        assert L.artifacts.get_artifact(name) is None or not L.artifacts.get_artifact(name), name
        assert L.artifacts.get_artifact(wn._prefix(L) + name), name
    assert wn.concept_ids("wolf.n.01", L) == ("wn-wolf.n.01", "wolf.n.01")
    assert wn.concept_ids("cn-wolf", L) == ("wn-cn-wolf", "cn-wolf")
    # the ontology's own prefix is applied once — `wn-wn-wolf.n.01` is nothing
    assert wn.concept_ids("wn-wolf.n.01", L) == ("wn-wolf.n.01",)


# ── provenance — a stated source, or the id that was read ─────────────────────────────────────

def test_a_concept_with_no_stated_source_is_not_cited_to_wordnet(tmp_path):
    """`concept-c0ffee` states no source, so `rank_fired` cites the record it did read: the concept's
    own id. Filling the absence with `cite.wordnet` would assert a provenance nobody measured
    ([[absence-is-not-an-affirmative-claim]]) for a record never read from that corpus.

    The WordNet row in the same call is the control: it carries `cite.wordnet` in the store and
    still says `cite.wordnet`. What a stated source says is unchanged."""
    from ember.ontology import activation
    from crystal.ontology import driver as wn
    L = _world(tmp_path)
    wn.bind(L)
    rows = {r["concept"]: r for r in activation.rank_fired(
        L, {"wolf.n.01": 9.0, "cn-wolf": 6.0, "concept-c0ffee": 7.0})}
    assert rows["wolf.n.01"]["cited_from"] == "cite.wordnet"
    assert rows["cn-wolf"]["cited_from"] == "cite.conceptnet"
    assert rows["concept-c0ffee"]["cited_from"] == "concept-c0ffee"
    # and the gloss arrives for the foreign ids as well as the prefixed one
    assert rows["cn-wolf"]["gloss"] and rows["concept-c0ffee"]["gloss"]


def test_the_cache_reverifies_against_the_id_it_resolved_to(tmp_path):
    """A cached Synset is re-stamped against the artifact it was built from. Re-deriving the
    candidate order on the verify path would stamp `wn-cn-wolf` for a concept living at `cn-wolf`,
    read "changed" on every call, and rebuild for ever — a cache that cannot hit.

    A real write to the concept still reaches a running reader, so both halves are asserted:
    unchanged reads the same object, changed reads a new one carrying the new value."""
    from crystal.ontology import driver as wn
    L = _world(tmp_path)
    wn.bind(L)
    first = wn.synset("cn-wolf", store=L)
    assert wn.synset("cn-wolf", store=L) is first, "re-verified entry was rebuilt"
    L.artifacts.put_many([
        {"id": "cn-wolf", "content_type": CN, "state": "committed",
         "lemmas": ["wolf"], "ic": 6.75, "content": "wolf - ConceptNet 5.7 concept /c/en/wolf",
         "cited_from": "cite.conceptnet"},
    ])
    wn.invalidate()
    again = wn.synset("cn-wolf", store=L)
    assert again.ic() == pytest.approx(6.75), again.ic()
