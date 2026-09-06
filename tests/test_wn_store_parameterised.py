"""`crystal.ontology.driver` reads the store it is handed, and its vocabulary comes from that
store's own spec.

An ontology read is a measurement, so the instrument is a parameter. `synsets(..., store=B)` reads
B; `bind(store)` parameterises the module for callers that cannot thread a store through; and a
Synset carries the store it was built from, so it resolves its own taxonomy after the module's
binding goes away. An ambient bind would let a persona hand in store B, ask B's ontology a question
and receive an answer measured on store A, with nothing raising because A answers perfectly well.

The process default is the host's to name. `crystal` does not import mantle, so `_arts`'s third rung
reads `driver._PROVIDER`, which `ember/__init__.py` registers as
`mantle.shard.local_store.open_store`. The monkeypatches below set `_PROVIDER` directly, which
states the property at its sharpest: "does not touch the process default" then covers anything that
resolves ambiently rather than only what resolves through mantle. With no store passed, none bound
and no provider registered, a read raises `driver.OntologyStoreRequired`; that contract and its
control live in `agience-crystal/tests/test_ontology_store_injection.py`.

The vocabulary is data. `id_prefix`, `content_type`, `isa_labels` and the POS order are read from
the `language:*` spec, so a second ontology drops in as data rather than as a code change, and the
existing corpus — whose stored `language:en` spec names none of those keys — keeps WordNet's own
values from the module defaults.
"""
from __future__ import annotations

import json
import logging
import sqlite3

import pytest

from crystal.ontology import driver as wn
from prism.grounding import TRANSDUCER_OP

TRANSDUCER_ID = TRANSDUCER_OP + "language.en"


# ── a store face real enough for the keyed path (artifacts + an `edge` table) ────────────────────
class _Db:
    def __init__(self):
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        self.con.execute("CREATE TABLE edge(src TEXT, dst TEXT, label TEXT, props TEXT)")

    def read(self):
        return self.con


class _Arts:
    def __init__(self):
        self.d = {}
        self.db = _Db()

    def put_artifact(self, doc):
        self.d[doc["id"]] = dict(doc)
        return doc

    def get_artifact(self, aid):
        a = self.d.get(aid)
        return dict(a) if a else None

    def list_artifacts(self, *, content_type=None, **kw):
        for a in self.d.values():
            if content_type and a.get("content_type") != content_type:
                continue
            yield dict(a)

    def add_edge(self, src, dst, label, props=None):
        self.db.con.execute("INSERT INTO edge(src,dst,label,props) VALUES(?,?,?,?)",
                            (src, dst, label, json.dumps(props or {})))


class _Store:
    def __init__(self):
        self.artifacts = _Arts()


def _ontology(word, *, prefix="wn-", content_type="text/x-wordnet", spec_extra=None,
              isa_label=None, parent=None):
    """A store holding one synset `<word>.n.01` reachable by the keyed path, plus its transducer."""
    st = _Store()
    name = "%s.n.01" % word
    spec = {"name": "language.en", "kind": "language", "lang": "en", "entry_label": "lex:en",
            "exceptions": {}}
    spec.update(spec_extra or {})
    # Both values are written out rather than derived from the module under test: a fixture that
    # derived them would agree with whatever the module holds, including a wrong value. Written
    # out, they are an independent statement of what the store contains.
    st.artifacts.put_artifact({"id": TRANSDUCER_ID, "content_type": "application/x-transducer",
                               "kind": "transducer", "spec": spec})
    st.artifacts.put_artifact({"id": prefix + name, "content_type": content_type, "pos": "n",
                               "lemmas": [word], "ic": 0.5})
    st.artifacts.add_edge("lemma:" + word, prefix + name, "lex:en",
                          {"pos": "n", "proper": 0, "rank": 0})
    if parent:
        st.artifacts.put_artifact({"id": prefix + parent, "content_type": content_type,
                                   "pos": "n", "lemmas": [parent.split(".")[0]], "ic": 0.1})
        st.artifacts.add_edge(prefix + name, prefix + parent, isa_label or "hypernym", {})
    return st


@pytest.fixture(autouse=True)
def _clean_module_state():
    """The driver holds a process singleton index, so a test that binds a store hands the module
    back the way it found it — other modules in this suite share it."""
    wn._release()
    wn._AMBIENT_WARNED = False
    yield
    wn._release()
    wn._AMBIENT_WARNED = False


# ── the store is a parameter ─────────────────────────────────────────────────────────────────────
def test_synsets_reads_the_store_it_was_handed(monkeypatch):
    """Handed a specific store, the read answers from that store rather than the process default.

    Both halves are asserted: the handed store's sense is present, and the process default's sense
    is absent through it. Asserting only the first would pass on a read that resolves ambiently and
    happens to see both.
    """
    default = _ontology("alpha")
    handed = _ontology("beta")
    # The process default is `default` — the store an ambient resolution would reach.
    monkeypatch.setattr(wn, "_PROVIDER", lambda: default)

    got = [s.name() for s in wn.synsets("beta", store=handed)]
    assert got == ["beta.n.01"], "did not read the store it was handed: %r" % got
    # …and the process default's own sense is not reachable through the handed store.
    assert wn.synsets("alpha", store=handed) == []


def test_a_second_store_invalidates_the_index_rather_than_answering_from_the_first(monkeypatch):
    """The index is per-store. A `store=` parameter over one process-wide index keyed on nothing
    would turn a wrong-store read into a first-store-wins read — the same ambiguity wearing a
    parameter. Reading A then B then A again catches it: each read answers from its own store.
    """
    monkeypatch.setattr(wn, "_PROVIDER", lambda: pytest.fail("must not touch the process default"))
    a, b = _ontology("alpha"), _ontology("beta")

    assert [s.name() for s in wn.synsets("alpha", store=a)] == ["alpha.n.01"]
    assert [s.name() for s in wn.synsets("beta", store=b)] == ["beta.n.01"]
    assert [s.name() for s in wn.synsets("alpha", store=a)] == ["alpha.n.01"]
    # and cross-reads stay empty in both directions — neither store's index leaks into the other
    assert wn.synsets("alpha", store=b) == [] and wn.synsets("beta", store=a) == []


# ── one level down: the Synset carries its store ─────────────────────────────────────────────────
def test_a_synset_outlives_the_modules_ambient_binding(monkeypatch):
    """A Synset resolves its own taxonomy after the module stops holding any store.

    The shape matters. Asking for the parent immediately after `synsets(store=B)` cannot detect
    this, because `_arts(None)` returns `_SOURCE` — the last store bound — and that is still B: an
    ambient path lands on the right store by luck, so a single-store, single-moment assertion says
    nothing about what the Synset carries. So the module's binding is released first, and only then
    is the taxonomy walked.

    An ambient resolution at that point falls through to `open_store()` -> `ensure_schema`, which
    takes a `BEGIN IMMEDIATE` write lock on the process default — a live multi-gigabyte lattice,
    from a unit test. That does not fail, it hangs, and the suite reports a timeout inside
    `geometry.tree_path` with no failing assertion. `_PROVIDER` is monkeypatched to fail so the
    reach is an error rather than a stall.

    Both halves are asserted: the default is not reached, and the parent resolves — `hypernyms()`
    returning nothing would satisfy the first half on its own.
    """
    handed = _ontology("beta", parent="gamma.n.01")
    s = wn.synsets("beta", store=handed)[0]

    wn._release()                     # the module forgets everything; the Synset must not
    monkeypatch.setattr(wn, "_PROVIDER", lambda: pytest.fail("walked the taxonomy through the process default"))

    parents = [p.name() for p in s.hypernyms()]
    assert parents == ["gamma.n.01"], "the parent did not resolve at all: %r" % parents
    # …and the next hop up resolves through the same store, so the carry survives the recursion
    # rather than exactly one call.
    assert [p.name() for p in s.hypernyms()[0].hypernyms()] == []


def test_two_stores_disagreeing_about_one_synset_are_read_apart(monkeypatch):
    """The strongest form: the same synset name with different parents in two stores.

    A Synset that resolved ancestry ambiently could not tell these apart — whichever store answered
    first would supply both taxonomies, and `jc_tree`/`canonical_parent` above it would report a
    distance measured on a tree the caller never handed in. Re-reading the first store after the
    second shows that order does not decide whose taxonomy is served.
    """
    monkeypatch.setattr(wn, "_PROVIDER", lambda: pytest.fail("must not touch the process default"))
    a = _ontology("delta", parent="animal.n.01")
    b = _ontology("delta", parent="machine.n.01")

    sa = wn.synsets("delta", store=a)[0]
    sb = wn.synsets("delta", store=b)[0]
    assert [p.name() for p in sa.hypernyms()] == ["animal.n.01"]
    assert [p.name() for p in sb.hypernyms()] == ["machine.n.01"]
    # re-read the first one after the second: order does not decide whose taxonomy is served
    assert [p.name() for p in wn.synsets("delta", store=a)[0].hypernyms()] == ["animal.n.01"]


def test_bind_parameterises_the_module_for_callers_that_cannot_pass_a_store(monkeypatch):
    """`bind(store)` is the explicit parameterisation for a caller that cannot thread a `store=`
    through: after it, no-arg reads use the bound store, and the process default is not consulted.
    """
    monkeypatch.setattr(wn, "_PROVIDER", lambda: pytest.fail("bound store must win over the process default"))
    wn.bind(_ontology("gamma"))
    assert [s.name() for s in wn.synsets("gamma")] == ["gamma.n.01"]
    assert wn.morphy("gamma", "n") == "gamma"


def test_the_ambient_fallback_is_loud(monkeypatch, caplog):
    """A caller with genuinely no store still reads, and the read says which instrument it used: a
    WARNING record names the process-default fallback. A silent fallback is indistinguishable at the
    call site from a parameterised read.
    """
    monkeypatch.setattr(wn, "_PROVIDER", lambda: _ontology("delta"))
    with caplog.at_level(logging.WARNING, logger="crystal.ontology.driver"):
        assert [s.name() for s in wn.synsets("delta")] == ["delta.n.01"]
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("PROCESS-DEFAULT" in m for m in msgs), (
        "the ambient fallback was silent; logged: %r" % msgs)


# ── the ontology's vocabulary is data, read from the language:* spec ─────────────────────────────
def test_the_id_prefix_and_content_type_come_from_the_spec(monkeypatch):
    """A second ontology drops in as data: it names its own prefix and content type, and both are
    read from the spec. With them fixed in the module, `_get_synset` would look up `"wn-" + name`
    and `_load_index` would filter on one content type, so `syn-epsilon.n.01` under `text/x-myonto`
    would be invisible.
    """
    monkeypatch.setattr(wn, "_PROVIDER", lambda: pytest.fail("must not touch the process default"))
    st = _ontology("epsilon", prefix="syn-", content_type="text/x-myonto",
                   spec_extra={"id_prefix": "syn-", "content_type": "text/x-myonto"})
    assert wn._prefix(st) == "syn-" and wn._ct(st) == "text/x-myonto"
    assert [s.name() for s in wn.synsets("epsilon", store=st)] == ["epsilon.n.01"]


def test_the_isa_edge_labels_come_from_the_spec(monkeypatch):
    """§13.1: relations are edges, and the label that means is-a is the source's word for it.

    With the labels fixed in the module, `'is_a'` is not among
    `('hypernym','instance_of','instance_hypernym')`, so the synset comes back with an empty parent
    list — nodes and no tree, which is the shape that makes every `jc_tree` measure 0.0.
    """
    monkeypatch.setattr(wn, "_PROVIDER", lambda: pytest.fail("must not touch the process default"))
    st = _ontology("zeta", spec_extra={"isa_labels": ["is_a"], "instance_labels": ["an_instance_of"]},
                   isa_label="is_a", parent="thing.n.01")
    assert wn._isa_labels(st) == (["is_a"], ["an_instance_of"])
    got = wn.synsets("zeta", store=st)
    assert [h.name() for h in got[0].hypernyms()] == ["thing.n.01"]


def test_the_pos_order_comes_from_the_spec(monkeypatch):
    """The POS alphabet and its order belong to the ontology, and the order is what orders
    `synsets()`. Built from a module-level tuple it would be the same for every corpus, whatever
    each corpus's spec said.
    """
    monkeypatch.setattr(wn, "_PROVIDER", lambda: pytest.fail("must not touch the process default"))
    st = _ontology("eta", spec_extra={"pos_order": ["v", "n"]})
    assert wn._pos_order(st) == ["v", "n"]


def test_a_silent_spec_keeps_wordnets_own_vocabulary(monkeypatch):
    """A spec that names none of these keys keeps WordNet's own vocabulary, which is what the stored
    `language:en` spec does — so the existing corpus reads exactly as before. Drifting defaults
    would silently re-home every synset id in a live corpus.
    """
    monkeypatch.setattr(wn, "_PROVIDER", lambda: pytest.fail("must not touch the process default"))
    st = _ontology("theta")                                # spec carries only exceptions/etc.
    assert wn._prefix(st) == "wn-"
    assert wn._ct(st) == "text/x-wordnet"
    assert wn._pos_order(st) == [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV, wn.ADJ_SAT]
    assert wn._isa_labels(st) == (["hypernym"], ["instance_of", "instance_hypernym"])
