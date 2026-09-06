"""The cache-invalidation hook — a data change reaches a running process.

Every cache under the ontology (`driver._CACHE`, `geometry._DENSE_CACHE`,
`projection._BASIS_CACHE`, `match._OFFER_CACHE`) is a process singleton, so each one is gated on a
freshness stamp the store itself reports. Without that gate a repair is invisible until a restart:
measured on node 71, a repair rewrote 141,102 stored `ic` values and rebuilt the corpus basis
(k 195 -> 280) while the running lumen/ember went on serving the pre-repair coordinates, and the
only remedy on offer was a restart — which the operator procedure in PLAN §0.0 rules out.

Every test here writes the change through a second, independent store handle, and that shape is what
gives them teeth. A test that writes through the same handle the reader uses would pass on a write
path that invalidated the cache directly, while a repair from another process — the case that
matters — still did not land. The second handle is the closest in-process stand-in for that separate
repair process.

Every positive test has its negative control beside it, because "the new value is served" is equally
true of a cache that has been quietly disabled, and that is the worse outcome: measured on 71, a
cold read of one synset plus its coordinate is 1,566 µs against 17.4 µs warm, a 90x margin, and the
warm read touches the store zero times. So each freshness assertion is paired with one asserting
that an unchanged store recomputes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import _paths

sys.path.insert(0, str(_paths.repo("agience-mantle") / "src"))

from mantle.db import open_lattice  # noqa: E402

WN = "text/x-wordnet"


def _lattice(tmp_path, name="lattice.db"):
    L = open_lattice(str(tmp_path / name), origin="test-node")
    L.artifacts.ensure_schema()
    L.graph.ensure_schema()
    return L


def _keyed(L):
    """Write the language transducer, which is what puts the driver on its keyed arm.

    Without it `_keyed_ready()` is False and every lookup goes through `_load_index`, which
    re-derives intrinsic IC over the whole corpus and overwrites the stored values — so a test that
    repaired a stored `ic` would watch the derivation erase it, and would be measuring the
    full-index path rather than the keyed one node 71 serves from."""
    from crystal.ontology import driver as wn
    L.artifacts.put_artifact({
        "id": wn._TRANSDUCER_ID, "content_type": "application/x-transducer",
        "state": "committed", "spec": {"kind": "language", "lang": "en"}})
    return L


def _seed(L, ic=1.0):
    L.artifacts.put_many([
        {"id": "wn-root.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["root"], "lemma_counts": {"root": 1}, "hypernyms": [], "ic": 0.0},
        {"id": "wn-thing.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "lemmas": ["thing"], "lemma_counts": {"thing": 1},
         "hypernyms": ["root.n.01"], "ic": float(ic)},
    ])


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Every cache in this chain is a process singleton, so a test that does not reset them is
    testing the previous test's state. Released both before and after, so that a failure mid-test
    leaves no bound store for the next module."""
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    wn.bind(None)
    g._DENSE_CACHE.clear()
    yield
    wn.bind(None)
    g._DENSE_CACHE.clear()


# ── the store's own write bookkeeping — the thing everything else hangs on ───────────────────
def test_write_mark_moves_on_every_write_and_only_on_a_write(tmp_path):
    """The gate is only as good as this: the mark moves for a write and stands still otherwise.

    Both halves are asserted. A mark that always moves disables every cache below it; a mark that
    never moves lets a repair go unseen. State the failure mode first
    ([[verification-that-cannot-fail]])."""
    from crystal.ontology import freshness as fr
    L = _lattice(tmp_path)
    _seed(L)
    m0 = fr.write_mark(L.artifacts)
    assert m0 is not None

    assert fr.write_mark(L.artifacts) == m0, "the mark moved with no write at all"
    L.artifacts.get_artifact("wn-thing.n.01")
    assert fr.write_mark(L.artifacts) == m0, "a READ moved the write mark"

    L.artifacts.put_artifact({"id": "wn-thing.n.01", "content_type": WN, "state": "committed",
                              "pos": "n", "lemmas": ["thing"], "hypernyms": ["root.n.01"],
                              "ic": 7.0})
    m1 = fr.write_mark(L.artifacts)
    assert m1 != m0, "an UPDATE did not move the write mark"

    L.graph.add_edge("wn-thing.n.01", "wn-root.n.01", "hypernym", {})
    m2 = fr.write_mark(L.artifacts)
    assert m2 != m1, "an EDGE write did not move the write mark — edges are half the ontology"

    L.artifacts.delete_artifact("wn-root.n.01")
    assert fr.write_mark(L.artifacts) != m2, "a DELETE did not move the write mark"


def test_an_unverifiable_store_is_not_treated_as_fresh(tmp_path):
    """A store that cannot report its freshness yields no mark, and the callers read `None` as "do
    not cache" ([[absence-is-not-an-affirmative-claim]]).

    Reading the absence as "unchanged" would be the `.get("grounded", True)` substitution: a claim
    nobody measured."""
    from crystal.ontology import freshness as fr

    class _NoMark:
        def get_artifact(self, aid):
            return None

    assert fr.write_mark(_NoMark()) is None
    assert fr.stamp(_NoMark(), "wn-thing.n.01") is None
    assert fr.set_stamp(_NoMark(), WN) is None


def test_edge_mark_over_a_prefix_refuses_rather_than_rounding(tmp_path):
    """A stamp taken over the first `cap` rows is stable while the tail moves, so it would verify
    clean for ever. A capped mark therefore reports itself as partial: `exhaustive=False` is an
    absent measurement rather than an approximation of one."""
    from mantle.db.edge import edge_mark
    L = _lattice(tmp_path)
    _seed(L)
    for i in range(12):
        L.graph.add_edge("wn-thing.n.01", "wn-x%02d" % i, "hypernym", {})
    n, hi, exhaustive = edge_mark(L.db, "wn-thing.n.01", cap=5)
    assert (n, exhaustive) == (5, False), "a capped mark did not report itself as partial"
    n, hi, exhaustive = edge_mark(L.db, "wn-thing.n.01", cap=100)
    assert (n, exhaustive) == (12, True)


# ── driver._CACHE — the repair reaching a warm reader ───────────────────────────────────────
def test_a_repair_written_by_another_handle_reaches_a_warm_reader(tmp_path):
    """Read `ic` warm, rewrite it through an independent store handle, read again: the new value is
    served with no restart, no `invalidate()` call and no refresh of any kind."""
    from crystal.ontology import driver as wn
    reader = _keyed(_lattice(tmp_path))
    _seed(reader, ic=1.0)
    wn.bind(reader)
    assert wn.synset("thing.n.01").ic() == 1.0

    writer = _lattice(tmp_path)                      # a second handle: the reader is never told
    writer.artifacts.put_artifact({
        "id": "wn-thing.n.01", "content_type": WN, "state": "committed", "pos": "n",
        "lemmas": ["thing"], "hypernyms": ["root.n.01"], "ic": 9.5})

    assert wn.synset("thing.n.01").ic() == 9.5, (
        "STALE: the repaired ic never reached the running reader — this is the 141,102-value "
        "defect of 2026-08-01 reproduced")


def test_an_unchanged_store_recomputes_nothing(tmp_path):
    """The negative control for the test above. Serving the new value is also what a cache that has
    been quietly turned off does, and that is the worse outcome: 1,566 µs cold against 17.4 µs warm
    for one synset plus its coordinate. So the store reads are counted — with no write there are
    none, and the same object comes back."""
    from crystal.ontology import driver as wn
    L = _keyed(_lattice(tmp_path))
    _seed(L)
    wn.bind(L)
    s0 = wn.synset("thing.n.01")

    reads = []
    orig = L.artifacts.get_artifact
    L.artifacts.get_artifact = lambda aid: (reads.append(aid), orig(aid))[1]
    try:
        for _ in range(50):
            assert wn.synset("thing.n.01") is s0, "a rebuilt object with no data change"
    finally:
        L.artifacts.get_artifact = orig
    assert reads == [], (
        "the cache re-read the store %d times with nothing written — the freshness check has "
        "replaced the cache rather than protected it" % len(reads))


def test_an_edge_only_write_invalidates_the_synset(tmp_path):
    """The vertex row does not move when only its edges change, and on the live corpus the taxonomy
    is the edges (`wn-dog.n.01` carries no `hypernyms` field). A stamp over the vertex alone would
    keep serving a synset whose parents had moved, so the stamp covers both."""
    from crystal.ontology import driver as wn
    reader = _keyed(_lattice(tmp_path))
    reader.artifacts.put_many([
        {"id": "wn-root.n.01", "content_type": WN, "state": "committed", "pos": "n", "ic": 0.0},
        {"id": "wn-leaf.n.01", "content_type": WN, "state": "committed", "pos": "n", "ic": 3.0},
        {"id": "wn-alt.n.01", "content_type": WN, "state": "committed", "pos": "n", "ic": 1.0},
    ])
    reader.graph.add_edge("wn-leaf.n.01", "wn-root.n.01", "hypernym", {})
    wn.bind(reader)
    before = wn.synset("leaf.n.01")
    ver_before = reader.artifacts.version_of("wn-leaf.n.01")
    assert [h.name() for h in before.hypernyms()] == ["root.n.01"]

    writer = _lattice(tmp_path)
    writer.graph.add_edge("wn-leaf.n.01", "wn-alt.n.01", "hypernym", {})
    assert reader.artifacts.version_of("wn-leaf.n.01") == ver_before, (
        "the vertex row moved, so this no longer tests the EDGE half of the stamp")

    after = wn.synset("leaf.n.01")
    assert after is not before, "an edge-only write did not invalidate the cached synset"
    assert sorted(h.name() for h in after.hypernyms()) == ["alt.n.01", "root.n.01"]


# ── geometry._DENSE_CACHE — the derived layer ───────────────────────────────────────────────
def test_the_coordinate_moves_with_the_ic_it_is_built_from(tmp_path):
    """The coordinate follows the IC it is built from. Keyed on `id(ic)` the cache could never
    miss, because `load_ic()` returns None and `id(None)` is a constant."""
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    reader = _keyed(_lattice(tmp_path))
    _seed(reader, ic=1.0)
    wn.bind(reader)
    ic = g.load_ic()
    v0 = g.dense_vec(wn.synset("thing.n.01"), ic)
    n0 = float((v0 * v0).sum()) ** 0.5
    assert n0 > 0

    writer = _lattice(tmp_path)
    writer.artifacts.put_artifact({
        "id": "wn-thing.n.01", "content_type": WN, "state": "committed", "pos": "n",
        "lemmas": ["thing"], "hypernyms": ["root.n.01"], "ic": 9.5})

    v1 = g.dense_vec(wn.synset("thing.n.01"), ic)
    n1 = float((v1 * v1).sum()) ** 0.5
    assert abs(n1 - n0) > 1e-9, "the coordinate did not move when the IC it is built from did"


def test_an_ancestors_repair_reaches_a_childs_coordinate(tmp_path):
    """The case per-artifact stamping alone cannot cover. `sparse_vec` walks `tree_path`, so a
    child's coordinate is a function of every ancestor's IC, and repairing the parent leaves the
    child's own artifact untouched. This is why the dense cache is keyed on the generation rather
    than on the child's stamp."""
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    reader = _keyed(_lattice(tmp_path))
    reader.artifacts.put_many([
        {"id": "wn-mid.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "hypernyms": [], "ic": 1.0},
        {"id": "wn-kid.n.01", "content_type": WN, "state": "committed", "pos": "n",
         "hypernyms": ["mid.n.01"], "ic": 5.0},
    ])
    wn.bind(reader)
    ic = g.load_ic()
    v0 = g.dense_vec(wn.synset("kid.n.01"), ic)

    writer = _lattice(tmp_path)                       # only the parent is repaired
    writer.artifacts.put_artifact({"id": "wn-mid.n.01", "content_type": WN, "state": "committed",
                                   "pos": "n", "hypernyms": [], "ic": 4.5})

    v1 = g.dense_vec(wn.synset("kid.n.01"), ic)
    assert abs(float((v1 * v1).sum()) ** 0.5 - float((v0 * v0).sum()) ** 0.5) > 1e-9, (
        "a parent's repaired IC never reached its child's coordinate")


def test_the_dense_cache_still_caches(tmp_path):
    """Negative control for the two above: with nothing written the coordinate is not rebuilt.
    Measured cost of rebuilding anyway: 37.6 µs per concept per call, on every row of every
    screen."""
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    L = _keyed(_lattice(tmp_path))
    _seed(L)
    wn.bind(L)
    ic = g.load_ic()
    s = wn.synset("thing.n.01")
    g.dense_vec(s, ic)

    builds = []
    orig = g.sparse_vec
    g.sparse_vec = lambda syn, _ic: (builds.append(syn.name()), orig(syn, _ic))[1]
    try:
        for _ in range(50):
            g.dense_vec(s, ic)
    finally:
        g.sparse_vec = orig
    assert builds == [], "the coordinate was rebuilt %d times with nothing written" % len(builds)


# ── projection._BASIS_CACHE — the derived basis ──────────────────────────────────────────────
def test_the_corpus_basis_follows_the_artifact(tmp_path):
    """A rebuilt basis reaches a running reader. Keyed on `id(store)` the cache says which store and
    never when, so when k went 195 -> 280 on 71 every running process kept projecting onto the old
    195 directions.

    The gate is the lineage's content type. A derivation writes a new generation and archives the
    old, so `stamp(store, BASIS_ID)` — one row's `(_origin, _seq)` — need never move across a
    rebuild and would verify clean for ever ([[verification-that-cannot-fail]]). This test writes
    the second generation the way `build_basis` does, as a revision rather than an overwrite, so it
    exercises that shape."""
    from ember.signal import projection as pj
    pj._BASIS_CACHE.clear()

    def body(k, d):
        return {"content_type": pj.BASIS_CONTENT_TYPE, "k": k, "d": d,
                "basis": [[float(i == j) for j in range(d)] for i in range(k)]}

    reader = _lattice(tmp_path)
    reader.artifacts.put_artifact(dict(body(3, 8), id=pj.BASIS_ROOT, state="committed"))
    B0 = pj.load_basis(reader)
    assert B0.shape == (3, 8)
    assert pj.load_basis(reader) is B0, "the basis was re-hydrated with nothing written"

    writer = _lattice(tmp_path)
    writer.artifacts.revise(writer.artifacts.head_of(pj.BASIS_ROOT)["id"], body(5, 8))
    B1 = pj.load_basis(reader)
    assert B1.shape == (5, 8), "STALE BASIS: the rebuilt basis never reached the reader"
    assert pj.load_basis(reader) is B1, (
        "NEGATIVE CONTROL: the basis is being re-hydrated on every read — that is not a fix, it "
        "is a disabled cache")
    pj._BASIS_CACHE.clear()


# ── match._OFFER_CACHE — a set-sized derivation ─────────────────────────────────────────────
def test_offers_follow_an_edited_operator_without_anyone_calling_invalidate(tmp_path):
    """The offers are derived from a whole content type, so that is what is stamped, and a newly
    written operator arrives without anyone calling `match.invalidate(store)`. With invalidation as
    the only route, an operator edited in place or replicated in from a peer stays invisible until
    a restart — `capability.register_remote_host` is its one caller."""
    from ember.ontology import match as m
    OP = "application/vnd.agience.operator+json"
    reader = _lattice(tmp_path)
    reader.artifacts.put_artifact({"id": "op.alpha", "content_type": OP, "state": "committed",
                                   "context": "dog"})
    assert m._offers(reader, refresh=True)["n"] == 1

    writer = _lattice(tmp_path)
    writer.artifacts.put_artifact({"id": "op.beta", "content_type": OP, "state": "committed",
                                   "context": "cat"})
    assert m._offers(reader)["n"] == 2, (
        "a newly written operator stayed invisible — invalidate() is not being called and must "
        "not need to be")


def test_offers_are_not_rebuilt_when_nothing_of_that_type_was_written(tmp_path):
    """Negative control, and the reason the offers are stamped per content type rather than on the
    whole-store write mark: rebuilding them measures 215–270 ms on 71, and on a live node an
    unrelated write (a chat message) lands every turn."""
    from ember.ontology import match as m
    OP = "application/vnd.agience.operator+json"
    reader = _lattice(tmp_path)
    reader.artifacts.put_artifact({"id": "op.alpha", "content_type": OP, "state": "committed",
                                   "context": "dog"})
    first = m._offers(reader, refresh=True)

    writer = _lattice(tmp_path)                       # a write of an unrelated type
    writer.artifacts.put_artifact({"id": "note-1", "content_type": "text/markdown",
                                   "state": "committed", "content": "hello"})
    assert m._offers(reader) is first, (
        "an unrelated write discarded the offers — the stamp is wider than the derivation")
