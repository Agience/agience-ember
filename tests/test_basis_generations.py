"""Each basis generation is its own artifact, and its id is derived from its content.

`build_basis` writes a new generation under the `geom.corpus-basis` lineage rather than overwriting
one fixed id. A basis generation is something that was measured
([[everything-is-an-artifact]]), so it survives being superseded: a Screen that pooled planes under
an older width can still ask what that basis was.

`pooling.coordinate_id` names the generation by that artifact id, not by `freshness.stamp(store,
BASIS_ID)`. A stamp is a side-car beside the artifact and is not portable — `_seq` is allocation
order in one store, so a replica holding the byte-identical basis would mint a different token and
its planes would be counted as a coordinate mismatch.

Each test is paired with the control that stops it passing vacuously. Hermetic: real
`mantle.db` stores in `tmp_path`, a stubbed coordinate, no live corpus, no running service.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

import _paths

sys.path.insert(0, str(_paths.repo("agience-mantle") / "src"))

from mantle.db import open_lattice  # noqa: E402

from ember.signal import pooling  # noqa: E402
from ember.signal import projection as pj  # noqa: E402

#: A narrow, tall cloud on purpose — the shape `build_basis` needs for a well-conditioned read
#: (~85k nouns against D=2048, T/F ~ 41). A near-square cloud sends `_basis_from_cloud` down its
#: SVD fallback, which is a different measurement and not this file's subject.
D = 16


def _lattice(tmp_path, name="lattice.db", origin="test-node"):
    L = open_lattice(str(tmp_path / name), origin=origin)
    L.artifacts.ensure_schema()
    L.graph.ensure_schema()
    return L


class _Syn:
    def __init__(self, name, vec):
        self._n, self.vec = name, vec

    def name(self):
        return self._n


def _corpus(monkeypatch, n, seed):
    """A stubbed noun coordinate cloud: `n` rows of a sparse D-wide coordinate. `build_basis` reads
    the corpus through the crystal ontology driver, which is stubbed here, so this is the whole
    corpus as far as it is concerned."""
    rng = np.random.default_rng(seed)
    vecs = {}
    for i in range(n):
        v = np.zeros(D)
        v[rng.choice(D, 4, replace=False)] = rng.normal(size=4)
        vecs["c%d" % i] = v
    syns = [_Syn(k, v) for k, v in vecs.items()]
    from crystal.ontology import geometry as g
    from crystal.ontology import driver as wn
    monkeypatch.setattr(wn, "all_synsets", lambda pos=None: list(syns))
    monkeypatch.setattr(wn, "synset", lambda nm: (_Syn(nm, vecs[nm]) if nm in vecs else None))
    monkeypatch.setattr(g, "load_ic", lambda: None)
    monkeypatch.setattr(g, "dense_vec", lambda s, ic=None, center=None: s.vec)
    return vecs


@pytest.fixture(autouse=True)
def _clean():
    """`_BASIS_CACHE` is a process singleton keyed on `id(store)`, and CPython reuses addresses —
    a test that does not clear it can read the previous test's basis out of a dead store's slot."""
    pj._BASIS_CACHE.clear()
    yield
    pj._BASIS_CACHE.clear()


# ═════ 1 — a second derivation does not destroy the first ════════════════════════════════════════

def test_a_rebuild_leaves_the_superseded_generation_QUERYABLE(tmp_path, monkeypatch):
    """After a second derivation lands, the first generation is still addressable by its own id,
    with its own width. Under one fixed id the second derivation would replace the first — same id,
    new matrix — and the old width would cease to exist.

    The control comes first: the two ids must differ and the two `k` must differ. If the stub
    produced the same basis twice, `revise` would idempotently return the same row and the
    assertions below would prove nothing."""
    L = _lattice(tmp_path)

    _corpus(monkeypatch, 200, seed=1)
    assert pj.build_basis(L) is not None
    gen1 = L.artifacts.head_of(pj.BASIS_ROOT)
    k1 = gen1["k"]

    pj._BASIS_CACHE.clear()
    _corpus(monkeypatch, 400, seed=2)                 # a different corpus -> a different basis
    assert pj.build_basis(L) is not None
    gen2 = L.artifacts.head_of(pj.BASIS_ROOT)

    assert gen1["id"] != gen2["id"], "the rebuild wrote under the same id — this IS the defect"
    assert k1 != gen2["k"], (
        "both derivations produced k=%r, so this store never saw a coordinate change and the "
        "assertions below cannot fail" % (k1,))

    # the superseded generation is still there, with its own width, addressable by its own id.
    old = L.artifacts.get_artifact(gen1["id"])
    assert old is not None and old["k"] == k1
    assert np.asarray(old["basis"]).shape == (k1, D)
    assert old["state"] == "archived" and old["superseded_by"] == gen2["id"]
    # ...and the lineage holds both, sharing one root.
    # Order is not asserted. `versions_of` documents "oldest first" (`ORDER BY _origin, _seq`), but
    # `revise` writes the new version before archiving the old, so the archived prior is re-stamped
    # with a later `_seq` and the lineage reads newest-first. That ordering is mantle's, and this
    # file measures the projection module.
    vs = L.artifacts.versions_of(pj.BASIS_ROOT)
    assert {v["id"] for v in vs} == {gen1["id"], gen2["id"]}
    assert {v.get("root_id") or v["id"] for v in vs} == {pj.BASIS_ROOT}


def test_the_generation_id_is_DERIVED_FROM_THE_BASIS_not_from_a_clock_or_a_counter(
        tmp_path, monkeypatch):
    """The generation id is a function of the basis. An id minted from a counter (`~1`, `~2`) or
    carrying a fresh `created_time` would make a re-derivation of an identical basis a new
    generation, and every Screen pooled under the old id would start counting `dropped:basis` on a
    coordinate that never moved — a fabricated coordinate change rather than a hidden one.

    Same corpus, derived twice: one artifact, one lineage, nothing new written."""
    L = _lattice(tmp_path)
    _corpus(monkeypatch, 200, seed=1)
    pj.build_basis(L)
    first = L.artifacts.head_of(pj.BASIS_ROOT)

    pj._BASIS_CACHE.clear()
    pj.build_basis(L)                                 # byte-identical basis, derived again
    second = L.artifacts.head_of(pj.BASIS_ROOT)

    assert second["id"] == first["id"], "an identical basis minted a second generation"
    assert len(L.artifacts.versions_of(pj.BASIS_ROOT)) == 1


def test_a_store_that_cannot_VERSION_is_refused_rather_than_overwritten(tmp_path, monkeypatch):
    """A store with no lineage support raises where an operator can see it. Falling back to
    `put_artifact(id=BASIS_ROOT)` would succeed quietly: the previous generation would be gone,
    every pooled Screen stranded, and nothing raised."""
    class _NoLineage:
        def __init__(self, inner):
            self.inner = inner
            self.written = []

        def put_artifact(self, doc, **kw):
            self.written.append(doc)
            return doc

        def __getattr__(self, k):
            if k in ("head_of", "revise"):
                raise AttributeError(k)
            return getattr(self.inner, k)

    class _Store:
        def __init__(self, arts):
            self.artifacts = arts

    L = _lattice(tmp_path)
    arts = _NoLineage(L.artifacts)
    _corpus(monkeypatch, 200, seed=1)
    with pytest.raises(TypeError, match="cannot version"):
        pj.build_basis(_Store(arts))
    # `_author_ref` mints the person artifact before the basis is written, so the assertion is about
    # the basis specifically: "nothing at all was written" would be false for a reason unrelated to
    # the raise.
    assert [d for d in arts.written if d.get("content_type") == pj.BASIS_CONTENT_TYPE] == [], \
        "it wrote the basis anyway — the refusal is decorative"


# ═════ 2 — the coordinate token is the artifact, not a stamp beside it ═══════════════════════════

def test_the_coordinate_token_IS_the_generation_and_survives_replication(tmp_path, monkeypatch):
    """The token names the generation's artifact, so it is the same on every node holding that
    generation. A token folding `freshness.stamp(store, BASIS_ID)` = `((origin, _seq),)` would not
    be: `_seq` is allocation order in one store, so two nodes holding the byte-identical basis would
    produce different tokens and a peer's plane would be counted `dropped:basis` against a coordinate
    that never differed.

    The control is the first assertion: the two stores must genuinely disagree about `_seq`, or
    equality here would say nothing about portability."""
    from crystal.ontology import freshness as fr

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    A = _lattice(tmp_path / "a", origin="node-a")
    _corpus(monkeypatch, 200, seed=1)
    pj.build_basis(A)
    doc = A.artifacts.head_of(pj.BASIS_ROOT)

    B = _lattice(tmp_path / "b", origin="node-b")
    B.artifacts.put_artifact(dict(doc))               # the same generation, replicated verbatim

    assert fr.stamp(A, doc["id"]) != fr.stamp(B, doc["id"]), (
        "the two stores agree on (_origin, _seq), so the retired token would have compared equal "
        "and this test could not detect a non-portable identity")

    pj._BASIS_CACHE.clear()
    ta = pooling.coordinate_id(A, corpus_basis=True)
    pj._BASIS_CACHE.clear()
    tb = pooling.coordinate_id(B, corpus_basis=True)
    assert ta is not None and ta == tb, "the same basis got two different tokens: %r vs %r" % (ta, tb)
    assert doc["id"] in ta, "the token does not name the generation it claims to identify"


def test_the_token_MOVES_when_the_basis_is_rebuilt(tmp_path, monkeypatch):
    """The negative control for the test above. A token equal everywhere could also be equal across
    a rebuild, which would let a Screen pool planes of one width onto a pool carrying another.
    Portable and constant are different properties, and this pins the difference."""
    L = _lattice(tmp_path)
    _corpus(monkeypatch, 200, seed=1)
    pj.build_basis(L)
    before = pooling.coordinate_id(L, corpus_basis=True)

    pj._BASIS_CACHE.clear()
    _corpus(monkeypatch, 400, seed=2)
    pj.build_basis(L)
    pj._BASIS_CACHE.clear()
    after = pooling.coordinate_id(L, corpus_basis=True)

    assert before is not None and after is not None
    assert before != after, "the token did not move across a rebuilt basis"


def test_a_store_with_NO_basis_cannot_name_the_corpus_coordinate(tmp_path):
    """A store holding no basis has no corpus coordinate to name, so `coordinate_id` gives `None`.
    `freshness.stamp` answers `(None,)` for an artifact that does not exist — a tuple, hence a
    confident-looking token — and two stores with no basis would then compare equal on a coordinate
    neither of them has ([[absence-is-not-an-affirmative-claim]]).

    The raw dense arm is the control: it does not come from an artifact, so it is always nameable and
    stays nameable here."""
    L = _lattice(tmp_path)
    assert L.artifacts.head_of(pj.BASIS_ROOT) is None
    assert pj.basis_head(L) is None
    assert pooling.coordinate_id(L, corpus_basis=True) is None
    assert pooling.coordinate_id(L, corpus_basis=False) is not None


def test_the_live_generation_follows_a_rebuild_WITHOUT_a_restart(tmp_path, monkeypatch):
    """A reader picks up a rebuilt basis without restarting, because the cache gate is the lineage's
    content type. A gate on `stamp(store, BASIS_ID)` — one row's `(_origin, _seq)` — cannot see a
    derivation that writes a new row instead of that one, so the old row's version never moves and
    the reader serves the superseded basis indefinitely.

    The negative control is in the same test: with nothing written the basis is not re-hydrated. A
    cache that is simply disabled would pass the freshness half and cost 327 ms per screen."""
    reader = _lattice(tmp_path)
    _corpus(monkeypatch, 200, seed=1)
    writer = _lattice(tmp_path)                       # a second handle — the repair process
    pj.build_basis(writer)

    B0 = pj.load_basis(reader)
    assert B0 is not None
    assert pj.load_basis(reader) is B0, "the basis was re-hydrated with nothing written"

    _corpus(monkeypatch, 400, seed=2)
    pj.build_basis(writer)
    B1 = pj.load_basis(reader)
    assert B1 is not None and B1.shape != B0.shape, (
        "STALE BASIS: the rebuilt basis never reached the reader (%r)" % (B1.shape,))
    assert pj.load_basis(reader) is B1, (
        "NEGATIVE CONTROL: the basis is re-hydrated on every read — that is a disabled cache, not "
        "a fix")


def test_the_cheap_gate_is_LOAD_BEARING_and_cannot_serve_a_store_that_will_not_say(
        tmp_path, monkeypatch):
    """The two-level cache gate is load bearing at both levels, and the two ways it can go wrong are
    opposite to each other.

    1. The cheap arm has to fire. `frame_rows` asks for the basis on every screen, so a warm read
       that always paid the set-sized discriminator would spend 230 µs learning that nothing
       anywhere was written. Asserted by counting discriminator reads: with nothing written it is
       not consulted at all.
    2. The cheap arm may only fire when the store reports a write mark. On a store that reports
       none, `None == None` would serve the cached value for ever — a check that cannot fail.
       Asserted by muting `write_mark` and showing the discriminator is consulted every time."""
    from crystal.ontology import freshness as fr

    reader = _lattice(tmp_path)
    _corpus(monkeypatch, 200, seed=1)
    pj.build_basis(_lattice(tmp_path))
    assert pj.load_basis(reader) is not None

    calls = []
    real_set = fr.set_stamp
    monkeypatch.setattr(fr, "set_stamp", lambda s, ct, **kw: (calls.append(ct), real_set(s, ct, **kw))[1])

    for _ in range(5):
        pj.load_basis(reader)
    assert calls == [], "the cheap write-mark arm never fired: %d discriminator reads" % len(calls)

    monkeypatch.setattr(fr, "write_mark", lambda s: None)      # a store that will not say
    for _ in range(3):
        pj.load_basis(reader)
    assert len(calls) == 3, (
        "a store that cannot report a write mark was served from the cheap arm anyway — "
        "`None == None` is a check that cannot fail (%d reads)" % len(calls))
