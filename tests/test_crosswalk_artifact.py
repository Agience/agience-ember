"""A cross-walk is an artifact: fit once, share, and verify rather than trust.

Every failure mode here is silent. A wrong cross-walk still projects, still returns unit vectors and
still routes; it routes into cells chosen by the wrong geometry, and nothing raises. So these tests
catch what would otherwise look like success.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantle.search.anchors.anchorset import AnchorSet
from mantle.search.anchors.crosswalk import fit_crosswalk
from prism.mass import Provenance, provenance_of
from mantle.search.anchors.crosswalk_artifact import (
    CROSSWALK_CONTENT_TYPE, anchorset_fingerprint, from_artifact, to_artifact,
    verify_crosswalk_artifact,
)
from ember.embed import HashEmbedder

CANON, CANON_DIM = "canonical-bge-like", 64
LEAF_DIM = 16
LABELS = ["ontology", "storage", "identity", "streaming", "embeddings", "grants"]


def _anchors(n: int = 6) -> AnchorSet:
    a = AnchorSet(model_id=CANON, dim=CANON_DIM)
    rng = np.random.default_rng(7)
    for i, label in enumerate(LABELS[:n]):
        v = rng.normal(size=CANON_DIM).astype(np.float32) * 0.05
        v[i] = 1.0
        a.add_text(label, v)
    return a


def _fit(anchors: AnchorSet, emb: HashEmbedder):
    return fit_crosswalk(
        emb.encode([a.label for a in anchors.anchors]),
        np.vstack([a.embedding for a in anchors.anchors]),
        source_model_id=emb.model_id, target_model_id=anchors.model_id, method="auto",
    )


def test_roundtrip_preserves_the_projection_exactly() -> None:
    """Serialisation must not perturb the matrix: a cross-walk that projects *almost* the same
    routes into a neighbouring cell, which is a wrong answer wearing a right answer's shape."""
    anchors, emb = _anchors(), HashEmbedder(dim=LEAF_DIM)
    cw = _fit(anchors, emb)
    back = from_artifact(to_artifact(cw, anchors))
    q = emb.encode(["ontology"])[0]
    assert np.allclose(cw.apply(q), back.apply(q), atol=1e-6)
    assert back.method == cw.method and back.dim_in == cw.dim_in and back.dim_out == cw.dim_out


def test_it_is_a_real_artifact_carrying_its_own_quality() -> None:
    anchors, emb = _anchors(), HashEmbedder(dim=LEAF_DIM)
    art = to_artifact(_fit(anchors, emb), anchors)
    assert art["content_type"] == CROSSWALK_CONTENT_TYPE
    # The residual travels with the walk, so a consumer decides before querying rather than after.
    # It is a block rather than a bare float, so a consumer names which number it means; `held_out`
    # is the one that carries a claim about unseen input.
    res = art["context"]["residual"]
    assert "in_sample" in res and "held_out" in res
    # observed: nobody asserted or guessed it — it is a deterministic computation over stated
    # inputs, reproducible by anyone holding the same anchors and embedder.
    assert provenance_of(art) is Provenance.OBSERVED


def test_every_leaf_on_the_same_embedder_fits_the_identical_walk() -> None:
    """The payoff: the fit is deterministic (lstsq/Procrustes — no seed, no stochasticity), so
    a thousand leaves compute the same matrix. Fit once, cache, share — instead of a thousand
    refits. Same id proves they're interchangeable in the cache."""
    anchors = _anchors()
    a1 = to_artifact(_fit(anchors, HashEmbedder(dim=LEAF_DIM)), anchors)
    a2 = to_artifact(_fit(anchors, HashEmbedder(dim=LEAF_DIM)), anchors)
    assert a1["id"] == a2["id"]
    assert a1["context"]["matrix_b64"] == a2["context"]["matrix_b64"]


def test_a_grown_anchorset_changes_the_id() -> None:
    """The trap: anchors grow continuously. A walk fitted against yesterday's anchors still
    projects and still routes — into yesterday's cells. Making the fingerprint part of the id
    turns that silent mis-projection into an ordinary cache miss."""
    small, grown = _anchors(4), _anchors(6)
    assert anchorset_fingerprint(small) != anchorset_fingerprint(grown)
    emb = HashEmbedder(dim=LEAF_DIM)
    assert to_artifact(_fit(small, emb), small)["id"] != to_artifact(_fit(grown, emb), grown)["id"]


def test_fingerprint_ignores_insertion_order() -> None:
    """An AnchorSet is a set. If order changed its identity, two leaves that learned the same
    anchors in different orders could never adopt each other's cross-walks."""
    a, b = AnchorSet(CANON, CANON_DIM), AnchorSet(CANON, CANON_DIM)
    rng = np.random.default_rng(3)
    vecs = [(l, rng.normal(size=CANON_DIM).astype(np.float32)) for l in LABELS[:4]]
    for l, v in vecs:
        a.add_text(l, v)
    for l, v in reversed(vecs):
        b.add_text(l, v)
    assert anchorset_fingerprint(a) == anchorset_fingerprint(b)


def test_verify_rejects_every_silent_mismatch() -> None:
    anchors, emb = _anchors(), HashEmbedder(dim=LEAF_DIM)
    art = to_artifact(_fit(anchors, emb), anchors)

    ok, why = verify_crosswalk_artifact(art, anchors, emb.model_id)
    assert ok, why

    # wrong embedder — would project from a space this leaf doesn't inhabit
    ok, why = verify_crosswalk_artifact(art, anchors, "some-other-embedder")
    assert not ok and "this leaf derives" in why

    # anchors grew — would route into stale cells, plausibly
    ok, why = verify_crosswalk_artifact(art, _anchors(4), emb.model_id)
    assert not ok and "changed" in why

    # not a cross-walk at all
    ok, why = verify_crosswalk_artifact({"content_type": "application/json"}, anchors, emb.model_id)
    assert not ok and "content_type" in why


def test_malformed_artifact_raises_rather_than_guessing() -> None:
    """A float matrix without its dtype is a very convincing wrong answer."""
    with pytest.raises(ValueError):
        from_artifact({"content_type": CROSSWALK_CONTENT_TYPE, "context": {"method": "linear"}})
    with pytest.raises(ValueError):
        from_artifact({"content_type": "text/plain", "context": {}})


# --------------------------------------------------------------------------- adoption
def test_a_leaf_adopts_a_shared_walk_instead_of_refitting() -> None:
    """The payoff, end to end: leaf A fits and publishes; leaf B adopts and never refits —
    and both project identically, so they route to the same cells."""
    from ember.embed import Aligner

    anchors = _anchors()
    a = Aligner(HashEmbedder(dim=LEAF_DIM), anchors).fit()
    assert not a.adopted                      # A did the work
    published = a.as_artifact(fitted_by="leaf-A")

    b = Aligner(HashEmbedder(dim=LEAF_DIM), anchors).fit(cached=published)
    assert b.adopted, "B must reuse the shared walk, not refit"

    q = "ontology"
    assert np.allclose(a.encode_query(q), b.encode_query(q), atol=1e-6)


def test_a_stale_walk_is_a_cache_miss_not_a_crash() -> None:
    """Anchors grow continuously, so leaves are handed walks fitted against older sets as a matter
    of course. Adopting one would route into yesterday's cells, plausibly and silently, and raising
    would make a routine event fatal; refitting quietly keeps a routine event routine."""
    from ember.embed import Aligner

    grown, stale_anchors = _anchors(6), _anchors(4)
    stale = Aligner(HashEmbedder(dim=LEAF_DIM), stale_anchors).fit().as_artifact()

    al = Aligner(HashEmbedder(dim=LEAF_DIM), grown).fit(cached=stale)
    assert not al.adopted, "a walk for other anchors must be ignored"
    assert al.encode_query("ontology").shape[-1] == CANON_DIM   # refitted, still works


def test_a_foreign_embedders_walk_is_ignored() -> None:
    from ember.embed import Aligner

    anchors = _anchors()
    other = HashEmbedder(dim=8)
    other.model_id = "someone-elses-embedder"
    foreign = Aligner(other, anchors).fit().as_artifact()

    al = Aligner(HashEmbedder(dim=LEAF_DIM), anchors).fit(cached=foreign)
    assert not al.adopted
    assert al.encode_query("ontology").shape[-1] == CANON_DIM


def test_native_embedder_has_no_walk_to_publish() -> None:
    from ember.embed import Aligner

    anchors = _anchors()
    native = HashEmbedder(dim=CANON_DIM)
    native.model_id = CANON
    al = Aligner(native, anchors).fit()
    assert al.native and not al.adopted
    with pytest.raises(ValueError):
        al.as_artifact()
