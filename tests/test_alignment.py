"""A small local embedder routes to the same cells as the canonical space.

This is the load-bearing claim behind "use something small if disconnected, upgrade during
sync". If a leaf's own embedder produced its own anchor ids, it would compute region names
nobody else uses and its shards would be unshareable — silently, because the routing would
still appear to work. The cross-walk is what prevents that.
"""
from __future__ import annotations

import numpy as np
import pytest

from ember.embed import Aligner, HashEmbedder
from mantle.search.anchors.anchorset import AnchorSet
from mantle.mesh.anchor_routing import cell_region, route_query_regions

CANON_MODEL = "canonical-bge-like"
CANON_DIM = 64          # the cloud's space
LEAF_DIM = 16           # the leaf's small space — deliberately different

LABELS = ["ontology", "storage", "identity", "streaming", "embeddings", "grants"]
# Artifact ids, not names: a collection is an artifact with the right edges.
PRINCIPAL, COLLECTION = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64", "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"


def _canonical_anchorset() -> AnchorSet:
    """Anchors in the canonical (big) space, one per topic."""
    a = AnchorSet(model_id=CANON_MODEL, dim=CANON_DIM)
    rng = np.random.default_rng(7)
    for i, label in enumerate(LABELS):
        v = rng.normal(size=CANON_DIM).astype(np.float32) * 0.05
        v[i] = 1.0
        a.add_text(label, v)
    return a


def test_crosswalk_is_dimension_agnostic() -> None:
    """A 16-dim leaf aligns to a 64-dim canonical space via a rectangular least-squares map."""
    anchors = _canonical_anchorset()
    al = Aligner(HashEmbedder(dim=LEAF_DIM), anchors).fit()
    assert not al.native
    q = al.encode_query("ontology")
    assert q.shape[-1] == CANON_DIM            # arrives in the canonical space
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)   # unit-norm, ready to route


def test_leaf_routes_to_the_same_region_ids_as_canonical() -> None:
    """The load-bearing claim: a small local embedder names the same cells, so its shards are
    shareable.

    We compare the leaf's routed region against the region the canonical anchor itself
    defines — not against another leaf. Region ids are anchor ids, so agreeing here means
    agreeing with the mesh.
    """
    anchors = _canonical_anchorset()
    al = Aligner(HashEmbedder(dim=LEAF_DIM), anchors).fit()

    agreed = 0
    for label in LABELS:
        canonical_region = cell_region(
            PRINCIPAL, COLLECTION,
            next(a.anchor_id for a in anchors.anchors if a.label == label),
        )
        routed = route_query_regions(anchors, al.encode_query(label), PRINCIPAL, COLLECTION,
                                     nprobe=3)
        # The leaf must at minimum consider the right cell; ideally it ranks it first.
        assert canonical_region in routed, f"{label!r}: leaf never routes to its own cell"
        if routed[0] == canonical_region:
            agreed += 1

    # The hash embedder is not semantic, so exact-nearest agreement on every label is not the
    # bar — the bar is that the ids are in the same namespace at all. A real embedder raises
    # this; the cross-walk's error_bound is what makes that measurable.
    assert agreed >= 1, "cross-walk produced no nearest-cell agreement at all"


def test_native_embedder_skips_the_crosswalk() -> None:
    """A leaf running the canonical model pays nothing for the abstraction."""
    anchors = _canonical_anchorset()
    native = HashEmbedder(dim=CANON_DIM)
    native.model_id = CANON_MODEL              # stand in for the canonical model
    al = Aligner(native, anchors).fit()
    assert al.native
    assert al.error_bound is None
    assert al.encode_query("ontology").shape[-1] == CANON_DIM


def test_alignment_needs_no_network() -> None:
    """Fitting uses only the cached AnchorSet: anchors carry both label and canonical embedding,
    so every training pair is available offline. Alignment is not a cloud dependency."""
    anchors = _canonical_anchorset()
    al = Aligner(HashEmbedder(dim=LEAF_DIM), anchors)
    al.fit()                                    # no channel exists in this process
    assert al.residual is not None              # and it reports what the projection cost
    assert 0.0 <= al.residual.in_sample <= 2.0


def test_upgrade_is_measurable_not_faith() -> None:
    """"Upgrade during sync" is justified by the held-out residual read against the derived null.

    The in-sample residual falls as the fit becomes more underdetermined, so more dimensions
    improve it by construction and a check like `bigger <= small + 0.5` would hold whether or not
    either cross-walk carried a single bit. The aligner reports `carries_information` as True,
    False or None instead, and leaves `error_bound` unset.

    On this path the projection carries nothing: held-out 1.026 against a null of 1.0. So what is
    asserted is that the aligner reports which of the two it is, and claims no information it
    lacks.
    """
    anchors = _canonical_anchorset()
    small = Aligner(HashEmbedder(dim=8), anchors).fit()
    bigger = Aligner(HashEmbedder(dim=32), anchors).fit()

    for al in (small, bigger):
        assert al.residual is not None, "a fitted non-native aligner must report its cost"
        # None is unknown; True and False are measured. A verdict, never a number standing in.
        assert al.carries_information in (True, False, None)

    # `error_bound` is unset, so it hands nobody a plausible-looking fidelity number.
    assert small.error_bound is None and bigger.error_bound is None

    # The claim that can fail: a cross-walk reporting that it carries information has a held-out
    # residual below the null. A fit that says "yes" while sitting at the null is caught here,
    # whichever way the measurement lands.
    from mantle.search.anchors.crosswalk import null_residual
    for al in (small, bigger):
        if al.carries_information:
            res = al.residual
            null, se = null_residual(res.dim_out, max(1, res.n_held_out))
            assert res.held_out < null - 2.0 * se, (
                "claimed information while sitting at the shared-nothing null")


def test_unfitted_aligner_fails_loudly() -> None:
    """An unfitted cross-walk raises rather than encoding: it would otherwise produce vectors in
    the wrong space and route them onto wrong-but-plausible cells."""
    anchors = _canonical_anchorset()
    al = Aligner(HashEmbedder(dim=LEAF_DIM), anchors)
    with pytest.raises(RuntimeError):
        al.encode_query("ontology")
