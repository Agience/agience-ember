"""Dark matter is surfacable through a workspace.

Mass gets an artifact into the geometry; edges get it into the answers. An undescribed artifact
still embeds and still bends routing, so it is retrieved by anything that reaches its region. The
scoping is structural: a region id carries the workspace, so a query addresses one workspace by
construction. These tests state that as a property, so a later cross-collection search meets it as
an assertion rather than as a coincidence.
"""
from __future__ import annotations

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ember import LocalCache
from mantle.search.anchors.anchorset import AnchorSet
from mantle.mesh.anchor_routing import cell_region

DIM = 16
MODEL = "test-embed"
ALICE = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
WORK = "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"      # a workspace (a collection artifact)
OTHER = "d8e2b3c4-5f6a-4b7c-9d8e-1f2a3b4c5d6e"      # a different one


def _anchorset() -> AnchorSet:
    a = AnchorSet(model_id=MODEL, dim=DIM)
    for i in range(4):
        v = np.full(DIM, 0.05, dtype=np.float32)
        v[i] = 1.0
        a.add_text(f"anchor-{i}", v)
    return a


def _vec(axis: int) -> np.ndarray:
    v = np.full(DIM, 0.05, dtype=np.float32)
    v[axis] = 1.0
    return v


def test_a_query_cannot_cross_collections() -> None:
    """The structural guarantee: every routed region carries this workspace's collection_id, so a
    question put to the manifold is scoped to a workspace by its address."""
    cache = LocalCache(_anchorset(), ALICE, WORK, nprobe=4)
    routed = cache.route(_vec(0)).regions
    assert routed
    for region in routed:
        principal, collection, _anchor = region.split("/", 2)
        assert principal == ALICE
        assert collection == WORK, f"query escaped its workspace into {collection}"


def test_another_workspaces_dark_matter_is_unreachable() -> None:
    """Same content, same anchor, different workspace: the region sets are disjoint, so one
    workspace's undescribed artifacts lie outside the addresses another's query reaches."""
    mine = LocalCache(_anchorset(), ALICE, WORK, nprobe=4)
    theirs = LocalCache(_anchorset(), ALICE, OTHER, nprobe=4)
    assert set(mine.route(_vec(0)).regions).isdisjoint(theirs.route(_vec(0)).regions)


def test_dark_matter_is_visible_inside_its_own_workspace() -> None:
    """The other half of the rule: freshly dropped, undescribed content is retrievable inside the
    workspace it was put in, so ingest is useful before a describer has run."""
    cache = LocalCache(_anchorset(), ALICE, WORK, nprobe=3)
    priv = Ed25519PrivateKey.generate()
    cache.put([("just-dropped", b"an undescribed note about ontologies", _vec(0))],
              version=1, authority="ember-local", priv=priv)
    hits = cache.search(_vec(0), k=5)
    assert [i.id for i, _ in hits] == ["just-dropped"]


def test_region_ids_are_built_from_the_workspaces_artifact_id() -> None:
    """The scoping is the collection artifact's id rather than its name, which is what makes the
    rule enforceable rather than advisory: the workspace is in the address."""
    cache = LocalCache(_anchorset(), ALICE, WORK, nprobe=2)
    anchor_id = cache.anchors.anchors[0].anchor_id
    assert cell_region(ALICE, WORK, anchor_id) == f"{ALICE}/{WORK}/{anchor_id}"
    assert cache.collection_id == WORK
