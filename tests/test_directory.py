"""Gossip region directory — a routed query maps to where the source lives.

    python tests/test_directory.py      # from the ember repo root

Three nodes each hold a different slice of the anchor cells. A directory learns
who-holds-what from their manifests, gossips transitively (merge), prefers the
denser/more-coherent replica, and drives a selective multi-peer pull that self-heals
when the best provider goes dark. All pulls stay blind + content_root-verified.
"""
# The mesh modules are named absolutely below. This file lives in `tests/`, outside the package, so
# a relative import would resolve against `tests` rather than against `mantle.mesh`.
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mantle.search.anchors.anchorset import AnchorSet
from mantle.mesh.anchor_routing import cell_region, route_query_regions
from mantle.mesh.directory import RegionDirectory, pull_regions
from mantle.mesh.node import MeshNode

DIM = 8
PRINCIPAL = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
COLLECTION = "default"
MODEL = "all-MiniLM-L6-v2"


def _ct(cluster: str) -> bytes:
    return b"ENCv1:" + hashlib.sha256(cluster.encode()).digest() * 4


def _anchorset() -> AnchorSet:
    a = AnchorSet(model_id=MODEL, dim=DIM)
    for i in range(6):
        v = np.full(DIM, 0.05, dtype=np.float32)
        v[i] = 1.0
        a.add_text(f"anchor-{i}", v)
    return a


def main() -> None:
    aset = _anchorset()
    anchors = aset.anchors
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    def region(i: int) -> str:
        return cell_region(PRINCIPAL, COLLECTION, anchors[i].anchor_id)

    def item(i: int) -> Dict[str, bytes]:
        return {f"cell:{anchors[i].anchor_id}": _ct(anchors[i].anchor_id)}

    # Working sets spread across three nodes; anchor#2's cell is replicated on A and B, and B's
    # replica names an origin while A's does not. B is therefore the copy a puller can compute
    # agreement from; A carries the bytes and nothing about who stands behind them.
    A, B, C = MeshNode("A"), MeshNode("B"), MeshNode("C")
    A.put_shard(region(0), item(0), version=1, authority="cell-authority", priv=priv)
    A.put_shard(region(1), item(1), version=1, authority="cell-authority", priv=priv)
    A.put_shard(region(2), item(2), version=1, authority="cell-authority", priv=priv,
                attest={f"cell:{anchors[2].anchor_id}": {}})   # replica that names no origin
    B.put_shard(region(2), item(2), version=1, authority="cell-authority", priv=priv,
                attest={f"cell:{anchors[2].anchor_id}": {"origin": "cell-authority"}})   # names its origin
    B.put_shard(region(3), item(3), version=1, authority="cell-authority", priv=priv)
    C.put_shard(region(4), item(4), version=1, authority="cell-authority", priv=priv)
    C.put_shard(region(5), item(5), version=1, authority="cell-authority", priv=priv)
    registry = {"A": A, "B": B, "C": C}
    print("[0] 3 nodes; anchor#2 cell replicated on A (weak) + B (strong)")

    # Gossip: node near A learns A+B directly; a far directory learns B+C; merge unions them.
    dir_near = RegionDirectory()
    dir_near.observe("A", A.manifests())
    dir_near.observe("B", B.manifests())
    dir_far = RegionDirectory()
    dir_far.observe("B", B.manifests())
    dir_far.observe("C", C.manifests())
    dir_near.merge(dir_far)   # transitive: A's directory now knows C's cells too
    assert set(dir_near.peers()) == {"A", "B", "C"}
    assert region(4) in dir_near.regions() and region(5) in dir_near.regions()
    print(f"[1] gossip merge: directory knows {len(dir_near.peers())} peers, "
          f"{len(dir_near.regions())} regions (incl. C's, learned transitively)")

    # Density/coherence preference: both A and B hold anchor#2, B preferred.
    provs = dir_near.providers(region(2))
    assert [p for p, _ in provs] == ["B", "A"], provs
    print(f"[2] anchor#2 providers ranked: {[p for p, _ in provs]} (B's stronger replica first)")

    # Route a query to anchor#2 and pull its working set from wherever the good copies live.
    q = anchors[2].embedding.astype(np.float32).copy()
    q[5] += 0.02
    routed = route_query_regions(aset, q, PRINCIPAL, COLLECTION, nprobe=4)
    assert routed[0] == region(2)
    into = MeshNode("query-node")

    def get_shard(peer_id, reg):
        return registry[peer_id].get_shard(reg)

    satisfied = pull_regions(dir_near, routed, into, pub, get_shard)
    assert satisfied[region(2)] == "B", satisfied           # pulled from the strong replica
    assert all(v is not None for v in satisfied.values()), satisfied
    assert all(into.has_region(r) for r in routed)
    print(f"[3] routed {len(routed)} cells; pulled from {sorted(set(satisfied.values()))}; "
          f"anchor#2 came from '{satisfied[region(2)]}'")

    # Self-heal: B goes dark → the pull fails over to A's (verified) replica.
    def get_shard_B_down(peer_id, reg):
        if peer_id == "B":
            raise OSError("connection refused")
        return registry[peer_id].get_shard(reg)

    into2 = MeshNode("query-node-2")
    satisfied2 = pull_regions(dir_near, [region(2)], into2, pub, get_shard_B_down)
    assert satisfied2[region(2)] == "A", satisfied2
    assert into2.has_region(region(2))
    print(f"[4] B down -> anchor#2 self-healed from '{satisfied2[region(2)]}' (failover)")

    # Coverage gap: a region no peer holds surfaces as no-provider (not hidden).
    ghost = cell_region(PRINCIPAL, COLLECTION, "no-such-anchor")
    assert dir_near.plan([ghost])[ghost] == []
    satisfied3 = pull_regions(dir_near, [ghost], MeshNode("q3"), pub, get_shard)
    assert satisfied3[ghost] is None
    print("[5] unheld region surfaces as a coverage gap (no provider), not a silent miss")

    # Gossip wire round-trip + merge idempotency.
    rt = RegionDirectory.from_dict(dir_near.to_dict())
    assert rt.to_dict() == dir_near.to_dict()
    before = dir_near.to_dict()
    dir_near.merge(rt)  # merging a copy changes nothing
    assert dir_near.to_dict() == before
    print("[6] directory to_dict/from_dict round-trips; merge is idempotent")

    print("\nOK - a routed query maps to where the source lives: best replica preferred, "
          "multi-peer pull, self-healing failover - all blind + verified.")


if __name__ == "__main__":
    main()
