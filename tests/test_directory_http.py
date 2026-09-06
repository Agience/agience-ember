"""Gossip directory over the wire — discover, route, pull from the best live peer.

    python tests/test_directory_http.py      # from the ember repo root

Stands up real HTTP mesh nodes, gossips their manifests into a directory, routes a query
to its anchor cells, and pulls each from its best provider over sockets — then kills the
preferred peer and shows the pull self-heal onto the replica.
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
from mantle.mesh.node import MeshNode
from mantle.mesh.service import gossip, pull_regions_http, serve

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

    def region(i):
        return cell_region(PRINCIPAL, COLLECTION, anchors[i].anchor_id)

    def item(i) -> Dict[str, bytes]:
        return {f"cell:{anchors[i].anchor_id}": _ct(anchors[i].anchor_id)}

    A, B, C = MeshNode("A"), MeshNode("B"), MeshNode("C")
    A.put_shard(region(0), item(0), version=1, authority="cell-authority", priv=priv)
    A.put_shard(region(2), item(2), version=1, authority="cell-authority", priv=priv,
                attest={f"cell:{anchors[2].anchor_id}": {}})   # replica that names no origin
    B.put_shard(region(2), item(2), version=1, authority="cell-authority", priv=priv,
                attest={f"cell:{anchors[2].anchor_id}": {"origin": "cell-authority"}})   # names its origin
    B.put_shard(region(3), item(3), version=1, authority="cell-authority", priv=priv)
    C.put_shard(region(4), item(4), version=1, authority="cell-authority", priv=priv)

    sA = serve(A, 9721, host="127.0.0.1")
    sB = serve(B, 9722, host="127.0.0.1")
    sC = serve(C, 9723, host="127.0.0.1")
    urls = ["http://127.0.0.1:9721", "http://127.0.0.1:9722", "http://127.0.0.1:9723"]
    try:
        directory = gossip(urls)
        print(f"[1] gossiped {len(directory.peers())} peers, {len(directory.regions())} regions over HTTP")
        assert len(directory.peers()) == 3

        # anchor#2 held by A (weak) and B (strong) — B must rank first.
        ranked = [p for p, _ in directory.providers(region(2))]
        assert ranked[0] == "http://127.0.0.1:9722", ranked
        print(f"[2] anchor#2 providers: {[u[-5:] for u in ranked]} (:9722 strong replica first)")

        q = anchors[2].embedding.astype(np.float32).copy()
        q[5] += 0.02
        routed = route_query_regions(aset, q, PRINCIPAL, COLLECTION, nprobe=4)
        into = MeshNode("query-node")
        satisfied = pull_regions_http(directory, routed, into, pub)
        assert satisfied[region(2)] == "http://127.0.0.1:9722", satisfied
        got = [r for r in routed if into.has_region(r)]
        print(f"[3] pulled {len(got)}/{len(routed)} routed cells over sockets; "
              f"anchor#2 from :{satisfied[region(2)][-4:]}")
        assert into.has_region(region(2))

        # Kill the strong provider; a fresh pull self-heals from A's verified replica.
        sB.shutdown()
        into2 = MeshNode("query-node-2")
        satisfied2 = pull_regions_http(directory, [region(2)], into2, pub)
        assert satisfied2[region(2)] == "http://127.0.0.1:9721", satisfied2
        assert into2.has_region(region(2))
        print(f"[4] :9722 down -> anchor#2 self-healed from :{satisfied2[region(2)][-4:]} over HTTP")
    finally:
        sA.shutdown()
        sC.shutdown()
        try:
            sB.shutdown()
        except Exception:
            pass

    print("\nOK - gossip discovery + anchor-routed multi-peer pull + failover, over the wire.")


if __name__ == "__main__":
    main()
