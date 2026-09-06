"""Anchor-region keying — the mesh partition IS the search partition.

    python tests/test_anchor_routing.py     # from the ember repo root

Builds a small AnchorSet, synthesizes one encrypted cell blob per anchor, packs them
into shards keyed by anchor cell (blind), then shows that a query routed by the same
geometry the index uses pulls exactly its ``nprobe`` cells — its working set — and no
others. Verifies content_roots, agreement between write- and query-side routing, and
that a tampered ciphertext cell does not import. All blind: cells are never decrypted.
"""
# The mesh modules are named absolutely below. This file lives in `tests/`, outside the package, so
# a relative import would resolve against `tests` rather than against `mantle.mesh`.
from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mantle.search.anchors.anchorset import AnchorSet
from mantle.mesh.anchor_routing import (
    build_node_from_cells,
    cell_region,
    parse_cell_key,
    route_query_regions,
    route_write_region,
)
from mantle.mesh.node import MeshNode, ShardVerifyError

DIM = 8
PRINCIPAL = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
COLLECTION = "default"
MODEL = "all-MiniLM-L6-v2"
PREFIX = "mantle-cells"


def _fake_ciphertext(cluster: str) -> bytes:
    """A deterministic stand-in for an encrypted cell blob (opaque bytes)."""
    return b"ENCv1:" + hashlib.sha256(("cell/" + cluster).encode()).digest() * 4


class _FakeS3:
    """The 3 boto3 calls build_node_from_cells uses — over an in-memory blob map."""

    def __init__(self, blobs: Dict[str, bytes]) -> None:
        self._blobs = blobs

    def get_paginator(self, _op):
        store = self._blobs

        class _P:
            def paginate(self, Bucket, Prefix):
                contents = [{"Key": k} for k in sorted(store) if k.startswith(Prefix)]
                yield {"Contents": contents}

        return _P()

    def get_object(self, Bucket, Key):
        class _B:
            def __init__(self, b):
                self._b = b

            def read(self):
                return self._b

        return {"Body": _B(self._blobs[Key])}


def _anchorset() -> AnchorSet:
    """K distinct, well-separated anchors in R^DIM (deterministic)."""
    a = AnchorSet(model_id=MODEL, dim=DIM)
    for i in range(6):
        v = np.full(DIM, 0.05, dtype=np.float32)
        v[i] = 1.0
        a.add_text(f"anchor-{i}", v)
    return a


def main() -> None:
    aset = _anchorset()
    anchors = aset.anchors
    print(f"[0] AnchorSet: {len(aset)} anchors in R^{DIM} (model {MODEL})")

    # One encrypted cell per anchor, laid out exactly like the S3 cell store.
    blobs: Dict[str, bytes] = {}
    for a in anchors:
        key = f"{PREFIX}/{PRINCIPAL}/{COLLECTION}/{a.anchor_id}{'.cell'}"
        blobs[key] = _fake_ciphertext(a.anchor_id)
    # a couple of non-cell objects under the prefix — must be skipped, blind:
    blobs[f"{PREFIX}/{PRINCIPAL}/{COLLECTION}/stats.enc"] = b"not-a-cell"
    s3 = _FakeS3(blobs)

    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    src, n = build_node_from_cells("genesis", s3, "agience-content", PREFIX, priv)
    regions_all = sorted(r for r, _ in src.summary()["regions"].items())
    print(f"[1] packed {n} ENCRYPTED cells into {len(regions_all)} shard(s), one per anchor cell")
    assert n == len(anchors), f"expected {len(anchors)} cells, packed {n}"  # stats.enc skipped
    assert len(regions_all) == len(anchors)

    # A query near anchor #2 — route it with the same geometry the index uses.
    target = anchors[2]
    q = target.embedding.astype(np.float32).copy()
    q[5] += 0.02  # small perturbation; nearest anchor is still #2
    nprobe = 3
    routed = route_query_regions(aset, q, PRINCIPAL, COLLECTION, nprobe=nprobe)
    print(f"[2] route_query (nprobe={nprobe}) -> {len(routed)} region(s); nearest first:")
    for r in routed:
        print(f"      {r}")
    assert len(routed) == nprobe
    assert routed[0] == cell_region(PRINCIPAL, COLLECTION, target.anchor_id), \
        "nearest routed region must be the query's own anchor cell"
    # write-side routing agrees with query-side nearest (a match indexes where we look):
    assert route_write_region(aset, q, PRINCIPAL, COLLECTION) == routed[0]
    print("[3] write-side route == query-side nearest (writes land where queries look)")

    # Anchor-routed pull: a node fetches only its working set — the routed cells.
    peer = MeshNode("query-node")
    synced = sorted(peer.sync_from(src, pub, regions=routed))  # blind + verified, selective
    assert synced == sorted(routed), f"pulled {synced}, expected {sorted(routed)}"
    assert len(synced) < len(regions_all), "must pull the working set, not the whole corpus"
    roots_ok = all(peer.get_shard(r)[0].content_root == src.get_shard(r)[0].content_root
                   for r in synced)
    assert roots_ok
    print(f"[4] pulled {len(synced)}/{len(regions_all)} cells (working set only); "
          f"content_roots verified: {roots_ok}")

    # The unrouted cells were never fetched — the whole point of anchor keying.
    not_pulled = [r for r in regions_all if r not in synced]
    assert all(not peer.has_region(r) for r in not_pulled)
    print(f"[5] {len(not_pulled)} unrelated cell(s) correctly NOT pulled")

    # A tampered ciphertext cell does not import (and is still never decrypted).
    region = routed[0]
    manifest, items = src.get_shard(region)
    ik = next(iter(items))
    bad = dict(items)
    bad[ik] = bad[ik] + b"\x00TAMPER"
    try:
        MeshNode("evil").import_shard(manifest, bad, pub)
        print("[6] TAMPER NOT DETECTED - BUG")
        raise SystemExit(1)
    except ShardVerifyError:
        print(f"[6] tampered ciphertext cell in '{region[-20:]}' REJECTED")

    # parse_cell_key round-trips; non-cell keys return None.
    assert parse_cell_key(f"{PREFIX}/{PRINCIPAL}/{COLLECTION}/{target.anchor_id}.cell", PREFIX) \
        == (PRINCIPAL, COLLECTION, target.anchor_id)
    assert parse_cell_key(f"{PREFIX}/{PRINCIPAL}/{COLLECTION}/stats.enc", PREFIX) is None
    print("[7] parse_cell_key round-trips; non-cell keys skipped")

    print("\nOK - the mesh partition == the search partition: a query pulls only its "
          "anchor cells, blind + verified.")


if __name__ == "__main__":
    main()
