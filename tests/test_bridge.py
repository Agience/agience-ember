"""Build mesh shards from a live Mantle node's real artifacts, then sync + verify.

    python -m mesh.test_bridge     # on a node with mantle:8082 + origin:8080 reachable

Logs in to Origin, packs the operator's visible artifacts into signed shards, syncs them
to a fresh peer, verifies content_root, and confirms a tampered real shard is rejected.
"""
# A test file inside the package ships in the wheel, so this module lives under `tests/` rather
# than inside `mantle.mesh`. Its imports name `mantle.mesh` explicitly instead of a package-relative
# import, since a relative import here would resolve against `tests`, not `mantle.mesh`.
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mantle.mesh.mantle_bridge import build_node_from_mantle
from mantle.mesh.node import MeshNode, ShardVerifyError

ORIGIN = os.getenv("ORIGIN_URL", "http://127.0.0.1:8080")
MANTLE = os.getenv("MANTLE_URL", "http://127.0.0.1:8082")
EMAIL = os.getenv("OP_EMAIL", "author@example.com")
PW = os.getenv("OP_PW", "genesis-dev-2026")


def _login() -> str:
    req = Request(ORIGIN + "/auth/password/login",
                  data=json.dumps({"identifier": EMAIL, "password": PW}).encode(),
                  headers={"content-type": "application/json"})
    with urlopen(req, timeout=20) as r:
        return json.load(r)["access_token"]


def main() -> None:
    token = _login()
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    src, arts = build_node_from_mantle("genesis", MANTLE, token, priv)
    print(f"[1] built {len(src.manifests())} shard(s) from {len(arts)} live artifacts:")
    for m in sorted(src.manifests(), key=lambda x: x.region_id):
        print(f"      region '{m.region_id}': {m.density} items, root {m.content_root[:12]}...")

    peer = MeshNode("peer")
    synced = peer.sync_from(src, pub)
    all_ok = all(peer.get_shard(r)[0].content_root == src.get_shard(r)[0].content_root for r in synced)
    print(f"[2] peer synced {sorted(synced)}; content_roots verified: {all_ok}")
    assert all_ok

    if synced:
        region = sorted(synced)[0]
        manifest, items = src.get_shard(region)
        ik = next(iter(items))
        bad = dict(items); bad[ik] = bad[ik] + b"TAMPER"
        try:
            MeshNode("evil").import_shard(manifest, bad, pub)
            print("[3] TAMPER NOT DETECTED - BUG"); raise SystemExit(1)
        except ShardVerifyError:
            print(f"[3] tampered real shard '{region}' REJECTED")

    print("\nOK - mesh now syncs REAL Mantle artifacts (content-addressed, signed, verified).")


if __name__ == "__main__":
    main()
