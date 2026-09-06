"""HTTP round-trip test — a node stood up on a real socket, a peer syncs over the wire.

    python -m mesh.test_http
"""
# This file lives under `tests/`, not inside `src/ember/mesh/` — a test file inside the package
# would ship in the wheel. Imports below name `mantle.mesh` explicitly rather than relatively:
# `from . import merkle` here would resolve to `tests.merkle`, not `mantle.mesh.merkle`.
from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mantle.mesh.node import MeshNode
from mantle.mesh.service import pull_from, serve

PORT = 9799


def main() -> None:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    seed = MeshNode("seed")
    seed.put_shard("geo-01",
                   {"a": b"ENC:paris", "b": b"ENC:tokyo", "c": b"ENC:cairo"},
                   version=1, authority="geo", priv=priv)
    httpd = serve(seed, PORT, host="127.0.0.1")
    try:
        peer = MeshNode("peer")
        synced = pull_from(f"http://127.0.0.1:{PORT}", peer, pub)
        assert synced == ["geo-01"], synced
        assert peer.get_shard("geo-01")[0].content_root == seed.get_shard("geo-01")[0].content_root
        print(f"[1] peer synced {synced} over HTTP; content_root verified")

        # freshness over the wire
        seed.put_shard("geo-01",
                       {"a": b"ENC:paris", "b": b"ENC:tokyo", "c": b"ENC:cairo", "d": b"ENC:lima"},
                       version=2, authority="geo", priv=priv)
        synced = pull_from(f"http://127.0.0.1:{PORT}", peer, pub)
        assert synced == ["geo-01"] and peer.version_of("geo-01") == 2
        print(f"[2] freshness: peer re-synced to v{peer.version_of('geo-01')} "
              f"(density={peer.get_shard('geo-01')[0].density}) over HTTP")

        # a peer trying to verify with the wrong key is refused
        wrong = MeshNode("wrong")
        try:
            pull_from(f"http://127.0.0.1:{PORT}", wrong, Ed25519PrivateKey.generate().public_key())
            print("[3] FAIL: wrong-key sync accepted"); raise SystemExit(1)
        except Exception as e:
            print(f"[3] wrong-authority sync rejected over HTTP: {type(e).__name__}")

        print("\nOK - HTTP mesh sync verified (stand up a node, shards sync over the wire).")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
