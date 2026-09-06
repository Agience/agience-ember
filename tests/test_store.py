"""Persistence: the cache survives a restart, and disk is treated as untrusted.

The point of a leaf is to be useful with the network unplugged. If shards die with the process,
that's true exactly once — the second boot has nothing and must reach the network to say
anything, which is the moment a leaf is supposed to earn its keep.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from _fakes import _StubAnswerer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ember import LocalCache, answer_query, build
from mantle.search.anchors.anchorset import AnchorSet
from mantle.mesh.node import MeshNode
from mantle.shard.store import ShardStore

DIM = 16
MODEL = "test-embed"
PRINCIPAL = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
COLLECTION = "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"


def _anchorset() -> AnchorSet:
    a = AnchorSet(model_id=MODEL, dim=DIM)
    for i in range(4):
        v = np.full(DIM, 0.05, dtype=np.float32)
        v[i] = 1.0
        a.add_text(f"anchor-{i}", v)
    return a


def _vec(axis: int, jitter: float = 0.0) -> np.ndarray:
    v = np.full(DIM, 0.05, dtype=np.float32)
    v[axis] = 1.0
    if jitter:
        v[(axis + 5) % DIM] += jitter
    return v


@pytest.fixture()
def seeded(tmp_path):
    priv = Ed25519PrivateKey.generate()
    cache = LocalCache(_anchorset(), PRINCIPAL, COLLECTION, nprobe=3)
    cache.put(
        [("art-1", b"Ontologies name the things a system can talk about.", _vec(0)),
         ("art-2", b"An anchor set is the shared coordinate system.", _vec(0, 0.02))],
        version=1, authority="ember-local", priv=priv,
    )
    store = ShardStore(tmp_path)
    store.save_node(cache.node)
    return store, priv.public_key(), cache


def test_shards_survive_a_restart(seeded) -> None:
    """The whole point: boot #2 answers from disk with no network in the process."""
    store, pub, _ = seeded
    fresh = MeshNode("ember-after-restart")
    loaded, rejected = store.load_node(fresh, pub)
    assert loaded == 1 and rejected == 0
    assert fresh.summary()["regions"], "a restarted leaf must still hold its shards"


def test_a_tampered_cache_file_is_rejected(seeded, tmp_path) -> None:
    """Disk is just another untrusted server. The signature is what makes the cache
    trustworthy — not the fact that we wrote it ourselves. Editing bytes on disk must fail the
    same check that catches a lying peer."""
    store, pub, _ = seeded
    f = next((tmp_path / "shards").glob("*.json"))
    d = json.loads(f.read_text(encoding="utf-8"))
    k = next(iter(d["items"]))
    d["items"][k] = base64.b64encode(b"TAMPERED").decode("ascii")
    f.write_text(json.dumps(d), encoding="utf-8")

    fresh = MeshNode("victim")
    loaded, rejected = store.load_node(fresh, pub)
    assert loaded == 0 and rejected == 1, "tampered cache must not load"
    assert not fresh.summary()["regions"]


def test_a_corrupt_file_degrades_to_a_miss_not_a_crash(seeded, tmp_path) -> None:
    """Half-written or garbage files happen. One bad file must not stop a leaf from booting —
    the region just becomes a miss and refills when a channel exists."""
    store, pub, _ = seeded
    next((tmp_path / "shards").glob("*.json")).write_text("{not json", encoding="utf-8")
    fresh = MeshNode("survivor")
    loaded, rejected = store.load_node(fresh, pub)          # must not raise
    assert loaded == 0 and rejected == 1


def test_region_ids_are_keys_not_paths(tmp_path) -> None:
    """A region id contains '/' and comes from the wire. Treating it as a path invites
    traversal; it is an opaque key."""
    store = ShardStore(tmp_path)
    priv = Ed25519PrivateKey.generate()
    node = MeshNode("n")
    nasty = "../../etc/passwd/../../evil"
    node.put_shard(nasty, {"i": b"x"}, version=1, authority="a", priv=priv)
    m, items = node.get_shard(nasty)
    store.put(m, items)
    # Everything lands flat under shards/, nothing escapes the cache dir.
    files = list((tmp_path / "shards").glob("*.json"))
    assert len(files) == 1
    assert files[0].parent == tmp_path / "shards"
    got = store.get(nasty)
    assert got is not None and got[0].region_id == nasty


def test_vectors_are_not_persisted(seeded, tmp_path) -> None:
    """Vectors are a derived retrieval view, cheap to recompute. Writing them would duplicate
    plaintext-adjacent data beside a cache that is otherwise blind. Storage holds what we were
    given; derivations get rebuilt."""
    _store, _pub, _cache = seeded
    blob = " ".join(p.read_text(encoding="utf-8") for p in (tmp_path / "shards").glob("*.json"))
    assert "vector" not in blob and "embedding" not in blob


def test_items_persist_byte_for_byte(seeded, tmp_path) -> None:
    """Ember holds ciphertext it cannot read; persistence must not assume otherwise."""
    store, pub, cache = seeded
    region = next(iter(cache.node.summary()["regions"]))
    _m, original = cache.node.get_shard(region)
    _m2, restored = store.get(region)
    assert restored == original


def test_artifacts_round_trip_and_are_findable(tmp_path) -> None:
    """One shape stores every artifact — a cross-walk today, anything else tomorrow."""
    store = ShardStore(tmp_path)
    art = {"id": "cw-1", "content_type": "application/vnd.agience.crosswalk+json",
           "context": {"error_bound": 0.1}}
    store.put_artifact(art)
    assert store.get_artifact("cw-1") == art
    assert store.find_artifact("application/vnd.agience.crosswalk+json") == art
    assert store.find_artifact("application/vnd.agience.nothing+json") is None


def test_anchors_round_trip(tmp_path) -> None:
    """The AnchorSet is cached like anything else — it's what alignment is fitted against."""
    store = ShardStore(tmp_path)
    store.save_anchors(_anchorset())
    back = store.load_anchors()
    assert back is not None and len(back) == 4 and back.model_id == MODEL


def test_missing_cache_is_empty_not_an_error(tmp_path) -> None:
    """A first boot has no cache. That is the normal case, not a failure."""
    store = ShardStore(tmp_path / "does-not-exist")
    assert list(store.regions()) == []
    assert store.load_anchors() is None
    assert store.get("anything") is None
    assert store.load_node(MeshNode("n"), Ed25519PrivateKey.generate().public_key()) == (0, 0)


def test_restarted_leaf_answers_offline(tmp_path) -> None:
    """End to end: persist, drop everything, reload, answer — no network in the process."""
    priv = Ed25519PrivateKey.generate()
    first = LocalCache(_anchorset(), PRINCIPAL, COLLECTION, nprobe=3)
    first.put([("art-1", b"Ontologies name the things a system can talk about.", _vec(0))],
              version=1, authority="ember-local", priv=priv)
    store = ShardStore(tmp_path)
    store.save_node(first.node)

    # A brand new process: nothing in memory.
    second = LocalCache(_anchorset(), PRINCIPAL, COLLECTION, nprobe=3)
    loaded, _ = store.load_node(second.node, priv.public_key())
    assert loaded == 1
    for region in second.node.summary()["regions"]:
        _m, items = second.node.get_shard(region)
        second.adopt(region, items, {"art-1": _vec(0)})   # rebuild the derived view

    res = answer_query(second, _StubAnswerer(), "what is an ontology?", _vec(0), refill=None)
    assert res.routing.hit and res.served_offline and res.answer.grounded
    assert "Ontologies" in res.answer.text


def test_dual_read_finds_legacy_cells_after_provisioning() -> None:
    """The migration is incremental. A node that holds cleartext cells and then provisions a
    blinding secret still finds those cells: new writes go blinded, and the existing cells stay
    reachable. That is what lets `principal.secret` roll out without re-keying the corpus in one
    shot."""
    import tempfile
    from mantle.shard import region
    priv = Ed25519PrivateKey.generate()
    items = [("art-%d" % i, b"c%d" % i, _vec(i % 4)) for i in range(12)]

    legacy = LocalCache(_anchorset(), PRINCIPAL, COLLECTION, nprobe=3, secret=None)
    legacy.put(items, version=1, authority="a", priv=priv)
    legacy_regions = set(legacy.node.summary()["regions"])
    assert any(PRINCIPAL in r for r in legacy_regions), "legacy cells should carry the principal"

    secret = region.principal_secret(tempfile.mkdtemp(), create=True)
    provisioned = LocalCache(_anchorset(), PRINCIPAL, COLLECTION, nprobe=3, secret=secret)
    routed = provisioned.route(_vec(0)).regions
    assert legacy_regions & set(routed), "dual-read did not find pre-existing legacy cells"
    assert any(PRINCIPAL not in r for r in routed), "no blinded id was offered for new writes"


def test_disabling_legacy_read_closes_the_reader() -> None:
    """Once the corpus is fully migrated, `legacy_read=False` stops emitting cleartext ids at all."""
    from mantle.mesh.anchor_routing import route_query_regions
    import tempfile
    from mantle.shard import region
    secret = region.principal_secret(tempfile.mkdtemp(), create=True)
    aset = _anchorset()
    both = route_query_regions(aset, _vec(0), PRINCIPAL, COLLECTION, nprobe=3, secret=secret)
    only = route_query_regions(aset, _vec(0), PRINCIPAL, COLLECTION, nprobe=3, secret=secret,
                               legacy_read=False)
    assert any(PRINCIPAL in r for r in both), "dual-read should still emit legacy ids"
    assert not any(PRINCIPAL in r for r in only), "legacy_read=False must emit no cleartext ids"
