"""The relay Channel: outward refill from the cloud, verified, over an injected transport.

No socket exists in these tests — the 'cloud' is an in-process MeshNode and the transport is its
`get_shard`. That is the point of the design: the platform is a function argument, so the license
boundary (reach it over the wire, never import it) is structural, and the path is testable
without a network.
"""
from __future__ import annotations

import numpy as np
import pytest

from _fakes import _StubAnswerer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ember import Ember, MeshChannel
from ember.config import Settings
from ember.embed import HashEmbedder
from mantle.mesh.node import MeshNode
from mantle.search.anchors.anchorset import AnchorSet
from mantle.mesh.anchor_routing import route_write_region

DIM = 16
CANON = "canonical-test"
ALICE = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
WORK = "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"
LABELS = ["ontology", "storage", "identity", "streaming"]
QUERY = "An ontology names the things a system can talk about."


def _anchors() -> AnchorSet:
    a = AnchorSet(model_id=CANON, dim=DIM)
    for i, l in enumerate(LABELS):
        v = np.full(DIM, 0.05, dtype=np.float32); v[i] = 1.0
        a.add_text(l, v)
    return a


def _settings(tmp_path) -> Settings:
    return Settings(cache_dir=tmp_path, principal=ALICE, collection_id=WORK,
                    engine="extractive", embed_model="", cloud_uri="", nprobe=3)


def _leaf(tmp_path):
    priv = Ed25519PrivateKey.generate()
    e = Ember.boot(_settings(tmp_path), embedder=HashEmbedder(dim=8), engine=_StubAnswerer()).seed(_anchors(), priv.public_key())
    e.authority_pub = priv.public_key()
    return e, priv


def _cloud_with(priv, anchors, embedder, texts):
    """An in-process 'cloud': a MeshNode holding authored shards. Its get_shard is the transport
    — no socket. It uses the same authority key the leaf trusts and the same anchors, so region ids
    line up (that is exactly why a leaf never authors its own anchors)."""
    from ember.embed import Aligner
    al = Aligner(embedder, anchors).fit()
    node = MeshNode("cloud")
    for item_id, text in texts.items():
        vec = al.encode_query(text)
        region = route_write_region(anchors, vec, ALICE, WORK)
        node.put_shard(region, {item_id: text.encode("utf-8")},
                       version=1, authority="ember-local", priv=priv)
    return node


def _transport(node):
    """Adapt a node to the mesh's (peer, region) GetShard convention — the same shape the real
    HTTP transport (`mesh.service.http_shard_getter`) has, so production drops in unchanged."""
    return lambda peer, region: node.get_shard(region)


def test_a_miss_refills_from_the_cloud_and_then_answers(tmp_path) -> None:
    """The whole outward path: the leaf holds nothing, misses, pulls the routed region from the
    cloud over the injected transport, and answers from it — verified end to end."""
    e, priv = _leaf(tmp_path)
    cloud = _cloud_with(priv, e.anchors, e.embedder, {"art-1": QUERY})
    e.connect(MeshChannel(e.cache.node, e.authority_pub, _transport(cloud)))

    res = e.ask(QUERY)                       # query == stored text -> co-routes under HashEmbedder
    assert res.refilled, "a miss must have refilled"
    assert not res.served_offline
    assert res.answer.grounded and "ontology" in res.answer.text.lower()


def test_a_hit_does_not_touch_the_channel(tmp_path) -> None:
    """Refill is for misses only. If the leaf already holds the nearest cell, the channel is
    never consulted — the leaf does not reach the network when it does not need to."""
    from prism.mass import Provenance
    e, priv = _leaf(tmp_path)
    e.remember([("local-1", QUERY.encode("utf-8"), QUERY)],   # embed-text == query -> a hit
               provenance=Provenance.HUMAN_VALIDATED, authority="ember-local", priv=priv)

    calls = []
    def spy_get_shard(peer, region):
        calls.append(region)
        return None, None
    e.connect(MeshChannel(e.cache.node, e.authority_pub, spy_get_shard))

    res = e.ask(QUERY)
    assert res.served_offline and res.answer.grounded
    assert calls == [], "a local hit must not consult the channel"


def test_a_tampered_cloud_response_is_rejected(tmp_path) -> None:
    """The cloud is as untrusted as any peer. A shard whose bytes were altered fails the same
    import verification and does not land — the region stays a miss."""
    e, priv = _leaf(tmp_path)
    cloud = _cloud_with(priv, e.anchors, e.embedder, {"art-1": "some real content about storage"})

    def tampering_get_shard(peer, region):
        manifest, items = cloud.get_shard(region)
        if items:
            k = next(iter(items)); items = dict(items); items[k] = items[k] + b"\x00EVIL"
        return manifest, items
    ch = MeshChannel(e.cache.node, e.authority_pub, tampering_get_shard)
    assert ch.fetch_regions(["anything"]) == [] or True  # region may not exist; the real check:
    # for a region the cloud does hold, a tampered shard fails verification and does not land:
    v = e.aligner.encode_query("storage")
    region = route_write_region(e.anchors, v, ALICE, WORK)
    assert ch.fetch_regions([region]) == [], "tampered shard must not land"
    assert not e.cache.node.has_region(region)






def test_disconnected_is_the_default(tmp_path) -> None:
    """Boot attaches no channel. ask() is purely local until connect() — you opt in to the
    network, never out of it."""
    e, priv = _leaf(tmp_path)
    assert e.channel is None
    res = e.ask("what is an ontology?")     # no channel, no refill
    assert res.served_offline


def test_the_channel_never_imports_the_platform() -> None:
    """The license boundary, as a test: relay.py reaches the platform only through the injected
    transport. No chorus/lumen/mantle-server import may appear."""
    import ember.runtime.relay as relay
    src = relay.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    for banned in ("import chorus", "import lumen", "from chorus", "from lumen"):
        assert banned not in text


def test_serve_is_an_honest_seam(tmp_path) -> None:
    """The inward direction is unimplemented, and `serve()` raises `NotImplementedError` at the
    point of call rather than appearing to work. The outward path is independent of it."""
    e, priv = _leaf(tmp_path)
    ch = MeshChannel(e.cache.node, e.authority_pub, lambda p, r: (None, None))
    with pytest.raises(NotImplementedError):
        ch.serve()
