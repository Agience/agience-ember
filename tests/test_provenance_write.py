"""Provenance at write time — the proofreading layer.

Fidelity at copy time is a different mechanism from the selection that acts later (density, aging,
human validation), and both are needed: proofreading alone does not adapt, and selection alone
cannot keep up once the error rate rises. It is also what buys corpus size — a store grows only as
large as its gates permit.

The invariant: nothing acquires a rung it did not earn, and no caller gets a free default.
"""
from __future__ import annotations

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from prism.extraction import Extraction, describe, rung_for_extraction
from prism.attestation import SELF_ORIGIN
from prism.mass import Provenance, has_referent, provenance_of, stamp
from ember.runtime.boot import Ember
from ember.config import Settings
from ember.embed import HashEmbedder
from mantle.search.anchors.anchorset import AnchorSet

DIM = 16
CANON = "canonical-test"
ALICE = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
WORK = "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"


def _anchors() -> AnchorSet:
    a = AnchorSet(model_id=CANON, dim=DIM)
    for i, l in enumerate(["ontology", "storage", "identity", "streaming"]):
        v = np.full(DIM, 0.05, dtype=np.float32); v[i] = 1.0
        a.add_text(l, v)
    return a


def _leaf(tmp_path):
    priv = Ed25519PrivateKey.generate()
    s = Settings(cache_dir=tmp_path, principal=ALICE, collection_id=WORK,
                 engine="extractive", embed_model="", cloud_uri="", nprobe=3)
    e = Ember.boot(s, embedder=HashEmbedder(dim=8)).seed(_anchors(), priv.public_key())
    e.authority_pub = priv.public_key()
    return e, priv


# --------------------------------------------------------------------------- stamp
def test_stamp_copies_never_mutates() -> None:
    """Artifacts are content-addressed, so editing one in place changes what it is while every
    holder of its id still reads the old identity."""
    a = {"id": "x", "content_type": "text/plain", "content": "hi"}
    s = stamp(a, Provenance.SPAN_CITED)
    assert provenance_of(s) is Provenance.SPAN_CITED
    assert "context" not in a, "the original must be untouched"


def test_stamp_refuses_a_nonsense_rung() -> None:
    """The write and read sides are deliberately asymmetric. A write raises on an unrecognised
    rung, because it is recording a claim and a typo would otherwise land as UNKNOWN. A read
    resolves an unrecognised rung to UNKNOWN, so a typo under-credits an artifact rather than
    promoting it."""
    with pytest.raises(ValueError):
        stamp({"id": "x"}, "hooman_validated")
    # the read side, same typo, stays conservative rather than raising
    assert provenance_of({"context": {"provenance": "hooman_validated"}}) is Provenance.UNKNOWN


def test_promotion_is_recorded_not_silent() -> None:
    """The ladder is climbed by acquiring better provenance. A rung change with no trail is
    indistinguishable from tampering, so the prior rung stays in `provenance_history`."""
    a = stamp({"id": "x"}, Provenance.HYPOTHESIS)
    b = stamp(a, Provenance.SPAN_CITED)          # earned a citation
    assert provenance_of(b) is Provenance.SPAN_CITED
    assert b["context"]["provenance_history"] == ["hypothesis"]


def test_evidence_is_kept_so_the_decision_can_be_re_judged() -> None:
    e = Extraction(n_models=3, n_agreeing=3, distinct_families=3)
    art = stamp({"id": "x"}, rung_for_extraction(e), evidence=describe(e))
    ev = art["context"]["provenance_evidence"]
    assert ev["n_models"] == 3 and ev["distinct_families"] == 3
    assert provenance_of(art) is Provenance.HYPOTHESIS   # a panel earns hypothesis, never more


# --------------------------------------------------------------------------- the leaf
def test_remember_requires_a_rung(tmp_path) -> None:
    """The caller is the only party that knows how the content was obtained, so `remember` takes
    the rung as a required argument: a default would hand every artifact a rung it did not earn."""
    e, priv = _leaf(tmp_path)
    with pytest.raises(TypeError):
        e.remember([("a", b"x", "ontology")], authority="ember-local", priv=priv)  # no provenance


def test_a_leafs_own_notes_get_no_local_exemption(tmp_path) -> None:
    """A leaf's own content records the same channel as anything from the mesh. A note you typed
    is human_validated; a model's guess you saved is not — being local buys nothing."""
    e, priv = _leaf(tmp_path)
    e.remember([("mine", b"I checked this myself.", "ontology")],
               provenance=Provenance.HUMAN_VALIDATED, authority="ember-local", priv=priv)
    e.remember([("guess", b"A model reckons this.", "storage")],
               provenance=Provenance.ASSERTION, authority="ember-local", priv=priv)

    got = {}
    for region in e.cache.node.summary()["regions"]:
        m, _items = e.cache.node.get_shard(region)
        for it in m.items:
            got[it["id"]] = it["channel"]
    assert got["mine"] == Provenance.HUMAN_VALIDATED.value
    assert got["guess"] == Provenance.ASSERTION.value
    assert has_referent(got["mine"])
    assert not has_referent(got["guess"]), "an unbacked claim stays unbacked even when it is yours"


def test_the_attestation_travels_with_the_shard_not_a_weight(tmp_path) -> None:
    """What travels with the shard is what this node observed: it originated these bytes, via this
    channel. A peer combines that with its own attestation and other peers', and computes
    agreement — something it could not have known alone.

    No consensus weight rides along. A weight derived from the channel label is a lookup the peer
    can perform for itself, so the field would carry no information."""
    e, priv = _leaf(tmp_path)
    e.remember([("cited", b"Extracted from a real source.", "ontology")],
               provenance=Provenance.SPAN_CITED, authority="ember-local", priv=priv)
    region = next(iter(e.cache.node.summary()["regions"]))
    m, _ = e.cache.node.get_shard(region)
    assert "consensus" not in m.items[0], "the constant is back on the wire"
    assert m.items[0]["origin"] == SELF_ORIGIN         # this node originated it
    assert m.items[0]["channel"] == Provenance.SPAN_CITED.value

    # ...and it reads back as one independent origin, resolved to the signing authority.
    r = e.cache.agreement("cited")
    assert r is not None and r.resolved and r.agreeing == 1
    assert r.by_hash[r.agreed_hash] == frozenset({"ember-local"})


def test_unlabelled_writes_land_at_unknown_not_at_zero(tmp_path) -> None:
    """`LocalCache.put` without a channel lands at UNKNOWN. Unlabelled is not fabricated, and
    conflating the two would describe the corpus wrongly.

    An unlabelled write is still attested. "We never recorded how this was obtained" says nothing
    about whether anyone stands behind the bytes, and this node does; channel and attestation are
    separate fields so the two stay distinguishable."""
    e, priv = _leaf(tmp_path)
    v = e.aligner.encode_query("ontology")
    e.cache.put([("bare", b"no rung given", v)], version=1, authority="a", priv=priv)
    region = next(iter(e.cache.node.summary()["regions"]))
    m, _ = e.cache.node.get_shard(region)
    assert m.items[0]["channel"] == Provenance.UNKNOWN.value
    assert not has_referent(m.items[0]["channel"])
    r = e.cache.agreement("bare")
    assert r is not None and r.agreeing == 1, "unlabelled is not unattested"
