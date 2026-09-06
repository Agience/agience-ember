"""Artifacts are immutable: an edit is a new version under a stable root_id, and the write decides
nothing about which version answers.

Every revision commits and stands. Which one answers is resolved at read time by the reader's own
measurement, injected as a seam because mantle does not import the instrument. With no resolver, every
revision answers: a store that has not measured which version is right has no basis for picking one.

Two properties settle the shape of this.

A count of attesting origins measures agreement — a reading of existence — rather than validity, and
the comparison grounds out in any case: `prism.resolution.separated([incoming, current])` is False
for every pair of counts, 10-vs-2 included, because at n=2 the computed null is exactly 1.0000 and
nothing can exceed it. Head decided from two counts would be a confident answer computed from
nothing.

And a rule that keeps a low-attestation revision from standing beside a well-attested one protects
the error, because a correction is precisely the claim that begins attested by one origin.
"""
from __future__ import annotations

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ember import Ember
from ember.config import Settings
from ember.embed import HashEmbedder
from mantle.search.anchors.anchorset import AnchorSet
from mantle.mesh.manifest import item_hash
from prism.mass import Provenance

from _fakes import _StubAnswerer

DIM = 16
CANON = "canonical-test"
ALICE = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
WORK = "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"
TOPIC = "An ontology names the things a system can talk about."


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
    e = Ember.boot(s, embedder=HashEmbedder(dim=8), engine=_StubAnswerer()).seed(_anchors(), priv.public_key())
    e.authority_pub = priv.public_key()
    return e, priv


def _standing(e, root="doc-1"):
    """Every revision the cache currently returns for `root`, by id."""
    hits = e.cache.search(e.aligner.encode_query(TOPIC), k=20)
    return [it.id for it, _ in hits if e.cache._find(it.id).root_id == root]


def _attest(e, item_id, *peers):
    from prism.attestation import Attestation
    held = e.cache._find(item_id)
    for peer in peers:
        e.cache.ledger.record(Attestation(item_id=item_id,
                                          content_hash=item_hash(held.content),
                                          authority=peer, origin=peer))


# ── lineage ──────────────────────────────────────────────────────────────────────────────────
def test_a_first_version_has_id_equal_to_root_id(tmp_path) -> None:
    """Mantle's rule: the first version's id is the root_id, and a leaf agrees, so version lineage
    means the same thing on both sides.

    Minting a separate root for the first version would hang every later revision off a root nothing
    else knows, and `_versions_of` would report one version forever.
    """
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"first draft about ontology.", TOPIC)],
               provenance=Provenance.HYPOTHESIS, authority="me", priv=priv)
    assert e.cache._find("doc-1").root_id == "doc-1"


def test_revising_an_unknown_root_raises(tmp_path) -> None:
    """revise() edits an existing artifact, so an unknown root_id raises rather than creating one.

    Treating an unknown root as a create would let a typo in a root id mint a parallel lineage, and
    nothing downstream could tell the two apart.
    """
    e, priv = _leaf(tmp_path)
    with pytest.raises(KeyError):
        e.revise("never-created", b"x", TOPIC,
                 provenance=Provenance.HUMAN_VALIDATED, authority="me", priv=priv)


# ── the write decides nothing ────────────────────────────────────────────────────────────────
def test_revise_returns_WHAT_LANDED_not_a_verdict(tmp_path) -> None:
    """`revise` returns what landed — root, id and version count — not a verdict. A `Revision`-shaped
    return would offer callers a decision nobody makes, and their branch would always take one
    arm."""
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"first draft about ontology.", TOPIC)],
               provenance=Provenance.HYPOTHESIS, authority="me", priv=priv)
    landed = e.revise("doc-1", TOPIC.encode(), TOPIC,
                      provenance=Provenance.HUMAN_VALIDATED, authority="me", priv=priv)
    assert landed.root_id == "doc-1"
    assert landed.id != "doc-1"                 # a distinct, content-derived version id
    assert landed.versions == 2
    assert not hasattr(landed, "value"), "a Revision-shaped verdict is back"


def test_NOTHING_is_destroyed_and_NOTHING_is_hidden(tmp_path) -> None:
    """Both versions are retained and both answer.

    This passes by grounding out, which is worth stating: `Ember.boot` wires a resolver, so what runs
    here is a resolver measuring a frame of two items, finding fewer than three neighbours to
    standardise against, and propagating nothing. The safe direction and the unwired direction agree,
    which is why nothing is hidden either way.

    It rules out a write-time rule that surfaces one version and suppresses the other, and equally a
    resolver that names a winner off a frame too small to measure.
    """
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"v1 content about ontology", TOPIC)],
               provenance=Provenance.HYPOTHESIS, authority="me", priv=priv)
    first = "doc-1"
    landed = e.revise("doc-1", b"v2 content, another ontology answer", TOPIC,
                      provenance=Provenance.SPAN_CITED, authority="me", priv=priv)

    assert e.cache._find(first) is not None          # retained
    standing = _standing(e)
    assert set(standing) == {first, landed.id}, standing


def test_a_SINGLY_attested_correction_still_answers_beside_a_well_attested_error(tmp_path) -> None:
    """A correction attested by one origin stands beside a claim attested by three, and a reader with
    a measurement decides between them.

    This is the case an `agreeing >=` comparison inverts: the well-attested claim would win because
    it is well-attested, and the correction would be invisible until peers came round.
    """
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"A widely repeated claim about ontology.", TOPIC)],
               provenance=Provenance.HUMAN_VALIDATED, authority="me", priv=priv)
    _attest(e, "doc-1", "peer-a", "peer-b")
    assert e.cache.agreement("doc-1").agreeing == 3          # control: the prior is well attested

    landed = e.revise("doc-1", b"A correction about ontology, attested by one.", TOPIC,
                      provenance=Provenance.ASSERTION, authority="me", priv=priv)
    assert e.cache.agreement(landed.id).agreeing == 1        # control: the correction is not

    standing = _standing(e)
    assert landed.id in standing, "the correction was hidden by a headcount"
    assert "doc-1" in standing, "the prior version was destroyed rather than retained"


def test_the_resolution_instrument_GROUNDS_OUT_on_two_counts() -> None:
    """The measurement the file rests on, asserted rather than described: two attestation counts are
    never separable. The control shows the instrument is not simply always-False, which would make
    the claim vacuous.
    """
    from prism import resolution as R
    for current, incoming in ((3, 1), (3, 4), (10, 2), (2, 10), (1, 1)):
        assert R.separated([float(incoming), float(current)]) is False, (current, incoming)
    assert R._null_separability(2) == 1.0, "at n=2 nothing can exceed the null"
    assert R.separated([9.0, 8.7, 8.5, 0.4, 0.2, 0.1]) is True, "the instrument cannot fire at all"


# ── read-time resolution ─────────────────────────────────────────────────────────────────────
def test_an_INJECTED_resolver_decides_which_revision_answers(tmp_path) -> None:
    """The seam: a reader that has a measurement supplies one, and mantle imports no instrument.

    An ignored resolver would return everything and look identical to the no-resolver default, so the
    resolver here picks one specific version and the assertion names it.
    """
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"v1 content about ontology", TOPIC)],
               provenance=Provenance.HYPOTHESIS, authority="me", priv=priv)
    landed = e.revise("doc-1", b"v2 content about ontology", TOPIC,
                      provenance=Provenance.SPAN_CITED, authority="me", priv=priv)

    seen = {}

    def only_the_newest(root, revisions, reads, frame):
        seen[root] = len(revisions)
        # The frame is the reader's recall set: the pool this query surfaced, with vectors. Asserted
        # here because a resolver handed no frame has nothing to measure against, and would fall back
        # to letting everything stand.
        assert frame, "the cache passed no frame to the resolver"
        return [landed.id]

    e.cache._resolve = only_the_newest
    assert _standing(e) == [landed.id]
    assert seen.get("doc-1") == 2, "the resolver was not offered every standing revision"


def test_a_resolver_that_GROUNDS_OUT_is_respected(tmp_path) -> None:
    """A reader whose measurement cannot separate the revisions says so, and the store does not pick
    something on its behalf [[one-resolution-not-thresholds]].

    The empty return grounds out: no revision is ruled against, the reader's measurement simply had
    nowhere to stand, and nothing propagates. Reading it as "no opinion" and falling back to all
    revisions would make a reader that measured nothing indistinguishable from one never asked.
    """
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"v1 content about ontology", TOPIC)],
               provenance=Provenance.HYPOTHESIS, authority="me", priv=priv)
    e.revise("doc-1", b"v2 content about ontology", TOPIC,
             provenance=Provenance.SPAN_CITED, authority="me", priv=priv)
    e.cache._resolve = lambda root, revisions, reads, frame: []
    assert _standing(e) == []


def test_a_resolver_cannot_conjure_an_id_it_was_not_offered(tmp_path) -> None:
    """The resolver's return is intersected with the revisions it was offered. Taken verbatim, it
    would let a buggy or hostile reader surface an artifact this cache does not hold."""
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"v1 content about ontology", TOPIC)],
               provenance=Provenance.HYPOTHESIS, authority="me", priv=priv)
    e.cache._resolve = lambda root, revisions, reads, frame: ["not-a-real-id"]
    assert _standing(e) == []


# ── restart ──────────────────────────────────────────────────────────────────────────────────
def test_version_lineage_survives_a_restart_and_NO_head_is_persisted(tmp_path) -> None:
    """The sidecar carries lineage — which versions share a root — and no head. Persisting a head
    would restore a write-time decision on the next boot, on a store that otherwise looks upgraded.
    """
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"first draft about ontology.", TOPIC)],
               provenance=Provenance.HYPOTHESIS, authority="me", priv=priv)
    landed = e.revise("doc-1", b"A second answer: an ontology names what a system talks about.",
                      TOPIC, provenance=Provenance.HUMAN_VALIDATED, authority="me", priv=priv)

    assert "heads" not in e.cache.export_views()

    again = Ember.boot(e.settings, embedder=HashEmbedder(dim=8), engine=_StubAnswerer(),
                       authority_pub=priv.public_key())
    ids = {it.id for it, _ in again.cache.search(again.aligner.encode_query(TOPIC), k=20)}
    assert {"doc-1", landed.id} <= ids, "a revision was lost or hidden across a restart"
    assert again.cache._find(landed.id).root_id == "doc-1", "lineage did not survive"


def test_an_OLD_sidecars_heads_are_discarded_and_the_drop_is_OBSERVABLE(tmp_path) -> None:
    """A sidecar carrying `heads` has it read, discarded, and counted. Honouring it would reinstate a
    write-time decision; dropping it without a count would be a migration nobody can observe.
    """
    e, priv = _leaf(tmp_path)
    e.remember([("doc-1", b"v1 content about ontology", TOPIC)],
               provenance=Provenance.HYPOTHESIS, authority="me", priv=priv)
    landed = e.revise("doc-1", b"v2 content about ontology", TOPIC,
                      provenance=Provenance.SPAN_CITED, authority="me", priv=priv)

    views = e.cache.export_views()
    views["heads"] = {"doc-1": "doc-1"}          # a sidecar naming v1 head and v2 a proposal

    again = Ember.boot(e.settings, embedder=HashEmbedder(dim=8), engine=_StubAnswerer(),
                       authority_pub=priv.public_key())
    again.cache.hydrate(views)
    assert getattr(again.cache, "_discarded_heads", 0) == 1, "the drop was not published"
    ids = {it.id for it, _ in again.cache.search(again.aligner.encode_query(TOPIC), k=20)}
    assert landed.id in ids, "the old sidecar's head decision was honoured"
