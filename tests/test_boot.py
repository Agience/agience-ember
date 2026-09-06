"""Boot: the assembly, and what a leaf reports about its own state.

The distinction under test is uninitialised versus empty. An empty leaf answers "I don't hold
that", which is a measurement of what it holds. An uninitialised leaf cannot route at all, so it
has no reading to give and `NotInitialised` names what is missing. One value for both states would
report the second as the first.
"""
from __future__ import annotations

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ember.runtime.boot import Ember, NotInitialised
from ember.config import Settings
from mantle.search.anchors.crosswalk_artifact import CROSSWALK_CONTENT_TYPE
from ember.embed import HashEmbedder
from mantle.search.anchors.anchorset import AnchorSet
from prism.mass import Provenance

DIM = 16
CANON = "canonical-test"
ALICE = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
WORK = "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"
LABELS = ["ontology", "storage", "identity", "streaming"]


def _anchors() -> AnchorSet:
    a = AnchorSet(model_id=CANON, dim=DIM)
    for i, l in enumerate(LABELS):
        v = np.full(DIM, 0.05, dtype=np.float32)
        v[i] = 1.0
        a.add_text(l, v)
    return a


def _settings(tmp_path) -> Settings:
    return Settings(cache_dir=tmp_path, principal=ALICE, collection_id=WORK,
                    engine="extractive", embed_model="", cloud_uri="", nprobe=3)


def test_a_fresh_leaf_is_uninitialised_not_broken(tmp_path) -> None:
    """No first light yet: it cannot route, and `ask()` raises `NotInitialised` naming that, so the
    caller learns which step is outstanding."""
    e = Ember.boot(_settings(tmp_path), embedder=HashEmbedder(dim=8))
    assert not e.ready
    assert e.status()["anchors"] == 0
    with pytest.raises(NotInitialised, match="first light"):
        e.ask("anything")


def test_anchors_without_an_authority_key_is_still_not_ready(tmp_path) -> None:
    """Anchors give a leaf coordinates; the authority key is what lets it verify a shard. Without
    the key an unverified shard is a rumour, so the leaf holds nothing it can answer from."""
    s = _settings(tmp_path)
    e = Ember.boot(s, embedder=HashEmbedder(dim=8))
    e.store.save_anchors(_anchors())
    e2 = Ember.boot(s, embedder=HashEmbedder(dim=8))       # anchors, no key
    assert e2.cache is not None                            # it can route...
    assert not e2.ready                                    # ...and it verifies no shard
    with pytest.raises(NotInitialised, match="authority public key"):
        e2.ask("anything")


def test_seed_gives_first_light(tmp_path) -> None:
    priv = Ed25519PrivateKey.generate()
    e = Ember.boot(_settings(tmp_path), embedder=HashEmbedder(dim=8))
    e = e.seed(_anchors(), priv.public_key())
    assert e.ready
    st = e.status()
    assert st["anchors"] == len(LABELS) and st["has_authority_key"]


def test_boot_touches_no_network(tmp_path, monkeypatch) -> None:
    """Boot reads the disk and nothing else, so it is fast, works offline, and has nothing to hang
    on when the cloud is unreachable. The socket layer is monkeypatched to make that measurable."""
    import socket

    def _forbidden(*a, **k):
        raise AssertionError("boot attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    priv = Ed25519PrivateKey.generate()
    s = _settings(tmp_path)
    Ember.boot(s, embedder=HashEmbedder(dim=8)).seed(_anchors(), priv.public_key())
    e = Ember.boot(s, embedder=HashEmbedder(dim=8), authority_pub=priv.public_key())
    assert e.ready


def test_remember_then_restart_answers_offline(tmp_path) -> None:
    """End to end: author, drop the process, boot again, answer — with no network in the process."""
    priv = Ed25519PrivateKey.generate()
    s = _settings(tmp_path)
    e = Ember.boot(s, embedder=HashEmbedder(dim=8)).seed(_anchors(), priv.public_key())
    e.authority_pub = priv.public_key()
    # A rung is required at write time — no default (see test_provenance_write.py).
    e.remember([("art-1", b"An ontology names the things a system can talk about.", "ontology")],
               provenance=Provenance.HUMAN_VALIDATED, authority="ember-local", priv=priv)

    # A brand new process.
    again = Ember.boot(s, embedder=HashEmbedder(dim=8), authority_pub=priv.public_key())
    assert again.ready
    assert again.loaded >= 1 and again.rejected == 0, "shards must survive the restart"


def test_the_crosswalk_is_fitted_once_then_adopted(tmp_path) -> None:
    """Boot #1 fits the cross-walk and publishes it; boot #2 adopts the published one. The fit is
    deterministic, so this holds for the same leaf and for any leaf on the same embedder and
    anchors."""
    priv = Ed25519PrivateKey.generate()
    s = _settings(tmp_path)
    first = Ember.boot(s, embedder=HashEmbedder(dim=8)).seed(_anchors(), priv.public_key())
    assert first.aligner is not None and not first.aligner.adopted    # it did the fitting
    assert first.store.find_artifact(CROSSWALK_CONTENT_TYPE) is not None  # published it

    second = Ember.boot(s, embedder=HashEmbedder(dim=8), authority_pub=priv.public_key())
    assert second.aligner.adopted, "a published cross-walk must be reused, not refitted"


def test_a_leaf_never_authors_its_own_anchors(tmp_path) -> None:
    """Anchors are a shared coordinate system, so a leaf receives them rather than authoring them.
    A home-made AnchorSet routes and answers perfectly well while sharing with nobody, which is why
    `NotInitialised` names this case in its message."""
    e = Ember.boot(_settings(tmp_path), embedder=HashEmbedder(dim=8))
    assert e.anchors is None
    with pytest.raises(NotInitialised) as ex:
        e.ask("q")
    assert "author its own anchors" in str(ex.value)


def test_status_is_honest_about_alignment_cost(tmp_path) -> None:
    """Status reports two separate things about the cross-walk: whether a walk exists, and whether
    it carries information.

    The in-sample residual measures how well the fit reproduces its own training points, so it
    stays out of status: on this path it reads 0.107 while the held-out residual is 1.026 against a
    derived null of 1.0. Published alone, 0.107 reads as high fidelity for a walk that says nothing.
    `alignment_carries_information` is a verdict against the null, and
    `alignment_residual_held_out` is the number behind it.
    """
    priv = Ed25519PrivateKey.generate()
    e = Ember.boot(_settings(tmp_path), embedder=HashEmbedder(dim=8)).seed(_anchors(), priv.public_key())
    st = e.status()

    assert st["aligned"], "a fitted aligner exists"
    assert "alignment_error" not in st, (
        "the in-sample number must not be published — a reader cannot tell it from a fidelity claim")
    assert "alignment_carries_information" in st and "alignment_residual_held_out" in st

    held = st["alignment_residual_held_out"]
    carries = st["alignment_carries_information"]
    assert carries in (True, False, None), "a verdict, never a number standing in for one"

    # The claim that can fail: when status says the walk carries information, the held-out residual
    # sits below the derived null by more than two standard errors.
    if carries:
        from mantle.search.anchors.crosswalk import null_residual
        res = e.aligner.residual
        null, se = null_residual(res.dim_out, max(1, res.n_held_out))
        assert held is not None and held < null - 2.0 * se, (
            "status claimed alignment carries information while sitting at the null")


def test_an_unwired_ember_refuses_instead_of_answering(tmp_path) -> None:
    """An ember with no answerer raises `NoAnswerer` rather than returning an empty answer.

    An empty answer is a measurement: nothing was found. No answerer is the absence of one, and the
    two are indistinguishable to a caller if both come back as an empty result. The computed null
    belongs to the persona that was asked, not to the runner.

    This also pins the default: `boot()` constructs no answerer — composing prose from evidence is
    a persona concern, outside ember — so `engine is None` until something injects one."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from ember.runtime.engine import NoAnswerer

    priv = Ed25519PrivateKey.generate()
    e = Ember.boot(_settings(tmp_path), embedder=HashEmbedder(dim=8)).seed(_anchors(),
                                                                          priv.public_key())
    assert e.engine is None, "boot constructed an answerer — the runner is answering again"
    with pytest.raises(NoAnswerer):
        e.ask("anything at all")
