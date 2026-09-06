"""The pattern runner — the gate on the single distribution path (the loader is `prism.runner`).

What holds: every shipped bundle verifies and loads; content addressing is enforced, so a tampered
bundle raises before any exec; the provenance seam declines an authorless or dangling-author store
bundle; a verified store bundle outranks the shipped copy; and shared modules resolve to one object
by content, process-wide.

The trust gate (signature/rung leg, env-gated): sign->verify roundtrips; a tampered bundle does not
verify; unsigned with the gate on raises `BundleTrustError`; unsigned with the gate off grounds; the
signer's channel is one that grounds something (explicit `EMBER_BUNDLE_CHANNELS`, or the
`has_referent` default); and the off state leaves its breadcrumb in the log.
"""
from __future__ import annotations

import hashlib
import json
import logging

import pytest

from prism.trust import opsign
# The loader is prism's. This file exercises bundle verification and exec — `_loaded`, `_canonical`,
# `_DATA_DIR`, the trust gate — all of which live in `prism.runner`, so it imports prism directly.
# Reaching them through ember's re-export shim would no-op silently: `monkeypatch.setattr` writes to
# the shim's namespace while `load()` reads prism's, so every gate test would pass without gating
# anything. Ember's share of the runner — registering the host seams — is covered by
# `test_runner_seam.py`.
from prism import runner


class _Arts:
    """Minimal artifact store double (get/put by id)."""
    def __init__(self):
        self.d = {}

    def get_artifact(self, k):
        return self.d.get(k)

    def put_artifact(self, doc):
        self.d[doc["id"]] = dict(doc)
        return doc


@pytest.fixture
def unpinned(monkeypatch):
    """A copy of the runner's pin table plus a detached store, restored afterwards, so a test can
    clear individual pins to force a fresh load without disturbing the process's real pins."""
    monkeypatch.setattr(runner, "_loaded", dict(runner._loaded))
    monkeypatch.setattr(runner, "_attached_store", None)
    return runner


def _shipped(group):
    return json.loads((runner._DATA_DIR / (group + ".json")).read_text(encoding="utf-8"))


def _store_with_bundle(bundle, *, created_by="person-author"):
    arts = _Arts()
    if created_by:
        arts.put_artifact({"id": created_by, "content_type": "application/vnd.agience.person+json"})
    doc = {"id": runner.BUNDLE_ARTIFACT_PREFIX + bundle["group"],
           "content_type": runner.BUNDLE_CONTENT_TYPE,
           "content": json.dumps(bundle)}
    if created_by:
        doc["created_by"] = created_by
    arts.put_artifact(doc)
    return arts


def test_every_shipped_bundle_verifies_and_loads():
    for g in runner.GROUPS:
        entry = runner.load(g)
        info = runner.loaded()[g]
        assert info["sha256"] == _shipped(g)["sha256"]
        assert entry.__name__.endswith("." + _shipped(g)["entry_module"])
    # the manifest-declared register fns resolve on every group
    for g in runner.GROUPS:
        for fn in runner.register_fns(g):
            assert callable(fn)


def test_shared_modules_are_one_object_by_content():
    """`evolution` reached from any bundle is the same module and `Answer` is one class: identity
    follows the distributed content, exactly like the store."""
    ev = runner.load("evolution")
    assert runner.load("fetch", "evolution") is ev
    assert runner.load("corpus", "evolution") is ev
    assert runner.load("dev_ops", "answer").Answer is runner.load("answer").Answer
    from ember.runtime.runner import answer as _answer_mod   # the bundle's own module, via the shim
    Answer = _answer_mod.Answer
    assert Answer is runner.load("answer").Answer


def test_a_tampered_bundle_is_refused_before_exec(unpinned):
    """The integrity gate. Content that does not hash to its ref never runs: a mismatch raises
    `BundleIntegrityError` before exec rather than falling back to the shipped copy."""
    b = _shipped("fetch")
    b["modules"]["fetch"] += "\nTAMPERED = True\n"      # content changed, sha claim stale
    runner.attach(_store_with_bundle(b))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleIntegrityError):
        runner.load("fetch")


def test_a_correctly_rehashed_store_bundle_outranks_the_shipped_one(unpinned):
    b = _shipped("fetch")
    b["modules"]["fetch"] += "\nSTORE_BUNDLE_MARK = True\n"
    b["sha256"] = hashlib.sha256(runner._canonical(b)).hexdigest()
    runner.attach(_store_with_bundle(b))
    runner._loaded.pop("fetch", None)
    mod = runner.load("fetch")
    assert getattr(mod, "STORE_BUNDLE_MARK", False) is True
    assert runner.loaded()["fetch"]["origin"] == "store"


def test_an_authorless_store_bundle_is_refused(unpinned):
    """Trust gate default (flagged seam): an executable pattern from the store with no resolvable
    author does not run, because provenance needs an authority to attach to."""
    b = _shipped("fetch")
    runner.attach(_store_with_bundle(b, created_by=None))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleIntegrityError):
        runner.load("fetch")


def test_a_dangling_author_is_refused(unpinned):
    b = _shipped("fetch")
    arts = _store_with_bundle(b, created_by="person-author")
    del arts.d["person-author"]                          # the claim does not resolve
    runner.attach(arts)
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleIntegrityError):
        runner.load("fetch")


def test_an_absent_store_bundle_falls_back_to_shipped(unpinned):
    runner.attach(_Arts())                               # store present, no bundle artifact in it
    runner._loaded.pop("fetch", None)
    runner.load("fetch")
    assert runner.loaded()["fetch"]["origin"] == "shipped"


# ── the trust gate: signature/rung leg (EMBER_REQUIRE_SIGNED) ─────────────────────────────────


@pytest.fixture
def gate_off(monkeypatch):
    monkeypatch.delenv("EMBER_REQUIRE_SIGNED", raising=False)
    monkeypatch.delenv("EMBER_MIN_BUNDLE_RUNG", raising=False)
    monkeypatch.delenv("EMBER_BUNDLE_CHANNELS", raising=False)


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv("EMBER_REQUIRE_SIGNED", "1")
    monkeypatch.delenv("EMBER_MIN_BUNDLE_RUNG", raising=False)
    monkeypatch.delenv("EMBER_BUNDLE_CHANNELS", raising=False)


def _rehashed(group="fetch", mark="STORE_BUNDLE_MARK"):
    b = _shipped(group)
    b["modules"][group] += "\n%s = True\n" % mark
    b["sha256"] = hashlib.sha256(runner._canonical(b)).hexdigest()
    return b


def _keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    priv = Ed25519PrivateKey.generate()
    return priv, opsign.public_key_hex(priv.public_key())


def _signed_store(bundle, priv, pub_hex, *, rung="observed", attest=True,
                  cited_from="cite.employer", cite_exists=True):
    """A store holding the signed bundle plus an author artifact that attests the signing key, names
    a provenance channel, and cites something — the store-side facts the gate binds together.

    `cited_from` and `cite_exists` are separable on purpose. A label cannot dangle and a citation
    can, which is why the gate resolves the citation rather than reading a string.
    `cite_exists=False` writes an author whose citation names an artifact absent from this store,
    which no check that reads only labels can tell apart from a good author."""
    arts = _Arts()
    author = {"id": "person-author",
              "content_type": "application/vnd.agience.person+json",
              "cited_from": cited_from,
              "context": {"provenance": rung}}
    if attest:
        author["signed_by"] = pub_hex
    arts.put_artifact(author)
    if cited_from and cite_exists:
        arts.put_artifact({"id": cited_from, "content_type": "application/x-citation",
                           "content": "the organisation that stands behind this author"})
    arts.put_artifact({"id": runner.BUNDLE_ARTIFACT_PREFIX + bundle["group"],
                       "content_type": runner.BUNDLE_CONTENT_TYPE,
                       "content": json.dumps(bundle),
                       "created_by": "person-author"})
    return arts


def test_sign_bundle_roundtrip_and_tamper():
    """opsign level: the envelope signs the same canonical payload the sha covers, so signing does
    not move the sha. Verify passes on the roundtrip, and a tampered bundle does not verify."""
    b = _shipped("fetch")
    sha_before = b["sha256"]
    priv, pub_hex = _keypair()
    signed = opsign.sign_bundle(b, priv)
    assert signed["signed_by"] == pub_hex
    assert hashlib.sha256(runner._canonical(signed)).hexdigest() == sha_before
    ok, why = opsign.verify_bundle(signed)
    assert ok, why
    ok, why = opsign.verify_bundle(signed, pub=priv.public_key())
    assert ok and "supplied key" in why
    signed["modules"]["fetch"] += "\nTAMPERED = True\n"
    ok, why = opsign.verify_bundle(signed)
    assert not ok and "does not verify" in why
    assert not opsign.verify_bundle({"group": "fetch"})[0]        # unsigned => False, no raise


def test_signed_store_bundle_grounds_with_gate_on(unpinned, gate_on):
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    runner.attach(_signed_store(b, priv, pub_hex, rung="observed"))
    runner._loaded.pop("fetch", None)
    mod = runner.load("fetch")
    assert getattr(mod, "STORE_BUNDLE_MARK", False) is True
    assert runner.loaded()["fetch"]["origin"] == "store"


def test_tampered_signature_refuses_with_gate_on(unpinned, gate_on):
    """Tamper after signing and re-sha so the integrity leg passes: the signature leg is then the
    one that raises, with `BundleTrustError`, before any exec."""
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    b["modules"]["fetch"] += "\nEVIL = True\n"
    b["sha256"] = hashlib.sha256(runner._canonical(b)).hexdigest()
    runner.attach(_signed_store(b, priv, pub_hex))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="does not verify"):
        runner.load("fetch")


def test_unsigned_store_bundle_refuses_with_gate_on(unpinned, gate_on):
    priv, pub_hex = _keypair()
    runner.attach(_signed_store(_rehashed(), priv, pub_hex))      # the bundle itself is unsigned
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="UNSIGNED"):
        runner.load("fetch")


def test_unsigned_store_bundle_grounds_with_gate_off(unpinned, gate_off):
    """With the gate off, an unsigned store bundle with a resolvable author grounds: the signature
    leg is the only thing the gate adds."""
    runner.attach(_store_with_bundle(_rehashed()))
    runner._loaded.pop("fetch", None)
    mod = runner.load("fetch")
    assert getattr(mod, "STORE_BUNDLE_MARK", False) is True
    assert runner.loaded()["fetch"]["origin"] == "store"


def test_unattested_or_wrong_key_refuses_with_gate_on(unpinned, gate_on):
    """The channel is meaningful only when the key is bound to the store-resolved author. An author
    artifact attesting no key, or a different key, leaves the binding unmade — however validly the
    bundle verifies against its own embedded key."""
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    runner.attach(_signed_store(b, priv, pub_hex, attest=False))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="attests no signing key"):
        runner.load("fetch")
    _other_priv, other_pub = _keypair()
    runner.attach(_signed_store(b, priv, other_pub))              # the author attests someone else
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="is not the key author"):
        runner.load("fetch")


def test_signer_channel_outside_the_explicit_set_refuses(unpinned, gate_on, monkeypatch):
    """`EMBER_BUNDLE_CHANNELS` is an explicit set of channel names: a signer whose channel is
    outside the set does not ground, and one inside it does. Channels carry no ordering, so a set is
    the only thing a caller can state."""
    monkeypatch.setenv("EMBER_BUNDLE_CHANNELS", "observed,human_validated")
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    runner.attach(_signed_store(b, priv, pub_hex, rung="hypothesis"))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="not in the required set"):
        runner.load("fetch")
    # in the set => grounds
    runner.attach(_signed_store(b, priv, pub_hex, rung="observed"))
    runner._loaded.pop("fetch", None)
    mod = runner.load("fetch")
    assert getattr(mod, "STORE_BUNDLE_MARK", False) is True


def test_an_ungrounded_signer_is_refused_by_the_default_gate(unpinned, gate_on):
    """The default gate is `has_referent`: the author cites something, and that citation resolves in
    this store. An author with no `cited_from` does not ground, whatever channel it stamps on
    itself, and an author that cites a present artifact grounds on any channel.

    A mass floor over the channel labels cannot state this. At `weigh(rung).mass <= GHOST_FLOOR`
    (0.10) the excluded set is one channel — assertion, at 0.02 — while UNKNOWN sits at 0.12 and
    passes, so an author whose provenance was never recorded could sign executable patterns."""
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    for channel in ("assertion", "unknown", "hypothesis", "human_validated"):
        runner.attach(_signed_store(b, priv, pub_hex, rung=channel, cited_from=""))
        runner._loaded.pop("fetch", None)
        with pytest.raises(runner.BundleTrustError, match="not grounded"):
            runner.load("fetch")
    for channel in ("span_cited", "observed", "human_validated"):
        runner.attach(_signed_store(b, priv, pub_hex, rung=channel))
        runner._loaded.pop("fetch", None)
        assert getattr(runner.load("fetch"), "STORE_BUNDLE_MARK", False) is True


def test_a_DANGLING_citation_grounds_nothing(unpinned, gate_on):
    """A citation can dangle, which is the failure mode a label cannot have and the reason the gate
    resolves.

    This author is `human_validated`, the strongest channel there is and a pass under any gate that
    reads a channel, and it cites a source that is absent from this store. Nothing stands behind it,
    and label-reading cannot notice, because a string cannot dangle. The control is the same author
    with the citation present, which grounds."""
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    runner.attach(_signed_store(b, priv, pub_hex, rung="human_validated",
                                cited_from="cite.vanished", cite_exists=False))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="not grounded"):
        runner.load("fetch")

    # control: the same author, same channel, with the citation actually present — grounds.
    runner.attach(_signed_store(b, priv, pub_hex, rung="human_validated",
                                cited_from="cite.vanished", cite_exists=True))
    runner._loaded.pop("fetch", None)
    assert getattr(runner.load("fetch"), "STORE_BUNDLE_MARK", False) is True


def test_a_SELF_ANCHORED_author_cannot_mint_its_own_grounding(unpinned, gate_on):
    """An artifact citing itself resolves — it is right there — so a bare "does it resolve" check
    would let anything self-anchor into being grounded, which is laundering with one extra step.
    An axiom is assumed rather than grounded, and only the system's root anchor is one."""
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    # cite_exists=False because the cited id is the author: it already resolves, which is the trap.
    # Writing a second artifact under that id would clobber the author's attested key.
    runner.attach(_signed_store(b, priv, pub_hex, rung="human_validated",
                                cited_from="person-author", cite_exists=False))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="not grounded"):
        runner.load("fetch")


def test_a_mis_set_gate_refuses_rather_than_weakens(unpinned, gate_on, monkeypatch):
    """A typo'd channel name fails closed. Mapping it to UNKNOWN, which is `provenance_of`'s read
    fallback, would silently weaken an execution gate."""
    monkeypatch.setenv("EMBER_BUNDLE_CHANNELS", "observedd")
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    runner.attach(_signed_store(b, priv, pub_hex, rung="human_validated"))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="not a prism.mass.Provenance"):
        runner.load("fetch")


def test_the_retired_env_var_refuses_rather_than_being_reinterpreted(unpinned, gate_on, monkeypatch):
    """A deployed setting keeps its meaning or goes red. `EMBER_MIN_BUNDLE_RUNG=observed` named a
    floor on an ordering; read as a set it would mean something different, and narrower, with nobody
    touching the environment. So a deployment still carrying it fails loudly and names
    `EMBER_BUNDLE_CHANNELS`."""
    monkeypatch.setenv("EMBER_MIN_BUNDLE_RUNG", "observed")
    priv, pub_hex = _keypair()
    b = opsign.sign_bundle(_rehashed(), priv)
    runner.attach(_signed_store(b, priv, pub_hex, rung="human_validated"))
    runner._loaded.pop("fetch", None)
    with pytest.raises(runner.BundleTrustError, match="EMBER_BUNDLE_CHANNELS"):
        runner.load("fetch")


def test_shipped_bundles_load_under_either_gate_state(unpinned, gate_on):
    """Shipped data files ride package-install trust and skip the store-provenance leg, so a node
    with no store bundles boots with the gate on."""
    runner.attach(_Arts())
    runner._loaded.pop("fetch", None)
    runner.load("fetch")
    assert runner.loaded()["fetch"]["origin"] == "shipped"


def test_gate_off_leaves_the_honest_breadcrumb(gate_off, caplog):
    with caplog.at_level(logging.INFO, logger="prism.runner"):   # the loader's own logger
        runner._log_gate_state()
    assert ("bundle signature gate OFF (EMBER_REQUIRE_SIGNED unset) — flagged seam"
            in caplog.text)


def test_gate_on_breadcrumb_states_the_floor(gate_on, caplog):
    with caplog.at_level(logging.INFO, logger="prism.runner"):   # the loader's own logger
        runner._log_gate_state()
    assert "bundle signature gate ON" in caplog.text
    assert "has_referent" in caplog.text


def test_the_first_load_pins_for_the_process(unpinned):
    """One logical runtime runs one version of a group throughout — the genesis `_pinned` rule, one
    layer down — so attaching a store after a group has loaded leaves the pin in place."""
    runner._loaded.pop("fetch", None)
    first = runner.load("fetch")
    b = _shipped("fetch")
    b["modules"]["fetch"] += "\nLATE = True\n"
    b["sha256"] = hashlib.sha256(runner._canonical(b)).hexdigest()
    runner.attach(_store_with_bundle(b))
    assert runner.load("fetch") is first, "a mid-process attach swapped a pinned bundle"
