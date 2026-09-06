"""The signal — one signed primitive, with the channel selecting message or event.

Grouped by invariant, covering PEERING-AND-MESSAGING P1 (envelope) and P2 (regime router). The
through-line: a signal is signed and content-addressed; its channel is derived at the receiver from
the signed fields rather than read off the wire; and that derived channel, rather than the sender's
say-so, decides whether the signal propagates or transforms stored state.
"""
from __future__ import annotations

import tempfile

import pytest

from ember.signal import signal
from ember.runtime import delegate
from prism.mass import Provenance
from _fakes import _FakeStore
from _fakes import _install_offline_wordnet


@pytest.fixture(scope="module")
def _wn():
    if not _install_offline_wordnet():
        pytest.skip("WordNet not available")
    return True


@pytest.fixture()
def store():
    s = _FakeStore()
    s.keys_dir = tempfile.mkdtemp()
    return s


@pytest.fixture()
def alice(store):
    """An authenticated delegate — real person and real origin (SYSTEM-kind signals)."""
    delegate.reset_cache()
    return delegate.Delegate.get(store, person="alice@x.com", id="d.alice",
                                 origin="https://agience.ai", restore=False)


@pytest.fixture()
def anon(store):
    """A local/anonymous delegate (CLIENT-kind signals)."""
    return delegate.Delegate.get(store, id="d.anon", restore=False)


# ── INVARIANT 1: the envelope is signed and content-addressed ─────────────────────────────────
def test_seal_stamps_the_provenance_quadruple(alice):
    sig = signal.seal(alice, to="op.host.gw.x", seeds={"wolf.n.01": 1.0})
    assert set(sig["from_"]) == {"origin", "person", "host", "observer"}
    assert sig["from_"]["observer"] == "d.alice"
    assert sig["from_"]["person"] == "alice@x.com"
    assert sig["content_type"] == signal.SIGNAL_CONTENT_TYPE
    assert sig["signature"] and sig["signed_by"]


def test_the_id_is_the_content_address(alice):
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0})
    assert sig["id"] == signal._content_address(sig)


def test_verify_accepts_a_valid_signal(alice):
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0})
    ok, _why = signal.verify(sig)
    assert ok is True


def test_an_unsigned_signal_is_refused(alice):
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0}, priv=None, keys_dir=None)
    # force the no-key path: strip the sig if one got attached from the store keys_dir
    sig.pop("signature", None)
    ok, why = signal.verify(sig)
    assert ok is False and "unsigned" in why


@pytest.mark.parametrize("field,badval", [
    ("to", "op.host.EVIL.x"),
    ("operator", "op.something.else"),
    ("claimed", "human_validated"),
    ("nonce", "deadbeefdeadbeef"),
])
def test_tampering_any_signed_field_breaks_the_signature(alice, field, badval):
    """`to`, `from`, the claimed channel and the nonce are all inside the signed payload, so an
    alteration in flight leaves the signature unverifiable."""
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0})
    sig[field] = badval
    ok, why = signal.verify(sig)
    assert ok is False


def test_an_id_that_disagrees_with_the_body_is_caught_even_when_signed(alice, store):
    """The content-address invariant: `id` equals `hash(body)`. A malicious sealer controls its own
    key, so it can sign a signal whose id does not match its body in order to confuse dedup.
    Re-signing a wrong id is the only route to a valid signature over a bad id, and verification
    checks the address separately."""
    from prism.trust import opsign
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0})
    priv, _ = opsign.authority_key(store.keys_dir, create=True)
    sig["id"] = "0" * 64                                  # a lie about the content address
    sig["signature"] = opsign.sign_bytes(signal._canonical(sig), priv)   # validly re-signed
    ok, why = signal.verify(sig)
    assert ok is False and "content" in why


# ── INVARIANT 2: the channel is derived at the receiver, from the signed fields ────────────────
def test_a_client_sender_cannot_inflate_mass_to_force_an_event(anon):
    """The anti-laundering invariant. A CLIENT-kind sender claiming `human_validated` derives to
    ASSERTION — its real channel — before the switch reads it, the same discipline as
    `derive_provenance`. A peer's standing is a property of the peer, not of what it writes."""
    from_ = {**anon.provenance(), "observer": anon.id}
    ch = signal.derive_channel(from_, "human_validated")
    assert ch is Provenance.ASSERTION, "a client's over-claim survived derivation"
    assert signal.regime(ch) == signal.MESSAGE, "a client claim reached the event regime"


def test_a_wire_mass_field_is_ignored_by_the_router(alice, store, _wn):
    """`route` re-derives the channel, so a validly-signed signal carrying a `channel` field routes
    on the derivation rather than on the field."""
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0}, claimed="hypothesis")
    sig_forged = dict(sig)
    sig_forged["channel"] = "human_validated"             # a claim in an unsigned field
    # `channel` is not in SIGNED_FIELDS, so the signature still verifies — which is why the router
    # re-derives from the signed `from_` and `claimed` instead of reading it.
    assert signal.verify(sig_forged)[0] is True
    r = signal.route(alice, sig_forged, store=store)
    assert r["regime"] == signal.MESSAGE, "the router trusted a wire mass and made it an event"


def test_an_authenticated_observation_reaches_the_event_regime(alice):
    from_ = alice.provenance()
    from_ = {**from_, "observer": alice.id}
    ch = signal.derive_channel(from_, "observed")
    assert ch is Provenance.OBSERVED and signal.regime(ch) == signal.EVENT


# ── INVARIANT 3: the switch — ungrounded propagates, grounded transforms ──────────────────────
def test_the_switch_is_a_partition_with_no_edge_to_sit_on():
    """The switch asks whether anything checks the claim, and each channel answers that on its own.
    So the regimes are a partition over the channel set with no numeric boundary between them, and
    no edge to tune: grounded channels are EVENT, ungrounded ones are MESSAGE."""
    for grounded in (Provenance.HUMAN_VALIDATED, Provenance.OBSERVED, Provenance.SPAN_CITED):
        assert signal.regime(grounded) == signal.EVENT
    for ungrounded in (Provenance.HYPOTHESIS, Provenance.UNKNOWN, Provenance.ASSERTION,
                       Provenance.ONTOLOGY_PROPOSAL):
        assert signal.regime(ungrounded) == signal.MESSAGE
    assert not hasattr(signal, "ELECTROWEAK_SCALE")


def test_a_message_propagates_and_changes_no_state(alice, store, _wn):
    """MESSAGE / ungrounded: activate the receiver, change nothing."""
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0}, claimed="hypothesis")
    before = len(store.artifacts.d)
    r = signal.route(alice, sig, store=store)
    assert r["regime"] == signal.MESSAGE and r["delivered"] is True
    assert len(store.artifacts.d) == before, "a message wrote to the store"
    # it deposited a trace on the delegate's own screen (activation happened)
    assert alice.tick >= 1


def test_an_event_LANDS_beside_what_is_held_and_radiates(alice, store, _wn):
    """EVENT / grounded: the content lands beside what is held, and the signal still activates the
    receiver — an event radiates as well as landing.

    Nothing arbitrates between versions here; the sibling test below carries why.
    """
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0}, claimed="observed")
    # carry a content artifact for it to land; nothing is held under this root yet
    sig["content"] = {"id": "art.x", "root_id": "art.x", "content": "a fact"}
    r = signal.route(alice, sig, store=store)
    assert r["regime"] == signal.EVENT
    assert r["stands_beside"] is False                   # nothing is held under this root
    assert "revision" not in r and "transforms" not in r, (
        "a displacement verdict is back in the answer")
    assert "radiated" in r                               # it also propagated


def test_an_event_against_held_state_LANDS_BESIDE_IT_and_decides_nothing(alice, store, _wn):
    """An event arriving against held state lands beside it, and the answer carries no displacement
    verdict. Which version answers is resolved at read time by the reader.

    There is nothing here to arbitrate on. At this seam the counts are always 1 against 1, and
    `prism.resolution.separated([1, 1])` is False — as it is for every pair, since at n=2 the
    computed null is exactly 1.0000. A "newer wins" rule would be a clock rather than evidence, and
    a `mass` field on the artifact is a claim by whoever holds the bytes.
    """
    store.artifacts.put_artifact({"id": "art.h", "root_id": "art.h", "content": "human fact"})
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0}, claimed="observed")
    sig["content"] = {"id": "art.h", "root_id": "art.h", "content": "an agent's correction"}
    r = signal.route(alice, sig, store=store)
    assert r["regime"] == signal.EVENT
    assert r["stands_beside"] is True                    # a version is already held under this root
    assert "revision" not in r, "a displacement verdict is back in the answer"


def test_an_ungrounded_claim_never_reaches_the_transform_path(alice):
    """A claim with nothing checking it stays in MESSAGE whatever artifact it carries, so it never
    reaches the path that transforms stored state.

    Asserted on the switch rather than through `route()`: routing a MESSAGE runs activation, so a
    green result there would also depend on the ontology driver being healthy, and the claim under
    test is about the switch. `test_a_message_propagates_and_changes_no_state` covers the routing
    half."""
    from_ = {**alice.provenance(), "observer": alice.id}
    for ungrounded in ("hypothesis", "unknown", "assertion", "ontology_proposal", "", "garbage"):
        ch = signal.derive_channel(from_, ungrounded)
        assert signal.regime(ch) == signal.MESSAGE, \
            "%r reached the transform path" % ungrounded


# ── INVARIANT 4: verify → authorize (D4) → switch, in that order ──────────────────────────────
def test_an_unverifiable_signal_never_reaches_the_switch(alice, store):
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0})
    sig["to"] = "tampered"
    r = signal.route(alice, sig, store=store)
    assert r["delivered"] is False and r["regime"] is None and "unverifiable" in r["reason"]


def test_a_signal_carrying_another_persons_private_content_is_refused(alice, store, _wn):
    """D4 — authorization is the extent of the field, and it sits upstream of the switch. A signal
    carrying another person's grant-gated artifact is undelivered as message and as event alike.
    Private is a grant: the content is grounded in bob's gated collection rather than marked by a
    flag."""
    from mantle.db import access
    access.mint_owner_read_grant(store, "private.bob@x.com", "bob@x.com")   # bob's grant gates it
    sig = signal.seal(alice, to="a", claimed="observed")
    sig["content"] = {"id": "secret", "collection_id": "private.bob@x.com", "content": "bob's diary"}
    r = signal.route(alice, sig, store=store)
    assert r["delivered"] is False and "field" in r["reason"].lower()


def test_a_pure_activation_is_always_within_the_field(alice, store, _wn):
    """Seeds-only, with no content artifact, is always within the field: hearing is not reaching."""
    sig = signal.seal(alice, to="a", seeds={"wolf.n.01": 1.0})
    r = signal.route(alice, sig, store=store)
    assert r["delivered"] is True


# ── INVARIANT 5: P3 — an arriving signal is delivered rather than left inert ──────────────────
def test_deliver_routes_a_signal_addressed_to_this_observer(alice, store, _wn):
    bob = delegate.Delegate.get(store, person="bob@x.com", id="d.bob", restore=False)
    sig = signal.seal(alice, to="d.bob", seeds={"wolf.n.01": 1.0})
    r = signal.deliver(bob, [sig], store=store)
    assert r["delivered"] == 1 and r["signals"][0]["regime"] in (signal.MESSAGE, signal.EVENT)


def test_deliver_skips_signals_for_other_observers(alice, store, _wn):
    carol = delegate.Delegate.get(store, person="carol@x.com", id="d.carol", restore=False)
    sig = signal.seal(alice, to="d.bob", seeds={"wolf.n.01": 1.0})   # addressed to bob, not carol
    r = signal.deliver(carol, [sig], store=store)          # carol is not the addressee
    assert r["delivered"] == 0 and r["skipped"] == 1


def test_deliver_skips_the_delegates_own_signals_echo_guard():
    """A delegate skips what it authored, even when the signal is addressed to itself — the same
    no-echo rule the mesh enforces."""
    from _fakes import _FakeStore as _S
    import tempfile
    s = _S(); s.keys_dir = tempfile.mkdtemp(); delegate.reset_cache()
    a = delegate.Delegate.get(s, person="a@x.com", id="d.a", origin="https://agience.ai", restore=False)
    own = signal.seal(a, to="d.a", seeds={"wolf.n.01": 1.0})   # addressed to self
    r = signal.deliver(a, [own], store=s)
    assert r["delivered"] == 0 and r["skipped"] == 1


def test_deliver_ignores_non_signal_artifacts(alice, store, _wn):
    bob = delegate.Delegate.get(store, person="bob@x.com", id="d.bob", restore=False)
    batch = [{"id": "x", "content_type": "text/markdown"},
             signal.seal(alice, to="d.bob", seeds={"wolf.n.01": 1.0})]
    r = signal.deliver(bob, batch, store=store)
    assert r["considered"] == 2 and r["delivered"] == 1 and r["skipped"] == 1


def test_a_broadcast_reaches_everyone(alice, store, _wn):
    bob = delegate.Delegate.get(store, person="bob@x.com", id="d.bob", restore=False)
    sig = signal.seal(alice, to="*", seeds={"wolf.n.01": 1.0})
    assert signal.addressed_to(sig, bob) is True
    assert signal.deliver(bob, [sig], store=store)["delivered"] == 1


# ── INVARIANT 6: P4 — the resolver reads the endpoint; send wraps a signal ────────────────────
def test_resolve_reads_the_dead_endpoint_of_a_remote_operator(store, _wn):
    from ember.runtime import capability
    capability.register_remote_host(store, name="gw", operators=["analyze"],
                                    endpoint="http://10.0.0.7:8083")
    r = signal.resolve("op.host.gw.analyze", store=store)
    assert r["kind"] == "remote" and r["endpoint"] == "http://10.0.0.7:8083"
    assert r["operator_name"] == "analyze"


def test_resolve_distinguishes_the_address_kinds(store):
    assert signal.resolve("d.carol", store=store)["kind"] == "observer"
    assert signal.resolve("a" * 64, store=store)["kind"] == "content"
    assert signal.resolve("agi://operator.x@agience.ai", store=store)["kind"] == "agi"
    assert signal.resolve("", store=store)["kind"] == "unknown"


def test_resolve_a_local_operator(store):
    store.artifacts.put_artifact({"id": "op.mine", "content_type": "application/vnd.agience.operator+json",
                                  "state": "committed", "kind": "composition", "spec": {"steps": []}})
    r = signal.resolve("op.mine", store=store)
    assert r["kind"] == "local" and r["runtime"] == "composition"


def test_resolve_an_unknown_operator_says_so(store):
    r = signal.resolve("op.nope", store=store)
    assert r["kind"] == "unknown" and "no operator" in r["reason"]


def test_send_to_a_remote_operator_wraps_a_signal_not_an_rpc(alice, store, _wn):
    """Invoking a remote operator emits a signed signal toward it rather than dialing an RPC. The
    sealed signal and its transport target come back for P6 to ship; nothing is called
    synchronously."""
    from ember.runtime import capability
    capability.register_remote_host(store, name="gw", operators=["analyze"],
                                    endpoint="http://10.0.0.7:8083")
    r = signal.send(alice, "op.host.gw.analyze", seeds={"wolf.n.01": 1.0}, store=store)
    assert r["dispatched"] == "remote"
    assert r["transport"]["endpoint"] == "http://10.0.0.7:8083"
    assert signal.verify(r["signal"])[0] is True          # it's a properly sealed, verifiable signal
    assert "not an RPC" in r["note"]


def test_send_to_an_observer_delivers_locally(alice, store, _wn):
    r = signal.send(alice, "d.someone", seeds={"wolf.n.01": 1.0}, store=store)
    assert r["dispatched"] == "local" and r["target"]["kind"] == "observer"
    assert signal.verify(r["signal"])[0] is True


def test_send_to_an_unroutable_address_reports_it(alice, store, _wn):
    r = signal.send(alice, "op.nowhere", store=store)
    assert r["dispatched"] is None and "unroutable" in r["reason"]
