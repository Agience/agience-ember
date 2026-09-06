"""Operator signing and admission — verify what you reached, then decide whether to run it.

Grouped by invariant. The through-line: a signature attests authorship rather than safety, and each
distinct outcome stays distinguishable — "unsigned", "forged" and "valid but not admissible" are
three different answers, and collapsing any two of them is how a security check becomes theatre.
"""
from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from prism.trust import opsign
from ember import genesis as g
from _fakes import _FakeStore


@pytest.fixture()
def keys(tmp_path):
    priv, pub = opsign.authority_key(tmp_path, create=True)
    return priv, pub


def _op(**over):
    d = {"id": "op.t.x", "kind": "composition", "spec": {"steps": [{"op": "op.a"}]},
         "requires": ["compute.basic"], "effects": {"writes": False}}
    d.update(over)
    return d


# ── Invariant 1: a verify path never mints an identity ────────────────────────────────────────
def test_verification_never_creates_a_key(tmp_path):
    """A missing key means there is nothing to verify against; minting one is a separate act, asked
    for with `create=True`. `content.py` draws the same line for a different key: `_content_key`
    raises rather than minting one when a read finds no key, because a freshly generated key would
    decrypt nothing while every health metric kept reporting normal."""
    assert opsign.authority_key(tmp_path) == (None, None)
    assert not (tmp_path / "operator.key").exists()
    priv, _ = opsign.authority_key(tmp_path, create=True)
    assert priv is not None and (tmp_path / "operator.key").exists()


def test_the_key_is_stable_across_loads(tmp_path):
    a, _ = opsign.authority_key(tmp_path, create=True)
    b, _ = opsign.authority_key(tmp_path, create=True)
    assert opsign.public_key_hex(a.public_key()) == opsign.public_key_hex(b.public_key())


# ── Invariant 2: unsigned, forged and valid stay three distinct answers ───────────────────────
def test_unsigned_is_refused_as_a_draft(keys):
    ok, why = opsign.verify_operator(_op())
    assert ok is False and "unsigned" in why


def test_a_valid_signature_verifies(keys):
    priv, pub = keys
    signed = opsign.sign_operator(_op(), priv)
    assert opsign.verify_operator(signed, pub=pub) == (True, "verified against a supplied key")


def test_tampering_with_the_spec_breaks_the_signature(keys):
    priv, pub = keys
    signed = opsign.sign_operator(_op(), priv)
    signed["spec"] = {"steps": [{"op": "op.EVIL"}]}
    ok, why = opsign.verify_operator(signed, pub=pub)
    assert ok is False and "does not verify" in why


def test_tampering_with_requires_breaks_the_signature(keys):
    """`requires` is the capability contract (D12) and it is inside the signed payload. Left
    unsigned, a relay could strip a requirement and make an operator admissible on a host that does
    not offer it."""
    priv, pub = keys
    signed = opsign.sign_operator(_op(), priv)
    signed["requires"] = []
    assert opsign.verify_operator(signed, pub=pub)[0] is False


def test_a_different_key_does_not_verify(keys):
    priv, _ = keys
    signed = opsign.sign_operator(_op(), priv)
    other = Ed25519PrivateKey.generate()
    assert opsign.verify_operator(signed, pub=other.public_key())[0] is False


def test_self_consistency_is_reported_as_weaker_than_attested_authorship(keys):
    """Verifying against the embedded key shows only that nothing was tampered with in transit,
    since anyone can sign anything with a key they generated. The result says `unattested`, which
    is a weaker statement than verification against a supplied key."""
    priv, pub = keys
    signed = opsign.sign_operator(_op(), priv)
    ok, why = opsign.verify_operator(signed)             # no key supplied
    assert ok is True
    assert "unattested" in why
    assert "verified against a supplied key" != why


def test_fitness_counters_are_not_signed(keys):
    """Fitness counters accrue locally and differ per node, so they sit outside the signed payload:
    signing them would let every invocation invalidate the signature."""
    priv, pub = keys
    signed = opsign.sign_operator(_op(), priv)
    signed["invocations"] = 41
    signed["verified"] = 7
    assert opsign.verify_operator(signed, pub=pub)[0] is True


def test_carrying_different_behaviour_is_caught_by_the_SIGNATURE(keys):
    """A doc cannot advertise one content address while carrying different behaviour.

    The property lives in the signature. `canonical_operator` — the bytes the Ed25519 signature
    binds — covers {id, kind, spec, requires, effects}, so a spec that carries injected behaviour
    fails verification directly, with no side-car hash to consult.
    """
    priv, pub = keys
    signed = opsign.sign_operator(_op(), priv)
    assert opsign.verify_operator(signed, pub=pub)[0] is True

    tampered = dict(signed)
    tampered["spec"] = {**(tampered.get("spec") or {}), "injected": "different behaviour"}
    ok, why = opsign.verify_operator(tampered, pub=pub)
    assert ok is False and "signature" in why


def test_a_stray_spec_hash_field_is_IGNORED_not_honoured(keys):
    """A `spec_hash` field on the doc is inert: signing does not stamp one, and verification does
    not consult one. Honouring an unsigned field is the vulnerability — a value outside the signed
    payload can be set by anyone, and the only thing it can report on an otherwise-valid doc is
    staleness, which is how a canonicalizer change becomes a fleet-wide verification failure."""
    priv, pub = keys
    signed = opsign.sign_operator(_op(), priv)
    assert "spec_hash" not in signed, "signing must not stamp a second content address"

    signed["spec_hash"] = "0" * 64          # attacker-supplied, unsigned, and never consulted
    ok, _ = opsign.verify_operator(signed, pub=pub)
    assert ok is True, "an unsigned stray field must not decide authenticity in either direction"


# ── Invariant 3: a signature never authorizes execution ───────────────────────────────────────
def test_admission_requires_the_host_to_offer_what_is_required(keys):
    priv, _ = keys
    signed = opsign.sign_operator(_op(), priv)
    assert opsign.admit(signed, host_offers={"compute.basic"})[0] is True
    ok, why = opsign.admit(signed, host_offers=set())
    assert ok is False and "does not offer: compute.basic" in why


def test_unprobed_host_capabilities_are_refused_not_assumed(keys):
    """Three-valued discipline, per `resource.py`: has, has not, and was not probed stay
    distinguishable. `None` is an absent measurement, and admission needs a measurement."""
    priv, _ = keys
    signed = opsign.sign_operator(_op(), priv)
    ok, why = opsign.admit(signed, host_offers=None)
    assert ok is False and "not probed" in why


def test_a_perfectly_signed_code_operator_is_still_not_runnable(keys):
    """A valid signature says who wrote an operator, not what it does. Code-backed operators are
    reachable and not runnable until a sandbox exists (D10), so the signature verifies and
    admission still declines for want of one."""
    priv, pub = keys
    code = opsign.sign_operator({"id": "op.t.c", "kind": "python", "spec": {"src": "..."}}, priv)
    assert opsign.verify_operator(code, pub=pub)[0] is True, "the signature itself is fine"
    ok, why = opsign.admit(code, host_offers=set())
    assert ok is False and "no sandbox" in why


def test_an_unsigned_operator_is_never_admitted(keys):
    ok, why = opsign.admit(_op(), host_offers={"compute.basic"})
    assert ok is False and "unsigned" in why


# ── Invariant 4: publishing signs, and says so when it cannot ─────────────────────────────────
def test_define_operator_signs_when_a_keys_dir_exists(tmp_path):
    s = _FakeStore()
    s.keys_dir = tmp_path
    g.bootstrap(s)
    r = g.invoke(s, "op.operator.define", {"id": "op.t.signed", "kind": "composition",
                                           "spec": {"steps": [{"op": "op.consistency"}]}})["result"]
    # Signing is what this invariant is about, so signing is what is asserted. No `spec_hash` is
    # stamped: `crystal/evolution.py` and `opsign.sign_operator` both leave it off, because a
    # second, weaker content address (digesting only kind+spec) beside an Ed25519 signature that
    # covers a superset would need migrating whenever its canonicalizer changed.
    #
    # `define_operator` reaches `evolution` through `ember.runtime.runner` — the bundle — rather
    # than through `crystal` directly, so an assertion on a stamped field here measures whatever
    # bundle happens to be shipped rather than the source it names.
    assert r["signed"] is True
    doc = s.artifacts.get_artifact("op.t.signed")
    assert opsign.verify_operator(doc)[0] is True


def test_an_unsignable_publish_reports_it_rather_than_pretending(tmp_path):
    """With no keys dir there is no key to sign with, so the result says `signed: False` and carries
    a note. An unsigned publish that said nothing would look identical to a signed one at every
    later step."""
    s = _FakeStore()                                   # no keys_dir
    g.bootstrap(s)
    r = g.invoke(s, "op.operator.define", {"id": "op.t.unsigned", "kind": "composition",
                                           "spec": {"steps": [{"op": "op.consistency"}]}})["result"]
    assert r["signed"] is False
    assert r["signature_note"]
    assert opsign.verify_operator(s.artifacts.get_artifact("op.t.unsigned"))[0] is False


def test_redefining_resigns_and_the_old_signature_does_not_carry(tmp_path):
    s = _FakeStore()
    s.keys_dir = tmp_path
    g.bootstrap(s)
    g.invoke(s, "op.operator.define", {"id": "op.t.v", "kind": "composition",
                                       "spec": {"steps": [{"op": "op.consistency"}]}})
    first = dict(s.artifacts.get_artifact("op.t.v"))
    g.invoke(s, "op.operator.define", {"id": "op.t.v", "kind": "composition",
                                       "spec": {"steps": [{"op": "op.health"}]}})
    second = s.artifacts.get_artifact("op.t.v")
    assert second["signature"] != first["signature"], "the spec changed but the signature did not"
    # The content address is computed rather than stored: `evolution.spec_hash()` answers the
    # behaviour question on demand. The signature above already covers {id, kind, spec, requires,
    # effects}, a superset of what a stored digest would hold, so this is the narrower of the two
    # checks and is here because it names the address directly.
    from crystal.evolution import spec_hash
    assert spec_hash(second) != spec_hash(first), "the spec changed but the computed address did not"
    assert opsign.verify_operator(second)[0] is True
