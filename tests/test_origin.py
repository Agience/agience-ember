"""Tests for the Origin artifact — the governing entity that owns an Authority (P5)."""
from __future__ import annotations

from origin import entity as origin
from _fakes import _FakeStore


# ── id shaping ────────────────────────────────────────────────────────────────────────────────────
def test_origin_id_from_name_is_urn():
    assert origin.origin_id("Ikailo") == "urn:agience:origin:ikailo"


def test_origin_id_url_used_verbatim():
    assert origin.origin_id("https://agience.ai") == "https://agience.ai"


def test_origin_id_urn_used_verbatim():
    assert origin.origin_id("urn:agience:origin:x") == "urn:agience:origin:x"


def test_origin_id_slug_strips_junk():
    assert origin.origin_id("My Origin!!") == "urn:agience:origin:my-origin"


# ── artifact shape ──────────────────────────────────────────────────────────────────────────────
def test_origin_artifact_references_its_authority():
    o = origin.origin_artifact("acme", issuer="https://iss.acme/")
    assert o["content_type"] == origin.ORIGIN_CONTENT_TYPE
    assert o["issuer"] == "https://iss.acme/"           # an Origin owns an Authority by reference
    assert o["id"] == "urn:agience:origin:acme"


def test_origin_artifact_is_human_provenance():
    # A person made the container on purpose; the container's own rung is human-authored.
    from ember import genesis
    o = origin.origin_artifact("acme", issuer="i")
    assert o["provenance"] == genesis.P_HUMAN


def test_origin_inherits_no_constants_at_all():
    """An Origin governs no constants: `DEFAULT_CONSTANTS` is empty and a fresh Origin's `economy`
    is too.

    The message/event switch is `mass.has_referent`, a partition with no threshold in it, so there
    is no quantity there for an Origin to set. A default carried here would be a value nobody
    measured, arriving as though it had been."""
    o = origin.origin_artifact("acme", issuer="i")
    assert origin.DEFAULT_CONSTANTS == {}
    assert o["economy"] == {}
    assert "electroweak_scale" not in origin.GOVERNABLE


def test_origin_economy_override_merges_over_defaults():
    o = origin.origin_artifact("acme", issuer="i", economy={"facilitation_fee": 0.05})
    assert o["economy"]["facilitation_fee"] == 0.05      # overridden
    for k, v in origin.DEFAULT_CONSTANTS.items():        # every default retained
        if k != "facilitation_fee":
            assert o["economy"][k] == v


def test_local_origin_owns_local_issuer():
    o = origin.local_origin_artifact()
    assert o["id"] == origin.LOCAL_ORIGIN
    assert o["issuer"] == origin.LOCAL_ISSUER


# ── registration + resolution: Origin → Authority ─────────────────────────────────────────────────
def test_register_and_resolve_authority():
    s = _FakeStore()
    iss = {"id": "https://iss.acme/", "content_type": origin.ISSUER_CONTENT_TYPE, "jwks": {}}
    s.artifacts.put_artifact(iss)
    origin.register_origin(s, "acme", issuer="https://iss.acme/")

    assert origin.authority_ref(s, "urn:agience:origin:acme") == "https://iss.acme/"
    got = origin.resolve_authority(s, "urn:agience:origin:acme")
    assert got is not None and got["id"] == "https://iss.acme/"


def test_authority_ref_never_guesses_for_unknown_origin():
    # An unknown origin resolves to no authority; there is no default one to fall back on.
    s = _FakeStore()
    assert origin.authority_ref(s, "urn:agience:origin:ghost") is None
    assert origin.resolve_authority(s, "urn:agience:origin:ghost") is None


def test_resolve_authority_none_when_issuer_artifact_missing():
    # Origin recorded, issuer artifact absent: there is nothing to validate against, so None.
    s = _FakeStore()
    origin.register_origin(s, "acme", issuer="https://iss.acme/")
    assert origin.authority_ref(s, "urn:agience:origin:acme") == "https://iss.acme/"
    assert origin.resolve_authority(s, "urn:agience:origin:acme") is None


def test_authority_ref_rejects_non_origin_artifact():
    # A row of another content type is not an Origin, so it carries no authority reference.
    s = _FakeStore()
    s.artifacts.put_artifact({"id": "x", "content_type": "text/plain", "issuer": "sneaky"})
    assert origin.authority_ref(s, "x") is None


# ── constants: the seam for per-origin physics ────────────────────────────────────────────────────
def test_constants_of_returns_defaults_for_unknown():
    s = _FakeStore()
    c = origin.constants_of(s, "urn:agience:origin:ghost")
    assert c == origin.DEFAULT_CONSTANTS
    assert c is not origin.DEFAULT_CONSTANTS       # a copy, not the shared dict


def test_constants_of_applies_overrides():
    s = _FakeStore()
    origin.register_origin(s, "acme", issuer="i", economy={"facilitation_fee": 0.05})
    c = origin.constants_of(s, "urn:agience:origin:acme")
    assert c["facilitation_fee"] == 0.05
    # Neither `demurrage_tau` nor `electroweak_scale` is a default: the 2nd-law clock is measured
    # off the node's screen and reaches `prism.demurrage` through the socket the runner fills at
    # boot, so a governed default would stand in for a measurement. The property under test is that
    # an unoverridden key keeps its default.
    assert "demurrage_tau" not in origin.DEFAULT_CONSTANTS
    assert "electroweak_scale" not in origin.DEFAULT_CONSTANTS
    for k, v in origin.DEFAULT_CONSTANTS.items():
        if k != "facilitation_fee":
            assert c[k] == v


def test_constants_of_ignores_unknown_and_nonnumeric_keys():
    s = _FakeStore()
    origin.register_origin(s, "acme", issuer="i")
    # tamper with a written artifact to include junk economy keys
    doc = s.artifacts.get_artifact("urn:agience:origin:acme")
    doc["economy"] = {"facilitation_fee": "not-a-number", "bogus": 1.0,
                      "demurrage_tau": 55.0, "salience_floor": 0.7}
    s.artifacts.put_artifact(doc)
    c = origin.constants_of(s, "urn:agience:origin:acme")
    assert "facilitation_fee" not in c                # non-numeric ignored, and there is NO default
    assert "bogus" not in c                           # unknown key rejected
    assert c["demurrage_tau"] == 55.0                 # valid override of a governable key applied
    assert "electroweak_scale" not in c               # not governable — cannot be re-added either
    # `salience_floor` is not governable either. The firing floor is a computed null on the hot
    # path, so an override here would put a chosen number back in front of a measured one.
    assert "salience_floor" not in c


# ── peers: exchange agreements default empty ──────────────────────────────────────────────────────
def test_peers_default_empty():
    s = _FakeStore()
    origin.register_origin(s, "acme", issuer="i")
    assert origin.peers_of(s, "urn:agience:origin:acme") == []


def test_peers_listed_when_present():
    s = _FakeStore()
    origin.register_origin(s, "acme", issuer="i", peers=["urn:agience:origin:beta", ""])
    assert origin.peers_of(s, "urn:agience:origin:acme") == ["urn:agience:origin:beta"]
