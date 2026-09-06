"""Blinded region ids — deterministic for one person's devices, opaque to everyone else.

Grouped by invariant. Without a secret the region id is `f"{principal}/{collection}/{cluster}"`,
which carries the email in cleartext, a publicly derivable collection, and the cluster naming the
concept. With a secret it is a length-prefixed blind of those same components.
"""
from __future__ import annotations

import uuid

import pytest

from mantle.shard import region

P, C, K = "author@example.com", "c0ffee-collection", "anchor-42"


@pytest.fixture()
def secret(tmp_path):
    return region.principal_secret(tmp_path, create=True)


# ── Invariant 1: a read never mints a secret ──────────────────────────────────────────────────
def test_reading_an_absent_secret_returns_none(tmp_path):
    """Creation is explicit because minting on read would give each device its own secret, and
    each device would then derive different region ids for the same data while still working
    alone — a divergence with no symptom until two devices try to meet."""
    assert region.principal_secret(tmp_path) is None
    assert not (tmp_path / region.SECRET_FILENAME).exists()


def test_creating_is_explicit_and_stable(tmp_path):
    a = region.principal_secret(tmp_path, create=True)
    b = region.principal_secret(tmp_path)
    assert a and a == b


# ── Invariant 2: deterministic for the holder, opaque to everyone else ────────────────────────
def test_same_secret_same_id(secret):
    """Devices must agree, or a person's data fragments into per-device universes."""
    assert region.cell_region(P, C, K, secret=secret) == region.cell_region(P, C, K, secret=secret)


def test_a_different_secret_gives_a_different_id(tmp_path, secret):
    other = region.principal_secret(tmp_path / "other", create=True)
    assert region.cell_region(P, C, K, secret=other)[0] != region.cell_region(P, C, K, secret=secret)[0]


def test_the_principal_is_no_longer_in_the_id(secret):
    rid, blinded = region.cell_region(P, C, K, secret=secret)
    assert blinded is True
    assert P not in rid and C not in rid and K not in rid
    assert "@" not in rid


def test_the_legacy_id_really_did_leak_all_three():
    """With no secret the id is the unblinded form, which spells out all three components."""
    rid, blinded = region.cell_region(P, C, K)          # no secret
    assert blinded is False
    assert rid == "author@example.com/c0ffee-collection/anchor-42"
    assert P in rid and C in rid and K in rid


# ── Invariant 3: unlinkable — one person's regions must not be groupable ──────────────────────
def test_two_regions_of_one_person_share_no_prefix(secret):
    """Blinded ids of one person share no leading run. The unblinded form shares
    `principal/collection/`, which lets a whole footprint be counted at a glance without asking
    about any concept."""
    a, _ = region.cell_region(P, C, "anchor-1", secret=secret)
    b, _ = region.cell_region(P, C, "anchor-2", secret=secret)
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    assert shared <= 2, "blinded ids of one person are still linkable by prefix"


# ── Invariant 4: the encoding is unambiguous ──────────────────────────────────────────────────
def test_distinct_tuples_never_collide(secret):
    """Components are length-prefixed rather than delimiter-joined. `"|".join(parts)` encodes
    `("a|b","c")` and `("a","b|c")` identically, so two different regions would derive one id —
    two unrelated cells sharing a shard, which is a correctness fault before it is a privacy
    one."""
    assert region.blind(secret, "a|b", "c")[0] != region.blind(secret, "a", "b|c")[0]
    assert region.blind(secret, "a:b", "c")[0] != region.blind(secret, "a", "b:c")[0]
    assert region.blind(secret, "", "ab")[0] != region.blind(secret, "a", "b")[0]
    assert region.blind(secret, "a")[0] != region.blind(secret, "a", "")[0]


def test_component_order_matters(secret):
    assert region.blind(secret, "a", "b")[0] != region.blind(secret, "b", "a")[0]


# ── Invariant 5: the collection id keeps its shape ────────────────────────────────────────────
def test_blinded_collection_id_is_still_a_uuid(secret):
    cid, blinded = region.local_collection_id(P, secret=secret)
    assert blinded is True
    uuid.UUID(cid)                                       # must not raise


def test_unsalted_collection_id_matches_the_legacy_derivation():
    """Existing data stays addressable during migration."""
    from mantle.shard.local_collection import local_collection_id as legacy
    cid, blinded = region.local_collection_id(P)
    assert blinded is False and cid == legacy(P)


def test_a_collection_needs_a_principal():
    with pytest.raises(ValueError):
        region.local_collection_id("")


# ── Invariant 6: what is still exposed is reported, not forgotten ─────────────────────────────
def test_the_remaining_leak_is_declared():
    """Blinding the region id while the storage key still spells out the same three fields moves
    the leak rather than closing it, so `remaining_leak()` names what is still open."""
    r = region.remaining_leak()
    assert r["closed"] and r["open"]
    assert any("parse_cell_key" in o["where"] or "object keys" in o["where"] for o in r["open"])
    for o in r["open"]:
        assert o.get("leak")


# ── Invariant 7: provisioning across a principal's parties ────────────────────────────────────
def test_provision_installs_a_matching_secret(tmp_path):
    """A second device or peer takes the same secret from the first; a fresh mint would fragment
    routing between them."""
    origin = region.principal_secret(tmp_path / "a", create=True)
    peer_dir = tmp_path / "b"
    installed = region.provision_secret(peer_dir, origin.hex())
    assert installed == origin
    assert region.principal_secret(peer_dir) == origin


def test_provision_refuses_to_overwrite_a_different_secret(tmp_path):
    """Silently replacing it would re-key every cell this node routes to."""
    region.principal_secret(tmp_path, create=True)
    other = region.principal_secret(tmp_path / "other", create=True)
    with pytest.raises(ValueError, match="re-key"):
        region.provision_secret(tmp_path, other.hex())


def test_provision_is_idempotent_on_matching_bytes(tmp_path):
    s = region.principal_secret(tmp_path / "a", create=True)
    region.provision_secret(tmp_path / "b", s.hex())
    region.provision_secret(tmp_path / "b", s.hex())        # again, same bytes: accepted
    assert region.principal_secret(tmp_path / "b") == s


def test_provision_rejects_a_wrong_length_secret(tmp_path):
    with pytest.raises(ValueError, match="bytes"):
        region.provision_secret(tmp_path, "abcd")


# ── Invariant 8: the fingerprint identifies without exposing ──────────────────────────────────
def test_fingerprints_match_iff_secrets_match(tmp_path):
    """Compare across parties before enabling blinding: a mismatch means they will not route to
    each other."""
    a = region.principal_secret(tmp_path / "a", create=True)
    region.provision_secret(tmp_path / "b", a.hex())
    region.principal_secret(tmp_path / "c", create=True)     # a different secret
    fa = region.secret_fingerprint(tmp_path / "a")
    fb = region.secret_fingerprint(tmp_path / "b")
    fc = region.secret_fingerprint(tmp_path / "c")
    assert fa == fb and fa != fc


def test_fingerprint_is_none_when_unprovisioned(tmp_path):
    assert region.secret_fingerprint(tmp_path) is None


def test_fingerprint_does_not_expose_the_secret(tmp_path):
    s = region.principal_secret(tmp_path, create=True)
    fp = region.secret_fingerprint(tmp_path)
    assert s.hex() not in fp and len(fp) < len(s.hex())


def test_a_whitespace_valued_secret_survives_a_round_trip(tmp_path):
    """`os.urandom` produces arbitrary bytes, so about 4.7% of secrets carry a leading or trailing
    ASCII-whitespace byte. The file is read back verbatim: stripping would change those bytes, and
    the node would derive region ids its own devices do not share. The pathological byte patterns
    are forced here rather than waited for."""
    import os
    from mantle.shard import region as _r
    for evil in (b"\x20" + os.urandom(30) + b"\x0a",   # space ... newline
                 b"\x09" + os.urandom(30) + b"\x0d",   # tab ... CR
                 b"\x0b" + os.urandom(30) + b"\x20"):  # vtab ... space
        (tmp_path / _r.SECRET_FILENAME).write_bytes(evil)
        got = _r.principal_secret(tmp_path)
        assert got == evil, "a whitespace-valued secret was corrupted on read"
        # and it still derives a stable, correct region id
        a, _ = _r.cell_region("p", "c", "k", secret=got)
        b, _ = _r.cell_region("p", "c", "k", secret=evil)
        assert a == b


def test_a_wrong_length_secret_file_is_not_half_used(tmp_path):
    """A truncated/oversized file is not a usable secret — using it would derive ids no peer shares."""
    from mantle.shard import region as _r
    (tmp_path / _r.SECRET_FILENAME).write_bytes(b"too short")
    assert _r.principal_secret(tmp_path) is None
