"""The access decision is grants, not flags: public is the un-keyed top, private is what a grant gates.

The decision is the lattice light-cone the mantle service uses, driven locally against ember's own
store. No `visibility`, `no_share`, `owner` or `private.` prefix enters it.
"""
import os
import tempfile

import pytest

from mantle.db import access

try:
    from mantle.db import open_lattice
except Exception:                       # pragma: no cover
    open_lattice = None

pytestmark = pytest.mark.skipif(open_lattice is None, reason="lattice store not importable")


def _store():
    d = tempfile.mkdtemp(prefix="access-test-")
    return open_lattice(os.path.join(d, "lat.db"), origin="node-71")


def test_public_is_the_ungated_top_no_grant_needed():
    """The shared corpus carries no grant, so it is public — readable by anyone, including an anonymous
    caller. 'No explicit grant' means public, not denied."""
    L = _store()
    L.artifacts.put_artifact({"id": "wn-dog", "content_type": "text/markdown",
                              "collection_id": "universe", "content": "x"})
    L.artifacts.put_artifact({"id": "loose", "content_type": "text/markdown", "content": "x"})

    dog = L.artifacts.get_artifact("wn-dog")
    assert access.is_public(L, dog) is True
    assert access.can_read(L, dog, principal="anyone") is True
    assert access.can_read(L, dog, principal=None) is True          # anonymous reads the commons
    assert access.is_public(L, L.artifacts.get_artifact("loose")) is True   # ungrounded == public


def test_a_grant_makes_a_collection_private_and_only_its_lightcone_reads_it():
    """Minting the owner's Read grant is what makes a collection private. Its members are then gated:
    public to nobody, readable by a principal whose light-cone includes the grant."""
    L = _store()
    # owner "alice" gets a private collection by minting a grant on it, not by a flag or prefix
    access.mint_owner_read_grant(L, "col.alice.notes", "alice")
    for i in range(3):
        L.artifacts.put_artifact({"id": "mem-%d" % i, "content_type": "text/markdown",
                                  "collection_id": "col.alice.notes", "content": "secret",
                                  "created_by": "alice"})           # authorship = provenance
    mem = L.artifacts.get_artifact("mem-0")

    assert access.gated_collections(L) == {"col.alice.notes"}
    assert access.is_public(L, mem) is False                        # gated -> not public
    assert access.can_read(L, mem, principal="alice") is True       # owner's light-cone reaches it
    assert access.can_read(L, mem, principal="bob") is False        # a stranger cannot
    assert access.can_read(L, mem, principal=None) is False         # anonymous cannot reach private


def test_public_and_private_coexist_and_the_decision_reads_no_flag():
    """A store with both: the public row is readable by all, the private row by its owner. The
    artifacts carry no visibility, no_share or owner flag — the decision is grants alone."""
    L = _store()
    L.artifacts.put_artifact({"id": "pub", "content_type": "text/markdown",
                              "collection_id": "subjects", "content": "shared"})
    access.mint_owner_read_grant(L, "col.bob.private", "bob")
    L.artifacts.put_artifact({"id": "priv", "content_type": "text/markdown",
                              "collection_id": "col.bob.private", "content": "owned",
                              "created_by": "bob"})

    pub, priv = L.artifacts.get_artifact("pub"), L.artifacts.get_artifact("priv")
    # no flags were written
    for a in (pub, priv):
        assert "visibility" not in a and "no_share" not in a and "owner" not in a
    assert access.can_read(L, pub, "carol") is True and access.can_read(L, pub, None) is True
    assert access.can_read(L, priv, "bob") is True
    assert access.can_read(L, priv, "carol") is False


def test_ensure_private_mints_a_grant_that_gates_the_owner_collection():
    """The write path makes a collection private by minting the owner's grant. After
    `_ensure_private`, `access` sees the collection as gated and the owner is the one principal
    whose light-cone reaches its members."""
    from ember import genesis
    L = _store()
    pid, _cite = genesis._ensure_private(L, "alice")
    assert pid == "private.alice"
    assert access.gated_collections(L) == {"private.alice"}          # a grant is what gates it
    L.artifacts.put_artifact({"id": "m1", "content_type": "text/markdown",
                              "collection_id": pid, "content": "secret", "created_by": "alice"})
    m = L.artifacts.get_artifact("m1")
    assert access.is_public(L, m) is False
    assert access.can_read(L, m, "alice") is True
    assert access.can_read(L, m, "bob") is False


def test_visible_to_uses_the_grant_lightcone_when_given_a_store():
    """The read gate is the light-cone. Given the store, `visible_to` decides by grants with no flag
    on the artifact: the owner sees, a stranger does not, and public is visible to all. Without a
    store it has only the artifact to judge by, and falls back to the flag gate."""
    from ember import genesis
    from mantle.db.access import visible_to
    L = _store()
    genesis._ensure_private(L, "alice")
    L.artifacts.put_artifact({"id": "s1", "content_type": "text/markdown",
                              "collection_id": "private.alice", "content": "secret",
                              "created_by": "alice"})          # no visibility/no_share flags
    s = L.artifacts.get_artifact("s1")
    assert visible_to(s, "alice", store=L) is True             # owner's light-cone reaches it
    assert visible_to(s, "bob", store=L) is False              # a stranger's does not — by grant
    L.artifacts.put_artifact({"id": "p1", "content_type": "text/markdown",
                              "collection_id": "universe", "content": "x"})
    assert visible_to(L.artifacts.get_artifact("p1"), "bob", store=L) is True   # public top
    # with no store there is only the artifact to read: the flag gate decides on its own
    assert visible_to({"id": "f", "no_share": True, "owner": "alice"}, "bob") is False


# ── Invoke: the discharge verb, closed by default ─────────────────────────────────────────────────
# Discharge happens only where the energy is granted access. These pin the polarity: Read is open by
# default (the public top) and Invoke is closed. Reusing the read light-cone for actuation would
# make every organon world-firable.

def test_invoke_is_CLOSED_by_default_unlike_read():
    """The load-bearing asymmetry: one public artifact, readable by anyone and invokable by nobody.
    No grant means public for Read and denied for Invoke."""
    L = _store()
    L.artifacts.put_artifact({"id": "crystal.pub", "content_type": "text/markdown",
                              "collection_id": "universe", "content": "x"})
    art = L.artifacts.get_artifact("crystal.pub")

    assert access.can_read(L, art, principal="anyone") is True      # open by default
    assert access.may_invoke(L, "crystal.pub", "anyone") is False   # closed by default
    assert access.may_invoke(L, "crystal.pub", None) is False       # anonymous fires nothing
    assert access.invokable_resources(L, "anyone") == set()


def test_an_invoke_grant_is_what_permits_firing():
    L = _store()
    access.grant_invoke(L, "crystal.sensing", "alice", "alice")
    assert access.may_invoke(L, "crystal.sensing", "alice") is True
    assert access.may_invoke(L, "crystal.sensing", "bob") is False   # not granted
    assert access.may_invoke(L, "crystal.other", "alice") is False   # granted narrowly


def test_a_READ_grant_does_NOT_imply_invoke():
    """Seeing a crystal and firing it are separate verbs with separate grants, which is why Invoke
    is not folded into the read light-cone."""
    L = _store()
    access.grant_read(L, "crystal.sensing", "alice", "alice")
    assert access.may_invoke(L, "crystal.sensing", "alice") is False


def test_an_INVOKE_grant_adds_no_READ_reach_to_a_private_collection():
    """The converse: the right to fire something hands over none of its contents.

    The collection is made private by a read grant first. An invoke-only grant gates nothing for
    reading, so without bob's grant `col.locked` would be public top and readable by all — gating
    is a read-grant question."""
    L = _store()
    access.mint_owner_read_grant(L, "col.locked", "bob")          # bob owns it -> now private
    L.artifacts.put_artifact({"id": "inner", "content_type": "text/markdown",
                              "collection_id": "col.locked", "content": "secret",
                              "created_by": "bob"})
    access.grant_invoke(L, "col.locked", "alice", "bob")          # alice may fire it, not read it

    inner = L.artifacts.get_artifact("inner")
    assert access.may_invoke(L, "col.locked", "alice") is True    # she can fire
    assert access.can_read(L, inner, principal="alice") is False  # ...and still cannot read
    assert access.can_read(L, inner, principal="bob") is True     # the owner still can


def test_discharge_authority_answers_crystal_by_content_address():
    """The duck-typed bridge crystal asks. A grant is minted against the crystal, so the authority
    answers on its sha, and an ungranted principal does not discharge."""
    L = _store()
    access.grant_invoke(L, "sha-of-crystal", "alice", "alice")
    auth = access.DischargeAuthority(L)

    assert auth.may_discharge("alice", "op.sense", "sha-of-crystal") is True
    assert auth.may_discharge("bob", "op.sense", "sha-of-crystal") is False
    assert auth.may_discharge(None, "op.sense", "sha-of-crystal") is False   # no provenance, no fire
    assert auth.may_discharge("alice", "op.sense", "sha-of-other") is False


def test_discharge_authority_also_accepts_a_narrow_organon_grant():
    """Granting one organon directly, rather than the whole crystal, still works."""
    L = _store()
    access.grant_invoke(L, "op.sense", "alice", "alice")
    auth = access.DischargeAuthority(L)
    assert auth.may_discharge("alice", "op.sense", "sha-of-crystal") is True
    assert auth.may_discharge("alice", "op.other", "sha-of-crystal") is False
