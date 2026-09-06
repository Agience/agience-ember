"""Curation — collections, ownership, and moving things in and out.

The properties under test are the limits: the commons stays out of a private collection, and every
row keeps a home. A curation surface without those is a data-loss surface with a friendly name.
"""
import os
import tempfile

import pytest

from mantle.shard import curate


def _lattice():
    from mantle.db import open_lattice
    return open_lattice(os.path.join(tempfile.mkdtemp(), "c.db"), origin="test")


def _put(L, aid, **kw):
    d = {"id": aid, "content_type": "text/markdown", "state": "committed",
         "created_by": "john", "created_time": "2026-01-01T00:00:00+00:00"}
    d.update(kw)
    L.artifacts.put_artifact(d)
    return d


@pytest.fixture()
def store():
    return _lattice()


def test_the_three_grounding_layers_are_reported_not_decided(store):
    assert curate.owner_of({"content_type": "application/vnd.agience.operator+json"})[0] == "foundation"
    assert curate.owner_of({"cited_from": "cite.oewn"})[0] == "commons"
    assert curate.owner_of({"collection_id": "stage.0.lexicon"})[0] == "commons"
    assert curate.owner_of({"created_by": "john"}) == ("observer", "john")


def test_THE_COMMONS_CANNOT_BE_CURATED_INTO_A_PRIVATE_COLLECTION(store):
    """Observing does not confer disposal. Filing a WordNet synset into your own collection would
    make the commons look owned, and every downstream grounding question would then read wrong."""
    _put(store, "wn-1", cited_from="cite.oewn", collection_id="stage.0.lexicon")
    r = curate.file_into(store, "wn-1", "private.john", person="john", apply=True)
    assert r["ok"] is False and "commons" in r["why"]
    assert store.artifacts.get_artifact("wn-1").get("collections") in (None, [])


def test_THE_ONTOLOGY_IS_NOT_ONE_PERSONS_TO_MOVE(store):
    """`op.math.add` carries `created_by` whoever ran bootstrap, so an authorization written as "you
    may move what you created" would hand the arithmetic operators to a random principal. Authorship
    is not ownership when the thing authored is the ontology."""
    _put(store, "op.math.add", content_type="application/vnd.agience.operator+json",
         created_by="john")
    ok, why = curate.may_curate(store, store.artifacts.get_artifact("op.math.add"), "john")
    assert ok is False and "foundation" in why


def test_another_persons_work_is_refused_with_a_reason(store):
    _put(store, "hers", created_by="ada")
    ok, why = curate.may_curate(store, store.artifacts.get_artifact("hers"), "john")
    assert ok is False and "ada" in why


def test_filing_and_unfiling_your_own_work(store):
    _put(store, "mine", created_by="john", collection_id="private.john")
    r = curate.file_into(store, "mine", "reading-list", person="john", apply=True)
    assert r["ok"] and "reading-list" in r["memberships"]
    r = curate.unfile(store, "mine", "reading-list", person="john", apply=True)
    assert r["ok"] and "reading-list" not in r["memberships"]
    assert store.artifacts.get_artifact("mine") is not None      # un-filed, not deleted


def test_UNFILING_THE_HOME_COLLECTION_IS_REFUSED(store):
    """A row with no home is unreachable by every listing that pages by collection — data loss
    wearing the clothes of a tidy-up. `move` exists for this, and it is one act."""
    _put(store, "mine", created_by="john", collection_id="private.john")
    r = curate.unfile(store, "mine", "private.john", person="john", apply=True)
    assert r["ok"] is False and "homeless" in r["why"]


def test_move_changes_the_home_in_one_act(store):
    _put(store, "mine", created_by="john", collection_id="private.john")
    r = curate.move(store, "mine", "shared.notes", person="john", apply=True)
    assert r["ok"] and r["from"] == "private.john" and r["to"] == "shared.notes"
    a = store.artifacts.get_artifact("mine")
    assert a["collection_id"] == "shared.notes"


def test_DRY_RUN_IS_THE_DEFAULT_HERE_TOO(store):
    _put(store, "mine", created_by="john", collection_id="private.john")
    r = curate.file_into(store, "mine", "reading-list", person="john")
    assert r["applied"] is False
    assert "reading-list" not in (store.artifacts.get_artifact("mine").get("collections") or [])


def test_an_UNDECLARED_collection_that_rows_belong_to_is_still_listed(store):
    """Membership is a plain field, so an artifact can name a collection nobody minted —
    `private.<x>` is created exactly this way. Listing only declared collections hides where things
    actually are."""
    _put(store, "mine", created_by="john", collection_id="private.john")
    cols = {c["id"]: c for c in curate.collections(store)}
    assert "private.john" in cols
    assert cols["private.john"]["declared"] is False and cols["private.john"]["count"] == 1


def test_members_reports_each_members_layer_and_holder(store):
    _put(store, "mine", created_by="john", collection_id="shared.notes")
    _put(store, "wn-1", cited_from="cite.oewn", collections=["shared.notes"])
    m = curate.members(store, "shared.notes")
    layers = {i["id"]: i["layer"] for i in m["items"]}
    assert m["total"] == 2 and layers["mine"] == "observer" and layers["wn-1"] == "commons"


def test_A_PRIVATE_COLLECTION_OUTRANKS_ANY_CITATION(store):
    """A grant-gated collection is the strongest statement of grounding there is, so it is checked
    first and no citation overrides it. Ownership is grant-derived — the grantee — rather than read
    off the `private.` name.

    A citation is not always a source. `cite.genesis` is the system's own provenance anchor, so a
    person's conversation in their private collection carries one while remaining theirs; reading
    any citation as evidence of commons would make that memory un-curatable by its author."""
    from mantle.db import access
    access.mint_owner_read_grant(store, "private.john", "john")   # the grant makes it private and owned
    _put(store, "convo.x", collection_id="private.john", cited_from="cite.genesis",
         created_by="ada", content_type="application/vnd.agience.message+json")   # note: created_by != owner
    gated = access.gated_owner_map(store)
    layer, who = curate.owner_of(store.artifacts.get_artifact("convo.x"), gated)
    assert (layer, who) == ("observer", "john")                  # owner = the grantee, not created_by
    ok, _why = curate.may_curate(store, store.artifacts.get_artifact("convo.x"), "john")
    assert ok is True


def test_a_real_source_citation_still_means_commons(store):
    _put(store, "wn-9", cited_from="cite.oewn", collection_id="stage.0.lexicon")
    assert curate.owner_of(store.artifacts.get_artifact("wn-9"))[0] == "commons"
