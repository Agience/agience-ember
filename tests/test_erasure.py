"""Erasure — wipe one higgs, and nothing else.

The property under test is what stays. An erasure that takes the commons with it cannot be measured
afterwards, because the evidence is what went.
"""
import os
import tempfile

import pytest

from mantle.shard import erasure


def _lattice():
    from mantle.db import open_lattice
    return open_lattice(os.path.join(tempfile.mkdtemp(), "c.db"), origin="test")


def _put(L, aid, **kw):
    d = {"id": aid, "content_type": "text/x-wordnet", "state": "committed",
         "created_by": "u", "created_time": "2026-01-01T00:00:00+00:00"}
    d.update(kw)
    L.artifacts.put_artifact(d)


@pytest.fixture()
def store(monkeypatch):
    L = _lattice()
    monkeypatch.setattr(erasure, "_person_ids", lambda _s, p: {p})
    return L


def test_the_training_set_is_NOT_attached_to_whoever_ingested_it(store):
    """Stage-0 stamps every synset with the ingesting principal, so a `created_by == me` sweep
    would take the entire corpus. A row carrying a source citation was ingested rather than
    authored: it is the commons, reached and not grounded, so it attaches to nobody."""
    for i in range(50):
        _put(store, "wn-%d" % i, created_by="john", cited_from="cite.oewn",
             collection_id="stage.0.lexicon")
    inv = erasure.attached(store, "john")
    assert inv["total"] == 0, inv["counts"]


def test_what_IS_grounded_at_the_higgs_is_found(store):
    _put(store, "note-1", created_by="john", collection_id="private.john")
    _put(store, "convo.abc", created_by="john", content_type="application/vnd.agience.message+json")
    _put(store, "thought-1", created_by="john")
    inv = erasure.attached(store, "john")
    assert inv["counts"]["private"] == 1
    assert inv["counts"]["conversation"] == 1
    assert inv["counts"]["authored"] == 1


def test_another_persons_work_is_never_taken(store):
    _put(store, "hers-1", created_by="ada", collection_id="private.ada")
    _put(store, "mine-1", created_by="john", collection_id="private.john")
    inv = erasure.attached(store, "john")
    assert inv["found"]["private"] == ["mine-1"]


def test_RESET_and_ERASURE_are_different_acts_and_must_be_said(store):
    """Keeping the person and dropping what they made is a reset; taking the person too is an
    erasure. `include_identity` selects between them, so each act is asked for by name."""
    _put(store, "john", content_type="application/vnd.agience.person+json", created_by="john")
    _put(store, "note-1", created_by="john", collection_id="private.john")
    assert erasure.attached(store, "john")["counts"]["identity"] == 0
    assert erasure.attached(store, "john", include_identity=True)["counts"]["identity"] == 1


def test_DRY_RUN_IS_THE_DEFAULT(store):
    _put(store, "note-1", created_by="john", collection_id="private.john")
    r = erasure.erase(store, "john")
    assert r["applied"] is False
    assert store.artifacts.get_artifact("note-1") is not None      # still there


def test_apply_removes_and_reports_completeness(store):
    _put(store, "note-1", created_by="john", collection_id="private.john")
    _put(store, "wn-1", created_by="john", cited_from="cite.oewn", collection_id="stage.0.lexicon")
    r = erasure.erase(store, "john", apply=True)
    assert r["applied"] is True and r["removed"] == 1 and r["complete"] is True
    assert store.artifacts.get_artifact("note-1") is None
    assert store.artifacts.get_artifact("wn-1") is not None        # the commons survives


def test_a_partial_erasure_must_not_report_as_complete(store, monkeypatch):
    """A partial erasure reported as partial is recoverable; one reported as complete is not."""
    _put(store, "note-1", created_by="john", collection_id="private.john")
    _put(store, "note-2", created_by="john", collection_id="private.john")

    real = store.artifacts.delete_artifact

    def _flaky(aid):
        if aid == "note-2":
            raise RuntimeError("locked")
        return real(aid)

    monkeypatch.setattr(store.artifacts, "delete_artifact", _flaky)
    r = erasure.erase(store, "john", apply=True)
    assert r["complete"] is False and r["failed"]


def test_THE_REGISTRY_IS_NOT_ONE_PERSONS_WORK(store):
    """`op.math.add` and its siblings are stamped `created_by` whoever ran bootstrap, with no
    collection, no provenance and no citation, so a `created_by == me` sweep would take the
    arithmetic operators as private work.

    Operators, type registries, rungs, collections and citations are the ontology, grounded at the
    foundation (§13.11.7). They come back as `not_yours` rather than being skipped: naming what was
    found and left is more useful than silence."""
    _put(store, "op.math.add", created_by="john",
         content_type="application/vnd.agience.operator+json")
    _put(store, "etype.hypernym", created_by="john",
         content_type="application/vnd.agience.etype+json")
    _put(store, "note-1", created_by="john", collection_id="private.john")
    inv = erasure.attached(store, "john")
    assert inv["total"] == 1                                   # only the note
    assert set(inv["not_yours"]) == {"op.math.add", "etype.hypernym"}
    r = erasure.erase(store, "john", apply=True)
    assert store.artifacts.get_artifact("op.math.add") is not None
    assert store.artifacts.get_artifact("note-1") is None
