"""Ember's local collection is an artifact, and its id is derived — not named.

Everything is an artifact; a collection is an artifact with the right edges. A leaf that
invented a collection *name* would compute region ids corresponding to no artifact — routable,
plausible, shareable with nobody, and silently wrong. These tests pin that shut.
"""
from __future__ import annotations


import pytest

from ember import COLLECTION_CONTENT_TYPE, local_collection_artifact, local_collection_id
from ember.config import load

ALICE = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
BOB = "1341c674-d22c-4e25-bcec-2a1e4fa96fef"


def test_collection_is_a_real_artifact_shaped_for_mantle() -> None:
    """It carries Mantle's discriminator and Mantle's field shape, so no translation layer
    exists to drift: the leaf and the server describe the same object."""
    a = local_collection_artifact(ALICE, "My Stuff")
    assert a["content_type"] == COLLECTION_CONTENT_TYPE == "application/vnd.agience.collection+json"
    assert a["id"] and a["context"] and "content" in a
    assert a["context"]["principal"] == ALICE


def test_same_person_two_devices_agree() -> None:
    """The property that makes sync possible. A laptop and a desktop derive the same local
    collection id, so one person's data stays one collection. If they diverged, each device would
    work perfectly alone and the split would only show up when sync was expected to work.
    """
    laptop = local_collection_id(ALICE)
    desktop = local_collection_id(ALICE)
    assert laptop == desktop, "one person's devices must compute identical region ids"


def test_different_people_do_not_collide() -> None:
    assert local_collection_id(ALICE) != local_collection_id(BOB)


def test_id_is_stable_across_versions() -> None:
    """The derivation is a wire constant: region ids are built from it, and shards already
    exist under those ids. If this value ever changes, every leaf silently re-homes its data
    into cells nobody else routes to. Pinned deliberately — changing it is a migration."""
    assert local_collection_id(ALICE) == "698fe8b1-53c4-51ba-8960-2e687c0ba649"


def test_a_principal_is_required() -> None:
    """A collection belongs to someone. Deriving one from nothing would hand every anonymous
    leaf the same id and silently merge strangers' data."""
    with pytest.raises(ValueError):
        local_collection_id("")


def test_config_derives_an_artifact_id_not_a_name(monkeypatch) -> None:
    """`config.load()` derives `collection_id` from the principal rather than defaulting to a
    name. A name such as "default" is not an artifact id, so it would route to cells that hold
    nothing."""
    monkeypatch.setenv("EMBER_PRINCIPAL", ALICE)
    monkeypatch.delenv("EMBER_COLLECTION_ID", raising=False)
    s = load()
    assert s.collection_id == local_collection_id(ALICE)
    assert s.collection_id != "default"


def test_explicit_collection_id_wins(monkeypatch) -> None:
    """Joining someone else's collection is just supplying its artifact id."""
    monkeypatch.setenv("EMBER_PRINCIPAL", ALICE)
    monkeypatch.setenv("EMBER_COLLECTION_ID", "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d")
    assert load().collection_id == "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"


def test_collection_provenance_describes_the_container_only() -> None:
    """A person deliberately made this container — that is a real claim about its origin.
    It says nothing about whether the contents are true; those carry their own channels.
    """
    from prism.mass import Provenance, has_referent, provenance_of

    a = local_collection_artifact(ALICE)
    assert provenance_of(a) is Provenance.HUMAN_VALIDATED
    assert has_referent(provenance_of(a))    # a person staked it — there is someone to ask
