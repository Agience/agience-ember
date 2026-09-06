"""A non-person author resolves to a foundation entity.

`ember-source` and `ember-local` are processes rather than people, and between them they authored
the 200,000+ rows (~98% of the corpus) that node-repair's `created_by resolves` check reads.
`mantle.services.principal.person_id` resolves them to a foundation entity, so `created_by` is a
column whose every value points at a vertex.

The negative controls carry as much weight as the positive one: an id that resolves is worthless if
it resolves to a *person*. Two of these tests exist to establish that a foundation entity is a
distinct kind of principal, readable as such from the row.
"""
from __future__ import annotations

import uuid

import pytest

from mantle.services import principal as person


PROCESSES = ("ember-source", "ember-local")


def test_person_id_resolves_a_process_author_to_the_foundation_entity_and_does_not_raise():
    """`person_id("ember-source")` returns the foundation entity's id: a real vertex id, not the
    raw claim string (contract §2.1).
    """
    for proc in PROCESSES:
        fid = person.person_id(proc)                       # resolves
        assert fid == person.foundation_id(proc)
        assert uuid.UUID(fid)                              # a real vertex id, not a bare string
        assert fid != proc, "the raw claim is not a vertex reference (contract §2.1)"


def test_the_foundation_id_is_the_same_derivation_under_a_different_issuer():
    """Every observer computes the id the same way, which is what makes `created_by` a column.

    The derivation is asserted exactly — `uuid5(_USER_NS, issuer\\nsub)` — so a second derivation
    (a hash of the name, a random uuid4, a literal) does not satisfy it.
    """
    for proc in PROCESSES:
        assert person.foundation_id(proc) == str(uuid.uuid5(
            person._USER_NS, "%s\n%s" % (person.FOUNDATION_ISSUER, proc)))
    # Deterministic, not random — two calls agree.
    assert person.foundation_id("ember-source") == person.foundation_id("ember-source")
    # Distinct processes stay distinct: collapsing them would destroy the only WHO those rows have.
    assert person.foundation_id("ember-source") != person.foundation_id("ember-local")


def test_a_foundation_entity_can_never_collide_with_a_person():
    """A negative control. The issuer is inside the hash, so one subject under the two issuers is
    two vertices, and a foundation id is distinguishable from a human's.

    Equal issuers, or an issuer dropped from the hash, would make the two ids identical.
    """
    assert person.FOUNDATION_ISSUER != person.LOCAL_ISSUER
    for sub in PROCESSES + ("author@example.com",):
        assert person.foundation_id(sub) != str(uuid.uuid5(
            person._USER_NS, "%s\n%s" % (person.LOCAL_ISSUER, sub)))
    # And a real person keeps the person derivation.
    assert person.person_id("author@example.com") == str(uuid.uuid5(
        person._USER_NS, "%s\n%s" % (person.LOCAL_ISSUER, "author@example.com")))


def test_the_foundation_artifact_is_visibly_not_a_human_person():
    """A negative control. The artifact carries `FOUNDATION_CONTENT_TYPE`, so a reader can tell a
    foundation entity from a person by looking at the row rather than at the id.
    """
    for proc in PROCESSES:
        doc = person.foundation_artifact(proc)
        assert doc["content_type"] == person.FOUNDATION_CONTENT_TYPE
        assert doc["content_type"] != person.PERSON_CONTENT_TYPE
        assert doc["context"]["kind"] == "foundation"
        assert doc["context"]["issuer"] == person.FOUNDATION_ISSUER
        assert doc["context"]["sub"] == proc
        # Self-attesting at the root of its own chain, so it resolves with no prior vertex to
        # point at and needs no bootstrap of its own.
        assert doc["id"] == person.foundation_id(proc) == doc["created_by"]


def test_a_foundation_vertex_carries_only_universal_fields():
    """§2.1 consequence 4 / §7.4 apply to any shared vertex, not only to people.

    Trust weight, a reputation score and observation counts are one observer's reading. Carried on
    a shared vertex, two observers hold different values under one id, and no clock orders them.
    """
    doc = person.foundation_artifact("ember-source")
    assert set(doc["context"]) <= {"kind", "issuer", "sub"}, (
        "a foundation vertex grew a field: %s" % sorted(doc["context"]))
    for banned in ("trust", "trust_weight", "reputation", "observations",
                   "observation_count", "grants", "fitness", "mass", "provenance", "cited_from"):
        assert banned not in doc["context"] and banned not in doc


def test_principal_artifact_dispatches_on_what_the_subject_actually_is():
    """One call for a writer, with the two kinds told apart in exactly one place.

    Each branch asserts both the type it carries and the type it does not, so an inverted dispatch
    shows up on either side.
    """
    proc = person.principal_artifact("ember-source")
    assert proc["content_type"] == person.FOUNDATION_CONTENT_TYPE
    assert proc["id"] == person.person_id("ember-source")

    human = person.principal_artifact("author@example.com")
    assert human["content_type"] == person.PERSON_CONTENT_TYPE
    assert human["content_type"] != person.FOUNDATION_CONTENT_TYPE
    assert human["id"] == person.person_id("author@example.com")


def test_an_empty_subject_still_raises_because_there_is_nothing_to_identify():
    """An empty claim names no principal, so there is nothing to derive an id from and `ValueError`
    says which input was empty. Minting a vertex for "" would create a single WHO that every
    unattributed row shares.
    """
    with pytest.raises(ValueError):
        person.person_id("")
    with pytest.raises(ValueError):
        person.foundation_id("")
