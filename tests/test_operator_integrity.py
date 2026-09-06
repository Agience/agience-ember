"""Operator integrity — an operator's evidence, its version, and its bounds.

Grouped by invariant: fitness is evidence about a behaviour and is keyed to it; one logical
invocation runs one version throughout; a composition is bounded and says why it stopped; and the
values a composition's author fixed are part of its definition.

These are the prerequisites for the declarative-spec work (AGENT-HOST-DESIGN.md D10 / Phase 3).
Expressiveness rests on a spec that already versions and scores correctly, and `op.operator.define`
is reachable over HTTP, so what can be defined is what any caller can define.
"""
from __future__ import annotations

import pytest

from ember import genesis as g
from ember.runtime.runner import evolution
from _fakes import _FakeStore


# ── Invariant 1: fitness is evidence about a behaviour, keyed to the behaviour ────────────────
def test_spec_hash_is_deterministic_and_content_addressed():
    a = {"id": "op.x", "kind": "composition", "spec": {"steps": [{"op": "op.a"}]}}
    b = {"id": "op.DIFFERENT-NAME", "kind": "composition", "spec": {"steps": [{"op": "op.a"}]}}
    c = {"id": "op.x", "kind": "composition", "spec": {"steps": [{"op": "op.b"}]}}
    assert evolution.spec_hash(a) == evolution.spec_hash(b), "the hash depends on the NAME"
    assert evolution.spec_hash(a) != evolution.spec_hash(c), "a spec change did not change the hash"
    # key order does not affect the hash
    d1 = {"kind": "source", "spec": {"repo": "r", "config": "c"}}
    d2 = {"kind": "source", "spec": {"config": "c", "repo": "r"}}
    assert evolution.spec_hash(d1) == evolution.spec_hash(d2)


def test_spec_hash_is_none_without_executable_content():
    """Code-backed operators registered by name carry no spec, so re-registration is the same
    operator and its fitness carries with it."""
    assert evolution.spec_hash({"id": "op.health"}) is None


def test_fitness_carries_across_identical_reregistration():
    """The behaviour this helper exists for: a fresh process re-registering an identical operator
    keeps what that operator has learned."""
    s = _FakeStore()
    doc = {"id": "op.k", "kind": "composition", "spec": {"steps": [{"op": "op.a"}]},
           "verified": 500, "invocations": 900}
    s.artifacts.put_artifact(evolution.preserve_fitness(s.artifacts, dict(doc)))
    again = evolution.preserve_fitness(s.artifacts, {"id": "op.k", "kind": "composition",
                                                     "spec": {"steps": [{"op": "op.a"}]}})
    assert again["verified"] == 500 and again["invocations"] == 900
    assert "spec_change" not in again


def test_fitness_does_not_survive_a_spec_change():
    """A spec change discards the fitness fields, because the evidence was gathered about the old
    behaviour. Carried across, `verified: 500` earned under one spec would rank an entirely
    different behaviour, and an edit would buy a score instead of earning one."""
    s = _FakeStore()
    s.artifacts.put_artifact(evolution.preserve_fitness(s.artifacts, {
        "id": "op.k", "kind": "composition", "spec": {"steps": [{"op": "op.harmless"}]},
        "verified": 500, "refuted": 0, "invocations": 900}))

    redefined = evolution.preserve_fitness(s.artifacts, {
        "id": "op.k", "kind": "composition", "spec": {"steps": [{"op": "op.something.else"}]}})

    for f in evolution.FITNESS_FIELDS:
        assert f not in redefined, f"{f} was laundered across a spec change"
    assert evolution.fitness(redefined) == pytest.approx(0.4)     # the agnostic prior, not 0.8+


def test_a_discarded_fitness_reset_is_recorded_not_silent():
    """The reset writes `spec_change` carrying what was discarded, so a zero score reads as
    "evidence dropped at this spec change" rather than as an operator nobody ever exercised."""
    s = _FakeStore()
    s.artifacts.put_artifact(evolution.preserve_fitness(s.artifacts, {
        "id": "op.k", "kind": "source", "spec": {"repo": "a", "config": "x"},
        "verified": 7, "invocations": 9}))
    redefined = evolution.preserve_fitness(s.artifacts, {
        "id": "op.k", "kind": "source", "spec": {"repo": "b", "config": "x"}})
    assert redefined["spec_change"]["discarded"] == {"verified": 7, "invocations": 9}
    assert redefined["spec_change"]["from"] != redefined["spec_change"]["to"]


def test_citation_still_survives_a_spec_change():
    """§12: an artifact keeps its citation across re-registration. A citation records where the
    artifact came from, which a spec change does not alter, so it survives the fitness reset."""
    s = _FakeStore()
    s.artifacts.put_artifact(evolution.preserve_fitness(s.artifacts, {
        "id": "op.k", "kind": "source", "spec": {"repo": "a"},
        "cited_from": g.CITE_GENESIS, "provenance": g.P_HUMAN, "verified": 3}))
    redefined = evolution.preserve_fitness(s.artifacts, {"id": "op.k", "kind": "source",
                                                         "spec": {"repo": "b"}})
    assert redefined["cited_from"] == g.CITE_GENESIS
    assert redefined["provenance"] == g.P_HUMAN
    assert "verified" not in redefined


def test_define_operator_resets_fitness_end_to_end():
    s = _FakeStore()
    g.bootstrap(s)
    g.invoke(s, "op.operator.define", {"id": "op.t.x", "kind": "composition",
                                       "spec": {"steps": [{"op": "op.consistency"}]}})
    op = s.artifacts.get_artifact("op.t.x")
    op["verified"] = 500
    s.artifacts.put_artifact(op)

    g.invoke(s, "op.operator.define", {"id": "op.t.x", "kind": "composition",
                                       "spec": {"steps": [{"op": "op.health"}]}})
    assert s.artifacts.get_artifact("op.t.x").get("verified") in (None, 0)


# ── Invariant 2: one logical invocation runs one version throughout ───────────────────────────
def test_a_redefinition_midflight_does_not_change_the_running_invocation():
    """An invocation resolves each operator once and runs that version to the end.

    Re-reading the operator artifact per composition step lets a redefinition landing mid-flight
    change behaviour between steps of one invocation — step 1 on one version, step 3 on another —
    and the result then describes neither."""
    s = _FakeStore()
    g.bootstrap(s)
    # inner: one step
    g.invoke(s, "op.operator.define", {"id": "op.t.inner", "kind": "composition",
                                       "spec": {"steps": [{"op": "op.consistency"}]}})
    # outer: inner -> redefine inner to TWO steps -> inner again
    g.invoke(s, "op.operator.define", {"id": "op.t.outer", "kind": "composition", "spec": {"steps": [
        {"op": "op.t.inner"},
        {"op": "op.operator.define", "args": {"id": "op.t.inner", "kind": "composition",
                                              "spec": {"steps": [{"op": "op.consistency"},
                                                                 {"op": "op.consistency"}]}}},
        {"op": "op.t.inner"},
    ]}})

    steps = g.invoke(s, "op.t.outer", {})["result"]["steps"]
    first_inner = steps[0]["result"]["steps"]
    last_inner = steps[2]["result"]["steps"]
    assert len(first_inner) == 1
    assert len(last_inner) == 1, "the redefinition took effect mid-invocation"
    # and the redefinition is visible to the next invocation
    assert len(g.invoke(s, "op.t.inner", {})["result"]["steps"]) == 2


# ── Invariant 3: a composition is bounded, and says so ──────────────────────────────────────────────
def test_self_referencing_composition_is_refused_not_recursed():
    """A composition that names itself stops with an error naming the cycle. `op.operator.define` is
    reachable over HTTP, so an unbounded composition is definable by any caller; an error that names
    the cycle tells that caller what to change, where a stack overflow tells them nothing."""
    s = _FakeStore()
    g.bootstrap(s)
    g.invoke(s, "op.operator.define", {"id": "op.t.loop", "kind": "composition",
                                       "spec": {"steps": [{"op": "op.t.loop"}]}})
    r = g.invoke(s, "op.t.loop", {})
    inner = r["result"]["steps"][0]
    assert "cycle" in (inner.get("error") or ""), inner


def test_indirect_cycle_is_refused():
    s = _FakeStore()
    g.bootstrap(s)
    g.invoke(s, "op.operator.define", {"id": "op.t.a", "kind": "composition",
                                       "spec": {"steps": [{"op": "op.t.b"}]}})
    g.invoke(s, "op.operator.define", {"id": "op.t.b", "kind": "composition",
                                       "spec": {"steps": [{"op": "op.t.a"}]}})
    r = g.invoke(s, "op.t.a", {})
    err = r["result"]["steps"][0]["result"]["steps"][0].get("error") or ""
    assert "cycle" in err, r


def test_depth_is_bounded(monkeypatch):
    s = _FakeStore()
    g.bootstrap(s)
    monkeypatch.setattr(g, "_MAX_COMPOSITION_DEPTH", 3)
    # a -> b -> c -> d, each a distinct id, so what is measured is depth rather than a cycle
    for name, nxt in (("op.t.d1", "op.consistency"), ("op.t.c1", "op.t.d1"),
                      ("op.t.b1", "op.t.c1"), ("op.t.a1", "op.t.b1")):
        g.invoke(s, "op.operator.define", {"id": name, "kind": "composition",
                                           "spec": {"steps": [{"op": nxt}]}})
    r = g.invoke(s, "op.t.a1", {})
    flat = repr(r)
    assert "nested deeper than 3" in flat, r


# ── Invariant 4: the composition author's fixed values are part of the definition ─────────────
def test_step_args_win_over_caller_args():
    """A step's own args win over the caller's, and caller args fill the slots the author left open.

    A value the author fixed is part of what the composition is: merged the other way round, a
    caller passing `limit=999` to a composition whose author pinned `limit=5` changes the
    definition, and the composition's name no longer describes what runs."""
    s = _FakeStore()
    g.bootstrap(s)
    seen = {}

    def _spy(store, args):
        seen.update(args)
        return {"ingested": 0}

    monkey = dict(g.SOURCE_INGESTERS)
    monkey["op.source.spy"] = _spy
    g.SOURCE_INGESTERS.update(monkey)
    try:
        g.invoke(s, "op.operator.define", {"id": "op.t.fixed", "kind": "composition", "spec": {
            "steps": [{"op": "op.source.spy", "args": {"limit": 5}}]}})
        g.invoke(s, "op.t.fixed", {"limit": 999, "extra": "passed-through"})
        assert seen["limit"] == 5, "the caller clobbered the author's fixed value"
        assert seen["extra"] == "passed-through", "caller args no longer fill open slots"
    finally:
        g.SOURCE_INGESTERS.pop("op.source.spy", None)
