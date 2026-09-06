"""Cuddler Process — the human-in-the-loop interface behind `human.ask`.

Grouped by invariant. A questionnaire is a typed state machine: it validates before it runs, it
enforces each answer's type, it routes by the answer, and a script asset stays data.
"""
from __future__ import annotations

import pytest

from ember.runtime import cuddler
from ember.runtime import capability


def _q(qid, atype, out, routing=None, **extra):
    d = {"questionId": qid, "title": qid.upper(), "answerType": atype, "outputKey": out,
         "routing": routing or {}}
    d.update(extra)
    return d


def _proc(*questions, entry=None, qid="p"):
    return {"questionnaireId": qid, "title": "T",
            "entryQuestionId": entry or questions[0]["questionId"],
            "questions": list(questions)}


# ── Invariant 1: validation before execution, with citable rule ids ───────────────────────────
def test_a_valid_questionnaire_loads():
    p = cuddler.Process.load(_proc(_q("a", "short-text", "a")))
    assert p.entryQuestionId == "a" and "a" in p.questions


def test_duplicate_question_ids_are_rejected():
    with pytest.raises(cuddler.ProcessError) as e:
        cuddler.Process.load(_proc(_q("a", "short-text", "a"), _q("a", "number", "b")))
    assert any(d.ruleId == "AUT-SEM-012" for d in e.value.diagnostics)


def test_entry_must_resolve():
    with pytest.raises(cuddler.ProcessError) as e:
        cuddler.Process.load(_proc(_q("a", "short-text", "a"), entry="nope"))
    assert any(d.ruleId == "AUT-SEM-011" for d in e.value.diagnostics)


def test_a_dangling_route_is_rejected():
    with pytest.raises(cuddler.ProcessError) as e:
        cuddler.Process.load(_proc(_q("a", "short-text", "a", {"nextQuestionId": "ghost"})))
    assert any(d.ruleId == "AUT-SEM-013" for d in e.value.diagnostics)


def test_select_without_options_is_rejected():
    with pytest.raises(cuddler.ProcessError):
        cuddler.Process.load(_proc(_q("a", "select-one", "a")))


def test_unknown_answer_type_is_rejected():
    with pytest.raises(cuddler.ProcessError):
        cuddler.Process.load(_proc(_q("a", "telepathy", "a")))


def test_diagnostics_carry_the_cuddler_shape():
    try:
        cuddler.Process.load(_proc(_q("a", "short-text", "a"), entry="nope"))
    except cuddler.ProcessError as e:
        d = e.diagnostics[0].as_dict()
        assert set(("message", "ruleId", "severity", "artifactPath")).issubset(d)


# ── Invariant 2: answers are typed — a bad answer is a diagnostic, not a coercion ────────────
def test_number_rejects_non_numeric():
    _, d = cuddler.coerce_answer(_q("a", "number", "a"), "seven")
    assert d is not None


def test_number_accepts_numeric_strings():
    v, d = cuddler.coerce_answer(_q("a", "number", "a"), "42")
    assert d is None and v == 42.0


def test_percentage_is_range_checked():
    assert cuddler.coerce_answer(_q("a", "percentage", "a"), 150)[1] is not None
    assert cuddler.coerce_answer(_q("a", "percentage", "a"), 42)[0] == 42.0


def test_date_must_be_iso_and_is_not_invented():
    assert cuddler.coerce_answer(_q("a", "date", "a"), "not-a-date")[1] is not None
    v, d = cuddler.coerce_answer(_q("a", "date", "a"), "2026-07-21")
    assert d is None and v == "2026-07-21"


def test_select_one_must_be_an_option():
    q = _q("a", "select-one", "a", options=["x", "y"])
    assert cuddler.coerce_answer(q, "z")[1] is not None
    assert cuddler.coerce_answer(q, "x")[0] == "x"


def test_select_many_validates_every_choice_and_dedupes():
    q = _q("a", "select-many", "a", options=["x", "y", "z"])
    assert cuddler.coerce_answer(q, ["x", "bad"])[1] is not None
    v, d = cuddler.coerce_answer(q, ["x", "y", "x"])
    assert d is None and v == ["x", "y"]


def test_select_many_rejects_a_bare_string():
    q = _q("a", "select-many", "a", options=["x"])
    assert cuddler.coerce_answer(q, "x")[1] is not None      # a string is not a list of choices


def test_options_may_be_value_label_objects():
    q = _q("a", "select-one", "a", options=[{"value": "v1", "label": "One"}])
    assert cuddler.coerce_answer(q, "v1")[0] == "v1"


# ── Invariant 3: routing follows the answer ───────────────────────────────────────────────────
def test_branching_follows_the_answer():
    proc = _proc(
        _q("q1", "select-one", "pet", {"branches": [{"when": "fish", "nextQuestionId": "q_tank"}],
                                       "nextQuestionId": "q_name"}, options=["dog", "fish"]),
        _q("q_tank", "number", "litres", {}),
        _q("q_name", "short-text", "name", {}),
    )
    dog = cuddler.run_process(proc, lambda q: {"q1": "dog", "q_name": "Rex"}[q["questionId"]])
    fish = cuddler.run_process(proc, lambda q: {"q1": "fish", "q_tank": 40}[q["questionId"]])
    assert dog["path"] == ["q1", "q_name"] and dog["outputs"]["name"] == "Rex"
    assert fish["path"] == ["q1", "q_tank"] and fish["outputs"]["litres"] == 40.0


def test_missing_routing_completes():
    r = cuddler.run_process(_proc(_q("a", "short-text", "a", {})), lambda q: "done")
    assert r["complete"] and r["outputs"]["a"] == "done"


def test_a_bad_answer_mid_run_raises_naming_the_question():
    proc = _proc(_q("a", "number", "a", {}))
    with pytest.raises(cuddler.ProcessError, match="'a'"):
        cuddler.run_process(proc, lambda q: "not a number")


def test_a_routing_cycle_fails_loudly():
    proc = _proc(_q("a", "short-text", "a", {"nextQuestionId": "b"}),
                 _q("b", "short-text", "b", {"nextQuestionId": "a"}))
    with pytest.raises(cuddler.ProcessError, match="terminate"):
        cuddler.run_process(proc, lambda q: "x", max_steps=50)


# ── Invariant 4: a script asset is reachable, and stays data ───────────────────────────────
def test_a_process_with_a_script_is_refused():
    proc = _proc(_q("a", "short-text", "a", {}))
    proc["scriptPath"] = "catalogs/x/a/script.ps1"
    with pytest.raises(cuddler.ProcessError, match="script"):
        cuddler.run_process(proc, lambda q: "x")


def test_a_per_question_script_is_also_detected():
    proc = _proc(_q("a", "short-text", "a", {}, scriptPath="x/script.ps1"))
    p = cuddler.Process.load(proc)
    assert p.has_script is True


# ── Invariant 5: it is the human.ask capability adapter ───────────────────────────────────────
def test_human_ask_adapter_runs_a_process():
    adp = capability.adapter_for("human.ask")
    assert adp is not None
    proc = _proc(_q("a", "short-text", "a", {}))
    r = adp({"process": proc}, ask=lambda q: "hi")
    assert r["outputs"]["a"] == "hi"


def test_human_ask_needs_an_answer_source():
    with pytest.raises(RuntimeError, match="answer-source"):
        capability._adapter_human_ask({"process": {}}, ask=None)


def test_human_ask_needs_a_process():
    with pytest.raises(ValueError, match="process"):
        capability._adapter_human_ask({}, ask=lambda q: "x")


def test_human_ask_is_an_offered_source_capability():
    assert capability.CAPABILITIES["human.ask"].kind == capability.SOURCE


# ── Invariant 6: collected answers become a private, human-provenanced artifact ───────────────
def test_answers_artifact_is_private_and_human_provenanced():
    from ember import genesis
    r = {"questionnaireId": "onboarding", "outputs": {"pet": "dog"}, "path": ["q1"]}
    a = cuddler.answers_artifact(r, principal="me@example.com")
    # Private is a grant rather than a flag: the answer is grounded in the owner's private collection,
    # whose owner Read grant gates it. No visibility/no_share flags are written.
    assert a["collection_id"] == "private.me@example.com"   # grounded in the owner's gated collection
    assert "visibility" not in a and "no_share" not in a and "no_promote" not in a
    assert "owner" not in a                        # no owner field — ownership is the collection grant
    assert a["provenance"] == genesis.P_HUMAN     # a person answered
    assert a["outputs"] == {"pet": "dog"}
