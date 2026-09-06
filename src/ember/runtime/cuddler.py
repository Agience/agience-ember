"""Cuddler Process — the human-in-the-loop interface (D13), backing the `human.ask` capability.

`cuddler.dev` is an external standard (not on disk — see memory `cuddler-is-the-real-world-interface`).
Its **Process** role declares a workflow as a typed questionnaire: `*.process.json` with an entry
question, typed questions, per-question `routing.nextQuestionId` (a state machine), and `for-ai`
guidance. AGENT-HOST-DESIGN.md D13 places it precisely:

    a Cuddler questionnaire is a capability an adapter provides.

An operator declares `requires: ["human.ask"]`; the host's adapter satisfies it by running a Cuddler
process against whatever can answer a question (a TTY, a test stub, an MCP relay later). A human is
part of the real world, so the same adapter boundary that covers a camera covers a person.

## What this module is

It parses, validates, and runs a Cuddler Process questionnaire. Deterministic, model-free,
stdlib-only — there is no generation anywhere: a questionnaire is a fixed script, and running it
means asking its questions in the order its routing dictates and collecting typed answers. The only
non-determinism is the human's answers, which arrive through the injected `ask` callable.

## Adopted from Cuddler's own spec

- **Stable semantic rule IDs** (`AUT-SEM-*`) — a validation failure is citable, not an ad-hoc assert.
- **Typed diagnostics** — `{message, severity, artifactPath, ruleId, instancePath?}` — so a caller
  learns exactly which question, under which rule, failed.
- **Fail closed** — an invalid process does not run; an answer that does not satisfy its type is
  refused, not coerced-and-hoped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

PROCESS_CONTENT_TYPE = "application/vnd.agience.cuddler-process+json"
ANSWER_CONTENT_TYPE = "application/vnd.agience.cuddler-answers+json"

# Cuddler's enumerated answer types (`document-role/process`). Each maps to a validator below.
ANSWER_TYPES = ("select-one", "select-many", "number", "short-text", "long-text",
                "date", "percentage")

# A routing target that ends the questionnaire rather than naming another question.
COMPLETE = None


@dataclass
class Diagnostic:
    """Cuddler's diagnostic shape. `severity == "error"` fails conformance; warn/info do not."""
    message: str
    ruleId: str
    severity: str = "error"
    artifactPath: str = ""
    instancePath: str = ""

    def as_dict(self) -> dict:
        return {"message": self.message, "ruleId": self.ruleId, "severity": self.severity,
                "artifactPath": self.artifactPath, "instancePath": self.instancePath}


class ProcessError(ValueError):
    """A process that could not be loaded/validated. Carries the diagnostics."""

    def __init__(self, diagnostics: List[Diagnostic]):
        self.diagnostics = diagnostics
        super().__init__("; ".join("%s: %s" % (d.ruleId, d.message)
                                   for d in diagnostics if d.severity == "error"))


@dataclass
class Process:
    """A validated Cuddler Process questionnaire — the state machine, ready to run."""
    questionnaireId: str
    title: str
    entryQuestionId: str
    questions: Dict[str, dict]           # questionId -> question
    order: List[str] = field(default_factory=list)
    has_script: bool = False

    # ── load + validate (structure AND the semantic rules) ────────────────────────────────────
    @classmethod
    def load(cls, doc: dict) -> "Process":
        diags: List[Diagnostic] = []

        def err(msg, rule, path=""):
            diags.append(Diagnostic(msg, rule, "error", "questionnaire", path))

        if not isinstance(doc, dict):
            raise ProcessError([Diagnostic("process document is not an object", "AUT-SCHEMA-003")])

        qid = doc.get("questionnaireId") or ""
        title = doc.get("title") or ""
        entry = doc.get("entryQuestionId") or ""
        questions = doc.get("questions")
        if not qid:
            err("missing questionnaireId", "AUT-SCHEMA-003")
        if not title:
            err("missing title", "AUT-SCHEMA-003")
        if not isinstance(questions, list) or not questions:
            err("questions must be a non-empty array", "AUT-SCHEMA-003", "questions")
            raise ProcessError(diags)

        by_id: Dict[str, dict] = {}
        order: List[str] = []
        has_script = bool(doc.get("scriptPath"))
        for i, q in enumerate(questions):
            path = "questions[%d]" % i
            if not isinstance(q, dict):
                err("question is not an object", "AUT-SCHEMA-003", path)
                continue
            q_id = q.get("questionId") or ""
            if not q_id:
                err("missing questionId", "AUT-SCHEMA-003", path)
                continue
            # AUT-SEM-012: question ids unique within the questionnaire.
            if q_id in by_id:
                err("duplicate questionId %r" % q_id, "AUT-SEM-012", path)
                continue
            if not q.get("title"):
                err("question %r missing title" % q_id, "AUT-SCHEMA-003", path)
            atype = q.get("answerType")
            if atype not in ANSWER_TYPES:
                err("question %r has answerType %r, not one of %s"
                    % (q_id, atype, ", ".join(ANSWER_TYPES)), "AUT-SCHEMA-003", path)
            if not q.get("outputKey"):
                err("question %r missing outputKey" % q_id, "AUT-SCHEMA-003", path)
            # select-* must carry explicit options.
            if atype in ("select-one", "select-many"):
                opts = q.get("options")
                if not isinstance(opts, list) or not opts:
                    err("question %r is %s but defines no options[]" % (q_id, atype),
                        "AUT-SCHEMA-003", path)
            by_id[q_id] = q
            order.append(q_id)
            if q.get("scriptPath"):
                has_script = True

        # AUT-SEM-011: entryQuestionId resolves to a declared question.
        if entry and entry not in by_id:
            err("entryQuestionId %r resolves to no declared question" % entry, "AUT-SEM-011")
        elif not entry:
            err("missing entryQuestionId", "AUT-SCHEMA-003")

        # AUT-SEM-013: every routing target resolves to a question or marks completion.
        for q_id, q in by_id.items():
            nxt = _next_of(q)
            if nxt is not COMPLETE and nxt not in by_id:
                err("question %r routes to %r, which resolves to no question and is not completion"
                    % (q_id, nxt), "AUT-SEM-013", "questions/%s" % q_id)

        if any(d.severity == "error" for d in diags):
            raise ProcessError(diags)
        return cls(questionnaireId=qid, title=title, entryQuestionId=entry,
                   questions=by_id, order=order, has_script=has_script)

    # ── run the questionnaire ─────────────────────────────────────────────────────────────────
    def run(self, ask: Callable[[dict], Any], *, max_steps: int = 1000) -> Dict[str, Any]:
        """Drive the state machine. `ask(question)` returns the human's raw answer for one question.

        Returns `{questionnaireId, outputs:{outputKey: value}, path:[questionId,...], complete}`.
        Each answer is coerced and validated against its `answerType` (see `coerce_answer`); an answer
        that cannot be made valid raises `ProcessError` naming the question — never silently kept.

        `max_steps` is a backstop against a routing cycle a validator might miss; a well-formed
        questionnaire terminates by reaching completion. Raises loudly if it trips."""
        outputs: Dict[str, Any] = {}
        path: List[str] = []
        current = self.entryQuestionId
        steps = 0
        while current is not COMPLETE:
            if steps >= max_steps:
                raise ProcessError([Diagnostic(
                    "questionnaire did not terminate within %d steps (routing cycle?)" % max_steps,
                    "AUT-SEM-013", severity="error", artifactPath="questionnaire",
                    instancePath="/".join(path[-8:]))])
            steps += 1
            q = self.questions[current]
            raw = ask(q)
            value, diag = coerce_answer(q, raw)
            if diag is not None:
                raise ProcessError([diag])
            outputs[q["outputKey"]] = value
            path.append(current)
            current = _route(q, value)
        return {"questionnaireId": self.questionnaireId, "outputs": outputs,
                "path": path, "complete": True}


# ── routing ────────────────────────────────────────────────────────────────────────────────────
def _next_of(q: dict):
    """The static next-question of a question (ignores answer-conditional routing), for validation.
    `None`/missing => completion."""
    r = q.get("routing")
    if isinstance(r, dict):
        return r.get("nextQuestionId", COMPLETE)
    return COMPLETE


def _route(q: dict, value: Any):
    """The next question given the answer. Cuddler routes primarily by `routing.nextQuestionId`;
    answer-conditional branching is expressed as `routing.branches: [{when, nextQuestionId}]` where
    `when` matches the answer. Unmatched branches fall back to `nextQuestionId`. A branch is a data
    comparison, never a predicate to evaluate — no code path here runs anything from the document."""
    r = q.get("routing")
    if not isinstance(r, dict):
        return COMPLETE
    for br in (r.get("branches") or []):
        if not isinstance(br, dict):
            continue
        when = br.get("when")
        if _matches(when, value):
            return br.get("nextQuestionId", COMPLETE)
    return r.get("nextQuestionId", COMPLETE)


def _matches(when: Any, value: Any) -> bool:
    """Answer-branch match: equality, or membership when the answer is a list (select-many)."""
    if isinstance(value, list):
        return when in value
    return when == value


# ── typed answer validation (the whole point of "typed questionnaire") ───────────────────────────
def coerce_answer(q: dict, raw: Any) -> Tuple[Any, Optional[Diagnostic]]:
    """`(value, diagnostic)`. Coerce a raw answer to the question's `answerType`, or return a
    diagnostic (value None). A `number` that isn't numeric is refused, and a `select-one` outside
    its options is refused, rather than coerced-and-hoped. That refusal is the type doing its job."""
    atype = q.get("answerType")
    qid = q.get("questionId", "?")
    path = "questions/%s" % qid

    def bad(msg):
        return None, Diagnostic(msg, "AUT-SEM-VALUE", "error", "answer", path)

    if atype == "short-text" or atype == "long-text":
        if raw is None:
            return bad("question %r expects text, got nothing" % qid)
        return str(raw), None

    if atype == "number":
        try:
            return float(raw), None
        except (TypeError, ValueError):
            return bad("question %r expects a number, got %r" % (qid, raw))

    if atype == "percentage":
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return bad("question %r expects a percentage, got %r" % (qid, raw))
        if not (0.0 <= v <= 100.0):
            return bad("question %r percentage %r is outside 0..100" % (qid, v))
        return v, None

    if atype == "date":
        # ISO-8601 date, validated without inventing a value. No timezone math, no now().
        s = str(raw).strip()
        try:
            import datetime as _dt
            _dt.date.fromisoformat(s)
        except (TypeError, ValueError):
            return bad("question %r expects an ISO date (YYYY-MM-DD), got %r" % (qid, raw))
        return s, None

    if atype in ("select-one", "select-many"):
        options = _option_values(q)
        if atype == "select-one":
            if raw not in options:
                return bad("question %r answer %r is not one of its options" % (qid, raw))
            return raw, None
        # select-many: a list, every element an option, order-preserved, de-duplicated.
        if isinstance(raw, (str, bytes)) or not _is_iterable(raw):
            return bad("question %r (select-many) expects a list of options, got %r" % (qid, raw))
        chosen, seen = [], set()
        for r in raw:
            if r not in options:
                return bad("question %r choice %r is not one of its options" % (qid, r))
            if r not in seen:
                seen.add(r)
                chosen.append(r)
        return chosen, None

    return bad("question %r has unknown answerType %r" % (qid, atype))


def _option_values(q: dict) -> List[Any]:
    """The selectable values. Options may be bare scalars or `{value, label}` objects."""
    out = []
    for o in (q.get("options") or []):
        if isinstance(o, dict):
            out.append(o.get("value", o.get("id")))
        else:
            out.append(o)
    return out


def _is_iterable(x) -> bool:
    try:
        iter(x)
        return True
    except TypeError:
        return False


# ── the human.ask adapter ────────────────────────────────────────────────────────────────────────
def run_process(doc: dict, ask: Callable[[dict], Any], *, max_steps: int = 1000) -> Dict[str, Any]:
    """Load + run a Cuddler Process in one call — the `human.ask` capability adapter (D13).

    Refuses a process carrying a script asset. `scriptPath` is Cuddler's code escape hatch, and
    per D10 a script is reachable-but-not-runnable without a sandbox that does not exist here.
    Loading the questionnaire is fine; running one that would execute a script is not."""
    proc = Process.load(doc)
    if proc.has_script:
        raise ProcessError([Diagnostic(
            "process declares a scriptPath asset — code execution is not admissible without a "
            "sandbox (D10); the questionnaire is reachable but not runnable",
            "AUT-SEC-SCRIPT", "error", "questionnaire")])
    return proc.run(ask, max_steps=max_steps)


def answers_artifact(result: Dict[str, Any], *, principal: str = "") -> dict:
    """The collected answers, as an artifact — a human's typed inputs, provenance HUMAN_VALIDATED
    (a person deliberately answered). Private + owner-scoped, like `op.remember`: what a human told
    the agent is theirs until an explicit share."""
    # `CITE_GENESIS`, `P_HUMAN`, and `_now` are single-sourced in `grounding` (Rule Zero), so they
    # are imported directly rather than through the instrument op-table. The id below is built
    # from `qid` directly; no slug transform applies here, and `_slug` lives on `capability.py`
    # and `origin.py`, not on this path.
    from prism.grounding import CITE_GENESIS, P_HUMAN, _now
    qid = result.get("questionnaireId", "process")
    return {
        "id": "cuddler.answers.%s.%s" % (qid,
                                         _now().replace(":", "").replace("-", "")[:15]),
        "content_type": ANSWER_CONTENT_TYPE,
        "state": "committed",
        "questionnaire": qid,
        "outputs": result.get("outputs", {}),
        "path": result.get("path", []),
        # Grounded in the owner's private collection so the grant on it gates this answer (privacy is
        # a grant, not a flag). Whoever persists this must call `genesis._ensure_private(store, owner)`
        # first so the collection and its owner Read grant exist.
        "collection_id": "private.%s" % (principal or "local"),
        "collections": ["private.%s" % (principal or "local")],
        "provenance": P_HUMAN,                       # a person answered, on purpose
        "cited_from": CITE_GENESIS,
        "context": "answers to %r" % qid,
        "content": "",
    }


__all__ = ["PROCESS_CONTENT_TYPE", "ANSWER_CONTENT_TYPE", "ANSWER_TYPES", "Diagnostic",
           "ProcessError", "Process", "coerce_answer", "run_process", "answers_artifact"]
