"""Ember as a LOCAL copilot — an OpenAI-compatible /v1 surface over the deterministic
leaf. Point VS Code Continue (or any OpenAI SDK) at ``http://127.0.0.1:8091/v1`` and
every completion is a grounded, attributed distillation of the local corpus — or a plain
statement that the corpus has nothing to ground it. No model weights, no network, no
hallucination; disconnected by default.

Contract (OpenAI-shaped, so any OpenAI SDK client works unchanged):
  POST /v1/chat/completions   {model, messages:[{role,content}], stream?}
  GET  /v1/models             -> {"data":[{"id":"agience-ember", ...}]}

The last user message is the need; `ember.ask` routes it to the nearest offers, distills
an answer from the retrieved CONTENT, and cites the artifacts it rests on. Streaming
emits the standard chat.completion.chunk SSE frames + a terminal ``data: [DONE]``.

stdlib only (http.server) — Ember stays lean; no FastAPI. That is a dependency decision; the
licence is the GNU Affero GPL v3.0 (`agience-ember/LICENSE`).
"""
from __future__ import annotations

import urllib.parse as _urlparse

import json
import os
import secrets as _secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

MODEL_ID = "agience-ember"


def _http_write_ops() -> frozenset:
    """Operator ids that are not reachable over HTTP.

    Two sources, because `writes` alone is not the whole hazard:

      1. Anything flagged `writes=True` in `DEV_OPS`, read from the flag rather than a hardcoded
         list, so a new write operator is covered the day it exists.
      2. The whole `op.dev.` namespace. serve.py registers these as local dev hands: a developer's
         local tooling rather than a remote control plane. `writes` alone leaves `op.dev.run_tests`
         reachable, and that is the operator that shells out
         (`subprocess.run([sys.executable, "-m", "pytest", ...], cwd=WORKSPACE)`) — the execution
         step of the chain, and a way to burn the box's CPU on demand.

    Fails closed: when the registry cannot be read, the whole `op.dev.` prefix is still blocked."""
    names = set()
    try:
        from ember.runtime.runner import dev_ops        # the single distribution path (runner.py)
        names |= {o.name for o in dev_ops.DEV_OPS if getattr(o, "writes", False)}
        names |= {o.name for o in dev_ops.DEV_OPS}
    except Exception:
        pass
    return frozenset(names)


def _is_local_only_op(op: str) -> bool:
    """`op.dev.*` is local tooling and is never invocable over HTTP, registry readable or not."""
    return str(op).startswith("op.dev.")


def _resolve_delegate(handler, store):
    """Resolve which delegate this request is acting as. Cognition belongs to a delegate, so this is
    the seam where an HTTP request becomes an agent.

    The principal resolved here is what reaches `activation.act`, and it is what makes `_OBS`'s
    per-principal keying separate one delegate from another. Without it every request in the process
    collapses onto whatever `genesis._principal()` reads from the environment.

    Identity from a header is honoured only behind the shared secret. An unauthenticated header
    naming a delegate is a spoofing vector: it would let any caller read another agent's memory by
    asking as them. So a header is trusted only when `EMBER_INVOKE_TOKEN` is set and matches, the
    same gate `/v1/invoke` uses. With no token set, the result is the commons delegate. Fails
    closed: any error resolving yields the commons delegate.

    The commons delegate is the fallback. The process delegate is not, because it carries
    `EMBER_PRINCIPAL`, a real person on every deployed node — see §3.
    """
    from mantle.shard import curate as _curate
    from ember.runtime.delegate import Delegate

    # ── 1. origin is the IdP (§13.3, §13.11.1) ───────────────────────────────────────────────────
    # A bearer token verified against origin's published JWKS is the caller's real identity. That is
    # what keeps the two ports honest: an authenticated caller gets their own scope, while an
    # unauthenticated one falls through to the commons principal (§3) and the public commons.
    #
    # An unauthenticated request resolves to the commons rather than to the process delegate, which
    # on a deployed node is a real person (a deployment sets `EMBER_PRINCIPAL` to one);
    # otherwise every anonymous visitor reads through that person's light cone. Assuming a user
    # costs as much as spoofing one. The resolution itself is at §3.
    #
    # Fails closed, exactly as the relay does: an invalid signature, unknown key, expired token or
    # missing trust anchor yields no identity, never a partial or assumed one. It falls through to
    # the commons rather than raising, because a commons read is a legitimate request; it simply is
    # not this caller's.
    try:
        _raw = (handler.headers.get("Authorization") or "")
        if _raw.lower().startswith("bearer "):
            _tok = _raw[7:].strip()
            if _tok:
                from prism.trust.authority_trust import verify_jwt as _verify
                _claims = _verify(_tok, expected_issuer_service="origin")
                _sub = str(_claims.get("sub") or "").strip()
                if _sub:
                    return Delegate.get(store, person=_sub)
    except Exception:
        pass                                     # no identity — fall through to anonymous

    # ── 2. shared-secret header (internal/trusted callers only) ──────────────────────────────────
    try:
        tok = os.getenv("EMBER_INVOKE_TOKEN", "")
        if tok and _secrets.compare_digest(str(handler.headers.get("X-Ember-Token", "")), tok):
            person = (handler.headers.get("X-Ember-Principal") or "").strip() or None
            did = (handler.headers.get("X-Ember-Delegate") or "").strip() or None
            if person or did:
                return Delegate.get(store, person=person, id=did)
    except Exception:
        pass
    # ── 3. the commons — no verified identity, which is not the same as no one.
    #
    # The principal is `curate.COMMONS_PRINCIPAL`, named for what it is rather than
    # for an absence. An unauthenticated reader is the commons itself, reading what the commons
    # holds; §13.11.7's ground-zero layer is a real grounding, and CC/OA material belongs to it.
    # It reads operationally too: everything read or written on this port grounds at this principal,
    # so a conversation in `private.common@ground` is the commons' own record.
    #
    # The principal is passed explicitly. `Delegate.get` otherwise falls through to
    # `os.getenv("EMBER_PRINCIPAL")` — which node 71's own `run-serve.ps1` sets to a real person —
    # and every unauthenticated request would then read through that person's light cone. That is an
    # extra grant rather than a missing one, so it shows up as a reader seeing more than the commons
    # holds rather than as a denied read. The deployment variable names who the node is; who an
    # anonymous reader is comes from here.
    #
    # The delegate id is pinned for the same reason, because without it the principal alone can
    # still resolve to a person. `Delegate.get` derives the id from `EMBER_DELEGATE_ID` when no `id=`
    # is given, and the delegate cache is keyed on the id rather than on the person (`Delegate.get`
    # says so). On a node that sets both env vars, the first authenticated-as-the-node call caches a
    # Delegate under `$EMBER_DELEGATE_ID` carrying `EMBER_PRINCIPAL`, and this call — asking for the
    # commons — would get that cached object back with the `person=` argument discarded, screen
    # included. Naming the id from the principal makes the commons its own cognition by
    # construction, out of reach of either deployment variable.
    return Delegate.get(store, person=_curate.COMMONS_PRINCIPAL,
                        id="d." + _curate.COMMONS_PRINCIPAL)

def _note_fault(faults: List[str], arm: str, exc: BaseException) -> None:
    """Record that an answering arm raised, and make that visible.

    An arm returning `None` means "this need is not mine" — ordinary control flow. An arm raising
    means the arm is broken. Both fall through to the next arm, because one broken arm should not
    500 a request another arm can answer, so the two need a way to be told apart: a raised fault is
    written to stderr and attached to the response as `x_agience.faults`, where it is observable
    from the client and from the log without a debugger. Without that, a broken arm reads exactly
    like a declining one and the next arm's blank goes out as an answer.

    Never raises: a failure to report a failure would be a third failure."""
    try:
        detail = "%s: %s: %s" % (arm, type(exc).__name__, str(exc)[:200])
        faults.append(detail)
        print("[ember.serve] answering arm faulted -> %s" % detail, file=sys.stderr, flush=True)
    except Exception:
        pass


def _last_user(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return (messages[-1].get("content") or "").strip() if messages else ""


def _control_answer(store, query: str):
    """Deterministic intent router: status / ingest / advance are answered by invoking the
    corresponding operator.

    Returns `(text, cited)`, or None when this is not a control need.

    `cited` is returned so `grounded` derives from it on this path exactly as it does on the others.
    Each operator-backed branch cites the operator it invoked; the branches that store nothing cite
    nothing, so a reply that grounded nothing does not go out flagged as grounded
    (`test_refusal_is_not_reported_as_grounded`)."""
    if store is None:
        return None
    q = query.lower().strip()
    # No scripted persona. A greeting is a signal like any other: it grounds through activation
    # ("hello" -> its own gloss) or returns the computed null, the same as every other turn. The
    # node does not recite a personality.
    from ember import genesis
    # Owner memory: only an explicit statement ingests, not a question or banter. Private + gated.
    for trig in ("remember that ", "remember: ", "remember ", "note that ", "note: "):
        if q.startswith(trig):
            text = query.strip()[len(trig):].strip()
            if not text:
                # Nothing was invoked, so nothing grounds this reply.
                return "Nothing to remember. Say e.g. “remember that my project ships Friday”.", []
            r = genesis.remember(store, text)
            if not r.get("stored"):
                return f"Did not store: {r.get('reason')}", []
            return (f"Noted — stored privately, only for you (owner-scoped, no_share + no_promote). "
                    f"id={r['id']}\n\n[via op.remember → {r['collection']}]", ["op.remember"])
    if any(p in q for p in ("status", "how is the universe", "how are you doing", "rho", "coverage")):
        r = genesis.invoke(store, "op.status.universe")["result"]
        lines = [f"ρ (compression) = {r.get('rho')}   keyed_coverage = {r.get('keyed_coverage')}   "
                 f"artifacts = {r.get('artifacts')}", "curriculum:"]
        for s in r.get("curriculum", []):
            lines.append(f"  - {s['stage']}: {s['have']}/{s['target']} "
                         f"({int((s.get('progress') or 0)*100)}%){'  ✓promoted' if s['promoted'] else ''}")
        top = sorted(((c, d) for c, d in r.get("collections", {}).items() if d["artifacts"]),
                     key=lambda kv: -kv[1]["artifacts"])[:6]
        lines.append("collections: " + ", ".join(f"{c}={d['artifacts']}(ρ{d['rho']})" for c, d in top))
        return "\n".join(lines) + "\n\n[via op.status.universe]", ["op.status.universe"]
    if ("advance" in q and "curriculum" in q) or q in ("advance", "ingest next", "next"):
        r = genesis.invoke(store, "op.curriculum.advance")["result"]
        return (f"advanced: {json.dumps(r)}\n\n[via op.curriculum.advance]",
                ["op.curriculum.advance"])
    if q.startswith("ingest ") and ("wiki" in q or "stage 1" in q or "stage1" in q):
        r = genesis.invoke(store, "op.source.wikipedia-simple", {"limit": 5000})["result"]
        return (f"ingested: {json.dumps(r)}\n\n[via op.source.wikipedia-simple]",
                ["op.source.wikipedia-simple"])
    return None


def _reach_route(store, ember, query: str, k: int = 6):
    """Reach lumen's responder (`op.respond`), or return `(None, None)` — the computed null.

    Routing dispatches and composes prose, both persona acts, so it lives in lumen. Ember imports no
    chorus, so this rides `ember.runtime.reach` exactly as `genesis._reach_conversation` does: keyed
    by the fleet root, gated by ember's own access light-cone, and dark unless a carrier is wired.
    Dark is a real state — the caller falls through to the keyed/semantic backstop, which is a real
    answer from ember's own grounding.
    """
    if store is None:
        return None, None
    try:
        from ember import genesis as _g
        carrier = _g._conversation_carrier(store)
        if carrier is None:
            return None, None
        # The capability addressed is `op.respond` (`lumen.reach_provider.RESPOND_CAP`), which is
        # what lumen serves and what aria's bff reaches — one name across every surface. A need
        # addressed to a capability no provider offers does not fail fast: it goes onto the plane,
        # nothing can ever discharge evidence for it, and the caller waits: one such unanswered need
        # ran for 794 s of CPU without returning, against 7.4 s for the same question routed through
        # `reach_provider.respond_handler`.
        #
        # The resolver the launcher hands in comes first. `_fleet/peers/71/ember/live_node.py` injects
        # `{"resolve": …, "fabric": None}` because the node's real transport is a store-and-forward
        # `StoreCarrier`, which has no `subscribe` and therefore cannot be a `fabric=`. Reaching with
        # `fabric=carrier["fabric"]` would pass `fabric=None` and skip the resolver entirely. Same
        # two rungs, in the same order, as `genesis._reach_conversation`.
        RESPOND = "op.respond"
        need = {"text": query, "k": k}
        resolver = carrier.get("resolve")
        if callable(resolver):
            ev = resolver(need, RESPOND)
        elif carrier.get("fabric") is not None:
            from ember.runtime import reach as _reach
            ev = _reach.reach(store, carrier.get("principal") or "ember", need, to=RESPOND,
                              root_secret=carrier["root_secret"], fabric=carrier["fabric"])
        else:
            # Neither rung wired is dark, and dark is a real state — the caller falls through to its
            # own backstop. Returning here rather than reaching with `fabric=None` is what keeps the
            # unanswerable-need wait above out of reach.
            return None, None
        if not ev:
            return None, None
        payload = ev[0] if isinstance(ev, list) else ev
        if not isinstance(payload, dict):
            return None, None
        # Normalize at the boundary. Evidence off the plane is a row; everything downstream reads
        # `.grounded` / `.read` / `.cited` / `.text`. The row is shaped into the operator bundle's
        # `Answer` here rather than rewriting ten call sites to dict access — the bundle is
        # distributed content ember runs, so using its shape leaves the contract with the bundle.
        # One place converts; the caller stays unaware.
        from ember.runtime.runner import answer as _answer_mod
        # `op.respond` spells it `answer` rather than `text`: `conversation.respond` returns
        # `{answer, activations, cited, …}`. Reading only `text` yields an empty answer that is not
        # `None`, so the caller would treat a real reply as ungrounded and fall through to its
        # backstop. Both spellings are accepted here, for the same reason the bff accepts both: this
        # is the boundary that normalises.
        cited = list(payload.get("cited") or [])
        # Groundedness is measured rather than assumed, and `conversation.respond` does not emit the
        # key at all — so defaulting it to False would call every real citation ungrounded, and
        # defaulting it to True would claim a grounding nobody checked. An answer is grounded iff it
        # cites something, and an explicit flag from the tekton wins, because that is the tekton
        # reporting its own measurement. The same rule as `aria/www/bff/main.py`.
        grounded = (bool(payload["grounded"]) if "grounded" in payload else bool(cited))
        a = _answer_mod.Answer(text=str(payload.get("answer") or payload.get("text") or ""),
                               grounded=grounded, cited=cited,
                               read=dict(payload.get("read") or {}))
        if not a.text.strip():
            return None, None          # the computed null — nothing fired, and that is not an answer
        return a, payload.get("path")
    except Exception:
        # A dark reach is indistinguishable from a missing one for the caller's purposes: both mean
        # "no persona answered", and the backstop below is the honest response to either.
        return None, None


#: The one sentence this module composes, and it is a computed null rather than an explanation: it
#: states that nothing answered, and nothing else. Ember is the runner; the persona composes. The
#: null still has to reach the caller, because an empty bubble says less than a short sentence, so
#: this is the irreducible minimum — a named constant, so there is exactly one of it.
NO_ANSWERER_TEXT = "No answerer is wired on this node."

#: Operator-facing diagnosis: stderr, once per process rather than once per keystroke. The chat
#: surface is the wrong audience for it.
_NO_ANSWERER_LOGGED = False


def _log_once_no_answerer() -> None:
    global _NO_ANSWERER_LOGGED
    if _NO_ANSWERER_LOGGED:
        return
    _NO_ANSWERER_LOGGED = True
    print("[ember.serve] no answerer wired: Ember.engine is None, so `ask()` raises NoAnswerer, and "
          "the op.respond reach returned a null (no live fabric). The ontology is unaffected. "
          "Wire a reach fabric + mint the op.respond grant, or inject an engine.",
          file=sys.stderr, flush=True)


def _answer_text(ember, query: str, k: int = 6, *, store=None) -> tuple:
    """Return (text, grounded, cited). Routes lexical needs to the dictionary (keyed) path and
    conceptual needs to embedding retrieval; citations ride in the returned `cited` rather than in
    the answer text."""
    # The router is lumen's. Ember cannot import chorus, so this reaches `op.respond` over the ground
    # plane and is inactive by default: with no live fabric the reach yields the computed null and
    # the code below falls through to the keyed/semantic backstop. It fabricates nothing and it does
    # not raise — the same shape `genesis._reach_conversation` established.
    a, path = _reach_route(store, ember, query, k=k)
    if a is not None:
        # Self-improvement: record operator fitness from real use (deterministic ops self-verify;
        # cited artifacts credit their producing operator). This is what activates selection.
        if a.grounded:
            try:
                from ember.runtime.runner import evolution
                op = (a.read or {}).get("operator")
                if op and str(op).startswith("op."):
                    evolution.record_invocation(store.artifacts, op, verified=True)
                for cid in (a.cited or [])[:1]:
                    if str(cid).startswith(("sym-", "wn-", "file-")):
                        evolution.record_use(store.artifacts, cid)
            except Exception:
                pass
    elif ember is not None:
        a = ember.ask(query, k=k).answer
        path = "semantic"
    else:
        # `ember is None` is a supported deploy: `serve_openai` sets it deliberately, twice, for a
        # store-only node (no semantic index, no embedder). This branch reports that rather than
        # calling `ember.ask(...)` on it, which would close the connection with no body.
        #
        # There are two independent gaps here, and a reindex closes only one:
        #   · `Ember.ready` is False because `cache is None` (the retrieval index). Reindex fixes it.
        #   · `Ember.engine` is None, and `ask()` raises `NoAnswerer` regardless of the cache:
        #     "ember runs energized crystals and prisms, it does not compose answers."
        # A rebuilt index alone therefore still cannot answer, so pointing an operator at `reindex`
        # would cost them a round trip back to here.
        #
        # The text is a computed null: one plain sentence, with nothing invented in it. An
        # explanation of ember's architecture rendered into a chat bubble would be the wrong audience
        # (whoever typed the question cannot act on it) and the wrong place — ember is the runner,
        # and composing words is the persona's job, which is what this branch exists to report is
        # missing. The diagnosis goes to stderr once per process (`_log_once_no_answerer`), where an
        # operator reads it and a human reading chat does not have to.
        _log_once_no_answerer()
        return NO_ANSWERER_TEXT, False, []
    if not a.grounded:
        return (a.text or "I don't have grounded local evidence for that."), False, list(a.cited or [])
    # No sources footer in the answer text. Citations ride in the structured `x_agience.cited`
    # (returned below), where a client can show provenance without it cluttering the prose.
    return a.text, a.grounded, list(a.cited or [])


def serve_openai(port: int = 8091, host: str = "127.0.0.1", ember=None, *, k: int = 6,
                 watch: Optional[str] = None) -> ThreadingHTTPServer:
    """Start the /v1 server in a background thread; return the handle (call
    ``.shutdown()`` to stop). Loads the local leaf once and reuses it per request."""
    if ember is None:
        try:
            from ember.corpus.ingest import open_local
            ember = open_local()
            if not ember.ready:            # index empty -> treat as store-only (no semantic)
                ember = None
        except Exception:
            ember = None                   # store-only deploy: no semantic index / no embedder
    try:
        from mantle.shard.local_store import open_store
        from ember.runtime import runner
        store = open_store()
        runner.attach(store)               # store bundles
        arithmetic = runner.load("arithmetic")
        operators = runner.load("operators")
        dev_ops = runner.load("dev_ops")
        fetch = runner.load("fetch")
        # Register operator artifacts best-effort: on a busy shard (a peer with a full ingest fleet
        # writing) any single registration can hit a transient ConcurrentModification, and that
        # should leave the rest of the store intact. The operators are idempotent and get
        # re-registered on the next invoke(), so a miss here costs nothing.
        # Author is the resolved principal (see the worker.py note): the default 'ember-local'
        # process author writes an unresolvable created_by on a fresh store.
        from ember import genesis as _genesis
        _author = _genesis._author_ref(store, os.getenv("EMBER_PRINCIPAL") or "ember-local")
        for _reg in (lambda: arithmetic.register_transform_operators(store, author=_author),
                     lambda: operators.register_operators(store, author=_author),
                     lambda: dev_ops.register_dev_operators(store, author=_author),
                     lambda: fetch.register_fetch_operators(store, author=_author)):
            try:
                _reg()
            except Exception:
                pass
        # Say what this node is. `capability.publish` had one call site — `runtime/worker.py`
        # — and `agience up ember` runs `python -m ember.cli serve`, a DIFFERENT process. So a node
        # that serves and never ingests never published its host artifact, and `agience restart
        # ember` could not repair one. Measured 2026-08-25: a live service was restarted to fix
        # exactly that and it had no effect, because the restart runs this path.
        #
        # Inside the same `try` as the store bundle, deliberately: with no store there is nothing to
        # publish INTO, and `publish_host` is itself best-effort and never raises.
        from ember.runtime import capability as _capability
        _capability.publish_host(store, who="ember.serve")
    except Exception:
        store = None            # dictionary/arithmetic-store path down; semantic-only still works

    # Warm the meaning-geometry IC cache in the background so the first chat query isn't the ~8s
    # cold load (the reasoning layer's load_ic()). Non-blocking.
    #
    # It warms the ontology driver (`crystal.ontology.driver`), which is the index the runtime reads:
    # `activation` and `geometry` take ontology structure from our own corpus through the
    # store-backed driver. Warming nltk's WordNet corpus instead would pay ~6 s of startup to
    # populate a cache nothing consults, and pull an undeclared dependency into the serve path —
    # verified that driver/activation/geometry/forgetting/templates/keyed all import and work with
    # `nltk` poisoned in sys.modules.
    #
    # The `except` swallows deliberately, since a cold cache should not stop the server booting, and
    # it names what failed, because a warm that never happens otherwise looks exactly like a warm
    # that worked until the first query pays the full cost.
    def _warm_ic():
        try:
            from crystal.ontology import geometry
            geometry.load_ic()                         # the Brown IC dict (~seconds)
            from crystal.ontology import driver as wn
            wn.synsets("dog")                          # force the store-backed index load
            _ = wn.synset("dog.n.01").hypernyms()      # and the hypernym graph
            # Warm the whole conversational path, not just the index. Measured, the first user query
            # paid ~16 s because seed -> full-DAG spread -> rank_fired (a store read per fired
            # concept) -> compose all ran cold. A read-only `recognize` over a few common words
            # exercises every one of those stages, so the first real turn is warm. It never writes.
            if store is not None:
                from ember.ontology import activation
                # Warm the machinery once — the seed/spread/rank/compose path and the instrument read
                # both load on first use. One representative turn warms them; the content is
                # incidental.
                _acts = activation.recognize(store, "dog")
                activation.output_membrane(store, _acts)           # warm the INSTRUMENT (~1s cold read)
        except Exception as e:                         # pragma: no cover - boot must not fail
            sys.stderr.write("warm_ic skipped (%s: %s) -- first query pays the cold load\n"
                             % (type(e).__name__, str(e)[:120]))
    threading.Thread(target=_warm_ic, daemon=True).start()

    # Self-improvement loop: illuminate dark matter, track corpus/operator health over time.
    if store is not None:
        try:
            from ember.runtime import improve
            improve.start(store, interval=300.0)
        except Exception:
            pass
        # No background loop is started here, and each of the four that could be is placed
        # elsewhere for a measured reason:
        #
        #   * Publishing belongs to `_fleet/peers/71/ember/health-loop.py`. A second publisher is not redundancy:
        #     both call `stats.write_stats`, which read-modify-writes the shared `stats.prev`
        #     differencing record, so two unsynchronized 20s/30s loops interleave read and write and
        #     each differences against a sample the other just overwrote — rates computed from
        #     mismatched (count, ts) pairs, corrupting the numbers the loop exists to publish. A box
        #     that runs `serve` without a health loop reports nothing, which shows on the status page
        #     as a missing peer rather than as wrong numbers.
        #   * Nothing on a timer full-scans the corpus. A metrics pass that walks every committed row
        #     summing bytes with no overlap guard, on a fixed-interval tick, stacks another full scan
        #     on the last once a single pass runs longer than the interval — at corpus scale that
        #     alone pins the database's CPU and GC threads, independent of anything else running.
        #     Metrics are a health concern and live on health-loop.py's slow cadence, computed from a
        #     bounded sample.
        #   * The anti-entropy digest is maintained incrementally on write (`sync._MERKLE_LIVE`, and
        #     mantle XORs the leaf inside the same transaction as the row), so there is no stale
        #     cache for a timer to refresh — a periodic recompute would be the corpus scan above.
        #   * The mesh daemon runs as its own process, launched by the node launcher. `mesh.daemon.run`
        #     does CPU-heavy Merkle hashing over the whole graph (2M artifacts) and holds the GIL,
        #     starving the chat handler in the same process (measured: chat went to 20-60 s), and it
        #     writes, competing for the DB. Its own docstring calls it one process per box.
        #     `op.mesh.reconcile` is invocable directly for a one-shot sync.
        #
        # Any thread started here fails loudly. A bare `threading.Thread(target=...)` makes a failure
        # in the target invisible, the same shape as the `except Exception: pass` `_note_fault`
        # replaces. `test_serve_thread_targets_resolve` proves every target this module starts is
        # callable, so a name that does not resolve is a red test rather than a silent no-op.

    # Live knowledge: watch the workspace and re-describe changed files (keeps the keyed
    # index current as you develop — the flywheel running in the background of the server).
    if watch and store is not None:
        try:
            from ember.corpus import sources
            rt = sources.SourceRuntime(store, sink=sources.describe_sink(store))
            fs = sources.FolderSource("workspace", watch, exts={".py", ".md", ".txt", ".rst"})
            fs.prime()                                   # only future changes trigger
            rt.register(fs)
            threading.Thread(target=rt.run, kwargs={"interval": 10.0}, daemon=True).start()
        except Exception:
            pass

    # a fixed created-time avoids Date.now() nondeterminism; clients ignore the value's exactness
    CREATED = 1_700_000_000

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            # ── ontology browser (same-origin UI + JSON API) ──
            if store is not None and (self.path in ("/browse", "/chat", "/", "/status", "/dashboard",
                                                    "/stats", "/console")
                                      or self.path.startswith("/facet/")
                                      or self.path.startswith("/library")
                                      or self.path.startswith("/api/")):
                from ember.facets import browse
                from urllib.parse import urlparse, parse_qs, unquote
                u = urlparse(self.path)
                if u.path == "/stats":                      # raw local snapshot (instant; the mesh reads this)
                    from ember.surface import stats as _stats
                    self._send(_stats.read_stats(store.keys_dir) or {"error": "no snapshot yet"}); return
                if u.path == "/dashboard":                  # the mesh dashboard — reads snapshots, never scans
                    body = browse.dashboard_page(store).encode()
                    self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body))); self.end_headers()
                    self.wfile.write(body); return
                if u.path == "/status":
                    body = browse.status_page(store).encode()
                    self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body))); self.end_headers()
                    self.wfile.write(body); return
                if u.path == "/browse":
                    body = browse.PAGE.encode()
                    self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body))); self.end_headers()
                    self.wfile.write(body); return
                if u.path in ("/chat", "/"):
                    body = browse.CHAT_PAGE.encode()
                    self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body))); self.end_headers()
                    self.wfile.write(body); return
                if u.path == "/library":
                    # The library does not pick the zoom (§9.6) — it offers the levels this corpus
                    # has, as links — so `resolution` is parsed here and carried through. Reading
                    # only `refresh` returns the same offer page for every one of those links:
                    # choices with no way to choose. Declining to pick works only if the caller can.
                    _q = _urlparse.parse_qs(u.query or "")
                    refresh = "refresh=1" in (u.query or "")
                    _res = (_q.get("resolution") or [None])[0]
                    try:
                        resolution = float(_res) if _res not in (None, "") else None
                    except ValueError:
                        resolution = None          # an unreadable level is no level, not an error
                    body = browse.library_page(store, refresh=refresh,
                                               resolution=resolution).encode()
                    self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body))); self.end_headers()
                    self.wfile.write(body); return
                # ── the local control console (ember/surface/console.py) ──────────────────────
                # Read-only here: it renders the node's config, the prisms that have registered,
                # and each one's live health beside what it once claimed. Running an operator is a
                # POST and is gated separately — see `/api/console/run`.
                if u.path == "/console":
                    from ember.surface import console as _console
                    body = _console.page(store).encode()
                    self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body))); self.end_headers()
                    self.wfile.write(body); return
                # ── facets — a crystal's view, served by the runner that runs the crystal ──────
                # `/facet/<name>/<path>`; `<name>/` alone serves its `index.html`. Read-only and
                # ungated: a facet is a public view, and its far side (the tekton it talks to)
                # carries its own authorization. Serving the view behind a token while the data is
                # gated separately would put the check on the wrong side of the membrane.
                if u.path.startswith("/facet/"):
                    from ember.surface import console as _console
                    seg = unquote(u.path[len("/facet/"):]).split("/", 1)
                    fname = seg[0]
                    rel = seg[1] if len(seg) > 1 else ""
                    target = _console.facet_file(fname, rel)
                    if target is None:
                        # 404 for a traversal attempt too — a 403 would confirm the path exists.
                        self._send({"error": "no such facet file"}, 404); return
                    data = target.read_bytes()
                    self.send_response(200)
                    self.send_header("content-type", _console.facet_content_type(target))
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data); return
                if u.path == "/api/console/state":
                    from ember.surface import console as _console
                    # `probe=0` skips dialling each host's /health — the page is then a pure store
                    # read, which is what you want when a host is hanging rather than down.
                    self._send(_console.state(
                        store, probe=("probe=0" not in (u.query or "")))); return
                if u.path == "/api/artifacts":
                    q = parse_qs(u.query)
                    self._send(browse.api_artifacts(
                        store, content_type=(q.get("type", [None])[0] or None),
                        skip=int(q.get("skip", [0])[0]), limit=min(200, int(q.get("limit", [40])[0])),
                        lemma=(q.get("lemma", [None])[0] or None))); return
                if u.path.startswith("/api/artifact/"):
                    aid = unquote(u.path[len("/api/artifact/"):])
                    self._send(browse.api_artifact(store, aid)); return
                self._send({"error": "not found"}, 404); return
            if self.path.rstrip("/") == "/v1/models":
                self._send({"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "created": CREATED, "owned_by": "agience"}]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                # A health check that raised is not a healthy node. `genesis.health` calls `status()`
                # and `consistency()` unguarded, so a store that is timing out — the wedged-ArcadeDB
                # state this codebase documents at length — raises, and that raise answers 503 here
                # rather than falling through to the `{"status": "ok"}` below. The fallback's `ready`
                # flag is computed from the semantic index and says nothing about the store, so a
                # supervisor or load balancer polling this endpoint would otherwise read 200 OK in
                # precisely the state that calls for the opposite.
                if store is not None:
                    try:
                        from ember import genesis
                        h = genesis.health(store)
                        self._send({"status": "ok" if h["healthy"] else "degraded",
                                    "model": MODEL_ID, **h})
                    except Exception as e:
                        # 503: the node cannot demonstrate it is healthy. Say why, and say which
                        # check failed — an unexplained 503 is only marginally better than a lie.
                        self._send({"status": "unhealthy", "model": MODEL_ID,
                                    "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                                    "check": "genesis.health"}, 503)
                    return
                # No store bound at all (chat-only serve). Report that honestly rather than "ok" —
                # this branch cannot speak to store health because there is nothing to ask.
                self._send({"status": "ok", "model": MODEL_ID, "store": None,
                            "ready": bool(ember and getattr(ember, "ready", False))})
            else:
                self._send({"error": {"message": "not found"}}, 404)

        def do_POST(self):  # noqa: N802
            path = self.path.rstrip("/")
            if path not in ("/v1/chat/completions", "/v1/invoke", "/api/invoke",
                            "/hosts/register", "/api/console/run"):
                self._send({"error": {"message": "not found"}}, 404)
                return
            length = int(self.headers.get("content-length", 0) or 0)
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send({"error": {"message": "invalid JSON"}}, 400)
                return

            # ── host registration — a prism host announcing itself and what it serves ──
            #   POST /hosts/register  {"name": "...", "operators": ["..."], "capabilities": [...]}
            #
            # This is the router for the call both prism legs make: they POST
            # `{api_uri}/hosts/register` (`prism/host/host.py`, the JS `src/host/host.ts`) wrapped in
            # a bare `except` logged "non-fatal", so a host whose registration goes nowhere starts
            # anyway and its operators are never announced.
            #
            # It is a write endpoint and is gated accordingly. It creates artifacts, so an
            # unauthenticated version would let any host that can reach this port inject operator
            # artifacts into the graph — and `mesh/federation.py` shows peers polling each other on
            # the LAN, the same shape as `/v1/invoke`'s gate below: an unauthenticated write endpoint
            # reachable on the network is remote code execution by another name.
            # Requires `EMBER_INVOKE_TOKEN` to be set and to match; no token configured means the
            # endpoint is closed. Fails closed in both directions.
            # ── console: run an operator served by a registered prism host ────────────────────
            #
            # Gated exactly like `/hosts/register`, and for a stronger reason: this causes work to
            # happen on another machine, the same class of exposure `/v1/invoke`'s gate below closes —
            # and serve.py already blocks the whole `op.dev.*` namespace over HTTP because one of
            # those operators shells out. So the token must be set and match. Unset means the console
            # is read-only.
            #
            # It executes nothing on this box: it seals a signal and ships it to a host that already
            # registered an endpoint. There is no shell on this path.
            if path == "/api/console/run":
                if store is None:
                    self._send({"error": "store unavailable"}, 503); return
                _tok = os.getenv("EMBER_INVOKE_TOKEN", "")
                if not _tok:
                    self._send({"error": "console run is disabled: set EMBER_INVOKE_TOKEN to enable "
                                         "it (it causes work on a remote host)"}, 403)
                    return
                presented = str(self.headers.get("Authorization", ""))
                presented = presented[7:] if presented.startswith("Bearer ") else \
                    str(self.headers.get("X-Ember-Token", ""))
                if not _secrets.compare_digest(presented, _tok):
                    self._send({"error": "invalid or missing console token"}, 401); return
                op = (req.get("operator") or "").strip()
                if not op:
                    self._send({"error": "missing 'operator'"}, 400); return
                try:
                    from ember.surface import console as _console
                    self._send(_console.run(store, _resolve_delegate(self, store), op))
                except Exception as exc:                        # noqa: BLE001 — reported, not hidden
                    self._send({"error": "%s: %s" % (type(exc).__name__, exc)}, 500)
                return

            if path == "/hosts/register":
                if store is None:
                    self._send({"error": "store unavailable"}, 503); return
                _tok = os.getenv("EMBER_INVOKE_TOKEN", "")
                if not _tok:
                    self._send({"error": "host registration is disabled: set EMBER_INVOKE_TOKEN "
                                         "to enable it (it writes to the graph)"}, 403)
                    return
                presented = str(self.headers.get("Authorization", ""))
                presented = presented[7:] if presented.startswith("Bearer ") else \
                    str(self.headers.get("X-Ember-Token", ""))
                if not _secrets.compare_digest(presented, _tok):
                    self._send({"error": "invalid or missing registration token"}, 401); return
                name = (req.get("name") or "").strip()
                if not name:
                    self._send({"error": "missing 'name'"}, 400); return
                try:
                    from ember.runtime import capability
                    self._send(capability.register_remote_host(
                        store, name=name,
                        operators=req.get("operators") or [],
                        endpoint=(req.get("endpoint") or "").strip(),
                        capabilities=req.get("capabilities") or []))
                except Exception as e:
                    # An unknown capability name is a 400 (the caller's payload), not a 500.
                    code = 400 if type(e).__name__ == "UnknownCapability" else 500
                    self._send({"error": f"{type(e).__name__}: {str(e)[:300]}"}, code)
                return

            # ── operator control plane: drive the universe by invoking operators (§4) ──
            #   POST /v1/invoke  {"operator":"op.status.universe"}                 -> status
            #   POST /v1/invoke  {"operator":"op.source.wikipedia-simple","arguments":{"limit":5000}}
            #   POST /v1/invoke  {"operator":"op.curriculum.advance"}             -> next increment
            if path in ("/v1/invoke", "/api/invoke"):
                if store is None:
                    self._send({"error": "store unavailable"}, 503); return
                op = req.get("operator") or req.get("op")
                if not op:
                    self._send({"error": "missing 'operator'"}, 400); return

                # Ungated, this endpoint is unauthenticated remote code execution reachable from a
                # web page. `op` goes to `genesis.invoke`, and serve.py registers the dev operators
                # into that same namespace: `op.dev.edit_file` / `rename_symbol` / `verify_change`
                # are `writes=True`, and `op.dev.run_tests` shells out to pytest with cwd=WORKSPACE.
                # The attack needs no credentials and no response: any page the developer visits
                # can `fetch('http://127.0.0.1:8091/v1/invoke', {method:'POST',
                # headers:{'content-type':'text/plain'}, body:'{"operator":"op.dev.edit_file",...}'})`
                # — `text/plain` makes it a CORS *simple request*, so there is no preflight to
                # block, and the attacker never reads the reply, so same-origin policy protects
                # nothing. Write a poisoned `conftest.py`, then invoke `op.dev.run_tests` to execute
                # it. It reaches past loopback too: `mesh/federation.py` shows peers polling each
                # other at `http://<lan-ip>:8091`, so any LAN host can invoke it directly.
                #
                # Three independent gates, because each blocks a different caller:
                #  1. Write operators are not reachable over HTTP. Gated on the real `writes` flag
                #     from `DEV_OPS` rather than on a name prefix, so a new write op is covered the
                #     day it is added.
                #  2. An `Origin` header means a browser sent it. A CLI, a peer or a script has no
                #     reason to set one; a drive-by page always does. This is what stops the
                #     no-preflight simple-request path above.
                #  3. A shared secret when `EMBER_INVOKE_TOKEN` is set. Off by default so the local
                #     dev loop is unchanged, and a node exposed on the LAN can require it.
                if op in _http_write_ops() or _is_local_only_op(op):
                    self._send({"error": f"{op!r} is local developer tooling (it writes files "
                                         f"and/or spawns processes) and is not invocable over "
                                         f"HTTP; run it from the CLI"}, 403)
                    return
                if self.headers.get("Origin"):
                    self._send({"error": "cross-origin invoke refused"}, 403); return
                _tok = os.getenv("EMBER_INVOKE_TOKEN", "")
                if _tok and not _secrets.compare_digest(
                        str(self.headers.get("X-Ember-Token", "")), _tok):
                    self._send({"error": "invalid or missing invoke token"}, 401); return
                try:
                    from ember import genesis
                    self._send(genesis.invoke(store, op, req.get("arguments") or {}))
                except Exception as e:
                    self._send({"error": f"{type(e).__name__}: {str(e)[:300]}"}, 500)
                return

            query = _last_user(req.get("messages", []))
            faults: List[str] = []
            control = _control_answer(store, query)
            if control is not None:
                # One rule for groundedness on every arm of this handler: citing implies grounded.
                text, cited = control
                grounded = bool(cited)
            else:
                text = grounded = cited = None
                # 1) Deterministic domains first, via the unified router — arithmetic (numbers), code,
                #    and the keyed dictionary answer before the wordnet response act, per design.
                #    'semantic' is excluded here so the conversational act (below) owns concept queries.
                if store is not None:
                    try:
                        # Reached rather than imported: the router belongs to lumen, and ember
                        # imports no chorus. `_reach_route` returns (None, None) with no carrier,
                        # so the `if a` gate below is skipped entirely and the conversational act
                        # owns the turn — the designed path for a non-deterministic query, not a
                        # hole.
                        a, path = _reach_route(store, ember, query, k=k)
                        # Accept only the deterministic, self-verifying domains here — arithmetic,
                        # code, dev, numeric reason. These carry their own certificate, so they
                        # rightly answer before the conversational act.
                        # `content` and `lexical` are excluded from this first gate: the
                        # conversational act below owns concept and natural-language turns — it
                        # seeds through the converged field, resolves the lead through the
                        # instrument, and routes learn/think/respond. `content` and `semantic` drop
                        # to `_answer_text`'s wiki/semantic fallback, reached only when the
                        # conversational act returns the instrument's null (no citation). That keeps
                        # the WordNet definition — grounded, cited, membrane-read — as the primary
                        # answer, with the encyclopedia as backstop for what the ontology cannot
                        # resolve: ranking a term-dense wiki page ahead of it on plain BM25 would
                        # answer "what does a dog say" from a wiki article about a person named
                        # Robert Woof instead of the learned concept.
                        # `a is None` when the reach is dark — skip the gate, do not treat it as
                        # "not grounded" and certainly not as an answer.
                        if a is not None and a.grounded and path in ("arithmetic", "arithmetic.learn",
                                                                     "code", "dev", "reason"):
                            # `a.grounded`, not `True`: the value is the Answer's own measurement.
                            # Restating it as a literal only holds while the `if` above keeps
                            # checking it — an edit to that condition would silently make the
                            # literal a claim nobody took a reading for.
                            text, grounded, cited = a.text, a.grounded, list(a.cited or [])
                    except Exception as e:
                        _note_fault(faults, "router", e)
                # 2) else the conversational response act: activation over the ontology (respond/think/
                #    learn over wordnet + taught triples). The turn is recorded private inside respond().
                if text is None and store is not None:
                    try:
                        from ember import genesis as _genesis
                        d = _resolve_delegate(self, store)        # whose cognition is this?
                        # The conversation act belongs to lumen and is reached rather than
                        # imported, inactive by default. Without a live fabric the reach yields
                        # the null (answer=None), so `text` stays None and falls through to the
                        # keyed/semantic backstop below rather than fabricating a conversational
                        # answer. `op.recognize` stays a direct ember call (grounding).
                        # The capability reached is `op.respond`: lumen's `serve_respond` is the
                        # only conversation provider in the workspace, and `reach_host.serve_lumen`
                        # registers it as the chat responder — `lumen.reach_provider.RESPOND_CAP`
                        # is that id.
                        r = _genesis._reach_conversation(
                            store, "op.respond", {"text": query, "principal": getattr(d, "person", None)})
                        if r.get("answer") is not None:
                            # Groundedness is read from the evidence rather than asserted:
                            # `activation.compose` returns "Nothing in the ontology resonates with
                            # that yet." with no citations when nothing activated, and `think`
                            # returns "Nothing resonates with that yet." with no `cited` key at
                            # all. A composed answer cites what it rests on; an empty result cites
                            # nothing, so a client can tell an answer from an empty result by the
                            # citation list alone.
                            cited = list(r.get("cited") or [])
                            text, grounded = r["answer"], bool(cited)
                    except Exception as e:
                        _note_fault(faults, "activation", e)
                if text is None:                                  # nothing activated -> keyed/semantic
                    text, grounded, cited = _answer_text(ember, query, k=k, store=store)
            cid = "ember-" + str(abs(hash(query)) % (10 ** 12))

            if req.get("stream"):
                self._stream(cid, text)
                return
            self._send({
                "id": cid, "object": "chat.completion", "created": CREATED, "model": MODEL_ID,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": len(query.split()), "completion_tokens": len(text.split()),
                          "total_tokens": len(query.split()) + len(text.split())},
                # `faults` is present only when an arm actually broke, so a healthy response is
                # byte-identical to before and no client has to learn a new field to stay correct.
                "x_agience": dict({"grounded": grounded, "cited": cited},
                                  **({"faults": faults} if faults else {})),
            })

        def _stream(self, cid: str, text: str):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.end_headers()

            def frame(delta, finish=None):
                payload = {"id": cid, "object": "chat.completion.chunk", "created": CREATED,
                           "model": MODEL_ID,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            frame({"role": "assistant"})
            # chunk on lines so an editor renders progressively but deterministically
            for line in text.splitlines(keepends=True):
                frame({"content": line})
            frame({}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, *a):  # quiet
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
