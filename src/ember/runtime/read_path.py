"""The read path: query -> route -> hit? answer local : miss -> refill -> answer.

This is Ember's whole loop, and it is the first thing worth building because it exercises
every part at once: the shards, the anchor routing, the hit/miss decision, the refill over
the relay, and the reasoning split.

The ordering matters and is not an optimisation:

  1. **Route.** Ask the geometry which cells could answer. This is computable, so "can I
     answer locally?" has a real answer rather than a hopeful one.
  2. **Answer locally if we hold the nearest cell.** Not "if we hold something" — the nearest
     cell is where a match would index into, so holding only outer probe cells is a miss.
  3. **Refill only what's missing.** The working set, not the corpus.
  4. **Reason over evidence.** The engine sees artifacts, never the network.

Disconnected is the default, not the fallback: with no cloud configured, step 3 is skipped and
the loop still terminates with an honest answer (possibly "I don't know", which is a fine
answer and an important one).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from mantle.shard.cache import LocalCache, Routing
# No typed gate sits at this import: the runner hands over the rows it retrieved and returns
# whatever the answerer made of them, declaring neither shape. The answerer is duck-typed —
# `.answer(query, rows)` — exactly as `crystal.discharge` duck-types its authority.


@dataclass
class Result:
    answer: Any                  # whatever the answerer produced — the type is born at the tekton
    routing: Routing
    refilled: List[str]          # regions pulled this turn (empty when disconnected/hit)
    served_offline: bool         # answered without touching the network


# A refill takes the missing region ids and returns those it actually obtained + imported.
# Injected rather than imported so the read path stays transport-agnostic: in-process peer
# for tests, the relay socket in production. The cache never learns what a socket is.
Refill = Callable[[Sequence[str]], List[str]]


def answer_query(
    cache: LocalCache,
    engine: Any,                 # duck-typed answerer: anything with .answer(query, rows)
    query_text: str,
    query_vec: Sequence[float] | np.ndarray,
    *,
    refill: Optional[Refill] = None,
    k: Optional[int] = None,
) -> Result:
    """Answer from local shards, refilling only on a genuine miss.

    When the caller does not state `k`, the read is not truncated to a fixed count: how many
    artifacts answer a question is a property of the question and the corpus, and the corpus has
    no bound to state in advance.

    Where relevance stops being signal is instead a measurement, taken by
    `prism.resolution.signal_end` — the same instrument `forgetting.Screen.lead` uses for "where
    does the present end", and `adaptive_cut` for "where does the result set end". It computes its
    own null, so a score series that does not separate returns everything rather than a number
    picked in advance: the difference between a derived cut and a cap.

    Nothing is truncated before that measurement runs: the search is asked for everything the
    cache holds — a count read off the cache itself, not a ceiling — because a top-N applied first
    would put the cut upstream of the instrument, and `signal_end` would only ever measure the
    inside of someone else's window.

    A `k` the caller does state is honoured exactly, and is the only way a fixed count enters —
    the same contract `projection.frame` has for the basis zoom: the read does not truncate the
    corpus by default, but a caller may state an instrument."""
    routing = cache.route(query_vec)
    refilled: List[str] = []

    # A miss is specifically: we do NOT hold the cell a match would index into.
    if not routing.hit and refill is not None and routing.missing:
        refilled = list(refill(routing.missing))
        if refilled:
            routing = cache.route(query_vec)   # re-route: we may hold the nearest cell now

    hits = cache.search(query_vec, k=(int(k) if k is not None else _everything(cache)))
    if k is None:
        hits = _signal_span(hits)
    # ROWS, not a declared type: id/text/score/region/vector is what an `EVIDENCE_CT` artifact
    # carries, so the runner passes the same shape the plane would and the answerer decides what to
    # make of it. (`lumen.composers.Evidence.of` is the far side of this boundary.)
    evidence = [
        {"id": item.id, "text": _as_text(item.content), "score": score, "region": item.region,
         "vector": cache._vectors.get(item.id)}
        for item, score in hits
    ]
    answer = engine.answer(query_text, evidence)

    # A miss that survives refill leaves the answer provisional, whatever the engine says.
    #
    # `search` legitimately returns items from the outer probe cells, and an engine with no
    # corpus cannot tell that they came from the wrong neighbourhood — it just sees spans.
    # Left alone, the engine answers with confident, well-formed content pulled from a cell
    # that happens to be held but is not the right neighbourhood.
    #
    # The runner holds cells, knows which one a match would index into, and knows it does not
    # have it, so it states that instead of guessing. Evidence found is still cited — "here is
    # what I have locally, and it is not the right neighbourhood" is more useful than silence.
    #
    # The miss is built via `type(answer)(...)` rather than a named `Answer` class: the content
    # type is born at the tekton, so the honest miss is expressed in whatever type the answerer
    # just produced. This is the same duck-typing `crystal.discharge` uses for its authority.
    if not routing.hit and answer.grounded:
        answer = type(answer)(
            text=("I don't hold the artifacts that would answer that. "
                  f"({len(routing.missing)} region(s) missing"
                  f"{', and no channel to fetch them' if refill is None else ''}.)"),
            grounded=False,
            cited=answer.cited,
            read={**answer.read, "provisional": True,
                  "missing_regions": routing.missing, "held_regions": routing.held},
        )

    return Result(answer=answer, routing=routing, refilled=refilled,
                  served_offline=not refilled)


def _everything(cache: LocalCache) -> int:
    """How many readable items this cache holds — "no truncation", stated as the count `search`
    needs to hear it.

    This is a count, not a limit: `LocalCache.search` takes a `k` and slices `scored[:k]`, so
    "give me everything" has to be spelled as a number. It is read off the cache
    (`summary()["readable"]` — the items that actually carry a vector, which are exactly the ones
    `search` can score), so it moves with the corpus instead of bounding it. An overestimate here
    truncates nothing; the floor of 1 exists because `scored[:0]` would return nothing at all on
    an empty cache, and an empty read is already the correct answer there."""
    try:
        return max(1, int(cache.summary().get("readable") or 0))
    except Exception:
        return 1


def _signal_span(hits):
    """The leading hits that are signal, per `prism.resolution.signal_end` on their own scores.

    Fails open: if the instrument cannot be reached or does not resolve, every hit is returned —
    the read is not narrowed by a measurement that did not run.

    `signal_end` is given no frame, because these are cosine scores — a column, not the ordered
    (T, F) evidence — and its contract takes a score column on its own terms rather than a frame.
    It answers from the column's own statistics against a computed null, which is why it can
    return the whole set when nothing separates."""
    if not hits:
        return hits
    try:
        from prism.resolution import signal_end
        n = signal_end([float(s) for _item, s in hits])
        n = int(n)
    except Exception:
        return hits
    if n <= 0 or n >= len(hits):
        return hits                      # nothing separated — the whole set IS the reading
    return hits[:n]


def _as_text(content: bytes) -> str:
    """Best-effort text view. Ember holds opaque bytes by design — plenty of what it caches
    is ciphertext or binary it has no business decoding — so undecodable content is reported
    as such rather than mangled into mojibake."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return f"<{len(content)} bytes, not utf-8>"


__all__ = ["Result", "Refill", "answer_query"]
