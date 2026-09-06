"""The delegate — one agent's cognition, owned by that agent and shared with no one.

Cognitive state is per-delegate: its own Screen, its own tick, its own taught-triple cache. Three
properties follow, and each is why the state is held at this grain rather than per process:

  1. **Attention stays with its witness.** `Screen.read()` scores every trace by geometric
     similarity, so a shared screen would surface delegate A's attention to `wolf.n.01` inside
     delegate B's query for `dog.n.01` as B's own present tense, for something B never observed.
     `forgetting.py`'s thesis is that the residual carries its witness ("I remember because I was
     there") and is therefore *provenance*; a per-delegate screen keeps first-hand memory
     first-hand.

  2. **Subjective time is per-observer.** The tick is not a clock: one tick is one observation
     step, the observer's own proper time. On one shared counter, a delegate handling 1000 turns
     alongside one handling 2 puts the quiet delegate's trace (stamped tick 5) at `now = 1005` —
     residual `0.25 * exp(-998/40) ≈ 3.6e-12`, far below `read()`'s 0.02 cutoff — so a busy
     neighbour would decay a quiet agent's memory. Each delegate advances its own clock.

  3. **Forgetting rates are fitted per delegate.** `Screen.measure()` fits `tau_fast`/`tau_slow`
     over the stream it is given and writes them onto every trace. Two delegates interleaving
     unrelated concepts look artificially decorrelated, so a pooled fit hands both a short
     `tau_fast` neither earned. That is measurement contamination, which is why the fit is
     per-delegate.

## The grain: a delegate is not a principal

**Memory is a property of the delegate; privacy is a property of the person.** `private.<person>`
and `owner`/`visibility` are person-scoped, because privacy belongs to the human. The screen, the
tick, and the taught-triple cache are delegate-scoped: two delegates of one person are two
cognitions, and keying them on `principal` collides their learning on
`obs.<sha256(who+subj+rel+obj)>`.

## Provenance: the delegate carries a 4-tuple

Provenance tracks **origin · person · host · operator**:

    origin    the authority whose policy and economy govern this (agience.ai; peering with another
              origin needs a trust boundary + an exchange agreement)   -> token `iss`
    person    whose claim it is                                        -> token `sub`
    host      the environment it was observed on                       -> token `host_id`
    operator  the observation/morphism that produced it                -> the operator edge

The delegate holds the first three and stamps them; the operator is per-observation. The delegate's
own id is not a provenance term: trust does not depend on *which* of your delegates observed
something, while cognition does.

The word "origin" carries two senses in the workspace. `Delegate.origin` is the authority. `_origin`
on a stored row — `mantle/mesh/sync.py`, `_fleet/peers/71/ember/_sqlite_db.py` — is the authoring node
(`EMBER_NODE_ID`).
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

# The reserved non-person identity. A process with no configured principal is not a human, so an
# unattributed cognitive act is recorded under this identity rather than under someone's name —
# the same vocabulary `prism.principals.PROCESS_AUTHORS` uses.
#
# `prism.delegate` is the contract: these constants and `SCREEN_TYPE` are declared there, with no
# dependencies, so a persona or a bare host can agree about the shape without importing this
# runtime. Re-exported here for ember's own callers (genesis, signal).
from prism.delegate import LOCAL_ORIGIN, LOCAL_PERSON, SCREEN_TYPE  # noqa: F401

_CACHE: Dict[str, "Delegate"] = {}
_CACHE_LOCK = threading.Lock()


def _host_id() -> str:
    """The environment this delegate is running in. Distinct from the delegate and from the author
    of a row. Falls back to the node id, the closest identity available."""
    return (os.getenv("EMBER_HOST_ID") or os.getenv("EMBER_NODE_ID") or "unknown-host").strip()


class Delegate:
    """One agent's cognition. Construct via `Delegate.get()`, never by sharing one across agents.

    Cognitive state (per-delegate, never shared): `screen`, `tick`, `obs`.
    Identity (carried, stamped onto what it authors): `id`, `origin`, `person`, `host`.
    Infrastructure (shared, injected — a store is durable truth, not cognition): `store`.
    """

    def __init__(self, store, *, id: str, person: str, origin: str = None, host: str = None,
                 capacity: int = None):
        from ember.signal import forgetting

        self.id = id
        self.person = person
        self.origin = origin or LOCAL_ORIGIN
        self.host = host or _host_id()
        self.store = store

        # ── cognition. One screen, one clock, one memory — this delegate's own. ──
        # The store is passed so the screen can read its own decay rates in the corpus basis: the
        # rates are fitted against `geom.corpus-basis`, which is an artifact. Without it
        # `measured_rates` has no reading to give, and the screen reports
        # `rates_source == "unmeasured"`. In the raw dense coordinate the fit resolves one mode and
        # `tau_fast == tau_slow`, a collapsed timescale that would carry the word "measured" (see
        # `forgetting.measured_rates`).
        self.screen = forgetting.Screen(witness=self.id, capacity=capacity, store=store)

        # ── the accumulating screen — the beam this delegate has seen, pooled ────────────────
        # entroptics is a streaming instrument, so a read certifies as samples accumulate. A single
        # turn is ~34 rows against a 195-dim basis, where the certified band `sqrt(F/T) + F/T` is
        # ~16 and the interval spans the whole range. Measured on the live corpus:
        #
        #     1 turn    T= 34  band=16.26  interval [2,195]
        #     3 turns   T= 88  band= 7.41  interval [6,195]
        #    20 turns   T=466  band= 2.13  interval [10,59]
        #
        # The band falls as sqrt(F/T), so the read certifies by conversing: the screen accumulates.
        #
        # Built lazily — the first frame names F — so a delegate that never conducts a beam pays
        # nothing and F is read rather than assumed.
        #
        # The mechanism is `ember.signal.pooling.PooledScreen`, with a second instance owned by the
        # node (`pooling.node_screen()`) for readings that belong to the box rather than to a
        # conversation, such as `op.measure`'s headroom. One accumulator, two owners.
        from ember.signal import pooling
        self._screen = pooling.PooledScreen("delegate.%s" % self.id)
        self.tick: int = 0
        self._tick_lock = threading.Lock()
        self.obs: List[Dict[str, Any]] = []
        self._obs_loaded: bool = False
        self._obs_lock = threading.Lock()

    # ── identity ─────────────────────────────────────────────────────────────────────────────
    def provenance(self) -> Dict[str, str]:
        """The 4-tuple minus the operator, which is per-observation and supplied by the caller."""
        return {"origin": self.origin, "person": self.person, "host": self.host}

    @property
    def screen_artifact_id(self) -> str:
        return "delegate.%s.screen" % self.id

    # ── the accumulating screen ──────────────────────────────────────────────────────────────
    def accumulate(self, plane, *, basis=None) -> str:
        """Pool one turn's beam onto this delegate's screen. `plane` is the ordered `(T, F)` frame.

        Never raises: accumulation is bookkeeping alongside an answer, so a screen that cannot pool
        leaves the answer standing. A plane whose F disagrees with the pooled basis is dropped
        rather than reshaped, because pooling two coordinate systems produces a confident spectrum
        of nothing.

        A drop is counted, logged, and reported — a dropped plane is data
        ([[absence-is-not-an-affirmative-claim]]) — and the outcome is returned, so a call site that
        wants to know can ask. A basis change (for example a corpus-basis rebuild moving `k` from
        195 to 280) therefore shows up as drops rather than as a pool that quietly stops growing
        while `accumulated()` keeps reporting the stale read.

        `basis` is the coordinate token the caller built the plane in (`pooling.coordinate_id`).
        Omitting it is recorded as unrecorded, which is not read as agreement."""
        return self._screen.place(plane, basis=basis)

    def accumulated(self) -> Optional[Dict[str, Any]]:
        """What the pooled screen currently holds — the read, the drops, and which basis.

        `None` only when nothing has ever been placed; an all-dropped screen reports itself. See
        `ember.signal.pooling.PooledScreen.accumulated` for the shape and for why the read is asked
        of `ember.optics`, the one module that reaches entroptics ([[one-instrument-enforced]])."""
        return self._screen.accumulated()

    # ── subjective time ──────────────────────────────────────────────────────────────────────
    def next_tick(self) -> int:
        """Advance this delegate's proper time. One tick is one observation step, for this observer
        only, so another delegate's activity leaves this counter where it was."""
        with self._tick_lock:
            self.tick += 1
            return self.tick

    def observe(self, concept) -> None:
        """Drop a concept into this delegate's screen as a fresh `is`, witnessed by this delegate."""
        self.screen.observe(concept, tick=self.next_tick(), witness=self.id)

    def observe_field(self, pairs) -> None:
        """Observe a whole turn's fired field — `[(concept, energy), ...]` — advancing the clock once.

        A turn is one observation step for this observer, however many concepts it lit. The energies
        are the turn's own measured shares and sum to 1, so every turn deposits the same total and a
        busy turn carries the same weight as a focused one. See `activation._observe_field` for why
        the whole field is observed rather than one picked lead."""
        self.screen.observe_field(pairs, tick=self.next_tick(), witness=self.id)

    # ── taught triples (delegate-scoped; the store rows stay person-scoped) ───────────────────
    def remember_obs(self, doc: Dict[str, Any]) -> None:
        """Add a taught offer, replacing any prior version with the same id — the offer's identity is
        its (context, operator), so a new value is a new version of the same artifact, and the field
        holds only the head."""
        oid = doc.get("id")
        with self._obs_lock:
            if oid:
                self.obs = [o for o in self.obs if o.get("id") != oid]
            self.obs.append(doc)

    def load_obs(self) -> None:
        """Load this delegate's taught triples from the store, once, in the background.

        The loaded flag is set after the load completes. A reader inside the load window therefore
        sees an unloaded delegate rather than a loaded one holding an empty list.
        """
        with self._obs_lock:
            if self._obs_loaded:
                return

        def _load():
            got: List[Dict[str, Any]] = []
            try:
                from mantle.db.typed_fetch import list_by_content_type
                from prism.grounding import TRIPLE_TYPE
                mine = "private.%s" % self.person
                for a in list_by_content_type(self.store.artifacts, TRIPLE_TYPE):
                    # Person-scoped on disk; delegate-scoped in memory. A row authored by another
                    # delegate of the same person is still that person's memory and is legitimately
                    # recalled; what stays unshared is attention (the screen).
                    #
                    # The filter is a raw string compare of `collection_id` against `mine`: this
                    # enumerates every triple in the store and keeps the ones whose collection
                    # string matches. The light-cone — `mantle.db.access.can_read` /
                    # `filter_visible`, the mechanism that answers "may this reader see this
                    # collection" — is not consulted on the recall path; its one live caller is
                    # `mantle/shard/keyed.py`.
                    #
                    # There is no ownership in the model: a collection is public or it is granted,
                    # and `is_creator` grants nothing. The string compare agrees with the grant
                    # whenever one principal exists, which is the case on the served chat, where
                    # every visitor resolves to the same `self.person` because no identity flows
                    # from the BFF. With two principals, this line is the read-access decision.
                    if a.get("collection_id") == mine and a.get("operator") and a.get("subject"):
                        got.append(a)
            except Exception:
                pass
            with self._obs_lock:
                self.obs.extend(got)
                self._obs_loaded = True

        threading.Thread(target=_load, daemon=True).start()

    def obs_matching(self, *, relation: str = None, obj: str = None,
                     subject: str = None) -> List[Dict[str, Any]]:
        from crystal.ontology import driver as wn
        self.load_obs()

        def lemma(v: str) -> str:
            return wn.morphy(v, wn.VERB) or v

        with self._obs_lock:
            rows = list(self.obs)
        out = []
        for f in rows:
            if obj and f.get("object") != obj:
                continue
            if subject and f.get("subject") != subject:
                continue
            if relation and lemma(f.get("relation", "")) != lemma(relation):
                continue
            out.append(f)
        return out

    # ── persistence: attention and subjective time survive restart, together ──────────────────
    def save_screen(self) -> bool:
        """Persist traces and the tick as one artifact.

        They are saved and restored together. Traces without the tick would come back stamped in a
        frame whose clock has been reset to 0: every restored trace would sit in the future,
        `amplitude()` would clamp `dt` to 0, and the whole screen would read as freshly observed —
        the backwards-time hazard `ember/signal/forgetting.py` documents.
        """
        try:
            doc = self.screen.serialize()
            doc.update({
                "id": self.screen_artifact_id,
                "content_type": SCREEN_TYPE,
                "state": "committed",
                "tick": self.tick,
                "delegate": self.id,
                "origin": self.origin,
                "host": self.host,
                # Private working state, gated by the owner's grant on their private collection
                # rather than by a flag. That collection and its grant exist by the time a screen is
                # written, because learn/chat ran first.
                "collection_id": "private.%s" % self.person,
                "collections": ["private.%s" % self.person],
            })
            self.store.artifacts.put_artifact(doc)
            return True
        except Exception:
            return False

    def load_screen(self) -> bool:
        try:
            doc = self.store.artifacts.get_artifact(self.screen_artifact_id)
            if not doc:
                return False
            tick = int(doc.get("tick") or 0)
            n = self.screen.restore(doc, witness=self.id)
            with self._tick_lock:
                self.tick = max(self.tick, tick)     # never let the clock run backwards
            return bool(n)
        except Exception:
            return False

    # ── construction ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def get(cls, store, *, person: str = None, id: str = None, origin: str = None,
            host: str = None, restore: bool = True) -> "Delegate":
        """Resolve a delegate, caching one instance per id per process.

        The cache is keyed by delegate id rather than by person, because two delegates of one
        person are two separate cognitions and each holds its own screen.
        """
        person = (person or os.getenv("EMBER_PRINCIPAL") or LOCAL_PERSON).strip()
        did = (id or os.getenv("EMBER_DELEGATE_ID") or ("d." + person)).strip()
        with _CACHE_LOCK:
            d = _CACHE.get(did)
            if d is not None and d.store is store:
                return d
            d = cls(store, id=did, person=person, origin=origin, host=host)
            _CACHE[did] = d
        if restore:
            d.load_screen()
        return d




# ── test seam ────────────────────────────────────────────────────────────────────────────────────
def reset_cache() -> None:
    """Drop every cached delegate. For tests; production code reaches delegates through `get`."""
    with _CACHE_LOCK:
        _CACHE.clear()
