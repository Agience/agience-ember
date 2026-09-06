"""The activation layer: recognition over the ontology (which concepts a signal fires) and the
rendering primitives around it. Ember is a runner ([[ember-is-a-runner]]), so this module is the
measurement layer; the conversation acts live in `lumen/conversation.py` and reach back into it over
the ground plane, as do the mesh signal runtime and the serve warm-up.

  * Seeding     — a signal (text / artifact / explicit map) -> weighted senses (`seeds_from_text`,
                  `seeds_from_artifact`, `_seed_field`).
  * Spread      — each seed up its hypernym DAG by the propagation kernel (`spread_seeds` via
                  `prism.propagation.spread_graph`), gap-gated to the mass horizon.
  * Rank/recall — activation × log1p(IC) plus this delegate's forgetting-screen memory
                  (`rank_fired`, `recall`); `recognize` / `activate` are the entrypoints, so a peer
                  message, a sensor reading, a mesh-applied artifact and a local query all enter by
                  the same door.
  * Render      — the keyed inverse (signal -> language): `_render_concept` / `compose`, lossless
                  because it is identity-keyed, and the instrument read of the result (`vertex_field`,
                  `output_membrane`).

The geometry decides what fires: there are no stopword lists, no `is_question()`, no part-of-speech
rules. Tense is amplitude on the forgetting screen (`ember/signal/forgetting.py`), not a flag.
"""
from __future__ import annotations

import math
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from prism import law as _law


# ── fire-and-forget writes: an answer (a read) must never block on a contended store write ───────
def _async_write(fn) -> None:
    threading.Thread(target=lambda: _swallow(fn), daemon=True).start()


def _swallow(fn) -> None:
    try:
        fn()
    except Exception:
        pass


# The ontology tokeniser has one home, `crystal.ontology.lookup`, and the alphabet it splits on is
# the lexicon's own. Re-typing the class here is how this file came to hold a copy that treated
# every non-ASCII letter as a separator: `Zürich` reached seeding as `rich`.
from crystal.ontology.lookup import lemma_tokens as _lemma_tokens


def _tokens(text: str) -> List[str]:
    return _lemma_tokens(text)


# ── Seeds: the several ways a signal can enter activation ────────────────────────────────────────
# Tokenization is one seed producer among several, not the entrypoint. Seeding is separate from
# spreading so that an arbitrary signal — a peer message, a sensor reading, a mesh-applied artifact —
# activates this delegate's screen without having to arrive as local text.
def seeds_from_text(text: str, *, info=None) -> Dict[str, float]:
    """Query tokens -> WordNet noun senses weighted by SemCor sense-frequency: the absolute count of
    how often each word is used in that sense (`lemma.count()`, tagged from SemCor, landed on the
    store by the `op.source.wordnet` tekton). Summed across tokens, so co-activation is context.

    The count is what separates a topic from rare junk. A common word used commonly (`dog`=42)
    out-weighs the rare noun reading of a function word (`me`->`Maine`=0, `does`->`Department of
    Energy`, `is`->`ice`), so "tell me about dogs" leads on dog(43) and "what does a cat say" on
    cat(19). A sense with no measured count falls back to its sense-rank prior (`1/(1+rank)`,
    most-frequent-sense order). Morphy handles plurals (`dogs`->`dog`, `wolves`->`wolf`) and
    single-character-lemma matches are dropped (`is`->`i`->iodine) — both via
    `match.wn_synsets_for`."""
    from crystal.ontology import driver as wn
    from ember.ontology import match as _match
    seeds: Dict[str, float] = {}
    raw = _tokens(text)
    n, i = len(raw), 0
    while i < n:
        # Multi-word lemmas are one concept. WordNet carries compound entries ("Albert Einstein",
        # "baseball bat", "polar bear") whose meaning is not the sum of their words; firing only the
        # singles grounds "who was Albert Einstein" on `Albert` (Prince Albert). Longest-match the
        # query's own tokens against the lexicon — the source's multi-word entries do the tokenizing,
        # so there is no phrase list — and a compound wins those tokens, else the single word stands.
        #
        # The window has no typed bound. `crystal.ontology.driver.entry_prefix_exists`, reached via
        # `match._wn_prefix`, asks the lexicon whether any entry still begins with this window; the
        # walk extends while one does and stops when none does, which is exact and linear. A typed
        # maximum would have to be wrong somewhere: `united_states_of_america` is four tokens and
        # `american_standard_code_for_information_interchange` is six.
        tok, base, names = None, None, []
        _best = None
        j = i + 1
        while j <= n:
            surface = "_".join(raw[i:j])
            if j - i >= 2:
                _ss = _match.wn_synsets_for(" ".join(raw[i:j]))
                if _ss:
                    _best = (" ".join(raw[i:j]), _ss, j)      # longest match wins: keep looking
            if not _match._wn_prefix(surface):
                break                                         # nothing longer can match
            j += 1
        if _best:
            tok, names, i = _best[0], _best[1], _best[2]
            base = tok
        if tok is None:
            tok = raw[i]
            base = wn.morphy(tok, wn.NOUN) or tok        # dogs -> dog, wolves -> wolf
            names = _match.wn_synsets_for(tok)           # morphy + single-char-lemma drop, sense-ranked
            i += 1
        # Within a word, which sense — `sense_prior`, the one within-word prior, which
        # `match.fired_field` also reaches for its bare-word case. Computed over the WHOLE sense
        # list, because the rank half is positional: `1/(1+k)` needs k to be the sense's index in
        # its word's own most-common-first ordering, and calling this per sense would make every
        # untagged sense rank 0.
        _priors = sense_prior(names, tok, base)
        for k, name in enumerate(names):
            try:
                s = wn.synset(name)
            except Exception:
                continue
            sense_w = _priors[k]

            # ── Across words, how much the word narrows the field ─────────────────────────────
            # `sense_w` alone has a coverage cliff: every untagged rank-0 sense scores exactly 1.0.
            #
            #     who -> world health organization  semcor=0  1.000
            #     me  -> maine                      semcor=0  1.000
            #     einstein                          semcor=0  1.000
            #     photosynthesis                    semcor=0  1.000
            #     cat                               semcor=18 19.000
            #
            # `who` and `einstein` are then numerically identical and the answer to "Who was Albert
            # Einstein" comes down to insertion order. SemCor is a small hand-tagged 1990s corpus, so
            # its coverage is common nouns and almost no proper nouns or technical terms.
            #
            # The measure with no cliff is information: I(x) = -log2(df/N), read off this corpus's
            # own index for every word that appears in it. A word in nearly every document narrows
            # nothing (`the` = 0.77 bits); a rare one narrows a lot (`photosynthesis` = 14.37,
            # `einstein` = 13.04, against `who` = 7.37). One measurement, no list.
            #
            # This is not IDF-as-preference. Raw IDF ranks rarity alone and makes `me`->`Maine` win,
            # ignoring which sense is actually used. Here the two are multiplied: information scales
            # the word, the SemCor/rank prior picks the sense. Rarity cannot outrank usage; it only
            # separates words that usage could not see. `None` (word absent from the index, or no
            # index at all) leaves the weight untouched.
            bits = info(tok) if info is not None else None
            seeds[name] = seeds.get(name, 0.0) + (sense_w * bits if bits is not None else sense_w)
    return seeds


def sense_prior(names, token: str, base: Optional[str] = None) -> List[float]:
    """Within a word, which sense — the SemCor count where the source tagged one, else sense rank.

    THE ONE within-word prior. `match.fired_field` reaches this too, for the case its cross-word
    coherence provably cannot answer: a bare single-word need has nothing to agree with, so there
    is no context to disambiguate from and a prior is the only honest answer.

    Both are DATA rather than a tuned knob:
      * SemCor is a hand-tagged corpus, so `(1 + count)` is measured usage. It is small and 1990s,
        so its coverage is common nouns and almost no proper nouns or technical terms — which is
        why the fallback exists rather than a zero.
      * WordNet lists a word's senses most-common-first, so `1/(1+rank)` is the source's own
        ordering. Measured on the kindergarten battery, flat weights give 14.6% primary-sense
        against 9/10 with the prior — a bare "dog" resolves to `frump` without it.

    Every sense keeps weight, so a single-word need still SHOWS its ambiguity; the prior only ranks
    the commonest above the rarest where the corpus says so.

    `match.fired_field` reaches this for the same case, so the two seeders answer it identically.
    """
    out: List[float] = []
    wanted = {t for t in (token, base) if t}
    for k, name in enumerate(names):
        try:
            # `wn` was not in scope here until 2026-08-25, and nothing said so. This module
            # imports `crystal.ontology.driver as wn` inside OTHER functions, never at module
            # scope — and it must stay function-local, because `match` reaches `activation` the
            # same way and one eager edge in either direction closes the cycle. This function
            # simply never had one.
            #
            # The `except Exception` below caught the resulting `NameError` on EVERY call and
            # substituted the positional fallback, so this returned `[1.0, 0.5, 0.333, …]` for
            # every word — measured, not inferred. The SemCor count prior this function exists to
            # provide had never once been applied, while the docstring above describes it working.
            from crystal.ontology import driver as wn

            s = wn.synset(name)
        except Exception:
            out.append(1.0 / (1.0 + k))
            continue
        cnt = max((l.count() for l in s.lemmas()
                   if l.name().lower().replace("_", " ") in wanted), default=0)
        out.append((1.0 + cnt) if cnt > 0 else 1.0 / (1.0 + k))
    return out


def _word_information(store):
    """Deprecated alias — the definition moved to `ember.ontology.information.word_information`.

    Kept as a name because this module's own prose refers to it, but it holds no implementation:
    `match.fired_field` had a SECOND, incompatible definition of the same quantity (natural log,
    clamped at 1.0) and the two are now one function. See that module for why bits."""
    from ember.ontology.information import word_information
    return word_information(store)


def _seed_field(store, text: str) -> Dict[str, float]:
    """The conversational seeder: SemCor-count-weighted senses (`seeds_from_text`) scaled by how much
    each word narrows this corpus (`_word_information`).

    Rarity alone is the retrieval seeder's measure, not this one. `match.fired_field` weights by
    IDF = rarity, and for a conversational turn that is backwards: the rare noun reading of a function
    word (`me`->`Maine`, `does`->`DoE`) out-weighs the common real topic. The SemCor count is a corpus
    measurement of actual usage, landed through the source tekton, and it picks the sense.

    Both measurements are needed. SemCor alone seeds every word the tagged corpus never saw at
    exactly 1.0, so `who` ties with `einstein`; the store carries the corpus's own index, which says
    how much each word narrows the field."""
    return seeds_from_text(text, info=_word_information(store))


def seeds_from_artifact(art: Dict[str, Any]) -> Dict[str, float]:
    """An artifact as a signal. This is the seam that lets something arriving from a peer activate
    a delegate rather than sit inert in a store.

    Reads the structured fields first (`lemmas`, and the triple's `subject`/`object`, which are the
    artifact's own statement of what it is about) and falls back to its context/content text. A
    wn-* artifact seeds its own synset directly — no round trip through a surface string."""
    seeds: Dict[str, float] = {}
    if not art:
        return seeds
    # The artifact's own id seeds it, whatever source it came from. `cn-cow` carries its meaning in
    # its id; asking its content string what it is about instead reads back the citation ("cow -
    # ConceptNet 5.7 concept /c/en/cow"). `wn._n` strips our prefix when it is there and passes a
    # foreign id through whole, which is the name space `fired` uses.
    #
    # The id seeds only when it resolves to a concept. This door takes documents, messages and sensor
    # readings too, and `doc-4f1a` is not a concept name, so the store decides and nothing here
    # enumerates which sources exist. `_resolve` returns `None` rather than raising, and it is
    # `_resolve` rather than `_get_synset` because it answers on both substrates: an installed index
    # has no keyed artifacts to read, and `_get_synset` returns `None` for every name there.
    from crystal.ontology import driver as _wns
    aid = str(art.get("id") or "")
    if aid:
        _name = _wns._n(aid)
        if _wns._resolve(_name) is not None:
            seeds[_name] = seeds.get(_name, 0.0) + 1.0
    for field in ("subject", "object"):
        v = art.get(field)
        if isinstance(v, str) and v:
            for name, w in seeds_from_text(v).items():
                seeds[name] = seeds.get(name, 0.0) + w
    lem = art.get("lemmas")
    if isinstance(lem, (list, tuple)):
        for v in lem:
            if isinstance(v, str) and v:
                for name, w in seeds_from_text(v).items():
                    seeds[name] = seeds.get(name, 0.0) + w
    if not seeds:
        for field in ("context", "content", "name", "description"):
            v = art.get(field)
            if isinstance(v, str) and v.strip():
                # The whole field is seeded: the bound on how much of an artifact is read is the
                # artifact, which says what it says. A caller genuinely bounded by machine size
                # bounds it at its own call site against the one measured envelope, where the cut is
                # visible to whoever reads the result ([[no-arbitrary-caps]]). A cut inside the
                # seeder leaves no record in the returned seed map that a document was shortened.
                for name, w in seeds_from_text(v).items():
                    seeds[name] = seeds.get(name, 0.0) + w
                break
    return seeds


def _discharge(fired: Dict[str, float], ic, xi: float, gap: float) -> Dict[str, float]:
    """The gap jump: charge accumulates along the taxonomy, then discharges across a gap once.

    The jump distance is per vertex, computed from the charge that vertex holds.
    `prism.resolution.horizon(xi, gap, weight)` = `xi * ln(weight/gap)` is the discharge distance, and
    `weight` is what the field accumulated here. Live: `cow.n.01` holds 245.14 and buys 4.2476 nats;
    `moo.n.01` is 1.6040 away and fires. A concept the need barely reached holds almost nothing and
    cannot cross at all — below the corpus's own propagation floor a contribution does not propagate at any
    distance, so the periphery is gated by the physics rather than by a cap.

    A per-vertex distance is what makes associative edges usable at all. `spread_graph` takes one
    global `reach` for a whole walk, which hands every vertex the strongest signal's budget; with
    `related_to` in the graph (1.67 M edges, connecting everything to everything) that takes the field
    from 89 to 85,105 concepts.

    One pass, not a walk. A capacitor discharges across the gap; it does not recharge and jump again
    within the same event, and re-entering compounds the reach.

    A jumped hop is the least certain step in any path. `crystal.ontology.geometry.jc_tree_se` is the
    channel that would report that certainty and has no consumers, so a jumped concept ranks beside a
    walked one as though equally certain."""
    from prism.resolution import horizon as _horizon
    # Module-local imports: `spread_seeds` imports these inside itself, so they are not in scope here.
    # A NameError on every iteration would be swallowed by the `except Exception: continue` below and
    # the discharge would do nothing while the field looked untouched.
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    if not fired or xi is None or gap is None:
        return fired
    try:
        arts = wn._arts()
        conn = arts.db.read()
    except Exception:
        return fired
    out = dict(fired)
    for name, w in list(fired.items()):
        try:
            reach_w = _horizon(xi, gap, weight=float(w))
        except Exception:
            continue
        if not reach_w or reach_w <= 0.0:
            continue                       # this vertex holds too little charge to cross anything
        try:
            here = wn.synset(name)
            # Both directions: evidence has no preferred direction, so a concept whose evidence
            # arrives as an in-edge is reachable too.
            #
            # Keyed on the ids this name actually addresses. `wn._n` strips our prefix when it is
            # there and passes a foreign id through unchanged, so `fired` holds a union of bare
            # WordNet names and whole ids (`cn-cow`, `concept-952227…`); re-prefixing that union asks
            # for `wn-cn-cow` and matches zero rows, while `cn-cow` itself carries 637 incident edges.
            # See `driver.concept_ids`; both candidates are point lookups on `ix_e_src` / `ix_e_dst`.
            _ids = list(wn.concept_ids(name)) or [name]
            _q = ",".join("?" * len(_ids))
            rows = conn.execute(
                "SELECT label,dst FROM edge WHERE src IN (%s)" % _q, _ids).fetchall()
            rows += [(l, a) for l, a in conn.execute(
                "SELECT label,src FROM edge WHERE dst IN (%s)" % _q, _ids).fetchall()]
        except Exception:
            continue
        if here is None:
            continue
        for lab, dst in rows:
            lab, dst = str(lab), str(dst)
            if lab in ("hypernym", "instance_hypernym"):
                continue
            # An endpoint is a concept because the store says so, not because of an id prefix.
            # 8,392 hypernym edges and 26,158 edges in total point at `concept-*` ids, and ConceptNet
            # is the larger half of the ontology (1,165,110 vertices to WordNet's 676,225), so a
            # `wn-` prefix test drops most of the neighbourhood. `lemma:*` vertices are the one thing
            # that is not a neighbouring concept, and the ontology's own spec names them
            # (`wn.is_lemma_id`). `wn._n` strips the prefix by asking the spec how long it is, so
            # `cn-cow` stays `cn-cow` rather than becoming a different concept named `cow`.
            if wn.is_lemma_id(dst):
                continue                   # a surface form is not a neighbouring concept
            nb = wn._n(dst)
            if nb == name:
                continue
            try:
                far = wn.synset(nb)
                d = g.jc_tree(here, far, ic) if far is not None else None
            except Exception:
                continue
            if d is None or d != d or d > reach_w:
                continue                   # no measurable distance, or beyond what this charge buys
            got = float(w) * float(_law.similarity(float(d)))
            if got > out.get(nb, 0.0):
                out[nb] = got              # nearest reach wins, as in the taxonomic walk
    return out


def spread_seeds(seeds: Dict[str, float], *, spread: Optional[int] = None) -> Dict[str, float]:
    """Spread each seed up its hypernym lineage by exp(-JC), to the horizon rather than for N hops.

    Linear in the seed weight, which is why accumulating seeds first and spreading once is exactly
    equivalent to spreading per token: `w1*exp(-d) + w2*exp(-d) == (w1+w2)*exp(-d)`.

    Distance bounds the walk, because the propagator is written in distance. A step count measures the
    walk rather than the signal: four hops through a dense part of the ontology and four through a
    sparse part are nothing alike. A contribution stops where it can no longer clear the propagation floor,
    which is `prism.resolution.horizon(ξ, gap)` = 1.394 JC nats on the live corpus, with ξ measured
    rather than chosen. Improve the corpus and ξ moves; the reach moves with it and nobody edits a
    number. An explicit `spread` is honored for a caller bounding cost, which is a claim about a
    machine rather than about the signal.

    The walk covers the full hypernym DAG rather than the `canonical_parent` spanning tree, so dog
    reaches `canine` as well as `domestic_animal`. A spanning tree is what the coordinate needs for JC
    exactness; propagation has no such need and signals travel every path. This BFS walks all
    hypernyms, accumulating the true per-edge JC distance (`IC(child) - IC(parent)`) so a direct parent
    is one hop and is never routed through a distant tree-LCS. The propagation horizon prunes it, and the
    min-distance `seen` guard keeps each ancestor at its nearest reach across branches, so a node
    reached by several paths accumulates once, at its nearest distance."""
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    from ember.ontology import match as _match
    from prism.resolution import horizon
    ic = g.load_ic()
    # The reach is built from both of the corpus's own scales. A typed gap of 0.05 in place of the
    # measured one computes a horizon of 1.2195 nats, 28% short of the corpus's own diameter (1.6909),
    # so the spread stops inside the corpus. `None` from either scale means this corpus cannot say how
    # far a signal travels; the seeds then stand and nothing propagates, which is the honest reading
    # of an unmeasurable reach rather than an unbounded one.
    _xi, _gap = _match.xi(), _match.propagation_floor()
    if _xi is None or _gap is None:
        return dict(seeds)
    reach = horizon(_xi, _gap)
    # The is-a field via the graph-propagation primitive `prism.propagation.spread_graph`: a
    # nearest-reach BFS over the full hypernym DAG — `energy · similarity(accumulated JC distance)` at
    # each ancestor, gap-gated by the horizon. Prism owns the algorithm; ember supplies the graph
    # (hypernym neighbours) and the kernel. A node reached by several paths accumulates once, at its
    # nearest distance. The horizon prunes only when depth is unbounded; an explicit `spread` bounds
    # hop count, which is a cost claim, so it does not also apply the propagation floor reach.
    from prism import propagation as _prop

    def _neighbours(name: str):
        try:
            s = wn.synset(name)
        except Exception:
            return ()
        if s is None:
            return ()
        return [(p.name(), max(g.ic_of(s, ic) - g.ic_of(p, ic), 0.0))
                for p in s.hypernyms() + s.instance_hypernyms()]

    seed_pairs = []
    for name, freq in seeds.items():
        try:
            resolves = wn.synset(name) is not None
        except Exception:
            resolves = False
        if resolves:
            seed_pairs.append((name, freq))
    fired: Dict[str, float] = _prop.spread_graph(
        seed_pairs, _neighbours, kernel=_law.similarity,
        reach=(reach if spread is None else float("inf")), max_steps=spread,
    )
    fired = _discharge(fired, ic, _xi, _gap)
    # ── The signed lateral couplings — the mix reads the sign off the relation, which is data, and
    # decides nothing per relation. The is-a field above is the +1 attraction; here the relation's own
    # `sign` places the concept on the far side of the coupling. Antonymy is −1, so the opposite lands
    # on the screen at negative energy: `hate` sits opposite `love` because the edge says so.
    from crystal.ontology.coupling import SEED_ETYPE_COUPLING
    conn = wn._conn()
    for name, freq in list(seeds.items()):
        for label, spec in SEED_ETYPE_COUPLING.items():
            sign = float(spec.get("sign", 0.0))
            if sign >= 0.0:                              # attraction is the IS-A field above
                continue
            try:
                # keyed on the ids this name actually addresses — see `driver.concept_ids`.
                # A ConceptNet seed has antonyms too.
                _ids = list(wn.concept_ids(name)) or [name]
                rows = conn.execute(
                    "SELECT dst FROM edge WHERE src IN (%s) AND label=?"
                    % ",".join("?" * len(_ids)), _ids + [label]).fetchall()
            except Exception:
                rows = []
            for (dst,) in rows:
                tgt = wn._n(dst)
                fired[tgt] = fired.get(tgt, 0.0) + freq * sign
    return fired


# `top=None` means "everything that fired", which is the honest default at the door every signal comes
# through — a peer message, a sensor reading, a mesh-applied artifact, a local query. A fixed width
# here costs a longer or richer signal proportionally more of itself, and because spreading
# distributes a seed's weight up its lineage the loss falls hardest on the general, well-connected
# seeds. Callers that want a short list for a human pass one, and that is presentation, not cognition.
def rank_fired(store, fired: Dict[str, float], *, top: Optional[int] = None,
               d=None) -> List[Dict[str, Any]]:
    """Rank fired concepts by activation * log1p(IC), and attach each one's stored artifact.

    The memory term: when a delegate is given, each candidate is also read against that delegate's own
    forgetting screen, so what it has recently attended to is more salient to it than to a delegate
    that has not. Reading the two-timescale screen here is what gives activation a past.

    The term is additive and reported separately (`present` / `past` / `memory`), never folded into
    `salience` — a caller must be able to see how much of a ranking came from the query and how much
    from this delegate's history. On an empty screen it is exactly 0 and ranking is unchanged.
    """
    from crystal.ontology import driver as wn
    from crystal.ontology import geometry as g
    ic = g.load_ic()

    def _ic_or_unit(name: str) -> float:
        """This concept's IC, or 1.0 when the lexicon does not own it — an unweighted activation. A
        zero would rank every non-WordNet concept last, and a raise would kill the turn."""
        try:
            return g.ic_of(wn.synset(name), ic)
        except Exception:
            return 1.0

    def score(name: str, act: float) -> float:
        """Activation weighted by the concept's information content. A concept that is not a WordNet
        synset still scores, on its activation alone.

        `wn.synset(name)` raises `WordNetError` for any id the lexicon does not own, and the ontology
        legitimately contains concepts that are not WordNet words: ConceptNet terms
        (`cn-low_by_mooing`) and the canonical ids the colimit mints when it merges duplicates
        (`concept-952227...`). After 5,473 certified merges, 8,392 hypernym edges point at
        `concept-*` ids, so a propagation that walks one would raise out of `sorted()` and take the
        whole answer with it. "Every concept is a WordNet synset" holds of the seed corpus and not of
        the ontology this system grows; a consolidated concept is more of a concept, not less, and is
        as answerable as its unmerged members.

        There is no IC term. The score is the activation.

        An `act * log1p(IC)` weight here and a `1.0 + log1p(IC)` weight in `match.propagate` are two
        shapes over the same quantity, disagreeing on IC-0 and off-lexicon nodes; neither is
        applied. `match.propagate` carries the full reasoning and the measured regression that
        absence reopens.

        The short version: on the corpus that runs, IC is near-inert as a weight — the basis is
        intrinsic IC, so nothing carries IC 0 and 91.4% of nouns sit at exactly 1.0 — and the
        intended replacement, corpus surprisal, is unavailable without an FTS index and would have
        gone silently constant rather than honestly absent.

        What is lost is the anti-generic guard: a generic concept sits a short distance from
        everything, so with distance alone it can outrank a specific one."""
        return act

    ranked = sorted(fired.items(), key=lambda kv: score(kv[0], kv[1]), reverse=True)
    # Memory is read for the whole ranked list, not for a shortlist cut by present activation first.
    # The memory term is additive and exists so that what this delegate has attended to can become
    # salient; pre-cutting by the present would let the past only re-order the present's shortlist,
    # never add to it, which is the one thing the two-timescale screen exists to do. Cost is a reason
    # to make the read cheaper — see `read_many` below — rather than to decide before reading.
    # ── The units reconciliation, derived ───────────────────────────────────────────────────────
    # A screen amplitude and a query activation are different units. A query activation for `dog`
    # reaches ~250, while a screen amplitude cannot exceed 1 because `delegate.observe_field` deposits
    # `salience/Σsalience` and every turn therefore deposits the same total. Adding them directly
    # asserts an equality two orders of magnitude off.
    #
    # The screen states its own unit, so nothing has to be picked: its quantity is a share of one
    # turn, which makes the commensurable form of this turn's reading likewise its share of this turn.
    # That is the same normalisation `_observe_field` applies when it puts this very field onto that
    # very screen — one definition, both directions.
    #
    # The query half's order is unchanged, because dividing a column by its own positive total is
    # monotone. What the normalisation buys is that the memory term enters at its true relative scale.
    _scores = {name: score(name, act) for name, act in ranked}
    _tot = sum(v for v in _scores.values() if v > 0.0)
    # One screen read for the whole turn, not one per concept. `recall(d, name)` per fired concept
    # walks every trace, and both factors are unbounded while the screen is designed to accumulate:
    # 1,281 concepts x 50 traces is 64,050 `_sim` calls in a single turn. `Screen.read_many` takes the
    # query-independent work once and groups traces by concept before evaluating the similarity, so
    # the reading is identical. `recall` remains for single-concept callers.
    _mem = {}
    if d is not None:
        try:
            _mem = d.screen.read_many([n for n, _ in ranked], d.tick)
        except Exception:
            _mem = {}                    # a memory read must never break an answer
    rows: List[Dict[str, Any]] = []
    for name, act in ranked:
        present, past = _mem.get(name, (0.0, 0.0))
        # `_tot <= 0` means nothing this turn carries a measured share; there is then no share to
        # take and the query contributes nothing — the memory is the only reading there is.
        share = (_scores[name] / _tot) if _tot > 0.0 else 0.0
        rows.append({"name": name, "act": act, "rank": share + present + past,
                     "present": present, "past": past})
    rows.sort(key=lambda r: -r["rank"])

    out: List[Dict[str, Any]] = []
    # `top=None` keeps everything that fired. A cap here is presentation, never cognition.
    for r in (rows if top is None else rows[:int(top)]):
        name = r["name"]
        # `wn.concept_artifact` asks the store which id space the name lives in. A typed
        # `"wn-" + name` fetches nothing for a ConceptNet or colimit-canonical concept — most of the
        # ontology — and the row then goes out with an empty `gloss` and a citation naming a corpus
        # the concept was never read from. `cn-cow`'s own artifact says `cite.conceptnet`.
        art = wn.concept_artifact(name, arts=store.artifacts) or {}
        # Nothing rounds a measurement here, because `salience` is consumed rather than displayed.
        # `compose` cuts on `_sal(a) > 0.0`, `_observe_field` skips `e <= 0.0`, and
        # `_stated_relations` builds its frame at amplitude `sqrt(salience)`, and all three read an
        # exact zero as absence. Rounding to three decimals maps every reading below 0.0005 to exactly
        # 0.0 and so manufactures that absence — a 5e-4 threshold with no author, falling hardest on
        # weakly-fired concepts, which is the population the forgetting screen exists to keep. A
        # caller that wants three decimals for a human rounds them where it prints.
        out.append({
            "concept": name,
            "activation": float(r["act"]),
            # Same rule as `score` above: a concept the lexicon does not own still carries its
            # measured activation. `wn.synset` raises for ConceptNet terms and for the colimit's
            # canonical ids, and 8,392 hypernym edges point at the latter.
            "salience": float(r["act"]) * _ic_or_unit(name),
            "present": float(r["present"]),
            "past": float(r["past"]),
            "memory": float(r["present"] + r["past"]),
            "gloss": (art.get("content") or "").strip(),
            # The artifact says where it came from; when it does not, the honest citation is the
            # record actually read — its own id — rather than the name of a corpus it may have
            # nothing to do with. `cn-cow` says `cite.conceptnet`; a colimit-canonical `concept-*`
            # says nothing, and citing `cite.wordnet` for it would assert a provenance nobody
            # measured ([[absence-is-not-an-affirmative-claim]]). Every WordNet row carries
            # `cited_from == "cite.wordnet"` in the store.
            "cited_from": art.get("cited_from") or art.get("id") or None,
            "operator": art.get("via") or art.get("operator"),
        })
    return out


def recall(d, concept_name: str):
    """(present, past) energy for a concept on this delegate's screen. Best-effort — a memory read
    must never break an answer."""
    try:
        from crystal.ontology import driver as wn
        s = wn.synset(concept_name)
        if s is None:
            return 0.0, 0.0
        present, past, _ = d.screen.read(s, d.tick)
        return float(present), float(past)
    except Exception:
        return 0.0, 0.0


def recognize(store, text: str, *, top: Optional[int] = None, spread: Optional[int] = None,
              d=None) -> List[Dict[str, Any]]:
    """Activation over the ontology from text: `_seed_field` -> `spread_seeds` -> `rank_fired`, with
    the memory term added when a delegate is supplied.

    Each firing concept carries its stored artifact (gloss, citation, operator) — the recognition and
    the material the conversation acts compose from.

    Seeds come from `_seed_field`, so the lead is the SemCor-count-weighted topic scaled by corpus
    information, which is a measurement rather than a most-frequent-sense argmax or an IDF rarity
    heuristic."""
    return rank_fired(store, spread_seeds(_seed_field(store, text), spread=spread), top=top, d=d)


def activate(d, seeds, *, source: str = "local", top: Optional[int] = None,
             spread: Optional[int] = None, observe: bool = True) -> Dict[str, Any]:
    """The entrypoint: activate this delegate's screen from any signal.

    `seeds` may be text (`str`), an artifact (`dict`), or an explicit `{synset_name: energy}` map, so
    a peer message, a sensor reading, a mesh-applied artifact and a local query all enter cognition by
    the same door, as activations on this delegate's own screen. There is no inbox and no queue: what
    happens next is decided by what resolves, not by arrival order.

    Returns the ranked activations plus `fired` — whether anything resolved (a lead exists), which is
    the computed null rather than a chosen floor. With nothing resolved the trace is still deposited;
    arrival is recorded.
    """
    if isinstance(seeds, str):
        seed_map = _seed_field(d.store, seeds)          # converged seeder when a store is present
    elif isinstance(seeds, dict) and any(isinstance(v, (int, float)) for v in seeds.values()):
        seed_map = {str(k): float(v) for k, v in seeds.items()}
    elif isinstance(seeds, dict):
        seed_map = seeds_from_artifact(seeds)
    else:
        seed_map = {}
    acts = rank_fired(d.store, spread_seeds(seed_map, spread=spread), top=top, d=d)
    lead = acts[0] if acts else None
    salience = float(lead["salience"]) if lead else 0.0
    if observe and acts:
        # The whole field that fired is observed, each at its measured share, on one tick. Nothing
        # measures that a recognition contributes exactly one observation, and observing the lead
        # alone starves the trajectory — the same forcing `compose` avoids (see `_observe_field`).
        _observe_field(d, acts)
    return {"source": source, "activations": acts, "lead": lead["concept"] if lead else None,
            "salience": salience,          # the measurement, unrounded — see `rank_fired`
            "fired": bool(lead),          # fired iff something resolved — the computed null, no chosen floor
            "delegate": d.id, "tick": d.tick}


def _word(concept: str) -> str:
    """The word a synset names — asked of the ontology, never parsed out of its id.

    An id-parse (`concept.split(".", 1)[0]`) reads the lemma out of an nltk-style name (`dog.n.01` ->
    `dog`), but OEWN names have no dot, so it returns the id unchanged and the answer renders as
    "Oewn-01044274-n — (Roman Catholic Church...) the celebration of the Eucharist", with the artifact
    itself perfectly fine (`title: "Mass"`, a clean gloss) and nothing raised. Parsing a name that no
    longer has the expected shape produces a wrong answer quietly.

    So ask the ontology for the lemma it holds. Fall back to the id-parse only when the synset cannot
    be resolved at all, and then say the id rather than a mangled guess."""
    try:
        from crystal.ontology import driver as wn
        syn = wn.synset(concept)
        if syn is not None:
            lem = syn.lemmas()
            if lem:
                return lem[0].name().replace("_", " ")
    except Exception:
        pass
    head = concept.split(".", 1)[0]
    return head.replace("_", " ") if head != concept else concept


# ── the keyed inverse: signal -> language, the exact half of the transducer conversion ────────────────
# Language is a transducer. Its forward map (`crystal.ontology.geometry.text_to_signal`) is JC-exact to
# ~1e-15 and the D=2048 hash ranks at Spearman 1.0 on the live corpus. Its inverse is exact because it
# is keyed rather than embedding: the resonance carries the concept's identity (its synset name), so
# the surface word and cited gloss come back by dictionary lookup, never by a lossy nearest-neighbour
# in the hash. `signal -> language -> signal` is therefore 1:1 and lossless in the keyed regime — the
# same reason [[keyed-vs-semantic-retrieval]] validates the dictionary over a vector cache. This is the
# far side of the membrane the instrument reads: the input screen measures which concepts a signal fires;
# this renders resolved concepts back to surface with no loss.
def _render_concept(store, name: str, ic, *, person: str = None) -> tuple:
    """One resolved concept -> (sentences, cites) by exact keyed lookup of its own definition, the
    cited gloss. Lossless: no nearest-neighbour and no template guess about meaning, because the word
    and gloss are the ontology's own, asked for by identity."""
    # The source's own definition, rendered whole — the concept condensed back to surface by the
    # language transducer's inverse. No hand-authored mold wraps it: a `{Word} — {gloss}. It is a kind
    # of {X}.` scaffold is a pre-conceived form. What is grounded is the source's definition, stated as
    # the source wrote it. An empty gloss grounds nothing, so the caller yields the computed null
    # rather than emitting an empty frame.
    from crystal.ontology import driver as wn
    # `wn.concept_artifact` asks the store which id space the name lives in rather than asserting one.
    # A typed `"wn-" + name` — the inverse of `wn._n` — fetches nothing for a ConceptNet or
    # colimit-canonical concept.
    art = wn.concept_artifact(name, arts=store.artifacts) or {}
    gloss = (art.get("content") or "").strip()
    cites: set = set()
    if not gloss:
        return [], cites
    # The record read is cited. `cited_from` for a WordNet row is the family anchor `cite.wordnet`,
    # which is a real artifact minted in `ember.genesis`, but it names the corpus rather than which of
    # 117,659 synsets this sentence was read from — source-granularity provenance where the answer
    # renders per-record glosses and reports a per-record citation count. This function already
    # fetched the artifact to read the gloss, so citing it is the evidence that was used rather than
    # new evidence. Additive: `cited_from` still rides along from `rank_fired`.
    # The artifact states its own id; fall back only to a derived candidate, never a typed prefix.
    aid = str(art.get("id") or (wn.concept_ids(name) or ("",))[0])
    if aid:
        cites.add(aid)
    sent = gloss[0].upper() + gloss[1:]
    if sent[-1] not in ".!?":
        sent += "."
    return [sent], cites


# ── the edge graph in the answer — a real edge, never a co-activation ──────────────────────────────
def _operator_concepts(store, text: str) -> List[str]:
    """The concepts the need's operator fires — the second terminal of the transistor.

    `what does a cat say` supplies a context (`cat`) and an operator (`say`). The context is what the
    answer is about; the operator is what is being asked of it, and `_stated_relations` needs both in
    its band or the content terminal is orthogonal to everything it can absorb (`meow` against `cat`
    measures 0.0 exactly).

    Read off the corpus, not from a list: every token is offered to the lexicon, and where the corpus
    has ever used that surface form as a content word (`attested`) all of its senses are returned.
    Tokens the lexicon does not know, and tokens the corpus has never attested, contribute nothing —
    a reading rather than an exclusion.

    The context's own tokens come back too, and that is harmless: `_stated_relations` unions this with
    `leads` before building the band, so a token that is already the subject adds no direction.
    Nothing is filtered by part of speech here — there is no POS tagger in this system and this is not
    the place to introduce one."""
    out: List[str] = []
    try:
        from crystal.ontology import driver as _wn

        def attested(word: str, senses=None) -> bool:
            """Has anyone ever used this surface form as a content word? SemCor counts, measured.

            Without this test, `what does a cow say` grounds `does` to the Department of Energy and
            `what is a cow` grounds `is` to iodine: taking noun senses of every token grounds each
            function word to its two-letter abbreviation sense, because a chemical symbol is a real
            synset and nothing consults frequency. A sense nobody has ever used is not a reading of
            the word.

            SemCor tags lemmas, so `do` and `be` are attested but the surface forms `does` and `is`
            are not — an inflected function word has no attested reading under its own spelling. That
            is a measurement about the corpus, not a stop-list; nobody named these tokens."""
            # Resolve once. Every resolution runs `freshness.stamp` -> `edge_mark`, a write per read,
            # so resolving a token in the caller and again here costs four tokens 20,440 sqlite
            # statements and 11.2 s — 82% of a whole turn. Taking the senses as an argument keeps that
            # work to one pass (`compose` 13.6 s -> 0.9 s, 1,729,689 calls -> 749,210).
            if senses is None:
                try:
                    senses = _wn.synsets(word, store=store) or []
                except Exception:
                    return False
            target = str(word).lower().replace("_", " ")
            for syn in senses:
                for lm in (syn.lemmas() or []):
                    if str(lm.name()).lower().replace("_", " ") == target and int(lm.count() or 0) > 0:
                        return True
            return False

        for tok in _tokens(text or ""):
            try:
                # Attestation decides whether the token is a word; the senses then supply the
                # coordinate. Filtering the senses by attestation instead removes the operator
                # itself — `say.n.01` has a SemCor count of zero as a noun ("she had her say" is
                # rare), while `say` is heavily attested as a verb, which is what says it is a
                # content word at all.
                #
                # `does`, `is`, `what`, `a` fail the test because SemCor tags lemmas: the corpus has
                # attested `do` and `be`, never the surface forms, so an inflected function word has
                # no attested reading under its own spelling. That is a measurement about the corpus,
                # not a stop-list — nobody named these four. The band this feeds decides which
                # relations are stated, so handing it the Department of Energy as the operator of
                # "what does a cow say" makes that question and "what is a cow" answer identically.
                # Resolve once and hand the senses to both consumers — see `attested`.
                try:
                    _senses = _wn.synsets(tok, store=store) or []
                except Exception:
                    continue
                if not attested(tok, _senses):
                    continue
                # Every attested sense is offered, of any part of speech and without a count.
                # Consolidating the operator puts `cn-say`'s knowledge on `wn-state.v.01` — a verb,
                # 1 edge -> 39 — while the noun readings of "say" (`say.n.01`, "a turn to speak",
                # and `oewn-14509110-n`) are nearly edgeless. Restricting to nouns, or to the first
                # two senses, hands the band a concept with no neighbourhood and the operator can
                # contribute nothing.
                #
                # A nominal restriction would only be needed if the coordinate were built from noun
                # hypernym paths; the frame is the exact union of the rows' own coordinate keys (see
                # `_stated_relations`), so a verb's nodes are simply more keys, not a different
                # space. The attestation test above is what bounds this: a word the corpus has never
                # used contributes nothing, which is a measurement rather than a chosen count.
                for ss in _senses:
                    n = ss.name()
                    if n not in out:
                        out.append(n)
            except Exception:
                continue
    except Exception:
        return []
    return out


def _stated_relations(store, leads: List[str], field: Dict[str, float],
                      operator: Optional[List[str]] = None) -> tuple:
    """The relations the answer's own band absorbs, stated as the store holds them.

    A relation is sayable because it **couples**. Nothing here tests and nothing here admits, because
    a gate that permits is still forcing even when it permits exactly the right things
    ([[free-will-is-intelligence-never-force-only-give]]):

      1. **The store's incident edges are read.** A read is mechanics, not a decision — this asks the
         corpus what it holds and gets an answer. Requiring a real edge is what keeps co-activations
         out, and it is not a gate: an edge the store does not hold produces no row to couple with.
         `wolf.n.01` and `department_of_energy.n.01` both fire on "what does a cat say", and nothing
         joins them because the corpus joins nothing.
      2. **The answer's band is its leads' own coupling subspace** (`match.tekton_basis` →
         `prism.frames.offer_basis`, the orthonormal span of the leads' coordinates).
      3. **The incident signal is one ordered (T, F) frame**, one row per incident edge: the far
         endpoint's ontology coordinate at the amplitude of its own measured salience (`√energy`, so
         `‖row‖²` is that row's energy — the same convention `projection.frame` uses for every other
         signal in this module).
      4. **`ember.optics.absorb_transmit(frame, basis=band)` splits it** into the band the answer
         absorbs and the residual it does not. Conservation is exact. What is absorbed is stated; the
         residual is not.

    A concept the propagation never fired has no measured energy, so its row is the zero vector and
    there is nothing of it to absorb. "The far endpoint fired this turn" is therefore not a condition
    anyone checks — it is what coupling means. The floor is not chosen and is not a small number: it
    is zero, and an absent reading is absence rather than a measurement (the same rule `compose`
    applies to zero-salience acts and `projection.frame` applies to absent coordinates). On the live
    store, absorbed-energy > 0 is exactly the set an admissibility predicate would admit, for every
    probe query: cat 6/15 rows, dog 19/155, einstein 42/940, france 2/34, and the same for "what does
    a cat say". The behaviour is reproduced by geometry rather than preserved by a leftover check.

    The coordinate is the raw dense one, and that is load-bearing. A coupling basis is `(2048, k)`
    unprojected `dense_vec`; `absorb_transmit` requires `basis.shape[0] == frame.shape[1]` and returns
    `None` otherwise, which is indistinguishable from a genuine non-coupling. So the rows here are
    stacked in the same raw coordinate the band lives in: `corpus_basis=True` gives `(T, 280)` and the
    split returns `None`, while the raw frame gives `(T, 2048)` and it couples. The rows are stacked
    here rather than through `projection.frame` for two reasons, both about honesty. `frame` drops
    rows it cannot place without saying which, so a returned matrix cannot be aligned back to its
    edges. And `frame` has no reading for fewer than `MIN_ROWS` samples, which is an instrument
    precondition (a noise-floor read needs samples) and does not apply to a projection onto a given
    basis — it returns `None` for "what is a cat" (3 leads) and "capital of france" (1 lead).

    There is no label whitelist. Naming the sayable relations (`hypernym`, `instance_of`, …) would
    impose a vocabulary ([[never-impose-knowledge-derive-it]]) and freeze out any label a later source
    brings. The label is data — the source's own word for the relation — and it is repeated verbatim.
    The ontology's lemma-entry edge (`lex:en`, `lemma:cat -> wn-cat.n.01`) needs no exclusion either:
    a `lemma:` id is not a concept and carries no ontology coordinate, so it cannot enter a frame.

    Returns `(triples, cites)` where a triple is `(src_word, label, dst_word)` in the stored
    direction, and `cites` are the far endpoints' artifact ids — real, resolvable artifacts. An edge
    is asserted only with the artifact it came from ([[absence-is-not-an-affirmative-claim]]: a
    relation with no citation is indistinguishable from one we made up)."""
    try:
        import numpy as _np
        from crystal.ontology import driver as _wn
        from crystal.ontology import geometry as _g
        from ember.ontology import match as _match
        from ember.optics import absorb_transmit as _absorb
        conn = store.artifacts.db.read()
        prefix = _wn._prefix(store)            # the ontology's own id prefix, from the spec
    except Exception:
        return [], set()                       # a store that cannot be read states no relations

    # ── 1. What the corpus holds — a read, not a decision ────────────────────────────────────────
    # (far_name, src_name, label, dst_name, far_artifact_id), in the store's own direction.
    incident: list = []
    for name in leads:
        aid = prefix + name
        try:
            for label, dst in conn.execute(
                    "SELECT label, dst FROM edge WHERE src=?", (aid,)).fetchall():
                far = dst[len(prefix):] if dst.startswith(prefix) else dst
                incident.append((far, name, label, far, dst))
            for label, src in conn.execute(
                    "SELECT label, src FROM edge WHERE dst=?", (aid,)).fetchall():
                far = src[len(prefix):] if src.startswith(prefix) else src
                incident.append((far, far, label, name, src))

            # The WordNet and ConceptNet halves of a concept are read separately, not joined.
            # `wn-cow.n.01` carries one outgoing edge (`hypernym -> cattle`) while `cn-cow` carries
            # 338 (`related_to` 182, `at_location` 103, `capable_of` 11), and no edge joins them, so
            # a WordNet lead states its taxonomy relations for every question asked of it.
            #
            # The anchor exists — `lemma:cow --lex:en--> wn-cow.n.01` and `lemma:cow --assoc:en-->
            # cn-cow` are both in the store — but the frame two blocks below is WordNet-only in two
            # independent places and would drop every joined row without saying so:
            #
            #   `_g.dense_vec(_wn.synset(edge[0]), ic)`  — `cn-moo` has no Synset, so v is None
            #   `field.get(edge[0], 0.0)`                — the fired field holds `wn-` concepts, so
            #                                              a cn row's amplitude is sqrt(0) = 0
            #
            # A join that contributes nothing while looking measured is worse than no join. What it
            # needs first is a coordinate and an energy for the ConceptNet side:
            # `crystal.ontology.seed_lattice.conceptnet_ic` derives the coordinate from ConceptNet's
            # own 221,203 `is_a` edges (cn-cow 0.6775, cn-moo 1.0000, generic-low/specific-high), and
            # the energy has to come through the anchor from the lemma the need fired.
        except Exception:
            continue                           # this lead contributes nothing rather than raising

    if not incident:
        return [], set()

    # ── 2. The band the answer absorbs — the leads' own coupling subspace, (F, k) ─────────────────
    # The band spans the leads and the operator both. A need has two terminals — what the answer is
    # about and what is being asked of it — and the band is the coupling subspace of both, or the
    # content terminal is orthogonal to everything the answer can absorb.
    #
    # A band built from `cow.n.01 / cattle.n.01 / bovine.n.01` alone is a taxonomy subspace and `moo`
    # is orthogonal to it, so every question about cow states the same four taxonomy relations however
    # it is phrased. That holds even with cow's edges grown from 1 to 91 (including `wn-moo.n.01`):
    # the knowledge is present and the band is what reaches it.
    _basis_of = list(leads) + [x for x in (operator or []) if x not in leads]
    band = _match.tekton_basis(_basis_of)
    if band is None:
        return [], set()                       # no coordinate to couple against — never a fabricated one

    # ── 3. The incident signal — one row per edge, at its far endpoint's measured energy ─────────
    # The operator's own field, spread and discharged exactly like the context's, so a junction is
    # measurable. `None` when the need names no operator, which is a different thing from an empty one.
    op_field = None
    if operator:
        try:
            # Exclude by lemma, not by synset. `_operator_concepts` grounds every token, so the
            # context's own senses come back in the operator list — `cow.n.01`, `cow.n.02`,
            # `cow.n.03` alongside `state.v.01`. Filtering only those in `leads` (two or three
            # concepts) leaves the rest, and they seed the operator field with the context's own word,
            # which charges the context's entire lineage: the operator's energies come back nearly
            # uniform (0.2-1.8), the strongest junctions are all generic ancestors — entity,
            # placental, mammal, organism, living_thing — and a junction weighted by a near-constant
            # is just the context's own ranking, so `beef` (e_ctx 188) beats `moo` (e_ctx 49).
            #
            # A junction means something only when the two signals are genuinely different signals.
            # The lemma is what makes them the same word, so the lemma is what excludes, measured off
            # the leads themselves rather than from a list of stop-tokens.
            #
            # It has to be the lemmas a sense answers to rather than the sense's own name.
            # `_word("overawe.v.01")` is "overawe", but that synset is a sense of the token "cow" — to
            # cow someone — so a name filter lets it through, the operator for `what is a cow` becomes
            # {overawe}, which reaches nothing near the animal, the junction goes to zero and the
            # definitional question states nothing. A sense of the context's own word is not an
            # operator.
            _lead_lemmas = set()
            for a in leads:
                try:
                    _lead_lemmas.update(str(l.name()).lower() for l in _wn.synset(a).lemmas())
                except Exception:
                    _lead_lemmas.add(_word(a))
            def _is_context(n):
                try:
                    return bool({str(l.name()).lower() for l in _wn.synset(n).lemmas()} & _lead_lemmas)
                except Exception:
                    return _word(n) in _lead_lemmas
            _op_seeds = {n: 1.0 for n in operator if n not in leads and not _is_context(n)}
            if _op_seeds:
                op_field = spread_seeds(_op_seeds)
        except Exception:
            op_field = None
    ic = _g.load_ic()
    rows: list = []
    sparse_rows: list = []                     # the frame in the exact (unhashed) coordinate
    lead_sparse: list = []
    for _ln in _basis_of:
        try:
            _lv = _g.sparse_vec(_wn.synset(_ln), ic)
        except Exception:
            _lv = None
        if _lv:
            lead_sparse.append(_lv)
    carried: list = []                         # the edge each row IS, so the split stays aligned
    for edge in incident:
        try:
            v = _g.dense_vec(_wn.synset(edge[0]), ic)
        except Exception:
            v = None
        if v is None:
            continue
        v = _np.asarray(v, dtype=float)
        if not v.any():                        # no coordinate is absence, not a measured zero
            continue
        # The junction: a row's amplitude is the product of the two signals' amplitudes. A need names
        # a context and an operator, and weighting a row by the context's energy alone answers "what
        # is a cow" for every question asked about a cow — every phrasing returns the same four
        # taxonomy relations.
        #
        # Amplitude is sqrt(energy), so two signals meeting at a vertex have joint amplitude
        # `sqrt(e_ctx) * sqrt(e_op)`, the correlation of the two fields there. It is zero unless both
        # reach it, which is what makes a junction a junction rather than a sum. Nothing is chosen:
        # the same `sqrt(energy)` convention the frame already uses, applied to both terminals.
        #
        # With no operator the second factor is absent and the row keeps its context amplitude
        # unchanged. "What is a cow" is a need with one terminal, not a need whose junction is empty,
        # and the two read differently.
        e = field.get(edge[0], 0.0)
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = 0.0
        if op_field is not None:
            try:
                eo = float(op_field.get(edge[0], 0.0))
            except (TypeError, ValueError):
                eo = 0.0
            e = math.sqrt(max(e, 0.0) * max(eo, 0.0))
        if not _np.isfinite(e) or e < 0.0:
            e = 0.0                            # negative energy is not a reading; it is a bug upstream
        rows.append(v * math.sqrt(e))          # amplitude is √energy, so ‖row‖² is the energy
        try:                                   # the same row in the exact coordinate — see the note below
            _sv = _g.sparse_vec(_wn.synset(edge[0]), ic)
            sparse_rows.append({k: val * math.sqrt(e) for k, val in (_sv or {}).items()})
        except Exception:
            sparse_rows.append({})
        carried.append(edge)
    if not rows:
        return [], set()

    # ── 4. The membrane — absorbed is stated, the residual is not ────────────────────────────────
    # The frame is built over the exact union of the rows' own coordinate keys: no hash, no `D`.
    # `dense_vec` hashes an unbounded set of node names into a fixed width so there is something dense
    # to do linear algebra on, and that width is a machine choice
    # (`crystal.ontology.geometry.D_BIG = 2048`, "a real node — the default"; `D_SMALL = 256` for a
    # Pi). Handed to the read as the frame's feature count, it decides what can be perceived. On cow's
    # incident-edge frame:
    #
    #     hashed          (76, 2048)  ->  resolvable = 0   contrast 0.34
    #     hashed, pruned  (76,  250)  ->  resolvable = 5
    #     exact sparse    (76,  270)  ->  resolvable = 5   contrast 1.80
    #
    #     distinct nodes on the paths: 270    hash columns used: 250   -> 20 nodes collided
    #
    # One constant does three separate harms: `k = 0` is an artefact of padding (the Marchenko-Pastur
    # noise floor computed for a matrix eight times wider than the data), twenty distinct concepts are
    # merged into shared columns, and the contrast is degraded 5x by the smearing. 76 is the corpus's
    # answer (cow's incident edges) and 270 is the corpus's answer (the nodes those rows touch).
    #
    # `sparse_vec` is the exact JC coordinate pre-hash — `{node: sqrt(IC(c) - IC(parent))}` keyed by
    # the actual nodes on the path — so a frame over the union of those keys is collision-free and its
    # dimension is derived, with conservation unaffected because an all-zero column carries no energy
    # ([[absence-is-not-an-affirmative-claim]]). `consolidate/diagram.py` builds frames the same way.
    #
    # The split is taken against the leads' band rather than against the frame's own self-resolution,
    # because a frame this shape cannot resolve itself:
    #
    #     frame (76, 2048)      principal_directions -> (2048, 0)      resolvable -> 0
    #     contrast 0.38   top_share 0.059
    #     absorb(self) k=0 -> 0 rows absorbed        absorb(band) k=3 -> 38 rows absorbed
    #
    # 76 rows in 2048 dimensions is massively underdetermined, so the instrument correctly reports no
    # structure above its noise floor and the answer states nothing at all. Is-a descent frames read
    # the same way at 5-8 rows. The band is not an imposition: it is the tekton's own tuning supplying
    # structure the frame is too small to carry.
    #
    # The coordinate is taxonomic by construction, and that bounds what any band in it can couple:
    # `sparse_vec` walks the hypernym path, so `cos(moo.n.01, cow.n.01) = 0.0` exactly while
    # `cos(moo.n.01, say.n.01) = 0.064`. An animal and a sound occupy disjoint support, so no band
    # containing `cow` absorbs `moo`, even though the corpus holds `cow --related_to-> moo` and the
    # propagation carried real signal across that gap (energy 49.27, earned by a discharge the
    # corpus's own propagation floor permitted). The limit is the coordinate, not the selector.
    _keys = sorted({k for d in sparse_rows for k in d})
    if _keys and lead_sparse:
        _W = _np.array([[d.get(k, 0.0) for k in _keys] for d in sparse_rows], dtype=float)
        _L = _np.array([[d.get(k, 0.0) for k in _keys] for d in lead_sparse], dtype=float)
        try:
            from prism.frames import offer_basis as _ob
            _band = _ob(_L)
        except Exception:
            _band = None
        if _band is not None and _np.asarray(_band).shape[0] == _W.shape[1]:
            split = _absorb(_W, basis=_band)
        else:
            split = _absorb(_np.vstack(rows), basis=band)
    else:
        split = _absorb(_np.vstack(rows), basis=band)
    if split is None:
        return [], set()                       # the split could not be taken; never fake a coupling
    absorbed, _residual, _k = split
    absorbed = _np.asarray(absorbed, dtype=float)
    coupled = (absorbed ** 2).sum(axis=1)

    # ── 5. The output screen — project on the way out, not only on the way in ────────────────────
    # Step 4 absorbs, which is the projection on the way in. Stating every row with `coupled > 0`
    # would leave the outbound leg with no instrument at all: "what is a dog" then states twenty-four
    # breed relations (poodle, corgi, dalmatian, pug…), every one a real cited edge, absorbed and
    # never screened. The corpus is not wrong there; the answer is unresolved.
    #
    # The absorbed rows are a frame — one row per relation, at the energy it absorbed — so the
    # outbound read is the same instrument as every other read here: `signal_end` over the ordered
    # absorbed energies, handed the frame so `ember.optics.resolvable` answers when it can certify and
    # the derived Otsu statistics answer when it cannot. Nothing is chosen: a set of relations that
    # genuinely does not separate is returned whole, and one that resolves to three is three.
    from prism import resolution as _res
    order = _np.argsort(-coupled)
    live_n = int((coupled > 0.0).sum())
    if live_n > 0:
        kept = order[:live_n]
        try:
            n_out = _res.signal_end([float(coupled[int(j)]) for j in kept],
                                    frame=absorbed[kept])
        except Exception:
            n_out = live_n                     # unreadable screen -> state what absorbed, unscreened
        order = kept[:max(1, int(n_out))]

    triples: list = []
    cites: set = set()
    said: set = set()                          # the rendered word-triples already stated
    undirected: set = set()                    # {src, dst} pairs — an inverse edge is the same fact
    for j in order:                            # strongest coupling first, screened on the way out
        if not coupled[int(j)] > 0.0:
            break                              # the rest lies wholly in the residual: nothing absorbed
        _far, s, label, dobj, endpoint = carried[int(j)]
        sw, dw = _word(s), _word(dobj)
        if sw == dw:                           # the two id families naming one concept — not a relation
            continue
        key = (sw, str(label), dw)
        if key in said:
            continue                           # the same statement, reached twice
        pair = frozenset((sw, dw))
        if pair in undirected:
            continue                           # the store holds both directions; one fact, said once
        said.add(key)
        undirected.add(pair)
        triples.append(key)
        cites.add(endpoint)
    return triples, cites


def vertex_field(d, acts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Read the response off the light-cone vertex, not the isolated query.

    The recognition is not an answer that terminates; it is a signal deposited back onto the field and
    read with everything else already live there. The field at the vertex is the incoming signal (this
    turn's fired concepts, `signal`) plus the live traces still cooling on this delegate's screen (the
    accumulated context — recent turns, `is` not yet settled to `was`, `memory`) plus the delegate's
    own offers whose ends are already active (`corpus`).

    The field selects: every concept live at the vertex is ranked by total energy, so a strong query
    dominates while a weak or contentless turn ("hello") is coloured by what the context has made
    salient."""
    now = int(getattr(d, "tick", 0))
    field: Dict[str, float] = {}
    src: Dict[str, str] = {}
    for a in acts:                                   # the incoming signal — the present
        n = a.get("concept")
        if not n:
            continue
        e = float(a.get("salience") or a.get("activation") or 0.0)
        if e >= field.get(n, 0.0):
            field[n] = e
            src[n] = "signal"
    try:
        for tr in list(d.screen.traces):             # the accumulated memory already at the vertex
            amp = float(tr.amplitude(now))
            # Only an exact zero is dropped: a measured zero is the absence of a reading, not a small
            # one. The amplitude is a reading on the screen's own decay curve, ranked a few lines
            # below and cut again downstream (`compose` -> `prism.resolution.signal_end`), so a
            # threshold here would be a third decision on the same quantity.
            if amp <= 0.0:
                continue
            n = tr.concept.name()
            if amp > field.get(n, 0.0):
                field[n] = amp
                src[n] = "memory"
    except Exception:
        pass
    # Our own corpus: attraction pulls in the delegate's learned offers. An offer whose subject or
    # object is already active on the field joins it, and its other end becomes an active signal on
    # the screen. The reasoning screen is therefore observation ⊕ our own memories and offers, and a
    # taught fact enters the moment either of its ends is on the vertex.
    try:
        d.load_obs()
        with d._obs_lock:
            offers = list(d.obs)
        # each field concept's energy, folded to its word, so an offer (which stores words) can find its
        # attracting end among the synset-keyed field.
        word_e: Dict[str, float] = {}
        for n, e in field.items():
            w = _word(n)
            if e > word_e.get(w, 0.0):
                word_e[w] = e
        for f in offers:
            ends = [str(f.get("subject") or "").lower(), str(f.get("object") or "").lower()]
            attractor = max((word_e.get(w, 0.0) for w in ends if w), default=0.0)
            if attractor <= 0.0:                      # neither end is active -> not attracted
                continue
            for w in ends:                            # the offer's inactive end is pulled onto the screen
                if w and word_e.get(w, 0.0) <= 0.0 and attractor > field.get(w, 0.0):
                    field[w] = attractor
                    src[w] = "corpus"
    except Exception:
        pass
    ranked = sorted(field.items(), key=lambda kv: -kv[1])
    return [{"concept": n, "energy": round(e, 3), "source": src.get(n, "signal")} for n, e in ranked]


def output_membrane(store, acts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The output screen: read the concepts the answer names back through the same instrument the input
    screen uses (`ember.signal.projection` -> `ember.optics`), so the response is measured rather than
    only emitted. The input screen measures which concepts a signal fires; the output screen measures
    whether the concepts about to be said are one coherent, resolved thing and how many
    distinguishable things they are. Returns a measurement, never text: `k_signal` (distinguishable
    modes), `coherence`, and `coherent` (the certified single-thing signal). Best-effort — a frame
    that cannot be read reports `read: False` rather than raising, and never blocks an answer."""
    try:
        from ember.signal import projection as SC
        names = [a["concept"] for a in acts if a.get("concept")]
        if not names:
            return {"read": False}
        rd = SC.read(store, names)
        if not rd.get("frame"):
            return {"read": False, "rows": rd.get("rows")}
        return {"read": True, "k_signal": rd.get("k_signal"),
                "coherence": round(float(rd.get("coherence") or 0.0), 4),
                "coherent": SC.coherent(store, names), "rows": rd.get("rows")}
    except Exception:
        return {"read": False}


def compose(store, text: str, acts: List[Dict[str, Any]], *, person: str = None,
            delegate: Any = None) -> tuple:
    """Say only what is grounded: the resolved concepts' own definitions (their cited glosses),
    rendered by the keyed inverse (`_render_concept`) — the exact, lossless half of the
    language-transducer conversion — plus the edges the answer's band absorbs (`_stated_relations`).

    There is no "related concepts" dump: those are co-activations, which is where noise leaks in
    (wolf -> 'department of energy'). The render goes through the same identity-keyed path on the way
    out that the signal came in on, and `output_membrane` reads the result back through the instrument
    so a caller can report the conversion as measured rather than asserted."""
    # An empty field is the instrument's computed null: nothing rose above the measured noise floor
    # ([[one-resolution-not-thresholds]]). The empty result carries no citation, so the grounded flag
    # reads false from the evidence and the caller routes on the null rather than on a scripted
    # string ([[state-what-it-is]]). The null is reported through the caller's `screen` certificate
    # (`k_signal`), which is the honest "whether"; this line names it in language.
    if not acts:
        return "", []                   # nothing rose above the floor — no outgoing signal, not a sentence
    from crystal.ontology import geometry as g
    from prism import resolution as _res

    # ── How many readings are the answer — measured, not argmaxed ─────────────────────────────
    # A raw argmax over `acts` breaks ties by insertion order. On "Who was Albert Einstein",
    # `world health organization`, `washington` and `einstein` all fire at salience 1.000, so `acts[0]`
    # returns whichever was inserted first.
    #
    # The zeros leave the series before any cut. `rank_fired` returns readings whose salience is
    # exactly 0.0 — measured, and measured to be nothing. Feeding them to the cut pads the series with
    # identical values and flattens the between-class variance against the null, so a field with an
    # obvious dominant reads as unseparated and `signal_end` falls back to "all of it":
    #
    #   "What is the capital of France"   24 readings, 7 nonzero
    #       with zeros    -> separated=False, k=24   (17 of them measured zero, reported as signal)
    #       zeros dropped -> separated=True,  k=1    -> ['paris']
    #
    # Dropping them is not a threshold: zero is not a small reading, it is the absence of one and
    # carries no information.
    # `.get`, not `[...]` — an act is not guaranteed to carry a salience. The recognition path always
    # sets one, but `compose` is reached with hand-built acts too (the isolation tests do exactly
    # this), and the honest reading there is "no measured salience, so nothing to cut on".
    def _sal(a) -> float:
        try:
            return float(a.get("salience") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # The lead pool is what the query fired: observations, not predictions. Rows carrying
    # `predicted_from` mark a concept placed by a caller's own continuation step, not by this
    # recognition — a one-step propagator rolled forward from the lead and snapped onto real
    # corpus concepts. They belong in the field, and they are exactly what
    # `_stated_relations` can absorb: "what is water" goes from one relation to seven, 2 citations
    # to 8.
    #
    # They are not eligible to become the lead, and the reason is structural. The lead is read off the
    # frame's resolved modes rather than off salience, so a tight cluster of mutually-similar
    # predictions dominates the principal direction even when every one of them carries less energy
    # than the concept they were predicted from — measured with the predicted rows' salience already
    # strictly below the seed's:
    #
    #   "what is a dog"   + hunting dog/working dog/terrier/hound (sal ~130-142 vs dog's 248.6)
    #                     -> the answer becomes terrier's gloss
    #   "what is a bank"  + riverbank/waterside/hillside/descent/ascent
    #                     -> the financial-institution sense falls below the cut and is lost
    #
    # What is being defined has to be something the need actually fired; what is said about it may
    # include what couples to it. That is a separation of roles, not a threshold — no quantity is
    # compared against anything here, and a predicted row keeps its full energy everywhere else.
    #
    # When nothing fires with positive salience the pool is empty and there is nothing to define
    # ([[absence-is-not-an-affirmative-claim]]). An empty answer says "this corpus does not carry
    # it", while taking whichever activation happened to be first would say something the corpus
    # never said.
    live = [a for a in acts if _sal(a) > 0.0 and not a.get("predicted_from")] or \
           [a for a in acts if _sal(a) > 0.0]

    # ── The answer is what condensed, read off the beam ─────────────────────────────────────────
    # The acts are not a score list — they are a beam: an ordered (T, F) frame whose rows are the
    # fired concepts, each carrying its activation energy. The answer is read from the beam's resolved
    # modes, not from an Otsu split of a bare column. This is the same quantity `crystal.condense()`
    # absorbs, so the answer and the crossing agree by construction rather than by coincidence.
    #
    # A bare column cannot separate here. Ranking the rows by their loading on the dominant mode mixes
    # a row's energy with its alignment to one direction, and the series has no reason to separate:
    # live on "who was albert einstein", η² = 0.6785 against a null of 0.7519 — less structure than a
    # featureless ramp — so `separated()` says no, the cut returns all twenty rows, and the answer is
    # Einstein followed by nineteen glosses about Washington State and the World Health Organization.
    # The instrument cannot rescue it either: `signal_end(frame=W)` reads the instrument only when the
    # Weyl interval has collapsed, and on a (T≈14–81, F=280) query screen it measures [1, 280] every
    # time.
    #
    # `projection.readings` asks the same beam a question it can answer: which modes resolve, and
    # which row is each one. Einstein, Washington and the WHO are three different modes — the field
    # separates perfectly, just not along the dominant mode's loading column. See that docstring for
    # the measurements and for why a mode is stated once, at its peak.
    #
    # The frame must carry the energy or the read inverts. `dense_vec` is unit-norm, so without
    # `energy=` a concept fired at 43 and one fired at 1.0 are the same magnitude and the instrument —
    # correctly — reports whatever varies most. On an ontology screen that is the unrelated senses: a
    # taxonomy is correlated by construction (one hypernym path, low variance) while an odd sense sits
    # alone and carries the variance. Dominant-mode loadings:
    #
    #   "what is a dog"           unit-norm: frump, unpleasant woman, cad  |  energised: dog, domestic animal
    #   "who was albert einstein" unit-norm: world health organization     |  energised: einstein, physicist
    #   "capital of france"       energised: paris, next 0.32 (a 77x margin)
    #
    # Falls back to the salience cut when no frame can be built (too few rows, no coordinates, a
    # caller passing hand-built acts): a read that cannot be taken yields no value to read.
    leads = None
    try:
        from ember.signal import projection as _proj
        # `frame_rows`, not `frame` — the frame together with which act each row is. `frame` drops a
        # name whose coordinate is absent and returns the matrix alone, which forces a caller to guard
        # on `_load.size == len(live)` and throw the entire beam read away when they disagree (live:
        # dog 66->64, bird 58->56, tree 43->41, river 12->10, each falling back to the salience
        # column). The correspondence removes the need for that guard: rows and acts cannot drift
        # apart.
        # `corpus_basis=True` is stated rather than inherited from the signature default, and it is
        # not free: over nine live queries, projecting onto the 280 corpus directions keeps
        # 0.449–0.620 of the frame's energy (mean 0.542) and discards the rest. It is right for this
        # call — this frame goes to an instrument (`readings`, and the delegate's pooled spectrum), and
        # the raw dense frame is (T≈14–81, F=2048), where those same nine queries resolve
        # k_signal = 0 every time. The membrane path chooses the other arm for the opposite reason
        # (`ember/signal/state.py`).
        _got = _proj.frame_rows(store, [a["concept"] for a in live], energy=[_sal(a) for a in live],
                                corpus_basis=True)
        if _got is not None:
            _W, _kept = _got
            # The screen accumulates. This turn's beam pools onto the delegate's own screen, so T
            # grows with the conversation and the certified band tightens (sqrt(F/T)). It is
            # bookkeeping beside the answer — `accumulate` never raises — and it is the only place a
            # turn's beam is retained, so a delegate's screen is exactly what it has seen.
            #
            # `delegate`, not `person`. `person` is the person id string (`d.person`), so a
            # `hasattr(person, "accumulate")` check would be False forever and the screen would never
            # accumulate, with no error to show for it. An explicit parameter cannot quietly do
            # nothing: a caller either passes its delegate or it does not accumulate, and that is
            # visible at the call site.
            #
            # The basis is stated. `_proj.frame_rows` above states `corpus_basis=True`, so this plane
            # is in the `geom.corpus-basis` artifact's coordinate. Handing the coordinate's identity
            # over is what lets the screen say which basis it holds and count what it dropped when
            # that artifact is rebuilt in place (k 195 -> 280 changes every live screen's coordinate),
            # rather than reporting a stale pool with full confidence.
            if delegate is not None:
                from ember.signal import pooling as _pooling
                delegate.accumulate(_W, basis=_pooling.coordinate_id(store, corpus_basis=True))
            _rd = _proj.readings(_W)
            if _rd:
                # ── How many of the resolved readings are the answer ───────────────────────────
                # `readings` answers which readings exist and what each one is; what is left is where
                # this beam's energy stops. That is `signal_end`'s own question, asked here of a real
                # measured series — the energy each resolved mode absorbs — rather than of a proxy
                # score column. The null is computed from a ramp of the same length, so a field whose
                # modes are evenly lit does not separate and every reading stands, which is the honest
                # report of an ambiguous query. Live:
                #
                #   what is a cat        [56.57, 19.55, 1.06, 0.99]  -> 1   cat
                #   who was einstein     [4.48, 3.09, 3.05, 2.38, 2.34] -> 1   einstein
                #   what is a bat        [6.02, 4.04, 1.23, …]       -> 2   bat(club), bat(animal)
                #   what is a bank       [97.28, 62.58, 6.54, …]     -> 2   bank, financial institution
                #   what is the capital of france  [2.97, 1.69]      -> 2   both Paris entries
                #
                # No frame is passed. `signal_end(frame=W)` asks the instrument `resolvable(W)`, which
                # returns `k_signal`, the number of modes — and the series here has exactly one entry
                # per mode, so the instrument would answer "all of them" by construction, every time.
                # How many modes exist is already measured (`len(_rd)`); this is the different
                # question of where their energy separates, answered from the energies themselves.
                #
                # No `max(1, …)` floor: `partition` searches splits from 1 and `signal_end` returns
                # `len(vals)` when nothing separates, so the count is >= 1 for any non-empty series.
                # A guard that cannot fire reads as protection and proves nothing
                # ([[verification-that-cannot-fail]]); the measurement stands on its own.
                _k = _res.signal_end([e for _i, e in _rd])
                leads = [live[_kept[i]] for i, _e in _rd[:int(_k)]]
    except Exception:
        leads = None                      # any failure -> the honest fallback below, never a guess
    if leads is None:
        k = _res.signal_end([_sal(a) for a in live])
        leads = live[:k]

    # Compose over every resolved reading, not one. When the cut says k=1 that is a single answer
    # (France -> Paris); when three readings tie it says so, which is the honest report of a field
    # that genuinely did not resolve — never a coin-flip presented as a fact.
    ic = g.load_ic()
    sents: list = []
    cites: set = set()
    # The same statement, reached twice, is stated once. The corpus carries two id families over one
    # concept (PWN `einstein.n.01` and OEWN `oewn-10974490-n`) and both resolve to the same gloss
    # text, so without this "who was albert einstein" says "Physicist born in Germany who formulated
    # the special theory of relativity…" twice, word for word. Two ids are not two readings; the
    # second occurrence adds no information, and repeating it asserts a corroboration nothing
    # measured. `_stated_relations` holds the same rule for edges (its `said` set). It is not a cap
    # and cannot shorten a real answer: two different glosses are two different strings and both
    # stand.
    _said: set = set()
    for a in leads:
        s, extra = _render_concept(store, a["concept"], ic, person=person)
        s = [x for x in (s or []) if x not in _said]
        if s:
            _said.update(s)
            sents.extend(s)
            # `_render_concept` returns the record it read as `extra`, so a row whose artifact states
            # no source contributes its record and nothing invented.
            cites |= {c for c in ({a.get("cited_from")} | extra) if c}
    if not sents:                       # nothing carried a definition — no signal out
        return "", []

    # ── The edge graph, stated ──────────────────────────────────────────────────────────────────
    # The glosses above already walk the taxonomy — "what is a cat" says cat's gloss, then feline's,
    # then carnivore's — but as an undifferentiated list, so a reader cannot tell that "a terrestrial
    # or aquatic flesh-eating mammal" is cat's ancestor rather than a second definition of cat. This
    # names that structure, and names nothing else, because coupling is what selects rather than a
    # predicate. `_stated_relations` puts the store's incident edges on a frame at their far
    # endpoints' measured energy and lets the leads' own band absorb what couples: a concept the
    # propagation did not fire has no amplitude, so there is nothing of it to absorb, and a relation
    # the corpus does not hold has no row at all ([[free-will-is-intelligence-never-force-only-give]]
    # — a gate that permits is still forcing). See its docstring.
    #
    # The relation is rendered as the store's own label between the ontology's own words — `cat
    # -hypernym-> feline` — rather than wrapped in a hand-authored sentence. An `It is a kind of {X}.`
    # scaffold is a pre-conceived form asserting a reading of the label; repeating the label verbatim
    # asserts only what is stored ([[state-what-it-is]]).
    _field = {a["concept"]: _sal(a) for a in acts if a.get("concept")}
    try:
        _rel, _rel_cites = _stated_relations(store, [a["concept"] for a in leads], _field,
                                            operator=_operator_concepts(store, text))
    except Exception:
        _rel, _rel_cites = [], set()      # relations are additive — never let them cost an answer
    #
    # The stated edges go on their own line — layout, not language. A connective ("and it is related
    # to…") would be the pre-conceived form again; a line break asserts nothing and just stops the
    # relations from reading as one more gloss.
    _rel_line = ""
    if _rel:
        _rel_line = "\n" + "; ".join("%s -%s-> %s" % t for t in _rel) + "."
        cites |= _rel_cites

    # ── What is remembered is what condensed ────────────────────────────────────────────────────
    #
    # The turn deposits what actually fired, at its measured share, on one tick. Every activation
    # carries a measured salience and the turn's energy is distributed over them by share, so a turn
    # deposits exactly one unit however many concepts it fired (0->1->0, not 0->N->0). Weakly fired
    # concepts enter weakly and cool out on their own — no list, no cut, no rule about how many a turn
    # is allowed to remember.
    if delegate is not None and acts:
        try:
            _observe_field(delegate, acts)
        except Exception:
            pass
    return " ".join(sents) + _rel_line, sorted(cites)

# ── a turn's fired field enters this delegate's screen as cooling `is` traces; the conversation acts
#    in `lumen/conversation.py` reach back for this recognition primitive ──
def _observe_field(d, acts) -> None:
    """Drop the turn's fired field into this delegate's screen, each at its measured share.

    The concepts are the ones that were resolved, not words looked up again. `acts` already carry the
    disambiguated synset name, so `wn.synset(name)` resolves the same concept the answer used, by
    name, with no sense pick at all; going back through `_word()` and `synsets(word)[0]` would discard
    the disambiguation and re-pick sense 0.

    Energy is share, so a turn deposits one unit. `energy_i = salience_i / Σ salience` is the turn's
    own measured distribution normalised by its own total. Nothing is chosen: a turn that fires eighty
    concepts deposits the same total as one that fires three, so a busy turn cannot out-shout a
    focused one (0->1->0). Concepts that fired weakly enter weakly and fall below the read's own floor
    on their own, which is what replaces keeping only the top one.

    One tick for the whole turn. A tick is one observation step — the observer's proper time — so
    calling `next_tick()` per concept would make a turn that fired eighty concepts eighty times older
    than one that fired three, and a busy turn would age a quiet one's memory out. `delegate.py`
    records the same property for the process-global tick, one level down.

    The store is threaded through, because the ontology read is a measurement and belongs on the store
    the caller holds. `wn.synset(name)` without a store resolves the process-default store; on a node
    where that default is the live lattice, a caller holding its own store reaches past it and can
    take a write lock on the production shard. The field size multiplies the effect: one unbound read
    per fired concept rather than one per turn.

    Best-effort — never let screen bookkeeping break an answer."""
    try:
        from crystal.ontology import driver as wn
    except Exception:
        return
    store = getattr(d, "store", None)
    rows = []
    for a in (acts or []):
        name = a.get("concept")
        # A prediction is not an observation. `compose`'s lead pool draws exactly this line
        # thirty lines up, filtering on the same `predicted_from` key, and the memory path receives
        # the same list. One live turn of "what is a dog" places 74,450 activations, and depositing
        # every one as something observed makes the screen's trace bag a record of what the
        # propagation reached rather than of what the signal lit:
        #
        #     activations placed :  74,450
        #     traces after 1 turn:  74,450
        #     rates_source       :  unmeasured
        #     tau_fast / tau_slow:  None / None
        #
        # With 74,450 rows sharing this function's one deposited unit, each enters at ~1e-5. The
        # energy share is still conserved; whose energy it is, is not. It shows up as a lost
        # measurement rather than as an obvious fault: `rates_source` reads `unmeasured` after a full
        # live turn, so the per-screen decay (`tau_fast` / `tau_slow`) that `prism.demurrage` prices
        # the economy's clock off is absent.
        #
        # Predicted rows keep their full energy everywhere else — in the field, in `_stated_relations`'
        # absorption, in the answer's citations. They are excluded here only, from the record of what
        # this observer saw. That is the same separation of roles the lead pool makes, not a cut:
        # nothing is compared against anything, and no count is chosen.
        if a.get("predicted_from"):
            continue
        try:
            e = float(a.get("salience") or 0.0)
        except (TypeError, ValueError):
            e = 0.0
        if not name or e <= 0.0:
            continue                    # a measured zero is not a small reading; it is no reading
        try:
            s = wn.synset(name, store=store)   # the store the delegate holds, never the default
        except Exception:
            continue
        if s is not None:
            rows.append((s, e))
    total = sum(e for _s, e in rows)
    if not rows or total <= 0.0:
        return
    d.observe_field([(s, e / total) for s, e in rows])


# The conversation triple artifact content-type, re-exported from the stable grounding surface so the
# recognition primitives' callers (`delegate.load_obs`, tests) reach it without the genesis op-table.
# The conversation acts that write triples and messages live in `lumen/conversation.py`;
# `_ensure_private`, the grant-minter those acts call, is reachable from `ember.genesis`.
from prism.grounding import TRIPLE_TYPE  # noqa: E402  (stable re-export)
