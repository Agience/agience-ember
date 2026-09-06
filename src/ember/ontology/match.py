"""Need -> offer, geometrically. The measurement half of signal propagation.

A signal propagates: it activates what fits, and everything geodesically near what fits activates
too. Selection is therefore k-nearest-neighbour over operator offers, rather than a lookup table or
a top-1 pick.

## Real distance, not a similarity score

An offer is kept as its set of nodes in the ontology rather than pooled into one direction, so the
geodesic distance to each node stays measurable. Selection is a screened propagator —
`energy * exp(-d/xi)` with a propagation floor below which nothing propagates — over `jc_tree` distances.
See the propagator note below for the shape and the figures.

Every match reports its `distance` alongside its `energy`, so a caller reads how far the match
actually was instead of inferring it from a score.

## Nouns and verbs

Both carry a hypernym tree, so both have a position the coordinate can measure. Adjectives and
adverbs carry an information content but no parent, so they have no least common subsumer and
`jc_tree` has nothing to measure over them; `coordinate_coverage` reports which tokens a text leaves
unplaced. An offer that grounds to nothing is reported as `unembeddable` rather than scored as
distance-infinity, which keeps "never measured" distinguishable from "measured and far".

## A corpus with no geometry has no reading to give

When the geometry cannot be loaded (no WordNet index, no IC), `xi()` and `propagation_floor()` return `None`
and `propagate` returns `(0.0, inf)`. The selection tekton built on them (`sage.match.select`)
reports `basis="unavailable"` and the caller decides; suffix matching is never returned in a
geometric answer's clothes.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from weakref import WeakKeyDictionary

from weakref import WeakKeyDictionary

from mantle.db.constants import CT_FETCH_CAP as _CT_FETCH_CAP
from prism import law as _law

from ember.ontology import information as _information

# The lookup half of this module lives in `crystal.ontology.lookup`: the part that needs only the
# store, which is the part the personas reach for. What lives here is the part that also needs
# prism — `propagate`, `fired_field`, `signal_offers`, `tekton_basis` and the geometry derivations.
# The split follows the transitive call closure rather than theme, because crystal does not import
# ember and the boundary has to fall exactly where the dependencies do.
#
# The lookup names are imported back because this half calls them: `fired_field` uses `_wn_prefix`,
# `offer_synsets` and `wn_synsets_for`; `expand_associative` uses `related`. ember may import
# crystal, so this direction holds.
from crystal.ontology.lookup import (  # noqa: F401  — re-exported: callers still say `match.X`
    _WORD, _wn_prefix, hop_cost, lemma_tokens, offer_synsets, projected_synsets_for,
    related, wn_synsets_for)

# `_OFFER_CACHE` below is keyed by the store, following `genesis._COLLECTION_COUNT_CACHE`'s shape
# rather than the unkeyed module dicts elsewhere in this package: an unkeyed cache would serve one
# store's offers to another.


# ── The propagator ──────────────────────────────────────────────────────────────────────────────
# Selection is a screened propagator, the propagation floor shape. A correlator falls off as `exp(-d/xi)` in
# the geodesic distance `d`, and below the gap nothing propagates:
#
#     contribution = energy * exp(-jc_tree(need_synset, offer_synset) / xi)
#     ...counted only if that weight clears the gap
#
# The distance term is what makes the ranking meaningful. Mean-pooling an offer's senses into one
# direction and ranking by cosine discards how far the need is from each offer: measured that way,
# the need "a python function" scored `op.health` ("reports node health status and disk memory") at
# 0.3491, ahead of `op.describe.python` ("describes python source code modules and functions") at
# 0.1902 — pooling had smeared five senses into one direction where "function" and "status" sit
# close, and with no distance term nothing separated them.
#
# This is the same kernel `activation.spread_seeds` propagates with (`freq * exp(-jc_tree(p, s,
# ic))`), so selection and propagation agree by construction rather than by coincidence.
#
# The attenuation length is in Jiang-Conrath nats and is measured rather than chosen: `xi()` below
# reads it from the corpus's own geometry.



#: Corpus statistics — df / salience / row-count over the store's FTS index. A module (or any object)
#: exposing `_corpus_rows(conn)`, `_df(conn, t)` and `_salient(conn, terms)`. None until wired.
#:
#: An injection seam rather than an import: `match` is measurement, and `corpus_stats` reads
#: `ember.corpus.fts`, so it stays out of this module. `ember/ontology/__init__.py` wires the two at the
#: package boundary, where the edge is visible; that keeps the data/measurement split inside
#: `ontology/` clean and keeps this module free of a direct FTS dependency.
#:
#: Unwired there is no corpus to measure against: salience weighting stays uniform and every word
#: stands. See the note in `fired_field` on why that case still fires every sense rather than
#: returning early.
_CORPUS_STATS = None


def set_corpus_stats(provider) -> None:
    """Wire the corpus-statistics provider (mantle's). `None` unwires it."""
    global _CORPUS_STATS
    _CORPUS_STATS = provider


def projected_positions_for(name, store=None):
    """Where a MODIFIER synset sits in meaning-space — the nouns it is about.

    `fired_field` already projects a modifier on the QUERY side: `viscous` fires `viscosity.n.01`,
    because an adjective carries an information content but no hypernym parent, so `jc_tree` has
    nothing to measure and a synset admitted directly would score a distance from nowhere.

    The CANDIDATE side does not, and that asymmetry is the whole of the modifier problem.
    `ranking._position` returns a `wn-` candidate's own synset name, so an adjective ANSWER is
    placed where no query can reach it. Measured on this corpus:

        glacier   (noun)   energy 4.1131   distance 0.0     <- the query's own position
        able      (adj)    energy 0.1674   distance 1.53
        unable    (adj)    energy 0.0276   distance 1.68
        abaxial   (adj)    energy 0.0000   distance inf

    A modifier answer is never at distance 0 from its own question and is usually at infinity. So
    "what does X mean" cannot be answered by reach at all — measured, 22% rank-1 against 72% for
    nouns, on 60 questions each.

    Exposed here because this layer is the only one both sides can reach: `crystal` holds
    `projected_nouns_for`, mantle may not import it, and the seam is how the ranking already gets
    `fired_field` and `propagate`. Same function, same relations (`derivation` / `attribute`, then
    `similar` to a satellite's head, then `pertainym` for an adverb), so the two sides project a
    modifier the same way or the asymmetry simply moves.

    `[]` for a synset that is not a modifier, or one the source holds no link for — 34% of them.
    An empty answer leaves the caller with the position it already had.
    """
    from crystal.ontology.lookup import projected_nouns_for

    try:
        return list(projected_nouns_for(store, str(name)) or [])
    except Exception:  # noqa: BLE001 — an unreadable projection places nothing
        return []


def subject_synsets(text):
    """Where an ARTIFACT sits, from the words it is keyed on — nouns AND verbs. The seam half of
    `crystal.ontology.lookup.subject_synsets`.

    Exposed beside `offer_synsets` rather than replacing it: the two answer different questions and
    the ranking needs both. An OFFER is a description, where a verb is the frame and firing it
    smears the offer; an ARTIFACT titled `frighten` is ABOUT frightening. mantle cannot import
    crystal, so the seam is how the ranking reaches either.

    Measured 2026-08-26 at n=2,500: 2.8% +/- 0.7pp of ConceptNet — ~33,000 artifacts — is placeable only once
    verbs are asked for. Falls back to `offer_synsets` on an older crystal, which is the noun-only
    answer the caller had before this existed."""
    from crystal.ontology import lookup as _lookup

    fn = getattr(_lookup, "subject_synsets", None)
    if fn is None:
        return offer_synsets(text)
    try:
        return list(fn(text) or [])
    except Exception:  # noqa: BLE001 — an unreadable lexicon places nothing
        return []


def salient_terms(terms, store=None):
    """Which of `terms` carry the question — the corpus's own measure, exposed on the seam.

    `_salient` keeps a term when it carries at least the query's own MEAN information, so nothing
    is chosen: the bar scales with the query and with the corpus, and when every term is equally
    informative every term is kept.

    Public because two paths need one answer and only one of them could reach it. `fired_field`
    has applied this to the words it fires since §13.20, and the comment there claims "one
    measure, both paths: recall and reach agree about which words carry the question". The recall
    path did NOT apply it — mantle cannot import `ember` (layer law), so the measure was simply
    unreachable from the narrowing, and the claim had gone stale without anything failing.

    Measured, that cost the served path its answers. `recall("what is a glacier")` narrowed on
    `what` / `is` / `glacier`, and the coverage ordering that fills the reach arm's pool counts
    stems WITHOUT weighting — deliberately, see `_by_coverage` — so canon documents titled
    "What this is (and is not)" matched two stems and filled the horizon before any glacier
    synset, which matched one, could enter it. Reach then re-ranked a pool the answer was never
    in:

        recall("what is a glacier")  ->  what-is-agience — The Problem It Solves
        recall("glacier")            ->  Piedmont glacier, continental glacier, ...

    Same corpus, same fired field — only the stems differed.

    Returns the input unchanged when there is nothing to measure against, which is `_salient`'s
    own convention for an unmeasurable corpus: an unreadable index must not silently narrow a
    query to nothing.
    """
    ts = [t for t in (terms or []) if t]
    if len(ts) <= 1 or _CORPUS_STATS is None:
        return ts
    try:
        conn = store.artifacts.db.read() if store is not None else None
        if conn is None:
            return ts
        keep = set(_CORPUS_STATS._salient(conn, ts))
    except Exception:       # noqa: BLE001 — a store read raises broadly; measure nothing, keep all
        return ts
    return [t for t in ts if t in keep] or ts


def _derive_geometry() -> Optional[dict]:
    """The corpus's own geometry — ξ and the propagation floor, read exhaustively off the is-a tree.

    Both quantities are exact reads over the whole population, in one pass, with no sampling error,
    no RNG and therefore no seed.

        d_edge   the median of `IC(child) − IC(parent)` over every canonical tree edge that carries
                 positive weight — one is-a step, in nats. Every node contributes its own edge
                 exactly once (each has exactly one canonical parent), so this is the population
                 median rather than an estimate of one.

        d_max    the corpus's diameter: the largest JC distance between two nodes that are
                 comparable. Per taxonomy this is `2·IC_max − 2·IC(root)`, attained by any two
                 maximally-specific leaves whose only common ancestor is the root; the corpus's
                 diameter is the largest such over its taxonomies. It is attained rather than merely
                 a bound — see the corroboration below.

        mu = ln(d_max / d_edge)          xi = 1/mu          gap = exp(−d_max / xi)

    From those, `prism.resolution.horizon(xi, gap) == d_max` exactly, by construction. That identity
    is what makes the gap derived rather than picked: the propagation reaches precisely as far as
    the corpus extends, and no further. A unit-weight signal travels to the furthest thing this
    corpus attests and stops there, because past the diameter the corpus holds nothing.

    A reach bound is set by the extreme, not by the centre. A gap taken from the median unrelated
    distance would put half of all attested pairs beyond the horizon: measured on this store, ξ =
    0.4071 with a gap of 0.05 gives a horizon of 1.2195 nats against a corpus diameter of 1.6909, so
    28% of the corpus's own reach would sit outside it.

    Both quantities are read per taxonomy, because `jc_tree` measures only within one. Measured on
    this store, the coordinate-carrying noun population has exactly two canonical roots —
    `oewn-00001740-n` (84,955 nodes, IC 0.154525) and `entity.n.01` (82,114, IC 0.157059) — sharing
    no ancestor. `tree_lcs_ic` returns 0.0 for such a pair, i.e. it assumes a universal IC-0 root
    that does not exist here, and `jc_tree` then returns `IC(a)+IC(b)` = 2.0. That is `jc_tree`'s own
    documented scope limit rather than a distance, so a cross-taxonomy pair carries no reading and
    contributes none ([[absence-is-not-an-affirmative-claim]]).

    Nothing filters the population by language, by id family or by anything else typed; the
    structure selects itself. A synset with no positive edge on its path contributes no edge and has
    no coordinate (`sparse_vec` keeps only `d > 0`, so `dense_vec` is the zero vector), and a
    taxonomy with a flat IC has `d_edge` undefined and `2·IC_max − 2·IC(root) = 0`, so it reports no
    geometry and is not counted. All 314,775 OMW nouns are exactly that — one IC value, zero
    positive tree edges — and they drop out without being named.

    Measured on this store, exhaustive over 481,846 nouns:

        d_edge 0.196588   d_max 1.690949   mu 2.151933   ξ 0.464698   gap 0.026284

    Two independent corroborations: ξ = 0.4647 against the 0.4607 an independent sampler read on the
    84,956-noun OEWN corpus (0.9%), and the two taxonomies — measured entirely separately — agree on
    ξ to 0.14% and on the gap to 1.6%. The computed diameter equals the largest distance observed
    over 20,000 real pairs, to the last digit, in both taxonomies.

    Returns None when the corpus reports no geometry at all. A caller handed None has learned that
    this corpus cannot say how far a signal travels, and propagates nothing on that basis."""
    import statistics as _stats
    try:
        from crystal.ontology import driver as _wn
        from crystal.ontology import geometry as _g
        pop = _wn.all_synsets(pos=_wn.NOUN)
    except Exception:
        return None
    edges: List[float] = []
    ic_max: Dict[str, float] = {}
    root_ic: Dict[str, float] = {}
    members: Dict[str, int] = {}
    for s in pop:
        try:
            path = _g.tree_path(s, None)
            rn = path[-1].name()
            v = _g.ic_of(s, None)
        except Exception:
            continue
        members[rn] = members.get(rn, 0) + 1
        if v > ic_max.get(rn, float("-inf")):
            ic_max[rn] = v
        if rn not in root_ic:
            try:
                root_ic[rn] = _g.ic_of(path[-1], None)
            except Exception:
                continue
        if len(path) >= 2:
            try:
                d = v - _g.ic_of(path[1], None)     # this node's own canonical edge, counted once
            except Exception:
                continue
            if d > 0.0:
                edges.append(d)
    if not edges:
        return None                 # no attested tree edge carries weight: there is no geometry
    d_edge = float(_stats.median(edges))
    # The diameter is a max over taxonomies: the furthest two comparable things in this corpus.
    # Averaging across taxonomies that share no ancestor would fold in pairs `jc_tree` cannot
    # measure.
    diam = [(2.0 * ic_max[r] - 2.0 * root_ic[r], r) for r in root_ic if r in ic_max]
    diam = [(d, r) for d, r in diam if d > 0.0]
    if not diam:
        return None                 # every taxonomy is flat: no measurable extent
    d_max, d_root = max(diam)
    if not (d_max > d_edge > 0.0):
        return None                 # a corpus whose diameter is one step has no contrast to read
    mu = math.log(d_max / d_edge)
    if not (mu > 0.0):
        return None
    xi_ = 1.0 / mu
    # The read carries its own provenance: which derivation produced it, and which corpus it was
    # taken against. Without both, a persisted copy cannot be told apart from one a different
    # derivation wrote — see `_persisted_geometry`.
    try:
        _n = _wn.ic_basis().get("n")
    except Exception:
        _n = None
    return {"basis": GEOMETRY_BASIS, "ic_basis_n": _n,
            "xi": xi_, "mu": mu, "d_edge": d_edge, "d_max": d_max,
            # `_law.attenuate(d, length=xi)` IS `exp(-d/xi)` — the one screened propagator, taken
            # from prism rather than re-typed here. This file already imports `_law` and calls it
            # in `expand_associative`; an inline `math.exp` was a second copy of the kernel living
            # 621 lines from the first, and the two would only ever be noticed diverging by their
            # results. `attenuate` additionally clamps a negative distance at 0, which a bare
            # `math.exp` does not.
            "gap": _law.attenuate(d_max, length=xi_), "n_edges": len(edges),
            "diameter_root": d_root,
            "taxonomies": {r: {"members": members.get(r, 0), "ic_root": root_ic[r],
                               "ic_max": ic_max.get(r), "diameter": 2.0 * ic_max[r] - 2.0 * root_ic[r]}
                           for r in root_ic if r in ic_max}}


def _derive_xi() -> Optional[float]:
    """ξ measured from this corpus's own geometry — `xi = 1/mu`, `mu = ln(d_max / d_edge)`.

    Thin wrapper over `_derive_geometry`, which takes the one exhaustive pass both ξ and the mass
    gap are read from; the full derivation and the measured figures live there. `None` when the
    corpus reports no geometry."""
    g = _derive_geometry()
    return None if g is None else g["xi"]


# The identity of the derivation above: a name rather than a tuned value, in the same class as
# `crystal.ontology.geometry.GEOMETRY_VERSION` and `crystal.ontology.driver.INTRINSIC_IC_SOURCE`.
# It lets a persisted reading say which derivation produced it, so a reading taken by a different
# one is recognised as such and re-derived.
GEOMETRY_BASIS = "diameter-contrast/v1"

_GEOM_CACHE: List[Any] = []         # [(generation, transducer stamp, the read or None)]


def _persisted_geometry() -> Optional[dict]:
    """The geometry read persisted on the language transducer, or None if it cannot be verified.

    A freshness stamp verifies which artifact was read; it cannot verify which method produced the
    number inside it. So the persisted reading has to state what it is, which is the same rule
    `crystal.ontology.driver.ic_basis` holds one layer down: a stored number that cannot say what it
    was measured from is not a measurement to keep. A reading is accepted only when

      · it names this derivation (`basis == GEOMETRY_BASIS`), and
      · it was taken against this corpus (`ic_basis_n` matches the corpus's own record of its size),
        because the geometry is a function of the whole corpus: ingest a source and both IC and the
        diameter move.

    A bare `spec.xi` float states neither, so it is not accepted and the derivation is re-taken. That
    costs one whole-corpus pass, once, on a corpus whose transducer predates this basis — the price
    of serving only numbers that can account for themselves
    ([[absence-is-not-an-affirmative-claim]])."""
    from crystal.ontology import driver as _wn
    try:
        doc = _wn._arts().get_artifact(_wn._TRANSDUCER_ID) or {}
        g = (doc.get("spec") or {}).get("geometry")
    except Exception:
        return None
    if not isinstance(g, dict):
        return None
    if g.get("basis") != GEOMETRY_BASIS:
        return None
    try:
        n_now = _wn.ic_basis().get("n")
    except Exception:
        n_now = None
    if n_now is None or g.get("ic_basis_n") != n_now:
        return None                 # taken against a different corpus, or against one that cannot say
    if not isinstance(g.get("xi"), (int, float)) or not isinstance(g.get("gap"), (int, float)):
        return None
    return dict(g)


def _geometry() -> Optional[dict]:
    """The corpus's geometry read — persisted-and-verified where possible, derived otherwise, `None`
    when this corpus reports none. ξ and the propagation floor both come from here, through one read behind
    one cache keyed on one generation, so they move together and cannot disagree.

    Two levels, and the order matters because the derivation is expensive. The generation is read
    first and touches no store at all; only when the store has actually been written to is the
    transducer's stamp taken. That ordering keeps an unrelated write from re-entering
    `_derive_geometry()`, which loads the whole corpus on a store that has no transducer built
    yet."""
    from crystal.ontology import freshness as _freshness
    from crystal.ontology import driver as _wn
    gen = _wn.generation()
    if _GEOM_CACHE and _GEOM_CACHE[0][0] == gen:
        return _GEOM_CACHE[0][2]
    st = None
    if _GEOM_CACHE:
        try:
            st = _freshness.stamp(_wn._arts(), _wn._TRANSDUCER_ID)
        except Exception:
            st = None
        if st is not None and st == _GEOM_CACHE[0][1]:
            _GEOM_CACHE[0] = (gen, st, _GEOM_CACHE[0][2])   # re-verified: same artifact
            return _GEOM_CACHE[0][2]
    try:                                # stamp before the read — see `wn_store._get_synset`
        st = _freshness.stamp(_wn._arts(), _wn._TRANSDUCER_ID)
    except Exception:
        st = None
    g = _persisted_geometry()
    if g is None:
        g = _derive_geometry()
        _persist_geometry(g)        # so the whole-corpus pass is paid once, not once per process
    _GEOM_CACHE[:] = [(gen, st, g)]
    return g


def _persist_geometry(g: Optional[dict]) -> None:
    """Write a freshly-derived geometry onto the transducer, so the next process reads it.

    `_persisted_geometry` accepts exactly what `_derive_geometry` returns (`basis`, `ic_basis_n`,
    `xi`, `gap`), so persisting here is what makes the whole-corpus pass cost once per corpus rather
    than once per process: `_GEOM_CACHE` is a module-level list and dies with the interpreter.

    Measured on node 71 (2.15 M vertices), first call in a fresh process:

        propagation_floor()  39.9 s … 158.4 s      second call 0.00 s
        spread_seeds 189 s … 213 s        second call 0.00 s

    and `spread_seeds` runs on every turn, so without a persisted read the first question after a
    restart waits one to three minutes. The answer path itself is ~3.5 s warm.

    The measurement itself is unchanged by persistence: the value goes out under the basis tag and
    corpus size `_persisted_geometry` verifies against, and a corpus that grows fails `ic_basis_n`
    and is re-derived. The freshness discipline is what that check is for; this only keeps the
    derived answer.

    Best-effort: a store that cannot take the write still serves the derived value for this process,
    so a read-only corpus pays the derivation once per process rather than failing a turn."""
    if not isinstance(g, dict) or g.get("basis") != GEOMETRY_BASIS:
        return                      # never persist a reading that cannot state what produced it
    from crystal.ontology import driver as _wn   # the store-backed ontology driver
    try:
        arts = _wn._arts()
        doc = arts.get_artifact(_wn._TRANSDUCER_ID)
        if not doc:
            return
        doc = dict(doc)
        spec = dict(doc.get("spec") or {})
        spec["geometry"] = dict(g)
        doc["spec"] = spec
        arts.put_artifact(doc)
    except Exception:
        return                      # the derived value still stands for this process


def xi() -> Optional[float]:
    """The propagator's attenuation scale, in JC nats — measured from this corpus, or `None`.

    `None` when the corpus reports no geometry: there is no reading to give, and a caller handed it
    has learned that this corpus cannot say how far a signal travels
    ([[absence-is-not-an-affirmative-claim]]). A stand-in value would not be neutral —
    `exp(-d/1.0)` attenuates 2.2x more gently than this corpus's measured 0.4647, so it would widen
    every reach in the system while reading as a measurement.

    A single ξ for a whole corpus is a summary of a per-taxonomy quantity.
    `_derive_geometry()["taxonomies"]` measures each rooted taxonomy separately, and on this store
    the two agree on ξ to 0.14%, so the one number is a stated measurement rather than an
    assumption."""
    g = _geometry()
    return None if g is None else float(g["xi"])


def propagation_floor() -> Optional[float]:
    """The measured propagation floor — the weight scored at the corpus's own diameter — or `None`.

    Named `propagation_floor` rather than `gap` because `propagate` / `screened_accumulate` take a
    `gap` parameter, which would shadow a module-level `gap()` inside those functions; that same
    shadowing is why `xi()` is reached through `globals()["xi"]()` further down."""
    g = _geometry()
    return None if g is None else float(g["gap"])


# ── what ξ is, and why it is measured here rather than read off the instrument ────────────────────
# `xi` is the correlation length: the propagator's `exp(-JC/xi)` — spelled `prism.law.attenuate(JC,
# length=xi)`, the one attenuation law — reproduces `spread_seeds`' `exp(-JC)` exactly, one length
# scale for both. prism owns the kernel; the length that goes into it is measured just above, from
# the corpus's own is-a tree (`xi = 1/mu`, `mu = ln(d_max / d_edge)`; see `_derive_geometry`).
#
# entroptics' `correlation_length` / `decay(X).xi` read a decay length off an ordered (T, F) frame:
# how fast a trajectory decorrelates along its row order. This ξ is a length in JC-nat space over a
# tree, and the corpus's is-a structure is not that frame — `forgetting.py` measured that
# `Dynamics.rates()` returns None on concept streams, an observation set being not a trajectory. So
# the corpus's own geometry is the instrument for this length, while the instrument measures the decay
# length the leaf does present as a frame (forgetting's τ).
#
# `xi()` and `propagation_floor()` return `Optional[float]`, and every caller reads `None` as "this corpus
# cannot say" rather than as a number.


def _derive_gap() -> Optional[float]:
    """The propagation floor, measured: the weight scored at the corpus's own diameter.

    The floor is a quantity in JC-nat space, which is the only space it can be measured in. Two
    other derivations import a number from a different geometry and do not transfer:

      * A floor derived from a lattice counting argument, where the constant comes from closed
        surfaces on a 4D hypercubic lattice. A WordNet hypernym DAG is not that lattice, so the
        floor does not transfer -- and the arithmetic says so: with this corpus's mu ~ 2.15 the
        expression goes negative, i.e. "no floor", which is a unit error wearing a result's clothes.
      * `ember.optics`'s `attenuation` (alpha = log(l1/max(l2,floor))). That is a decay rate in the
        feature-correlation eigenspectrum of an ordered (T, F) frame. This length is a geodesic
        over a tree. Same reason `_derive_xi` measures the corpus's own extent rather than reading
        the instrument's -- see the note above.

    What is in the right space is the corpus's own diameter -- the largest JC distance between two
    nodes it attests as comparable at all. The weight scored there is the floor below which a
    contribution corresponds to a distance this corpus does not contain:

        gap = exp(-d_max / xi)

    and equivalently `prism.resolution.horizon(xi, gap) == d_max`, exactly. The propagation reaches
    precisely as far as the corpus extends and no further. Nothing is chosen; see `_derive_geometry`
    for the exhaustive read, for why a reach bound is set by the extreme rather than the centre, and
    for why cross-taxonomy pairs `jc_tree` cannot measure contribute nothing.

    Measured on this store, exhaustive over 481,846 nouns: d_max 1.690949, xi 0.464698,
    gap 0.026284.

    Returns None when the corpus reports no geometry: an unmeasured gap is not a small gap, and
    there is no screened propagation without one."""
    g = _derive_geometry()
    return None if g is None else g["gap"]


# ── energy is a sum, so a wordier offer absorbs more ────────────────────────────────────────────
# Energy accumulates over (need node x offer node) pairs, so an offer naming more nodes scores
# higher at equal distance. Measured on the smoke panel: the need "an encyclopedia article about
# animals" put `op.describe.markdown` ("markdown prose documents and text", 5 nodes) at 47.5 ahead
# of `op.source.wikipedia` ("encyclopedia articles about the world", 3 nodes) at 2.0 — both at
# distance 0.000, so the ordering came from node count rather than fit.
#
# Summing is the propagator (received energy at a target is the sum over sources), and dividing by
# node count would penalise breadth and so undo "nearby fitting contexts activate too". The choice
# between sum (current), mean, max and sqrt(n) normalisation is open and wants a real offer corpus
# to settle it. `nodes` is reported on every match, so the bias is visible at the call site.

# ── how many offers come back, and whether they separate ────────────────────────────────────────
# Neither question is answered by a constant.
#
#   * "How many of these ranked readings are signal?" is `prism.resolution.signal_end`, which
#     `activation.py` already uses on exactly this shape (an ordered energy column). It needs no
#     constant and imposes no cap: a series that does not separate is returned whole
#     ([[no-arbitrary-caps]]), where a fixed `k = 5` would truncate a neighbourhood of nine to five
#     in silence, and return five where the data resolve two. Five is a page size, not a property of
#     a need, of the offer corpus, or of the propagator.
#
#   * "Do they separate at all?" is `prism.resolution.separated`, which asks over the whole energy
#     column — does it split into two groups beyond what featureless data of the same length would
#     show — and obtains that baseline by running the identical statistic on a uniform ramp. Change
#     the statistic and the baseline follows; no constant survives anywhere in it.
#
#     A floor on `margin = (e0 - e1) / e0` would instead be a local statistic deciding a global
#     question, which is the error `prism/resolution.py`'s header dissects ("Two neighbours know
#     nothing about it") and the reason `signal_end` is not an argmax of adjacent ratios. Fitting a
#     measured floor would carry the same defect with better provenance ([[premonitions-vs-bounds]]:
#     a gate that permits is still forcing).
#
# `margin` is reported as an amplitude: the continuous separation sits in the result beside
# `separability` (η², the fraction of variance the split explains), so a reader sees how separated
# and not only whether.
#
# `separated` is three-valued — True / False / None, where None means not measurable. The case that
# turns on this is a two-candidate column, which no statistic over two points can answer. See
# `signal_offers` below for the measurement, and for why n = 1 is separated by the propagation floor rather
# than by a statistic.


def signal_offers(energies: Sequence[float], *, considered: Optional[int] = None) -> Dict[str, Any]:
    """How many ranked offer energies are signal, and whether they separate at all.

    The measurement half of operator selection. `energies` is the propagated energy column in
    descending rank order with zeros already dropped — the same shape `activation.py` hands
    `signal_end`, and for the same measured reason (see `compose`: padding a series with measured
    zeros flattens between-class variance against the null, and "capital of France" then read
    `separated=False, k=24` with 17 of those 24 measured to be nothing). `considered` is how many
    offers the propagator looked at in total, so the zeros can still be reasoned about without being
    fed to the statistic.

    Returns `{"k", "separated", "separability"}`:
      · `k`            — how many leading readings are signal (`prism.resolution.signal_end`). When
                         the column does not separate this is `len(energies)`: the whole
                         neighbourhood, unclipped. Never a page size.
      · `separated`    — the `whether`. `None` means not measurable, which is a different statement
                         from `False` ("measured, and they did not separate").
      · `separability` — η² for the best split; the continuous evidence behind the boolean, and
                         `None` on the paths where no η² was computed.

    Why `separated` has three values
    --------------------------------
    `prism.resolution.separated` compares the best split's η² against `_null_separability(n)` — what
    a featureless ramp of the same length would score. That instrument needs no constant, and it has
    an exact blind spot at the short end where operator selection lives:

      n = 1   `prism.resolution` returns False (a one-element series cannot split). That is not the
              propagator's situation: `propagate` reports energy only for offers that cleared the
              propagation floor, and `prism.resolution.reach_limit` states what the others are — a weight
              already below the gap cannot clear it at any distance, including zero. They are
              absent, not close. So when one offer survives out of `considered > 1`, the propagation floor
              separated it, and the gap is itself measured (`propagation_floor()`) rather than chosen.
              → `separated=True`, `separability=None` (no η² was computed, so none is reported).
              With `considered == 1` there was no competitor at all → `None`: winning uncontested
              is not a measurement.

      n = 2   Vacuous, and this is the case that matters. Any two distinct readings split perfectly
              (η² = 1.0) — and so does the two-point null ramp, which is also exactly 1.0, so the
              test can never fire in either direction. Measured on the live offer table: the need
              "an encyclopedia article about animals" scores markdown 148.45 against generic 0.99
              and reads `separated=False`; a mis-grounded pair (generic 109.806 against python
              109.374) reads `separated=False` too. Those two are not alike, and no statistic over
              two points tells them apart — two readings carry no information about the distribution
              they came from.
              → `None`. `False` would publish an absent measurement as a finding, and `True` on the
              148-against-0.99 case would be reading the `margin` amplitude with extra steps. The
              caller sees `margin` and may act on the amplitude; the selector states nothing.

      n >= 3  The null has something to say and the instrument answers. On the same table, "a source
              code module with functions" → [4.589, 3.337, 0.605], η² = 0.9056 against a null of
              0.75 → `separated=True`, and `k = 2` — a neighbourhood of two rather than a winner,
              reached without a number.

    The blind spot is a corpus fact rather than a missing threshold: `n <= 2` is the common case
    because the operator table is three entries long. `ember/optics.py` records the same story for
    short frames (a T=34 turn gives band=16.26 and certifies nothing). More evidence answers it,
    where a constant here would only cover it over.

    The measurement stays in ember and the tekton reaches it ([[ember-is-a-runner]]) — the same
    split `sage/match.py` documents for `propagate` / `_offers`."""
    from prism.resolution import separability as _sep2, separated as _sep, signal_end as _end
    vals = [float(e) for e in energies]
    n = len(vals)
    if n == 0:
        # Nothing measured, so nothing asserted: `separated: False` would be a verdict on a column
        # that does not exist. Same rule as `OpticsRead.scale_hazard` / `.coherence`
        # ([[absence-is-not-an-affirmative-claim]]).
        return {"k": 0, "separated": None, "separability": None}
    if n == 1:
        sole = True if (considered is not None and int(considered) > 1) else None
        return {"k": 1, "separated": sole, "separability": None}
    if n == 2:
        return {"k": int(_end(vals)), "separated": None, "separability": None}
    return {"k": int(_end(vals)), "separated": bool(_sep(vals)),
            "separability": float(_sep2(vals))}

# ── every need synset is carried into scoring ───────────────────────────────────────────────────
# There is no cap on how many need synsets reach the scoring pass, because a cap truncates whole
# words rather than an even tail. Measured on the live corpus, against a cap of 24:
#
#     1 word   ->  7 fired nodes,  7 propagate,  0 dropped
#    12 words  -> 26 fired nodes, 24 propagate,  2 dropped  (8%)
#    24 words  -> 42 fired nodes, 24 propagate, 18 dropped  (43% of the signal)
#
# On "what role light plays in the process of converting carbon dioxide into sugar" that cut every
# sense of `light`, along with `illumination`, `brightness` and `light source` — a subject word of
# the question, silenced.
#
# Sorting by weight and cutting also biases against ambiguity. Splitting conserves a word's
# constraint across its senses (§13.15), so an unambiguous word keeps its whole weight on one node
# while a 14-sense word carries a fourteenth on each; the cut therefore keeps the unambiguous words
# and discards the ambiguous ones, and an ambiguous word is not a less important word. The tail of
# this spread is attenuated by polysemy, not by distance.
#
# The walk is bounded by a derived fact instead ([[no-arbitrary-caps]]): a contribution is
# `weight * exp(-d/xi)` and `exp(-d/xi) <= 1`, so a node whose weight is already below the propagation floor
# cannot clear the gap at any distance, including zero. Skipping those changes no result — it is an
# exact statement about the propagator rather than a budget.




def coordinate_coverage(text: str, store=None) -> Dict[str, Any]:
    """Which of a text's tokens the ontology can place, and which it cannot.

    Nouns and verbs carry a hypernym parent and so have a position `jc_tree` can measure.
    Adjectives and adverbs do not:

        nouns       481,846   placed
        verbs        91,393   placed (27,030 sit in a tree)
        adjectives   66,805   no parent
        adj-sat      21,413   no parent
        adverbs      14,768   no parent

    Every synset carries an information content, at every part of speech, so what an adjective
    lacks is the parent rather than the weight. Without one it has no least common subsumer with
    anything, and `jc_tree` returns `IC(a) + IC(b)` — the disjoint case, which is not a distance.

    The store holds nothing that would place them: `derivationally_related_form` has no edges here,
    `similar_to` attaches to none of these ids, and `attribute` is the only adjective-to-noun bridge
    at 1,278 edges against 88,218 adjectives. Placing them means importing edges the corpus does not
    carry, which is a corpus change rather than a code change.

    An unplaced token is silent in the field, so this read exists to say so. It weights nothing and
    decides nothing: `fired_field` is unchanged and this is a second pass over the same tokens.

    Coverage detects absence, not a wrong sense. `"a quick brown dog"` reports coverage 1.00,
    because `quick` and `brown` both have noun senses and those senses are the wrong ones — the
    flesh under a fingernail, and a colour. A high coverage says something was placed, not that the
    placement is right. Telling the two apart needs a part-of-speech signal this system does not
    have.

    Returns ``{"placed", "projected", "unplaced", "coverage", "projected_coverage", "tokens"}``.

    ``placed`` are tokens with a position of their own. ``projected`` are modifiers with no position
    that `projected_synsets_for` can place onto the noun they are about — reported apart, because a
    projection says where a concept lives and not what the word is a kind of, and one number
    covering both would present the second as the first. ``coverage`` counts only ``placed``;
    ``projected_coverage`` counts both. Both are ``None`` for a text with no tokens: no tokens and
    no coverage are different measurements.
    """
    toks = lemma_tokens(text)
    if not toks:
        return {"placed": [], "projected": [], "unplaced": [],
                "coverage": None, "projected_coverage": None, "tokens": 0}
    placed, projected, unplaced = [], [], []
    for t in toks:
        try:
            got = bool(wn_synsets_for(t))
        except Exception:
            got = False
        if got:
            placed.append(t)
            continue
        # No position of its own. A modifier may still be placeable by projection onto the noun it
        # is about, and that is counted apart from a positioned token rather than added to it — a
        # single number covering both would report a projection as a measurement of position.
        try:
            proj = bool(projected_synsets_for(t, store))
        except Exception:
            proj = False
        (projected if proj else unplaced).append(t)
    n = float(len(toks))
    return {"placed": placed, "projected": projected, "unplaced": unplaced,
            "coverage": len(placed) / n,
            "projected_coverage": (len(placed) + len(projected)) / n,
            "tokens": len(toks)}


def fired_field(text: str, store=None) -> Dict[str, float]:
    """The need's position, weighted by how much each word constrains it.

    The weighting is the corpus's own measure of informativeness rather than a stopword list:
    document frequency, the same signal BM25 trusts. `things` appears in 392 synsets and `math` in
    20, so the corpus reports `math` as ~20x the constraint. Weight = `log(N/df)`, the standard IDF,
    and a word the corpus has never seen contributes nothing rather than dominating.

    Equal weighting cannot tell a subject from a filler word. Measured on the live corpus with every
    fired node at weight 1.0, the need "math about linear things" fires `math` and `things`, and
    `things` resolves to a leaf sense ("any movable possession") scoring intrinsic IC 1.0
    (maximally specific), so it matches itself at distance 0.0 and the query's own filler word wins.

    Weights stay uniform when no store is available: there is nothing to measure against."""
    syn = offer_synsets(text)
    if not syn:
        return {}
    words = lemma_tokens(text)
    if store is None or not words:
        return {s: 1.0 for s in syn}
    # The corpus size comes from the maintained counter rather than `count(*) FROM vertex`, a query
    # shape that dereferences every record and would sit on the hot retrieval path. The df figure is
    # the one `corpus_stats` measures — exact, off `fts5vocab` — so there is a single N and a single
    # df guard rather than a second IDF implementation here. With nothing to count the weights stay
    # uniform, rather than collapsing to `log(max(1, 1/df)) = 0.0` for every term, which would read
    # as "no word constrains anything".
    try:
        cs = _CORPUS_STATS
        if cs is None:
            return {s: 1.0 for s in syn}     # nothing wired: uniform, matching the no-store case above
        conn = store.artifacts.db.read()
        _rows, _corpus_df = cs._corpus_rows, cs._df
        total = float(_rows(conn))
    except Exception:
        return {s: 1.0 for s in syn}
    # An unmeasurable corpus does not return early here. `syn` is only the seed set
    # (`offer_synsets` takes one sense per word); the sense-coherence block below is what fires
    # every sense, so an early return would collapse a polysemous word to a single sense and take
    # sense 1 as a prediction. Unmeasurability is handled where it belongs: `_idf` returns None
    # and the weights stay uniform.

    # One definition of how much a word narrows the corpus, in bits, shared with
    # `activation.seeds_from_text`. `information.word_information` reads exact document frequency
    # off `fts5vocab` and converts it through `entroptics.entropy.surprisal_bits`, so a surprisal
    # here composes with the entropy reads the rest of the instrument reports.
    _info = _information.word_information(store)

    def _idf(word: str) -> Optional[float]:
        # `None` from `word_information` means the corpus itself is unmeasurable; `None` from
        # `_info(word)` means this word is. Both leave the weight uniform, which is what the
        # surrounding block's "unmeasurability is handled where it belongs" note requires.
        if _info is None:
            return None
        return _info(word)

    # ── sense coherence: the matching primitive, applied inside the need ─────────────────────────
    # Taking sense 1 is a prediction. Firing `senses[0]` and nothing else makes the need's position
    # depend on a guess about which meaning a bare word carries, and the guess fails in both
    # directions: measured, `star` fires the network-topology sense when the index sorts by synset
    # offset, and `calculus` fires the kidney-stone sense when it sorts by WordNet sense order,
    # because that genuinely is sense 1. A better-sourced guess is still a guess, and neither
    # reading of "what did they mean" is available to the corpus alone.
    #
    # So no sense is chosen. Every sense of a word fires, and two measurements decide the weights:
    #
    #   1. How much the word constrains at all — its IDF. A word's total contribution is that
    #      number, whatever its senses: the word is one piece of evidence, not N.
    #   2. Which of its senses the rest of the need agrees with. Each sense is scored by its reach
    #      to the other words' senses (the same screened propagator used against candidates, so the
    #      need is disambiguated by exactly the machinery that ranks answers), then normalized
    #      within the word. Normalizing conserves (1): the split redistributes the word's own
    #      constraint rather than amplifying it, so no decay constant is chosen anywhere.
    #
    # A single-word need has nothing to agree with, and then the senses stay evenly split — the
    # honest reading, because "calculus" alone is ambiguous and the answer shows that.
    #
    # ── which words carry the question ───────────────────────────────────────────────────────────
    # Only words the corpus measures as informative fire, and with morphology in the path that
    # matters. Measured on "what is a star": `is` strips to `i` and `a` is itself a lemma — both are
    # letters as nouns — and letters sit near printing characters, so the two function words
    # coherently drag `star` to its `asterisk` sense. Noise that agrees with the wrong sense of the
    # subject costs more than noise that outvotes it, because coherence is the mechanism that
    # resolves ambiguity.
    #
    # This is a measurement rather than a stop-list: no token is tagged or pre-defined. `_salient`
    # reads it from the corpus's own document frequencies — a term is informative when it carries at
    # least the query's own mean IDF — and the recall path has used it since §13.20. One measure,
    # both paths: recall and reach agree about which words carry the question.
    try:
        _keep = set(_CORPUS_STATS._salient(conn, words))
        if _keep:
            words = [w for w in words if w in _keep]
    except Exception:
        pass                    # no index to measure against: every word stands, matching the unmeasurable-corpus case above
    per_word: Dict[str, List[str]] = {}
    order: List[str] = []
    for w in words:
        if w in per_word:
            continue
        try:
            senses = wn_synsets_for(w)
        except Exception:
            senses = []
        if not senses:
            # ── a modifier has no position of its own; it has a noun it is ABOUT ────────────────
            # `wn_synsets_for` returns nouns and verbs only, and correctly: an adjective carries an
            # information content but no hypernym parent, so it has no least common subsumer with
            # anything and `jc_tree` would have nothing to measure. Admitting one directly would
            # score a distance that came from nowhere.
            #
            # Its projection is a different thing and is measurable. `projected_nouns_for` walks
            # `derivation` / `attribute` to the noun the modifier is a value of, or `similar` to a
            # satellite's head adjective and on from there — every hop a relation the source names,
            # and the nouns it lands on have real positions. So the modifier fires THROUGH them:
            #
            #     viscous     -> viscosity.n.01
            #     beautiful   -> beauty.n.01
            #     igneous     -> fire.n.03, fieriness.n.01
            #     radioactive -> radiation.n.04
            #
            # Without this the field is seeded by whatever else the question carried — measured,
            # "what does viscous mean" fired `department_of_energy` (from `does`) and answered
            # confidently about it. 66.4% of modifiers project on this corpus; the rest place
            # nowhere and `_unplaced_subject` says so rather than answering around them.
            try:
                senses = projected_synsets_for(w, store)
            except Exception:
                senses = []
        if senses:
            per_word[w] = senses
            order.append(w)
    # Multi-word lemmas are single concepts. WordNet carries compound entries ("baseball bat",
    # "polar bear", "power plant", "computer mouse") whose meaning is not the sum of their words, so
    # firing only the singles leaves the compound unreachable and "what is a baseball bat" grounds
    # on `bat` while "computer mouse" fires the rodent. Compounds are found by longest-match over
    # the query's own tokens against the lexicon (the source's multi-word entries do the tokenizing,
    # so no phrase list is maintained here).
    #
    # A matched compound consumes its words, and they stop firing on their own. A window that
    # matches nothing consumes nothing, so its words still stand and an unresolved compound falls
    # back to them. Keeping the singles when the compound does resolve puts the parts in
    # competition with the whole, and the parts can win on weight alone. On `what is a solar
    # panel`:
    #
    #     sun.n.01           11.53      <- from `solar`, projected, weighted by `solar`'s own IDF
    #     solar_array.n.01    5.90      <- the compound the question actually names
    #
    # `solar` is rarer than `panel`, so the modifier it is part of out-weighted the concept it is
    # part OF, and the answer to "what is a solar panel" was `sun`. The same shape put `energy`
    # above `alternative energy` and `form` above `rock`. A compound "whose meaning is not the sum
    # of their words" cannot also be scored as the sum of their words.
    #
    # There is no maximum compound length, because the lexicon holds four- and six-token entries
    # (`united_states_of_america`, `american_standard_code_for_information_interchange`). `_wn_prefix`
    # (`crystal.ontology.driver.entry_prefix_exists`) asks the lexicon whether any entry still begins
    # with this window, and the walk extends exactly while one does. When none does, no window longer
    # than this one can match either — a trie's stopping rule, exact, and linear in the token count where trying
    # every window would be quadratic.
    _raw = lemma_tokens(text)
    _consumed: set = set()
    _i, _n = 0, len(_raw)
    while _i < _n:
        _best = None
        _j = _i + 1
        while _j <= _n:
            _surface = "_".join(_raw[_i:_j])
            if _j - _i >= 2:
                try:
                    _ss = wn_synsets_for(" ".join(_raw[_i:_j]))
                except Exception:
                    _ss = []
                if _ss:
                    _best = (_surface, _ss, _j)          # longest match wins: keep looking
            if not _wn_prefix(_surface):
                break                                    # no entry begins with this: nothing longer can
            _j += 1
        if _best:
            _key, _ss, _j = _best
            if _key not in per_word:
                per_word[_key] = _ss
                order.append(_key)
            _consumed.update(_raw[_i:_j])
            _i = _j
        else:
            _i += 1
    # The words a matched compound swallowed. Removed by WORD rather than by position: a query that
    # names the same word twice, once inside a compound and once outside it, is naming one word, and
    # firing it separately on the strength of the second occurrence would restore exactly the
    # competition this removes.
    if _consumed:
        order = [k for k in order if k not in _consumed]
        per_word = {k: v for k, v in per_word.items() if k not in _consumed}
    if not per_word:
        return {s: 1.0 for s in syn}

    def _support(sense: str, others: List[str]) -> float:
        """How much the rest of the need reaches this sense. The same propagator used everywhere
        else, run without the propagation floor.

        The gap answers "was this offer reached at all", dropping distant contributions so that
        arbitrarily far offers cannot accumulate a score from sheer count. The question here is
        purely relative: given that the need names this word, which of its senses does the rest of
        the need sit nearest to? Measured on "a star in the night sky", with the gap applied seven
        of star's eight senses score exactly 0.0 support — including the celestial one, because
        `star(celestial)` to `sky` exceeds the gap's ~1.39-nat reach — leaving the geometric "plane
        figure" to win by default. A comparison in which almost every option reads zero is not a
        comparison.

        The gap still governs whether an answer counts as reached (`propagate` as called from
        `sage.content_search._reach_rank`). The count accumulation it guards against is handled here
        by normalizing within the word, which fixes the word's total contribution however many
        senses it has."""
        if not others:
            return 1.0
        e, _d = propagate({sense: 1.0}, others, gap=0.0)
        return float(e)

    out: Dict[str, float] = {}
    for w in order:
        senses = per_word[w]                      # most-frequent-sense first (wn_synsets_for)
        wgt = _idf(w)
        wgt = wgt if wgt is not None else 1.0
        others = [s for v in order if v != w for s in per_word[v]]
        sup = [_support(s, others) for s in senses] if others else []
        tot = sum(sup)
        if not others or tot <= 0.0:
            # No context to disambiguate — a bare word ("what is a dog"), or the rest of the need
            # reaches every sense equally. Cross-word coherence PROVABLY cannot answer this case:
            # there is nothing to agree with. So the prior answers it, and the prior is data rather
            # than a tuned knob — with flat weights a bare "dog" resolves to `frump` rather than the
            # animal (measured on the kindergarten battery: flat 14.6% primary-sense against 9/10
            # with the prior). A single-word need still shows its ambiguity, since every sense keeps
            # weight; the prior only ranks the commonest above the rarest where the corpus says so.
            #
            # The prior is `activation.sense_prior`, shared with the seeder: the SemCor count
            # where the source tagged one, else the sense-rank order. Both paths have to answer
            # this same case alone, so they answer it with one function.
            #
            # Imported inside the branch because `activation` reaches `match` the same way, in
            # function bodies; one eager edge in either direction would close the cycle.
            from ember.ontology.activation import sense_prior as _sense_prior

            _base = None
            try:
                # Same defect as `activation.sense_prior`, fixed the same day: `wn` was not in
                # scope in this function — the module's other `import ... as wn` sits in a
                # different function body — so this raised `NameError` every time and the handler
                # below silently substituted the unmorphed word. "dogs" never became "dog".
                from crystal.ontology import driver as wn

                _base = wn.morphy(w, wn.NOUN) or w
            except Exception:
                _base = w
            sup = _sense_prior(senses, w, _base)
            tot = sum(sup)
        for s, v in zip(senses, sup):
            out[s] = out.get(s, 0.0) + wgt * (v / tot)
    for s in syn:                             # anything unpaired keeps a neutral weight
        out.setdefault(s, 1.0)
    return out






# ── travelling a relation that is not is-a ──────────────────────────────────────────────────────
# The store holds 305,146 edges across 24 relation types (§13.25). Travelling a relation that has no
# subsumer and no IC needs a distance in the same units as `jc_tree`; a weight blending "tree
# distance" with "associative reach" would supply a number nothing measured.
#
# The derived answer is already in nats: the cost of traversing a link is the ambiguity it
# introduces. If a node has k outgoing edges of a label, following one costs `log(k)` nats — you had
# one thing, now you have one of k. That is branching entropy, measured per node off the graph
# itself (`hop_cost`), with nothing chosen.
#
# It produces the right asymmetry, which is the check that it reads the graph rather than inventing
# one. Measured on the live corpus (mean out-degree -> nats):
#
#     hypernym      1.03 -> 0.026    going up is nearly free: a thing has one kind
#     hyponym       4.53 -> 1.510    going down is expensive: a kind has many things
#     holo_member   1.01 -> 0.007    mero_member  2.21 -> 0.795
#     part_of       1.16 -> 0.147    mero_part    2.45 -> 0.897
#
# The same relation costs 58x more in the ambiguous direction, and the graph is what says so. A
# label that fans out is a weak step; a label that converges is a strong one.






def expand_associative(store, fired: Dict[str, float]) -> Dict[str, float]:
    """Carry the fired field one associative hop, screened by the same propagator.

    A neighbour enters with `energy · exp(-hop_cost/ξ)` — the identical kernel the tree uses, so the
    two steps compose rather than blend: weights multiply, which is distances adding, in nats. No
    coefficient decides how much an associative link is "worth" relative to a taxonomic one; both
    are distances and the arithmetic is the same.

    Measured on the live corpus (ξ = 0.4653, horizon 1.3939 nats):

        star --domain_topic--> astronomy              0.000 nats  reached   (its only such edge)
        star --holo_member--> constellation           0.693       reached   (one of two)
        star --instance_hyponym--> Alpha Crucis       2.303       past the horizon (one of eleven)
        Canis familiaris --> puppy / barker / cur     2.890       past the horizon (one of eighteen)

    The graph's own branching produces that: knowing "star" tells you a great deal about astronomy
    and nothing about which star. Nodes past the horizon are absent rather than weakly included,
    because a gap is a discontinuity.

    Both scales come from the corpus's own measurement (`xi()` / `propagation_floor()`). When the corpus
    reports no geometry there is no reading to give and the field comes back unexpanded: nothing was
    measured, so nothing is added."""
    if not fired:
        return fired
    xi_now = xi()
    gap_now = propagation_floor()
    if xi_now is None or gap_now is None:
        return dict(fired)          # no measured scale ⇒ no measured hop ⇒ nothing to add
    out = dict(fired)
    for name, energy in list(fired.items()):
        if abs(energy) < gap_now:
            continue
        for nb, d in related(store, name):
            w = float(_law.attenuate(d, length=xi_now))
            contrib = float(energy) * w
            if contrib < gap_now:      # cannot clear the gap: not a weak link, no link
                continue
            if contrib > out.get(nb, 0.0):
                out[nb] = contrib      # the strongest path to a node, not the sum of paths
    return out


def propagate(fired: Dict[str, float], targets: List[str], *,
              xi: Optional[float] = None, gap: Optional[float] = None) -> Tuple[float, float]:
    """Propagate an activation field onto a set of target nodes. `(energy, nearest_distance)`.

    The screened propagator: each (need, target) pair contributes
    `energy * exp(-jc_tree(need, target) / xi)`, counted only if that weight clears the propagation floor.
    Returns the accumulated energy and the smallest actual geodesic distance encountered, so a
    caller can always see how far the match really was rather than inferring it from a score.

    `xi` and `gap` both default to the corpus's own measurement (`xi()` / `propagation_floor()`). When the
    corpus reports no geometry this returns `(0.0, inf)`: nothing was measured, so nothing was
    reached. A substituted number here would send an unscreened propagation out in a screened one's
    clothes."""
    from crystal.ontology import geometry as g
    from crystal.ontology import driver as wn
    if xi is None:
        xi = globals()["xi"]()            # measured from the corpus; `xi` is shadowed by the param
    if gap is None:
        gap = propagation_floor()                  # measured from the corpus's own diameter
    if xi is None or gap is None:
        return 0.0, float("inf")          # no measured scale ⇒ no measured propagation
    if not fired or not targets:
        return 0.0, float("inf")

    tsyn = []
    for t in targets:
        try:
            s = wn.synset(t)
            if s is not None:
                tsyn.append(s)
        except Exception:
            continue
    if not tsyn:
        return 0.0, float("inf")

    # Distance alone decides. Every target carries the same weight.
    #
    # A specificity weight is available here and is not applied. The cost is measured and real: a
    # hypernym tree puts generic concepts near the root, where they sit a short geodesic distance
    # from almost anything, so `op.describe.generic` ("describes source and text files") outranks
    # `op.describe.python` for "a source code module with functions".
    #
    # It is unweighted anyway because neither candidate weight is sound on this corpus. Information
    # content is near-inert as a weight: the basis is intrinsic IC, so nothing carries zero and
    # 91.4% of nouns sit at exactly 1.0, leaving almost no dynamic range. Corpus surprisal, the
    # alternative, needs an FTS index — `information.word_information` returns None without one —
    # so it would go silently constant wherever an index is missing, which is worse than an honest
    # absence.
    #
    # `tests/test_specificity_guards_against_generic.py` holds the evidence for reversing this: it
    # asserts no weight is applied and carries the lost ordering property as a strict xfail, so
    # re-introducing a weight surfaces as an XPASS.
    ic_w = [1.0] * len(tsyn)

    # The screened-propagator accumulation lives in `prism.propagation.screened_accumulate`:
    # `Σ energy · attenuate(d, xi) · spec`, gap-gated, strongest-first, nearest tracked. prism owns
    # the algorithm and the kernel; ember passes the ontology in as callbacks, so the dependency runs
    # one way and prism imports no ember. Each target carries its synset (opaque to prism) and its
    # specificity; `_dist` resolves a source name once (cached) and reads the JC-tree geodesic.
    from prism import propagation as _prop
    tgt_pairs = list(zip(tsyn, ic_w))                 # (synset, specificity) — synset opaque to prism
    _srccache: Dict[str, Any] = {}

    def _dist(name: str, tgt_synset: Any) -> float:
        if name in _srccache:
            src = _srccache[name]
        else:
            try:
                src = wn.synset(name)
            except Exception:
                src = None
            _srccache[name] = src
        if src is None:
            return float("inf")
        try:
            return float(g.jc_tree(src, tgt_synset, None))
        except Exception:
            return float("inf")

    return _prop.screened_accumulate(fired.items(), tgt_pairs, distance=_dist, xi=xi, gap=gap)






# ── §A.2 — a tekton's coupling basis from its offer (signal-native reach) ────────────────────────────────
# A measurement (offer coordinates → the subspace they span), so it lives in ember where both personas reach
# it and lumen need not import sage. Reads `_offers` (ember) and `crystal.ontology.geometry.dense_vec`, and
# takes the span via `prism.frames.offer_basis`. A `(T,F)` frame reaching a tekton is absorbed against this
# basis (`prism.frames.absorb_at_tekton`). [[reach-carries-tf-frames-absorb-propagate]]
def tekton_basis(offered_synsets):
    """A tekton's `(F, k)` coupling basis from its offer's synsets — the subspace its offered ontology nodes
    span, in the same dense coordinate a signal frame is carried in. Stacks each synset's
    `crystal.ontology.geometry.dense_vec` (uncentered) into an `(n, F)` frame and takes its orthonormal span
    (`prism.frames.offer_basis`, an SVD — the offered coords are the directions, not a noise-resolved read).
    None when no offered synset grounds to a coordinate; the tekton then couples by self-resolution or not at
    all."""
    import numpy as _np
    from crystal.ontology import geometry as _g
    from crystal.ontology import driver as _wn
    from prism.frames import offer_basis as _ob
    ic = _g.load_ic()
    coords = []
    for name in (offered_synsets or []):
        try:
            v = _g.dense_vec(_wn.synset(name), ic)
        except Exception:
            continue
        if v is not None and _np.any(v):
            coords.append(_np.asarray(v, dtype=float))
    return _ob(_np.vstack(coords)) if coords else None


def tekton_basis_for(store, operator_id):
    """The coupling basis for a registered operator (tekton) in `store` — its offered synsets (from the
    `_offers` table, cached + invalidated ember-side) → `tekton_basis`. None if it has no grounded offer here."""
    syns = (_offers(store).get("nodes") or {}).get(operator_id)
    return tekton_basis(syns) if syns else None


# The operator-selection tekton — `select` (k-nearest operators) and the `Cascade` termination guard — lives
# in `sage/match.py`. Per [[ember-is-a-runner]] the tool belongs to the persona, and its only consumer is
# `sage/operators.py`. It reaches the measurements kept here: `propagate` (the physics), `offer_synsets`,
# `_offers` / `invalidate` (the per-store operator-offer table and its cache, which stay in ember because
# `capability.register_operators` invalidates them and ember does not reach sage), and
# `activation.spread_seeds`. One definition of each, on one side of the line.


_OFFER_CACHE: "WeakKeyDictionary[Any, Dict[str, Any]]" = WeakKeyDictionary()


def _offers(store, *, content_type: str = "application/vnd.agience.operator+json",
            refresh: bool = False) -> Dict[str, Any]:
    """`{id: unit vector}` for every operator whose offer embeds, plus what did not.

    Cached per store. `unembeddable` is carried alongside rather than dropped, so "no operator
    matched" stays distinguishable from "some operators could never be measured" — the difference
    between a real miss and a blind spot."""
    # The cache is keyed on the resolved artifacts store rather than on the object it was handed.
    # This package passes both a bundle (`.artifacts`) and a bare artifact store around — `select` is
    # called with a bundle from `serve`, and with the artifacts store from `operators.select_for` and
    # `capability.register_remote_host`. Keying on the handed object would make two entries for one
    # store's offers, so an `invalidate` on one would leave the other stale and a freshly registered
    # host would stay invisible until restart. Normalising the key collapses them to one entry, so a
    # single `invalidate(anything-resolving-to-this-store)` clears it.
    #
    # Freshness is verified on every call rather than left to that invalidation, because an operator
    # can change at several places: registered, edited in place, or replicated in from a peer, and
    # only the first of those has a caller that could remember to invalidate.
    #
    # The offers derive from every artifact of one content type, so that is the stamp to take:
    # `freshness.set_stamp` asks whether anything of this type has been written. Not the whole-store
    # write mark, which is true every time a chat message lands and would throw away a measured
    # 215–270 ms rebuild each turn; not a TTL, which is a window nobody measured. The stamp itself
    # measures at 81 µs over the 48 operator rows.
    from crystal.ontology import freshness as _freshness
    artifacts = getattr(store, "artifacts", None) or store
    # The stamp's cap is the enumeration's cap, read from the same place: `CT_FETCH_CAP`, the cap
    # `list_by_content_type` builds the table below under. A stamp taken over a narrower set would
    # verify clean while rows past its cap changed; a stamp over a wider one would fail to verify a
    # set that enumerated fine. One source for one fact — a second constant here would agree today
    # and stop agreeing later.
    #
    # With no cap available there is no stamp and therefore no cache: slower, and the only reading
    # the data supports ([[absence-is-not-an-affirmative-claim]]).
    try:
        _cap = _CT_FETCH_CAP
    except Exception:
        _cap = None
    mark = None if _cap is None else _freshness.set_stamp(artifacts, content_type, cap=_cap)
    cached = None if refresh else _OFFER_CACHE.get(artifacts)
    if cached is not None and mark is not None and cached.get("_mark") == mark:
        return cached
    nodes: Dict[str, List[str]] = {}
    unembeddable: List[str] = []
    try:
        from mantle.db.typed_fetch import list_by_content_type
        rows = list(list_by_content_type(artifacts, content_type))
    except Exception:
        rows = []
    for a in rows:
        oid = a.get("id")
        if not oid:
            continue
        offer = a.get("context") or a.get("content") or ""
        syn = offer_synsets(offer if isinstance(offer, str) else str(offer))
        if syn:
            nodes[oid] = syn
        else:
            unembeddable.append(oid)
    out = {"nodes": nodes, "unembeddable": unembeddable, "n": len(rows), "_mark": mark}
    if mark is not None:
        _OFFER_CACHE[artifacts] = out       # unverifiable ⇒ not cached rather than assumed current
    else:
        _OFFER_CACHE.pop(artifacts, None)
    return out


def invalidate(store) -> None:
    """Drop cached offers.

    Correctness does not rest on this call: `_offers` verifies itself against `freshness.set_stamp`
    on every call, so an operator that is defined, redefined, registered, edited or replicated in is
    picked up whether or not anyone invalidates. This is the immediate drop for a caller that has
    just written and does not want to pay one extra stamp read, and the path for stores that cannot
    report a stamp at all.

    Resolves to the same key `_offers` caches under (the artifacts store), so a bundle and its own
    `.artifacts` collapse to one entry and invalidating with either clears it."""
    artifacts = getattr(store, "artifacts", None) or store
    try:
        _OFFER_CACHE.pop(artifacts, None)
    except TypeError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# The coherent core of a position set (§110)
# ═══════════════════════════════════════════════════════════════════════════════════════════════

_CORE_CACHE: Dict[tuple, tuple] = {}
_CORE_CACHE_MAX = 8192


def coherent_core(names: Sequence[str], store=None, anchor: Optional[Sequence[str]] = None):
    """The positions that agree with what the document SAYS it is about (§110).

    A SYNSET candidate has one position, itself. A PROSE candidate has none of its own, so
    `ranking._position` falls through to `offer_synsets(title + lemmas)` and gets one position per
    key term the document was indexed on. Measured on 30 canon documents from the live shard, that
    is a median of 26 positions whose mean pairwise distance is 1.4888 — 88% of the corpus
    diameter. A document is not AT a place in meaning-space; it is smeared across nearly all of it.

    `anchor` is the document's TITLE positions and `names` its full set. A title position is kept
    unconditionally: it is the document's own statement of its subject, and `ranking` already
    records why that is the trustworthy part — a section's body-extracted terms do not repeat the
    name of the document they belong to, so `canon:AGENT-HOST-DESIGN#0` carries 109 lemmas and NONE
    of them is agent, host or design. A `names` position is kept when it lies within one
    correlation length of some anchor position: the body agreeing with the title.

    Without an anchor nothing is returned but `names` itself. There is no coherence without
    something to be coherent WITH, and the obvious substitute is measured below and is a trap.

    ## The radius is the corpus's, not a tuning

    `xi` is the screened propagator's own length scale, derived in `_derive_geometry` from
    `mu = ln(d_max / d_edge)`, `xi = 1/mu` — here 0.464698. It is exactly the distance beyond which
    the propagation this ranking is about to run carries no energy between two nodes, so positions
    further apart than `xi` do not interact under the very measure being applied to them. Nothing
    is chosen.

    ## Why the core is anchored rather than densest

    An unanchored version takes the densest cluster: the position with the most others inside `xi`,
    plus those others. It tightens beautifully — spread 1.4888 to 0.4576, 27% of the diameter, 26
    positions down to 3 — and the three are:

        canon:prism-protocol#1     ->  oewn-13764713-n `1`,  oewn-13765409-n `2`
        canon:AGENT-HOST-DESIGN#0  ->  the same, plus oewn-13766862-n `6`

    The cardinal numbers. They are siblings under one parent, so they are maximally tight, they
    appear in every document's lemma bag, and they are about nothing. **Tightness is not
    aboutness** — a set of taxonomic siblings is the tightest thing the corpus contains and the
    least characteristic. Spread was the wrong objective, and a measurement that improved by 61
    points of diameter was selecting boilerplate.

    Anchoring costs almost all of that apparent gain and is right: spread 1.4880 to 1.4545, because
    a document genuinely about several things HAS a wide spread and should. What it removes is the
    noise — 28 positions to 5 — while keeping `authority.n.01`/`host.n.01` for a document about
    authority host topology, and `key.n.01`/`guidance.n.01` for one about key guidance.
    """
    wanted = [str(n) for n in (names or []) if n]
    seeds = [str(n) for n in (anchor or []) if n]
    if len(wanted) < 3 or not seeds:
        return wanted
    key = (tuple(wanted), tuple(seeds))
    hit = _CORE_CACHE.get(key)
    if hit is not None:
        return list(hit)

    try:
        from crystal.ontology.geometry import jc_tree, load_ic
        # `load_ic` is a no-op sentinel and always returns `None`. IC lives on the synset, stored
        # during enrichment; the `(ic)` argument is threaded through the geometry API so the
        # signatures stay undisturbed, and `jc_tree(a, b, None)` is the correct call. A guard of
        # `if not ic: return wanted` is unconditionally true and makes this function inert in every
        # process.
        ic = load_ic()
        radius = xi()
    except Exception:       # noqa: BLE001 — no geometry: nothing can be grouped, keep everything
        return wanted
    if not radius:
        return wanted

    anchors = [s for s in (_synset_or_none(n) for n in seeds) if s is not None]
    if not anchors:
        return wanted

    far = float("inf")
    keep = list(seeds)                     # the title's own positions, kept unconditionally
    for name in wanted:
        if name in keep:
            continue
        node = _synset_or_none(name)
        if node is None:
            continue
        for a in anchors:
            try:
                d = float(jc_tree(a, node, ic))
            except Exception:   # noqa: BLE001 — an unmeasurable pair is not a close pair
                d = far
            if d == d and d <= radius:
                keep.append(name)
                break
    out = keep or wanted
    if len(_CORE_CACHE) >= _CORE_CACHE_MAX:
        _CORE_CACHE.clear()
    _CORE_CACHE[key] = tuple(out)
    return list(out)


def _synset_or_none(name: str):
    """The synset for `name`, or None. A name the driver cannot resolve is not a position."""
    try:
        from crystal.ontology import driver as _wn
        return _wn.synset(name)
    except Exception:       # noqa: BLE001 — an unresolvable name simply is not a node
        return None
