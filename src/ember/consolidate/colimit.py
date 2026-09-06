"""The colimit: one object carrying every member's provenance, position and mass, with a morphism
from each member. Conservation is the acceptance test.

COMPACTIFICATION §4: *"A colimit is not deletion. It is the universal object with morphisms from
every member of the diagram... one object that carries both provenances and both positions, and
therefore knows something neither original did: that it is one thing seen two ways."* §5: *"fewer
objects, more mass each."*

What the merged object carries, and why each part is derived rather than chosen.

`provenance` — every member's, kept whole. The rung is the minimum over the members, because a claim
    is only as well-grounded as its weakest support and taking the maximum would launder an
    assertion into an observation. `colimit_of` lists the member ids and `sources` their distinct id
    namespaces, so *"one thing seen two ways"* is a fact recorded on the object rather than an
    inference someone has to redo.

`evidence` — the union, as `(count, sum)`. A colimit has a morphism from every member, so every
    member's evidence factors through it; the smallest object with that property is the one whose
    incidence is exactly the union of the members'. Incidence counting is linear, so that union is
    the sum of the members' incidence vectors — the order-free `(count, sum)` accumulator
    `[[consolidation-is-learning]]` names. It is an accumulation rather than an averaging or a
    choice of representative: `sum` with `count` beside it loses nothing and can be merged again
    with a third member in any order.

`position` — the sum of the members' dense coordinates, for the same reason, and it is a genuine
    superposition rather than a blur. Cross-source members are orthogonal in the dense coordinate
    (`cos = +0.000000` exactly — see `diagram`'s docstring), and for mutually orthogonal `vᵢ` the sum
    obeys `(Σv)·vᵢ = ‖vᵢ‖²`. So every member's own position is exactly recoverable from the
    colimit's by projection, with each member's energy intact. An average divides that energy away
    and lands on a point that is none of the members — the "blur" a colimit is defined against — and
    `position_recovers_members` measures the difference.

`mass` — summed, with the count beside it. §5's *"fewer objects, more mass each"* is arithmetic: the
    members' mass accumulates onto the one object rather than being discarded on merge. A member
    carrying no measurable mass is counted as unmeasured, and no number is supplied for it.

`consolidates` — a morphism colimit → each member. The members remain and stay reconstructible;
    this is compression, not deletion.

Conservation is the acceptance test, and it can fail. Stack every member's evidence rows into one
incident frame and split it against the colimit's own band. If the colimit is the universal object,
its band spans every member's evidence and the residual is zero — the `PathLedger` terminates
without emitting. If any member carries a direction the colimit does not, the residual is that
direction's energy and the certificate says `balanced=False` with the leak measured. A compaction
that leaks is a broken merge, detectably (COMPACTIFICATION §6).

The test has teeth because the ledger is left un-emitted. `emit()` declares the residual handed back
as output, which makes `terminated` true for any residual at all. See `diagram.separation`.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import diagram as _diagram

__all__ = ["colimit", "smooth_edges", "COLIMIT_LABEL", "CONTENT_TYPE"]

COLIMIT_LABEL = "consolidates"          # canonical -> member, the declared compression morphism
CONTENT_TYPE = "application/x-concept"

# The rung order genesis declares (`SEED_RUNGS`), weakest first. The colimit takes the minimum.
_RUNG_ORDER = ("asserted", "hypothesis", "derived", "span_cited", "fetched", "observed")


def _colimit_id(member_ids: Sequence[str]) -> str:
    """A content id: blake2b over the sorted member ids.

    Sorted member ids make the id a function of the diagram alone, so re-deriving the same diagram
    re-mints the same id and the write is an idempotent upsert. A clock-derived hash would give the
    same colimit a different artifact on every take — the side-car-identity defect
    `[[EVERYTHING IS AN ARTIFACT]]` names."""
    h = hashlib.blake2b("\x00".join(sorted(member_ids)).encode("utf-8"), digest_size=16)
    return "concept-" + h.hexdigest()


def _rung(docs: Iterable[Dict[str, Any]]) -> Tuple[Optional[str], int]:
    """The colimit's provenance rung: the minimum over the members, and how many had none.

    Minimum rather than maximum: merging an observation with an assertion leaves the assertion an
    assertion. A member whose rung is absent or unrecognised is counted as unmeasured and reported
    as such, so it is neither the weakest rung nor the strongest."""
    best: Optional[int] = None
    unknown = 0
    for d in docs:
        p = (d.get("provenance") or "")
        p = p.rsplit(":", 1)[-1].strip().lower() if isinstance(p, str) else ""
        try:
            i = _RUNG_ORDER.index(p)
        except ValueError:
            unknown += 1
            continue
        best = i if best is None else min(best, i)
    return (_RUNG_ORDER[best] if best is not None else None), unknown


def _mass(docs: Iterable[Dict[str, Any]]) -> Tuple[float, int, int]:
    """`(sum, count, unmeasured)` — §5's *fewer objects, more mass each*, as arithmetic.

    A member with no numeric mass is counted in `unmeasured` and contributes nothing to the sum.
    `.get("mass", 1.0)` would supply a mass nobody measured, which is what
    `[[absence-is-not-an-affirmative-claim]]` names."""
    total, n, unmeasured = 0.0, 0, 0
    for d in docs:
        v = d.get("mass")
        if isinstance(v, (int, float)) and np.isfinite(float(v)):
            total += float(v)
            n += 1
        else:
            unmeasured += 1
    return total, n, unmeasured


def _position(store, member_ids: Sequence[str]) -> Tuple[Optional[np.ndarray], List[str], int]:
    """`(Σ dense_vec, the members it summed, how many had no coordinate)`, over the members'
    `crystal.ontology.geometry` coordinates.

    The sum rather than the mean — see the module docstring. A member with no resolvable coordinate
    is reported in the third value and left out of the sum. A zero vector is itself a position, so
    zero-filling would state a coordinate for a concept whose geometry was never read."""
    from crystal.ontology import geometry
    from crystal.ontology import driver as wn_store
    acc: Optional[np.ndarray] = None
    got: List[str] = []
    missing = 0
    for mid in member_ids:
        try:
            s = wn_store.synset(mid[3:] if mid.startswith("wn-") else mid)
            v = geometry.dense_vec(s, None)
        except Exception:
            v = None
        if v is None or not np.any(v):
            missing += 1
            continue
        v = np.asarray(v, dtype=float)
        acc = v.copy() if acc is None else acc + v
        got.append(mid)
    return acc, got, missing


def recover_from_position(pos: np.ndarray, vecs: Sequence[np.ndarray]) -> Dict[str, Any]:
    """The negative control on the position, as pure arithmetic: does `pos` still hold each `vᵢ`?

    For mutually orthogonal members `(Σv)·vᵢ = ‖vᵢ‖²` exactly, so projecting the colimit's position
    onto a member returns that member's full energy. Reports the worst relative error.

    An average fails this check, which is what gives it discriminating power. With two orthogonal
    members the mean returns `‖vᵢ‖²/2` — a relative error of 0.5, half of each member's energy
    divided away into a point that is neither member, the "blur" a colimit is defined against. A
    check both the sum and the mean passed would prove nothing, so
    `test_mean_position_loses_half_of_each_member` asserts the mean's 0.5 explicitly."""
    p = np.asarray(pos, dtype=float)
    worst, n = 0.0, 0
    for v in vecs:
        v = np.asarray(v, dtype=float)
        e = float(v @ v)
        if e <= 0:
            continue
        n += 1
        worst = max(worst, abs(float(p @ v) - e) / e)
    return {"members_checked": n, "worst_relative_error": worst}


def position_recovers_members(store, member_ids: Sequence[str], pos: np.ndarray) -> Dict[str, Any]:
    """`recover_from_position` over the members' live `crystal.ontology.geometry` coordinates — see
    it for the failure mode."""
    from crystal.ontology import geometry
    from crystal.ontology import driver as wn_store
    vecs = []
    for mid in member_ids:
        try:
            s = wn_store.synset(mid[3:] if mid.startswith("wn-") else mid)
            vecs.append(np.asarray(geometry.dense_vec(s, None), dtype=float))
        except Exception:
            continue
    return recover_from_position(pos, vecs)


def certify_merge(store, member_ids: Sequence[str], *, label_keyed: bool,
                  include_unanchored: bool,
                  anchor_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The acceptance test: does the colimit's band absorb every member's evidence, with no leak?

    Stacks the members' evidence rows into one incident frame over the union basis and splits it
    against the band that same union spans. The colimit is that union, so a balanced certificate
    says the merged object carries every member's evidence, whole.

    Failure mode: `balanced=False` with `loss > 0` means a member carries a direction the colimit
    does not — the merge lost a fact. `certificate['why']` names the hop. The ledger is left
    un-emitted, so a residual of any size fails the test."""
    cache = anchor_cache if anchor_cache is not None else {}
    evs = [_diagram.evidence(store, m, label_keyed=label_keyed,
                             include_unanchored=include_unanchored, anchor_cache=cache)
           for m in member_ids]
    keys = sorted({k for e in evs for k in e.weights})
    if not keys:
        return {"balanced": False, "measured": False,
                "why": "no member carried an evidence frame — nothing to certify"}
    if not all(e.exhaustive for e in evs):
        return {"balanced": False, "measured": False,
                "why": "a member's evidence could not be drained inside the measured instrument"}
    W = np.vstack([e.matrix(keys) for e in evs])
    B = _diagram.band(W)
    if B is None:
        return {"balanced": False, "measured": False, "why": "the union spans nothing"}

    from ember.optics import absorb_transmit
    from prism.conservation import PathLedger
    got = absorb_transmit(W, basis=B)
    if got is None:
        return {"balanced": False, "measured": False, "why": "the stacked frame carried no read"}
    absorbed, residual, k = got
    led = PathLedger(W, at="members")
    led.absorb(absorbed, residual, at="colimit-band", k=k)
    cert = led.certificate()          # left un-emitted -- see the module docstring
    return {"measured": True, "balanced": bool(cert["balanced"]), "k": int(k),
            "incident": cert["incident"], "absorbed": cert["absorbed"],
            "leak": cert["unaccounted"], "tolerance": cert["tolerance"],
            "closed": cert["closed"], "why": cert["why"], "features": len(keys),
            "rows": int(W.shape[0]), "certificate": cert}


def colimit(store, member_ids: Sequence[str], *, label_keyed: bool, include_unanchored: bool,
            author: Optional[str] = None, anchor_cache: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
    """Take the colimit of a derived diagram. Returns the artifact doc, the morphisms and the
    certificate, and writes nothing. Applying is the caller's separate, explicit act (`apply`).

    `member_ids` comes from `diagram.derive_diagram`. This function leaves the diagram as given —
    deciding what the diagram is belongs upstream — and certifies that the object it builds carries
    all of them, so a bad diagram surfaces as an unbalanced certificate."""
    ids = sorted(set(member_ids))
    if len(ids) < 2:
        return {"applied": False, "members": len(ids),
                "why": "a colimit of fewer than two objects is the object itself"}
    cache = anchor_cache if anchor_cache is not None else {}
    docs = [d for d in (store.artifacts.get_artifact(i) for i in ids) if d]
    if len(docs) < 2:
        return {"applied": False, "members": len(docs), "why": "fewer than two members exist"}
    ids = sorted(d["id"] for d in docs)

    cert = certify_merge(store, ids, label_keyed=label_keyed,
                         include_unanchored=include_unanchored, anchor_cache=cache)
    rung, rung_unknown = _rung(docs)
    mass_sum, mass_n, mass_missing = _mass(docs)
    pos, pos_members, pos_missing = _position(store, ids)
    recovery = (position_recovers_members(store, pos_members, pos)
                if pos is not None and pos_members else None)

    # The evidence union as (count, sum) — order-free, re-mergeable with a third member later.
    evs = [_diagram.evidence(store, m, label_keyed=label_keyed,
                             include_unanchored=include_unanchored, anchor_cache=cache)
           for m in ids]
    union: Dict[str, float] = {}
    for e in evs:
        for k, v in e.weights.items():
            union[k] = union.get(k, 0.0) + float(v)

    cid = _colimit_id(ids)
    lemmas = sorted({l for d in docs for l in (d.get("lemmas") or [])})
    doc: Dict[str, Any] = {
        "id": cid,
        "content_type": CONTENT_TYPE,
        "word": (docs[0].get("word") or (lemmas[0] if lemmas else cid)),
        "lemmas": lemmas,
        # The offer, without which a colimit is unfindable.
        #
        # Measured 2026-08-24 on 71/home: of 5,484 colimit objects in the store, **0 carry an
        # `offer`**. `search/ingest/pipeline_unified` indexes the OFFER — *"The lexical arm indexes
        # the offer rather than the body"* — so every merged object was invisible to the arm that
        # narrows a recall, while its own members stayed indexed.
        #
        # That inverts the whole proposition. COMPACTIFICATION §5 is *"fewer objects, more mass
        # each"*; consolidating into a heavier object and then indexing only the lighter members
        # means a search can never reach the thing the merge produced.
        #
        # The offer is the members' own lemmas, which is exactly what this object announces itself
        # as — nothing is invented, and the same set is already carried in `lemmas` and keyed in
        # `evidence`. A colimit whose members named nothing announces nothing, and that is the
        # honest empty rather than a placeholder.
        "offer": " ".join(lemmas),
        # NO GENERATED CONTENT. This carried `"colimit of %d records of one concept"` — a
        # sentence about the object, written by the code that made it, sitting in the field that
        # means "the bytes this artifact IS". It is a description at best, and it made every
        # colimit a row with a plaintext body for no reader's benefit.
        #
        # A colimit's content is its members'. `colimit_of` names them, `consolidates` edges reach
        # them, and they are all still present — 📄 *"The members remain and stay reconstructible;
        # this is compression, not deletion."* So there is nothing to restate here, and restating
        # it cost 5,484 rows a body they did not have.
        "context": {"kind": "concept", "colimit_of": ids,
                    "sources": sorted({i.split("-")[1] if i.startswith("wn-oewn-")
                                       else i.split("-")[0] for i in ids})},
        # Every member's provenance, kept whole, plus the rung the minimum resolves to.
        "colimit_of": ids,
        "member_provenance": {d["id"]: d.get("provenance") for d in docs},
        "provenance": rung,
        "provenance_unmeasured": rung_unknown,
        # (count, sum) — the order-free accumulators.
        "evidence_count": len(union),
        "evidence": union,
        "mass": mass_sum,
        "mass_count": mass_n,
        "mass_unmeasured": mass_missing,
        "position": (pos.tolist() if pos is not None else None),
        "position_members": pos_members,
        "position_unmeasured": pos_missing,
        "via": "op.consolidate.colimit",
    }
    if author:
        doc["created_by"] = author

    # ── File the colimit where its members live ─────────────────────────────────────────────────
    # Measured on 71/home 2026-08-23: 5,484 of 1,170,594 `application/x-concept` artifacts had NO
    # `collection_id` and no origin edge, and every one of them was a colimit written here. The
    # other 1,165,110 sit in `stage.0.lexicon`. The consequence is not cosmetic: with no collection,
    # `resolve_cell_principal` walks to the artifact ITSELF as its own origin root, so each colimit
    # becomes its own encrypted-search principal that no collection grant can reach — 5,484
    # principals the index writer holds nothing on. They indexed `sse=failed` and were unsearchable.
    #
    # Derived from the members, never a constant. A colimit belongs where the records it merges
    # belong; hardcoding `stage.0.lexicon` would be right on this corpus and wrong on the next one.
    # Members are read in id order, so the choice is deterministic rather than dict-order dependent.
    member_colls = [c for c in (d.get("collection_id") for d in docs) if c]
    coll = member_colls[0] if member_colls else None
    if coll:
        doc["collection_id"] = coll
    # `state` was absent too, and every member carries one. An artifact with no state is not
    # archived, so it is not skipped — it is routed to a segment chosen from nothing.
    states = [st for st in (d.get("state") for d in docs) if st]
    if states:
        doc["state"] = states[0]

    morphisms = [(cid, m, COLIMIT_LABEL, {"via": "op.consolidate.colimit"}) for m in ids]
    if coll:
        # The ORIGIN containment edge — collection -> colimit, `is_origin=1`. This is the edge
        # `get_origin_parent` looks for and `list_origin_descendants` walks, so without it a grant
        # on the collection reaches the members and not the thing that replaced them.
        #
        # NO `propagate` KEY, deliberately. The column's absent value is "unrestricted", which is
        # what every member's own containment edge carries. Writing `propagate=[]` here — the
        # default `add_context_edge` uses for context edges — would be `attenuation`'s ABSORBING
        # DENY: authority would stop dead at this edge and the colimit would be unreachable for a
        # second, subtler reason than the one this fix exists to repair.
        morphisms.append((coll, cid, "contains", {"is_origin": 1,
                                                 "via": "op.consolidate.colimit"}))
    return {"applied": False, "id": cid, "doc": doc, "members": ids,
            "morphisms": morphisms, "certificate": cert,
            "position_recovery": recovery,
            # The verdict `apply_colimit` gates on: an unbalanced certificate means a member's
            # evidence did not survive the merge, so the merge is not written.
            "acceptable": bool(cert.get("balanced"))}


def smooth_edges(store, mapping: Dict[str, str]) -> Dict[str, Any]:
    """Land the members' edges on their colimits without duplicating facts.

    `mapping` is member id -> colimit id. Every edge incident to a member is rewritten with both
    endpoints replaced by their colimit (an endpoint that merged into nothing maps to itself), and
    the rewritten set is returned as `(src, dst, label, props)` tuples ready for `add_edges`.

    The deduplication is free, and the edge table gives it. `edge_key = blake2b(src ‖ dst ‖ label)`
    is the primary key and `add_edges` is an `ON CONFLICT DO UPDATE`, so `a --instance_of--> b` and
    `a' --instance_of--> b'` — with `{a,a'}` and `{b,b'}` merged — rewrite to the same
    `(src, dst, label)` triple, hash to the same `edge_key`, and land as one row. That means no
    dedup pass, no `composed_of` bookkeeping and no set arithmetic here: idempotency is already
    load-bearing for mesh replay and this inherits it. `emitted` against `distinct` below measures
    the collapse.

    Self-edges are dropped and counted. Two members of one diagram that pointed at each other
    (OEWN's `instance_hyponym` back-edge, say) become an edge from the colimit to itself, which
    asserts nothing. It is reported as `self_edges`.

    `composed_of` stays out of this write. `genesis.py`'s `SEED_ETYPES` declares it as
    `operator -> operator` ("this morphism = composition of these generators"), which is about
    operators composing rather than a concept's edges landing on a colimit."""
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    emitted = 0
    self_edges = 0
    not_exhaustive: List[str] = []
    for member in sorted(mapping):
        for direction in ("out", "in"):
            rows, ex = _diagram._all_edges(store, member, direction=direction)
            if not ex:
                not_exhaustive.append(member)
                continue
            for e in rows:
                if e["label"] == COLIMIT_LABEL:
                    continue                    # the compression morphism is not a fact to relocate
                src = mapping.get(e["src"], e["src"])
                dst = mapping.get(e["dst"], e["dst"])
                emitted += 1
                if src == dst:
                    self_edges += 1
                    continue
                out[(src, dst, e["label"])] = dict(e.get("props") or {})
    edges = [(s, d, l, p) for (s, d, l), p in sorted(out.items())]
    return {"edges": edges, "emitted": emitted, "distinct": len(edges),
            "collapsed": emitted - self_edges - len(edges), "self_edges": self_edges,
            "not_exhaustive": sorted(set(not_exhaustive))}


def apply_colimit(store, built: Dict[str, Any], *, smooth: bool = True) -> Dict[str, Any]:
    """Write a built colimit. An unbalanced certificate is not written; the certificate comes back
    with the result instead.

    Morphisms land before the artifact's edges, as `genesis.consolidate_nearvdup` orders its writes:
    a crash after the `consolidates` edges leaves the members merely un-smoothed, and every colimit
    on disk still points at the members it is the colimit of."""
    if not built.get("acceptable"):
        return {"applied": False, "why": "certificate is not balanced — refusing to write a merge "
                                         "that loses a member's evidence",
                "certificate": built.get("certificate")}
    store.artifacts.put_many([built["doc"]], batch=1)
    store.graph.add_edges(built["morphisms"], batch=500)
    res = {"applied": True, "id": built["id"], "members": len(built["members"]),
           "morphisms": len(built["morphisms"])}
    if smooth:
        mapping = {m: built["id"] for m in built["members"]}
        sm = smooth_edges(store, mapping)
        # Report what was written, rather than what was offered. `add_edges` returns the number
        # handled, and the shortfall beside it is the data-loss signal the caller reads
        # ([[mesh-cursor-must-not-skip]]: the apply reports what it actually wrote).
        handled = store.graph.add_edges(sm["edges"], batch=500)
        res["smoothed"] = dict(sm, handled=handled, shortfall=len(sm["edges"]) - handled)
        res["smoothed"].pop("edges", None)
    return res
