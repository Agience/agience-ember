"""Derive the diagram of a concept: the artifacts whose evidence cannot separate them from it.

Two artifacts are one object when the evidence cannot separate them, and that is a measurement
rather than a string rule. The diagram comes from the store; nothing is passed in but a concept id.

The basis. PWN and OEWN records of one concept occupy strictly orthogonal subspaces of the
`crystal.ontology.geometry` dense coordinate::

    cos(dense_vec(wn-einstein.n.01), dense_vec(wn-oewn-10974490-n))  =  +0.000000

`dense_vec` signed-hashes the names of the nodes on a synset's own hypernym path, and the two id
families share not one name::

    wn-einstein.n.01    einstein.n.01 -> physicist.n.01 -> scientist.n.01 -> ... -> entity.n.01
    wn-oewn-10974490-n  oewn-10974490-n -> oewn-10447768-n -> ... -> oewn-00001740-n

A band built from `dense_vec` therefore transmits the whole residual for every cross-source pair, by
construction: the coordinate is keyed on the very identity the measurement is meant to establish, so
it can only confirm an identity it already had. This module uses a different basis.

The two universes are joined by vertices shared by construction. `lemma:einstein` is one artifact in
the lattice and carries a `lex:en` edge into `wn-einstein.n.01` and into `wn-oewn-10974490-n`. Those
edges are observations the ingest already recorded, so reading them is a lattice walk rather than a
comparison this module performs. The evidence frame is built over those shared anchor vertices:

    a concept's evidence  =  its incident edges, each neighbour expanded into the shared vertices
                             that neighbour anchors on.

No hash, no width, no collision. The basis is the exact union of the two concepts' feature keys —
one column per distinct key, built per pair. A corpus-wide coordinate hashes into a fixed width
because it must be comparable everywhere; a pairwise separation has no such need, so no width is
invented and no collision is incurred. A collision here could only make two distinct facts share a
column and merge things that differ.

The verdict is `prism.conservation`, and it has no threshold.
`absorb_transmit(E(b), basis=band(a))` splits b's evidence into the part a's band absorbs and the
residual it does not; the split is an orthogonal projection, so `‖E(b)‖² = ‖absorbed‖² + ‖residual‖²`
exactly. A `PathLedger` opened on `E(b)` records that one hop and is not `emit()`ed, so
`certificate()["terminated"]` is true exactly when the residual is zero within the tolerance the
ledger derives from the arithmetic it performed. "Terminated without emitting" is the ledger's own
name for *the elements absorbed everything*, and here the element is the other concept's band: a's
evidence accounts for all of b's, with nothing left travelling. There is no similarity score and
nothing to tune.

It is measured both ways. `residual(b|a) == 0` alone says a's band is at least as wide as b's, which
is subsumption rather than identity — `dog` is absorbed by `animal`'s band. One object requires both.

The corpus is unbounded; the instrument is finite. There is no top-N candidate list, no maximum
diagram size and no sampling. Every read drains (`_all_edges` grows its batch geometrically until
the page comes back short), and the only bound is the one `prism.envelope` measures on this box.
When a read cannot be drained inside the envelope the evidence is marked not exhaustive and nothing
merges — that is an absence, and it is reported as one.

Two keyings and two treatments of the unanchored make four questions, and the difference between
their answers is itself a measurement:

  · `label_keyed` — is the relation label part of the evidence? With it, `wn-einstein.n.01`
    (`instance_of -> physicist`) and `wn-oewn-10974490-n` (`instance_of -> physicist` and
    `instance_hyponym <- physicist`) are separated by OEWN having recorded the converse of a
    relation PWN recorded once — the same fact written twice. The corpus declares no inverse pairs
    (`SEED_ETYPES` in `genesis.py` has no inverse field), so whether `instance_of` and
    `instance_hyponym` are converses is itself unmeasured, and both keyings are read.
  · `include_unanchored` — is a neighbour the shared lexicon cannot name (a citation, a collection,
    a source-private node) evidence about the concept, or apparatus around the record? Included, it
    is a private feature no other source can absorb, and two records of one concept ingested from
    two files are separated by their provenance. Excluded, provenance stops being grounds for
    separation, which is what a colimit is for: it carries both provenances rather than being
    blocked by them. Both are measured.

Both are required arguments at every entry point, so the choice always appears in the caller's
report.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ANCHOR_LABEL", "anchors", "evidence", "band", "separation", "candidates",
           "derive_diagram", "Evidence", "instrument_rows"]

# The lexical anchor edge: `lemma:<form> --lex:en--> <concept>`. Its source vertices are `lemma:*`
# artifacts co-cited by PWN, OEWN and ConceptNet alike, which is what makes it a bridge between
# ingest sources. `anchors()` returns empty when a concept has none, and empty is read as absence.
ANCHOR_LABEL = "lex:en"


def instrument_rows(row_bytes: int) -> int:
    """How many rows fit in the resource envelope, expressed in rows.

    The instrument has a single resource envelope, and it is the only thing that limits instrument flow.
    Consolidation therefore has no bound of its own: no batch size, no candidate cap, no
    diagram-size limit, no sampling rate. This function mints nothing. It reads the one envelope and
    divides by the measured size of one row, so the machine-wide reading can be stated in the unit
    this call site works in.

    `prism.envelope.mem_available_bytes()` is the reader: cgroup headroom, else the Windows job
    object, else `MemAvailable`. It is the single envelope call site on the consolidation path.

    Returns `0` when the envelope could not be read. That is an unmeasurable envelope rather than an
    infinite one, and callers treat it as such."""
    try:
        from prism.envelope import mem_available_bytes
        avail = mem_available_bytes()
    except Exception:
        return 0
    if not avail or int(row_bytes) <= 0:
        return 0
    return int(int(avail) // int(row_bytes))


# One edge row as materialised by `edges_of`: nine columns plus a props dict. `_row_bytes` sizes an
# actual row the store returned, so the figure is a measurement of the real object.
def _row_bytes(sample: Any) -> int:
    import sys as _sys
    try:
        n = _sys.getsizeof(sample)
        if isinstance(sample, dict):
            for k, v in sample.items():
                n += _sys.getsizeof(k) + _sys.getsizeof(v)
        return max(1, int(n))
    except Exception:
        return 1


def _all_edges(store, aid: str, *, direction: str, label: Optional[str] = None
               ) -> Tuple[List[Dict[str, Any]], bool]:
    """Every incident edge on one side of `aid`, and whether the read drained.

    `LatticeGraphStore.edges_of` takes a `limit` and offers no cursor, so draining it means asking
    for more until the page comes back short — the page is short exactly when it is the whole set.
    The batch doubles, so the total work is geometric in the true degree and the corpus is nowhere
    bounded; the growth stops at the instrument `prism.envelope` measures on this box.

    Returns `(rows, exhaustive)`. `exhaustive=False` means the node is wider than this box can hold
    at once: not yet read, which is a different state from approximately read. Callers treat it as
    unmeasured, because a merge decided on a prefix of the evidence is a check that cannot fail."""
    want = 1
    rows: List[Dict[str, Any]] = []
    while True:
        # `partial_ok=True` is required, and it is the honest flag here.
        #
        # `edges_of` RAISES `EdgesTruncated` rather than returning a clipped list — it refuses to
        # hand back a prefix a caller cannot distinguish from a complete set. That is the right
        # default and this loop is the documented exception to it: `partial_ok=True` is "a caller
        # that genuinely wants a bounded peek and has said so", which is exactly what a doubling
        # probe is. It asks for `want` rows in order to LEARN whether there are more, and the short
        # page below is how it finds out.
        #
        # Without the flag this loop could not run at all: it opens at `want = 1`, so ANY node with
        # two or more incident edges raised on the first iteration. That is what
        # `test_consolidate_colimit.py` was failing on (`oewn-e has more than 1 edge(s) outbound`) —
        # the loop was written against an older `edges_of` that truncated, and was never updated
        # when the contract changed to raise.
        #
        # The truncation is safe precisely because the "did it drain" answer does not come from the
        # row count alone: `partial_ok` caps the page at `want`, so `len(rows) < want` still means
        # the whole set, and every other exit below returns `exhaustive=False`.
        rows = store.graph.edges_of(aid, label=label, direction=direction, limit=want,
                                    partial_ok=True)
        if len(rows) < want:
            return rows, True                      # short page == the whole set
        cap = instrument_rows(_row_bytes(rows[0]))
        if cap <= 0:
            return rows, False                     # envelope unmeasurable -> unverified, not "done"
        if want >= cap:
            return rows, False                     # wider than the instrument -> not yet read
        want = min(want * 2, cap)


class Evidence:
    """One concept's evidence as a sparse incidence over feature keys, plus what it could not read.

    `weights` maps a feature key to the number of incident edges asserting it — a measured count, and
    the order-free `(count, sum)` accumulator consolidation uses elsewhere. `rows` is the row
    grouping: each entry is the subset of feature keys contributed by one incident edge, so the frame
    has one row per observation rather than one row per concept. A single row spans a line, which
    would make every pair of concepts trivially one object."""

    __slots__ = ("aid", "rows", "weights", "anchor_ids", "unanchored_rows", "exhaustive")

    def __init__(self, aid: str, rows: List[Dict[str, float]], anchor_ids: List[str],
                 unanchored_rows: int, exhaustive: bool) -> None:
        self.aid = aid
        self.rows = rows
        self.anchor_ids = anchor_ids
        self.unanchored_rows = unanchored_rows
        self.exhaustive = exhaustive
        w: Dict[str, float] = {}
        for r in rows:
            for k, v in r.items():
                w[k] = w.get(k, 0.0) + float(v)
        self.weights = w

    @property
    def keys(self) -> List[str]:
        return sorted(self.weights)

    def matrix(self, basis: Sequence[str]) -> np.ndarray:
        """The `(T, F)` frame over the given feature basis, in row order."""
        idx = {k: i for i, k in enumerate(basis)}
        W = np.zeros((len(self.rows), len(basis)), dtype=np.float64)
        for i, r in enumerate(self.rows):
            for k, v in r.items():
                j = idx.get(k)
                if j is not None:
                    W[i, j] = float(v)
        return W

    def __repr__(self) -> str:  # pragma: no cover - diagnostic
        return "<Evidence %s rows=%d features=%d anchors=%d exhaustive=%s>" % (
            self.aid, len(self.rows), len(self.weights), len(self.anchor_ids), self.exhaustive)


def anchors(store, aid: str) -> Tuple[List[str], bool]:
    """The shared vertices this artifact anchors on — the sources of its incoming anchor edges — and
    whether that read drained.

    An indexed `ix_e_dst` seek over one artifact's incoming anchor edges, drained by `_all_edges`.
    Empty-and-exhaustive is a real answer meaning *this artifact anchors on nothing shared*; empty
    -and-not-exhaustive means *not read yet*. They are different states and are returned as such."""
    # Both anchor labels are read. ConceptNet entries anchor through `assoc:en`
    # (`lemma:1530s --assoc:en--> cn-1530s`) rather than `lex:en`, so reading the lexical label alone
    # leaves them with no shared vertex, no features and no evidence frame — every ConceptNet
    # candidate then comes back unmeasured with `rows_b = 0`.
    rows, ex = _all_edges(store, aid, direction="in", label=ANCHOR_LABEL)
    rows2, ex2 = _all_edges(store, aid, direction="in", label="assoc:en")
    return sorted({r["src"] for r in rows} | {r["src"] for r in rows2}), bool(ex and ex2)


def evidence(store, aid: str, *, label_keyed: bool, include_unanchored: bool,
             anchor_cache: Optional[Dict[str, Tuple[List[str], bool]]] = None) -> Evidence:
    """Derive one concept's evidence frame over the shared-anchor basis.

    One row per incident edge (both directions). A row's features are the neighbour's shared anchors,
    keyed `<label>|<anchor>` when `label_keyed`, else `<anchor>`. A neighbour with no shared anchor
    is either carried as the private feature `<label>|id:<neighbour>` or dropped as apparatus,
    per `include_unanchored`; either way it is counted and reported (`unanchored_rows`).

    The concept's own anchors are not a row — see the inline note below. The anchor is what makes a
    pair a candidate, so it is held out of the evidence that decides.

    Both keyword arguments are required. They select which of four questions is being asked (see the
    module docstring), and a default would hide that choice from the answer."""
    cache = anchor_cache if anchor_cache is not None else {}

    def _anchors_of(nid: str) -> Tuple[List[str], bool]:
        got = cache.get(nid)
        if got is None:
            got = anchors(store, nid)
            cache[nid] = got
        return got

    def _prefetch(nids: List[str]) -> None:
        """Fill `cache` for many neighbours in one read instead of one probe each.

        Pure performance — the same anchors, by the same definition, in the same order.
        `anchors()` is `_all_edges` twice, and `_all_edges` is a doubling probe: it asks for 1 row,
        then 2, then 4, until a page comes back short. That is the right shape for one unknown node
        and the wrong shape for four thousand known ones.

        Measured 2026-08-25 on 71/home: `superposition(cn-singlish)` spent **80.4 s of 111.4 s**
        inside a single candidate, `cn-music` — 4,164 edges, so 4,164 neighbours, so ~8,300 probe
        sequences. Three candidates accounted for **93%** of the whole call.

        The envelope discipline is preserved, not bypassed. `_all_edges` refuses to hand back a
        prefix a caller cannot distinguish from a complete set, so this reads a bounded page and
        **falls back to the per-id probe for the whole chunk** the moment a page comes back full —
        the one case where the batch cannot prove it drained. A cheap read that is sometimes wrong
        about exhaustiveness would be worse than the slow one it replaces.
        """
        want = [n for n in nids if n not in cache]
        if not want:
            return
        try:
            conn = store.artifacts.db.read()
        except Exception:
            return                              # no batched read: the per-id path still answers
        marks = ",".join("?" * len(_ANCHOR_LABELS))
        for i in range(0, len(want), 400):
            chunk = want[i:i + 400]
            cap = len(chunk) * 64 + 64
            try:
                got = conn.execute(
                    "SELECT dst, src FROM edge WHERE dst IN (%s) AND label IN (%s) LIMIT %d"
                    % (",".join("?" * len(chunk)), marks, cap),
                    list(chunk) + list(_ANCHOR_LABELS)).fetchall()
            except Exception:
                continue                        # fall through to the per-id probe for this chunk
            if len(got) >= cap:
                continue                        # cannot prove it drained — leave it to `anchors()`
            found: Dict[str, set] = {}
            for dst, src in got:
                found.setdefault(str(dst), set()).add(str(src))
            for n in chunk:
                cache[n] = (sorted(found.get(n, ())), True)

    rows: List[Dict[str, float]] = []
    unanchored = 0
    own, exhaustive = _anchors_of(aid)

    # The concept's own anchors are held out of its evidence, because including them is circular.
    # `candidates()` generates the pair because the two artifacts share an anchor, so feeding that
    # same anchor back in as evidence has every candidate begin the measurement already agreeing on
    # its only discriminating row, and the reach then decides the verdict it is meant to offer.
    #
    # Satellite adjectives make the effect concrete: they carry no relation edges, so the self-anchor
    # row would be their entire frame and every sense of a word would carry an identical one:
    #
    #     wn-oewn-88455131-a  "causing indignation due to hypocrisy"
    #       with  wn-rich.s.01  "of great worth or quality"
    #       with  wn-rich.s.04  "very productive"
    #       with  wn-rich.s.05  "containing plenty of fat, or eggs, or sugar"
    #
    # Four distinct senses of `rich` collapsing into one object is the `bank` failure in miniature.
    # Holding the row out is neither a tightening nor a threshold: it removes an input that can only
    # assert what the reach already assumed. A concept whose evidence is then empty is reported
    # `measured=False` and merges with nothing, which is the honest answer — there is no evidence
    # about it beyond what it is called.
    _edges = []
    for direction in ("out", "in"):
        got, ex = _all_edges(store, aid, direction=direction)
        exhaustive = exhaustive and ex
        _edges.append((direction, got))
    # Every neighbour this frame will ask about, resolved in one pass rather than one probe each.
    _prefetch([str(e["dst"] if d == "out" else e["src"]) for d, g in _edges for e in g])
    for direction, got in _edges:
        for e in got:
            label = e["label"]
            if label == ANCHOR_LABEL:
                continue                      # the anchor is the reach; the evidence is elsewhere
            nid = e["dst"] if direction == "out" else e["src"]
            na, nex = _anchors_of(nid)
            exhaustive = exhaustive and nex
            if na:
                feats = ["%s|%s" % (label, a) for a in na] if label_keyed else list(na)
            else:
                unanchored += 1
                if not include_unanchored:
                    continue
                feats = ["%s|id:%s" % (label, nid)] if label_keyed else ["id:%s" % nid]
            rows.append({f: 1.0 for f in feats})

    return Evidence(aid, rows, own, unanchored, exhaustive)


def band(W: np.ndarray):
    """The `(F, k)` coupling band a frame's evidence spans — `prism.frames.offer_basis`.

    The same construction `match.tekton_basis` uses for a tekton's offer, and for the same reason:
    these coordinates are the directions (an incidence that was observed is an observation, not a
    noisy sample of one), so the band is the span rather than an instrument read. `None` when the
    evidence spans nothing.

    `offer_basis` drops singular values below `1e-9·σ_max`. That is numerical rank determination —
    which directions the arithmetic can represent at all — and it is independent of any pair. The
    verdict is the residual, below."""
    from prism.frames import offer_basis
    return offer_basis(W)


def separation(store, a_id: str, b_id: str, *, label_keyed: bool, include_unanchored: bool,
               ev_a: Optional[Evidence] = None, ev_b: Optional[Evidence] = None,
               anchor_cache: Optional[Dict[str, Tuple[List[str], bool]]] = None) -> Dict[str, Any]:
    """Measure whether the evidence can separate two artifacts. The whole verdict is a conservation
    certificate; there is no score and no cutoff.

    Both frames are built over the exact union of their feature keys, so each is represented in full:
    a feature only `b` has is a real column, and `a`'s band leaves it in the residual.

    Returns `one_object` plus, in both directions, the residual energy, the residual fraction, the
    ledger's derived tolerance, and the certificate. `one_object` is true exactly when both
    certificates terminate without emitting — each band absorbed the whole of the other's evidence.
    One direction alone is subsumption (`dog` lies inside `animal`'s band) rather than identity."""
    ev_a = ev_a if ev_a is not None else evidence(
        store, a_id, label_keyed=label_keyed, include_unanchored=include_unanchored,
        anchor_cache=anchor_cache)
    ev_b = ev_b if ev_b is not None else evidence(
        store, b_id, label_keyed=label_keyed, include_unanchored=include_unanchored,
        anchor_cache=anchor_cache)
    basis_keys = sorted(set(ev_a.weights) | set(ev_b.weights))
    out: Dict[str, Any] = {
        "a": a_id, "b": b_id, "label_keyed": bool(label_keyed),
        "include_unanchored": bool(include_unanchored),
        "features": len(basis_keys), "rows_a": len(ev_a.rows), "rows_b": len(ev_b.rows),
        "unanchored_a": ev_a.unanchored_rows, "unanchored_b": ev_b.unanchored_rows,
        "exhaustive": bool(ev_a.exhaustive and ev_b.exhaustive),
    }
    # A merge is decided on the whole of the evidence. When either side's incidence could not be
    # drained inside this box's instrument the pair is not yet measured: the unread tail is exactly
    # where a separating fact would be, so merging on what fit is a check that cannot fail.
    if not out["exhaustive"]:
        out.update({"one_object": False, "measured": False,
                    "why": "evidence could not be drained inside the measured instrument — "
                           "NOT YET READ, which is not the same as identical"})
        return out
    if not basis_keys or not ev_a.rows or not ev_b.rows:
        # No evidence is an absence, not an agreement. Two artifacts about which nothing was observed
        # have a residual of zero trivially, and reading that as one object would merge the whole
        # unobserved tail of the corpus into a single blob.
        out.update({"one_object": False, "measured": False,
                    "why": "no evidence frame on one or both sides — unmeasured, not identical"})
        return out

    Wa, Wb = ev_a.matrix(basis_keys), ev_b.matrix(basis_keys)
    Ba, Bb = band(Wa), band(Wb)
    if Ba is None or Bb is None:
        out.update({"one_object": False, "measured": False,
                    "why": "evidence spans nothing on one or both sides"})
        return out

    from ember.optics import absorb_transmit
    from prism.conservation import PathLedger

    def _hop(W, basis, at):
        got = absorb_transmit(W, basis=basis)
        if got is None:
            return None
        absorbed, residual, k = got
        led = PathLedger(W, at=at)
        led.absorb(absorbed, residual, at="band", k=k)
        # Deliberately left un-emitted. `emit()` declares the residual handed back as output, which
        # makes `terminated` true for any residual whatsoever — a check that cannot fail. Un-emitted,
        # `terminated` is exactly `|residual| <= derived tolerance`, which is the question.
        cert = led.certificate()
        e0 = cert["incident"]
        return {"k": int(k), "incident": e0, "absorbed": cert["absorbed"],
                "residual": cert["unaccounted"],
                "residual_fraction": (cert["unaccounted"] / e0) if e0 else 0.0,
                "tolerance": cert["tolerance"], "closed": cert["closed"],
                "absorbed_everything": bool(cert["terminated"]), "balanced": cert["balanced"],
                "why": cert["why"]}

    b_into_a = _hop(Wb, Ba, "b-evidence")
    a_into_b = _hop(Wa, Bb, "a-evidence")
    if b_into_a is None or a_into_b is None:
        out.update({"one_object": False, "measured": False,
                    "why": "frame carried no read in one direction"})
        return out

    out.update({
        "measured": True,
        "b_into_a": b_into_a, "a_into_b": a_into_b,
        # The symmetric read: the larger of the two residual fractions. Reported so a caller can see
        # how far from one object a pair is; the verdict is `one_object`, from the two certificates.
        "residual_fraction": max(b_into_a["residual_fraction"], a_into_b["residual_fraction"]),
        "one_object": bool(b_into_a["absorbed_everything"] and a_into_b["absorbed_everything"]),
        "subsumed": bool(b_into_a["absorbed_everything"] != a_into_b["absorbed_everything"]),
    })
    return out


_ANCHOR_LABELS = (ANCHOR_LABEL, "assoc:en")

#: The containment edge, which is never a relation. `mantle/db/vertex.py::_place` writes
#: `(collection, artifact, "contains")`, so treating it as one reaches every member of the
#: collection — 1.16M of them for `stage.0.conceptnet`.
_NOT_A_RELATION = "contains"


def relates_to(store, aid: str) -> Dict[str, float]:
    """The vertices `aid` relates to, and what sharing each one is WORTH — `{vertex: weight}`.

    ## Why this exists beside `anchors()`

    `anchors()` reads `lex:en` and `assoc:en`: both are LEMMA-HUB edges, so an artifact the English
    lexicon never named has no anchors, no candidates and no evidence frame — whatever relations it
    actually carries. Measured 2026-08-25 over 40 such ConceptNet terms:

        with NO relation at all                       0 of 40
        non-containment degree   1-4: 27 · 5-19: 10 · 20+: 3
        `anchors()` -> 0 anchors                     38 of 40

    📄 `candidates()` justifies its reach as a property of the graph — *"there is no path in the
    lattice along which evidence about it could reach `aid`"*. `cn-nosode --related_to-->
    cn-homeopathy` is such a path. The bound was a label whitelist, and this is the graph itself.

    ## The weight is the ambiguity, and nothing is chosen

    Sharing `astronomy` says almost nothing — hundreds of terms point at it. Sharing `autonosodes`
    says a great deal. That difference is `1/(1 + log(in-degree))`: the same branching-entropy
    reading `lookup.hop_cost` already takes, in the same direction, with no coefficient.

    Measured against the alternatives over the same 14 terms — flat weights, and a degree ceiling
    of 500 — entropy landed 11/14 against 8/12 for both, and landed BETTER: the other two put
    `cn-laminarin` on `cn-1,1,2_trichloro_1,2,2_t` and `cn-checkback` on `cn-52_card_pickup`. It is
    also the only one of the three with no chosen number in it.

    **Containment is excluded and that is load-bearing**, not tidiness — see `_NOT_A_RELATION`.

    **Deliberately NOT memoised, and that was measured rather than assumed.** A per-store cache
    was added on the theory that recomputing neighbour degrees drove `superposition(cn-singlish)`'s
    **96.7 s**, and it changed nothing — 122 s after, inside the noise of a contended store. Profiled
    instead: `candidates()` 2.4 s, `evidence(source)` 0.0 s, `separation()` 0.27 s each warm, so the
    39 candidates account for ~13 s of it. **The remainder is building evidence for the few
    candidates that are themselves high-degree**, which no cache on this function reaches. The cache
    also held every neighbourhood it ever read, keyed on `id(store)`, for the life of the process.
    """
    import math as _math
    out: Dict[str, float] = {}
    conn = store.artifacts.db.read()
    for sql, col in (("SELECT dst, label FROM edge WHERE src = ?", "dst"),
                     ("SELECT src, label FROM edge WHERE dst = ?", "src")):
        try:
            rows = conn.execute(sql, (aid,)).fetchall()
        except Exception:
            return {}                          # NOT cached: an unread node is not an empty one
        for r in rows:
            nid, label = str(r[0]), str(r[1])
            if label == _NOT_A_RELATION or nid == aid:
                continue
            if nid not in out:
                try:
                    deg = conn.execute(
                        "SELECT count(*) FROM edge WHERE dst = ? AND label <> ?",
                        (nid, _NOT_A_RELATION)).fetchone()[0]
                except Exception:
                    continue
                out[nid] = 1.0 / (1.0 + _math.log(max(1.0, float(deg))))
    return out


def candidates(store, aid: str,
               anchor_cache: Optional[Dict[str, Tuple[List[str], bool]]] = None) -> Iterable[str]:
    """The artifacts reachable from `aid` through a shared anchor vertex — the diagram's search space.

    A two-hop indexed walk: `aid <-anchor- lemma -anchor-> other`, both legs `ix_e_dst`/`ix_e_src`
    seeks. This is the reach, and the verdict comes later. Anything sharing no anchor with `aid` is
    outside it because there is no path in the lattice along which evidence about it could reach
    `aid` — the same reach bound the mesh works under, rather than a rule about strings.

    It is deliberately generous and uncapped: `lemma:bank` reaches both the slope and the
    institution, so the negative control is offered to the measurement. No top-N, no cap —
    `neighbors()` streams the whole indexed range."""
    cache = anchor_cache if anchor_cache is not None else {}
    own = cache.get(aid)
    if own is None:
        own = anchors(store, aid)
        cache[aid] = own
    seen = {aid}
    # The second leg follows both anchor labels, which is what puts ConceptNet inside the reach:
    #     wn-cow.n.01  <-lex:en-  lemma:cow  -assoc:en->  cn-cow
    # differs from the WordNet path only in the second leg's label. `assoc:<lang>` is a distinct
    # label from `lex:<lang>` (`seed_lattice.ASSOC_LABEL_PREFIX`: the entry edge belongs to the
    # language transducer and the associative edge does not), and both are anchors.
    #
    # Walking `lex:en` on both legs reaches WordNet alone — `wn-cow.n.01` finds 7 candidates and
    # certifies a merge with `wn-oewn-02406106-n`, which is where the ~75,000 PWN/OEWN duplicates
    # come from — while `cn-cow`, carrying 338 edges against `wn-cow.n.01`'s one, stays outside it.
    for a in own[0]:
        for lab in _ANCHOR_LABELS:
            for nid in store.graph.neighbors(a, lab, direction="out"):
                if nid not in seen:
                    seen.add(nid)
                    yield nid

    # ── and everything sharing a RELATION, not only a surface form ───────────────────────────────
    # ADDITIVE. Everything the lemma-hub walk above found has already been yielded, so no artifact
    # that was a candidate stops being one and no existing merge loses its reach. What this adds is
    # the artifacts an English lemma hub could never reach — measured, `anchors` returns nothing
    # for 38 of 40 ConceptNet terms that each carry 1-20 real relations.
    #
    # It is not free of consequence even so: `superposition` decides by whether the series of
    # residuals SEPARATES, and adding candidates changes that series. It can therefore move a
    # verdict on the WordNet side too, which is why `relates_to` weights by branching entropy — a
    # hub shared by hundreds contributes almost nothing — and why the suite is the check.
    # ── the horizon prunes it, and without that this does not terminate ─────────────────────────
    # MEASURED, not anticipated: the first version pruned nothing — the entropy weight is never
    # zero — and `superposition(cn-cow)` went from **16.1 s to past 10 minutes**. `cn-cow` carries
    # 338 edges and every neighbour's whole neighbourhood became a candidate.
    #
    # The bound is the corpus's own and is the one `lookup.hop_cost` already states: travelling
    # through a vertex costs `log(in-degree)` nats, and `prism.resolution.horizon(xi, gap)` is how
    # far anything travels before it stops clearing the propagation floor. Measured 2026-08-25: xi 0.4647,
    # gap 0.0263, horizon **1.6909 nats** — so a vertex reachable from more than e^1.69 ≈ 5 places
    # is past it. That is not a cap on candidates; it is a statement that a vertex hundreds of
    # things point at is not evidence about any of them.
    #
    # No measured geometry means NO PRUNE — not no arm. Making the horizon a precondition for
    # running at all was tried and reverted: `xi()`/`propagation_floor()` are None on any store without a
    # measured corpus, so the whole feature went silently inert on every test fixture, and a feature
    # that only works on the production corpus cannot be tested. The prune is a cost bound; the
    # correctness of the reach does not depend on it, and a store small enough to lack a geometry is
    # small enough not to need one.
    from crystal.ontology import driver as _wn
    try:
        from ember.ontology.match import xi as _xi, propagation_floor as _gap
        from prism.resolution import horizon as _horizon
        _x, _g = _xi(), _gap()
        _reach = None if (_x is None or _g is None) else float(_horizon(_x, _g))
    except Exception:  # noqa: BLE001 — no measured geometry, no widened reach
        _reach = None
    # Only where the lemma hub finds nothing. Measured: unguarded, this arm took
    # `superposition(cn-cow)` from **16.1 s to past 10 minutes** even with the horizon prune,
    # because the second level expands every surviving neighbour's neighbourhood uncached —
    # O(degree x neighbour-degree) counting queries, ~34,000 of them for a 338-edge concept.
    #
    # The guard is not a narrowing of the rule, it is the rule: an artifact's own relations place it
    # when the lexicon cannot. An artifact the lexicon named already has a reach, that reach is
    # cheaper and better attested, and widening it buys nothing while costing everything. 38 of 40
    # ConceptNet tail terms have no lemma anchor at all, so this arm is where the whole population
    # that needed it lives.
    #
    # It also makes the change strictly additive and therefore incapable of moving an existing
    # verdict: a candidate set that was non-empty is untouched, so no merge already certified on the
    # WordNet side can change. The unguarded form could move those, and remains unmeasured.
    for nid, w in (relates_to(store, aid).items() if not own[0] else ()):
        if w <= 0.0:
            continue
        if _reach is not None and (1.0 / w - 1.0) > _reach:
            continue                          # log(in-degree) past the horizon: not evidence                          # log(in-degree) past the horizon: not evidence
        for other in relates_to(store, nid):
            if other in seen:
                continue
            # A lemma is not a candidate. It is travelled through — that is what an anchor is —
            # and offering one as a thing to merge with is a category error: `lemma:physicist` is a
            # surface form, not a concept. 📄 `driver.is_lemma_id`: *"a lexical entry edge joins a
            # concept to its own surface, not to a neighbouring concept."*
            #
            # Caught by `test_diagram_is_derived_from_the_concept_alone`, which requires
            # `unmeasured == 0`: the first version of this walk yielded `lemma:physicist` for
            # `pwn-e`, and a lemma has no evidence frame, so it came back unmeasurable. The test was
            # right and the widening was wrong.
            if _wn.is_lemma_id(other, store):
                continue
            seen.add(other)
            yield other


def superposition(store, source_id: str, *, label_keyed: bool = False,
                  include_unanchored: bool = False,
                  anchor_cache: Optional[Dict[str, Tuple[List[str], bool]]] = None
                  ) -> Dict[str, Any]:
    """Where a source concept lands, and what it adds — the partial-absorption verdict.

    `derive_diagram` gates on `one_object`, both bands absorbing the whole of the other's evidence,
    and treats everything else as separate. That reads the certificate as a yes/no when it already
    reports the residual, and it is why ConceptNet consolidates through this call instead:

        wn-cow.n.01 <-> cn-cow    measured, residual 0.9703, rows_a 5, rows_b 286

    Five taxonomy rows cannot absorb 286 associative ones. A taxonomy stub and a relational
    neighbourhood describe the same thing from different angles, which is the point of having more
    than one source. So the verdict here is three-way: absorbed fully (one artifact), absorbed
    partially (this — superimpose the shared core, keep the residual and add it), or not absorbed
    (separate).

    The residual itself picks the sense, so one mechanism answers the whole question. Measured
    against every sense sharing the anchor:

        cn-cow  ->  wn-cow.n.01  0.9703      cn-dog  ->  wn-oewn-02086723-n  0.9555
                    oewn-...-v   0.9930                  wn-dog.n.01         0.9575
                    wn-cow.n.02  0.9991                  wn-dog.n.03         0.9996
                    wn-cow.n.03  1.0000  <- zero shared  wn-frump.n.01       1.0000  <- zero

    A residual of exactly 1.0 is "no shared evidence" rather than a near miss, and the profile is
    graded.

    `prism.resolution` decides, and the minimum alone does not: taking the lowest residual would also
    merge `cow.n.01` into `cow.n.02`, since some sense is always nearest. The series of residuals
    over the sibling senses is a reading, cut by the same instrument every other reading in this
    system is cut by — the landing sense separates from its siblings. When it does not, the source is
    ambiguous here and nothing is merged, which is itself a measurement."""
    from prism.resolution import separated, partition
    cache = anchor_cache if anchor_cache is not None else {}
    # ── an artifact the lexicon never named has only its neighbours to be evidence ───────────────
    # `evidence()` features a row by the NEIGHBOUR'S anchors, and for a source with no anchors the
    # neighbours have none either — so the frame comes back empty, the pair is `measured=False`, and
    # a term with real relations refuses for want of a lexicon entry rather than for want of
    # evidence. Measured: `cn-nosode` reached 3 candidates once `candidates` was widened and
    # still refused, with an empty frame on both sides.
    #
    # `include_unanchored` already carries exactly what is left — `id:<neighbour>` — and is simply
    # off by default. Turning it on HERE, only where the source has no anchors, is not a second
    # policy: it is the same rule the widened reach follows, that an artifact's own relations place
    # it when the lexicon cannot. A caller that asked for it explicitly still gets what it asked for.
    if not include_unanchored and not anchors(store, source_id)[0]:
        include_unanchored = True
    ev_s = evidence(store, source_id, label_keyed=label_keyed,
                    include_unanchored=include_unanchored, anchor_cache=cache)
    reads: List[Dict[str, Any]] = []
    for cand in candidates(store, source_id, anchor_cache=cache):
        r = separation(store, cand, source_id, label_keyed=label_keyed,
                       include_unanchored=include_unanchored, ev_b=ev_s, anchor_cache=cache)
        if not r.get("measured"):
            continue
        rf = r.get("residual_fraction")
        if rf is None:
            continue
        reads.append({"candidate": cand, "residual_fraction": float(rf),
                      "one_object": bool(r.get("one_object")), "rows": r.get("rows_a")})
    out: Dict[str, Any] = {"source": source_id, "rows": len(ev_s.rows),
                           "candidates": len(reads), "reads": reads}
    # A candidate sharing no evidence is not a landing site — residual 1.0 is the computed null.
    shared = [x for x in reads if x["residual_fraction"] < 1.0]
    if not shared:
        out.update({"lands_on": None, "why": "no candidate shares any evidence with this source"})
        return out
    # Ordered ascending by residual, so the series `separated` reads is "how much better is the
    # best landing site than the rest".
    shared.sort(key=lambda x: x["residual_fraction"])
    series = [1.0 - x["residual_fraction"] for x in shared]      # shared-evidence fraction
    if len(series) > 1 and not separated(series):
        out.update({"lands_on": None,
                    "why": "the candidates do not separate — this source is ambiguous here"})
        return out
    k = partition(series)[0] if len(series) > 1 else 1
    out.update({"lands_on": shared[0]["candidate"],
                "residual_fraction": shared[0]["residual_fraction"],
                "resolved": int(k),
                "why": "superimpose the shared core; the residual is kept and added"})
    return out


def derive_diagram(store, aid: str, *, label_keyed: bool, include_unanchored: bool,
                   anchor_cache: Optional[Dict[str, Tuple[List[str], bool]]] = None
                   ) -> Dict[str, Any]:
    """The diagram of a concept — the artifacts the evidence cannot separate from it. The concept id
    is the only input.

    Reaches the candidates through shared anchor vertices, then measures each one. Returns the member
    ids (including `aid`), every candidate's separation read, and what could not be measured. A
    concept whose evidence separates it from everything comes back a diagram of one, which is a
    correct and common answer and is reported as such."""
    cache = anchor_cache if anchor_cache is not None else {}
    ev_a = evidence(store, aid, label_keyed=label_keyed,
                    include_unanchored=include_unanchored, anchor_cache=cache)
    members: List[str] = [aid]
    reads: List[Dict[str, Any]] = []
    unmeasured = 0
    n_cand = 0
    for c in candidates(store, aid, anchor_cache=cache):
        n_cand += 1
        r = separation(store, aid, c, label_keyed=label_keyed,
                       include_unanchored=include_unanchored, ev_a=ev_a, anchor_cache=cache)
        reads.append(r)
        if not r.get("measured"):
            unmeasured += 1
        elif r.get("one_object"):
            members.append(c)
    return {"concept": aid, "members": members, "member_count": len(members),
            "candidates": n_cand, "unmeasured": unmeasured,
            "label_keyed": bool(label_keyed), "include_unanchored": bool(include_unanchored),
            "reads": reads, "exhaustive": bool(ev_a.exhaustive)}
