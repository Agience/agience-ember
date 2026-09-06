"""The projection: the read taken of the ordered frame that every cut and every condensation sits on.

The screen and the projection are two things. The screen is the ordered `(T, F)` frame signals are
placed on; the projection is what the instrument returns when it reads that frame. This module builds
a screen (`frame`) and hands back projection reads (`read`, `resolvable`, `coherent`). Every one of
those reads passes `with_screen=False`: it declines the entropy-folded `Screen`, which is what
`ember/optics.py` exists to route around.

The frame is the candidates that were reached, in the order they were reached, each carrying its own
coordinate — exactly the `(T, F)` the optics contract asks for, axis 0 ordered and axis 1 feature:

    T   the reach ordering. Reached-ness is the ordered axis, and it carries information:
        coherence is a lag-1 statistic, so a shuffled screen collapses toward the null
        (11.32 -> 0.42). Reversal is exactly invariant; shuffling is not.
    F   `crystal.ontology.geometry.dense_vec` — the signed feature-hash of the JC coordinate into
        D=2048, whose inner products preserve the metric to 1e-15.

The coordinate is dense, and that is what makes the frame readable. A sparse one-hot frame reads as
pure noise (phi_F 0.94, k_signal 0, coherence 0.10); `dense_vec` is the hashed projection of that
sparse coordinate into a dense basis an instrument can read.

Build the screen once per read and hand it to everything: one screen, one instrument, every cut.

The feature axis comes from the corpus, not from the screen
-----------------------------------------------------------
Read in its own occupied columns, a query screen leaves `F` growing with `T`. Sweeping the screen
size on one query ("cats and dogs", 208 reached):

    T      F      k_signal   phi_F   margin_last   margin_next
    8      21     1          0.162   2.4453        -5.7224
    32     45     1          0.436   5.7536        -2.6018
    60     79     2          0.498   0.4754        -1.7262
    100    161    5          0.399   0.3138        -0.1995
    208    439    11         0.314   0.0873        -0.0402

`k_signal` grows monotonically because `F` grows with `T`: every additional row brings ~2 new
occupied columns, so the frame stays near-square however many rows arrive, and a near-square
correlation matrix is where the spectral read stops being trustworthy. The instrument reports that
state — the marginal mode sits 5.75 above the floor at T=32 and 0.0873 at T=208, with the first
unresolved mode at -0.0402, and the certified interval [0, 439] certifies nothing. Small magnitudes
either side of the floor are a threshold artefact rather than a measurement.

The cause is the coordinate, not the instrument: a synset occupies ~10 of 2048 hashed dimensions and
different synsets occupy different ones, so the union of occupied columns grows linearly with the
row count. Dropping empty columns makes the read possible; it cannot make the frame tall.

`build_basis` therefore derives the directions from the corpus itself, so `F` does not depend on
how many rows a query reached. The width of that basis stays a measurement rather than a constant:
`geom.corpus-basis` measures `k = 280` with a certified interval `[123, 956]` and `k_certain=False`,
which says the evidence admits every resolution in that band. Because the directions come out
ordered by eigenvalue, `B[:k]` is a basis for every `k <= k_stored`, so one stored generation
contains every zoom in the band. `frame` reads at the corpus basis's stored width unless a caller
states a `k`, clamps a stated `k` into the generation's certified band, and reports the zoom it used
as `read()["basis_k"]`.

Two frames at different `k` are in different coordinates and must not couple, pool or compare
(PAPER §5.5: coupling raises without a shared basis). The `k` is therefore part of the coordinate
token `pooling.coordinate_id` builds, not a detail of the frame.
"""
from __future__ import annotations

from ember.authorship import DEFAULT_AUTHOR

from typing import Any, Dict, List, Optional, Sequence, Tuple


# The corpus basis: a fixed set of directions the whole lattice occupies, so a query screen is
# projected onto axes that do not depend on how many rows the query happened to reach. Stored as an
# artifact so it is derived reproducibly and replicates like everything else.
#
# This is a lineage root, not an artifact id. A basis generation is a thing that was measured
# ([[everything-is-an-artifact]]), so each derivation is its own row and `head_of(BASIS_ROOT)`
# resolves which one is live. Handing the root straight to `put_artifact` would instead overwrite
# the previous generation in place: a rebuild that moves `k` then leaves every Screen that pooled
# planes at the old width unable to pool, `accumulated()` reporting a stale pool, and no artifact
# anywhere recording what the old basis was.
BASIS_ROOT = "geom.corpus-basis"
BASIS_CONTENT_TYPE = "application/vnd.agience.corpus-basis+json"


def _basis_from_cloud(M):
    """From an `(n, D)` coordinate cloud, the corpus's own basis `(k_stored, D)`, the instrument's
    point-estimate count, and the read behind it — `(B, k, rd, source)`.

    The matrix is stored out to `k_hi` rather than to the point estimate, wherever the decomposition
    is this function's own to widen. The instrument reports a certified interval (`rd.k_lo … rd.k_hi`)
    and, on this corpus, `k_certain=False`: `k=280` inside `[123, 956]`, with 280 well off the centre
    of it (157 below, 676 above). Because the directions come out ordered by eigenvalue, `B[:k]` is a
    valid basis for every `k <= k_stored`, so storing the wider matrix costs nothing and keeps the
    point estimate recoverable as a prefix, while storing the narrow one discards that freedom
    irreversibly.

    On the instrument path the matrix is the point estimate, and stays that width.
    `ember.optics.principal_directions` returns exactly `resolved_modes` columns (entroptics computes
    the count and slices `evecs[:, order[:k]]` in one call), so that is the widest matrix reachable
    through the one sanctioned instrument. Widening it would take either a second eigendecomposition
    here — a private re-implementation of the instrument, which `ember/optics.py` exists to prevent
    ([[one-instrument-enforced]]), and which would also break the property below that the count and the
    directions are one read — or a `k`/floor argument on `principal_directions`, which belongs to the
    instrument module. That is a flagged seam, not taken here. The SVD fallback is a decomposition this
    function owns outright, so it is widened.

    Consequence for a caller: on the instrument path `len(body["basis"])` equals `body["k"]`; on the
    fallback path it equals `k_certified_hi`. Those are two different measurements — how many modes
    the instrument resolved, and how many the evidence admits — and `k` is the count, not the row count
    of the matrix.

    The count and the directions are one read. `k` is the instrument's resolved-mode count and `B` is
    the subspace those modes span, both read off the same correlation spectrum through
    `ember.optics.principal_directions`, so the basis is the directions of the modes that were
    actually counted.

    A hashed coordinate leaves most of D structurally empty, so the columns no row occupies are
    dropped before the read — measure the basis the corpus is actually in, not 2,000 empty channels —
    and the resulting directions are scattered back to full width D so the basis still projects a
    full-D screen. Falls back to the covariance SVD only if the instrument returns nothing (too few
    rows, no resolved mode), so the artifact is always derivable."""
    import numpy as np
    from ember.optics import read_ordered, principal_directions
    D = int(M.shape[1])
    live = np.nonzero(M.any(axis=0))[0]                 # the columns some row actually occupies
    Mr = M[:, live] if 0 < live.size < D else M
    rd = read_ordered(Mr, with_screen=False)
    Br = principal_directions(Mr)                        # (len(Mr_cols), k) correlation eigenbasis
    if Br is not None and Br.ndim == 2 and Br.shape[1] >= 1 and Br.shape[0] == Mr.shape[1]:
        k = int(Br.shape[1])
        B = np.zeros((k, D), dtype=float)
        B[:, live] = Br.T                               # live-column directions back into full D
        return B, k, rd, "principal_directions"
    # Fallback: nothing resolved (or the instrument could not read) — the covariance SVD, so a basis
    # is always produced rather than the consolidation job failing.
    k = int(rd.k_signal) if rd.k_signal and rd.k_signal >= 2 else int(rd.k_hi or 2)
    _u, _sv, vt = np.linalg.svd(Mr, full_matrices=False)
    # Out to `k_hi`. `vt` is ordered by singular value, so every prefix is a basis and the point
    # estimate `k` remains exactly `B[:k]`. This decomposition is this function's own, so widening it
    # stays one reading of the instrument — see the docstring for why the instrument path holds its
    # width.
    k_stored = max(k, int(rd.k_hi or 0))
    k_stored = int(min(k_stored, vt.shape[0]))            # never more directions than exist
    B = np.zeros((k_stored, D), dtype=float)
    B[:, live] = vt[:k_stored]
    return B, k, rd, "svd-fallback"


def build_basis(store):
    """Derive the corpus basis: the directions the noun coordinate occupies, and how many of them,
    both read from the corpus itself. A consolidation-phase job — it needs the corpus present, and
    runs against a corpus that has finished being written.

    The derivation is deterministic and takes no seed. The coordinate cloud is the corpus in
    `crystal.ontology.driver` order and the directions come from `principal_directions` or a
    covariance SVD, which is what makes a basis generation content-addressable and identical across
    observers. The lineage in `_write_generation` rests on two derivations of the same corpus
    agreeing byte for byte.

    `k` is measured rather than typed. How many directions the corpus occupies is read by presenting
    the coordinate cloud tall — all ~85k nouns against D=2048, so T/F ~ 41 and the read is
    well-conditioned — and asking the instrument how many modes rise above its own floor.

    The measurement's own verdict is that there is no clean rank: the full cloud reads
    `k_signal = 202` with a certified band `[73, 1008]` and a marginal-mode gap of 0.0008. The
    feature hash spreads the coordinate nearly evenly, so the singular spectrum decays smoothly with
    no valley. `k` is the derived `k_signal` and the read carries `k_certain=False`, which is the
    true state.

    The interval is the zoom range every read is bounded by, not decoration on the artifact.
    `k_certified_lo`/`k_certified_hi` are the admissible band `_zoom` clamps a caller-stated `k`
    into, so the point estimate is one reading inside the interval rather than the only one that
    exists. See `_zoom`."""
    import numpy as np
    from crystal.ontology import geometry as g
    from crystal.ontology import driver as wn
    from prism import vector as _vec
    ic = g.load_ic()
    rows = []
    for n in (x.name() for x in wn.all_synsets(pos=wn.NOUN)):
        try:
            v = g.dense_vec(wn.synset(n), ic)
        except Exception:
            continue
        nv = _vec.norm(v)
        if nv:
            rows.append(v / nv)
    if len(rows) < 2:
        return None
    M = np.vstack(rows)
    B, k, rd, _src = _basis_from_cloud(M)
    from ember.lattice_mint import _author_ref
    from prism.grounding import P_OBSERVED
    body = {
        "content_type": BASIS_CONTENT_TYPE,
        "title": "corpus basis", "k": int(k), "d": int(B.shape[1]),
        "k_certified_lo": int(rd.k_lo), "k_certified_hi": int(rd.k_hi),
        "k_certain": bool(rd.k_lo == rd.k_hi == rd.k_signal),
        "basis": [[float(x) for x in row] for row in B],
        "collection_id": "ontology", "provenance": P_OBSERVED,
        "via": "op.consolidate.basis",
    }
    _write_generation(store, body, author=_author_ref(store, DEFAULT_AUTHOR))
    return B


def _write_generation(store, body: Dict[str, Any], *, author: str):
    """Write one basis generation into the `BASIS_ROOT` lineage and return the stored doc.

    The version id is mantle's. `LatticeArtifactStore.revise` derives
    `"<root>~<blake2b(canonical_json(doc minus id and store bookkeeping), 8)>"`, archives the prior
    head, and shares `root_id`, so `head_of(BASIS_ROOT)` keeps resolving the live basis and every
    earlier width stays queryable. Deriving an id here would be a second implementation of identity
    that can drift from the one the rest of the lattice versions by; mantle's has the two properties
    this needs — deterministic across observers, and idempotent, so identical content re-derives the
    same id and writes nothing new.

    A revision carries no fresh `created_time`, which is the content-addressing working.
    `created_time` sits inside mantle's canonical fingerprint, so stamping `_now()` on every
    derivation would mint a new generation for a basis byte-identical to the live one, and every
    Screen pooled under the old id would start counting `dropped:basis` on a coordinate that never
    moved. When a generation was written is the store's own `(_origin, _seq)`; what it is, is the id.

    `(_origin, _seq)` orders the lineage newest-first rather than oldest-first as `versions_of`
    describes: `revise` writes the new version before archiving the old — deliberately, so a crash
    between the two leaves two committed heads rather than zero — so the archived prior is re-stamped
    with the later `_seq`. That is mantle's to reconcile; `superseded_by` is the link that says which
    came after which.

    Versioning is a requirement on the store, not a fallback. On a store without `head_of`/`revise`
    the derivation raises, because `put_artifact(id=BASIS_ROOT)` would succeed while destroying the
    previous generation and leaving every pooled Screen stranded with no error anywhere. Raising
    makes the consolidation job fail on a store that cannot hold a lineage, which is a state an
    operator can see.

    On a store where no basis has ever been derived, the first generation is written under the root
    id itself, following mantle's lineage rule (`vertex.py:146`) that the first version of an
    artifact has `id == root_id`. Generation 1 is therefore identified by the lineage it opens and
    generation 2 onwards by their content; the move happens on the next derivation, with no
    migration."""
    arts = getattr(store, "artifacts", None) or store
    head_of = getattr(arts, "head_of", None)
    revise = getattr(arts, "revise", None)
    if head_of is None or revise is None:
        raise TypeError(
            "%s cannot version an artifact (no head_of/revise), so a basis derivation here would "
            "OVERWRITE the live generation in place under %r — the 2026-08-01 02:13:33Z defect: "
            "k moved 195->280, every pooled Screen silently stopped pooling, and the 195 basis no "
            "longer existed to compare against. Derive the basis on a lineage-capable store."
            % (type(arts).__name__, BASIS_ROOT))
    head = head_of(BASIS_ROOT)
    # The same basis is the same generation, whatever id it currently sits under. `revise` is
    # idempotent against its own derived ids, but the first generation is written under the root id
    # (see below), so a second byte-identical derivation would otherwise fork it into `root~hash` —
    # a new coordinate token for a coordinate that did not move, and every Screen pooled under the
    # old one counting `dropped:basis` for nothing. The content answers the question directly.
    if head is not None and all(head.get(k) == v for k, v in body.items()):
        return head
    if head is None:
        from prism.grounding import _now
        doc = dict(body)
        doc.update({"id": BASIS_ROOT, "state": "committed", "created_by": author,
                    "created_time": _now()})
        return arts.put_artifact(doc)
    return revise(head["id"], dict(body), author=author)


#: `id(store) -> (the store's write mark, the lineage's stamp, the live generation's id, B, band)`.
#: The two marks are the point: see `_live_generation`. All five live in one tuple rather than in
#: parallel dicts, because dicts that have to be kept in step are the shape that goes stale.
_BASIS_CACHE: Dict[int, Any] = {}


def _live_generation(store):
    """`(generation_id, B, band)` — which basis generation is live, the `(k_stored, D)` matrix it
    holds, and the certified zoom band `(k_certified_lo, k_certified_hi)` it was measured with.
    `(None, None, None)` when the store has no basis, or cannot say which one is live.

    `band` is `None` when the generation states none, and an unstated band stays unstated. A
    generation that records no interval says nothing about which resolutions its evidence admits, so
    `_zoom` takes no zoom on it and the frame comes back at the full stored width. Supplying a
    default band would fabricate a certification nobody measured
    ([[absence-is-not-an-affirmative-claim]]).

    The cache is keyed by `id(store)`, which identifies which store was read and says nothing about
    when: the store object outlives any number of writes to it. A basis rebuilt underneath a running
    process — `k` moving 195 -> 280, say — would therefore keep being projected onto the old
    directions until the service restarted, and restarting the service is the operator procedure
    PLAN §0.0 rules out.

    The discriminator is the lineage rather than one row. A derivation writes a new row and archives
    the old, so any single row's `(_origin, _seq)` can stay put across a rebuild and a gate watching
    it would verify clean for ever ([[verification-that-cannot-fail]]). `freshness.set_stamp` asks
    the true question — has anything of `BASIS_CONTENT_TYPE` been written? — as `(rows, max _seq)`,
    which moves on a new generation, on an archive, and on a deletion. Nothing else in the corpus
    carries this content type, so it is exact.

    It sits behind the cheap gate, the two-level shape `crystal.ontology.driver._gate` establishes.
    Measured on a live 5.7 GB lattice, 200 warm samples each: `write_mark` 15.9 µs · `set_stamp`
    31.2 µs (230 µs on the first, cold call) · resolving `head_of` and `np.asarray`-ing a live
    (280, 2048) generation 316–392 ms. An unchanged write mark means nothing was written at all, so
    this derivation is still the derivation of what is in the store, with no further read of any
    kind. End to end a warm `load_basis` is 15.7 µs against 5.2 µs for a single-row stamp — ~10 µs
    more per screen, against a 42 ms frame build, to gate on the question that is true.

    A `None` mark means the store cannot say, and that is not an answer to cache against. The cheap
    arm requires a mark that is reportable and did match, so a store answering `None` falls through
    to the discriminator rather than comparing `None == None` and serving for ever. With no stamp
    either it caches nothing and re-reads, which is slower and honest
    ([[absence-is-not-an-affirmative-claim]]). `(None, None)` is cached normally, though: "this
    corpus has no basis yet" is a real answer, verified against the same stamp as any other, and it
    is what `frame()` falls back on.

    The cost of `head_of` grows with the number of generations: it hydrates every version in the
    lineage, and a generation is an ~11 MB inline matrix. Measured at one generation: 327 ms against
    293 ms for a bare `get_artifact`, so the cost is the matrix rather than the lineage walk. It is
    paid only when the gate has moved, i.e. once per derivation. Storing the matrix as content (a
    `content_ref` into the CAS) would make both reads cheap, and is a flagged seam not taken here."""
    from crystal.ontology import freshness as _freshness
    # The stamp's cap is the fetch's own cap, read from its one source rather than typed here.
    # `freshness.set_stamp` requires a cap and answers `None` without one, because a mark over a
    # prefix would verify clean for ever and the extent it covers is not something it can infer.
    # Without the cap every `load_basis` gets `st = None`, drops the entry (`_BASIS_CACHE.pop`) and
    # re-hydrates the ~11 MB generation from scratch — a cache that reads as present while doing
    # nothing. It is the same cap `match._offers` and `mantle.db.typed_fetch` take, imported
    # rather than restated so the three stay in step; a lineage longer than the cap reports
    # non-exhaustive and degrades to no cache, which is slower and honest.
    from mantle.db.constants import CT_FETCH_CAP as _CT_FETCH_CAP
    key = id(store)
    hit = _BASIS_CACHE.get(key)
    mk = _freshness.write_mark(store)
    if hit is not None and mk is not None and hit[0] == mk:
        return hit[2], hit[3], hit[4]              # nothing was written to this store at all
    st = _freshness.set_stamp(store, BASIS_CONTENT_TYPE, cap=_CT_FETCH_CAP)
    if hit is not None and st is not None and hit[1] == st:
        # the store moved, but no basis did — re-verified, so carry the value and the new mark
        _BASIS_CACHE[key] = (mk, st, hit[2], hit[3], hit[4])
        return hit[2], hit[3], hit[4]
    gid, B, band = None, None, None
    try:
        import numpy as np
        arts = getattr(store, "artifacts", None) or store
        # `head_of`, not `get_artifact(BASIS_ROOT)`, with no fallback to the root id. A store that
        # cannot resolve a lineage cannot tell a generation from a handle: the root id answers with
        # whatever was last written over it and holds still when the content changes, so
        # `basis_head` would hand out a stable token for a coordinate that had been replaced. Such a
        # store reports no basis.
        doc = arts.head_of(BASIS_ROOT)
        if doc:
            gid = doc.get("id") or None
            if doc.get("basis"):
                B = np.asarray(doc["basis"], dtype=np.float64)
            band = _band_of(doc)
    except Exception:
        gid, B, band = None, None, None
    if st is not None:
        _BASIS_CACHE[key] = (mk, st, gid, B, band)
    else:
        _BASIS_CACHE.pop(key, None)
    return gid, B, band


def _band_of(doc) -> Optional[Tuple[int, int]]:
    """`(k_certified_lo, k_certified_hi)` off a basis generation, or None when it does not state one.

    Both ends or neither. A generation carrying only one end of the interval has not stated an
    interval, and deriving the other end from `k` or from the stored width would manufacture a
    certification nobody measured — the `.get("grounded", True)` shape applied to a zoom range.
    Ends that do not bracket a usable width (`hi < lo`, or `lo < 2`, which no instrument can read)
    likewise read as no band, because a band that cannot be entered is not a band."""
    try:
        lo = doc.get("k_certified_lo")
        hi = doc.get("k_certified_hi")
    except Exception:
        return None
    if lo is None or hi is None:
        return None
    try:
        lo, hi = int(lo), int(hi)
    except (TypeError, ValueError):
        return None
    if lo < 2 or hi < lo:
        return None
    return lo, hi


def load_basis(store):
    """The corpus basis `(k_stored, D)` at its full stored width, or None if it has not been derived
    yet.

    This is the widest admissible zoom, rather than the zoom any particular read takes. `frame`
    truncates it per read (see `_zoom`); a caller that projects with this matrix directly is reading
    at the stored width and should say so, because that is a different coordinate from one `frame`
    took a zoom on — `pooling.coordinate_id(store, corpus_basis=True, k=…)` is how it says which."""
    return _live_generation(store)[1]


def basis_band(store) -> Optional[Tuple[int, int]]:
    """The admissible zoom range this store's live basis was measured with —
    `(k_certified_lo, k_certified_hi)` — or `None` when there is no basis or it does not state one.

    The interval is the degrees of freedom the read has. A corpus measuring `k = 280` with
    `k_certified_lo = 123`, `k_certified_hi = 956` and `k_certain = False` is saying that the
    evidence admits every resolution from 123 to 956 — 833 of them — with 280 well off the centre
    (157 below, 676 above), rather than that the answer is 280 with some uncertainty around it.

    The two ends mean different things. Below `k_lo` a truncation discards directions the corpus
    certified carry structure; above `k_hi` it admits directions the corpus certified are noise.
    Both are the instrument's own statement about its own evidence rather than tuning knobs, which
    is why a caller-stated `k` can be bounded by them with no typed constant appearing anywhere.
    `_zoom` bounds a stated `k` from above by `k_hi`; it leaves a smaller `k` where it is, because
    raising every read to the band's floor puts every query at the same number."""
    return _live_generation(store)[2]


def basis_head(store) -> Optional[str]:
    """Which basis generation this store is projecting onto — the live artifact's id, or `None`
    when there is no basis or the store cannot resolve the lineage.

    The id is the identity, which is what content-addressing it buys. A store-local
    `(_origin, _seq)` for the basis row is a proxy that does not travel: two nodes holding the same
    generation allocate different `_seq`, so a replica's plane would count as a `dropped:basis`
    against a coordinate that is byte-identical. A generation is its own artifact with an id derived
    from its content, so every observer names the same basis the same way and the token needs no
    side-car ([[everything-is-an-artifact]]).

    `None` means unrecorded, and a store with no basis is a real and common state.
    `frame(corpus_basis=True)` then falls back to dropping this screen's empty columns, which is a
    different feature axis for every screen and so is not a coordinate anyone can name."""
    return _live_generation(store)[0]


def _zoom(P, band: Optional[Tuple[int, int]], *, k: Optional[int] = None) -> int:
    """How many of the corpus directions this frame is read in. `P` is the frame already projected
    onto the full stored basis, so the answer is a column count on `P` and `P[:, :k]` is the frame at
    that zoom.

    The rule in three lines: an unstated `k` is the full stored width, a stated `k` is bounded above
    by the generation's certified `k_hi` and by the width itself, and a generation with no band takes
    no zoom. The bounds are the artifact's own measurement (`basis_band`) and nothing here is typed.

    The inbound frame takes no zoom of its own, and that is a measured result rather than an
    omission. Reading the inbound frame at its own `ember.optics.resolvable(P)` count degrades the
    answer — same frame, three zooms, asking which concept dominates the top mode:

        what is a dog     k=280 -> dog.n.01     k=123 -> dog.n.01     k=4 -> canine.n.02
        what is water     k=280 -> water.n.01   k=123 -> water.n.01   k=4 -> binary_compound.n.01
        capital of france k=280 -> paris.n.01   k=123 -> paris.n.01   k=2 -> k_signal 0, nothing

    The lead falls to its own generic ancestor, and the answers follow it: canine's gloss for "what
    is a dog", binary compound's for "what is water", seven glosses about capitals and regions in
    general for "capital of france".

    The cause is structural. The top-variance directions of a dog-field are the ancestor chain —
    canine, carnivore, mammal — because that is what the field's members share, and `dog` is
    distinguished from `canine` by directions further down. Truncating to the few modes that rise
    above the noise floor keeps exactly the part every member agrees on and discards the part that
    discriminates. A count of resolved modes is not a resolution for a read that has to tell things
    apart. Raising a small count to `k_lo` instead has the shape of a typed constant with a different
    literal: `resolvable` measures 4 modes on "what is a dog", `k_lo` is 123, and every conversational
    query then comes out at 123.

    So the zoom belongs on the outbound projection. Inbound, the corpus basis at its stored width is
    the corpus's resolution — that is what a corpus basis is, and reading through all of it is not a
    chosen zoom. Outbound is `activation._stated_relations` step 5, which screens the absorbed frame
    through `prism.resolution.signal_end`; that is where "it returns too much" is answered. `k` here
    is caller-stated only, and unstated means the corpus's own width.

    An unreadable frame takes no zoom, and a generation that states no band takes none either: with
    no admissible range there is no zoom to take and the frame comes back at the full stored width.
    `k_signal == 0` is a different case — "nothing rose above the floor" is a measurement, and a
    stated `k` clamps against it like any other.

    A caller-stated `k` is bounded by the same band, because a resolution outside the band is not one
    the evidence admits, whoever asks for it."""
    width = int(P.shape[1])
    if k is None:
        return width                                  # inbound: the corpus's own resolution
    if band is None:
        return width                                  # no admissible range stated: no zoom to take
    return max(1, min(min(int(band[1]), width), int(k)))


def _build(store, names: Sequence[str], *, center=None, energy=None,
           corpus_basis: bool = True, k: Optional[int] = None):
    """`(W, kept, zoom)` — the frame, the correspondence, and which zoom it is in.

    One builder, so the zoom stays tied to the frame it describes. `zoom` is the number of
    corpus-basis directions the frame occupies, and it is `None` — rather than the width — whenever
    the corpus basis was not the coordinate: the raw dense arm, and the no-basis fallback that drops
    this screen's own empty columns. Both of those widths are numbers that are not zooms, and a zoom
    is only what it says it is ([[state-what-it-is]]). `frame_rows`, `frame` and `read` are all this
    function with different parts of the answer taken.

    See `frame_rows` for the correspondence contract and `frame` for the coordinate fork."""
    try:
        import numpy as np
        from crystal.ontology import geometry as g
        from crystal.ontology import driver as wn
        from ember.optics import MIN_ROWS
    except Exception:
        return None
    ic = g.load_ic()
    rows: List[Any] = []
    amps: List[float] = []
    kept: List[int] = []                     # which input each row is — see `frame_rows`
    e_in = list(energy) if energy is not None else None
    for idx, n in enumerate(names):
        try:
            s = wn.synset(n)
        except Exception:
            s = None
        if s is None:
            continue
        try:
            _v = g.dense_vec(s, ic, center=center)
        except Exception:
            continue
        # A concept with no coordinate arrives as a row of zeros, and this is where it leaves the
        # frame. `dense_vec` returns an all-zero vector for an unplaced synset rather than raising,
        # so the `except: continue` above covers only a failed lookup; the emptiness has to be
        # tested for directly.
        #
        # On a live corpus every `oewn-*` synset the seeder fires has no geometry at all:
        #
        #     "what is a cat?"   16 seeds -> 8 measured, 8 all-zero, 0 raised, 0 skipped
        #                        frame T=16, rows carrying signal=8
        #     "capital of france" 2 seeds -> 1 measured, 1 all-zero
        #
        # `T` is the sample count the certified band is computed from (`band ~ sqrt(F/T) + F/T`), so
        # phantom rows buy certification with evidence that does not exist. Same corpus, same query,
        # the two readings:
        #
        #     claimed T=16 -> band 15.68        honest T=8 -> band 29.31
        #
        # The honest number is the wider one, which is the point: counting the empty rows reports the
        # instrument as nearly twice as certain as its evidence supports. An absent coordinate is an
        # absence rather than a measured zero — the rule behind `.get("grounded", True)`, applied to
        # geometry. Dropping the row loses no information; the information was never there.
        if not np.asarray(_v).any():
            continue
        rows.append(_v)
        kept.append(idx)
        # Amplitude is sqrt(energy). Collected alongside the row, so a synset that failed to resolve
        # leaves the alignment between a row and its energy intact.
        if e_in is not None:
            try:
                amps.append(float(e_in[idx]))
            except Exception:
                amps.append(0.0)
    if len(rows) < MIN_ROWS:
        return None
    W = np.vstack(rows)
    if e_in is not None and len(amps) == W.shape[0]:
        a = np.asarray(amps, dtype=float)
        a[~np.isfinite(a)] = 0.0
        a[a < 0.0] = 0.0                     # energy is non-negative; a negative value is a defect
        W = W * np.sqrt(a)[:, None]
        if not W.any():
            return None                      # every row had zero energy: nothing was carried
    if not W.any():
        return None
    if not corpus_basis:
        # The membrane's coordinate: the raw dense space, columns intact, because that is where
        # `match.tekton_basis` builds its `(F, k)` coupling bases and `absorb_transmit` requires a
        # basis whose first axis equals `F`. Dropping empty columns (below) or projecting onto the
        # corpus basis would both misalign it. See the docstring for the fork.
        return W, kept, None                 # no zoom: this is the raw dense coordinate
    # The feature axis is the basis this screen occupies. Read in the raw hash width, every query
    # reads `k_signal = 0` with `phi_F` between 0.004 and 0.028 — a (60, 2048) frame where the screen
    # fills 2% of the basis. `dense_vec` hashes a ~10-node path into D=2048, so 99% of every column
    # is structurally zero, and `F >> T` is the regime `ember.optics` calls vacuous by construction.
    #
    # A column that is zero for every row on this screen carries no information about this screen.
    # Dropping it removes dimensions that hold no data, so the instrument measures the basis the
    # evidence is in; the coordinate is unchanged, and the instrument stops being asked about 2,000
    # empty channels.
    #
    # When a corpus basis exists, project onto it: a fixed feature axis, so F stops growing with T.
    # `W @ B.T` is (T, k) — the screen expressed in the directions the whole corpus occupies, rather
    # than the ad-hoc union of whatever these rows happened to touch.
    #
    # The basis rows are ordered by eigenvalue, so `B[:k]` is a basis for every `k <= k_stored` and
    # `P[:, :k]` is `W @ B[:k].T`: the truncation is a slice of a projection already taken rather
    # than a second projection, so a zoom costs no linear algebra. See `_zoom` for what sets `k` and
    # what bounds it.
    #
    # `load_basis` / `basis_band` rather than `_live_generation` — one seam, two questions. Both
    # resolve through the same cached generation (the second call is a cache hit), and going through
    # the public readers keeps one substitution point a caller can stand a basis in at.
    B = load_basis(store)
    band = basis_band(store)
    if B is not None and B.ndim == 2 and B.shape[1] == W.shape[1]:
        P = W @ B.T
        if not P.any():
            return None
        z = _zoom(P, band, k=k)
        Pk = P[:, :z]
        # A truncation that empties the frame has deleted it rather than zoomed it, and a frame whose
        # every row is zero is not a screen anyone can read. The absence is what comes back.
        return (Pk, kept, z) if Pk.shape[1] >= 2 and Pk.any() else None
    # No basis yet: fall back to dropping empty columns, which makes the read possible while leaving
    # F growing with T. The read still happens; it is the near-square regime the margins warn about.
    occupied = np.flatnonzero(np.any(W != 0.0, axis=0))
    if occupied.size < 2:
        return None
    # This width is how many columns these rows happened to touch — a different feature axis for
    # every screen — so the zoom is reported as absent rather than as this number.
    return W[:, occupied], kept, None


def frame_rows(store, names: Sequence[str], *, center=None, energy=None,
               corpus_basis: bool = True, k: Optional[int] = None):
    """`(W, kept)` — the `(T, F)` screen and which of `names` its rows are, in row order.

    Rows are dropped: a name with no resolvable synset, and a name whose coordinate is absent (see
    `_build`). `kept[i]` is the index into `names` of row `i`, so `W[i]` and `names[kept[i]]` are the
    same object by construction and stay aligned. That correspondence is what makes a per-row read a
    measurement of something rather than of an anonymous row.

    The drop rate is large enough to matter — `T` and `len(names)` differ on roughly half of live
    queries:

        "what is a dog"    66 live concepts -> 64 rows    "what is a bird"   58 -> 56
        "what is a tree"   43 live concepts -> 41 rows    "what is a river"  12 -> 10

    A caller holding only the matrix has to detect the mismatch and discard the whole read, falling
    back to cutting a bare salience column, which `prism.resolution` states is not a frame and does
    not decide. `frame_rows` is the entry point that removes that fallback.

    It does not report the zoom. `F` is the zoom on the corpus arm, but only `read` says so in words;
    a caller that needs to name the coordinate it built — to pool a plane, or to compare with a peer
    — passes that width to `pooling.coordinate_id(store, corpus_basis=True, k=W.shape[1])`, and
    `PooledScreen.place` completes it from the plane itself when the caller does not.

    See `frame` for the full contract — this is that function, with the correspondence returned."""
    got = _build(store, names, center=center, energy=energy, corpus_basis=corpus_basis, k=k)
    return None if got is None else (got[0], got[1])


def frame(store, names: Sequence[str], *, center=None, energy=None, corpus_basis: bool = True,
          k: Optional[int] = None):
    """The `(T, F)` screen for these synsets, in the given order.

    `k` — which zoom. `None`, the default and what most callers want, reads at the corpus basis's
    own stored width. Stating `k` materialises the same concepts at a resolution the caller names,
    bounded by the generation's certified band, because a resolution outside the band is not one the
    evidence admits. Ignored when `corpus_basis=False`, where the coordinate is the raw dense one
    and has no zoom to take. See `_zoom`.

    `corpus_basis` — which feature axis the frame comes back in. This is a fork between two real
    coordinate systems, and the caller makes it because only the caller knows whether the frame is
    going to an instrument or to a membrane. Both arms cost something, measured over nine
    conversational queries on a live corpus (the ruling in PLAN §1.5 is membrane -> False,
    instrument -> True):

        True   the screen is projected onto the stored corpus basis, so `F` is the fixed number of
               corpus directions rather than the raw hash width D=2048. `B` is orthonormal to
               2.3e-15 and spans 280 of 2048 dimensions, so `W @ B.T` keeps only the in-span energy:
               retained energy per query 0.449 – 0.620, mean 0.542, i.e. ~45.8% of a conversation
               frame's energy is dropped at this line. It buys the only read there is — on the same
               queries the instrument resolved `k_signal` 2–5 projected and 0 raw, every time, because
               `(T≈14–81, F=2048)` is the vacuous `F >> T` regime.
        False  the screen stays in the raw dense coordinate the coupling bases live in, with all D
               columns kept, which is what a caller placing a signal for a tekton to absorb needs
               (`ember/signal/state.py`). `ontology.match.tekton_basis` stacks unprojected
               `dense_vec` into an `(F=2048, k)` coupling basis and `ember.optics.absorb_transmit`
               requires `basis.shape[0] == frame.shape[1]`, so a projected `(T, 280)` frame comes
               back as None — "the frame carried no read" — which is indistinguishable from a
               genuine non-coupling. No energy is lost on this arm.

    Empty columns are dropped only in the no-basis fallback: with `corpus_basis=False` a coupling
    basis is indexed by the raw dimension, so dropping columns would misalign it. Each caller states
    the arm rather than inheriting it, which is why `read`, `resolvable` and `coherent` take the same
    parameter and `activation.compose` passes it explicitly.

    `energy` — the per-row activation energy, in the same order as `names`. Rows are scaled by
    `sqrt(energy)`, because a beam's amplitude is the square root of its energy; the frame is then
    the beam the propagation actually produced, and `‖row‖²` is that row's energy.

    Without it the beam carries coordinates and no energy, which inverts the answer.
    `crystal.ontology.geometry.dense_vec` returns a unit-norm coordinate, so a concept the propagator
    fired at 43 and one it fired at 1.0 arrive as rows of the same magnitude, and the instrument then
    reports whatever varies most. For an ontology screen that is the unrelated senses, because a
    taxonomy is highly correlated by construction — one shared hypernym path, low variance — while a
    far-off sense sits alone and carries the variance. Dominant-mode row loadings on a live corpus:

        "what is a dog"          unit-norm: frump 61.2, unpleasant woman 55.5, cad 44.4
                                 energised: dog 320.7, domestic animal 289.7, canine 56.4
        "who was albert einstein" unit-norm: world health organization 51.0, un agency 50.9
                                 energised: einstein 56.8, physicist 16.1
        "capital of france"      unit-norm: paris 14.6 (next 0.71) | energised: paris 24.8 (next 0.32)

    `None` keeps the unit-norm behaviour for callers measuring pure geometry. The corpus basis is
    built that way on purpose: there every noun counts once, rather than by how loudly one query
    happened to fire it.

    `names` are bare synset names (no `wn-` prefix), already ordered — by reach, by score, by
    whatever the caller measured. The order is data and is preserved exactly; this function never
    sorts, because sorting a screen destroys the lag-1 structure the instrument reads.

    Returns None when a frame cannot be built (fewer rows than the optics minimum, no resolvable
    coordinates) rather than a padded or partial frame, because a fabricated row is a fabricated
    measurement.

    The matrix alone cannot be aligned back to `names`, because rows are dropped (see above) and this
    signature has nowhere to say which. A caller taking a per-row reading calls `frame_rows`, which
    returns the same frame together with the correspondence; this entry point serves the callers that
    read the frame whole (`read`, `resolvable`, `coherent`, the pooling placers), where no row needs
    a name."""
    got = _build(store, names, center=center, energy=energy, corpus_basis=corpus_basis, k=k)
    return None if got is None else got[0]


def readings(W):
    """The distinguishable readings this beam carries — one per resolved mode, named by the row that
    is it, ordered by the energy that mode absorbs. `[(row_index, mode_energy), …]`, or None when the
    frame carries no read.

    Ranking rows by their loading on the dominant mode alone does not discriminate, for two
    structural reasons:

      1. The instrument has no reading to give on this shape. `signal_end(frame=W)` consults the
         instrument only when the Weyl interval has collapsed (`require_certain=True`), and on a query
         screen that is `(T≈14–81, F=280)` — `F ≫ T`, the regime this module's header calls vacuous
         — the certified interval measures `[1, 280]` on every probe query, with
         `concentration_band` between 10.6 and 48.9. The answer then falls through to an Otsu split
         of a bare column, which `prism.resolution` states is not a frame and does not decide.
      2. A column of loadings is the wrong measurement. A loading mixes a row's energy with its
         alignment to one direction, so the series has no reason to separate. On "who was albert
         einstein" the 20 loadings decay smoothly, `η² = 0.6785` against a null of `0.7519` — less
         structure than a featureless ramp — so `separated()` reads no separation and the cut
         correctly returns all 20, giving Einstein followed by nineteen glosses about Washington
         State and the World Health Organization.

    The field does separate, along other axes. Those nineteen are different readings, and the
    instrument resolves them as different: `principal_directions` returns the `k` correlation modes
    standing above the noise floor, and routing each row to the mode that absorbs the most of it
    separates the field cleanly. Same query:

        mode 0  E=4.478  einstein, physicist, scientist
        mode 1  E=3.049  washington, american state, state
        mode 3  E=3.090  world health organization, un agency, administrative unit

    Three subjects, three modes. The discrimination is in the evidence; it is the loading column that
    flattens it.

    A mode is one reading, named by its peak. A mode is a direction, and the concept at that
    direction is the row carrying most of the mode's energy. That is what keeps a taxonomy from being
    read as several answers: `cat`, `feline` and `carnivore` are one line in the ontology, so they
    are collinear and land in one mode — the same thing measured at three resolutions. The taxonomy
    survives as structure: `_stated_relations` states `cat -hypernym-> feline` as an edge.

    Routing is `next_by_coupling`'s own rule — which absorber takes the most of this signal — applied
    per row across the frame's own resolved modes instead of across a set of tektons. Both are
    comparisons of measured absorbed energy rather than an argmax over a chosen score. A mode no row
    belongs to names nothing and is left out, which is why the count of readings can be smaller than
    `k_signal` and never larger.

    Ordered by absorbed energy rather than by eigenvalue, and the difference is measurable.
    `principal_directions` orders by the correlation eigenvalue, which is scale-invariant: it
    discards magnitude, and magnitude is precisely what `frame(energy=…)` puts into this beam. On
    "who was albert einstein" the eigenvalue order gives modes `[4.478, 3.049, 2.340, 3.090, 2.380]`,
    so the third-largest energy sits fourth. Ordering readings by what they absorb is the same choice
    as energising the frame in the first place.

    Returns None — rather than an empty list — when the frame cannot carry a read, so "unreadable"
    and "read, and nothing resolved" stay different answers."""
    try:
        import numpy as np
        from ember.optics import principal_directions
    except Exception:
        return None
    P = principal_directions(W)
    if P is None or getattr(P, "ndim", 0) != 2 or P.shape[1] < 1:
        return None
    E = (np.asarray(W, dtype=float) @ P) ** 2      # (T, k): energy each row places on each mode
    if not np.isfinite(E).all() or not E.any():
        return None
    home = np.argmax(E, axis=1)                    # each row's own strongest coupling
    out = []
    for j in range(int(P.shape[1])):
        mem = np.flatnonzero(home == j)
        if mem.size == 0:
            continue                               # a mode no row belongs to names nothing
        peak = int(mem[int(np.argmax(E[mem, j]))])
        out.append((peak, float(E[:, j].sum())))
    if not out:
        return None
    out.sort(key=lambda t: -t[1])                  # by absorbed energy — see the note above
    return out


def read(store, names: Sequence[str], *, center=None, corpus_basis: bool = True,
         k: Optional[int] = None) -> Dict[str, Any]:
    """Screen these candidates through the instrument and report what it saw.

    This is what building the frame buys: `k_signal` is a measurement of how many distinguishable
    things were reached rather than an approximation of one. Everything is reported — including
    `frame: False` when no frame could be built — so a caller can always tell a measurement from an
    absence.

    `corpus_basis` — see `frame`. The ruling (PLAN §1.5) is instrument -> True, and it is stated at the
    call site rather than inherited from a signature; the callers that go through here
    (`activation.output_membrane`, `lumen.output_screen.generate`) pass it explicitly. It is reported
    back in the result, so a reading is always attributed to the coordinate it was taken in.

    `k` / `basis_k` — which zoom the read was taken at, asked for and reported back. `k=None` reads
    at the corpus basis's stored width (`_zoom`). `basis_k` is the zoom that was used and is `None`
    on the raw dense arm, which has no zoom. A `k_signal` is a count of modes in a coordinate, and
    two counts taken at different `k` are not comparable, so the reading carries the `k` it was taken
    at ([[state-what-it-is]])."""
    got = _build(store, names, center=center, corpus_basis=corpus_basis, k=k)
    if got is None:
        return {"frame": False, "rows": 0, "k_signal": None, "corpus_basis": bool(corpus_basis),
                "basis_k": None}
    W, _kept, zoom = got
    from ember import optics
    from ember.optics import read_ordered
    out: Dict[str, Any] = {"frame": True, "rows": int(W.shape[0]), "features": int(W.shape[1]),
                           "corpus_basis": bool(corpus_basis), "basis_k": zoom}
    rd = read_ordered(W, with_screen=False)
    out["k_signal"] = rd.k_signal
    # The count travels with its margins. `k_signal` is `#(eigenvalue > floor)`, so a mode sitting
    # 5.75 above the floor and one sitting 0.087 above it are reported with identical authority, and
    # the margins are what tell them apart. Sweeping the screen size collapses the margin from 5.75
    # at T=32 to 0.0873 at T=208, with the first unresolved mode at -0.0402 — a threshold artefact
    # rather than a measurement, in the terms `ember.optics` states.
    out["k_margin_last"] = rd.k_margin_last
    out["k_margin_next"] = rd.k_margin_next
    out["k_lo"], out["k_hi"] = rd.k_lo, rd.k_hi
    out["k_certain"] = bool(rd.k_lo == rd.k_hi == rd.k_signal)
    out["coherence"] = optics.coherence(W)
    out["fill"] = optics.fill(W)
    out["spots"] = optics.spots(W)
    return out


def resolvable(store, names: Sequence[str], *, center=None,
               corpus_basis: bool = True, k: Optional[int] = None) -> Optional[int]:
    """How many of these are distinguishable — `k_signal` off the screen, or None if unreadable.

    `corpus_basis` — see `read`. It is which coordinate the count is a count in, and it decides
    whether there is a count at all: five conversational queries read `k_signal` 2–5 on the corpus
    basis and 0 on every one of them in the raw dense coordinate, because `(T≈14–81, F=2048)` is the
    `F >> T` regime `ember.optics` calls vacuous by construction.

    The zoom matters the same way — a count taken at one `k` is comparable only with one taken at the
    same `k`. Sweeping the corpus basis width on "what is a dog" (T=64), `k_signal` reads 1 at k=8, 2
    at k=32, 3 at k=64, 4 at k=280. Each is a count of what is distinguishable at that resolution.
    `read(...)["basis_k"]` reports the zoom, and a caller comparing two counts compares that too."""
    W = frame(store, names, center=center, corpus_basis=corpus_basis, k=k)
    if W is None:
        return None
    from ember.optics import resolvable as _r
    return _r(W)


def coherent(store, names: Sequence[str], *, center=None,
             corpus_basis: bool = True, k: Optional[int] = None) -> Optional[bool]:
    """Is this reached set one coherent, scale-invariant thing? The certified DEFINE signal.

    Read through the wrapper's `scale_read`, so the answer path goes through the one instrument rather
    than a locally constructed one. True when the coupling peaks across essentially the whole
    instrument, i.e. the coordinate stayed scale-invariant throughout the reach. None when unreadable,
    which is a different answer from False and the caller keeps them apart.

    `corpus_basis` and `k` — see `read`."""
    W = frame(store, names, center=center, corpus_basis=corpus_basis, k=k)
    if W is None:
        return None
    from ember.optics import scale_read as _sr
    r = _sr(W)
    return None if r is None else bool(r.get("scale_invariant"))


# The read collection's own coordinate — the same construction, over what was read.
#
# `build_basis` above derives the ontology's basis from the noun cloud. A read collection needs the
# same thing derived from itself — no WordNet, no preloaded coordinate — because the whole claim of
# the reading path is that structure comes from what was read.
#
# This lives here rather than in a persona because both astra (ingest) and lumen (reasoning) need
# it, and `test_no_persona_SOURCE_imports_another_persona` forbids one persona importing another —
# correctly: a persona that imports a sibling cannot be deployed alone. A measurement both personas
# need is the host's, reached by seam, which is exactly what `build_basis` already is.
#
# Nothing here is preselected: the width is the number of distinct contexts the reading formed — a
# count it produced, not a size anyone chose — and the amplitude is `sqrt(self_information_bits(...))`
# through the one instrument. Measured on `read:mine`, the three coordinates tried:
#
#     one-hot over vocabulary   F=3067  T=12000   contrast 0.4671   resolved_modes  0   no read
#     signed hash into D=2048   F=2048  T= 3870   contrast 4.2303   resolved_modes  4   reads
#     contexts x sqrt(bits)     F= 180  T= 3870   contrast 9.5719   resolved_modes 22   reads
#
# The chosen width was costing most of the signal, and the one-hot width grew with the corpus.

_OBSERVED = ("observed", "observed_alone")


def read_unit_contexts(ro, collection: str) -> Dict[str, List[str]]:
    """unit -> the contexts holding it, over one indexed range on `src`. Never a table scan, and
    never a `LIKE`, which cannot use the index the way a range can."""
    import collections as _c
    out = _c.defaultdict(list)
    q = ("SELECT src, dst FROM edge WHERE src >= ? AND src < ? AND label IN (%s)"
         % ",".join("?" * len(_OBSERVED)))
    for src, dst in ro.execute(q, (collection + ":", collection + ";") + _OBSERVED):
        out[dst].append(src)
    return out


def read_cloud(ro, collection: str):
    """The read corpus as `(M, names, contexts)` — one row per unit, one column per context.

    Each cell is `sqrt(I)`, the amplitude whose square is that context's information in bits, so
    `‖row‖²` is the unit's energy (the `A² + T² = 1` convention the organon reads with). A context
    holding nearly everything narrows nothing and arrives at ~0 bits; a rare one arrives loud — no
    stop-list, and nothing excluded. `(None, [], [])` when the reading holds no unit in any
    context."""
    import collections as _c

    import numpy as np
    from ember.optics import self_information_bits
    uc = read_unit_contexts(ro, collection)
    if not uc:
        return None, [], []
    deg = _c.Counter()
    for ctxs in uc.values():
        for c in ctxs:
            deg[c] += 1
    names, contexts = sorted(uc), sorted(deg)
    col = {c: j for j, c in enumerate(contexts)}
    total = float(sum(deg.values()))
    amp = {}
    for c, k in deg.items():
        bits = self_information_bits(float(k), total)
        amp[c] = float(np.sqrt(bits)) if bits and bits > 0.0 else 0.0
    M = np.zeros((len(names), len(contexts)), dtype=np.float64)
    for i, u in enumerate(names):
        for c in uc[u]:
            M[i, col[c]] += amp[c]
    return M, names, contexts


def read_basis(ro, collection: str) -> Optional[dict]:
    """The read collection's basis and the read behind it, or `None` when it holds nothing.

    `k` and the directions come from ONE spectrum through `optics.principal_directions`, so the
    count and the subspace cannot drift apart — the same property `_basis_from_cloud` holds."""
    import numpy as np
    from ember.optics import principal_directions, read_ordered
    M, names, contexts = read_cloud(ro, collection)
    if M is None:
        return None
    rd = read_ordered(M, with_screen=False)
    B = principal_directions(M)
    base = {"rows": int(M.shape[0]), "d": int(M.shape[1]), "contrast": float(rd.contrast),
            "k_lo": rd.k_lo, "k_hi": rd.k_hi, "k_certain": bool(rd.k_certain),
            "names": names, "contexts": contexts}
    if B is None or B.ndim != 2 or B.shape[1] < 1:
        base.update({"basis": None, "k": 0,
                     "refusal": "the cloud resolved no mode above the instrument's floor"})
        return base
    base.update({"basis": B, "k": int(B.shape[1]), "has_signal": bool(rd.has_signal),
                 "live_columns": int(np.count_nonzero(M.any(axis=0))), "refusal": None})
    return base


__all__ = ["frame", "frame_rows", "readings", "read", "resolvable", "coherent", "build_basis",
           "load_basis", "basis_band", "basis_head", "BASIS_ROOT", "BASIS_CONTENT_TYPE",
           "read_unit_contexts", "read_cloud", "read_basis"]
