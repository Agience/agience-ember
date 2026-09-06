"""The zoom is caller-stated, and the inbound frame is read at the corpus basis's own stored width.

A corpus basis at its stored width *is* the corpus's resolution, so reading through all of it is not
a chosen zoom. What the stored generation carries is an interval, not a point: `geom.corpus-basis`
on 71 reads `k = 280` with `k_certified_lo = 123`, `k_certified_hi = 956`, `k_certain = False`. The
instrument is saying the evidence admits every resolution from 123 to 956, and 280 is not the centre
of it. So a caller may state `k`, only `k_hi` bounds what it may state, and nothing lifts a stated
`k` to the corpus floor (COMPACTIFICATION §2).

Inbound, `resolvable(P)` is not consulted. The top-variance directions of a field are its ancestor
chain — what its members share — so truncating to the few modes above the noise floor keeps the
agreement and discards what discriminates. A count of resolved modes is not a resolution for a read
that has to tell things apart. Measured live on 71, taking each read at `resolvable(P)`:

    what is a dog     k=280 -> dog.n.01     k=4 -> canine.n.02
    what is water     k=280 -> water.n.01   k=4 -> binary_compound.n.01
    capital of france k=280 -> paris.n.01   k=2 -> k_signal 0, nothing resolved

The zoom that a read returning too much needs is outbound instead
(`activation._stated_relations` step 5 / `prism.resolution.signal_end`), which is not this file.

Half this file is the price of a stated `k`. PAPER §5.5: coupling *"raises without"* a shared basis.
One generation serving many widths means `pooling.coordinate_id` has to carry the `k`, or two frames
sharing no feature axis mint the same token and pool.

Every test states its failure mode first and carries the control that keeps it from passing
vacuously. Hermetic: a stubbed coordinate and a stubbed basis store, no live corpus, no running
service.
"""
from __future__ import annotations

import numpy as np
import pytest

from ember.signal import pooling
from ember.signal import projection as pj

D = 64                 # the raw dense width
K_STORED = 40          # how many directions the stored generation holds


# ── an exactly plantable frame ───────────────────────────────────────────────────────────────────
# `B` is orthonormal, so for a row `w = B.T @ c` the projection `w @ B.T` is `c` exactly. That makes
# the projected frame something the test authors rather than measures: the ground truth is the
# number of subjects mixed into `c`, computed from the generator rather than from the thing under
# test.

def _basis(seed=5):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(D, K_STORED)))
    return Q.T                                        # (K_STORED, D), orthonormal rows


def _coeffs(n_subjects, *, T=48, seed=0, gain=6.0):
    """`(T, K_STORED)` coefficients carrying exactly `n_subjects` independent directions."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(K_STORED, n_subjects)))
    A = np.abs(rng.normal(size=(T, n_subjects))) * gain
    return A @ U.T + 0.02 * rng.normal(size=(T, K_STORED))


class _BasisStore:
    """A store holding exactly one thing: a corpus basis generation. `head_of` is the lineage
    resolution `projection` uses; `content_type_mark` is the freshness gate."""

    def __init__(self, B, band=(4, 30), gen="geom.corpus-basis~stub"):
        self.doc = {"id": gen, "root_id": "geom.corpus-basis",
                    "content_type": pj.BASIS_CONTENT_TYPE, "state": "committed",
                    "k": int(B.shape[0]), "d": int(B.shape[1]),
                    "basis": [[float(x) for x in row] for row in B]}
        if band is not None:
            self.doc["k_certified_lo"], self.doc["k_certified_hi"] = int(band[0]), int(band[1])
            self.doc["k_certain"] = band[0] == band[1]
        self.artifacts = self

    def head_of(self, root_id):
        return self.doc if root_id == "geom.corpus-basis" else None

    def content_type_mark(self, content_type, cap=2000):
        return (1, 1, True) if content_type == pj.BASIS_CONTENT_TYPE else (0, 0, True)


@pytest.fixture()
def planted(monkeypatch):
    """Two named sets whose projected frames carry a different number of subjects by construction:
    `few.*` mixes 2 directions, `many.*` mixes 6. Returns `(store, B, few, many)`."""
    B = _basis()
    Cf, Cm = _coeffs(2, seed=1), _coeffs(6, seed=6)
    vecs = {}
    for i, c in enumerate(Cf):
        vecs["few.%d" % i] = B.T @ c
    for i, c in enumerate(Cm):
        vecs["many.%d" % i] = B.T @ c
    from crystal.ontology import geometry as g
    from crystal.ontology import driver as wn

    class _Syn:
        def __init__(self, n):
            self._n = n

        def name(self):
            return self._n

    monkeypatch.setattr(wn, "synset", lambda n: (_Syn(n) if n in vecs else None))
    monkeypatch.setattr(g, "load_ic", lambda: None)
    monkeypatch.setattr(g, "dense_vec", lambda s, ic, center=None: vecs[s.name()])
    pj._BASIS_CACHE.clear()
    yield B, [n for n in vecs if n.startswith("few.")], [n for n in vecs if n.startswith("many.")]
    pj._BASIS_CACHE.clear()


# ═════ 1 — the inbound read is the corpus's own width, and not the frame's ═══════════════════════

def test_two_frames_that_resolve_DIFFERENTLY_still_come_back_at_THE_CORPUS_WIDTH(planted):
    """The inbound frame takes no zoom: two frames that resolve differently come back at the same
    width, the corpus basis's stored one.

    Taking each read at `resolvable(P)` — the frame's own resolved-mode count — moves the lead to
    its own generic ancestor, and the live answers follow it down: "what is a dog" returns canine's
    gloss, "what is water" returns binary compound's, "capital of france" floods to seven glosses
    about capitals and regions in general. The cause is structural rather than a tuning miss: the
    top-variance directions of a field are the ancestor chain, because that is what its members
    share, so truncating to the few modes above the noise floor keeps the agreement and throws away
    what discriminates. A count of resolved modes is not a resolution for a read that has to tell
    things apart. The zoom for a read returning too much lives outbound
    (`activation._stated_relations` step 5 / `prism.resolution.signal_end`).

    The control is the whole test. Equal widths prove nothing if the two frames resolve the same,
    because the assertion would hold on a build that auto-zoomed. So the frames are first shown to
    resolve genuinely differently through the instrument itself (`resolvable` reads 2 and 5 on
    them), and only then required to come back at the same width. That separates "the zoom did not
    fire" from "there was nothing for it to fire on"."""
    from ember.optics import resolvable as _r
    B, few, many = planted
    store = _BasisStore(B, band=(3, 30))
    Wf = pj.frame(store, few)
    Wm = pj.frame(store, many)
    assert Wf is not None and Wm is not None
    assert Wf.shape[0] == Wm.shape[0], "the control needs equal T so width cannot be tracking T"
    # The control: these two frames are not the same read, and the instrument says so.
    kf, km = _r(Wf), _r(Wm)
    assert kf is not None and km is not None and kf != km, (
        "the control collapsed: both frames resolve %r, so equal widths below would prove nothing "
        "about whether the inbound zoom fired" % (kf,))
    # ...and the widths agree anyway, because the inbound read is taken at the corpus's own width.
    assert Wf.shape[1] == Wm.shape[1] == K_STORED, (
        "the frames came back at %d and %d columns — the inbound auto-zoom is back, and it is what "
        "made 'what is a dog' answer with canine's gloss" % (Wf.shape[1], Wm.shape[1]))


def test_the_zoom_is_a_PREFIX_of_the_full_projection_not_a_second_projection(planted):
    """The basis rows are ordered by eigenvalue, so `B[:k]` is a basis for every `k <= k_stored` and
    `P[:, :k]` is `W @ B[:k].T`. That identity makes the zoom a slice rather than a re-projection,
    and leaves the point estimate recoverable.

    A zoom implemented as a fresh decomposition per read would produce columns that are not the
    corpus's directions, and two reads at different `k` would share no prefix, so nothing downstream
    could relate them. Checked against the projection computed here from the stored matrix, which is
    an oracle independent of `frame`.

    Exercised through a stated `k`, because that is where the truncation lives. The unstated read is
    the full projection, which satisfies a prefix assertion trivially, so every stated zoom in the
    band is checked against the oracle instead."""
    B, few, _many = planted
    store = _BasisStore(B, band=(3, 30))
    raw = pj.frame(store, few, corpus_basis=False)
    full = raw @ B.T
    # the unstated read IS the full projection — no truncation, and the oracle says so exactly.
    assert np.allclose(pj.frame(store, few), full, atol=1e-12)
    # ...and every stated zoom is the leading columns of that one projection, not a re-decomposition.
    for k in (3, 7, 18, 30):
        Wk = pj.frame(store, few, k=k)
        assert Wk.shape[1] == k
        assert np.allclose(Wk, full[:, :k], atol=1e-12), (
            "the frame at k=%d is not the leading columns of the full projection — the zoom is a "
            "second decomposition, and two reads at different k would share no prefix" % k)


# ═════ 2 — the band is the bound, and the band is the artifact's ══════════════════════════════════

def test_ONLY_k_hi_bounds_a_stated_zoom_and_the_corpus_floor_never_lifts_it(planted):
    """Only `k_hi` bounds a stated zoom, and the corpus floor never lifts one. The asymmetry is not
    incidental: above `k_hi` there are no directions stored and the corpus certified the rest as
    noise, so asking for them asks for something that does not exist. Below `k_lo` there is nothing
    missing — the corpus's floor is a statement about ~85k nouns pooled, not permission for a caller
    to ask a coarser question about fourteen. A caller that states `k` has measured something this
    function has no standing to overrule, so a stated `k` is carried through untouched.

    `k=1` has no frame to give: one direction is a line, and the instrument's own contract admits no
    frame narrower than two columns — `_build` returns None rather than handing back a one-column
    "frame", the same null that `readings`/`read` give everywhere else. That is an absence, and it
    is reported as one.

    The control: an admissible stated `k` must come back exactly as asked (`k=15` -> 15), or every
    assertion here would also hold on a build that ignored `k` entirely and always returned the
    stored width."""
    B, few, _many = planted
    store = _BasisStore(B, band=(6, 24))
    assert pj.basis_band(store) == (6, 24)
    # the control: `k` is honoured exactly, so the assertions around it are about clamping and not
    # about `k` being ignored.
    assert pj.frame(store, few, k=15).shape[1] == 15, "an admissible zoom must be taken as asked"
    # below `k_lo` is not lifted: 3 is under the certified floor of 6 and is served as 3.
    assert pj.frame(store, few, k=3).shape[1] == 3, (
        "a stated zoom below the corpus floor was lifted to it — that is the 123 defect returning")
    # ...down to the instrument's own shape contract, which admits no frame narrower than two columns.
    assert pj.frame(store, few, k=2).shape[1] == 2
    assert pj.frame(store, few, k=1) is None, "one direction is a line; it must be refused, not served"
    # above `k_hi` is brought to the ceiling — beyond it nothing is stored and nothing was certified.
    assert pj.frame(store, few, k=999).shape[1] == 24, "above k_hi was not brought to the ceiling"
    # ...and `k_hi` can never exceed what the generation actually stores.
    wide = _BasisStore(B, band=(6, 10_000), gen="geom.corpus-basis~wide")
    pj._BASIS_CACHE.clear()
    assert pj.frame(wide, few, k=9_999).shape[1] == K_STORED


def test_a_generation_that_states_NO_band_takes_NO_zoom_even_when_one_is_ASKED_for(planted):
    """[[absence-is-not-an-affirmative-claim]]. A generation written before the interval was
    recorded says nothing about which resolutions its evidence admits. With no certified band there
    is no statement about what is admissible, so there is nothing to bound a stated `k` by — and
    honouring it anyway would serve a truncation against a certification nobody measured. No band,
    no zoom: the frame comes back at the full stored width whether or not a `k` was asked for.

    The control distinguishes "no band -> no zoom" from "this build never zooms at all" through the
    stated `k`: with a band it must bite (`k=5` -> 5), and with no band the identical call must not
    (`k=5` -> the stored width). Same call, same frames, one difference — the band."""
    B, few, many = planted
    bare = _BasisStore(B, band=None)
    assert pj.basis_band(bare) is None
    assert pj.frame(bare, few).shape[1] == K_STORED
    assert pj.frame(bare, many).shape[1] == K_STORED
    assert pj.frame(bare, few, k=5).shape[1] == K_STORED, (
        "a stated k was honoured against a generation that certified no admissible range")

    pj._BASIS_CACHE.clear()
    banded = _BasisStore(B, band=(3, 30), gen="geom.corpus-basis~banded")
    # the control: the very same stated zoom does bite once the generation certifies a range.
    assert pj.frame(banded, few, k=5).shape[1] == 5, (
        "a stated zoom did not bite even WITH a band — this build ignores `k` entirely and the "
        "assertions above prove nothing about the band")
    # ...and unstated is still the corpus's own width on both arms, band or no band.
    assert pj.frame(banded, few).shape[1] == K_STORED
    assert pj.frame(banded, many).shape[1] == K_STORED


def test_HALF_an_interval_is_not_an_interval(planted):
    """Reading `k_certified_lo` alone and filling `k_certified_hi` in from `k` or from the stored
    width would use numbers that are lying around rather than a measured ceiling — the
    `.get("grounded", True)` shape applied to a zoom range. Half a certified interval carries no
    band."""
    B, few, _many = planted
    for missing in ("k_certified_lo", "k_certified_hi"):
        pj._BASIS_CACHE.clear()
        s = _BasisStore(B, band=(3, 30), gen="gen~%s" % missing)
        del s.doc[missing]
        assert pj.basis_band(s) is None, "%s alone was read as an interval" % missing
        assert pj.frame(s, few).shape[1] == K_STORED
    # a band that brackets nothing readable is not a band either
    for bad in ((1, 30), (30, 3)):
        pj._BASIS_CACHE.clear()
        assert pj.basis_band(_BasisStore(B, band=bad, gen="gen~%r" % (bad,))) is None


def test_the_INBOUND_zoom_never_consults_the_instrument_readable_frame_or_not():
    """`_zoom` never falls back to `resolvable(P)` as its unstated default: the inbound width does
    not move when the frame's readability moves, including between two frames the instrument reads
    as genuinely different. Taking the inbound read at `resolvable(P)` moves the lead to a field's
    generic ancestor chain, which is what produces wrong answers for reads that have to tell things
    apart.

    The control is a frame the instrument reads (`resolvable` -> 3, asserted here so the case is not
    vacuous) and a frame it has no reading for (`None` — too few rows, all-zero) coming back at the
    same width. If those ever differ, something inbound is consulting the instrument after all.

    The absence still matters on the other path: `resolvable` returning `None` is an absence, not a
    small number, and nothing here converts it into one — the inbound read does not consult it at
    all."""
    from ember.optics import resolvable as _r
    band = (6, 24)
    unreadable_rows = np.ones((2, K_STORED))            # below MIN_ROWS
    unreadable_zero = np.zeros((20, K_STORED))          # nothing to read
    readable = _coeffs(3, seed=4)
    # the control: these three are genuinely different reads to the instrument.
    assert _r(unreadable_rows) is None and _r(unreadable_zero) is None
    assert _r(readable) is not None, "the readable control is not readable — the test is vacuous"
    # ...and the inbound zoom is the same width for all three, because it never asked.
    assert pj._zoom(unreadable_rows, band) == K_STORED
    assert pj._zoom(unreadable_zero, band) == K_STORED
    assert pj._zoom(readable, band) == K_STORED, (
        "a READABLE frame came back narrower than an unreadable one — `resolvable` is back on the "
        "inbound path")
    # a stated `k` still bites on the very same frames, so "no inbound zoom" is not "no zoom".
    assert pj._zoom(readable, band, k=9) == 9


# ═════ 3 — the coordinate token carries the zoom ═══════════════════════════════════════════════════

def test_the_token_NAMES_THE_ZOOM_so_two_widths_of_one_generation_cannot_compare_equal(planted):
    """Once one generation serves every width in its band, `B[:6]` and `B[:24]` share no feature
    axis — `B[:6]` is not a sub-coordinate of `B[:24]` that anything may compare across. A token
    naming only the generation would say "same coordinate" about two frames that share nothing
    (PAPER §5.5: coupling *"raises without"* a shared basis; REASONING-PORT R3.2 flags exactly
    this).

    The control: the same generation at the same zoom must still compare equal, or the token has
    become unstable and would drop every plane forever."""
    B, _few, _many = planted
    store = _BasisStore(B, band=(6, 24))
    t6 = pooling.coordinate_id(store, corpus_basis=True, k=6)
    t24 = pooling.coordinate_id(store, corpus_basis=True, k=24)
    assert t6 is not None and t6 != t24, "two zooms of one generation minted the same token"
    assert t6 == pooling.coordinate_id(store, corpus_basis=True, k=6), "the token is not stable"
    assert store.doc["id"] in t6 and 6 in t6
    # the raw dense arm has no zoom and must not grow one
    dense = pooling.coordinate_id(store, corpus_basis=False)
    assert dense == pooling.coordinate_id(store, corpus_basis=False, k=6)
    # and a store that cannot name its generation still names nothing, zoom or no zoom
    assert pooling.coordinate_id(object(), corpus_basis=True, k=6) is None


def test_an_UNSTATED_zoom_is_completed_from_the_plane_never_defaulted(planted):
    """`F` is the zoom on a corpus-basis frame, so `place` reads it off the plane it was handed — a
    measurement of the plane, not an assumption about it. That is what lets a caller which did not
    thread the width through (`activation.compose`) still pin an exact coordinate.

    Treating an unstated `k` as "matches whatever is pinned" would mean the check could never fire
    ([[verification-that-cannot-fail]]), and every zoom would pool onto every other."""
    B, _few, _many = planted
    store = _BasisStore(B, band=(6, 24))
    unstated = pooling.coordinate_id(store, corpus_basis=True)
    assert unstated[-1] is None, "an unstated zoom must be carried as unstated, not filled in"
    assert pooling._at_zoom(unstated, 6) == pooling.coordinate_id(store, corpus_basis=True, k=6)
    assert pooling._at_zoom(unstated, 24) != pooling._at_zoom(unstated, 6)
    # a token that did state its zoom is never overwritten, and a token with no zoom slot is untouched
    stated = pooling.coordinate_id(store, corpus_basis=True, k=6)
    assert pooling._at_zoom(stated, 24) == stated
    assert pooling._at_zoom(None, 24) is None
    assert pooling._at_zoom(("geometry", "v", "dense"), 24) == ("geometry", "v", "dense")


def test_two_zooms_of_one_generation_DO_NOT_POOL_and_the_drop_is_counted(planted):
    """Two zooms of one generation must not pool, and the drop is counted. A per-read `k` inside one
    generation produces frames that share no feature axis, and pooling across them would let a
    Screen accumulate silently under one stale pool.

    The control is the first placement pair: two planes at the same zoom must pool, or a Screen that
    never pools anything would pass this test while holding nothing."""
    B, _few, _many = planted
    store = _BasisStore(B, band=(6, 24))
    rng = np.random.default_rng(3)
    sc = pooling.PooledScreen("t.zoom")
    tok = pooling.coordinate_id(store, corpus_basis=True)          # unstated: completed on place

    assert sc.place(rng.normal(size=(10, 6)), basis=tok) == "pooled"
    assert sc.place(rng.normal(size=(10, 6)), basis=tok) == "pooled", \
        "the control failed: two planes at the same zoom did not pool"
    assert sc.summary()["basis"][-1] == 6, "the Screen did not record WHICH zoom it pooled"

    assert sc.place(rng.normal(size=(10, 24)), basis=tok).startswith("dropped:"), \
        "a plane in a different zoom of the same generation pooled anyway"
    s = sc.summary()
    assert s["planes"] == 2 and sum(s["drops"].values()) == 1


# ═════ 4 — the read says which zoom it was taken at ═════════════════════════════════════════════

def test_read_reports_the_zoom_and_reports_NO_zoom_where_there_is_none(planted):
    """`k_signal` is a count of modes in a coordinate, and once the coordinate can differ per read,
    a bare count is uncomparable with any other. Sweeping the corpus width on one query, "what is a
    dog" (T=64): `k_signal` reads 1 at k=8, 2 at k=32, 3 at k=64 and 4 at k=280. None of those is
    wrong and none of them means anything without its `k`. The reading has to carry the zoom it was
    taken at ([[state-what-it-is]]).

    The control: the raw dense arm has no zoom, so `basis_k` must be `None` there — not the frame's
    width, which is a number that is not a resolution.

    `basis_k` unstated is the corpus's stored width, not a per-read measurement. The reading still
    has to carry its `k` for exactly the reason above; what changed is which `k` that is. A stated
    zoom is reported as stated, so the two cases stay distinguishable in the result."""
    B, few, _many = planted
    store = _BasisStore(B, band=(6, 24))
    r = pj.read(store, few)
    assert r["frame"] is True and r["basis_k"] == r["features"] == K_STORED, (
        "an unstated zoom reported %r — the read is not being taken at the corpus's stored width"
        % (r["basis_k"],))
    # the control: a stated zoom is reported as itself, so `basis_k` is not simply `features` always.
    stated = pj.read(store, few, k=17)
    assert stated["basis_k"] == 17 and stated["features"] == 17
    off = pj.read(store, few, corpus_basis=False)
    assert off["frame"] is True and off["basis_k"] is None and off["features"] == D
    assert pj.read(store, few[:1])["basis_k"] is None          # an absence claims no zoom either


# ═════ 5 — the stored matrix goes out to `k_hi`, not to the point estimate ═════════════════════════

def test_the_svd_derivation_stores_the_WHOLE_CERTIFIED_BAND_not_the_point_estimate(monkeypatch):
    """`_basis_from_cloud` stores the certified band, not the point estimate. Because the rows are
    ordered by singular value, storing out to `k_hi` costs nothing and loses nothing: the point
    estimate is still exactly `B[:k]`, while the directions between `k` and `k_hi` — every finer
    zoom the evidence admits — remain available to a later read.

    The instrument is muted here because there are two derivation paths and only one of them can be
    widened from inside this module. `ember.optics.principal_directions` returns exactly
    `resolved_modes` columns — entroptics counts and slices in one call — so on the instrument path
    the widest matrix reachable through the one sanctioned door is the point estimate, and widening
    it would take either a private eigendecomposition here ([[one-instrument-enforced]]) or a `k`
    argument on `principal_directions`, which is `ember.optics`'s to add — a seam stated in
    `_basis_from_cloud` itself, not faked. The SVD fallback is a decomposition this module owns
    outright, so muting the instrument is how the real fallback branch — the one a corpus with no
    clean rank actually takes — is entered.

    The control is the second assertion: this cloud must produce `k_hi > k`, or a matrix stored at
    the point estimate would satisfy everything below and the test would prove nothing."""
    import ember.optics as _bo
    monkeypatch.setattr(_bo, "principal_directions", lambda *a, **kw: None)

    rng = np.random.default_rng(6)
    n, d = 48, 40
    U, _ = np.linalg.qr(rng.normal(size=(d, 6)))
    M = np.abs(rng.normal(size=(n, 6))) * 6.0 @ U.T + 0.02 * rng.normal(size=(n, d))
    B, k, rd, src = pj._basis_from_cloud(M)
    assert src == "svd-fallback"
    assert k >= 2 and rd.k_hi > k, (
        "the certified interval collapsed onto the point estimate (k=%r hi=%r), so a matrix stored "
        "at `k` would pass every assertion below" % (k, rd.k_hi))
    assert B.shape == (min(rd.k_hi, min(n, d)), d), (
        "the basis was stored at %d rows, not out to the certified ceiling %d"
        % (B.shape[0], rd.k_hi))
    # the point estimate is not lost — it is the prefix, and every prefix is orthonormal, which is
    # what makes `B[:j]` a usable basis at every zoom `j` and not only at the two named ones.
    for j in (k, B.shape[0]):
        assert np.allclose(B[:j] @ B[:j].T, np.eye(j), atol=1e-9)
