"""The screen — the ordered frame every cut and every condensation sits on.

These pin the two properties that make the frame readable: the order is data, and the feature axis
must be the basis the screen actually occupies.
"""
import numpy as np
import pytest

from ember.signal import projection as scr


class _Syn:
    def __init__(self, name, vec):
        self._n, self.vec = name, vec

    def name(self):
        return self._n


@pytest.fixture()
def stubbed(monkeypatch):
    """A 2048-wide coordinate that is sparse, like the real one: ~6 non-zeros per row."""
    rng = np.random.default_rng(0)
    D = 2048
    vecs = {}
    for i in range(40):
        v = np.zeros(D)
        v[rng.integers(0, D, 6)] = rng.normal(size=6)
        vecs["s%d" % i] = v
    from crystal.ontology import geometry as g
    from crystal.ontology import driver as wn
    monkeypatch.setattr(wn, "synset", lambda n: (_Syn(n, vecs[n]) if n in vecs else None))
    monkeypatch.setattr(g, "load_ic", lambda: None)
    monkeypatch.setattr(g, "dense_vec", lambda s, ic, center=None: s.vec)
    return list(vecs)


def test_THE_FEATURE_AXIS_IS_THE_BASIS_THE_SCREEN_OCCUPIES(stubbed):
    """With the full 2048 columns every query reads `k_signal = 0` and `phi_F` between 0.004 and
    0.028 — a (60, 2048) frame filling 2% of its basis, the `F >> T` regime the optics wrapper's own
    header calls vacuous by construction.

    A column that is zero for every row on this screen carries no information about this screen.
    Dropping it removes dimensions holding no data — not a model, not a choice — and the instrument
    then measures the basis the evidence is actually in."""
    W = scr.frame(None, stubbed)
    assert W is not None
    assert W.shape[0] == len(stubbed)
    assert W.shape[1] < 2048, "the empty channels are still being handed to the instrument"
    assert not np.any(np.all(W == 0.0, axis=0)), "an all-zero column survived"


def test_the_order_is_data_and_is_never_sorted(stubbed):
    """Coherence is a lag-1 statistic: a shuffled screen collapses toward the null. The caller
    ordered these by what it measured, and `frame` must preserve that exactly."""
    W1 = scr.frame(None, stubbed)
    W2 = scr.frame(None, list(reversed(stubbed)))
    assert not np.allclose(W1, W2)                       # order changed the frame
    assert np.allclose(W1, scr.frame(None, stubbed))     # and is reproducible


def test_it_REFUSES_rather_than_padding(stubbed):
    """A fabricated row is a fabricated measurement."""
    assert scr.frame(None, stubbed[:2]) is None          # below the optics minimum
    assert scr.frame(None, ["nope"] * 10) is None        # nothing resolves
    assert scr.read(None, stubbed[:2])["frame"] is False


def test_read_reports_a_measurement_or_an_absence_never_a_guess(stubbed):
    r = scr.read(None, stubbed)
    assert r["frame"] is True and r["rows"] == len(stubbed)
    assert r["k_signal"] is not None and r["fill"]["phi_F"] > 0.05
    absent = scr.read(None, stubbed[:1])
    assert absent["frame"] is False and absent["k_signal"] is None


# ── The correspondence — which input each row is ─────────────────────────────────────────────────
@pytest.fixture()
def stubbed_with_holes(monkeypatch):
    """The live shape: some names resolve to a coordinate, some resolve to an all-zero vector.

    `dense_vec` does not raise for an unplaced synset — it returns zeros — so the frame drops those
    rows. On 71, live, that is not rare: "what is a dog" fired 66 concepts and produced 64 rows."""
    rng = np.random.default_rng(1)
    D, vecs, holes = 2048, {}, set()
    for i in range(20):
        v = np.zeros(D)
        if i % 5 == 3:                       # every fifth concept has no geometry at all
            holes.add("h%d" % i)
        else:
            v[rng.integers(0, D, 6)] = rng.normal(size=6)
        vecs["h%d" % i] = v
    from crystal.ontology import geometry as g
    from crystal.ontology import driver as wn
    monkeypatch.setattr(wn, "synset", lambda n: (_Syn(n, vecs[n]) if n in vecs else None))
    monkeypatch.setattr(g, "load_ic", lambda: None)
    monkeypatch.setattr(g, "dense_vec", lambda s, ic, center=None: s.vec)
    return list(vecs), holes


def test_frame_rows_SAYS_WHICH_ROWS_IT_KEPT(stubbed_with_holes):
    """`frame` drops a row whose coordinate is absent and returns the matrix alone, so a caller
    holding a per-row read cannot say which input each row was. `frame_rows` returns `kept` alongside
    the matrix so the correspondence survives the drop; `activation.compose` guards on
    `size == len(names)` and discards the whole read when they disagree.

    Catches `kept` that is merely `range(len(names))`, or a frame that padded the holes back in.
    Both are checked below, and the fixture guarantees holes exist so the assertion cannot pass
    vacuously."""
    names, holes = stubbed_with_holes
    got = scr.frame_rows(None, names)
    assert got is not None
    W, kept = got
    assert holes, "the fixture must actually contain unplaceable names or this proves nothing"
    assert W.shape[0] == len(kept) < len(names), "rows were dropped and `kept` must say so"
    assert [names[i] for i in kept] == [n for n in names if n not in holes]
    # the correspondence is exact, not merely the right length: each row is that name's own
    # coordinate, projected the same way, so a row and its name cannot drift apart.
    solo = scr.frame_rows(None, [names[i] for i in kept])
    assert solo is not None and np.allclose(W, solo[0])


def test_frame_and_frame_rows_are_the_SAME_frame(stubbed_with_holes):
    """`frame` is `frame_rows` minus the correspondence: one implementation, not two that could
    drift. If these two ever disagree, one of them is measuring something else."""
    names, _ = stubbed_with_holes
    got = scr.frame_rows(None, names)
    assert np.allclose(scr.frame(None, names), got[0])
    assert scr.frame(None, names[:2]) is None and scr.frame_rows(None, names[:2]) is None


# ── Readings — a mode is one reading, named by its peak ──────────────────────────────────────────
def _two_subject_frame(seed=0, gains=(3.0, 1.2)):
    """An independent oracle: a frame whose right answer is known before it is ever measured.

    Two orthonormal directions `u, v` (a QR of a random matrix, so they are exactly orthogonal and
    neither can leak into the other), and every row is `a_i·u + b_i·v + noise` with `a` and `b` drawn
    independently. Two subjects, mixed into every row — the real shape, where a concept is not
    confined to one direction — rather than two disjoint blocks, which the instrument reads as a single
    anti-correlated mode (a block frame resolves `k=1`, so it could not test this at all).

    Ground truth, computed from the generator and not from the thing under test:
      · exactly two modes;
      · the `u` mode is named by `argmax(a)`, the `v` mode by `argmax(b)`;
      · the `u` mode carries more energy, because `gains` says so."""
    rng = np.random.default_rng(seed)
    T, F = 40, 12
    Q, _ = np.linalg.qr(rng.normal(size=(F, 2)))
    u, v = Q[:, 0], Q[:, 1]
    a = np.abs(rng.normal(size=T)) * gains[0]
    b = np.abs(rng.normal(size=T)) * gains[1]
    W = a[:, None] * u + b[:, None] * v + 0.01 * rng.normal(size=(T, F))
    return W, int(np.argmax(a)), int(np.argmax(b))


def test_readings_names_each_mode_by_its_PEAK():
    """A mode is a direction; the concept at that direction is the row carrying most of the mode's
    energy. Against the oracle the two answers are known in advance — `argmax(a)` and `argmax(b)` —
    computed from the generator, not from the read.

    Catches returning every row that routes to the dominant mode (~30 of the 40 here), or naming a
    mode by anything other than the row that actually carries it."""
    W, peak_u, peak_v = _two_subject_frame()
    rd = scr.readings(W)
    assert rd is not None
    assert len(rd) == 2, "two subjects, two readings — not one per row"
    assert [i for i, _e in rd] == [peak_u, peak_v], "each mode must be named ONCE, by its own peak"
    energies = [e for _i, e in rd]
    assert energies[0] > energies[1]                        # the stronger subject leads


def test_readings_are_ordered_by_ABSORBED_ENERGY_not_by_eigenvalue():
    """`principal_directions` orders by the correlation eigenvalue, which is scale-invariant — it
    discards exactly the magnitude that `frame(energy=…)` puts into the beam. Ordering the readings
    by what they absorb is the same choice as energising the frame at all.

    A check taken on a frame where the two orders agree — as they usually do — cannot fail. This
    seed is chosen because they disagree, and the assertion below states that disagreement first, so
    the test cannot pass by the orders happening to coincide."""
    from ember.optics import principal_directions
    W, _pu, _pv = _two_subject_frame(seed=34)
    P = principal_directions(W)
    native = [float(e) for e in ((W @ P) ** 2).sum(axis=0)]
    assert P.shape[1] == 2 and native[0] < native[1], (
        "this frame must be one where eigen-order and energy-order DISAGREE, else the test is "
        "vacuous (native=%r)" % (native,))
    rd = scr.readings(W)
    energies = [e for _i, e in rd]
    assert energies == sorted(energies, reverse=True)
    assert energies[0] == pytest.approx(native[1]), "the stronger mode must lead, not the first one"


def test_readings_SEPARATE_WHERE_THE_LOADING_COLUMN_CANNOT():
    """On a frame whose ground truth is constructed by `_two_subject_frame`.

    Ranking rows by their loading on the dominant mode mixes a row's energy with its alignment to
    that one direction, so two genuinely separable subjects can still produce a loading order with
    no separable break: `prism.resolution.signal_end` finds no more structure in it than a null
    model already explains, `separated()` reports no signal, and every row comes back
    undifferentiated.

    This test constructs a frame that demonstrably contains two separable subjects and shows the
    mode read separates them where the loading-column ranking cannot. The loading-column path is
    recomputed inside the assertion, so this compares two live reads rather than appealing to a
    stored result."""
    from prism import resolution as _res
    from ember.optics import beam as _as_beam
    W, peak_u, peak_v = _two_subject_frame()
    b = _as_beam(W)
    assert b is not None and b.modes
    load = np.abs(np.asarray(b.modes[0].profile)[:, 0])
    order = np.argsort(-load)
    old_k = _res.signal_end([float(load[i]) for i in order], frame=W)
    assert old_k == W.shape[0], (
        "the retired cut must be shown FAILING here, or this proves nothing (old=%r)" % (old_k,))

    rd = scr.readings(W)
    new_k = _res.signal_end([e for _i, e in rd])
    assert [i for i, _e in rd] == [peak_u, peak_v]
    assert new_k == len(rd) == 2 < old_k, (
        "the mode read must resolve TWO named subjects where the loading column could not separate "
        "at all (old=%r new=%r)" % (old_k, new_k))
    # `signal_end` declines to cut two modes, which is the right answer: `_null_separability(2)` is
    # 1.0, so no 2-point series can beat its own null. Both subjects stand. The gain is not that the
    # cut got sharper — it is that the thing being cut is now two readings instead of forty anonymous
    # rows.
    assert not _res.separated([e for _i, e in rd])


def test_readings_returns_NONE_not_an_empty_list_when_unreadable():
    """`None` (the frame carried no read) and `[]` (read, nothing resolved) are different answers
    and a caller must be able to tell them apart — the same rule `absorb_transmit` follows."""
    assert scr.readings(np.zeros((8, 16))) is None          # nothing to resolve
    assert scr.readings(np.ones((2, 16))) is None           # below the optics minimum


# ── The fork — membrane -> False, instrument -> True, and the choice is said ───────────────────────
class _BasisStore:
    """A store holding exactly one thing: a corpus basis. `head_of` is the lineage resolution
    `projection` uses; `content_type_mark` is the freshness gate. Nothing else is needed to make
    `frame(corpus_basis=True)` take its real arm rather than the empty-column fallback."""

    def __init__(self, B):
        self.doc = {"id": "geom.corpus-basis~stub", "root_id": "geom.corpus-basis",
                    "content_type": scr.BASIS_CONTENT_TYPE, "state": "committed",
                    "k": int(B.shape[0]), "d": int(B.shape[1]),
                    "basis": [[float(x) for x in row] for row in B]}
        self.artifacts = self

    def head_of(self, root_id):
        return self.doc if root_id == "geom.corpus-basis" else None

    def content_type_mark(self, content_type, cap=2000):
        return (1, 1, True) if content_type == scr.BASIS_CONTENT_TYPE else (0, 0, True)


@pytest.fixture()
def basis_store(stubbed):
    """An orthonormal `(k, D)` corpus basis spanning a strict subspace of the 2048-wide coordinate,
    which is the live shape: 280 directions out of 2048, orthonormal to 2.3e-15."""
    scr._BASIS_CACHE.clear()
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.normal(size=(2048, 24)))
    store = _BasisStore(Q.T)
    yield store, stubbed
    scr._BASIS_CACHE.clear()


def test_the_instrument_path_can_STATE_which_basis_it_is_reading(basis_store):
    """`read`, `resolvable` and `coherent` take a `corpus_basis` parameter, so the two callers that
    reach the instrument through them — `activation.output_membrane` and `lumen.output_screen.generate`
    — can state which basis they are reading rather than defaulting to one silently. A ruling nobody
    can state at the call site is a default, not a decision (PLAN §1.5).

    The two coordinates must give different feature widths here, or a `corpus_basis` parameter that
    was accepted and ignored would pass this test unchanged."""
    store, names = basis_store
    on = scr.read(store, names, corpus_basis=True)
    off = scr.read(store, names, corpus_basis=False)
    assert on["frame"] is True and off["frame"] is True
    assert on["features"] == 24, "the frame was not projected onto the corpus basis"
    assert off["features"] == 2048, "the raw dense coordinate must keep every column"
    assert on["corpus_basis"] is True and off["corpus_basis"] is False
    # the same fork reaches the other two instrument entry points
    assert scr.resolvable(store, names, corpus_basis=False) is not None
    assert scr.coherent(store, names, corpus_basis=False) in (True, False, None)
    # an absence still reports which coordinate was asked for — and no zoom, because none was taken
    assert scr.read(store, names[:1], corpus_basis=False) == {
        "frame": False, "rows": 0, "k_signal": None, "corpus_basis": False, "basis_k": None}


def test_the_instrument_basis_DISCARDS_the_out_of_span_energy_and_that_is_the_price(basis_store):
    """`B` is orthonormal and spans a strict subspace, so `W @ B.T` keeps only the in-span energy. On
    the live corpus over nine conversational queries, retained energy ranges 0.449–0.620, mean 0.542
    — 45.8% of every conversation frame's energy is dropped at that line, and PLAN §1.5 rules it
    right anyway because the raw frame is unreadable by the instrument at (T≈14–81, F=2048).

    The control is the membrane arm: it must retain the energy exactly. If both arms lost energy the
    measurement would be of the stub, not of the projection."""
    store, names = basis_store
    raw = scr.frame(store, names, corpus_basis=False)
    prj = scr.frame(store, names, corpus_basis=True)
    e_raw, e_prj = float((raw ** 2).sum()), float((prj ** 2).sum())
    assert e_raw > 0.0
    assert e_prj < e_raw, "the projection lost nothing — it is not projecting onto a subspace"
    # the membrane arm is lossless by construction: it is the coordinate itself, untouched.
    solo = scr.frame(store, names, corpus_basis=False)
    assert float((solo ** 2).sum()) == pytest.approx(e_raw)
    # and the loss is exactly the out-of-span energy, not an artefact of the frame builder
    B = np.asarray(store.doc["basis"])
    assert e_prj == pytest.approx(float(((raw @ B.T) ** 2).sum()))
