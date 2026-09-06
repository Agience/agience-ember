"""`op.measure` — a state reading placed as a signal, and the coupling that reads it.

These pin the properties the design stands on, each stated as a failure mode first:

  · a threshold could creep back in — so the shortfall is checked to be a subtraction that is
    continuous through the floor, with no step at the crossing;
  · an unmeasured reading could be read as a measured zero — the `.get("grounded", True)` shape;
  · the frame could be built in the corpus basis, where a tekton's coupling basis cannot reach it,
    so `absorb_transmit` yields None — indistinguishable at the call site from "nothing coupled";
  · and `k` could be mistaken for a measurement of the incident signal. On the basis path it is the
    offer's rank and is identical in every regime. That negative result is pinned here so that a
    signal-dependent `k` in `absorb_transmit` turns this red and the coupling story gets re-read
    rather than silently changing meaning.

Hermetic: the ontology coordinate is stubbed exactly as `test_projection.py` stubs it, and the
envelope is measured against a real `tmp_path`. Nothing here reads the live corpus.
"""
import numpy as np
import pytest

from ember import optics                      # the instrument — ember holds it
from prism import conservation, frames       # the wire — prism holds it
from ember.signal import state as S


class _Syn:
    def __init__(self, name, vec):
        self._n, self.vec = name, vec

    def name(self):
        return self._n


D = 2048


@pytest.fixture()
def coords(monkeypatch):
    """A sparse 2048-wide coordinate for each reading's concept — the real one's shape (~6 nnz)."""
    rng = np.random.default_rng(11)
    vecs = {}
    for syn, _key in S.READINGS:
        v = np.zeros(D)
        v[rng.choice(D, 6, replace=False)] = rng.normal(size=6)
        vecs[syn] = v / np.linalg.norm(v)
    from crystal.ontology import geometry as g
    from crystal.ontology import driver as wn
    monkeypatch.setattr(wn, "synset", lambda n: (_Syn(n, vecs[n]) if n in vecs else None))
    monkeypatch.setattr(g, "load_ic", lambda: None)
    monkeypatch.setattr(g, "dense_vec", lambda s, ic, center=None: s.vec)
    return vecs


@pytest.fixture()
def node(tmp_path, monkeypatch):
    """A store root with a lattice file, and a declared floor read from the environment, which is
    where the deployed policy lives (`_fleet/peers/71/ember/serve.env`: keep >=20GB free on D:)."""
    (tmp_path / "lattice.db").write_bytes(b"x" * 4096)
    (tmp_path / "cas").mkdir()
    (tmp_path / "cas" / "aa").mkdir()
    (tmp_path / "cas" / "aa" / "blob").write_bytes(b"y" * 2048)
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(tmp_path))
    monkeypatch.setenv("EMBER_SQLITE_DB", "lattice.db")
    monkeypatch.setenv("MANTLE_CACHE_MIN_FREE_GB", "20")
    monkeypatch.delenv("EMBER_CACHE_MIN_FREE_GB", raising=False)
    return tmp_path


GB = 2 ** 30
FLOOR = 20 * GB


def _basis(coords, names):
    return frames.offer_basis(np.vstack([coords[n] for n in names]))


def test_the_shortfall_is_a_subtraction_and_has_no_step_at_the_floor(node):
    """The reading is `floor - free`, floored at zero because a surplus is not a negative demand;
    no question of the form "is headroom low?" is asked anywhere. Walking the free reading across
    the declared floor produces a continuous quantity: zero above it, rising linearly from zero
    below it, with no step at the crossing."""
    at = {}
    for free in (FLOOR + GB, FLOOR + 1, FLOOR, FLOOR - 1, FLOOR - GB, 0):
        at[free] = S.envelope(None, free_bytes=free)["shortfall_bytes"]
    assert at[FLOOR + GB] == 0 and at[FLOOR + 1] == 0 and at[FLOOR] == 0
    assert at[FLOOR - 1] == 1, "the first byte below the floor is a shortfall of ONE byte, not a state change"
    assert at[FLOOR - GB] == GB
    assert at[0] == FLOOR
    # continuity across the crossing: the step from just-above to just-below is one byte.
    assert at[FLOOR - 1] - at[FLOOR] == 1


def test_an_unmeasured_reading_is_not_a_measured_zero(node, monkeypatch):
    """[[absence-is-not-an-affirmative-claim]]. `disk_free_bytes` yields None when it could not
    measure, so a missing path and a full disk produce different output. The energy is 0.0 — an
    unmeasured row carries nothing and therefore couples to nothing — and the envelope still reports
    `free_measured: False`, so the two stay distinguishable downstream."""
    import prism.envelope as pe
    monkeypatch.setattr(pe, "disk_free_bytes", lambda p: None)
    env = S.envelope(None)
    assert env["free_measured"] is False
    assert env["free_bytes"] is None and env["shortfall_bytes"] is None
    e = dict(zip([k for _s, k in S.READINGS], S.energies(env)))
    assert e["shortfall_bytes"] == 0.0


def test_no_declared_floor_means_no_demand_never_an_invented_one(node, monkeypatch):
    """An unset policy stays unset. `content_cache._min_free_bytes()` returns 0 with no
    declaration, so there is no floor and therefore no shortfall. A default here would manufacture
    eviction pressure out of an absent policy."""
    monkeypatch.delenv("MANTLE_CACHE_MIN_FREE_GB", raising=False)
    monkeypatch.delenv("EMBER_CACHE_MIN_FREE_GB", raising=False)
    env = S.envelope(None, free_bytes=1)
    assert env["floor_bytes"] == 0 and env["floor_declared"] is False
    assert env["shortfall_bytes"] == 0


def test_the_footprint_is_measured_not_assumed(node):
    """The occupancy rows are real bytes off the real tree: the lattice file and the CAS blob."""
    env = S.envelope(None, free_bytes=FLOOR)
    assert env["lattice_bytes"] == 4096
    assert env["evictable_bytes"] == 2048 and env["cas_files"] == 1
    assert env["store_bytes"] == 4096 + 2048


def test_eviction_demand_is_bounded_by_what_exists_to_evict(node):
    """`eviction_bytes` is the demand a reclamation could ACTUALLY meet: the shortfall, bounded by
    the evictable mass. Both terms measured; the `min` is derived, not a chosen cap."""
    env = S.envelope(None, free_bytes=0)
    assert env["shortfall_bytes"] == FLOOR
    assert env["eviction_bytes"] == 2048, "a node with 2KB of cache cannot release 20GB"


def test_scarcity_is_amplitude_the_coupling_rises_with_the_deficit(node, coords):
    """Scarcity is amplitude. Ample headroom is a shortfall of zero, so the `headroom` row is scaled
    by sqrt(0) and carries nothing — present and silent. As the deficit grows the row's energy grows
    with it, so the fraction a reclamation offer absorbs rises monotonically. The mechanism is the
    scaling, with no comparison anywhere in it."""
    B = _basis(coords, ["eviction.n.01", "cache.n.01", "headroom.n.01"])
    fracs = []
    for free in (FLOOR + 10 * GB, FLOOR, FLOOR - GB, FLOOR - 5 * GB, FLOOR - 15 * GB, 0):
        W = S.place(None, S.envelope(None, free_bytes=free))
        assert W is not None and W.shape == (len(S.READINGS), D)
        a, t, _k = optics.absorb_transmit(W, basis=B)
        ei = conservation.energy(W)
        fracs.append(conservation.energy(a) / ei)
    assert fracs == sorted(fracs), "the coupled fraction must not fall as the deficit grows: %r" % fracs
    assert fracs[-1] > fracs[0] * 2, "the deficit did not change the coupling: %r" % fracs


def test_k_on_the_basis_path_is_the_OFFERS_RANK_not_a_read_of_the_signal(node, coords):
    """A negative result, pinned on purpose.

    `absorb_transmit(frame, basis=B)` returns `k = B.shape[1]` — the rank of the offer. It is a
    property of the tekton, identical for every incident frame, so a reading of the form "k > 0 when
    short, k == 0 when ample" is outside what this path expresses. The quantity that discriminates
    is the absorbed energy. Written as a test so the claim is checkable, and so a signal-dependent
    `k` turns this red rather than quietly changing what `k` means to every reader."""
    B = _basis(coords, ["eviction.n.01", "cache.n.01", "headroom.n.01"])
    ks = set()
    for free in (FLOOR + 10 * GB, 0):
        W = S.place(None, S.envelope(None, free_bytes=free))
        ks.add(optics.absorb_transmit(W, basis=B)[2])
    assert ks == {int(B.shape[1])}
    assert len(ks) == 1, "k varied with the signal — the coupling story needs re-reading, not this test"


def test_conservation_is_exact_across_the_membrane(node, coords):
    """`‖incident‖² = ‖absorbed‖² + ‖transmitted‖²` at the split, in both regimes. This is the
    correctness statement for the residual: a leak here is evidence that never gets asked about,
    and it looks exactly like a question nothing could answer."""
    B = _basis(coords, ["eviction.n.01", "cache.n.01", "headroom.n.01"])
    for free in (FLOOR + 10 * GB, 0):
        W = S.place(None, S.envelope(None, free_bytes=free))
        a, t, k = optics.absorb_transmit(W, basis=B)
        led = conservation.PathLedger(W, at="op.measure")
        led.absorb(a, t, at="op.reclaim", k=k)
        led.emit(at="residual")
        cert = led.certificate()
        assert cert["balanced"], cert["why"]
        assert abs(cert["loss"]) <= cert["tolerance"]


def test_the_frame_is_in_the_coordinate_a_coupling_basis_can_reach(node, coords, monkeypatch):
    """`place` hands back the raw dense coordinate, which is the width a coupling basis can reach.

    A corpus-projected frame is `(T, 195)` and a tekton's coupling basis is `(2048, k)`.
    `absorb_transmit` needs them to agree on the feature axis and yields None when they do not — an
    absent read, indistinguishable at the call site from a genuine non-coupling."""
    from ember.signal import projection
    B = _basis(coords, ["eviction.n.01", "headroom.n.01"])
    env = S.envelope(None, free_bytes=0)
    W = S.place(None, env)
    assert W.shape[1] == D
    assert optics.absorb_transmit(W, basis=B) is not None

    # A stand-in "corpus basis": three directions the screen occupies, so the projection is
    # non-degenerate. What matters is its width, which is 3 rather than D.
    corpus = np.vstack([coords[n] for n in ("headroom.n.01", "occupancy.n.01", "cache.n.01")])
    monkeypatch.setattr(projection, "load_basis", lambda store: corpus)
    Wc = projection.frame(None, [s for s, _ in S.READINGS], energy=S.energies(env),
                          corpus_basis=True)
    assert Wc is not None and Wc.shape[1] == 3
    assert optics.absorb_transmit(Wc, basis=B) is None, (
        "a corpus-projected frame silently failed to couple — that is the seam this fixes")


def test_measure_places_a_frame_it_does_not_return_a_verdict(node, coords):
    """`op.measure` hands back the frame, encoded, plus the readings that built it. The result
    carries no verdict, no severity and no 'should evict', and the module's source names no call to
    `op.reclaim`: deciding what to do with a reading belongs to whoever couples to it."""
    out = S.measure(None, free_bytes=0)
    assert out["placed"] is True and out["shape"] == [len(S.READINGS), D]
    W = frames.decode_frame(out[frames.FRAME_KEY])
    assert W is not None and W.shape == (len(S.READINGS), D)
    assert abs(conservation.energy(W) - out["incident_energy"]) < 1e-6
    assert {r["concept"] for r in out["rows"]} == {s for s, _ in S.READINGS}
    assert not any(k in out for k in ("verdict", "severity", "should_evict", "action"))
    import inspect
    src = inspect.getsource(S)
    assert "op.reclaim" not in src.replace("`op.reclaim`", ""), "op.measure must never call reclaim"
