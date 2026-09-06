"""The decay is measured per screen; where it is not measured, there is no reading to report.

Every claim that could pass vacuously carries its negative control.
"""
import numpy as np
import pytest

from ember.signal import forgetting as F


# ── the module carries no typed decay rate, under any name ───────────────────────────────────────
def test_the_typed_seed_timescales_are_DELETED():
    """The module holds no default decay rate under any name.

    A typed pair such as `TAU_FAST_SEED = 1.6` / `TAU_SLOW_SEED = 40.0` would assert one pair of
    timescales for every screen there will ever be. Measured in `geom.corpus-basis` the values are
    per stream: 0.340/91.04, 0.640/14.24 and 0.285/81.56 — three streams, three answers, ratios
    268/22/286 against a single typed 25."""
    assert not hasattr(F, "TAU_FAST_SEED")
    assert not hasattr(F, "TAU_SLOW_SEED")


def test_FLOOR_and_the_read_cutoffs_are_DELETED():
    """The is/was boundary, the residual height and both `read()` cutoffs are read off the screen,
    so none of them is a typed number under any name.

    A single `FLOOR` would be three chosen things at once. Measured against a screen's own
    `tau_fast = 0.5770`, a floor of 0.25 is crossed by the fast band at dt = 0.80 ticks, so the
    two-timescale model collapses to `0.25*exp(-dt/tau_slow)` at every lag anyone reads and the floor
    supplies the whole height: 0.2269 at dt=1 where the screen's measured curve reads 0.0021.

    This walks the AST rather than the text, because such numbers live as literals inside
    expressions (`a <= 0.02`, `e < 1e-3`, `a >= FLOOR`) rather than as module attributes: a `hasattr`
    check alone would be a check that cannot fail, and a text grep would trip over prose."""
    import ast
    import inspect
    assert not hasattr(F, "FLOOR")
    assert not hasattr(F, "RESIDUAL_A")

    floats = {n.value for n in ast.walk(ast.parse(inspect.getsource(F)))
              if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    # The invariant, stated as a set: the only float literals this module executes are the identity
    # and the zero of an amplitude. Every tuning number is excluded by construction rather than by a
    # list of known names that the next constant would walk straight past.
    assert floats <= {0.0, 1.0}, "typed float literal(s) back in forgetting.py: %r" % (
        floats - {0.0, 1.0})
    assert floats, "the AST walk found no float literals at all — the check would pass vacuously"


# ── capacity is the resource envelope, not a cognitive parameter ─────────────────────────────────
def test_CAPACITY_SEED_is_DELETED_and_no_module_level_number_replaced_it():
    """No module-level binding holds a working-memory size under any name.

    How many traces fit is a property of the box, and the box can be asked. There is no reading of
    the screen that says how much an agent should hold, so a typed capacity would be a number nobody
    measured.

    The float-literal invariant above cannot see an `int`, so this walks the module's top-level
    bindings and requires that none of them binds a number at all — the shape a typed capacity, floor
    or timescale would have."""
    import ast
    import inspect
    assert not hasattr(F, "CAPACITY_SEED")

    tree = ast.parse(inspect.getsource(F))
    bound = []
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in targets:
            if isinstance(t, ast.Name):
                bound.append((t.id, node.value))
    numeric = [n for n, v in bound
               if isinstance(v, ast.Constant) and isinstance(v.value, (int, float))
               and not isinstance(v.value, bool)]
    assert not numeric, "module-level typed number(s) back in forgetting.py: %r" % (numeric,)
    # The control for the control: the walk has to be able to see a top-level numeric binding, or it
    # is a check that cannot fail. The module has none, so the machinery is exercised on a probe
    # rather than asserting on an empty list forever.
    probe = ast.parse("CAPACITY_SEED = 2048\n")
    assert [t.id for n in probe.body for t in n.targets
            if isinstance(n, ast.Assign) and isinstance(t, ast.Name)
            and isinstance(n.value, ast.Constant)] == ["CAPACITY_SEED"]


def test_capacity_IS_the_measured_envelope_over_the_measured_cost():
    """Capacity is `available bytes // bytes per trace` exactly: neither half substituted, rounded to
    a nicer number, nor clamped.

    Both halves are readings, so the test dictates both and demands the exact quotient. The worked
    case: available 7,704,268,800 B, cost 18,624 B (a 16,384 B `dense_vec` row at D=2048 plus its
    2,240 B row after `geom.corpus-basis` projects it, both matrices alive at once inside `_fit`)
    gives 413,674."""
    from prism import envelope as env

    scr = F.Screen(ic={}, witness="w", store=None)
    scr._cost_bytes = 18_624
    scr._cap_read = False
    orig = env.mem_available_bytes
    try:
        env.mem_available_bytes = lambda: 7_704_268_800
        assert scr.measured_capacity() == 413_674
        assert scr.capacity_source == "measured"
        # a smaller box measures a smaller working memory — nothing here is pinned to this machine
        env.mem_available_bytes = lambda: 4 << 30
        assert scr.measured_capacity() == (4 << 30) // 18_624
    finally:
        env.mem_available_bytes = orig


def test_an_UNMEASURED_envelope_leaves_the_screen_UNBOUNDED_and_never_at_a_default():
    """A platform that reports no envelope leaves the screen unbounded. A substituted capacity would
    be a number nobody measured, published to peers as though it had been.

    Unbounded is the honest reading: no bound was measured, so no bound is applied."""
    from prism import envelope as env

    scr = F.Screen(ic={}, witness="w", store=None)
    scr._cost_bytes = 1024
    orig = env.mem_available_bytes
    try:
        env.mem_available_bytes = lambda: None
        assert scr.measured_capacity() is None
        assert scr.capacity_source == "unmeasured"
        for t in range(500):
            scr.observe("c%d" % t, tick=t, witness="w")
        assert len(scr.traces) == 500 and scr.evicted == 0     # nothing was forgotten
    finally:
        env.mem_available_bytes = orig


def test_a_MEASURED_envelope_DOES_evict_so_the_control_above_is_not_vacuous():
    """The negative control for the test above. Dead eviction would make "unbounded" indistinguishable
    from "the bound was never wired".

    It also pins what a bound is for: an unbounded screen grows without limit and `read()` is O(all
    history). An envelope-derived bound is finite, and it is exactly the count that fits."""
    from prism import envelope as env

    scr = F.Screen(ic={}, witness="w", store=None)
    scr._cost_bytes = 1024
    orig = env.mem_available_bytes
    try:
        env.mem_available_bytes = lambda: 10 * 1024               # room for exactly 10
        for t in range(25):
            scr.observe("c%d" % t, tick=t, witness="w")
        assert len(scr.traces) == 10
        assert scr.evicted == 15
        # the coldest goes first — the module's own decay model, not an arbitrary trim
        assert [tr.tick for tr in scr.traces] == list(range(15, 25))
    finally:
        env.mem_available_bytes = orig


def test_the_trace_cost_is_READ_OFF_A_COORDINATE_and_never_computed_from_D():
    """The per-trace cost is read off a coordinate, never derived from a `D` written down somewhere:
    a node at D=256 and one at D=2048 each measure their own row without being told which they are.

    The store half is dictated too. `_fit` holds `W` and `W @ B.T` at once, so the cost counts both;
    charging only for the raw row would let the screen fill past what the read survives."""
    import numpy as np

    class _B:
        pass

    big = np.zeros(2048)
    small = np.zeros(256)
    seen = {"v": big}
    orig_dense = F.geo.dense_vec
    orig_basis = None
    try:
        F.geo.dense_vec = lambda cpt, ic: seen["v"]
        assert F.trace_bytes("c", {}) == 16_384                  # 2048 float64
        seen["v"] = small
        assert F.trace_bytes("c", {}) == 2_048                   # a Pi measures its own
        seen["v"] = big
        from ember.signal import projection as pj
        orig_basis = pj.load_basis
        pj.load_basis = lambda store: np.zeros((280, 2048))
        assert F.trace_bytes("c", {}, store=_B()) == 16_384 + 280 * 8
    finally:
        F.geo.dense_vec = orig_dense
        if orig_basis is not None:
            from ember.signal import projection as pj
            pj.load_basis = orig_basis


def test_no_coordinate_means_NO_COST_hence_no_bound_rather_than_a_substituted_one():
    """The negative control for the test above. A substituted cost would publish a capacity for a
    screen nobody could price. A screen whose concepts carry no geometry never builds those matrices
    (`_fit` returns None), so there is nothing to charge for and no cost to report."""
    orig = F.geo.dense_vec
    try:
        F.geo.dense_vec = lambda cpt, ic: (_ for _ in ()).throw(KeyError("no geometry"))
        assert F.trace_bytes("c", {}) is None
        F.geo.dense_vec = lambda cpt, ic: []
        assert F.trace_bytes("c", {}) is None
    finally:
        F.geo.dense_vec = orig


def test_a_DECLARED_capacity_is_never_reported_as_a_measured_one():
    """A caller-supplied working set is labelled `declared`. The two are different claims — one is a
    statement about this box, the other about the caller — and a bound whose provenance is stated is
    one a reader can tell apart from a typed number."""
    class _C:
        def __init__(self, n):
            self._n = n

        def name(self):
            return self._n

    scr = F.Screen(ic={}, witness="w", capacity=5)
    assert scr.capacity == 5 and scr.capacity_source == "declared"
    for t in range(9):
        scr.observe(_C("c%d" % t), tick=t, witness="w")
    assert len(scr.traces) == 5 and scr.evicted == 4
    assert scr.serialize()["capacity_source"] == "declared"
    assert F.Screen(ic={}, witness="w").capacity is None         # control: the default measures


def test_vertex_field_reads_the_MEMORY_AMPLITUDE_it_used_to_cut(monkeypatch):
    """`activation.vertex_field` ranks on the memory amplitude it is given, with no cutoff of its
    own. A second copy of a threshold in another module is invisible to anyone auditing the first.

    The control is in the same assertion: a trace at exactly 0.0 is still dropped, because a zero is
    the absence of a reading rather than a small one — so this cannot pass by removing the filter
    wholesale."""
    from ember.ontology import activation as A

    class _C:
        def __init__(self, n):
            self._n = n

        def name(self):
            return self._n

    class _Tr:
        def __init__(self, n, a):
            self.concept, self._a = _C(n), a

        def amplitude(self, now):
            return self._a

    class _Scr:
        traces = [_Tr("dog.n.01", 0.004), _Tr("cat.n.01", 0.0)]

    class _D:
        tick = 0
        screen = _Scr()

        def load_obs(self):
            raise RuntimeError("no corpus in this test")

    names = [r["concept"] for r in A.vertex_field(_D(), [{"concept": "wolf.n.01",
                                                         "salience": 1.0}])]
    assert "wolf.n.01" in names                  # control: the incoming signal is there at all
    assert "dog.n.01" in names, "a measured memory amplitude was cut by a typed threshold"
    assert "cat.n.01" not in names, "a measured ZERO carries no information and must not enter"


# ── an unmeasured screen has no decay, and therefore no `was` ────────────────────────────────────
def test_an_unmeasured_trace_is_FULLY_PRESENT_and_never_a_typed_decay():
    """An unmeasured trace holds its energy at every `now`, and its tense is `is`.

    A screen that has not measured its decay has not measured a slow decay; it has measured nothing,
    and a cooled amplitude would assert cooling nobody observed."""
    tr = F.Trace(concept="x", tick=0, witness="w")
    assert tr.decay is None
    assert tr.amplitude(0) == 1.0
    assert tr.amplitude(10_000) == 1.0          # no decay, however long we wait
    assert tr.tense(10_000) == "is"


def test_a_MEASURED_curve_does_decay_so_the_control_above_is_not_vacuous():
    """The negative control for the test above. An amplitude of 1.0 even with a curve present would
    mean the decay is dead and the previous test passes for the wrong reason."""
    tr = F.Trace(concept="x", tick=0, witness="w")
    tr.decay = F.DecayCurve([1.0, 0.5, 0.25, 0.125])       # as if a screen had measured this
    assert tr.amplitude(0) == pytest.approx(1.0)
    assert tr.amplitude(2) == pytest.approx(0.25)
    assert tr.amplitude(10_000) == 0.0
    assert tr.tense(10_000) == "—"


def test_an_UNMEASURED_screen_holds_everything_as_is():
    """A screen that measured nothing carries no is/was split: every trace reads `is`.

    Without a curve every trace sits at its own energy; a bag of identical readings has zero
    between-class variance, and `signal_end` computes its own null, so it returns all of them."""
    scr = F.Screen(ic={}, witness="w", store=None)
    for t in range(6):
        scr.observe("c%d" % t, tick=t, witness="w")
    assert scr.decay_source == "unmeasured"
    lead = scr.lead(now=100)
    assert lead == 1.0
    assert all(tr.tense(100, lead=lead) == "is" for tr in scr.traces)


def test_the_is_was_cut_DOES_split_a_separated_screen():
    """The negative control for the test above. A `lead` that returned the smallest amplitude
    unconditionally would make every screen entirely `is` and the cut decorative.

    A screen holding one fresh trace and five deeply cooled ones separates cleanly, so the cut lands
    at 1 and the cooled traces read `was`."""
    scr = F.Screen(ic={}, witness="w", store=None)
    scr.decay = F.DecayCurve([1.0] + [1e-4] * 20)
    for t in range(6):
        scr.observe("c%d" % t, tick=t, witness="w")
    lead = scr.lead(now=5)
    bands = [tr.tense(5, lead=lead) for tr in scr.traces]
    assert bands.count("is") == 1 and bands[-1] == "is"     # only the trace at dt=0
    assert bands.count("was") == 5


def test_tense_is_read_off_the_SCREEN_not_off_a_constant():
    """Tense is relative to the screen the trace sits on, which is what a constant boundary cannot
    be: under a constant, one amplitude would always yield one tense.

    One amplitude, 0.30, read against two screens. Against a screen whose other traces sit far below
    it, 0.30 is the present; against a screen whose others sit above it, the same 0.30 is the past.
    Nothing about the trace changed."""
    tr = F.Trace(concept="x", tick=0, witness="w", energy=0.30)
    assert tr.amplitude(0) == pytest.approx(0.30)

    hot = F.Screen(ic={}, witness="w", store=None)
    hot.traces = [tr] + [F.Trace("y%d" % i, 0, "w", energy=1e-4) for i in range(5)]
    assert tr.tense(0, lead=hot.lead(0)) == "is"

    cold = F.Screen(ic={}, witness="w", store=None)
    cold.traces = [tr] + [F.Trace("y%d" % i, 0, "w", energy=1.0) for i in range(5)]
    assert tr.tense(0, lead=cold.lead(0)) == "was"


# ── the curve is a survival fraction: monotone, in [0,1], and it says so ─────────────────────────
class _OscOperator:
    """An operator whose reconstructed C(tau) oscillates — the phase-flipping case, as measured on a
    4-concept delegate stream (`C(1) = -0.4296`, `C(3) = +0.6685`)."""

    def __init__(self, values):
        self._v = list(values)

    def forgetting(self):
        return {"margin": 0.9, "forgets": True, "n_modes": 2}

    def reconstruct_decay(self, max_lag):
        out = list(self._v[:int(max_lag)])
        out += [0.0] * (int(max_lag) - len(out))
        return np.asarray(out, dtype=float)


def test_an_ANTICORRELATED_lag_is_not_a_negative_amplitude_and_the_clip_is_counted():
    """A negative autocorrelation reads as 0, and the clip is counted on the curve.

    `C` is an autocorrelation: a 15-concept screen reads `C(1) = -0.0763`, a 4-concept delegate
    screen `C(1) = -0.4296`. Anti-correlation is not negative presence — an amplitude is a surviving
    fraction — and the count of clipped lags is on the curve, where a reader can see it."""
    op = _OscOperator([1.0, -0.4296, -0.3957, 0.6685, -0.2717, 0.0])
    c = F._decay_from(op, span=5)
    assert c is not None
    assert min(c.values) >= 0.0 and max(c.values) <= 1.0
    assert c.clipped >= 3                                   # the three negative lags, counted


def test_the_curve_is_MONOTONE_so_a_was_is_never_stronger_than_the_is_before_it():
    """The reconstructed curve never rises, so a `was` is never stronger than the `is` before it.

    Clamping the measured curve at 0 alone is not enough: on a 4-concept stream the clamped curve
    reads 1.00, 0.00, 0.00, 0.669, 0.00, 0.00, 0.447 — a trace stronger at tick 3 than at tick 1.
    The zig-zag is phase (`arg(mu) = pi`), and an amplitude read takes the envelope."""
    op = _OscOperator([1.0, -0.4296, -0.3957, 0.6685, -0.2717, -0.2778, 0.4465, 0.0, 0.0])
    c = F._decay_from(op, span=8)
    v = c.values
    assert v[0] == pytest.approx(1.0)
    assert all(v[i + 1] <= v[i] + 0.0 for i in range(len(v) - 1)), v
    assert v[1] == pytest.approx(0.6685)                    # the envelope, not the zero


def test_the_monotone_hull_changes_NOTHING_on_an_already_decaying_curve():
    """The negative control for the test above. The monotone hull is an envelope, not a smoothing
    that rewrites readings: on an already-decaying curve it is identical to the clamped curve for
    all seven lags."""
    raw = [1.0, 0.4922, 0.2276, 0.0975, 0.03749, 0.01185, 0.002025]
    c = F._decay_from(_OscOperator(raw + [0.0] * 8), span=6)
    assert c.values == pytest.approx(raw)
    assert c.clipped == 0


def test_a_curve_extends_itself_rather_than_holding_at_its_last_value():
    """A read past the reconstructed window extends the curve from the operator. Clamping to the
    final value would be a floor by another name.

    The curve is built for span 2 and then asked for lag 5."""
    op = _OscOperator([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125] + [0.0] * 60)
    c = F._decay_from(op, span=2)
    assert len(c) == 3
    assert c.at(5) == pytest.approx(0.03125)                 # extended, not held at 0.25
    assert len(c) > 3


def test_a_curve_with_NO_operator_does_not_fabricate_past_its_window():
    """The negative control for the test above. A curve restored from disk carries no operator, so
    there is nothing to extend from and nothing past its window to give."""
    c = F.DecayCurve([1.0, 0.5, 0.25])
    assert c.at(2) == pytest.approx(0.25)
    assert c.at(3) == 0.0
    assert len(c) == 3


def test_a_non_forgetting_operator_is_reported_as_not_forgetting():
    """A persistent unit-circle mode is reported as it was fitted, not coerced into a decay. A stream
    of the same concept twelve times fits `margin = 1.0000` and `C(tau) = 1` at every lag: the screen
    does not forget, and a cooled curve would be cooling nobody observed."""
    class _Persistent(_OscOperator):
        def forgetting(self):
            return {"margin": 1.0, "forgets": False, "n_modes": 1}

    c = F._decay_from(_Persistent([1.0] * 40), span=20)
    assert c.margin == 1.0
    assert all(v == pytest.approx(1.0) for v in c.values)


# ── where there is no reading, the instrument is why — not a typed rule ──────────────────────────
class _Rates:
    def __init__(self, short, long):
        self.short_range, self.long_range = short, long


class _FakeDynamics:
    """Stands in for the entroptics operator so the rate spread can be dictated exactly."""
    spread = (2.0, 1.0)                          # (short_range, long_range) -> two timescales

    def __init__(self, n):
        self.n_pairs = 5

    def update_block(self, W):
        pass

    def rates(self):
        return _Rates(*self.spread)


class _FakeStore:
    pass


@pytest.fixture
def one_mode_rig(monkeypatch):
    """A store + basis + operator whose rate spread the test controls."""
    import ember.optics
    from ember.signal import projection

    monkeypatch.setattr(ember.optics, "dynamics_state", _FakeDynamics)
    monkeypatch.setattr(projection, "load_basis", lambda store: np.eye(3))
    monkeypatch.setattr(F.geo, "dense_vec", lambda cpt, ic: np.array([1.0, 2.0, 3.0]) * (cpt + 1))
    return [0, 1, 2, 3]


def test_ONE_resolved_mode_is_refused_rather_than_reported_as_two_timescales(one_mode_rig,
                                                                            monkeypatch):
    """`measured_rates` gives no reading when the operator resolves a single mode.

    `short_range = max(alpha)` and `long_range = min(alpha)`, so a one-mode operator hands back the
    same number twice, and a pair whose halves are equal is not two timescales. A delegate screen
    observing one concept per turn produces exactly that: `1.122/1.122` at four traces and
    `0.5507/0.5507` at seven."""
    monkeypatch.setattr(_FakeDynamics, "spread", (1.0, 1.0))     # one mode: max(alpha)==min(alpha)
    assert F.measured_rates(one_mode_rig, {}, store=_FakeStore()) is None


def test_TWO_resolved_modes_ARE_reported_so_the_refusal_is_not_blanket(one_mode_rig, monkeypatch):
    """The negative control for the test above. If the one-mode case swallowed genuine two-timescale
    readings too, `rates_source` would be permanently "unmeasured" and look identical to the
    measurement never having been wired."""
    monkeypatch.setattr(_FakeDynamics, "spread", (2.0, 0.5))     # alpha 2.0 and 0.5 -> tau 0.5, 2.0
    got = F.measured_rates(one_mode_rig, {}, store=_FakeStore())
    assert got is not None
    tau_fast, tau_slow = got
    assert tau_fast == pytest.approx(0.5)
    assert tau_slow == pytest.approx(2.0)
    assert tau_fast < tau_slow


def test_GROWTH_is_refused_because_a_diverging_screen_has_no_forgetting_time(one_mode_rig,
                                                                            monkeypatch):
    """A diverging screen has no forgetting time, so there is no pair to give. `alpha <= 0` is
    growth, not decay, and inverting it yields a negative tau: a delay lift at d>=3 drives `tau_slow`
    to -288, -1381 and -2789."""
    monkeypatch.setattr(_FakeDynamics, "spread", (2.0, -0.5))
    assert F.measured_rates(one_mode_rig, {}, store=_FakeStore()) is None


def test_a_basis_of_the_WRONG_WIDTH_is_refused_not_reshaped(one_mode_rig, monkeypatch):
    """A basis whose width does not match the frame yields no reading, and the frame is not reshaped
    to fit it. Projecting a frame through a basis of another generation's width measures something
    other than this frame."""
    from ember.signal import projection
    monkeypatch.setattr(projection, "load_basis", lambda store: np.eye(7))   # frame is 3-wide
    assert F.measured_rates(one_mode_rig, {}, store=_FakeStore()) is None
    assert F.measured_decay(one_mode_rig, {}, store=_FakeStore()) is None


def test_the_CURVE_and_the_RATES_refuse_INDEPENDENTLY(one_mode_rig, monkeypatch):
    """The curve and the rates are two readings, and one being unavailable says nothing about the
    other.

    A one-mode operator has no fast-versus-slow to report, so `measured_rates` gives nothing — but it
    does have a decay, `C(tau) = P mu^tau`, and that curve is a real measurement in its own right.
    Conversely, a fit the rates accept still yields a curve."""
    class _WithCurve(_FakeDynamics):
        spread = (1.0, 1.0)                                  # one mode -> no rate pair

        def forgetting(self):
            return {"margin": 0.5, "forgets": True, "n_modes": 1}

        def reconstruct_decay(self, max_lag):
            return np.asarray([0.5 ** t for t in range(int(max_lag))], dtype=float)

    import ember.optics
    monkeypatch.setattr(ember.optics, "dynamics_state", _WithCurve)
    assert F.measured_rates(one_mode_rig, {}, store=_FakeStore()) is None
    c = F.measured_decay(one_mode_rig, {}, store=_FakeStore(), span=3)
    assert c is not None and c.values == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_an_operator_that_cannot_reconstruct_yields_NO_curve_and_never_a_default(one_mode_rig):
    """The negative control for the test above. `_FakeDynamics` has no `reconstruct_decay` at all,
    so there is no curve to give and none is synthesised."""
    assert F.measured_decay(one_mode_rig, {}, store=_FakeStore()) is None


def test_no_store_means_no_measurement_and_never_a_default():
    """A screen with no store reports "unmeasured".

    The fit is taken in `geom.corpus-basis`, which is an artifact: without a store there is nothing
    to project onto, and in the raw dense coordinate the fit is degenerate at every N. A screen that
    cannot reach a basis has not measured its decay."""
    scr = F.Screen(ic={}, witness="w", store=None)
    assert scr.rates_source == "unmeasured" and scr.decay_source == "unmeasured"
    assert scr.measure() == "unmeasured"
    assert scr.tau_fast is None and scr.tau_slow is None and scr.decay is None


# ── persistence: a measurement that survives to disk must survive coming back ────────────────────
def test_restore_KEEPS_the_measured_timescales_and_the_measured_CURVE():
    """A measurement that reaches the disk survives the trip back. `serialize()` writes the
    timescales and the curve, and `restore()` reads both: a screen that has measured comes back
    measured, with its `margin` and `clipped` count intact."""
    scr = F.Screen(ic={}, witness="w")
    scr.tau_fast, scr.tau_slow, scr.rates_source = 0.34, 91.04, "measured"
    scr.decay = F.DecayCurve([1.0, 0.4922, 0.2276], margin=0.507, clipped=2)
    scr.decay_source = "measured"
    doc = scr.serialize()
    assert doc["rates_source"] == "measured" and doc["tau_fast"] == 0.34
    assert doc["decay"] == [1.0, 0.4922, 0.2276] and doc["decay_source"] == "measured"

    back = F.Screen(ic={}, witness="w")
    assert back.rates_source == "unmeasured" and back.decay_source == "unmeasured"   # control
    back.restore(doc, witness="w")
    assert back.rates_source == "measured"
    assert back.tau_fast == 0.34 and back.tau_slow == 91.04
    assert back.decay_source == "measured"
    assert back.decay.tolist() == [1.0, 0.4922, 0.2276]
    assert back.decay.margin == pytest.approx(0.507) and back.decay.clipped == 2


def test_restore_does_not_INVENT_a_curve_that_was_never_written():
    """The negative control for the test above. A doc carrying no decay restores to a screen with
    none: the persistence path is a carrier for readings, not a source of them."""
    back = F.Screen(ic={}, witness="w")
    back.restore({"traces": []}, witness="w")
    assert back.rates_source == "unmeasured" and back.decay_source == "unmeasured"
    assert back.tau_fast is None and back.tau_slow is None and back.decay is None


# ── the re-measure trigger is memoisation, not a cadence ─────────────────────────────────────────
def test_the_decay_is_reread_when_the_TRACE_BAG_MOVES_and_not_on_a_schedule():
    """The decay is a function of the trace bag, so the trigger to re-read it is that the bag is no
    longer what it was read from. `(len(traces), evicted)` is the mutation witness the screen already
    keeps; a "measure every N turns" cadence would re-read on a clock instead."""
    scr = F.Screen(ic={}, witness="w", store=None)
    calls = []
    real = scr.measure
    scr.measure = lambda *a, **k: (calls.append(1), real(*a, **k))[1]

    scr._rates_current()
    assert len(calls) == 1                       # never measured -> measure
    scr._rates_current()
    assert len(calls) == 1                       # bag unchanged -> reuse, no schedule fires

    scr.traces.append(F.Trace(concept="x", tick=1, witness="w"))
    scr._rates_current()
    assert len(calls) == 2                       # bag moved -> the old reading describes nothing


def test_an_ADVANCING_CLOCK_alone_does_not_retrigger_the_fit():
    """The negative control for the test above. `now` stays out of the memo key: were it in, every
    read of an idle screen would re-fit the operator — a cadence wearing a clock."""
    scr = F.Screen(ic={}, witness="w", store=None)
    calls = []
    real = scr.measure
    scr.measure = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    scr._rates_current(now=0)
    scr._rates_current(now=1_000)
    scr._rates_current(now=1_000_000)
    assert len(calls) == 1


def test_the_reconstruction_window_is_the_BAG_S_OWN_SPAN_not_a_constant():
    """The reconstruction window is the demand — `now - oldest tick` — read off the bag rather than a
    fixed lag count, and `DecayCurve` extends itself if a later read reaches further."""
    scr = F.Screen(ic={}, witness="w", store=None)
    assert scr._span() == 0                                   # empty bag: no lags at all
    scr.observe("a", tick=3, witness="w")
    scr.observe("b", tick=9, witness="w")
    assert scr._span() == 6                                   # 9 - 3, from the traces
    assert scr._span(now=40) == 37                            # 40 - 3, from the read
    assert scr._span(now=0) == 0                              # a backwards clock never goes negative
