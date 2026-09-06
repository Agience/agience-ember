"""Forgetting: tense is decay rather than deletion. There is no is_state/was_state flag — the
amplitude on a signal's own decay curve is its tense.

The picture: the dog is howling, then it is not, but it did, and that is another residual signal
representing state. I remember the dog howled because I was there. The footprint that represents `is`
disappears quickly, replaced by `was`.

An observation is one signal cooling on one measured curve:

  * `a(dt) = energy * C(dt)`, where `C` is this screen's own decay, rebuilt from its operator's
    spectrum (`Dynamics.reconstruct_decay` — `C(tau) = sum_k P_k mu_k^tau`, every resolved mode at
    its measured weight, normalised so `C(0) = 1`, extrapolating past the observed window).
  * `is` and `was` are two bands of that one curve, and the boundary between them is a cut on the
    screen's own ordered amplitudes (`prism.resolution.signal_end`) — the same instrument every other
    "how many of these are signal" question in this codebase answers with.

`is` and `was` are therefore one residual signal read at two rungs of the same forget, rather than two
facts or two content-types. Reading "is X happening?" queries the leading band; "did X happen?"
queries the tail.

The residual carries its witness ("I remember because I was there"), which is its provenance and
authority: a `was` is trustworthy to the degree it was witnessed, rather than because it was ever
asserted true.

    python -m ember.signal.forgetting     # watch `is` cool into a witnessed `was`

Discrete integer ticks — the substrate is the observation stream, not a wall clock; one tick = one
observation step.
"""
from __future__ import annotations

from typing import List, Optional

from prism import resolution as _res
from prism import law as _law

from crystal.ontology import geometry as geo

# ── the whole decay is measured per screen ──────────────────────────────────────────────────
# `Screen.measure()` fits one operator on this screen's own observation history — rows = ticks (a
# genuinely ordered axis), columns = the concept's JC coordinate, taken in `geom.corpus-basis` — and
# reads three things off it:
#
#   tau_fast  <- 1 / Dynamics.rates().short_range   fastest per-mode decay a_k = -log|mu_k|
#   tau_slow  <- 1 / Dynamics.rates().long_range    slowest
#   C(tau)    <- Dynamics.reconstruct_decay(...)    the whole decay curve, all resolved modes
#
# There is no floor, no seed timescale and no capacity constant in this module. Each would be a
# chosen number standing where a measurement belongs, and the measurements disagree with them.
#
# A crossover floor of 0.25 supplies the whole residual height rather than bounding it. On a
# 15-concept screen fitting in `geom.corpus-basis` (tau_fast 0.5770, tau_slow 2.0658) the crossing
# `tau_fast * ln(1/0.25)` lands at dt = 0.80 ticks, so for every lag anyone reads the "two-timescale"
# model is `0.25*exp(-dt/tau_slow)` and the floor is the entire model:
#
#     dt        1        2        3        5       10       20
#     floor  0.2269   0.1398   0.0862   0.0327   0.0029   2.3e-05
#     C(dt)  0.0021   0.0021   0.0021   0.0021   9.6e-05  1.6e-11
#
# — two orders of magnitude of memory that nothing measured.
#
# Seed timescales (1.6 fast, 40.0 slow) describe no real screen. Measured live, the timescales differ
# by an order of magnitude from screen to screen:
#
#     dog        tau_fast 0.340   tau_slow 91.04    ratio 268
#     water      tau_fast 0.640   tau_slow 14.24    ratio  22
#     physicist  tau_fast 0.285   tau_slow 81.56    ratio 286
#
# There is no replacement constant. A screen that cannot measure its decay has measured nothing
# rather than a slow decay, and reporting an unmeasured tense would assert cooling nobody observed
# ([[absence-is-not-an-affirmative-claim]]). An unmeasured screen therefore has no `was` band:
# everything it holds is `is`, at full amplitude, until the curve resolves — at which point the whole
# bag cools on its own measured curve. `decay_source` reports which state the screen is in.
#
# Amplitude cutoffs in `read()` — 0.02 on the amplitude, 1e-3 on the energy — are thresholds rather
# than display filters. On a capacity-bounded screen, measured on the live store:
#
#     now= 5   kept 15   dropped by a<=0.02:  0
#     now=10   kept 11   dropped by a<=0.02:  4
#     now=20   kept  1   dropped by a<=0.02: 14
#     now=40   kept  0   dropped by a<=0.02: 15
#
# and `e < 1e-3` can never fire, because `e = a*s` with `s = exp(-JC) > 0` is reachable only through
# an `a` the amplitude test already took. Nothing stands in their place: the amplitude already
# carries how present a trace is, so a separate cutoff on top of it would duplicate that test under
# another name. The only rows `read()` drops sit at amplitude exactly zero, which is the absence of
# a reading rather than a small one.

# ── capacity is the box's, and it is measured ────────────────────────────────────────────────
# There are exactly two legitimate external inputs to this module, and capacity is the first: how
# many traces this machine can hold and still read is a property of the machine rather than of the
# agent, and the box can be asked ([[associative-reach-bounded-by-envelope]]).
#
# Capacity is `prism.envelope.holds(cost)` — the measured available envelope divided by the measured
# cost of one trace. Both halves are read:
#
#     envelope   `mem_available_bytes()`  cgroup headroom / MemAvailable / MEMORYSTATUSEX
#     cost       `trace_bytes()`          the rows a read materialises for one trace
#
# Measured on one Windows box with no cgroup: available 7,704,268,800 B; one trace costs 16,384 B
# (its `dense_vec` row at D=2048) + 2,240 B (its row after `geom.corpus-basis` projects it, 280 wide)
# = 18,624 B, both matrices alive at once inside `_fit`, giving capacity = 413,674. A 4 GB Pi at
# D=256 measures its own, which is the point.
#
# An unmeasured platform leaves the screen unbounded rather than at a default
# ([[absence-is-not-an-affirmative-claim]]): a typed 2048 would be a claim about a box nobody read,
# the same shape as a fixed `8 * _GB` memory limit under which a 2 GB Pi and a 512 GB node report the
# same ceiling. `capacity_source` records which happened. An envelope-derived bound is still a bound
# — it is finite, and it is exactly the number of traces that fit before the read stops being
# possible — so it closes the unbounded growth that makes `read()` O(all history).
#
# Capacity and the per-delegate split land together. A bound on a shared screen would make agents
# compete for working memory, a busy delegate evicting a quiet one's traces, so bounding is safe
# because a screen belongs to exactly one delegate.

def trace_bytes(concept, ic, *, store=None, coordinate=None) -> Optional[int]:
    """What one trace costs this box to hold and still read. `None` when it cannot be measured.

    Two measured parts, read off real objects, neither computed from `D`:

      1. the row `_fit` materialises for it — `geo.dense_vec(concept, ic).nbytes`: 16,384 B at
         D=2048, 2,048 B on a Pi at D=256, and neither node was told which;
      2. the row that same trace occupies once `geom.corpus-basis` projects it
         (`B.shape[0] * B.itemsize` — 2,240 B at 280 wide). Both matrices are alive at once inside
         `_fit` (`W`, then `W @ B.T`), so a cost counting only one would let the screen fill past
         what the read can survive.

    The `Trace` object itself is not priced. It costs ~142 B against the ~18,600 B of rows beside it,
    and measuring it takes either a chosen sample size or `sys.getsizeof(tr.__dict__)`, which
    materialises the shared-key instance dict CPython was holding compactly and so changes the thing
    it measures.

    No coordinate means no measurement, hence no bound. A screen whose concepts carry no geometry
    never builds these matrices at all (`_fit` returns None), so there is nothing here to price and
    `None` travels out to `_evict` as unbounded."""
    # A screen watching units the reading formed — not synsets — has no coordinate under
    # `geo.dense_vec`, so this returns None, `measured_capacity` reports "unmeasured", and `_evict`
    # treats the screen as unbounded: an unpriced trace cannot be bounded. The `coordinate` seam is
    # what supplies a price in that case, and it answers the same missing-geometry question the decay
    # fit answers via the same seam in `_fit`.
    try:
        import numpy as np
        v = np.asarray(coordinate(concept) if coordinate is not None
                       else geo.dense_vec(concept, ic))
    except Exception:
        return None
    if not v.size:
        return None
    total = int(v.nbytes)
    # A caller-supplied coordinate is already in its own basis (see `_fit`), so `W @ B.T` never runs
    # for it and there is no second matrix alive beside the first to price.
    if coordinate is None and store is not None:
        try:
            from ember.signal import projection as _pj
            B = np.asarray(_pj.load_basis(store))
            if B.ndim == 2 and B.shape[1] == v.size:
                total += int(B.shape[0] * B.itemsize)
        except Exception:
            pass                                 # no basis -> no projected matrix to pay for
    return total


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# The operator — fitted once, read three ways
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def _fit(concepts, ic, *, store=None, coordinate=None):
    """This screen's own Koopman/DMD operator, fitted in the corpus basis. `None` if unreadable.

    The signal is the observation stream itself. Rows = ticks (a genuinely ordered axis rather than
    an arbitrary ordering read for meaning), columns = the concept's JC coordinate. If observations
    at nearby ticks resemble each other the present persists; if they decorrelate immediately, `is`
    is gone almost at once. The decay is the correlation structure of what this screen is watching,
    and it differs per screen.

    The fit is taken in the corpus basis, and that is what makes it readable at all. The same
    concepts fitted both ways on a live corpus:

        raw 2048-dim dense coordinate      in `geom.corpus-basis` (280)
        dog        n_modes=1  degenerate   n_modes=7  tau 0.340 / 91.04   ratio 268
        water      n_modes=1  degenerate   n_modes=6  tau 0.640 / 14.24   ratio  22
        physicist  n_modes=1  degenerate   n_modes=5  tau 0.285 / 81.56   ratio 286

    In the raw coordinate the operator resolves exactly one mode at every N from 3 to 500, so
    `short_range == long_range` and there is no fast/slow structure to read. Projected first, it
    resolves five to seven distinct timescales. The degeneracy is the basis rather than a shortage
    of observations.

    No delay lift is taken. In the corpus basis, d=1 already resolves 5-7 modes, and at d>=3
    `tau_slow` goes negative (-288, -1381, -2789); alpha < 0 is growth, which `_rates_from` reads as
    no decay timescale. The corpus basis has already decorrelated these coordinates, so lifting adds
    only unstable modes."""
    try:
        import numpy as np
        # One seam onto the instrument — see `ember/optics.py`.
        from ember.optics import dynamics_state as Dynamics
    except Exception:
        return None
    # `geo.dense_vec` needs a synset. A screen watching a read stream holds units the reading
    # formed — `'Kit'`, `'the'`, `' Th'` — none of which has one, so every row is dropped, `rows`
    # comes back empty, and the fit returns None: such a screen cannot measure a decay from
    # `geo.dense_vec` alone.
    #
    # The `coordinate` parameter is the seam that covers this case. A caller that has its own
    # coordinate — a reading has exactly one, derived from what it read
    # (`projection.read_basis`) — supplies it; everything else keeps the corpus coordinate
    # unchanged. Forgetting still requires a coordinate its concepts live in.
    _coord = coordinate if coordinate is not None else (lambda c: geo.dense_vec(c, ic))
    rows = []
    for cpt in concepts:
        try:
            v = _coord(cpt)
        except Exception:
            continue
        v = np.asarray(v, dtype=float)
        # A zero-norm vector is finite but carries no coordinate, so `np.all(isfinite)` alone admits
        # the all-zero vector every concept with no stored geometry returns and adds a row with no
        # rank. The norm test is what keeps such a row out of the fit, on the candidate side as well
        # as the prediction side (`output_screen._operator_flow`).
        if v.size and np.all(np.isfinite(v)) and float(np.linalg.norm(v)) > 0.0:
            rows.append(v)
    # There is no typed minimum row count. The instrument bounds the fit on its own terms and those
    # bounds are measured: `n_pairs < 2` below (which fires at N=2), a non-finite rate, and
    # `alpha <= 0` (growth rather than decay). A fit at N=3 succeeds and is stable on the live
    # corpus, so a typed floor would stand in front of the instrument's own answer
    # ([[no-arbitrary-caps]] — the measurement already bounds this).
    if len(rows) < 2:                       # not a trajectory at all — there is no pair to fit
        return None
    W = np.vstack(rows)
    # ── into the corpus basis ────────────────────────────────────────────────────────────────────
    # Absent a store or a basis the fit would stay in the raw coordinate, where it is degenerate, so
    # the caller is told nothing was measured rather than handed one collapsed reading labelled
    # "measured".
    # A caller-supplied coordinate arrives in its own basis — that is the point of supplying one —
    # so projecting it onto the corpus basis would be re-basing a frame that is already based, and
    # the width check below would reject it outright.
    if coordinate is not None:
        try:
            d = Dynamics(W.shape[1])
            d.update_block(W)
            return d if d.n_pairs >= 2 else None
        except Exception:
            return None
    B = None
    if store is not None:
        try:
            from ember.signal import projection as _pj
            B = _pj.load_basis(store)
        except Exception:
            B = None
    if B is None:
        return None
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[1] != W.shape[1]:
        return None                          # a basis of another width is not this frame's basis
    W = W @ B.T
    try:
        d = Dynamics(W.shape[1])
        d.update_block(W)
        if d.n_pairs < 2:
            return None
    except Exception:
        return None
    return d


def _rates_from(d):
    """`(tau_fast, tau_slow)` off a fitted operator, or `None`.

      tau_fast  1 / rates().short_range   the fastest per-mode decay a_k = -log|mu_k|
      tau_slow  1 / rates().long_range    the slowest

    Phase plays no part: `rates()` takes `alpha` off `abs(mu)` and carries `beta = angle(mu)`
    separately, so a timescale read here is unaffected by phase or sign.

    These do not drive `Trace.amplitude` — the whole curve does — but they are a real reading of this
    screen, and `prism.demurrage` prices the economy's clock off `tau_slow`
    (`ember/runtime/boot.py`), so they stay measured and reported."""
    try:
        import numpy as np
        r = d.rates()
        fast, slow = float(r.short_range), float(r.long_range)
    except Exception:
        return None
    # alpha <= 0 is growth rather than decay — a screen whose observations are diverging has no
    # forgetting timescale to read, and inventing one would describe the opposite of the data.
    if not (np.isfinite(fast) and np.isfinite(slow)) or fast <= 0 or slow <= 0:
        return None
    # One mode is not a two-timescale measurement. `short_range = max(alpha)` and
    # `long_range = min(alpha)`, so an operator resolving a single mode returns them equal and this
    # function would hand back `tau_fast == tau_slow` for the caller to stamp "measured". On a live
    # delegate screen, which observes one concept per turn: `1.122/1.122` at 4 traces and
    # `0.5507/0.5507` at 7.
    #
    # This function is asked for a fast rate and a slow rate. Where the data resolve one timescale
    # there is no fast-versus-slow in them, so there is no separation to report
    # ([[absence-is-not-an-affirmative-claim]]). The comparison is the data against itself, with no
    # chosen value in it.
    #
    # The decay curve is read independently of this. A one-mode operator has no two timescales but it
    # does have a decay — `C(tau) = P mu^tau` — and reporting that claims nothing about separation.
    # The two readings stand or fall separately because they claim different things.
    if not (slow < fast):                   # min(alpha) < max(alpha) — i.e. two distinct rates
        return None
    return (1.0 / fast, 1.0 / slow)


def measured_rates(concepts, ic, *, store=None):
    """Read this screen's own decay timescales off its observation history. `None` if unreadable.

    A history too short or an operator too degenerate to separate yields `None` rather than a
    default: "could not measure" is a different answer from "measured 1.6", and there are no seed
    constants to substitute (see the module header). The caller records
    `rates_source == "unmeasured"`."""
    d = _fit(concepts, ic, store=store)
    return None if d is None else _rates_from(d)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# The decay curve — one measured curve, no bands and no floor
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def _reconstruct(op, span: int, margin: float):
    """`(values, clipped)` — the measured decay over lags `0..span`, as an amplitude.

    `Dynamics.reconstruct_decay` gives `C(tau) = sum_k P_k mu_k^tau` from the operator's own
    spectrum, normalised so `C(0) = 1`, extrapolating past the observed window. Three things happen
    to it here, and each is a statement about what an amplitude is:

    1. Clamped to [0, 1]. `C` is an autocorrelation and may legitimately go negative — that is
       anti-correlation. Measured: a 15-concept screen reads `C(1) = -0.0763`, a 4-concept delegate
       screen `C(1) = -0.4296`. A negative correlation is not a negative amplitude: an amplitude is
       a surviving fraction, whose definitional bottom is 0 and whose top is `C(0) = 1` by the
       instrument's own normalisation. The clip is counted (`DecayCurve.clipped`) rather than
       silent.

    2. Taken to its monotone hull — `E(tau) = max_{s >= tau} C+(s)`, the least non-increasing
       majorant of the clamped curve. This is what keeps a `was` from reading stronger than the `is`
       before it. On a live 4-concept delegate stream the clamped curve reads

           tau     0      1      2      3      4      5      6
           C+    1.00   0.00   0.00   0.669  0.00   0.00   0.447

       — a trace 1.92x stronger at tick 3 than at tick 1. The zig-zag is phase (`arg(mu) = pi`: the
       stream flips sign turn to turn), and phase may not corrupt an amplitude read. The hull reads
       the envelope and leaves the phase alone. On a stream whose decay is already monotone it
       changes nothing — measured on the hypernym chain, hull == C+ to seven lags.

    3. The window is derived from the demand and the instrument's own margin. The hull needs enough
       tail to be final. Because the modal powers are non-negative and normalised
       (`sum_k P_k = 1`), `|C(tau)| <= margin^tau` where `margin = max_k|mu_k|` is what
       `Dynamics.forgetting()` publishes. So the hull over `0..span` can no longer move once
       `margin**L` has fallen to or below the smallest value already in it — an exact bound, from a
       published measurement, with no epsilon anywhere. The secondary stop is the fixed point
       (doubling the window changed nothing), also exact. `margin >= 1` is the operator saying this
       screen does not forget: there is no tail to wait for, and the curve stays where it is."""
    import numpy as np
    span = max(0, int(span))
    L = span + 1
    prev = None
    while True:
        C = np.asarray(op.reconstruct_decay(L), dtype=float)
        if C.size == 0 or not bool(np.all(np.isfinite(C))):
            # A non-finite reconstruction is not a decay, so no curve comes back. `nan_to_num` here
            # would hand out a curve nobody measured, and the clip counter would not show it.
            return [], 0
        Cc = np.clip(C, 0.0, 1.0)
        clipped = int((Cc != C).sum())
        E = np.maximum.accumulate(Cc[::-1])[::-1]          # least non-increasing majorant
        head = E[:span + 1]
        if margin >= 1.0 or C.size < L:                    # no tail to wait for / instrument capped
            return [float(v) for v in head], clipped
        if prev is not None and prev.shape == head.shape and bool((prev == head).all()):
            return [float(v) for v in head], clipped       # fixed point: the tail added nothing
        tail_bound = margin ** L                           # |C(tau)| <= margin^tau  (sum_k P_k = 1)
        if tail_bound == 0.0 or (head.size and tail_bound <= float(head.min())):
            return [float(v) for v in head], clipped       # nothing later can lift the hull
        prev = head
        L *= 2


class DecayCurve:
    """`C(tau)` — one screen's own measured decay, evaluable at any lag.

    There is no crossover, no two bands and no residual height: every resolved mode contributes at
    its measured weight, and a trace's amplitude is `energy * C(dt)`.

    It keeps the operator it was rebuilt from, so a read at a lag beyond the reconstructed window
    extends the curve rather than clamping to its last value — clamping would be a floor by another
    name. A curve restored from disk has no operator (see `Screen.restore`); the next `read()`
    re-measures against the restored bag and rebuilds one."""

    def __init__(self, values, *, operator=None, margin: Optional[float] = None, clipped: int = 0):
        self.values: List[float] = [float(v) for v in values]
        # `max_k |mu_k|` on the connected spectrum, from `Dynamics.forgetting()`. `< 1` is the
        # instrument saying this screen forgets; `>= 1` is a persistent mode — it does not.
        #
        # The default is `None` rather than `1.0`. `1.0` is the assertion "this screen does not
        # forget", which stops `_reconstruct` waiting for the tail that certifies the hull final, so
        # a curve nobody measured a radius for carries none
        # ([[absence-is-not-an-affirmative-claim]]).
        #
        # It is only ever `None` on a restored curve, which has no use for it: `at()` consults
        # `margin` solely to extend the reconstruction, and extending needs `_op`, which a curve read
        # back from disk does not have (`Screen.restore` passes no operator). So an absent radius
        # costs nothing here and the next `read()` re-measures the whole curve against the restored
        # bag. A live curve always carries a published radius, because `_decay_from` builds one only
        # when the instrument states it.
        self.margin: Optional[float] = None if margin is None else float(margin)
        # How many reconstructed lags were anti-correlated (clamped to 0). Reported, never silent.
        self.clipped = int(clipped)
        self._op = operator

    def __len__(self) -> int:
        return len(self.values)

    def at(self, dt) -> float:
        """`C(dt)`. `dt` is clamped at 0, so a read with a backwards clock returns `C(0)`."""
        i = int(max(0, dt))
        if i >= len(self.values) and self._op is not None and self.margin is not None:
            try:
                vals, clipped = _reconstruct(self._op, i, self.margin)
            except Exception:
                vals, clipped = [], 0
            if len(vals) > len(self.values):
                self.values, self.clipped = vals, clipped
        if i < len(self.values):
            return self.values[i]
        # Past the reconstruction, with no operator to extend it (a curve restored from disk whose
        # store has gone). The hull is non-increasing, so this lag is at or below the smallest value
        # the measurement resolved — below the end of the measurement, not at a chosen floor.
        return 0.0

    def tolist(self) -> List[float]:
        return list(self.values)


def _decay_from(d, span: int) -> Optional["DecayCurve"]:
    """The curve off a fitted operator, or `None` if the instrument cannot produce one.

    `margin` is `max_k |mu_k|`, the operator's own spectral radius, and `Dynamics.forgetting()`
    publishes it, so it is read rather than defaulted ([[never-handroll-probes]] — read the published
    stat; a missing stat is a reason to add it).

    A default would not be neutral here. `margin >= 1.0` is the instrument saying this screen does
    not forget, which makes `_reconstruct` return its first window immediately without waiting for
    the tail to certify the hull final. An absent margin read as 1.0 therefore produces a curve too
    high at long lags — traces reported as more present than they are, which is the defect a floor
    produces, arriving through the measurement instead.

    A screen whose operator does not report its own spectral radius has not measured a decay, and
    `Screen.measure` records "unmeasured": every trace stays fully present, which is the honest
    reading and the one this module takes everywhere else."""
    try:
        published = d.forgetting()
        if not isinstance(published, dict) or published.get("margin") is None:
            return None                  # the instrument did not state its radius — no curve
        margin = float(published["margin"])
        vals, clipped = _reconstruct(d, span, margin)
    except Exception:
        return None
    if not vals or vals[0] <= 0.0:
        # `C(0) = 1` by the instrument's own normalisation. Anything else means the reconstruction
        # had nothing to normalise against — not a decay curve, so not reported as one.
        return None
    return DecayCurve(vals, operator=d, margin=margin, clipped=clipped)


def measured_decay(concepts, ic, *, store=None, span: int = 0) -> Optional["DecayCurve"]:
    """Read this screen's own decay curve off its observation history. `None` if unreadable."""
    d = _fit(concepts, ic, store=store)
    return None if d is None else _decay_from(d, span)


def _concept_key(c):
    """The identity `_sim` compares on — a name rather than the object.

    `_sim` is `_law.similarity(geo.jc_tree(a, b, ic))` — `prism.law` over
    `crystal.ontology.geometry` — which reads concept names, so two objects naming one concept are
    indistinguishable to every reading this module takes. Grouping on the
    object instead would depend on whether a resolver happens to intern its results, which is not a
    property any caller guarantees and not one a measurement may rest on."""
    n = getattr(c, "name", None)
    if callable(n):                       # some synset APIs expose `name()` rather than `.name`
        try:
            return str(n())
        except Exception:
            return str(c)
    return str(n) if n is not None else str(c)


def _sim(a, b, ic) -> float:
    """Concept similarity from the proven geometry: `prism.law.similarity` of the JC distance from
    `crystal.ontology.geometry.jc_tree` — `exp(-JC)`, 1.0 identical, decaying with meaning-distance.
    So a query resonates with a stored trace even when it is a more or less specific relative
    (generalizes). The kernel is the same attenuation law the propagator and demurrage obey, applied
    at the natural unit length."""
    return float(_law.similarity(geo.jc_tree(a, b, ic)))


class Trace:
    """One observation, cooling. Its tense is read off `amplitude(now)` rather than stored."""

    def __init__(self, concept, tick, witness, energy: float = 1.0):
        # `None` until the owning screen measures it (see `Screen.measure`). A trace whose decay has
        # not been measured has no decay curve, and `amplitude` reports it as fully present rather
        # than inventing one.
        self.decay: Optional[DecayCurve] = None
        # What share of its turn this observation was. Energy is information: a concept that fired
        # weakly enters weakly and falls to the bottom of the screen's own cut by itself. Entering
        # every trace at 1.0 instead records a concept the propagation barely lit and the one it
        # resolved as equally witnessed, so the screen cannot tell attention from noise and the split
        # has to come from a rule ("only observe the lead") rather than from an amplitude.
        self.energy = float(energy)
        self.concept = concept          # the synset observed (the footprint's meaning)
        self.tick = tick                # when it entered the light cone
        self.witness = witness          # who was there — the provenance the residual carries

    def amplitude(self, now: int) -> float:
        """Cooling amplitude: `energy * C(now - tick)` on this screen's own measured curve.

        One curve, no bands, no floor. A two-band piecewise construction with a floor of 0.25 has its
        crossover at `dt = 0.80` on the live corpus, so the fast band is never read and the floor
        supplies the entire height — at dt=1 it reports 0.2269 where the screen's own curve reads
        0.0021. See the module header for the full comparison.

        Three properties follow from reading one measured curve, and each closes a failure the
        two-band construction carries:

        1. Consolidation is not timed by when someone looked. A `settle_tick = now` records the read
           that happened to observe the crossing, so an idle screen retains everything at full
           strength indefinitely: a trace read at t=1000 returns 0.550000 where the same trace read
           every tick returns 8.23e-12. There is no crossing here to time.
        2. The amplitude is non-increasing by construction, because `_reconstruct` takes the monotone
           hull. A residual height above the floor steps the curve upward as time advances
           (0.2865 -> 0.5500, a 1.92x increase), and a phase-flipping stream reproduces that through
           the measured curve when the hull is not taken.
        3. A backwards clock reads `C(0)`: `dt` is clamped at 0 in `DecayCurve.at`.

        An unmeasured screen has no decay and therefore no `was`. With no curve there is nothing to
        evaluate, and the honest amplitude is the energy this observation carried — fully present —
        rather than a number produced by a typed timescale (see the module header). It is
        self-correcting: the moment the curve resolves, the screen pushes it onto every trace and the
        whole bag cools on its own measured decay."""
        if self.decay is None:
            return self.energy
        return self.energy * self.decay.at(now - self.tick)

    def tense(self, now: int, *, lead: Optional[float] = None) -> str:
        """Emergent rather than stored. `lead` is the smallest amplitude still in the signal group of
        the owning screen's own ordered amplitudes — `Screen.lead(now)`, which is
        `prism.resolution.signal_end` and computes its own null, so a screen whose amplitudes do not
        separate returns everything and is entirely `is`.

        Where the present ends is a property of the whole screen — where this bag of amplitudes
        splits into signal and tail — so a typed threshold on a single trace, which knows nothing
        about the others, cannot answer it.

        Without a `lead` the cut is taken over this trace alone, which is the honest degenerate
        answer: one reading does not separate into two groups. Only an amplitude of exactly zero
        reads `—`, and zero is the absence of a reading rather than a small one."""
        a = self.amplitude(now)
        if a <= 0.0:
            return "—"
        return "is" if lead is None or a >= lead else "was"


class Screen:
    """The persistent activator: a bag of cooling traces. Observing adds one; reading resonates a
    query concept against every trace, weighted by its current amplitude. Present and past are the
    same read at two bands of one measured curve, and the result reports which band each contribution
    came from."""

    def __init__(self, ic=None, *, witness: Optional[str] = None, capacity: Optional[int] = None,
                 store=None, coordinate=None):
        self.ic = ic or geo.load_ic()
        # Where the corpus basis is read from. The fit is taken in `geom.corpus-basis` (see `_fit`),
        # and a basis is an artifact, so the screen needs the store its owner holds. `None` means the
        # decay is unmeasurable and the screen reports it as such, rather than measuring against
        # whatever store the process happened to resolve.
        self.store = store
        self.traces: List[Trace] = []
        # Whose screen this is. A screen belongs to exactly one observer; see `delegate.py`. `None`
        # is an unowned screen — the `_demo()` below and some tests — and an unowned screen skips the
        # witness check rather than reporting an enforcement it is not making.
        self.witness = witness
        # `None` means measure it: the default, and the only thing `Delegate` passes. An integer is a
        # caller declaring its own working set — a test rig, or an owner who knows something about
        # its box that the box does not publish. `capacity_source` records which, so a declared bound
        # reads back as declared.
        self.capacity: Optional[int] = None if capacity is None else int(capacity)
        self.capacity_source: str = "declared" if capacity is not None else "unmeasured"
        self._cost_bytes: Optional[int] = None      # measured once — see `trace_bytes`
        self._cap_hint: Optional[int] = None        # last capacity reading (None = unbounded)
        self._cap_read: bool = False                # distinct from `_cap_hint is None`, which is a
        #                                             measured unbounded rather than "not yet asked"
        # Traces dropped because the screen was full. Reported rather than silent: an agent whose
        # working memory is saturating is visible in this counter.
        self.evicted: int = 0
        # Traces seen bearing someone else's witness. Reads 0 in a correct deployment; non-zero means
        # cognitive state is being pooled across agents — see `foreign_witnesses`.
        self.foreign: int = 0
        # How a concept on this screen becomes a vector. `None` = the corpus coordinate
        # (`geo.dense_vec` projected onto the corpus basis), which is right for every screen whose
        # concepts are synsets. A screen watching a stream of units the reading formed has no
        # synsets, so it supplies its own — see `_fit`. Without this the fit finds no rows, no decay
        # is measured, and the screen never forgets.
        self.coordinate = coordinate
        # This screen's own decay curve. `None` until measured, and `decay_source` records which, so
        # an absence reads as an absence.
        self.decay: Optional[DecayCurve] = None
        self.decay_source: str = "unmeasured"
        # Per-screen timescales, read off the same operator. They do not drive `amplitude` (the curve
        # does) but they are a real reading, and `prism.demurrage` prices the economy's clock off
        # `tau_slow` — see `runtime/boot.py`.
        self.tau_fast: Optional[float] = None
        self.tau_slow: Optional[float] = None
        self.rates_source: str = "unmeasured"
        # What the trace bag looked like when the decay was last read off it. `(len, evicted)` is a
        # faithful mutation witness — `observe` moves the first, `_evict` moves both — so the key is
        # bookkeeping the bag already keeps rather than a side-car identity. `None` = never measured.
        self._rates_at: Optional[tuple] = None

    # ── measurement ─────────────────────────────────────────────────────────────────────────────
    def _span(self, now: Optional[int] = None) -> int:
        """The largest lag this screen can be asked for — `now - oldest tick`, from the bag itself.

        This is the demand rather than a window length or a horizon: the reconstruction covers the
        lags the traces actually sit at, and nothing beyond (`DecayCurve.at` extends on demand if a
        later read reaches further). Any "how many lags" constant would be a guess at this number."""
        ticks = [int(t.tick) for t in self.traces]
        if not ticks:
            return 0
        base = int(now) if now is not None else max(ticks)
        return max(0, base - min(ticks))

    def measure(self, store=None, *, now: Optional[int] = None) -> str:
        """Fit one operator on this screen's observation history and read the decay off it.

        The decay belongs to the screen rather than to the module. A screen watching a fast,
        decorrelating stream forgets its present almost immediately; one watching a slow, coherent
        stream holds it. A module-level floor asserts one residual height, and a module-level
        timescale one rate, for every screen that will ever exist.

        `store` is optional in the signature only. The fit is taken in the corpus basis and the basis
        is an artifact, so without a store there is nothing to project onto: `_fit` returns None and
        the screen reports "unmeasured". See the measurement table in `_fit` — in the raw coordinate
        the operator resolves one mode at every N.

        The curve and the rates are read independently. A one-mode operator has no fast-versus-slow
        to report (`_rates_from` returns None) but it does have a decay, so they are two fields with
        two sources and one is not evidence for the other.

        Returns `decay_source` — "measured" or "unmeasured". A failure to measure is reported as
        itself; there is no constant to fall back to."""
        d = _fit([t.concept for t in self.traces], self.ic,
                 store=store or self.store, coordinate=self.coordinate)
        r = _rates_from(d) if d is not None else None
        if r is None:
            self.tau_fast, self.tau_slow = None, None
            self.rates_source = "unmeasured"
        else:
            self.tau_fast, self.tau_slow = r
            self.rates_source = "measured"
        c = _decay_from(d, self._span(now)) if d is not None else None
        # An unreadable stream leaves the screen with no decay curve, and `Trace.amplitude` reports
        # its traces as fully present. "Could not measure" is recorded as itself; there is no
        # constant standing behind it.
        self.decay = c
        self.decay_source = "measured" if c is not None else "unmeasured"
        for t in self.traces:                       # traces read the screen's decay
            t.decay = self.decay
        self._rates_at = (len(self.traces), self.evicted)
        return self.decay_source

    def _rates_current(self, now: Optional[int] = None) -> None:
        """Re-read the decay when — and only when — the stream it describes has changed.

        The decay is a function of the trace bag, so this is memoisation on that function's input: if
        the bag has not moved the previous reading still stands, and if it has moved the previous
        reading describes a stream that no longer exists. Any "measure every k turns" cadence would
        be a typed constant standing in for exactly this. Key on the bookkeeping the thing already
        keeps ([[lattice-store-mutates-artifacts]] is the same move for artifacts).

        `now` is not part of the key. A clock that advanced without the bag moving does not
        invalidate the reading; it only asks the curve for a longer lag, and `DecayCurve` extends
        itself from the operator it kept."""
        at = (len(self.traces), self.evicted)
        if self._rates_at != at:
            self.measure(now=now)

    def _own(self, tr: "Trace") -> bool:
        """Is this trace this screen's own memory?

        A populated `witness` field is only a boundary once something compares it. With one
        process-wide screen serving every agent and no comparison, delegate A's attention scores into
        delegate B's read as B's own present tense. This is the comparison that makes the field mean
        something: it costs one string compare, and it makes any re-pooling fail loudly rather than
        bleed silently."""
        if self.witness is None:                    # unowned screen (demo/tests) — nothing to check
            return True
        if tr.witness == self.witness:
            return True
        self.foreign += 1
        return False

    def foreign_witnesses(self) -> int:
        """Traces seen bearing another observer's witness. This reads 0 in a correct deployment;
        non-zero means cognitive state is being pooled across agents.

        Zero does not establish the converse, and the scope of the check is why. It compares
        `tr.witness` against `self.witness`, and both are the delegate's id, so it catches pooling at
        the screen level (two delegates handed one Screen) and is structurally blind to pooling at
        the delegate level (two people handed one Delegate).

        The served chat is the second case. `aria/www/bff/main.py` sends `{"text": q}` with no
        principal, so `conversation._as_delegate` calls `Delegate.get(store, person=None)`, which
        falls back to `EMBER_PRINCIPAL`; every visitor resolves to the same delegate id, so every
        trace is stamped with the witness this screen checks against and `_own` returns True for all
        of them. The counter reads 0 while attention is fully shared.

        The consequence reaches past privacy. `_rates_current` re-measures whenever
        `(len(self.traces), self.evicted)` changes — on every visitor's turn — and `measure()` writes
        `t.decay = self.decay` onto every trace, so one person's turns re-fit the decay curve applied
        to another person's memory, and `rates_source` moves between "measured" and "unmeasured" off
        a stream they never produced. That is defect #3 from `delegate.py`'s own module header, one
        layer above the layer that closes it.

        So a zero here means no cross-witness trace reached this screen. Establishing that this
        screen belongs to one observer takes a check at the delegate boundary, which is where the
        identity question lives."""
        return self.foreign

    def observe(self, concept, tick: int, witness: str, energy: float = 1.0):
        tr = Trace(concept, tick, witness, energy=energy)
        if self.decay is not None:                  # inherit whatever the screen has measured
            tr.decay = self.decay
        self.traces.append(tr)
        self._evict()

    def observe_field(self, pairs, tick: int, witness: str):
        """Observe a whole turn's fired field on one tick — `[(concept, energy), ...]`.

        The tick is why this takes the whole field rather than being called per concept. A tick is
        one observation step, the observer's own proper time, so advancing it per concept would make
        a turn that fired eighty concepts eighty times older than one that fired three, and the busy
        turn would age the quiet one's memory out of the screen. That is the process-global-tick
        defect `delegate.py` documents, one level down. One turn, one tick."""
        for concept, energy in pairs:
            self.observe(concept, tick, witness, energy=energy)

    def measured_capacity(self) -> Optional[int]:
        """How many traces this box holds and can still read — or `None` when it did not say.

        `envelope / cost`, both measured (see the module header). `None` propagates out to `_evict`,
        which then does nothing: an unmeasured envelope leaves the screen unbounded, because "this
        platform reports no limit" is a different reading from "this platform reports 2048".

        The envelope is re-read; the cost is kept once measured. The cost is a property of the
        coordinate and does not move. The envelope moves continuously, because the box is shared, so
        the screen re-reads it at the one moment the answer matters (`_evict`, when the bag has grown
        past the last reading) rather than pinning whatever the box happened to be doing when this
        screen was constructed. A cost that could not be measured is retried rather than latched: a
        screen whose first concept carried no coordinate has learned nothing about the concepts that
        follow it."""
        if self.capacity is not None:
            return self.capacity                    # declared by the caller — not re-measured
        if self._cost_bytes is None:
            cpt = self.traces[0].concept if self.traces else None
            self._cost_bytes = trace_bytes(cpt, self.ic, store=self.store,
                                           coordinate=self.coordinate)
        from prism import envelope as _env
        cap = _env.holds(self._cost_bytes)
        self.capacity_source = "unmeasured" if cap is None else "measured"
        return cap

    def _evict(self) -> None:
        """Hold the screen to its measured capacity by forgetting the coldest traces first.

        The curve is non-increasing in elapsed ticks (`_reconstruct` takes its monotone hull), so at
        equal energy the lowest `tick` is the lowest amplitude: eviction is "forget what has cooled
        most", the module's own model carried to its conclusion rather than an arbitrary cap.
        Bulk-trimmed rather than one per observe, so a restored screen collapses in one pass.

        The cheap test runs first, and it is exact. `measured_capacity` costs a ctypes call (~96 us)
        and a possible `dense_vec`; a screen below its last known capacity can exceed a re-read one
        only by however far the envelope has moved, so the reading is taken when the bound is in
        reach rather than on every `observe`."""
        if (not self._cap_read or self._cost_bytes is None
                or (self._cap_hint is not None and len(self.traces) > self._cap_hint)):
            self._cap_hint = self.measured_capacity()      # first read, or the bound is in reach
            self._cap_read = True
        cap = self._cap_hint
        if cap is None:
            return                                  # unmeasured envelope -> unbounded, and it says so
        over = len(self.traces) - cap
        if over <= 0:
            return
        self.traces.sort(key=lambda t: t.tick)
        del self.traces[:over]
        self.evicted += over

    # ── the is/was boundary — a cut on this screen's own amplitudes ──────────────────────────────
    @staticmethod
    def _lead_from(amps) -> Optional[float]:
        """The cut itself, over amplitudes already gathered.

        Separate from `lead` so `read` counts each trace once. `_own` has a side effect — it
        increments `foreign`, an invariant counter that reads 0 — so walking the bag twice in one
        read would report two bleeds where one trace crossed."""
        live = sorted((a for a in amps if a > 0.0), reverse=True)
        if not live:
            return None
        k = _res.signal_end(live)
        k = max(1, min(int(k), len(live)))
        return live[k - 1]

    def lead(self, now: int) -> Optional[float]:
        """The smallest amplitude still in the signal group of this screen's ordered amplitudes.
        `None` when the screen holds nothing live.

        "Where does the present end and the past begin" is a property of the whole screen, and
        `prism.resolution.signal_end` answers it the way every other "how many of these are signal"
        question in this codebase is answered: maximum between-class variance against the
        separability a featureless series of the same length would show. It computes its own null, so
        it can answer no — a screen whose amplitudes do not separate returns all of them, which is
        why an unmeasured screen (every trace at full energy, nothing to separate) is entirely `is`
        and stays that way.

        No frame is passed: these are amplitudes, a score column, rather than the ordered (T, F)
        evidence behind them, and `signal_end` keeps the two apart."""
        return self._lead_from([t.amplitude(now) for t in self.traces if self._own(t)])

    def read(self, query_concept, now: int):
        """Return (present_energy, past_energy, contributors) for the query. There is no branch on
        tense — every trace contributes amplitude*similarity, and the tense label says which band the
        energy sits in.

        Only this screen's own traces are read — see `_own`.

        There are no amplitude or energy cutoffs here; see the module header for what such cutoffs
        were measured to remove (14 of 15 traces at lag 20 on the live store). The amplitude already
        says how present a trace is, and the only rows dropped carry amplitude exactly zero, which is
        the absence of a reading rather than a small one."""
        self._rates_current(now)        # the decay describes this stream, as it stands now
        mine = [tr for tr in self.traces if self._own(tr)]
        amps = [tr.amplitude(now) for tr in mine]
        lead = self._lead_from(amps)
        present, past, rows = 0.0, 0.0, []
        for tr, a in zip(mine, amps):
            if a <= 0.0:
                continue                # a measured zero carries no information
            s = _sim(query_concept, tr.concept, self.ic)
            e = a * s
            band = tr.tense(now, lead=lead)
            if band == "is":
                present += e
            else:
                past += e
            rows.append((tr, band, a, s, e))
        rows.sort(key=lambda r: -r[4])
        return present, past, rows

    def read_many(self, query_concepts, now: int):
        """`{name: (present, past)}` for many queries against one screen — the same reading
        :meth:`read` gives, taken once instead of once per query.

        Taken per query, the cost is a product of two unbounded, growing factors: `rank_fired` calls
        `recall(d, name)` for every fired concept, and each call walks every trace. Measured: 1,281
        concepts x 50 traces = 64,050 `_sim` calls in one turn, on a screen designed to accumulate.

        Two reductions, neither of which is a cap:

        1. The query-independent work is done once. `_rates_current`, the `_own` filter, every trace
           amplitude and the lead band do not depend on `query_concept`, so computing them per query
           is repetition rather than a trade.
        2. `_sim` is evaluated per distinct trace concept rather than per trace. A delegate attends
           to the same concepts repeatedly, so a screen of `T` traces carries far fewer than `T`
           distinct concepts; amplitudes are summed within a `(concept, band)` group first and the
           similarity is taken once for the group. `_sim` depends only on the two concept names, so
           this is an exact re-association of the same sum rather than an approximation of it.

        Every query concept is read against every trace. Pre-cutting the candidate list by present
        activation — a `ranked[:top*3]` shortlist in `rank_fired` — would let the past only re-order
        the present's shortlist and never add to it, which is the one thing a two-timescale screen
        exists to do. The cost here is lower because repeated work was removed, not because the
        answer is smaller.
        """
        names = list(query_concepts)
        out = {n: (0.0, 0.0) for n in names}
        if not names:
            return out
        self._rates_current(now)                    # once — describes this stream as it stands
        mine = [tr for tr in self.traces if self._own(tr)]
        if not mine:
            return out
        amps = [tr.amplitude(now) for tr in mine]
        lead = self._lead_from(amps)

        # Grouped by the concept's name rather than by the concept object. Two observations of one
        # concept can be two distinct objects (a resolver returns a fresh synset per call unless it
        # interns), so an identity key produces one group per trace and the reduction evaporates
        # while every equivalence test still passes — visible only by asserting the `_sim` call count
        # rather than the timing.
        #
        # The name is the right key because it is what the measurement itself keys on: `_sim` is
        # `_law.similarity(geo.jc_tree(a, b, ic))`, and two traces whose concepts share a name are
        # the same concept to that function. One representative object is kept per group and handed
        # to `_sim`, which is exact rather than approximate for the same reason.
        #
        # A measured zero carries no information and is dropped here exactly as `read()` drops it.
        groups = {}                        # (concept name, band) -> [summed amplitude, a concept]
        for tr, a in zip(mine, amps):
            if a <= 0.0:
                continue
            key = (_concept_key(tr.concept), tr.tense(now, lead=lead))
            slot = groups.get(key)
            if slot is None:
                groups[key] = [a, tr.concept]
            else:
                slot[0] += a
        if not groups:
            return out

        for n in names:
            q = self._query_concept(n)
            if q is None:
                continue                            # not a concept this screen can measure against
            present = past = 0.0
            for (_name, band), (a, concept) in groups.items():
                e = a * _sim(q, concept, self.ic)
                if band == "is":
                    present += e
                else:
                    past += e
            out[n] = (float(present), float(past))
        return out

    def _query_concept(self, name):
        """Resolve a query to whatever `_sim` compares.

        A non-string passes through unchanged, so this method takes the same thing `read()` takes as
        well as a name. That symmetry is what makes the two paths comparable: if the bulk read only
        accepted names it would resolve them through the store while `read()` resolved nothing, and
        the two could then disagree about what a concept is, leaving the equivalence between them
        untestable — which is how a fast path becomes a different measurement from the slow one.

        Best-effort on the name path, holding the same contract `recall` holds: a memory read leaves
        the answer intact."""
        if not isinstance(name, str):
            return name
        try:
            from crystal.ontology import driver as wn
            return wn.synset(name)
        except Exception:
            return None

    # ── persistence: attention must survive a restart, and so must the clock that dates it ──────
    def serialize(self) -> dict:
        """The traces, as data. The caller persists the owning delegate's `tick` alongside this — see
        `delegate.Delegate.save_screen`. Traces without their clock are meaningless."""
        return {
            # `energy` rides with each trace. Without it a restored screen brings every observation
            # back at full share, discarding the measured distribution of the turn that made them.
            "traces": [{"concept": t.concept.name(), "tick": int(t.tick), "witness": t.witness,
                        "energy": float(t.energy)}
                       for t in self.traces],
            # Written, and deliberately not read back by `restore`. Capacity is a property of the box
            # this screen is running on, measured now, rather than of the box that wrote the file: a
            # screen restored onto a Pi from a 32 GB node's dump measures the Pi. `None` here is the
            # honest record of a screen that was unbounded because its platform reported no envelope;
            # `capacity_source` says whether the number was measured or declared.
            "capacity": self.capacity if self.capacity is not None else self._cap_hint,
            "capacity_source": self.capacity_source,
            "evicted": self.evicted,
            "rates_source": self.rates_source,
            "tau_fast": self.tau_fast,
            "tau_slow": self.tau_slow,
            # The curve goes to disk with the rest of the measurement, and `restore` reads it back. A
            # measurement written here and dropped on the way in is the persistence equivalent of
            # never having measured.
            "decay_source": self.decay_source,
            "decay": self.decay.tolist() if self.decay is not None else None,
            "decay_margin": self.decay.margin if self.decay is not None else None,
            "decay_clipped": self.decay.clipped if self.decay is not None else None,
        }

    def restore(self, doc: dict, *, witness: Optional[str] = None) -> int:
        """Rebuild traces from `serialize()`. Returns how many were restored.

        Foreign-witness rows are dropped rather than loaded: restoring another observer's traces
        would reintroduce the bleed this class prevents. Counted in `foreign` either way.

        The measurement is restored too — the rates, the decay curve and its clip count — so a screen
        that had measured comes back measured rather than reporting "unmeasured" with everything at
        full amplitude. `_rates_at` is left `None` so the next `read()` re-measures against the
        restored bag, which also gives the restored curve an operator again and lets it extend past
        the window that was written."""
        from crystal.ontology import driver as wn
        want = witness if witness is not None else self.witness
        src = doc.get("rates_source")
        if src in ("measured", "seed"):
            tf, ts = doc.get("tau_fast"), doc.get("tau_slow")
            if tf is not None and ts is not None:
                self.tau_fast, self.tau_slow = float(tf), float(ts)
                self.rates_source = src
        dvals = doc.get("decay")
        if doc.get("decay_source") == "measured" and dvals:
            # The margin is carried through as it was written, `None` included. A persisted curve
            # that recorded no spectral radius did not measure one, and reading it as `1.0` would
            # restore it as "this screen does not forget" — a reading manufactured on the way in. See
            # `DecayCurve.__init__` for why a restored curve has no use for the radius anyway.
            _m = doc.get("decay_margin")
            self.decay = DecayCurve(dvals, margin=(None if _m is None else float(_m)),
                                    clipped=int(doc.get("decay_clipped") or 0))
            self.decay_source = "measured"
        n = 0
        for row in (doc.get("traces") or []):
            w = row.get("witness")
            if want is not None and w != want:
                self.foreign += 1
                continue
            try:
                cpt = wn.synset(row["concept"])
            except Exception:
                continue
            if cpt is None:
                continue
            # A row carrying no `energy` field was observed at full share, so `1.0` is what it
            # recorded rather than a default standing in for a measurement.
            tr = Trace(cpt, int(row.get("tick") or 0), w,
                       energy=float(row.get("energy", 1.0) or 0.0))
            tr.decay = self.decay
            self.traces.append(tr)
            n += 1
        self.evicted += int(doc.get("evicted") or 0)
        self._evict()
        return n


# ── self-check: watch `is` cool into a witnessed `was` ───────────────────────────────────────────
def _syn(name):
    from crystal.ontology import driver as wn
    try:
        return wn.synset(name)
    except Exception:
        return None


def _demo():
    ic = geo.load_ic()
    # The demo needs a store to demonstrate anything. The fit is taken in `geom.corpus-basis`, which
    # is an artifact; without it the screen is honestly "unmeasured" and every trace stays at full
    # amplitude — a correct reading and a useless demonstration. The print below says which
    # happened.
    try:
        from mantle.shard.local_store import open_store
        store = open_store()
        from crystal.ontology import driver as _wn
        _wn.bind(store)
    except Exception:
        store = None
    scr = Screen(ic, store=store)

    howl = _syn("howl.n.01") or _syn("bark.n.03") or _syn("sound.n.04")
    dog = _syn("dog.n.01")
    print("=" * 78)
    print("TENSE = amplitude on the MEASURED decay curve. No is/was flag, and no floor.")
    print("-" * 78)
    print(f"tick 0: OBSERVED  '{dog.name()} {howl.name()}'  (witness: john — 'I was there')\n")
    scr.observe(howl, tick=0, witness="john")
    for t, c in enumerate([dog, _syn("animal.n.01"), _syn("sound.n.04")], start=1):
        if c is not None:
            scr.observe(c, tick=t, witness="john")

    header = f"{'tick':>4} | {'a(now)':>9} | {'tense':>5} | reading of  \"the dog ___\""
    print(header)
    print("-" * len(header))
    for now in [0, 1, 2, 3, 5, 10, 30]:
        present, past, rows = scr.read(howl, now)
        tr = scr.traces[0]
        a = tr.amplitude(now)
        tense = tr.tense(now, lead=scr.lead(now))
        if tense == "is":
            reading = f"IS howling            (present={present:.3f})"
        elif tense == "was":
            reading = f"WAS howling  [witness: {tr.witness}]   (past={past:.3f})"
        else:
            reading = "— (nothing left of it)"
        print(f"{now:>4} | {a:>9.6f} | {tense:>5} | {reading}")

    print()
    print("decay_source:", scr.decay_source, " rates_source:", scr.rates_source)
    print("The SAME signal, on ONE curve this screen measured for itself. Nothing was deleted; it")
    print("cooled — and the is/was boundary is where THIS screen's amplitudes separate, not 0.25.")
    print()
    print("=" * 78)
    print("PAST is still queryable, and still GENERALIZES through the geometry")
    print("-" * 78)
    now = 8
    for q in ["howl.n.01", "sound.n.04", "cry.n.01"]:
        qs = _syn(q)
        if qs is None:
            continue
        present, past, rows = scr.read(qs, now)
        verb = "did the dog make this?"
        print(f"  tick {now}, query '{q:12s}' -> present={present:.3f}  past={past:.3f}   ({verb})")
    print("  (the witnessed `was` answers 'did it happen?' even for a related query — past tense that")
    print("   still reasons through meaning-distance, not a boolean fact that was stored as true.)")
    print("=" * 78)


if __name__ == "__main__":
    _demo()
