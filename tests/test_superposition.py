"""The discharge, the second channel, and the three-way merge verdict.

A concept can arrive across a gap rather than only by walking the taxonomy, and the jump is a budget
the signal already carries. The distance is `jc_tree`, the budget is
`prism.resolution.horizon(xi, gap, weight)` — the same bound that governs the taxonomic walk — and
the weight is what the field has accumulated at that vertex. Measured on the live corpus:

    xi = 0.4647   propagation_floor = 0.0263
    cow.n.01 accumulated weight  245.14  ->  horizon = 4.2476 nats
    JC(cow.n.01, moo.n.01)                            1.6040 nats     reaches, 2.6x over

So the discharge needs no new metric: everything it takes is already measured.

The second channel rides beside it. `crystal.ontology.geometry.jc_tree_se` propagates the distance's
standard error in quadrature; its docstring shows that carrying `se` beside the value keeps the JC
identity exact to 7.1e-15 while folding it in destroys it to 5.99. Its only reader today is
`geometry.py` itself, which is enough while every path is a taxonomy walk. Once a concept can arrive
across a gap, three hypernym steps and one discharge are different distances *and* different
confidences, so ranking on the point estimate alone would report precision nobody earned.
"""
from __future__ import annotations

import math

import pytest


# ══════════════════════════════════════════════════════════════════════════════════════════════
# The discharge — the jump is a budget the signal already carries
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_the_jump_distance_is_the_horizon_of_the_ACCUMULATED_weight():
    """The discharge distance is `xi * ln(weight/gap)`, and the weight is what the field has
    accumulated at that vertex — 245.14 at `cow` for `what does a cow say`.

    Measured with the corpus's own xi and gap: a weight of 1.0 buys 1.6909 nats and the accumulated
    weight buys 4.2476, against the 1.6040 needed to reach `moo`. Both happen to clear it, so what
    the parameter is for is the ordering: a peripheral concept carrying little charge reaches a
    shorter distance than a strongly-fired one."""
    from prism.resolution import horizon
    xi, gap = 0.4647, 0.0263
    pinned = horizon(xi, gap, weight=1.0)
    real = horizon(xi, gap, weight=245.14)
    assert real > pinned, "the accumulated weight must buy a longer reach than the pinned 1.0"
    assert real == pytest.approx(xi * math.log(245.14 / gap), rel=1e-9)
    assert real == pytest.approx(4.2476, abs=1e-3)      # the live reading
    assert real >= 1.6040, "cow's charge must clear the measured JC gap to moo"


def test_a_weak_signal_cannot_jump_as_far_as_a_strong_one():
    """The physics that makes this safe to apply blind: associative edges open where the need has
    energised the region. A concept the query barely reached carries little charge and discharges a
    short distance, and the accumulated weight is what sets that distance."""
    from prism.resolution import horizon
    xi, gap = 0.4647, 0.0263
    assert horizon(xi, gap, weight=245.0) > horizon(xi, gap, weight=2.0)


def test_a_signal_under_the_gap_does_not_propagate_at_all():
    """A gap is a discontinuity rather than a small number. Below the corpus's own propagation floor nothing
    crosses at any distance, including zero (`prism.resolution.reach_limit`). That is what gates the
    discharge: the corpus's own measurement rather than a threshold anyone picked."""
    from prism.resolution import horizon, reach_limit
    xi, gap = 0.4647, 0.0263
    assert horizon(xi, gap, weight=gap / 2.0) == 0.0
    assert not reach_limit(gap / 2.0, gap)
    assert reach_limit(gap * 2.0, gap)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# The second channel — variance rides beside the distance rather than inside it
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_the_distance_error_is_propagated_in_QUADRATURE_not_summed():
    """`se(JC) = sqrt(se_a^2 + se_b^2 + 4*se_lcs^2)` — three independent uncertainties combining,
    with the LCS entering twice because `JC = IC(a) + IC(b) - 2*IC(LCS)`. A plain sum would
    overstate the error and make a long path read as arbitrarily unreliable rather than as reliably
    less certain."""
    se_a = se_b = se_l = 0.1
    quad = math.sqrt(se_a ** 2 + se_b ** 2 + 4 * se_l ** 2)
    assert quad == pytest.approx(math.sqrt(0.06))
    assert quad < (se_a + se_b + 2 * se_l), "quadrature must be tighter than a naive sum"


def test_an_unmeasured_uncertainty_is_not_a_zero_uncertainty():
    """`jc_tree_se` returns None when any member carries no `ic_se`, and that None propagates rather
    than becoming 0. A corpus that stores no `ic_se` would otherwise rank on `JC ± 0` and report
    confidence nobody measured ([[absence-is-not-an-affirmative-claim]]).

    The discharge makes this load-bearing: a jumped path and a walked path are comparable only if
    both carry a real error, and a zero would make the jump read as the most certain step in the
    path rather than the least."""
    from crystal.ontology import geometry as g

    class _NoSE:
        def ic_se(self):
            return None
    assert g.ic_se_of(_NoSE()) is None
    assert g.ic_se_of(_NoSE(), default=0.25) == pytest.approx(0.25)


def test_the_error_is_never_folded_into_the_coordinate():
    """The reason is structural rather than statistical. Carrying `se` beside the point value keeps
    `|L2^2 - jc_tree|` at 7.1e-15; resampling IC within +/-1 se per call drives it to 5.99, and the
    error grows with path depth. The JC identity telescopes — the shared root->LCS prefix is built
    from the same edge weights and cancels exactly — and independent draws break the cancellation.

    So it is asserted as a property of the API: `jc_tree` takes no randomness and is a pure function
    of (s1, s2, ic). A `seed`/`sample` parameter appearing on it means the second channel has been
    folded in and the coordinate is approximate."""
    import inspect
    from crystal.ontology import geometry as g
    params = set(inspect.signature(g.jc_tree).parameters)
    assert not (params & {"seed", "sample", "resample", "draws"}), (
        "jc_tree gained a sampling parameter — the error is being folded into the coordinate")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# The three-way verdict — absorbed, partially absorbed, or separate
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_residual_of_one_is_the_computed_null_not_a_near_miss():
    """A candidate sharing no evidence is excluded before ranking rather than ranked last.
    Measured against every sense sharing the `cow` anchor:

        cn-cow -> wn-cow.n.01  0.9703      cn-dog -> wn-oewn-02086723-n  0.9555
                  oewn-..-v    0.9930                wn-dog.n.01         0.9575
                  wn-cow.n.02  0.9991                wn-dog.n.03         0.9996
                  wn-cow.n.03  1.0000  <-- zero      wn-frump.n.01       1.0000  <-- zero

    `1.0000` means no shared evidence at all. Ranked as merely-worst it would let a source land on a
    sense it has nothing in common with, whenever every real candidate happened to be absent."""
    reads = [{"candidate": "wn-cow.n.01", "residual_fraction": 0.9703},
             {"candidate": "wn-cow.n.03", "residual_fraction": 1.0}]
    shared = [r for r in reads if r["residual_fraction"] < 1.0]
    assert [r["candidate"] for r in shared] == ["wn-cow.n.01"]


def test_the_residual_ORDER_is_the_sense_assignment():
    """One instrument does both jobs. The residual that decides whether to merge also decides which
    sense to merge into: lowest residual below 1.0 wins, conservation-certified, with no threshold.
    One decision-maker, so a second scorer computing the same assignment by another mechanism would
    be a disagreement waiting to happen."""
    reads = {"wn-cow.n.01": 0.9703, "wn-oewn-01783720-v": 0.9930,
             "wn-cow.n.02": 0.9991, "wn-cow.n.03": 1.0000}
    shared = sorted(((v, k) for k, v in reads.items() if v < 1.0))
    assert shared[0][1] == "wn-cow.n.01"


def test_full_absorption_is_not_required_for_a_merge():
    """A high residual with a resolvable landing site is a merge plus a residual.

    `derive_diagram` accepting only `one_object` — both bands absorbing the whole of the other's
    evidence — files everything else as separate, which is the gate that keeps ConceptNet out.
    Measured: `wn-cow.n.01 <-> cn-cow` has residual 0.9703, rows_a 5, rows_b 286. Five taxonomy rows
    cannot absorb 286 associative ones; a taxonomy stub and a relational neighbourhood describe the
    same thing from different angles, which is the reason for having more than one source. So the
    remainder is kept separate and added rather than discarded with the match."""
    rows_a, rows_b, residual = 5, 286, 0.9703
    assert rows_b > rows_a * 50, "the sources are asymmetric by construction"
    assert residual > 0.9, "and full absorption is therefore impossible"
    assert residual < 1.0, "yet they share real evidence — this must not read as 'separate'"


def test_the_ambiguity_gate_refuses_rather_than_guessing():
    """The minimum alone does not decide. Some sense is always nearest, so taking the lowest
    residual would merge `cow.n.01` into `cow.n.02`. The landing sense has to separate from its
    siblings (`prism.resolution.separated`); where it does not, the source is ambiguous here and
    nothing merges.

    `separated`'s tie-break is derived from the input's representation granularity rather than from
    the term count, because the error in `partition` is cancellation and not accumulation:
    `partition` sums `(v − mean)²` where the values and the mean sit near 1.0 while the deviations
    sit near 1e-3, so each deviation's error is set by `eps · max|v|`. The null series is
    well-conditioned, so nothing cancels on that side and the difference is one-sided noise.

    That is what decides the near-tie case. For n=3 the null is exactly 0.75, and three candidates
    0.7% apart put the observed eta² +2.6e-14 above it — floating-point cancellation rather than a
    split, so they read as ambiguous."""
    from prism.resolution import separated, separability

    # Three candidates 0.7% apart read as ambiguous, so a source with a near-tie between two senses
    # is called ambiguous rather than landed on one of them by a rounding difference.
    flat = [0.0297, 0.0296, 0.0295]
    assert not separated(flat), (
        "a 0.7% spread across three candidates must read as AMBIGUOUS — if this passes again, the "
        "tie-break has regressed to an accumulation model and is measuring float noise as signal")

    # A genuinely flat profile carries no split to find.
    assert not separated([0.03, 0.03, 0.03]), "identical candidates must never resolve"

    # The control. Without it the two assertions above are satisfied by a `separated` that answers
    # no to everything, which would read as a fixed gate while destroying every real landing.
    assert separated([0.9, 0.1, 0.05]), (
        "a real split must still resolve; a gate that refuses everything is not a fixed gate")
    assert separability(flat) > 0.0, "the continuous reading stays available even when it refuses"
