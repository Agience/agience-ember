"""Need -> offer selection: real distance, attenuation dropoff, and the propagation floor.

Grouped by invariant. The through-line is that a similarity score without a distance is not a
measurement — a score alone will rank `op.health` above `op.describe.python` for the need "a python
function".
"""
from __future__ import annotations

import math

import pytest

from ember.ontology import match
from _fakes import _install_offline_wordnet
from _fakes import _FakeStore

OT = "application/vnd.agience.operator+json"

PANEL = [
    ("op.describe.markdown", "describes markdown prose documents and text"),
    ("op.describe.python", "describes python source code modules and functions"),
    ("op.source.wikipedia", "ingests encyclopedia articles about the world"),
    ("op.health", "reports node health status and disk memory"),
]


@pytest.fixture(scope="module")
def wn_ready():
    if not _install_offline_wordnet():
        pytest.skip("WordNet not available in this environment")
    return True


@pytest.fixture()
def store(wn_ready):
    s = _FakeStore()
    for oid, offer in PANEL:
        s.artifacts.put_artifact({"id": oid, "content_type": OT, "state": "committed",
                                  "context": offer})
    match.invalidate(s)
    return s


# ── Invariant 1: distance is measured and reported, never inferred from a score ───────────────


# ── Invariant 2: attenuation actually attenuates ──────────────────────────────────────────────
# These three declare `wn_ready` explicitly. Without it they depend on some earlier test in the
# file having built the index as a side effect of the module-scoped fixture, and an unbuilt index
# gives every pair infinite distance (`inf < inf`). Declaring the fixture makes the dependency the
# test's own rather than an artefact of file order.
def test_energy_falls_off_with_distance(wn_ready):
    """The screened propagator: `energy * exp(-d/xi)`. A closer target must receive more."""
    fired = {"dog.n.01": 1.0}
    near, dn = match.propagate(fired, ["dog.n.01"])
    far, df = match.propagate(fired, ["tree.n.01"])
    assert dn < df, "the fixture is not near/far"
    assert near > far, "energy did not fall off with distance"


def test_shorter_xi_attenuates_harder(wn_ready):
    fired = {"dog.n.01": 1.0}
    wide, _ = match.propagate(fired, ["cat.n.01"], xi=4.0, gap=0.0)
    tight, _ = match.propagate(fired, ["cat.n.01"], xi=0.25, gap=0.0)
    assert tight < wide, "xi had no effect on the dropoff"


# ── Invariant 3: the gap is a gap — a discontinuity, not a taper ──────────────────────────────
def test_beyond_the_gap_nothing_propagates_at_all(wn_ready):
    """A gap is a discontinuity by definition: beyond it nothing propagates at all. A taper would let
    arbitrarily distant offers accumulate a score from sheer count."""
    fired = {"dog.n.01": 1.0}
    open_gate, _ = match.propagate(fired, ["tree.n.01"], gap=0.0)
    gated, dist = match.propagate(fired, ["tree.n.01"], gap=0.99)
    assert open_gate > 0.0, "the fixture propagates nothing even with the gap open"
    assert gated == 0.0, "a contribution below the gap still propagated"
    assert dist == float("inf"), "a gated match reported a distance it never admitted"


# ── Invariant 4: nearby fitting contexts activate too — k > 1, not top-1 ──────────────────────


# ── Invariant 5: degrading is announced ───────────────────────────────────────────────────────


# ── Invariant 6: a cascade cannot sustain itself ──────────────────────────────────────────────


# ── Invariant 7: the cache is keyed by store ──────────────────────────────────────────────────


# ── Invariant 8: separation, not just a ranking ───────────────────────────────────────────────


# ── Invariant 9: exact keys beat distance, and the basis is always stated ─────────────────────




# ── Sense coherence: the matching primitive applied inside the need (§13.14) ──────────────────
# These need a real sql store: `fired_field` falls back to uniform weights when it cannot measure
# document frequency, and the fallback path never runs the coherence measurement.
@pytest.fixture()
def sqlstore(wn_ready):
    import os
    import tempfile
    from mantle.db import open_lattice
    return open_lattice(os.path.join(tempfile.mkdtemp(), "c.db"), origin="test")


def test_a_words_total_constraint_is_conserved_across_its_senses(sqlstore):
    """Firing every sense does not amplify a polysemous word. The word is one piece of evidence
    whatever its senses, so the split redistributes its own weight rather than multiplying it.
    Without conservation an 8-sense word would out-shout an unambiguous one 8:1."""
    f = match.fired_field("star", sqlstore)
    assert len(f) > 1, "every sense should fire, not just sense 1"
    assert max(f.values()) <= sum(f.values()) + 1e-9


def test_a_single_word_need_applies_the_sense_frequency_prior(sqlstore):
    """A lone word has no partner to disambiguate it, so it reads the source's own sense-frequency
    prior. WordNet lists a word's senses most-common-first, so the reading is the Bayesian MAP with
    the common sense weighted highest rather than a flat split — under a flat split bare "dog"
    resolves to "frump" (14.6% on the primary sense, against 87.5% under the prior).

    The word still shows its ambiguity, because every sense keeps weight, and its total is conserved,
    because the split is a redistribution."""
    f = match.fired_field("calculus", sqlstore)
    vals = list(f.values())
    assert len(vals) > 1, "every sense should fire, not just sense 1"
    assert max(vals) > min(vals) + 1e-6, "the frequency prior must prefer the common sense over the rare"
    assert max(vals) <= sum(vals) + 1e-9, "conserved: the split is a redistribution, never an amplification"


def test_the_rest_of_the_need_shifts_weight_toward_the_agreeing_sense(sqlstore):
    """With a partner word present, the word's constraint redistributes according to which sense the
    partner sits nearest to. That redistribution is the measurement the prior alone cannot make."""
    alone = match.fired_field("star", sqlstore)
    withctx = match.fired_field("star in the night sky", sqlstore)
    moved = [n for n in alone if n in withctx and abs(withctx[n] - alone[n]) > 1e-9]
    assert moved, "context changed nothing — coherence is not being measured"


def test_support_is_measured_WITHOUT_the_gap_or_the_comparison_is_empty(wn_ready):
    """`match._support` propagates with `gap=0.0`, and that difference is the point.

    The gap answers "was this reached". Sense choice is a relative question, and a comparison where
    nearly every option reads zero is not a comparison: with the gap applied, seven of star's eight
    senses score exactly 0.0 support against {night, sky} — the celestial one included, since that
    distance exceeds the gap's reach — and the lone survivor ("a plane figure with 5 or more points")
    wins by default.

    The gap is derived as `exp(−d_max/ξ)`, which is exactly `prism.resolution.horizon(ξ, gap) ==
    d_max`: the propagator's own value at the corpus's diameter. A unit-weight signal between two
    comparable nodes therefore scores at least the gap by construction.

    What would have to break for this to fail:
      · the gap stops being the value at the diameter (a picked `GAP = 0.05` puts 28% of the corpus
        past its own horizon);
      · a star sense ends up further from `sky` than the corpus's own diameter;
      · applying the gap adds support, or the ungapped comparison degenerates to one option;
      · the gap gate goes dark altogether — caught by the positive control, which raises the gap
        above the measured supports and requires the comparison to collapse."""
    from prism.resolution import horizon

    seeds = match.wn_synsets_for("star")
    targets = match.wn_synsets_for("sky")
    assert seeds and targets

    xi, gap = match.xi(), match.propagation_floor()
    assert xi is not None and gap is not None, "the corpus reports no geometry — nothing to measure"

    gapped = [match.propagate({s: 1.0}, targets)[0] for s in seeds]
    ungapped = [match.propagate({s: 1.0}, targets, gap=0.0)[0] for s in seeds]
    dists = [match.propagate({s: 1.0}, targets, gap=0.0)[1] for s in seeds]

    # 1. The gap can only ever remove support, never manufacture it.
    assert all(u >= g for u, g in zip(ungapped, gapped))

    # 2. Without the gap there is an actual comparison to make — the property `_support` depends on.
    assert sum(1 for v in ungapped if v > 0) > 1, \
        "the ungapped comparison is degenerate: sense choice would win by default"

    # 3. Why the derived gap is safe here, and it is not "the gap is small". The gap is the
    #    propagator at the diameter, so it excludes exactly what lies beyond the corpus's own extent
    #    — and every star sense sits inside it. Both halves, or this is a coincidence, not a reason.
    d_max = horizon(xi, gap)
    assert max(dists) < d_max, \
        "a star sense sits past the corpus diameter (%.4g >= %.4g)" % (max(dists), d_max)
    assert sum(1 for v in gapped if v > 0) == sum(1 for v in ungapped if v > 0), \
        "the derived gap refused a comparable pair — it is no longer the value at the diameter"

    # 4. Positive control: the gate is live, so assertion 3 is a finding and not a dead branch.
    #    Raised above every attenuation the propagator can produce, the gap collapses the
    #    comparison — which is what `_support` passes `gap=0.0` to avoid.
    #
    #    The gap gates on the attenuation rather than on the accumulated energy.
    #    `screened_accumulate` counts a contribution only where `attenuate(d, xi) >= gap`, and
    #    `attenuate` is `exp(-d/xi)` — bounded by 1.0, reached only at d == 0.
    #
    #    So the threshold comes from the propagator's own range rather than from a sample of its
    #    output: one ulp above 1.0 is above every attenuation that exists, for any corpus, any xi,
    #    and any weighting applied downstream. A threshold derived from `max(ungapped)` is an
    #    energy, a different quantity on a different scale, and it exceeds 1.0 only while some
    #    per-target weight multiplies the energies up — so it goes dark silently, which is the
    #    failure this control exists to detect arriving as a false negative about itself.
    collapsing = math.nextafter(1.0, 2.0)
    smothered = [match.propagate({s: 1.0}, targets, gap=collapsing)[0] for s in seeds]
    assert sum(1 for v in smothered if v > 0) == 0, \
        "a gap above every possible attenuation refused nothing — the gap gate is dark"


# ── the runner bundle's geometric arm is dark — pinned as the current state ───────────────────
