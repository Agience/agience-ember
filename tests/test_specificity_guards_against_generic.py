"""Specificity is not applied. Distance alone decides — and this file is why that is legible.

The IC specificity weight is absent from both `match.propagate` (`1.0 + log1p(IC)`) and
`activation.rank_fired` (`act * log1p(IC)`). This file pins that absence and keeps the property it
gave up executable, so what was traded away stays measured rather than untraceable.

## What was given up

A hypernym tree puts generic concepts near the root, where they sit a short geodesic distance from
almost anything. With distance alone, recorded in `match.propagate`:

    op.describe.generic   beat   op.describe.python     for "a source code module with functions"
    op.describe.generic   beat   op.describe.markdown   for "markdown prose document and text"

("describes source and text files" — `source`, `text` and `file` are all generic.) The weight
existed to fix exactly that, so that failure is expected to return.

## Why distance alone decides anyway

Two measurements against the live node-71 lattice:

  * IC is near-inert as a weight here. The basis is intrinsic IC (`1 - log(desc+1)/log(N+1)`,
    n=676,225), which is positive by construction: 0 of 676,225 synsets carry IC 0, minimum noun
    IC is 0.157, and 91.4% of nouns sit at IC exactly 1.0 (p25 through p99 all 1.000). The
    "~43% of synsets carry IC 0" the code cites belongs to frequency-based Resnik IC on classic
    WordNet 3.0 — a corpus and an IC source that are not in use. `ic_se`, the frequency channel,
    is populated on zero vertices.
  * Corpus surprisal is the intended replacement and cannot be relied on: `word_information`
    returns `None` without an FTS index, so specificity would be a constant wherever one is
    missing, including this test fixture. Silently inert is worse than honestly absent.

## What this file asserts

That specificity is genuinely not applied — no residue, no half-applied weight — so the state is
unambiguous, and that the probe still works, so re-enabling specificity is detected here rather
than discovered in a ranking.

## Reversing this

Delete the assertions below and restore either shape in `match.propagate` and
`activation.rank_fired`. The property to re-pin is the ordering one: a specific concept must
receive strictly more energy than a generic one at the same distance. `_specificity_of` is the
probe — `attenuate(0) == 1.0` exactly, so propagating unit energy from a node onto itself returns
that node's weight and nothing else.
"""
from __future__ import annotations

import pytest

from ember.ontology import match
from _fakes import _install_offline_wordnet


@pytest.fixture(scope="module")
def wn_ready():
    if not _install_offline_wordnet():
        pytest.skip("WordNet not available in this environment")
    return True


def _specificity_of(name: str) -> float:
    """The weight `propagate` gives this target, isolated: unit energy at distance 0."""
    energy, distance = match.propagate({name: 1.0}, [name])
    assert distance == 0.0, (
        f"{name} is not at distance 0 from itself (got {distance}) — the probe assumes "
        "`attenuate(0) == 1.0`, so a non-zero distance would leave an attenuation term in the "
        "number and this would stop measuring the weight alone."
    )
    return energy


# The taxonomy root (the most generic concept that exists), a mid node, and a leaf-ward one. If any
# weighting were applied, these three would differ.
_PROBES = ("entity.n.01", "animal.n.01", "dog.n.01")


def test_no_specificity_weight_is_applied(wn_ready):
    """Unit energy in, unit energy out — for a generic node and a specific one alike."""
    weights = {n: _specificity_of(n) for n in _PROBES}
    off_by = {n: w for n, w in weights.items() if abs(w - 1.0) > 1e-9}
    assert not off_by, (
        f"a specificity weight is being applied again: {off_by}.\n\n"
        "Unit energy propagated onto a node at distance 0 must come back as 1.0 — anything else "
        "is a weight. If specificity was deliberately restored, this file's docstring is the "
        "record of what it was traded against, and the ordering property is what to pin instead."
    )


def test_the_probe_still_measures_what_it_claims(wn_ready):
    """Control: without this, the test above would pass on a broken probe that returns 1.0 always.

    Energy must still fall off with distance — the screened propagator is untouched by the
    specificity removal, and a probe that could not see attenuation could not see a weight either.
    """
    near, d_near = match.propagate({"dog.n.01": 1.0}, ["dog.n.01"])
    far, d_far = match.propagate({"dog.n.01": 1.0}, ["tree.n.01"])
    assert d_near < d_far, "the fixture is not near/far — the probe proves nothing"
    assert near > far, (
        "energy did not fall off with distance, so this probe cannot distinguish a weight from "
        "an attenuation and neither assertion in this file means anything."
    )


@pytest.mark.xfail(
    reason="Accepted regression: specificity is not applied, so a generic concept is not scored as "
           "weaker evidence than a specific one. Recorded as xfail rather than deleted so the cost "
           "stays visible and `-rx` reports it on every run. An XPASS here means specificity has "
           "been re-introduced somewhere.",
    strict=True,
)
def test_a_generic_concept_would_be_weaker_evidence_if_specificity_were_applied(wn_ready):
    """The property that was given up. Kept executable so re-enabling it is a one-line proof."""
    assert _specificity_of("dog.n.01") > _specificity_of("entity.n.01")
