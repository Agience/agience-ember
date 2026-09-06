"""`ember.optics` is a valid embodiment for a crystal, and a real crystal runs on it.

This is the half of crystal's injection proof that names the instrument, and it lives here because
the arrow points this way: ember declares `agience-crystal` and imports it, crystal declares
nothing of ember and imports nothing of it. Asserting the pair from crystal's suite required a
checkout of the repository ABOVE it — which made crystal's CI depend on a private sibling and, on a
fork, unable to run at all.

What stays in crystal is the other half, and it is the half that proves injection is real: a stub
embodiment written against the contract in numpy and the stdlib, which crystal runs on without
knowing anything about it. Crystal proves the slot is a slot. This file proves the thing the host
puts in it fits.

    crystal/tests/test_embodiment_injection.py   the slot, and a stub that fills it
    this file                                    the instrument the host actually hands over

**One claim did not survive the split, and it is named rather than quietly dropped.** Crystal's
file used to parametrise every flow assertion over BOTH embodiments and then compare their outputs
directly — same invariants, `k` free to differ, because the two use different rank rules on
purpose. That comparison needs both implementations in one process, and the two now live in
different repositories, with crystal's stub reachable only from crystal's own suite (a wheel ships
no tests). What is asserted here is that the instrument satisfies the contract and holds the
invariants; what is no longer asserted anywhere is that it and the stub agree on structure while
differing on `k`. If that cross-check is wanted back, the stub belongs in `prism` — it is written
against `prism.embodiment` and neither repository owns it — not copied into a second suite where
the copies can drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from crystal import Crystal
from prism import conservation as prism_conservation
from prism.embodiment import (
    CONSERVATION_MEMBERS,
    Embodiment,
    EMBODIMENT_MEMBERS,
    Ledger,
    members_of,
)

from ember import optics as ember_optics

SPEC = {
    "name": "crystal.test.injected",
    "facets": [{"name": "a", "direction": "both"}, {"name": "b", "direction": "both"}],
    "tektons": [{"name": "sage", "domain": "test"}],
    "organons": [{"name": "op.retrieve", "requires": ["store.read"]}],
    "created_by": "author@example.com",
}

# an ordered (T=6, D=3) frame with structure on every axis
_F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=float)


def _bound(spec=SPEC) -> Crystal:
    """A crystal with the real instrument injected — what a full node assembles.

    The modules go in unadapted: `ember.optics` is an `Embodiment` and `prism.conservation` is a
    `Conservation`, because `prism/embodiment.py` names its members to match the implementation
    that already existed rather than the reverse.
    """
    c = Crystal(spec, embodiment=ember_optics, conservation=prism_conservation)
    c.bind("a", entry=lambda x: np.asarray(x, float), inverse=lambda X: X)
    c.bind("b", entry=lambda x: np.asarray(x, float), inverse=lambda X: X)
    return c


# ── the contract ────────────────────────────────────────────────────────────────────────────────

def test_the_aperture_satisfies_the_prism_contract():
    """Structural, and it is what makes `ember.optics` usable unadapted."""
    assert isinstance(ember_optics, Embodiment)
    assert members_of(ember_optics, "embodiment") == EMBODIMENT_MEMBERS
    assert isinstance(prism_conservation.PathLedger(_F, at="x"), Ledger)


def test_no_single_module_fills_conservation_and_that_is_the_design():
    """`Conservation` is filled by two modules together, and a host injects both:

        prism.conservation  ->  energy, PathLedger      (numpy, and it may never import an instrument)
        ember.optics        ->  entropy_bits            (an adapter over the instrument's entropy)

    `entropy_bits` is the one accounting member a real implementation may legitimately not have.
    Having `prism.conservation` implement `−Σ p log₂ p` on the numpy it already carries would make
    the contract fillable whole — and would forfeit the property that makes `entropy_bits` worth
    having: a caller's entropy and the instrument's own internal entropy over geometry marginals
    and mode weights are the same callable and cannot drift. `ember/optics.py` clips at `1e-12` and
    guards at `1e-30`; a restatement in prism would have to match those exactly, forever, or
    diverge silently.

    This assertion moved here from crystal's suite with the rest of the instrument arm: it reads
    `ember.optics`, so crystal could only make it by importing a package it must not name.
    """
    from prism.instrument import _FILLED_BY

    prism_side = set(members_of(prism_conservation, "conservation"))
    instrument_side = set(members_of(ember_optics, "conservation"))

    assert prism_side and instrument_side, "both sides must fill something, or the split is imaginary"
    assert not (prism_side & instrument_side), (
        "the two modules OVERLAP on %s — a member filled twice is a member that can drift, which is "
        "the whole reason entropy_bits was not re-implemented" % sorted(prism_side & instrument_side))
    assert prism_side | instrument_side == set(CONSERVATION_MEMBERS), (
        "together they must fill the contract exactly; missing %s"
        % sorted(set(CONSERVATION_MEMBERS) - (prism_side | instrument_side)))

    # The control: neither alone is enough. Without it, the assertions above are satisfied by a
    # contract of one member that one module happens to hold.
    assert prism_side != set(CONSERVATION_MEMBERS), "prism alone would make this test meaningless"
    assert instrument_side != set(CONSERVATION_MEMBERS), "the instrument alone likewise"
    assert "conservation" in _FILLED_BY


# ── the flow, through the real instrument ───────────────────────────────────────────────────────

def test_a_crystal_conducts_condenses_and_transmits_on_the_instrument():
    c = _bound()
    c.conduct("a", _F)
    absorbed, transmitted = c.condense(), c.transmit()
    assert np.asarray(transmitted).shape == _F.shape
    if absorbed is not None:
        assert np.asarray(absorbed).shape == _F.shape
        # the split is orthogonal — the property, not the number
        assert abs(float((np.asarray(absorbed) * np.asarray(transmitted)).sum())) < 1e-8


def test_energy_is_conserved_at_the_crossing():
    c = _bound()
    c.conduct("a", _F)
    led = c.ledger()
    assert led["conserved"] is True, led
    assert led["incident"] > 0.0
    assert abs(led["incident"] - (led["absorbed"] + led["transmitted"])) < 1e-8 * led["incident"]
    cert = c.certificate()
    assert cert["balanced"] is True, cert["why"]
    # the tolerance is derived from the arithmetic performed, so it scales with the frame rather
    # than being a constant that happens to pass
    assert 0.0 < cert["tolerance"] < led["incident"]


def test_the_coupling_sign_is_measured_not_declared():
    c = _bound()
    c.conduct("a", _F)
    c.conduct("b", _F)
    assert c.couple("a", "b") > 0.0, "identical sides attract (+)"
    c.clear("b")
    c.conduct("b", -_F)
    assert c.couple("a", "b") < 0.0, "opposed sides detract (−)"


def test_silence_stays_silence():
    """A surface that fires nothing is an empty crossing, trivially conserved — never a crash and
    never a synthesised answer."""
    c = Crystal(SPEC, embodiment=ember_optics, conservation=prism_conservation)
    c.bind("a", entry=lambda x: None, inverse=lambda X: X)
    assert c.conduct("a", "zzqxwv plorbnak") is None
    assert c.condense() is None
    assert c.transmit() is None
    assert c.ledger() == {"incident": 0.0, "absorbed": 0.0, "transmitted": 0.0,
                          "k": 0, "conserved": True}
    assert c.certificate()["balanced"] is True


def test_an_unreadable_frame_returns_the_null():
    """One row cannot carry a read. The instrument must say so by returning None from
    `absorb_transmit`, and the crystal must then propagate the frame whole — the entire incident
    energy transmits on, unabsorbed, rather than a zero band being invented for it."""
    c = _bound()
    c.conduct("a", np.array([[1.0, 2.0, 3.0]]))
    assert c._crossing() is None
    led = c.ledger()
    assert led["k"] == 0 and led["absorbed"] == 0.0
    assert led["transmitted"] == led["incident"] > 0.0
    assert led["conserved"] is True


def test_identity_is_unchanged_by_holding_the_instrument():
    """The point of the slot: the crystal's shareable identity is a property of its structure, so a
    node with the instrument and a store with a reduced embodiment address the same crystal."""
    c = _bound()
    assert c.sha == Crystal(SPEC).sha
    assert c.artifact() == Crystal(SPEC).artifact()
    assert c.required_capabilities() == ["store.read"]


# ── the control lives next door ─────────────────────────────────────────────────────────────────
#
# Every assertion above would hold against something that merely answers the protocol, so the proof
# that `membrane_screen()` hands back the REAL instrument is what makes them mean anything. That
# assertion already exists, in `test_ember_holds_the_instrument.py` — search it for `membrane_screen`.
# It moved out of crystal for the same reason this file did, and it is not restated here: a second
# copy is a copy that can drift, and it reads `__module__` off a CLASS, which is the detail a
# restatement gets wrong (writing `type(screen).__module__` reads `builtins` and fails on a correct
# instrument — that file records the measurement).
