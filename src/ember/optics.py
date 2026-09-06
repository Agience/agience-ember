"""The single seam onto entroptics. Every read in this codebase goes through here.

Why the instrument is a wrapper
-----------------------------
Three defects arise at a call site that reaches for entroptics directly, and each of them is
silent:

1. **Wrong door.** `entroptics.read()` / `Screen(W)` bypass the streaming front door and apply the
   entropy fold guard (`screen.py:619`, `fold = H_F < log2F - band_F`), which the library itself
   documents as the guard that destroys a sparse carrier (`batch.py:293`: *"Unlike the entropy
   (H_F) guard, this does NOT fold a sparse carrier (which averaging would dilute)"*). Ontology
   coordinates are sparse. Measured on a named-anchor frame, the entropy guard folded 256 feature
   channels into F_eff = 1 and reported `K_signal = 1` — a total loss of the basis,
   indistinguishable at the call site from "there is one real mode".

2. **A dimensionful floor read as if it were dimensionless.** `Screen.noise_floor` is a
   singular-value floor: it scales with the magnitude of the whitened screen. Per-channel MAD
   whitening amplifies a frame with heterogeneous channel occupancy by ~1e12 (see the measurement
   below), so the floor moves with it. Comparing that number across two differently scaled frames
   measures the scale, not the conditioning. `K_signal` over the same pair barely moves.

3. **The i.i.d.-Gaussian null on correlated data.** `spectral_optics`' own docstring says the
   default `mp` floor *"conflates bulk correlation with signal"* and directs the caller to
   `null_providers.permutation()` for correlated data. Retrieved evidence rows are correlated
   by construction — that is what retrieval means.

Measured (Unit AA, 32 real documents from the lattice corpus extract; see
`agience-ember/scripts/aa_fold_probe.py`)::

    D=256   oov=surface   Screen: K=44  floor=8.5e-1     spectral: K=45  floor=1.36
    D=256   oov=skip      Screen: K=50  floor=4.3e+12    spectral: K=34  floor=1.46
    D=2048  oov=surface   Screen: K=53  floor=1.2e-1     spectral: K=278 floor=2.14
    D=2048  oov=skip      Screen: K=62  floor=6.2e+11    spectral: K=210 floor=2.50

The Screen floor moves 13 orders of magnitude between two frames whose `K_signal` moves by 6
modes. The spectral floor holds still, because a correlation matrix has unit diagonal and
per-channel scale cancels exactly. That is why `k_signal` below is the spectral read.

The ordered-axis contract
-------------------------
Axis 0 is the ordered axis (`_T`), axis 1 is the feature axis (`_F`). Coherence is a lag-1
statistic: `mean_i Re<row_i, row_{i+lag}>²` — squared, over a symmetric row-Gram — so row reversal
is exactly invariant (bit-identical), as is global negation. Shuffling is what destroys it
(measured 11.32 -> 0.42). Passing an unordered collection is a meaningless read rather than a
degraded one, so `read_ordered` takes a `Sequence`, and a `set` / `frozenset` / `dict_keys` / bare
iterator raises `UnorderedInput`.

Where the instrument lives
------------------------
`prism.instrument`'s table names the two embodiments and their domains: the instrument (ember), whose
domain is the signal — an ordered (T, F) frame — against beacon (mantle), whose domain is the
corpus. ember's four signal modules (`signal/projection`, `signal/pooling`, `signal/forgetting`,
`consolidate/*`) are the heaviest callers, and ember carries the numpy floor the reads need.
`chorus` and `crystal` reach the instrument only from their tests, as hosts: `chorus -> ember` and
`crystal -> ember` are both 0 in non-test `src/`, ratcheted.

`ember/__init__.py` registers this module as the process-default instrument through
`prism.instrument.set_default`, passing a factory so that `import ember` does not pull entroptics
and numpy into a process that only wanted the cache. A measurement site resolves its instrument in
that order: an explicit keyword argument, then the process default, then no reading.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import math

import numpy as np

# The floating-point rounding law, from prism's dependency-free base — see `_float_noise`.
# `ember` already depends on `agience-prism-py[trust,vector,wire]`; this import is on the base, so
# it costs nothing an install here does not already have.
from prism.rounding import split_walk_rounding


# ═══════════════════════════════════════════════════════════════════════════════
# Binding the entroptics that the measurements were verified against
# ═══════════════════════════════════════════════════════════════════════════════
# entroptics is the source of every measured value here, so *which* entroptics is a correctness
# question rather than a packaging convenience. Two things can win the import: a `pip install
# entroptics` wheel in `site-packages`, or a second source checkout elsewhere on the box. In either
# case an edit made in the tree you are reading changes nothing while appearing to.
#
# What is checked is identity, and location is only one signal of it. A pure location test rejects
# every correct off-workspace install — ember on a container, a CI runner or an edge node resolves
# entroptics from site-packages by construction — so two direct signals carry the check:
#
#   1. **The version**, pinned here to a major.minor. Catches an install that moved underneath us.
#   2. **The numbers**, pinned in `prism.vectors/screen_read_vectors.json`. entroptics 0.2.0
#      shipped three value changes behind an unchanged API, so a matching version is necessary and
#      not sufficient — the vectors are what close it.
#
# The rival-source-checkout check below is derived (computed from this file's own location, so
# moving the workspace preserves it) and self-disables when the intended tree is absent — which is
# exactly the off-workspace install it must let through.
#
# The intended tree is `entroptics`, a MEMBER of this workspace since the three workspaces merged
# into `Repos/agience/` (2026-08-23). `agience-entroptics` is the older copy and is named below as
# a rival: it is still a source checkout, still imports, and still reports 0.2.0, so nothing but
# this check tells them apart. They are not the same code — measured, they differ by 948 lines in
# `proximity.py` and 257 in `instrument.py` — which is exactly why a version cannot arbitrate between
# them and why the vectors are the second half of this binding.
#
# The guard self-disables when the intended tree is absent, so a wheel passes — and that makes the
# repo name here load-bearing in a way no test failure announces. A stale name leaves
# `os.path.isdir(want)` False on every box, so the guard takes its off-workspace-install branch and
# admits any checkout, which is the one thing it exists to refuse. `ci_runner`'s PYTHONPATH carries
# the same name, so if the entroptics repo is renamed both move together.
#
# This seam is the only import site (the AST test in `tests/test_one_instrument.py` enforces that), so
# verifying the binding once here covers the whole codebase.
ENTROPTICS_REQUIRED = "0.2"      # major.minor; matched exactly, patch releases float

# Where the names live, because a patch release moved them and the next edit will want to move them
# back. 0.2.1 narrowed the top-level re-exports: `concentration`, `decay`, `etendue`,
# `mercer_certificate`, `phi_F`, `phi_T`, `principal_directions`, `scale_profile`,
# `space_bandwidth` and `strehl` are imported below from `entroptics.reads`, which is where they are
# DEFINED — `entroptics.instrument` also re-exports them, so importing from there would work and would
# be one more indirection to keep in step. `sequence` is a submodule and is imported as one.
#
# Nothing was removed in that release and no measured value moved: all seven cases in
# `screen_read_vectors.json` reproduce exactly, on both engines. The change surfaced as ImportErrors
# at collection — the loudest failure available — which is the two-check design working. The version
# alone would not have caught it, because "patch releases float" is a statement about what a patch
# release is SUPPOSED to carry, and this one carried a surface change.


def _bind_local_entroptics() -> str:
    import entroptics                                     # the one eager import; this file IS the seam
    src = os.path.abspath(entroptics.__file__).replace("\\", "/")
    low = src.lower()

    have = getattr(entroptics, "__version__", "")
    if ".".join(have.split(".")[:2]) != ENTROPTICS_REQUIRED:
        raise ImportError(
            "entroptics %s is installed; the instrument's measurements are pinned to %s.x.\n    %s\n"
            "Every number the instrument reports comes from this library, and 0.2.0 shipped value changes "
            "that carried no API change — so the version is the coarse check and "
            "`prism.vectors/screen_read_vectors.json` is the fine one. Install the matching version, or "
            "move the pin and the vectors together in one commit."
            % (have or "(no __version__)", ENTROPTICS_REQUIRED, src))

    # …/agience-ember/src/ember/optics.py -> the workspace is three hops up, and the entroptics
    # checkout is a MEMBER of it (it was a sibling one further out until the workspaces merged into
    # `Repos/agience/`). Derived by construction rather than written as a
    # literal, so moving the pair together preserves the check. The hop count is why the instrument
    # sits at `ember/optics.py` and not a directory deeper: from `ember/signal/optics.py` the same
    # expression resolves a level too high, `os.path.isdir` is False for every candidate, and this
    # guard switches itself off instead of failing. Count the hops before moving this file.
    _workspace = os.path.join(os.path.dirname(__file__), "..", "..", "..")

    def _root(*parts):
        return os.path.abspath(os.path.join(*parts)).replace("\\", "/").lower().rstrip("/") + "/"

    want = _root(_workspace, "entroptics", "src")
    stale = _root(_workspace, "agience-entroptics", "src")

    # Enforced only against a tree that is actually on disk. An off-workspace install — a wheel, a
    # CI container — matches neither and must pass; the version check above is what covers it there.
    if os.path.isdir(want) and not low.startswith(want):
        raise ImportError(
            "entroptics resolved to a DIFFERENT source checkout:\n    %s\nthe instrument must bind "
            "the current tree:\n    %s\n%sA rival checkout is still a source checkout, so this is "
            "not a packaging nit: every measured value here comes from whichever tree wins the "
            "import, and edits to the other one change nothing while appearing to. Re-point the "
            "editable install (`pip uninstall entroptics && pip install -e <workspace>/entroptics`)."
            % (src, want,
               ("That path is `agience-entroptics`, the older copy: same version, different code.\n"
                if low.startswith(stale) else "")))
    return src


# The resolved source path, verified at import. Runs once when the seam is first imported (which is
# whenever any measurement is taken), and is exposed so a caller / health check can see exactly which
# entroptics tree the instrument is reading through.
ENTROPTICS_SOURCE = _bind_local_entroptics()


# ═══════════════════════════════════════════════════════════════════════════════
# The membrane — the Screen object model, a second door
# ═══════════════════════════════════════════════════════════════════════════════
# Two doors live in this module and they answer different questions. Everything above is the
# projection read: hand it an ordered (T, F) frame, get back an `OpticsRead` — a spectral summary.
# Below is the folded `Screen` object model: `place`/`register`/`render`/`couple`/`transfer`/
# `certify` — the two-way membrane where facets meet. `crystal/crystal.py` uses all thirteen of
# those methods, and an `OpticsRead` carries no placement or coupling concept, so the two surfaces
# stay distinct.
#
# The membrane is reached by an explicit call and is out of `__all__`, so `from ember.optics import
# …` offers only the measurement reads. §1 of this module's header names `Screen(W)` as the door
# that applies the entropy fold guard which destroys a sparse carrier (256 channels folded to
# f_eff=1, reported as K_signal=1) — a caller who wants the membrane names it, and a caller who
# wants a measurement cannot arrive here by autocomplete. It lives in this file rather than an
# `ember/membrane.py` because the one-instrument rule (`tests/test_one_instrument.py`, `_ALLOWED =
# {"optics.py"}`) allows exactly one module to import entroptics.
def membrane_screen():
    """The folded `Screen` class — the membrane's object model, distinct from the measurement
    instrument.

    Use this to build a two-way membrane (facet placement, coupling, transfer). To measure
    structure in a frame, use `read_ordered` / `resolvable` / `principal_directions` above — those
    read the scale-invariant correlation spectrum and leave the feature axis unfolded."""
    from entroptics.screen import Screen
    return Screen


def membrane_types():
    """The membrane's result types, for callers that must `isinstance`-check what a Screen returns
    (`ScreenRead`, `Balance`, `Transfer`, `Coupling`, and the three law records). Same caveat as
    `membrane_screen`: these belong to the folded Screen, not to the projection read."""
    from types import SimpleNamespace
    from entroptics.screen import (Lens, ScreenRead, Balance, Transfer,
                                   Linearity, Realisation, Losslessness)
    from entroptics.reads import Coupling
    return SimpleNamespace(Lens=Lens, ScreenRead=ScreenRead, Balance=Balance, Transfer=Transfer,
                           Linearity=Linearity, Realisation=Realisation,
                           Losslessness=Losslessness, Coupling=Coupling)

__all__ = ["OpticsRead", "read_ordered", "UnorderedInput", "DegenerateFrame",
           "ENTROPTICS_SOURCE", "principal_directions", "absorb_transmit", "propagate_residual",
           "route_by_coupling", "next_by_coupling",
           "resolvable", "correlation_length", "diffraction", "coherence",
           "spots", "fill", "scales", "correlated_null", "derived_null",
           "embed", "fit_dynamics", "dynamics_state", "decay_profile", "resolution_limit",
           # The ordered-stream operator and its own bound: `roll()` stays within the horizon, so
           # the bound applies whether or not the caller reads it.
           "OperatorRead", "sequence_operator", "surrogate_significance",
           # Screen normalisation — MAD whitening on the whole frame. Per-vector normalisation is
           # how cosine similarity gets written, and cosine is out of scope in the reasoning domain.
           "screen_normalize",
           "scale_read", "certificate", "entropy_bits", "self_information_bits",
           # Proximity — the collection digest and its probe. The one entroptics surface mantle is
           # built to consume through injected seams and cannot import for itself.
           "proximity_read", "proximity_engine_id", "proximity_probe_factory", "spectral_distance",
           # The streaming surface — the screen accumulates; condensation is an event
           "accumulator", "stream", "beam",
           # Set reads (order-invariant) — see the "which axis" note at the foot of this module
           "concentration", "mode_significance", "error_bar",
           # The membrane — the folded Screen object model, reached by an explicit call. The types
           # themselves stay unexported.
           "membrane_screen", "membrane_types"]

# ═══════════════════════════════════════════════════════════════════════════════
# Degeneracy is computed from the frame, not compared against an integer
# ═══════════════════════════════════════════════════════════════════════════════
# `OpticsRead.coherence` is a z against the exact row-permutation null
# (`entroptics.projection.coherence`), and its variance is the Cliff-Ord / Mantel second moment —
# an expectation over ordered pairs of lag-pairs. Its highest-order term is the disjoint one,
#
#     E_disj = (S1² - 4U + 2S2) / (N(N-1)(N-2)(N-3)),
#
# which ranges over two lag-pairs sharing no row: 2 pairs × 2 rows = four distinct rows. At N = 3
# that denominator is zero — there is no disjoint pair of pairs to average over — so the exact
# variance cannot be assembled and the statistic has no value.
#
# The instrument does not announce that absence; it returns a finite, plausible `0.0`. Measured
# across twelve trials at each size:
#
#     T=2  walk coherence 0.0000 (sd 0)   noise 0.0000 (sd 0)   identical — unmeasurable
#     T=3  walk coherence 0.0000 (sd 0)   noise 0.0000 (sd 0)   identical — unmeasurable
#     T=4  walk coherence -1.6017         noise -0.1922         discriminates
#
# A strongly lag-1-coherent random walk and i.i.d. noise read exactly equal, with zero variance, at
# T <= 3 — and that fabricated 0.0 is indistinguishable from the real reading of a frame with no
# lag-1 correlation. So the degeneracy is computed rather than waited for.
#
# The algebra is evaluated on the frame instead of being solved once and frozen. The exact
# permutation variance divides by the falling factorial `N(N−1)(N−2)(N−3)`, a quantity of this
# frame, computable from its own shape; when it is zero the variance is a division by zero. A
# denominator either vanishes or it does not, and no shape is compared against a chosen integer.
# `4` appears nowhere below: it is a consequence of the algebra, recovered by asking the algebra
# (see `MIN_ROWS`), never an input to it. `_carries_read` is the whole of the predicate, and what
# it produces is an absence the caller reads (`None` / `screened=False`).
#
# The boundary is a precondition rather than a quality bar. It says the statistic is defined; how
# good a short read is comes from `k_band` / `k_lo,k_hi` / `k_margin_last`, which report honestly on
# a short frame (a T=34 turn gives band=16.26 and certifies nothing).

def _null_dof(W: np.ndarray) -> tuple:
    """`(ordered_dof, feature_dof)` — the degrees of freedom this frame gives the two nulls.

    `ordered_dof` is the falling factorial `T(T−1)(T−2)(T−3)`: the denominator of the disjoint term
    of the Cliff-Ord / Mantel second moment, which ranges over two lag-pairs sharing no row (2 pairs
    × 2 rows). Zero means there is no disjoint pair of pairs to average over and the exact
    permutation variance cannot be assembled.

    `feature_dof` is `F(F−1)`: the number of ordered off-diagonal entries of the feature correlation
    — the structure a correlation read is about. Zero means the matrix is all diagonal and there is
    no correlation to resolve.

    Both are counts of terms in the estimator, taken from the frame's own shape. Neither is a
    threshold, a budget, or a policy: an estimator with no terms has no value."""
    T = int(W.shape[0]) if W.ndim >= 1 else 0
    F = int(W.shape[1]) if W.ndim >= 2 else 0
    return (T * (T - 1) * (T - 2) * (T - 3), F * (F - 1))


def _carries_read(W: np.ndarray) -> bool:
    """Can this frame carry a read at all? True iff both nulls have terms and the frame is not
    silent. The single degeneracy predicate in this module — every read below asks it instead of
    comparing a shape against an integer."""
    o, f = _null_dof(W)
    return o > 0 and f > 0 and bool(np.any(W))


def _smallest_readable_rows() -> int:
    """The smallest row count at which `_null_dof` is non-zero — recovered from the algebra rather
    than typed. Ascends from an empty frame and returns the first `T` whose falling factorial
    survives, so a change in the statistic's dof follows through with no edit here.

    Exported as `MIN_ROWS` for readers that want to know where the boundary falls
    (`ember/signal/projection.py` imports it to explain an absence to a caller). It is a report
    rather than a gate: nothing in this module compares against it."""
    # The loop needs no iteration bound: a falling factorial of four consecutive integers is
    # strictly increasing once positive, so it terminates by the algebra itself, and a `T > 64`
    # guard here would be an invented limit standing in front of the instrument.
    T = 0
    while T * (T - 1) * (T - 2) * (T - 3) <= 0:
        T += 1
    return T


MIN_ROWS = _smallest_readable_rows()          # == 4, computed, not written down


# ═══════════════════════════════════════════════════════════════════════════════
# The false-alarm level lives in the null
# ═══════════════════════════════════════════════════════════════════════════════
# A read here takes two external inputs: the local resource envelope and the caller's noise
# provider. A false-alarm level is a property of the null rather than a third input, and entroptics
# is explicit that it is (`null_providers.py`, its own header):
#
#     "The false-alarm level (alpha, `far`) travels WITH the null, not beside it: the cutoff is ONE
#      decision, so the provider owns both the threshold and the alpha it is drawn at (`ctx.far` is
#      the read's target; a provider may honour it or pin its own)."
#
# So no read in this module takes a `far=` keyword beside `null=`. The two would be one knob exposed
# twice, and the two copies disagree by construction: `permutation(far=0.01)` pins its own level and
# ignores the `far` a caller passes to the read, so `read_ordered(rows, null=permutation(far=0.01),
# far=0.05)` would report a 0.05 that no cutoff was ever drawn at.
#
# `null=None` runs the read on entroptics' own derived default provider at its own level — the
# library's number, in the library, where the derivation that produced it lives. A caller with an
# opinion states it inside a null: `correlated_null(far=...)` (sampled, distribution-free) or
# `derived_null(far=...)` (closed-form, analytically sharp). One decision, one place, and the level
# travels with the thing it belongs to.


class UnorderedInput(TypeError):
    """Raised when the caller passes a collection with no defined row order."""


class DegenerateFrame(ValueError):
    """Raised when the frame cannot carry a read at all (empty / non-finite / rank-0)."""


_UNORDERED = (set, frozenset)


def _own_precision(a) -> np.ndarray:
    """Materialise at the array's own precision — complex128 if it carries phase, else float64.

    One rule in one place, applied on both the inbound and the outbound edge of every read. A
    `dtype=float` cast of a complex array does not raise: it drops the imaginary part behind a
    `ComplexWarning`, so a cast at either edge keeps phase from reaching an instrument that is
    complex-native throughout (Hermitian correlation, `Aperture.phase`, `Dynamics` promoting
    real->complex on contact). A shared helper is what keeps the two edges saying the same thing.

    A real array takes the float64 path."""
    A = np.asarray(a)
    return np.asarray(A, dtype=np.complex128 if np.iscomplexobj(A) else np.float64)


@dataclass(frozen=True)
class OpticsRead:
    """One screened read. Every field is scale-invariant unless named otherwise.

    k_signal
        Resolved modes above the noise floor, from the instrument's correlation eigenspectrum
        (`Aperture.spectral`). Invariant to per-channel scale. This is the number to band on.
    k_margin_last, k_margin_next
        The continuous evidence behind that integer. `k_signal` is `#(eigenvalue > edge)` — a
        hard threshold — so a mode sitting 0.0003 above the floor and one sitting 2.28 above
        it are reported with identical authority, and a `k_signal` of 3 that would have been
        2 under a hair's-width different null looks like a decisive 3.
        `k_margin_last` is `eigenvalue[k-1] - edge`: how far above the floor the marginal
        resolved mode sat — the coin-flip number. `k_margin_next` is
        `eigenvalue[k] - edge` (negative by construction): how far below the floor the first
        unresolved mode sat. Small magnitudes on either side mean the count is a threshold
        artefact rather than a measurement. Both are `None` when the mode in question does not
        exist (`k_signal == 0`, or every mode resolved).
    k_lo, k_hi, k_band
        The Weyl-certified interval for `k_signal` from
        `entroptics.reads.resolved_dimension_interval`: with `|C_read - C_true|_2 <= band`,
        `k_lo` counts modes certainly above the floor and `k_hi` modes possibly above it, so
        the true count lies in `[k_lo, k_hi]`. `k_band` is the a-priori spectral-norm band
        from `concentration_band(n_rows, n_features)`. `k_certain` (`k_lo == k_hi ==
        k_signal`) is the reading that says the count is settled at this sample size.

        Measured (planted K=3, F=16, permutation null)::

            T=64     k=3  [2, 3]   band=1.500   certain=False
            T=256    k=3  [3, 3]   band=0.625   certain=True
            T=2048   k=3  [3, 3]   band=0.192   certain=True
            T=20000  k=3  [3, 3]   band=0.058   certain=True

        The band is a-priori — `c * (sqrt(F/T) + F/T)` — so on a wide frame it is vacuous
        by construction, and the fleet's frames are wide. Measured: a (64, 256) planted-K=3
        frame gives band=12.0 and the interval `[0, 256]`, certifying nothing at all. That is
        an honest statement about 64 samples of a 256-dim vector, and it is why
        `k_margin_last` is carried alongside: on that frame the margin is 4.65 against a
        first-unresolved of -2.90, decisive evidence the interval cannot express. Read the
        margins on wide frames and the interval on tall ones, and use `k_certain` as a gate
        only where T exceeds F — for retrieval reads F >= T is the norm.

        `Sigma(1 - p_k)` — the natural "how many modes, weighted by confidence" summary — is
        not computable on this side of the library, so it is not supplied. There is no
        per-mode p-value on the spectral read: `SpectralOptics` exposes `eigenvalues`,
        `noise_floor`, `contrast`, `top_share` and friends, and nothing probabilistic per
        mode. entroptics' per-mode p-values live in `projection.mode_significance`, reached
        via `Aperture.significance` -> `self.projection().significance` — i.e. behind the
        entropy-folded Projection, the door this module routes around. Sourcing a p-value
        there takes it off a basis measured to fold 256 feature channels to f_eff=1 and
        report `K_signal=0` against 3 planted modes; a confidence-weighted count computed on
        a destroyed basis is the same wrong number wearing a decimal point. The margins and
        the certified interval above are what the spectral side can supply.
    noise_floor
        The finite-size Johnstone / Tracy-Widom correlation edge. Dimensionless, O(1).
    contrast
        lambda_1 / edge. > 1 means structure.
    top_share
        Dominant-mode power fraction.
    coherence
        Ordered-axis lag-1 coherence z-score, or `None` when it could not be measured (the screen
        produced a non-finite z). 0.0 is a real reading — no lag-1 correlation — so an unmeasurable
        screen reports `None` instead. Callers that do arithmetic gate on `is None`; `mass.compute`
        accepts `coherence=None` for this. Sign is meaningful and depends on row order.
    has_signal
        `k_signal > 0`: at least one mode resolves above the floor.
    screened
        False when the frame had too few rows to screen. `has_signal` is then False, and the read
        is not evidence of structure: a one-row or two-row answer is unscreened rather than
        screened-and-passed.
    k_screen, screen_floor
        The folded `Screen` values, for continuity with LATTICE §7.2's merge threshold, which is
        defined as `Screen(W).K_signal == 1`. `screen_floor` is dimensionful and comparable only
        within one frame. See `scale_hazard`.
    f_eff
        The width the entropy guard folded the feature axis to. `f_eff < n_features` means the
        `Screen` read coarsened the basis; `f_eff == 1` means it destroyed it.
    scale_hazard
        True when the screen half of the read cannot be trusted. Two triggers, both measurements
        rather than levels — neither compares anything to a chosen number:
        (a) the coherence z is non-finite — the screen could not produce a statistic at all; or
        (b) `f_eff <= 1` — the entropy guard folded the feature axis to a single column, so the
        screen read a destroyed basis. When True, `k_screen` / `screen_floor` may be flatly wrong
        and `k_signal` is the trustworthy count.

        A collapsed feature axis is outside this flag's two triggers. The obvious third one —
        "some channel has zero dispersion" — fires on every sparse frame: measured, an
        ontology-like frame at 2%, 5% and even 20% density has zero-MAD in all 64 of its
        channels, because a sparse coordinate is what this module carries. A permanently true
        flag carries as little information as one that never fires.

        `mantle.search.beacon` does flag it, via `live_channels` / `degraded`, and both engines
        are right for their own domain: beacon's domain is the corpus, a set of vectors where a
        dead channel is anomalous, while this instrument's domain is the signal, where it is the
        normal case. Each wrapper knows its own domain. The divergence is pinned in
        `screen_read_vectors.json` so neither side is aligned to the other by accident, and
        `top_share` / `k_signal` report the collapse directly (1.0 and 0 on such a frame) — what
        is absent is a flag, not the reading.
    """

    n_rows: int
    n_features: int
    k_signal: int
    noise_floor: float
    contrast: float
    top_share: float
    # `None` means not measurable — the screen produced a non-finite lag-1 z.
    # 0.0 is a real reading (no lag-1 correlation), so the two carry different values.
    coherence: Optional[float]
    has_signal: bool
    screened: bool
    k_screen: int
    screen_floor: float
    f_eff: int
    # `None` = not measured (the caller passed `with_screen=False`, so the screen half never ran),
    # distinct from `False` = measured and no hazard found. A caller writing `if not scale_hazard`
    # treats "unmeasured" as "safe"; write `if scale_hazard is False` when that distinction matters.
    scale_hazard: Optional[bool]
    # The continuous companions of `k_signal`. Defaulted so the two construction sites in
    # this module are the only places that need to know about them.
    k_margin_last: Optional[float] = None
    k_margin_next: Optional[float] = None
    k_lo: int = 0
    k_hi: int = 0
    k_band: float = 0.0

    # ── The propagation constant ─────────────────────────────────────────────────────────────────
    # `SpectralOptics` computes ten reads off the feature-correlation eigenspectrum; the five below
    # are the ones beyond contrast / top_share / noise_floor / resolved_modes. `attenuation` is the
    # attenuation constant alpha = log(lambda1 / max(lambda2, floor)): the decay rate of the
    # dominant mode, the real part of the propagation constant gamma (PAPER.md §6), and the number
    # a screened propagator is built on.
    #
    # alpha and `contrast` answer different questions and are not interchangeable. alpha's
    # reference is `max(lambda2, floor)`, so with two resolved modes it measures the top mode's
    # separation from the second mode; `contrast` is lambda1/floor, its separation from the noise
    # edge. Mixing them is the axis error this module exists to prevent.
    attenuation: Optional[float] = None        # alpha = log(l1/max(l2,floor)) — the decay rate
    phase: Optional[float] = None              # beta — per-step phase advance of the dominant mode
    dispersion: Optional[float] = None         # std of per-mode attenuation across resolved modes
    resolved_power: Optional[float] = None     # sum over resolved modes of (lambda_k - edge)
    dominance: Optional[float] = None          # (lambda1 - 1)/(F - 1) in [0, 1]

    # The Weyl-certified interval for alpha (PAPER.md Lemma 6.2), propagated through the same
    # `k_band` this read already computes. `attenuation_certified` is the honest whether: it is
    # True only when a positive attenuation survives the band — a computed null, not a threshold.
    attenuation_lo: Optional[float] = None
    attenuation_hi: Optional[float] = None
    attenuation_certified: Optional[bool] = None

    # ── The whitening amplitude ──────────────────────────────────────────────────────────────────
    # `max|whitened screen| / max|incident|` — how much the per-channel MAD whitening rescaled the
    # frame. Reported as evidence a reader can look at, alongside `f_eff` and `screen_floor`; it is
    # dimensionless but frame-relative, so it carries no band of its own. `None` when the screen
    # half did not run (`with_screen=False`, or an unscreened frame), which is distinct from 0.0.
    whitening_amplification: Optional[float] = None

    @property
    def k_certain(self) -> bool:
        """True when the certified interval collapses onto the point read — the count is
        settled at this sample size and this band. Anything else means `k_signal` is a point
        estimate inside `[k_lo, k_hi]` and should be reported as such."""
        return self.k_lo == self.k_hi == self.k_signal

    def as_read(self) -> dict:
        """The flat dict that goes into an `Answer.read` / a log line."""
        return {
            "k_signal": self.k_signal,
            # The margins and the certified interval travel with the integer, so a log line
            # shows whether a count was decided or was a threshold coin flip. `None` (not
            # NaN) for an undefined margin: NaN is not JSON round-trippable and would break
            # `as_read()` equality, and "there is no such mode" is a different statement
            # from "the margin was zero".
            #
            # Every value goes in at full precision. `as_read()` is what lands in `Answer.read` and
            # the log line — a record a future reader re-judges from — so rounding here would put a
            # truncated measurement into storage. A caller that wants fewer digits formats them at
            # the point of display, where the loss is visible.
            "k_margin_last": self.k_margin_last,
            "k_margin_next": self.k_margin_next,
            "k_lo": self.k_lo,
            "k_hi": self.k_hi,
            "k_band": self.k_band,
            "k_certain": self.k_certain,
            "noise_floor": self.noise_floor,
            "contrast": self.contrast,
            "coherence": self.coherence,
            "has_signal": self.has_signal,
            "screened": self.screened,
            "n_rows": self.n_rows,
            "scale_hazard": self.scale_hazard,
            # f_eff and k_screen are logged so a coarsened or destroyed basis is visible in the
            # read itself and not only in the boolean. `f_eff < n_features` means the guard
            # coarsened the basis; `f_eff == 1` means it destroyed it and `k_screen` carries no
            # information. These two are what separate a collapse from a clean read in a log line.
            "f_eff": self.f_eff,
            "n_features": self.n_features,
            "k_screen": self.k_screen,
        }


def _as_ordered_matrix(rows) -> np.ndarray:
    """Enforce the ordered contract, then materialise (T, F) at float64 — or complex128.

    Anything without a defined row order raises `UnorderedInput`. A bare iterator raises too: it
    has an order, but it is single-shot, so neither the caller nor this module can show the same
    order twice, and an ordering that cannot be reproduced cannot be reasoned about.

    Precision follows the frame (`_own_precision`), so a complex frame keeps its phase all the way
    to the instrument. entroptics is complex-native underneath — the correlation is Hermitian
    (`Xc.conj().T @ Xc`), `Aperture` reads a complex frame and publishes a `phase` for it, and
    `Dynamics` promotes real->complex on contact.
    """
    if isinstance(rows, _UNORDERED):
        raise UnorderedInput(
            f"{type(rows).__name__} has no row order. The entroptics screen is ORDERED: axis 0 "
            "is the ordered axis, and an arbitrary order collapses coherence toward the null "
            "(measured 11.32 -> 0.42 under shuffle). Sort or sequence the "
            "rows deliberately and pass a list/tuple/ndarray."
        )
    if isinstance(rows, np.ndarray):
        W = rows
    elif isinstance(rows, Sequence):
        # No dtype is pinned while stacking. `_own_precision` below is the one place the frame's
        # precision is decided, so a Sequence of complex rows arrives with its phase intact.
        W = np.asarray([np.asarray(r).ravel() for r in rows])
    elif isinstance(rows, Iterator):
        raise UnorderedInput(
            "a bare iterator is single-shot; its order cannot be reproduced or audited. "
            "Materialise it into a list in the order you mean."
        )
    else:
        raise UnorderedInput(f"cannot read row order from {type(rows).__name__}")

    # The frame's own dtype decides — `_own_precision` is that one rule. A real frame takes the
    # float64 path; a complex one keeps its phase all the way to the instrument.
    W = np.ascontiguousarray(_own_precision(W))
    if W.ndim == 1:
        W = W.reshape(1, -1)
    if W.ndim != 2:
        raise DegenerateFrame(f"expected a 2-D (ordered, feature) frame; got shape {W.shape}")
    if W.size and not np.isfinite(W).all():
        raise DegenerateFrame("frame contains non-finite values; clean or mask before reading")
    return W


def _unscreened(W: np.ndarray) -> OpticsRead:
    # The read for a frame that was never screened. The margin/interval fields take their declared
    # defaults — `None`, `None`, 0, 0, 0.0 — because there is no marginal mode to measure a margin
    # for and no eigenspectrum to certify an interval over. `screened=False` is what a caller gates
    # on; the margins carry the same absence rather than restating it.
    #
    # `coherence` and `scale_hazard` are `None` here, meaning not measured, as against `0.0` and
    # `False`, which are readings. This is the path every frame below `MIN_ROWS` rows, below two
    # features, or all-zero takes — precisely the frames the instrument cannot read, where a
    # lag-1-coherent walk and i.i.d. noise both come back as 0.0 with zero variance (see the
    # degeneracy block above). A 0.0 published from here would be indistinguishable from a real
    # measured absence of lag-1 structure. [[absence-is-not-an-affirmative-claim]] — derive it or
    # carry the absence.
    return OpticsRead(n_rows=int(W.shape[0]), n_features=int(W.shape[1]) if W.ndim == 2 else 0,
                      k_signal=0, noise_floor=float("inf"), contrast=0.0, top_share=0.0,
                      coherence=None, has_signal=False, screened=False, k_screen=0,
                      screen_floor=float("inf"), f_eff=0, scale_hazard=None)


def read_ordered(rows, *, null=None, seed: int = 0,
                 window: Optional[int] = None, with_screen: bool = True) -> OpticsRead:
    """Screen an ordered frame through the instrument — the sanctioned entroptics read.

    `rows` is (T, F): axis 0 ordered, axis 1 feature. Pass an ndarray or a Sequence of
    equal-length vectors, in the order you mean. A `set` raises `UnorderedInput`.

    `null` is the noise provider, the caller's one say in this read. `None` runs on entroptics'
    derived `mp` default, the i.i.d.-Gaussian edge: correct for an i.i.d. bulk and optimistic for
    correlated rows. For retrieved evidence (correlated by construction) pass `correlated_null()`,
    the distribution-free permutation null the library directs correlated callers to. Deterministic
    per `seed`.

    There is no `far` keyword. The false-alarm level is a property of the null — see the `far`
    block near the head of this module. State a level by handing over a null that holds it:
    `correlated_null(far=…)` or `derived_null(far=…)`.

    `window` defaults to None, which reads the whole frame. entroptics' `Aperture` defaults to an
    adaptive 128-frame window, truncating a longer batch read; a one-shot screen over a fixed
    evidence set wants the whole of it.

    `with_screen=False` skips the folded `Screen` (the expensive half) when the caller only
    needs the invariant count.
    """
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return _unscreened(W)

    from entroptics import Aperture

    ap = Aperture(W, window=window, null=null, seed=seed)
    spec = ap.spectral                       # correlation eigenspectrum: scale-invariant
    k_signal = int(spec.resolved_modes)

    # ── The continuous evidence behind the integer ────────────────────────────────────────
    # `resolved_modes` is `#(eigenvalue > edge)`. The comparison is paid for in full, so the
    # distance of each mode from the floor — what says whether the count was decided or
    # coin-flipped — is carried out alongside the count. `eigenvalues` is descending, so index
    # k-1 is the marginal resolved mode and index k is the first unresolved one.
    ev = np.asarray(spec.eigenvalues, dtype=float).ravel()
    edge = float(spec.noise_floor)
    k_margin_last: Optional[float] = None
    k_margin_next: Optional[float] = None
    if np.isfinite(edge) and ev.size:
        if 0 < k_signal <= ev.size:
            k_margin_last = float(ev[k_signal - 1] - edge)
        if 0 <= k_signal < ev.size:
            k_margin_next = float(ev[k_signal] - edge)

    # Weyl-certified interval for the count. `concentration_band` is the a-priori
    # spectral-norm bound on ||C_hat - C_true||_2 for T samples of an F-dim vector;
    # `resolved_dimension_interval` propagates it through the eigenvalues. `sg=spec` is
    # passed so it reuses the read already taken — without it the function recomputes
    # `spectral_optics(data)` from scratch, and does so against the library's default `mp`
    # floor, discarding whatever `null` the caller passed. With `sg`, `data` is untouched
    # and the caller's null is what the interval is certified against.
    from entroptics.reads import concentration_band, resolved_dimension_interval

    k_band = float(concentration_band(int(W.shape[0]), int(W.shape[1])))
    certified = resolved_dimension_interval(W, band=k_band, sg=spec)
    k_lo, k_hi = int(certified.resolved_lo), int(certified.resolved_hi)

    # The skip path carries what it did not measure as `None`. A `with_screen=False` read — which
    # is every read taken through `ember/signal/projection.py`, `resolvable()` and therefore
    # `prism.resolution.signal_end` — never runs the screen half, so `coherence` and `scale_hazard`
    # have no reading to give. `None` means "not measured" on this path and inside the branch
    # below, so the two agree.
    k_screen, screen_floor, f_eff = 0, float("inf"), 0
    coherence: Optional[float] = None
    hazard: Optional[bool] = None
    if with_screen:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sc = ap.projection()                 # the folded Screen surface, reached through Aperture.projection()
            k_screen = int(sc.K_signal)
            screen_floor = float(sc.noise_floor)
            f_eff = int(sc.screen.shape[1])
            coh = float(sc.coherence)
            # 0.0 is a legitimate coherence — it means no lag-1 correlation — so a screen that
            # could not produce a finite z has no reading to give and carries `None` instead.
            # `mass.compute` takes `coherence: Optional[float] = None` for exactly this case.
            coherence = coh if np.isfinite(coh) else None
            # Whitening amplification: the signature of heterogeneous channel occupancy, reported
            # as an amplitude on the read. No divide guard is needed here — `_carries_read` already
            # required `np.any(W)`, so `max|W| > 0`.
            amplification = float(np.abs(sc.screen).max()) / float(np.abs(W).max())
            # The hazard has two triggers, both measurements rather than levels: a non-finite z
            # (the screen could not produce a statistic) and a feature axis folded to one column
            # (the basis was destroyed). Neither compares anything to a chosen number.
            #
            # The amplitude is not a third trigger. Measured across 60 adversarial frames
            # (dead/zero-MAD channels, one dominant channel at 1e9, one-hot occupancy, 1e-9 scale
            # ratios, sparse carriers) the maximum amplification observed was 4.71, and every frame
            # a large amplification would have been meant to catch was already caught by
            # `f_eff <= 1`. `amp` is `max|screen| / max|W|`, a ratio of maxima, which stays O(1)
            # because whitening rescales columns rather than inflating the peak — the
            # 13-orders-of-magnitude movement this module's header measures lives in
            # `Screen.noise_floor`, a singular value, which is a different quantity.
            #
            # The obvious alternative — flag when any channel has zero dispersion — fires on almost
            # every real frame here, because ontology coordinates are sparse and all-zero channels
            # are the normal case; §1 of this module's header is about carrying that sparsity.
            hazard = (not np.isfinite(coh)) or f_eff <= 1

    return OpticsRead(
        n_rows=int(W.shape[0]), n_features=int(W.shape[1]),
        k_signal=k_signal, noise_floor=float(spec.noise_floor),
        contrast=float(spec.contrast), top_share=float(spec.top_share),
        coherence=coherence, has_signal=k_signal > 0, screened=True,
        k_screen=k_screen, screen_floor=screen_floor, f_eff=f_eff, scale_hazard=hazard,
        whitening_amplification=amplification if with_screen else None,
        k_margin_last=k_margin_last, k_margin_next=k_margin_next,
        k_lo=k_lo, k_hi=k_hi, k_band=k_band,
        # The five further spectral reads (see the field block above). `spec` is already computed,
        # so forwarding costs nothing and saves the caller re-deriving them.
        attenuation=float(spec.attenuation), phase=float(spec.phase),
        dispersion=float(spec.dispersion), resolved_power=float(spec.resolved_power),
        dominance=float(spec.dominance),
        **_certified_attenuation(W, spec, k_band),
    )


def _certified_attenuation(W: np.ndarray, spec, band: float) -> dict:
    """The Weyl-certified interval for the attenuation constant (PAPER.md Lemma 6.2), reusing the
    spectral read and the band `read_ordered` already computed — so this adds no second eigenread.

    Returns `None`s rather than zeros when it cannot certify. `attenuation_certified` is the
    whether (`alpha_lo > 0`: a positive attenuation survives the band), so `False` is the reading
    for a frame measured to have no certified attenuation, and `None` is an unread frame."""
    try:
        from entroptics.reads import attenuation_interval
        ci = attenuation_interval(W, band=float(band), sg=spec)
        return {"attenuation_lo": float(ci.attenuation_lo),
                "attenuation_hi": float(ci.attenuation_hi),
                "attenuation_certified": bool(ci.certified)}
    except Exception:
        return {"attenuation_lo": None, "attenuation_hi": None, "attenuation_certified": None}


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar reads off an ordered frame
# ═══════════════════════════════════════════════════════════════════════════════
# Each read below takes an ordered (T, F) frame and returns a single value the instrument measures on
# it. The frame is the measurement: the instrument reports the structure of whatever it is handed, so
# a caller presents real ordered evidence — rows in the order they mean — rather than a synthetic
# frame. Every read returns None rather than a number when the frame cannot carry one.

def resolvable(rows, *, null=None, seed: int = 0, require_certain: bool = False) -> Optional[int]:
    """Modes above the noise floor of the feature-correlation spectrum (`k_signal`).

    Which matrix is the whole content of this docstring. This returns `#{k : λ_k > λ₊}` on the
    unit-diagonal correlation matrix. The paper's `K_signal` is a different statistic,
    `#{k : s_k > Φ}` on the entropy-folded, MAD-whitened screen, and this function does not return
    it: it calls `read_ordered` with `with_screen=False`, so the screen half never runs. For the
    screen statistic, read `read_ordered(rows).k_screen` with `with_screen` left on. For this one,
    `principal_directions` is the directional companion and reads the same spectrum.

    The two are distinguishable, measured on a synthetic sparse frame (D=2048, density ~0.007),
    8 seeds:

        rows   structured        sparse-noise control
         8     k_signal 0        k_signal 0        <- this function: reads them alike
         8     k_screen 2        k_screen 0        <- the screen: discriminates
        63     k_signal 0        k_signal 0
        63     k_screen 16       k_screen 0        (0 in 23 of 24 noise runs)

    That separation is a property of how the synthetic frame was built: its "structured" rows
    shared literal feature indices — exact column overlap, which real signed-hash coordinates of a
    shared path do not produce. It is a true statement about frames of that construction, and not
    evidence about the corpus. On the real ontology coordinate
    (`crystal.ontology.geometry.dense_vec` over actual hypernym descents) neither statistic
    separates a descent from a per-row shuffle:

        seed              k_signal   k_screen   k_screen shuffled
        dog.n.01              0          2            3
        physicist.n.01        0          2            2
        bank.n.01             0          2            3
        water.n.06            0          2            1
        cat.n.01              0          2            6

    The shuffle reads higher on three of five. On a sparse frame the correlation read has nothing
    to find, which is what a `k_signal` of 0 there states.

    `rows` is an ordered (T, F) frame with F >= 2. With `require_certain=True`, returns the count
    only when the Weyl interval has collapsed to it (`k_lo == k_hi == k_signal`), else None.
    Returns None when the frame cannot carry a read (too few rows, one feature, all-zero) — distinct
    from a resolved count of 1."""
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    rd = read_ordered(W, null=null, seed=seed, with_screen=False)
    if require_certain and not (rd.k_lo == rd.k_hi == rd.k_signal):
        return None
    return int(rd.k_signal)


def position_coherence(rows, position: int, *, lag: int = 1) -> Optional[float]:
    """The per-position read: how well does one row sit where it sits, against its frame's own null?

    entroptics' own decomposition, read at a position instead of over a frame.
    `projection.coherence` is `A = mean_i R[i, i+lag]` standardised against `mu`, the off-diagonal
    mean of `R = Re(G)**2`. This takes the same `R` and the same `mu`, and reads the terms belonging
    to `position` — `R[p-lag, p]` and `R[p, p+lag]`. It is not a new statistic: the frame-level mean
    is the sum of these.

    The frame-level read cannot answer a one-row question. Replacing one row changes only 2 of a
    window's ~w terms, so `coherence` dilutes the question by w/2 whatever window it is given.
    Measured on the canon at windows 6..48: separation 0.1-0.36 sd, win-rate at chance. Read at the
    position instead, on the same frames: 0.53-0.72 sd, 72-82%.

    The answer does not depend on a window. Across w = 6, 8, 12, 16, 24, 32, 48 the separation is
    flat (0.53, 0.71, 0.58, 0.68, 0.66, 0.68, 0.72), so there is no size to choose — a bigger frame
    sharpens the null rather than changing the verdict.

    Rows must be comparable in scale. `R = Re(G)**2` on unnormalised rows is dominated by
    ``||row||``, so on a corpus whose rows span 90-100x in norm the read measures length. Rows are
    unit-normalised below, so the read takes its input already comparable.

    Sign is the verdict and the zero is computed: above 0 = this row is more like its ordered
    neighbours than a random row of this frame would be; below 0 = less. The zero is the frame's own
    off-diagonal null. Measured: true rows +0.13..+0.52, foreign -0.41..-0.57.

    Returns None when the frame carries no off-diagonal spread to standardise against, or the
    position has no ordered neighbour. The measurement has nowhere to stand, so nothing propagates
    and no verdict is issued; `None` is that absence rather than a default.
    """
    W = _as_ordered_matrix(rows)
    if W.ndim != 2 or W.shape[0] < 2:
        return None
    norms = np.linalg.norm(W, axis=1)
    W = W / np.maximum(norms[:, None], 1e-12)
    R = np.real(W @ W.T) ** 2
    n = R.shape[0]
    p = int(position)
    if not (0 <= p < n) or lag < 1:
        return None
    terms = []
    if p - lag >= 0:
        terms.append(float(R[p - lag, p]))
    if p + lag < n:
        terms.append(float(R[p, p + lag]))
    if not terms:
        return None
    if n < 4:
        return None            # the closed-form moment needs four distinct indices

    # The exact permutation moment, assembled as `entroptics.projection.coherence` assembles it
    # (Cliff-Ord / Mantel, Theorem 5.2) and evaluated for this statistic's term count.
    #
    # The sharing term carries the weight here: `R[p-lag,p]` and `R[p,p+lag]` share row p, which is
    # the correlated case the closed form exists for. Standardising by `off.std()` instead would be
    # the naive variance the library's own docstring names and sets aside — *"consecutive terms at
    # lag=1 share a row, so a naive var/(N-lag) mis-standardises"*. M = len(terms), n_share = 2 when
    # both terms are present (the ordered pairs of terms sharing an index), n_disj = 0.
    d = np.diagonal(R)
    S1 = float(R.sum() - d.sum())                         # sum over off-diagonal pairs
    S2 = float((R * R).sum() - (d * d).sum())             # sum of squares, off-diagonal
    rowsum = R.sum(axis=1) - d
    U = float((rowsum * rowsum).sum())
    Dp = n * (n - 1)
    mu, mu2 = S1 / Dp, S2 / Dp
    mu_sq = mu * mu
    E_share = (U - S2) / (n * (n - 1) * (n - 2))
    M = len(terms)
    n_share = 2 if M == 2 else 0
    var = (M * (mu2 - mu_sq) + n_share * (E_share - mu_sq)) / (M * M)
    # The degeneracy bound is derived from the float type and the formula's own structure. On a
    # frame whose rows are identical the variance is zero in exact arithmetic; in floats each moment
    # difference leaves ~eps of cancellation, and the assembly multiplies those by
    # (M + n_share)/M**2, so a tolerance of one eps is too tight exactly where n_share is large and
    # the caller would get a confident 0.0 from a frame that can standardise nothing.
    scale = max(abs(mu_sq), abs(mu2), 1.0)
    # S1, S2 and U each accumulate O(n**2) products, so their relative rounding grows with the
    # summation length (the standard floating-point bound) before the assembly amplifies it.
    tol = (np.finfo(float).eps * scale * (n * n)
           * max((M + n_share) / float(M * M), 1.0))
    if not math.isfinite(var) or var <= tol:
        return None
    return (float(np.mean(terms)) - mu) / math.sqrt(var)


def belonging(rows, position: int) -> Optional[float]:
    """The unordered sibling of `position_coherence`: does this row belong in this neighbourhood?

    Same decomposition, same null. Where `position_coherence` reads the two ordered terms
    `R[p-lag,p]` and `R[p,p+lag]`, this reads the candidate's entire off-diagonal row of
    `R = Re(G)**2` — its agreement with every member of the frame — against the frame's own `mu`.

    It is the read for a collection with no ordered axis. `mantle.shard.cache` has no sequence and
    no adjacency: items live in regions keyed by geometry, so a lag-1 read there would need an order
    imposed on it. The canon has document order and the cache does not, so the cache gets the read
    that needs none.

    All terms share index p, so every pair of them is a sharing pair — `n_share = M(M-1)` in the
    Cliff-Ord/Mantel second moment. The disjoint form would understate the variance and inflate
    every z.

    Sign is the verdict against a computed zero, exactly as in `position_coherence`: above the
    frame's own null = more like this neighbourhood than a random row of it; below = less. Returns
    None when the frame cannot standardise, rather than a default.
    """
    W = _as_ordered_matrix(rows)
    if W.ndim != 2 or W.shape[0] < 4:
        return None
    norms = np.linalg.norm(W, axis=1)
    W = W / np.maximum(norms[:, None], 1e-12)
    R = np.real(W @ W.T) ** 2
    n = R.shape[0]
    p = int(position)
    if not (0 <= p < n):
        return None
    d = np.diagonal(R)
    S1 = float(R.sum() - d.sum())
    S2 = float((R * R).sum() - (d * d).sum())
    rowsum = R.sum(axis=1) - d
    U = float((rowsum * rowsum).sum())
    Dp = n * (n - 1)
    mu, mu2 = S1 / Dp, S2 / Dp
    mu_sq = mu * mu
    E_share = (U - S2) / (n * (n - 1) * (n - 2))
    M = n - 1                                   # the candidate's off-diagonal row
    n_share = M * (M - 1)                       # every pair shares index p
    var = (M * (mu2 - mu_sq) + n_share * (E_share - mu_sq)) / (M * M)
    # The tolerance follows the formula's own structure. On an identical-row frame the variance is
    # zero in exact arithmetic; in floats each moment difference leaves ~eps of cancellation, and
    # the assembly multiplies those by (M + n_share)/M**2, so a tolerance of one eps is too tight
    # exactly where n_share is large and the caller would get a confident 0.0 from a frame that can
    # standardise nothing.
    scale = max(abs(mu_sq), abs(mu2), 1.0)
    # S1, S2 and U each accumulate O(n**2) products, so their relative rounding grows with the
    # summation length (the standard floating-point bound) before the assembly amplifies it.
    tol = (np.finfo(float).eps * scale * (n * n)
           * max((M + n_share) / float(M * M), 1.0))
    if not math.isfinite(var) or var <= tol:
        return None
    return (float(rowsum[p] / M) - mu) / math.sqrt(var)


def revision_resolver():
    """Build the read-time head resolver `mantle.shard.cache` takes as its `resolve` seam.

    Called as `resolve(root, revisions, reads, frame)`. `frame` is the reader's own recall set —
    `{item_id: vector}` for the pool this query surfaced, rather than the region or the whole held
    set. That is what makes head observer-relative: the same root may answer with different
    revisions to different questions, because the neighbourhood it is read against differs. A fixed
    frame would be a stored verdict under a new name.

    The read is `belonging`, which the consumer decides. `mantle.shard.cache` has no ordered axis —
    no sequence, no adjacency; items live in regions keyed by geometry — so a lag-1 ordered read
    would need an order imposed on it. `belonging` reads the candidate's whole off-diagonal row
    against the frame's own null and needs none. Measured on the canon it is also the stronger read:
    0.78-0.85 sd and 78-82%, against 0.52-0.70 sd for the ordered one, because it weighs n-1 terms
    rather than 2.

    Where there is no measurement to carry, every revision stands — the same result mantle has with
    no resolver at all. Each case is a computed null rather than a constant: no frame, a candidate
    with no vector, no readable measurement, or no candidate reading above the frame's own null.

    The result is a reading, published as one: 72-82% agreement with the true row on the canon.
    """
    def resolve(root, revisions, reads, frame):
        ids = [r["id"] if isinstance(r, dict) else r.id for r in revisions]
        if not frame:
            return ids
        # the neighbourhood is the recall set minus this root's own revisions — a candidate must be
        # read against its surroundings, not against its competitors.
        others = [v for k, v in frame.items() if k not in set(ids)]
        if len(others) < 3:
            return ids                      # too little frame to standardise against: nothing hides
        scored = []
        for rid in ids:
            vec = frame.get(rid)
            if vec is None:
                return ids                  # an unmeasurable candidate leaves the others standing
            rows = list(others) + [vec]
            z = belonging(rows, len(rows) - 1)
            if z is not None:
                scored.append((z, rid))
        above = [(z, i) for z, i in scored if z > 0.0]
        if not above:
            return ids                      # none reads above the null: all stand
        best = max(z for z, _ in above)
        return [i for z, i in above if z == best]

    return resolve


def principal_directions(rows, *, null=None, seed: int = 0):
    """The resolved subspace basis — an `(F, k)` array whose columns are the feature-space directions
    of the `k` resolved correlation modes (`k = resolvable(rows)`), descending by eigenvalue.

    The directional companion to `resolvable`: that returns how many modes stand above the instrument's
    noise floor, this returns which directions they span, read off the same scale-invariant
    correlation spectrum — the entropy-folded screen is a different door, and this one leaves a
    sparse carrier intact. A caller wanting the resolved subspace, to project onto it or hold it as
    a fixed basis, reads it here rather than taking its own SVD. `rows` is an ordered (T, F) frame;
    None when the frame cannot carry a read; an `(F, 0)` array when nothing resolves."""
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics.reads import principal_directions as _pd
    try:
        return _own_precision(_pd(W, null=null, seed=seed))
    except Exception:
        return None


def absorb_transmit(rows, *, basis=None, null=None, seed: int = 0):
    """The membrane operation — absorb-and-propagate-the-residual (the signal-native reach's 0a mechanism).

    A signal (an ordered (T, F) frame) arriving at a tekton is split at its membrane into the band that
    couples to this tekton and the band that does not. Coupling is measured rather than declared: the
    coupled band is the frame's projection onto the resolved subspace (`principal_directions` — the `k`
    correlation modes standing above the instrument's noise floor), and the residual is what is left. A tekton
    absorbs its coupled band (condense -> a typed artifact = the work) and transmits the residual, which
    keeps propagating to the next coupling capability, so the signal stays intact for its entire length.

    Returns `(absorbed, transmitted, k)`:
      · `absorbed`    — the coupled band as a (T, F) frame (the projection onto the resolved subspace);
      · `transmitted` — the residual as a (T, F) frame (`incident − absorbed`), what propagates onward;
      · `k`           — the number of resolved (absorbed) modes.
    Conservation is exact by construction: the projector onto the resolved subspace is orthogonal, so
    `‖incident‖² = ‖absorbed‖² + ‖transmitted‖²` — energy is neither created nor lost at the membrane (the
    `Screen.balance` 0→0 closure, per hop). Every norm here is summed over magnitudes, `Σ|z|²`, so the
    identity holds for a complex frame as it does for a real one. When nothing resolves (`k == 0`) nothing couples here:
    `absorbed` is all-zero and `transmitted` is the incident frame, so the whole signal propagates on
    intact. Returns None when the frame cannot carry a read (too few rows, one feature, all-zero); the
    caller then propagates the frame unabsorbed rather than fabricating a coupling.

    The absorbed band is unsigned. Provenance is signed at the tekton boundary, on the typed artifact the
    tekton condenses the band into — the waveform-provenance boundary keeps signatures off mid-stream
    frames. This returns the physical split; signing is the tekton's, after condensation. The read stays
    inside the one instrument (`principal_directions`), so entroptics is reached only through
    `ember.optics`."""
    W = _as_ordered_matrix(rows)
    if basis is None:
        # Self-resolution: the tekton is tuned to whatever the frame resolves — absorb the frame's own
        # structured band, transmit the unresolved residual. This is the default membrane read.
        B = principal_directions(W, null=null, seed=seed)
        if B is None:
            return None
    else:
        # Tekton coupling: project onto the tekton's own directions (its offer/tuning as an (F, k) basis
        # over this frame's features), so it absorbs the band that couples to it rather than the frame's
        # self-structure.
        B = _own_precision(basis)
        if B.ndim != 2 or B.shape[0] != W.shape[1]:
            return None                               # a basis is (F, k) over the incident frame's features
    if int(B.shape[1]) == 0:
        return np.zeros_like(W), W.copy(), 0          # nothing couples here → the whole signal transmits on
    # ── the projector, applied without ever forming it ───────────────────────────────────────────
    # `P = B @ pinv(B)` then `W @ P` is the definition and it is two avoidable costs, both pure
    # arithmetic and neither a modelling choice:
    #
    # 1. **`pinv(B)` is `B.T` when B has orthonormal columns**, exactly — it is the SVD's own
    #    definition, not an approximation. `B` here is `prism.frames.offer_basis`'s output, the left
    #    singular vectors `U[:, :r]`, orthonormal by construction; `principal_directions` returns
    #    directions too. It is CHECKED rather than assumed: `Bᵀ B ≈ I` is a `k x k` read, and
    #    anything that fails it falls through to `pinv` unchanged. The tolerance is the same rank
    #    tolerance `offer_basis` derives — `max(shape) · eps` — so no number is introduced here.
    # 2. **`W @ (B @ Bᵀ)` materialises an `F x F` projector; `(W @ B) @ Bᵀ` never does.** Matrix
    #    multiplication is associative, so the result is identical; the cost is not. For a frame
    #    with F features and k directions the first is `O(F²(F + T))` and the second `O(TFk)`.
    #
    # Measured 2026-08-25 on 71/home: `superposition(cn-singlish)` spent **73.3 s of 91.9 s in
    # `numpy.linalg.svd`** — 156 calls, two per `separation()`, one for the band and one for this
    # pseudo-inverse. The evidence frame of a single candidate, `cn-music`, is 4,165 rows wide.
    #
    # The conservation identity is what must not move: the projector has to stay ORTHOGONAL, or
    # `‖incident‖² = ‖absorbed‖² + ‖transmitted‖²` stops holding and every certificate built on it
    # is quietly wrong. `B Bᵀ` is orthogonal exactly when B is orthonormal, which is the condition
    # tested above — the check and the identity are the same condition.
    _k = int(B.shape[1])
    _gram = B.T @ B
    _tol = max(B.shape) * float(np.finfo(B.dtype).eps) * max(1.0, float(np.abs(_gram).max()))
    if np.allclose(_gram, np.eye(_k, dtype=B.dtype), atol=_tol, rtol=0.0):
        absorbed = (W @ B) @ B.T                       # orthonormal: pinv(B) IS B.T
    else:
        absorbed = W @ (B @ np.linalg.pinv(B))
    transmitted = W - absorbed
    return absorbed, transmitted, _k


def propagate_residual(rows, bases, *, null=None, seed: int = 0):
    """Propagate a signal through an ordered chain of tekton couplings, absorbing each tekton's band and
    carrying the residual onward — the signal-native reach's full-length propagation. The residual of one
    hop is the incident of the next, so the signal travels its whole length, shedding a band at each
    coupling capability rather than being consumed at one resolver.

    `bases` is an ordered iterable, one entry per tekton: an `(F, k)` coupling basis (the tekton's tuning),
    or `None` for a self-resolution membrane (`absorb_transmit`'s default). Returns::

        {"hops": [{"k", "absorbed_energy"} | {"k": None, "reason"}, …],   # per tekton, in order
         "residual": <(T, F) frame>,          # what reached the end UNcoupled (fully transmitted)
         "incident_energy", "residual_energy",
         "conserved": bool}                    # ‖incident‖² == Σ‖absorbed_i‖² + ‖final residual‖²

    Conservation holds across the whole chain by construction: every hop is an orthogonal split
    (`absorb_transmit`), and orthogonal splits telescope, so energy is neither created nor lost along the
    length — the `Screen.balance` 0→0 closure, composed. A hop whose frame cannot carry a read passes the
    signal through unabsorbed. Each absorbed band is the work: it condenses to a typed artifact at its
    tekton, where provenance is signed (waveform-provenance boundary). This returns the physical
    propagation; the signatures are the tektons'."""
    W = _as_ordered_matrix(rows)
    total_in = float((np.abs(W) ** 2).sum())
    hops = []
    residual = W
    absorbed_sq = 0.0
    for b in bases:
        res = absorb_transmit(residual, basis=b, null=null, seed=seed)
        if res is None:
            hops.append({"k": None, "absorbed_energy": 0.0, "reason": "frame carried no read — passed through"})
            continue
        absorbed, residual, k = res
        e = float((np.abs(absorbed) ** 2).sum())
        absorbed_sq += e
        hops.append({"k": k, "absorbed_energy": e})
    residual_sq = float((np.abs(residual) ** 2).sum())
    # The band is derived from the arithmetic this walk performed — see `_float_noise`.
    conserved = abs(total_in - (absorbed_sq + residual_sq)) <= _float_noise(W, total_in, splits=len(hops))
    return {"hops": hops, "residual": residual, "conserved": conserved,
            "incident_energy": total_in, "residual_energy": residual_sq}


def next_by_coupling(rows, bases, *, fired=(), null=None, seed: int = 0,
                     min_energy: Optional[float] = None, incident_energy=None):
    """One hop of coupling-based routing: which tekton absorbs the most of this residual, and what is left.

    This is the hop granularity the reach plane works at. `route_by_coupling` walks the whole chain in one
    call, which suits an in-process read where every basis is known locally; on the plane each hop happens
    in a different process, so the decision a provider can take is *"given the residual I am holding, who
    couples next?"* — one hop, then the residual is re-placed on the plane and the next process decides.
    `route_by_coupling` calls this in a loop, so the plane and the in-process walk share one selection rule.

    `fired` is the tektons already coupled — on the plane that is the need artifact's `path`, carried in
    provenance so a distributed hop can honour the Cascade guard (a tekton fires at most once) without
    shared memory. Coupling is idempotent, so re-visiting absorbs nothing: the guard is a derived
    termination rather than a hop cap ([[no-arbitrary-caps]]).

    `incident_energy` scopes the `min_energy` floor to the original incident signal when the caller knows
    it (the plane does, via the root need). Omitted, it falls back to this residual's own energy, the local
    reading — the floor then rises as the signal is absorbed, so a long chain terminates slightly earlier
    than the in-process walk.

    Returns `None` when nothing couples above the floor (the signal has finished its path), else::

        {"tekton": name, "transmitted": <(T, F) residual after this hop>,
         "absorbed_energy": float, "k": int}
    """
    W = _as_ordered_matrix(rows)
    total_in = float((np.abs(W) ** 2).sum()) if incident_energy is None else float(incident_energy)
    # This floor decides whether a tekton coupled at all — the routing decision itself — so
    # [[capability-is-an-artifact-matched-by-propagation]] wants it measured. `min_energy=None`
    # means the reader states no floor, and the floor is then the level below which a difference in
    # energy is floating-point noise rather than a coupling: `_float_noise`, derived from the
    # frame's own dtype and the arithmetic the projection performs. A caller that states a
    # `min_energy` scales it by the incident energy instead.
    thresh = (_float_noise(W, total_in) if min_energy is None
              else float(min_energy) * max(total_in, 1.0))
    skip = set(fired or ())
    best = None                                        # (name, transmitted, k, energy)
    for name, b in (bases or {}).items():
        if name in skip:
            continue
        res = absorb_transmit(W, basis=b, null=null, seed=seed)
        if res is None:
            continue
        ab, tr, k = res
        # Energy is Σ|z|², summed over magnitudes, so a complex band contributes its magnitude and
        # the quantity stays non-negative on any dtype. This line is the attention weight — `best`
        # below ranks tektons by it. On a real frame it is bit-identical to Σz², since `abs` clears
        # a sign bit that squaring would clear anyway.
        e = float((np.abs(ab) ** 2).sum())
        if e > thresh and (best is None or e > best[3]):
            best = (name, tr, int(k), e)
    if best is None:
        return None
    return {"tekton": best[0], "transmitted": best[1], "k": best[2], "absorbed_energy": best[3]}


def route_by_coupling(rows, bases, *, null=None, seed: int = 0,
                      min_energy: Optional[float] = None):
    """Auto-route a signal by coupling — the residual finds its next tekton by measurement rather than name.

    "Signals propagate to where they resolve" ([[capability-is-an-artifact-matched-by-propagation]]): at each
    hop the tekton whose coupling absorbs the most of the current residual takes its band, and the residual
    propagates to its next-strongest coupling — nearest / hop-the-gap, in place of a routing table. This is
    the ordering `propagate_residual` takes as given, selected here by coupling strength: a signal placed
    once finds its own path through the coupling capabilities.

    `bases` is `{tekton_name: (F, k) coupling basis}` (e.g. `ember.ontology.match.tekton_basis_for` per
    registered tekton). A tekton fires at most once (the `Cascade` termination guard, applied to
    propagation). Routing stops when no remaining tekton couples above `min_energy · ‖incident‖²` — the
    signal has been fully absorbed, or nothing left resolves. Returns::

        {"route": [{"tekton", "absorbed_energy", "k"}, …],   # the path it found, in order
         "residual": <(T, F) frame>, "incident_energy", "residual_energy",
         "conserved": bool}                                   # ‖incident‖² == Σ absorbed + ‖final residual‖²

    Conservation is exact across the routed path (every hop is an orthogonal split). Signing is each
    tekton's, at its boundary; mid-stream frames stay unsigned (waveform-provenance)."""
    W = _as_ordered_matrix(rows)
    total_in = float((np.abs(W) ** 2).sum())
    residual = W
    fired: set = set()
    route = []
    absorbed_sq = 0.0
    while True:
        # One selection rule, shared with the plane — see `next_by_coupling`. `incident_energy` is pinned to
        # the original signal so the floor holds steady as the residual shrinks, which is what makes this
        # walk's termination depend on the whole signal rather than on how much of it is left.
        hop = next_by_coupling(residual, bases, fired=fired, null=null, seed=seed,
                               min_energy=min_energy, incident_energy=total_in)
        if hop is None:                                # nothing left couples → the signal has finished its path
            break
        residual = hop["transmitted"]
        fired.add(hop["tekton"])
        absorbed_sq += hop["absorbed_energy"]
        route.append({"tekton": hop["tekton"], "absorbed_energy": hop["absorbed_energy"], "k": hop["k"]})
    residual_sq = float((np.abs(residual) ** 2).sum())
    # The band is derived from the arithmetic this walk performed — see `_float_noise`.
    conserved = abs(total_in - (absorbed_sq + residual_sq)) <= _float_noise(W, total_in, splits=len(route))
    return {"route": route, "residual": residual, "conserved": conserved,
            "incident_energy": total_in, "residual_energy": residual_sq}


def _float_noise(W: np.ndarray, energy: float, *, splits: int = 1) -> float:
    """The energy scale below which a difference is floating-point noise rather than a measurement.

    Used on the two questions this module would otherwise have to guess at: *did the signal
    conserve?* and *did anything couple here?* Both compare a real energy against a band derived
    from the arithmetic that produced it.

    The law itself is `prism.rounding.split_walk_rounding`; this function is the dtype read plus the
    call. Keeping the derivation in prism's dependency-free base is what lets `prism.conservation`
    and `mantle/search/beacon/instrument.py` answer the same question the same way — one derivation,
    reachable by any component that has numpy or does not.

    It models accumulation error rather than cancellation, which is right here because every term
    summed is a `‖·‖²` and therefore non-negative: partial sums increase monotonically, so
    catastrophic cancellation cannot arise. The full argument is at `prism.rounding`.

    What stays here is the line prism cannot hold: `ε` read off the frame's own dtype, so a float32
    frame earns a wider band than a float64 one with no level restated anywhere. That is numpy
    vocabulary, and prism's base install has no numpy.

    The band is sharp enough to see a real leak. A path that loses `1e-7` of its incident energy —
    orders of magnitude above float noise — fails this test on every frame size, where a flat `1e-6`
    tolerance would pass it. Measured slack of a flat tolerance against this bound, by frame:

        frame            flat 1e-6 (conserved)      flat 1e-9 (routing floor)
        (4, 16)                2.3e+07 ×                   3.5e+04 ×
        (64, 256)              3.9e+04 ×                       275 ×
        (466, 195)                4505 ×                      49.6 ×
        (2048, 2048)              51.1 ×                      1.07 ×

    A flat tolerance tilts the wrong way — loosest on the smallest frames, where a spurious coupling
    is most likely. A band that tracks the arithmetic has no such tilt.
    """
    eps = float(np.finfo(W.dtype).eps) if np.issubdtype(W.dtype, np.floating) \
        else float(np.finfo(float).eps)
    return split_walk_rounding(int(W.size), energy, eps, splits=splits)


def correlation_length(rows, *, seed: int = 0) -> Optional[float]:
    """The attenuation scale ξ, from the instrument's decay profile — the autocorrelation length of the
    given ordered frame. `rows` is (T, F) with F >= 2; None when the frame carries no read."""
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics import Aperture
    try:
        return float(Aperture(W, window=None, seed=seed).correlation_length)
    except Exception:
        return None


def diffraction(rows, *, seed: int = 0) -> Optional[dict]:
    """The full diffraction read: `a_delta` (the resolvable spot), `xi`, and the Abbe factor.

    `a_delta` is the instrument's resolution — the smallest separation it can distinguish. That is
    the same quantity `minhash.estimator_limit` derives for a sampled proportion and
    `resolution.exact_limit` derives for an exact set, expressed for a decay profile."""
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics import diffraction_limit
    from entroptics.reads import decay
    try:
        dl = diffraction_limit(decay(W))
        return {"a_delta": float(dl.a_delta), "xi": float(dl.xi),
                "a_delta_abbe": float(dl.a_delta_abbe), "H": float(dl.H)}
    except Exception:
        return None


def spots(rows) -> Optional[int]:
    """The space-bandwidth product `n_F * n_T` — the frame's capacity in resolvable spots: how many
    independent readings the instrument can separate at this size and fill. Capacity, not content —
    `resolvable` reports how many things are present. `rows` is (T, F); None when unreadable."""
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics.reads import space_bandwidth
    try:
        return int(space_bandwidth(W))
    except Exception:
        return None


def fill(rows) -> Optional[dict]:
    """The instrument's fill fractions and area: `phi_F`, `phi_T`, `etendue = phi_F * phi_T` — how much
    of each axis the signal occupies (near 1 uses the whole basis; near 0 occupies a corner).
    `rows` is (T, F); None when unreadable."""
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics.reads import etendue, phi_F, phi_T
    try:
        return {"phi_F": float(phi_F(W)), "phi_T": float(phi_T(W)), "etendue": float(etendue(W))}
    except Exception:
        return None


def scales(rows, windows=None) -> Optional[list]:
    """Structure vs observation window — the signal read at several window sizes (resolution vs
    instrument size). `rows` is (T, F); None when unreadable.

    One entry per window, ascending, each carrying that window's own read::

        [{"window", "k_signal", "contrast", "coherence", "a_delta", "phi_T", "transition"}, ...]

    Same shape as `mantle/search/beacon/instrument.py::scales`, which is the other embodiment of
    this contract member, so a caller comparing two profiles compares rungs by `window` rather than
    by position. `transition` marks a window where the resolved count changed from the previous one.

    The two scalars `ScaleProfile` derives are recoverable from the list rather than being returned
    beside it: `resolved_window` is the first entry with `k_signal >= 1` — the shortest window at
    which anything stands above the floor, which is the principled minimum context length — and
    `dominant_window` is the entry of maximal `coherence`.

    The member names `k_signal`, `coherence`, `noise_floor` and `mp_edge` directly, because
    `entroptics.reads.ScaleProfile` carries no `rows` / `profile` / `scales` / `points`, and every
    per-window quantity on it is a numpy array — a dict comprehension filtered on
    `isinstance(..., (int, float))` drops all of them and leaves a single dict of the two scalars.
    The profile is the point of the read (a
    resolved count that keeps growing as the instrument widens is what scale-free means, stated
    directly rather than inferred from a log-log slope), so it is unpacked by name here.
    """
    import numpy as np
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics.reads import scale_profile
    try:
        prof = scale_profile(W, windows)
        ws = np.asarray(prof.windows).tolist()
        if not ws:
            return None                 # no rung carried a read — not a profile measured and flat
        trans = {int(w) for w in np.asarray(prof.transitions).tolist()}
        # Read by name off the documented fields. A missing one is a contract break rather than
        # something to paper over, so this does not getattr-with-default.
        cols = (("k_signal", prof.K_signal), ("contrast", prof.contrast),
                ("coherence", prof.coherence), ("a_delta", prof.a_delta), ("phi_T", prof.phi_T))
        out = []
        for i, w in enumerate(ws):
            row = {"window": int(w), "transition": bool(int(w) in trans)}
            for name, arr in cols:
                v = np.asarray(arr)[i]
                row[name] = int(v) if name == "k_signal" else float(v)
            out.append(row)
        return out or None
    except Exception:
        return None


def correlated_null(*, draws: Optional[int] = None, far: Optional[float] = None):
    """The distribution-free null for correlated rows — what retrieved evidence always is.

    The library's own default (`mp`, the i.i.d.-Gaussian edge) "conflates bulk correlation with
    signal" by its own docstring. Every read in this codebase is over rows that are correlated by
    construction, because that is what retrieval means, so this is the null they pass.

    `far` pins this provider's own false-alarm level; `None` leaves it at the read's — see the `far`
    block at the head of this module for why the level lives with the null.

    `draws` is compute rather than a modelling choice. A sampled null is empirical, so the draw
    count fixes the finest p it can resolve (`1/draws`) and costs CPU linearly, which makes it a
    question for the instrument's resource envelope. `None` leaves it at
    `entroptics.null_providers.permutation`'s own default, stated in the one place it is written
    down.

    The envelope cannot answer it yet: `prism.envelope` measures memory (cgroup / job object /
    host), CPU quota and disk, and a draw count is time. Converting a measured core count into a
    draw count needs a per-draw cost and an allowed wall-clock, and neither is published here. So
    the instrument states no number, the instrument's own default stands, and a caller that has
    measured its own budget passes `draws` explicitly (as `chorus/ophan/market_frame.py` does)."""
    from entroptics import permutation
    kw = {}
    if draws is not None:
        kw["draws"] = int(draws)
    return permutation(far=far, **kw)


def derived_null(*, far: float):
    """The library's own derived edge (`mp`, finite-size Johnstone / Tracy-Widom), pinned to a
    stated false-alarm level — the closed-form counterpart of `correlated_null`.

    This is where a level goes. It is a property of the null, in entroptics' own words
    (`null_providers`: *"The false-alarm level (alpha, `far`) travels WITH the null, not beside it:
    the cutoff is ONE decision, so the provider owns both the threshold and the alpha it is drawn
    at"*). A caller with an opinion about how often it will call noise a coupling states it by
    handing over a null that holds that opinion.

    The TW1 quantile is inverted from the survival function rather than looked up, so an arbitrarily
    sharp level (1e-5 and beyond) is still a derived edge, with nothing tabulated or fitted."""
    from dataclasses import replace
    from entroptics.null_providers import mp

    level = float(far)
    if not (0.0 < level < 1.0):
        raise ValueError("far is a false-alarm probability and must lie in (0, 1); got %r" % (far,))

    def _provider(ctx):
        return float(mp(replace(ctx, far=level)))

    _provider.__name__ = "derived_null"
    _provider.far = level
    return _provider


def entropy_bits(weights) -> float:
    """Shannon entropy in bits of a non-negative weight array — `H(w) = -sum p log2 p`, `p = w/sum w`.

    The one entropy definition entroptics uses (geometry marginals, mode weights, spectra). Normalises
    internally. A single weight or an all-zero array is 0.0; a flat array of n weights is log2(n).

    An adapter, not an implementation: the four lines below are `numpy.asarray`, a finite-and-positive
    mask, and a call to `entroptics.entropy.shannon_bits`. Its floor is numpy + entroptics, identical
    to the rest of this module, which is why it lives at the instrument.

    A stdlib home in prism would add a copy rather than remove an import. prism does not import
    entroptics (the publication boundary — `test_contract_install_is_pure.py::PRIVATE`), so a
    `prism.entropy_bits` would re-implement the arithmetic, and the property that makes this function
    worth having is that a caller's entropy and entroptics' own entropy over geometry marginals and
    mode weights are the same callable.

    `mantle/search/beacon/engine.py::shannon_bits` is the other embodiment's copy, and the two
    embodiments do not import each other by design, so their agreement is evidence rather than a
    shared dependency.

    A host with no instrument has no way to compute a normalised entropy over a list of floats, and
    reports that absence rather than duplicating the arithmetic."""
    import numpy as np
    from entroptics.entropy import shannon_bits
    w = np.asarray([float(x) for x in weights], dtype=float)
    w = w[np.isfinite(w) & (w > 0.0)]
    if w.size <= 1:
        return 0.0
    return float(shannon_bits(w))


def joint_entropies(rows_x, rows_y, mask_x=None, mask_y=None) -> dict:
    """The three entropies of two co-registered frames and the three differences, from one joint
    table: `{"H_X", "H_Y", "H_XY", "I_XY", "H_X_given_Y", "H_Y_given_X"}` in bits.

    `I_XY` is the mutual information; `H_X_given_Y` is Shannon's equivocation (1948 §12) — what remains
    uncertain about the first frame once the second is known. It reads a failure the conservation
    certificate cannot: a hop may absorb every joule and still leave which signal arrived ambiguous,
    because `‖·‖²` counts energy and this counts distinguishability.

    One call rather than three, because the shared table is what makes the identities exact:
    `I_XY = H_X + H_Y − H_XY` and `I_XY = H_X − H_Y(X)` hold to the last bit rather than to float
    noise. The ordered lengths must match and a mismatch raises — frames on private axes superpose
    into noise, so a truncated read would report the misalignment as a measurement. The read is
    directed: `H_X_given_Y` and `H_Y_given_X` differ, and both come back named.

    An adapter, exactly as `entropy_bits` is: the arithmetic lives in `entroptics.entropy`, so a
    caller's joint read and the instrument's own are the same callable."""
    from entroptics.entropy import joint_entropies as _joint
    return dict(_joint(_as_ordered_matrix(rows_x), _as_ordered_matrix(rows_y), mask_x, mask_y))


# ═══════════════════════════════════════════════════════════════════════════════
# The streaming surface — the screen accumulates, and condensation is an event
# ═══════════════════════════════════════════════════════════════════════════════
# Everything above this line is a batch function: construct, read, discard. entroptics is also a
# stateful streaming instrument, and the surface below is its front door. `Aperture.update(frame)`
# accumulates, streams the dynamical operator from frame 0, and forgets adaptively — its own words:
# *"keep >= window frames, and MORE while a coherent signal is still active; forget only the
# decorrelated tail (the signal decides, not a clock)"*.
#
# Accumulating is what lets a count certify, because the certified band falls as sqrt(F/T):
#
#   one turn        T=34, F=195   band=16.26   interval [2,195]   never certifies
#   pooled 20 turns T=466         band= 2.13   interval [10,59]   certifies
#
# A single turn is a snapshot where the instrument expects a stream, and a count that cannot
# certify leaves a caller falling back on an Otsu split over a bare score column. As `T` grows with
# the conversation the band tightens and the count becomes a measurement rather than a fit.
def accumulator(n_features: int, *, whiten: bool = False):
    """A pooling screen: feed it one plane per turn, read one spectrum over everything seen.

    `.add(plane)` pools an intact `(T_p, F)` plane (each turn's frame), keeping the within-plane
    correlation intact; `.spectral()` reads the pooled `SpectralOptics`; `.band()` is the current
    certified band, which shrinks as the pooled `T` grows. `F` is constant across planes, which is
    what a fixed coordinate basis provides.

    `.merge(other)` is the across-peers path. Two nodes that have each been accumulating combine
    into one spectrum with only the pooled covariance travelling between them, so a peer
    contributes its evidence while its observations stay its own."""
    from entroptics import SpectralAccumulator
    return SpectralAccumulator(int(n_features), whiten=bool(whiten))


def accumulated_read(acc) -> Optional[dict]:
    """What a pooled `accumulator` currently resolves — the certified whether, alongside the count.

    Returns `{"planes"?, "T", "F", "band", "k_signal", "interval", "certified"}`, or None when the
    accumulator holds nothing. `certified` is True only when the Weyl interval has collapsed onto
    the count (`lo == hi == k`). `band` and `interval` come back alongside it so a caller can see how
    far from certification it is; the band shrinks as pooled `T` grows, making this a progress
    reading rather than a verdict.

    The read lives here, at the one instrument, and callers call it. A second entry point into
    `entroptics.reads` is a second place that can read a different instrument than everyone else:
    `read()`/`Screen()` apply the entropy fold guard that destroys a sparse carrier (measured: 256
    channels -> F_eff = 1, reported as K_signal = 1)."""
    if acc is None:
        return None
    try:
        import numpy as _np
        from entroptics.reads import resolved_dimension_interval
        T = int(getattr(acc, "T", 0) or 0)
        if T <= 0:
            return None
        F = int(acc.F)
        spec = acc.spectral()
        band = float(acc.band())
        # The interval is a property of the pooled spectrum; the frame argument only carries F.
        ci = resolved_dimension_interval(_np.zeros((2, F)), band=band, sg=spec)
        lo, hi = int(ci.resolved_lo), int(ci.resolved_hi)
        k = int(spec.resolved_modes)
        return {"T": T, "F": F, "band": band, "k_signal": k, "interval": (lo, hi),
                "certified": lo == hi == k}
    except Exception:
        return None


def stream(*, window: Optional[int] = None, null=None, seed: int = 0,
           forgetting: float = 1.0, rank=None):
    """A living instrument — hold it, feed it `update(frame)`, read it whenever.

    `read_ordered` is the call for a one-shot read of a complete frame; this is for a signal that
    keeps arriving — turns, peer signals, sensor frames — where the screen remembers between reads.
    The dynamical (predictive) operator is streamed from frame 0, which is what gives
    `Dynamics`-backed reads their meaning; a per-call refit has no history to stream from.

    `window=None` keeps the adaptive horizon (the signal decides how much to keep); an integer pins
    a floor under it."""
    from entroptics import Aperture
    return Aperture(window=window, null=null, seed=seed, forgetting=forgetting, rank=rank)


def beam(rows, *, null=None, seed: int = 0):
    """The frame as a beam — what this signal carries.

    Singular by design. A `Beam` is a bundle of beams: ask it for `.modes` and it decomposes into
    its constituents, each of which decomposes the same way, down to a leaf that spans one
    direction. So the surface is one `beam` property containing the others, with `footprints` as
    the flat mode list.

    A `Beam` is the flow carrier: `energy` (by its own law, about its own zero), `flow` (that energy
    per ordered step), `basis` (the directions it spans), `profile` (amplitude along each), and
    `modes`, which decompose the same way at every depth.

    `etendue = phi_T * phi_F` is, in entroptics' own words, *"the phase space this beam occupies,
    and THE CONSERVED INVARIANT A CROSSING IS SETTLED BY"*. That makes conservation of information a
    measurable quantity: a crossing that changes it has created or destroyed something, and the beam
    can be asked whether it did.

    Returns None when the frame cannot carry a read, rather than a fabricated decomposition."""
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics import Projection
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return Projection(W, null=null, seed=seed).beam


def self_information_bits(observed: float, total: float) -> Optional[float]:
    """`I(x) = -log2(p)` — how much observing x narrows the field, in bits.

    The summand of the entropy `entropy_bits` sums: `H = Σ p·I(x)`. Both come from
    `entroptics.entropy` — same base, same clip, same module — so the two stay one definition of
    "information in bits".

    A thing seen in nearly every observation carries almost nothing: `p→1` gives `I→0`. A thing seen
    rarely carries a lot. That is the whole content of the measure, and it is what lets it rank a
    word by evidence rather than by a hand-written stop-list.

    Returns `None` when no probability can be formed — `total <= 0`, `observed <= 0`, or
    `observed > total`. Zero bits is a real reading (the thing is everywhere and narrows nothing),
    so an unmeasurable count carries `None` instead.

    A wrapper rather than an implementation: the arithmetic is
    `entroptics.entropy.surprisal_bits`, beside `shannon_bits`, sharing its base and its clip. This
    delegates exactly as `entropy_bits` does."""
    from entroptics.entropy import surprisal_bits
    return surprisal_bits(observed, total)


def coherence(rows, *, seed: int = 0) -> Optional[float]:
    """Strehl — how much of the energy sits in the leading mode of the ordered axis.

    Precisely (entroptics): `lambda_1 / sum(lambda)` of `axis_spectrum(W, 0)`, the correlation
    eigenspectrum taken with the T rows as the variables. The question it answers is "are the
    frame's rows all the same thing?" — 1.0 means every row is one coherent object, low means the
    rows do not agree.

    Which axis is the whole content of this docstring, because the feature-mode share is a
    different quantity (`OpticsRead.top_share`) and the two separate sharply. Measured, 240 rows:

        frame (240 x 8)                       strehl   top_share   lag1_z
        ────────────────────────────────────  ───────  ─────────   ──────
        i.i.d. noise                           0.171     0.169     + 0.65
        rows are the same vector + noise       0.969     —         —
        columns share a driver, rows i.i.d.    0.185     0.968     - 0.78   <-- the trap
        one slow trend across all columns      0.189     0.983     +15.39

    Row 3 is the trap: the columns are ~0.96 correlated and `top_share` sees it at 0.968, while
    strehl reads 0.185 — noise. Both are right. A shared driver whose value changes sign and
    magnitude every step gives the columns a common mode but leaves each row a different scalar
    multiple of it, so the rows genuinely are not the same thing. Column structure is not row
    structure.

    Pick by the axis your structure lives on:
      * rows are repeated observations of one object (a retrieved evidence set, a set of senses that
        should agree) -> this function. It discriminates strongly in both aspect regimes: 0.93 vs
        0.13 at 8x2048, 0.97 vs 0.18 at 240x8.
      * features share a common mode (instruments under one driver, channels under one source)
        -> `read_ordered(...).top_share` plus `k_signal` for how many clear the null.

    This module carries a name collision worth knowing: `OpticsRead.coherence` is a third quantity,
    the ordered-axis lag-1 coherence z-score `mean_i Re<row_i, row_{i+lag}>²` (module header), which
    is what `certificate()` reports. Lag-1 asks "does each row resemble its neighbour"; strehl asks
    "do all the rows agree". Row 4 above separates them: a slow trend is strongly lag-1 coherent
    (+15.4) while its rows, being sign-flipping multiples, are not globally coherent (0.189)."""
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics.reads import strehl
    try:
        return float(strehl(W))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# The dynamics side — same door
# ═══════════════════════════════════════════════════════════════════════════════
# `entroptics.dynamics`, `delay_embed` and `null_providers` come through this seam like everything
# else: one import site, so a defect in how the instrument is used is fixed once.
#
# These wrappers are thin by design. Their job is to be the import site and to carry the measured
# caveats, rather than to re-abstract a library that is already the right shape.

def screen_normalize(rows, mask=None):
    """Normalise a screen — the whole (T, F) frame — by entroptics' own MAD whitening.

    Normalisation happens on the screen. Dividing an individual vector by its norm is the move that
    produces cosine similarity, which is out of scope in this domain, and it discards the magnitude
    the conservation certificate is stated over.

    This is the published way to normalise a screen. `entroptics.entropy.normalize` is reachable
    only from inside this module (`test_only_the_instrument_seam_imports_entroptics`), so a persona that
    needs a whitened frame calls this rather than hand-rolling per-vector division.

    Whitening can amplify. Per-channel MAD whitening rescales a frame with heterogeneous channel
    occupancy by up to ~1e12 (measured), and a sparse coordinate can have zero MAD in every channel.
    `whitening_amplification` on the spectral read reports how far a given frame moved; callers that
    care about comparability across frames read that field."""
    W = _as_ordered_matrix(rows)
    if W is None or W.size == 0:
        return None
    from entroptics.entropy import normalize as _normalize
    try:
        return _own_precision(_normalize(W, mask))
    except Exception:
        return None


class OperatorRead(object):
    """The dynamical operator of an ordered stream, together with how far it may be rolled.

    The horizon is what this type carries beyond the operator. `fit_dynamics` hands back an operator
    that steps forever, and rolled past its own decay it reports the collapse rather than a
    prediction — 60 steps against a horizon of 4 produce output that reads as an answer and is
    noise. So the bound travels with the operator, and `roll()` stays within it.

      ℓ = ⌈ln(contrast) / (−ln m)⌉    contrast = λ₁/edge (the instrument's), m = the forgetting margin

    `horizon` is `None` when no bound can be computed — contrast ≤ 1 (nothing resolved above the
    floor) or m ∉ (0,1) (the dominant mode does not decay, so it is growth). `None` carries the
    reason in `.refusal` and is distinct from "zero steps": treating it as 0 turns "there is no
    reading" into "the reading is nothing" ([[absence-is-not-an-affirmative-claim]]).
    """

    __slots__ = ("contrast", "resolved_modes", "margin", "horizon", "refusal", "_dyn", "_width")

    def __init__(self, contrast, resolved_modes, margin, horizon, refusal, dyn, width):
        self.contrast, self.resolved_modes = contrast, resolved_modes
        self.margin, self.horizon, self.refusal = margin, horizon, refusal
        self._dyn, self._width = dyn, width

    def roll(self, state):
        """Yield `(step, distribution)` for each step within the horizon.

        The distribution comes back whole, with no argmax. What the state carries across the feature
        block is the answer's shape, so "multiple items at once" falls out of the operator rather
        than from a cut applied afterwards. Yields nothing when there is no horizon to roll."""
        import numpy as np
        if self.horizon is None:
            return
        x = state
        for step in range(1, int(self.horizon) + 1):
            x = self._dyn.predict(x)
            blk = np.asarray(x).ravel()[-self._width:]
            total = float(np.sum(np.abs(blk))) or 1.0
            yield step, (blk / total)


def sequence_operator(rows, *, depth: int, window_rows_per_feature: int = 1):
    """Fit the operator of an ordered stream and measure how far it may be rolled.

    `rows` is the trajectory (T×D, one row per step, order carrying the whole signal); `depth` is the
    delay-embedding depth the caller asks for. Returns an `OperatorRead`, or `None` when the stream
    is too short to embed and there is nothing to fit."""
    import math

    import numpy as np
    A = _own_precision(rows)
    if A.ndim != 2 or A.shape[0] <= depth:
        return None
    from entroptics import Aperture, delay_embed, koopman_lift
    Z = delay_embed(A, depth)
    F = int(Z.shape[1])
    # The window is sized so that T > F: below that the correlation cannot be read, and a
    # degenerate frame reads identically to "no signal".
    ap = Aperture(Z, window=int(window_rows_per_feature) * F)
    sp = ap.spectral
    dyn = koopman_lift(A, depth)
    m = float(dyn.forgetting()["margin"])
    contrast = float(sp.contrast)
    horizon, refusal = None, None
    if contrast <= 1.0:
        refusal = "nothing resolved above the floor (contrast <= 1)"
    elif not (0.0 < m < 1.0):
        refusal = "the dominant mode does not decay (|mu| >= 1 is growth)"
    else:
        horizon = int(math.ceil(math.log(contrast) / (-math.log(m))))
    return OperatorRead(contrast, int(sp.resolved_modes), m, horizon, refusal, dyn, int(A.shape[1]))


def surrogate_significance(seq, *, draws: int = 200, n_max: int = 8, seed: int = 0):
    """Is this stream's block structure distinguishable from its own shuffle?

    The shuffled control, read off the sequence itself rather than against a typed-in threshold —
    the measurement that separates "the reader learned the text" from "the reader learned the
    alphabet's frequencies". Returns entroptics' dict, or `None` when the sequence carries no
    read."""
    import entroptics.sequence as _sequence
    try:
        return _sequence.surrogate_test(seq, draws=draws, n_max=n_max, seed=seed)
    except Exception:
        return None


def embed(X, d: int):
    """Takens delay embedding — `d` copies of the trajectory, lagged.

    Returns `X` unchanged when the series is too short to embed at that depth, rather than producing
    a frame with fabricated rows."""
    import numpy as np
    A = _own_precision(X)
    if d <= 1 or A.ndim != 2 or A.shape[0] <= d:
        return A
    from entroptics import delay_embed
    return delay_embed(A, d)


def fit_dynamics(X, *, forgetting: float = 1.0, rank=None):
    """Fit the Koopman/DMD dynamics of an ordered trajectory. Returns the entroptics object.

    Order is the whole signal here, more so than for a static read: a shuffled trajectory has no
    dynamics to fit, and the fit still returns an object, so the caller supplies the trajectory in
    the order it happened."""
    import numpy as np
    A = _own_precision(X)
    if A.ndim != 2 or A.shape[0] < 2:
        return None
    from entroptics.dynamics import dynamics as _fit
    return _fit(A, forgetting=forgetting, rank=rank)


def dynamics_state(n_features: int, *, forgetting: float = 1.0, rank=None):
    """An empty streaming `Dynamics` accumulator of the given width — for callers that ingest
    frame by frame rather than fitting a whole trajectory at once."""
    from entroptics.dynamics import Dynamics
    try:
        return Dynamics(int(n_features), forgetting=forgetting, rank=rank)
    except TypeError:
        return Dynamics(int(n_features))


def decay_profile(W):
    """The signal's own ordered-axis autocorrelation C(tau) — its optical transfer function."""
    import numpy as np
    A = _own_precision(W)
    if A.ndim != 2 or _null_dof(A)[0] <= 0:
        return None
    from entroptics.reads import decay
    return decay(A)


def resolution_limit(profile):
    """`DiffractionLimit` from a 1-D decay profile: `a_delta`, `xi`, the Abbe factor."""
    if profile is None:
        return None
    from entroptics import diffraction_limit
    return diffraction_limit(profile)


# ═══════════════════════════════════════════════════════════════════════════════
# The certified reads — coherence, scale-invariance, and the Mercer certificate
# ═══════════════════════════════════════════════════════════════════════════════
# The screen-count investigation (§13.34) established, from four independent measurements, that
# WordNet's coordinate is smooth rather than clustered, so there is no sharp integer mode-count to
# read on it. What is certified and clean there is the coherence and its behaviour across scale: a
# reached set is "one thing" exactly when its coupling holds constant as the instrument widens.
#
# `.strehl`, `.scale_profile` and `.mercer` are published here so the answer path reads them through
# the instrument rather than by constructing its own `Aperture`.

def scale_read(rows) -> Optional[dict]:
    """Structure vs observation window. The instrument sweeps trailing windows of the ordered axis
    itself and reports how the resolved structure changes with how much you look at.

    Returns `dominant_window` (where the coherent structure peaks — the full set for a single
    coherent thing, earlier for a set whose tail adds unrelated senses), `resolved_window` (where
    structure first settles), and the per-window `coherence` profile. `scale_invariant` is True when
    the dominant window is essentially the whole frame — the certified "one thing" reading."""
    import numpy as np
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics.reads import scale_profile
    try:
        p = scale_profile(W)
    except Exception:
        return None
    win = np.asarray(p.windows).tolist()
    coh = np.asarray(p.coherence).tolist()
    T = W.shape[0]
    dom = int(p.dominant_window)
    return {"dominant_window": dom, "resolved_window": int(p.resolved_window),
            # Full precision — see the truncation note in `as_read`.
            "windows": win, "coherence": [float(c) for c in coh],
            # The scale-invariant reading is exact and needs no fraction of T. The profile sweeps a
            # discrete ladder of trailing windows and reports which one the coupling peaks at, so
            # "the coupling peaks over the whole instrument" is `dominant_window == the widest window
            # examined`. Measured across T = 40 / 64 / 200 / 466 on coherent frames,
            # `dominant_window` equals `max(windows)` equals `T` in every case. Comparing against a
            # fraction of T instead would admit a profile peaking at the second-widest window
            # whenever the ladder's spacing happened to put it within that fraction — a property of
            # the ladder, not of the signal. `ember/signal/projection.py` consumes this reading on
            # the live answer path.
            "scale_invariant": bool(win) and dom == max(win)}


def certificate(rows) -> Optional[dict]:
    """The Mercer certificate: `a_delta` read the temporal way (decay entropy) and the spectral way
    (stationary eigenspectrum). `ratio` ~ const validates the read; a departure flags
    non-stationarity — that is, more than one mode.

    `rows` is (T, F); None when the frame carries no read."""
    import numpy as np
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics.reads import mercer_certificate
    try:
        m = mercer_certificate(W)
        return {"ratio": float(m.ratio), "a_delta_temporal": float(m.a_delta_temporal),
                "a_delta_spectral": float(m.a_delta_spectral), "n_dof": float(m.n_dof)}
    except Exception:
        return None



# ── Set reads — order-invariant, for a cloud rather than a trajectory ────────────────────────────
# Which axis. Everything above this line reads the ordered axis: `read_ordered` takes a (T, F) frame
# whose row order is meaningful, and `coherence` is a lag-1 z-score that a shuffle destroys by
# design. The reads below are the opposite kind — they take a set of row-vectors and are invariant
# under permutation. Match the read to what you hold: a bag of members belongs here, an ordered
# trajectory above.

def concentration(vectors, *, normalize: bool = True) -> Optional[dict]:
    """Fisher-information concentration of a cloud of row-vectors — how sharply it focuses on its
    dominant axis. Order-invariant: this is the read for a set.

    Returns `{intensity, focus, resultant, n, dim}`, or None when the cloud carries no read.

      * `intensity` — sigma_1^2, the top squared singular value. Extensive: it grows with the
        number of vectors, so it compares clouds only at equal `n`.
      * `focus` — sigma_1^2 / M, the power fraction on the leading principal axis, in (0, 1] for
        unit rows. Intensive, and the one to compare across clouds of different size.
      * `resultant` — the mean-vector length; 1.0 when every row points the same way.

    Answers *"is this focused, or a pile?"* natively, in the instrument's own terms, which is what
    makes it the read that proposes anchors: it takes no `k` chosen in advance and its result does
    not depend on the order the cloud arrived in.
    """
    import numpy as np
    V = np.asarray(vectors, dtype=float)
    if V.ndim != 2 or V.shape[0] < 2 or V.shape[1] < 1 or not np.any(np.isfinite(V)):
        return None
    from entroptics.reads import concentration as _c
    try:
        c = _c(V, normalize=normalize)
    except Exception:
        return None
    return {"intensity": float(c.intensity), "focus": float(c.focus),
            "resultant": float(c.resultant), "n": int(c.n), "dim": int(c.dim)}


def mode_significance(rows) -> Optional[list]:
    """Per-mode Tracy-Widom tail probabilities for the screen's singular spectrum — evidence rather
    than a cutoff.

    Returns `[p_0, p_1, …]`, one per mode, or None when the frame carries no read. The resolved
    count satisfies `K_signal == #(p_k < alpha)` for whatever level the read's null was drawn at, so
    this shows *why* a mode did or did not resolve rather than only how many did, as a continuous
    probability from which no cutoff can be read back out. The level belongs to the null
    (`correlated_null` / `derived_null`) — see the `far` block at the head of this module.
    """
    import numpy as np
    W = _as_ordered_matrix(rows)
    if not _carries_read(W):
        return None
    from entroptics import mode_significance as _ms
    try:
        return [float(p) for p in _ms(W).p]
    except Exception:
        return None


def error_bar(samples, read, *, n_bins: Optional[int] = None) -> Optional[dict]:
    """Delete-one jackknife point estimate and standard error for any scalar `read`.

    `read` is `callable(subset) -> float`, evaluated on the full set and on each delete-one subset.
    Returns `{value, se}`, or None when the sample carries no read.

    This is how a number acquires a stated uncertainty rather than an implied one. A read without an
    error bar invites the reader to treat its last digits as meaningful; this makes the claim
    checkable, and a wide bar is itself the finding.
    """
    import numpy as np
    S = np.asarray(samples, dtype=float) if not isinstance(samples, (list, tuple)) else samples
    # The delete-one jackknife's spread is taken over the delete-one subsets, so it exists exactly
    # when there is more than one of them to vary over: `n - 1 >= 1`. Stated as the estimator's own
    # degrees of freedom rather than as a chosen minimum count.
    if (len(S) - 1) < 1:
        return None
    from entroptics import jackknife
    try:
        j = jackknife(S, read, n_bins=n_bins)
    except Exception:
        return None
    val = getattr(j, "value", None)
    se = getattr(j, "se", None)
    if val is None or se is None:                      # tuple-shaped return
        try:
            val, se = j
        except Exception:
            return None
    return {"value": float(val), "se": float(se)}


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# Proximity — the collection digest, and the probe over it
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#
# `entroptics.proximity` is a magnitude-carrying, width-free spectral digest of a frame. It is the
# one entroptics surface mantle is BUILT to consume and cannot reach: `collection_frame`,
# `digest_refresh` and `collection_proximity` all take `read` / `engine_id` / `probe_factory` as
# injected seams, and every one of their docstrings names these functions as the intended filler.
# Mantle never imports entroptics (the publication boundary), so a host supplies them — and the
# host may only reach entroptics through this module (`tests/test_one_instrument.py`).
#
# These are thin by design. Unlike the reads above they add no policy: proximity's own contract is
# already "not a recall path, not a ranking, not a retrieval policy — a capability", so wrapping it
# in an opinion here would be inventing one.

def proximity_read():
    """`mp_deviation` — the digest function `digest_collection` / `CollectionDigestRefresher` want.

    Every direction a frame resolves, expressed as how far it stands from where the frame's OWN
    noise law says it should. No width, no rank, no rate, no tolerance, no learned constant: each
    quantity is a function of the frame.

    Pass with :func:`proximity_engine_id`, never alone. A digest carries the id of the instrument
    that took it because a digest taken against the bulk edge is not comparable with one taken
    against the per-mode prediction, and `collection_proximity` refuses the cross-engine comparison
    rather than silently mixing them."""
    from entroptics.proximity import mp_deviation
    return mp_deviation


def proximity_engine_id() -> str:
    """The instrument id that travels with every digest `proximity_read` takes."""
    from entroptics.proximity import ENGINE_ID_PROXIMITY
    return ENGINE_ID_PROXIMITY


def proximity_probe_factory():
    """`SpectrumProbe` — the `probe_factory` `CollectionProximityNarrower` wants.

    EXACT, not approximate: a one-dimensional sorted key index whose range scan is lossless by the
    componentwise bound `|x_j - y_j| <= ||x - y||`, so `within(q, R)` returns precisely what a full
    scan returns and `nearest(q, k)` stops only when the key gap exceeds the k-th best full
    distance. That is why it can sit under a narrowing: it prunes work, never candidates."""
    from entroptics.proximity import SpectrumProbe
    return SpectrumProbe


def spectral_distance(a, b) -> float:
    """Distance between two proximity digests — plain L2 on their common prefix.

    Deliberately un-normalised: a relative distance would destroy the magnitude the digest is built
    to keep. Modes only one record has are DROPPED rather than zero-padded, because the per-mode
    prediction is undefined beyond a record's own width — measured, zero-padding recalls 0.717 under
    a 5% row deletion where the common prefix recalls 0.983."""
    from entroptics.proximity import spectral_distance as _sd
    return float(_sd(a, b))
