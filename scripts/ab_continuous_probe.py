"""Unit AB — does the fold-criteria disagreement dissolve on continuous input?

Unit AA (`aa_fold_probe.py`) measured, on symbolic Jiang-Conrath frames, that entroptics'
two fold criteria disagree: the entropy guard (`entropy.py:135`) folds a sparse carrier that
the locality criterion (`batch.py:277-306`) holds must stay unfolded. `ember/optics.py` exists
to route around that door.

The library's own locality docstring states the condition under which folding is valid:

    "Fold is valid iff ADJACENT feature columns are correlated MORE than FAR ones (a smooth
     ordered continuum where area-averaging coarsens). ... sparse spike (narrowband carrier)
     adj ~ 1/T -> no fold"

An RF spectral frame is a smooth ordered continuum. So the prediction under test is:

    On genuinely continuous spectral input the two criteria agree, K_signal tracks the
    planted mode count, and the noise floor stops swinging orders of magnitude.

This script is where that prediction is measured; it is the first run of entroptics on a real
spectral frame in this codebase. It is falsifiable: if the criteria still disagree on a smooth
waterfall, the fold door is a library problem rather than an input-mismatch problem, and Phase 6
of AGENT-HOST-DESIGN.md is wrong.

No trained models, no corpus, no store, no nltk. Pure synthetic signals with known ground truth,
so every read can be scored.

The knob that matters is resolution bandwidth. A delta-thin tone occupies one bin (sparse spike ->
no fold). A real receiver has finite resolution, so a real tone occupies several adjacent bins
(smooth continuum -> fold). Sweeping it is the experiment: at what bandwidth do the criteria
converge?

Run:  python scripts/ab_continuous_probe.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aa_fold_probe import fold_diagnostics, reads, fmt  # noqa: E402  (same-dir sibling)

SEED = 20260721
T_DEFAULT = 256          # ordered axis = time (real, physical, reproducible)
F_DEFAULT = 256          # feature axis = frequency bin


# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic waterfalls with KNOWN mode count
# ─────────────────────────────────────────────────────────────────────────────
def _bins(F: int) -> np.ndarray:
    return np.arange(F, dtype=np.float64)


def _kernel(F: int, centre: float, bw: float) -> np.ndarray:
    """One spectral line with finite resolution bandwidth `bw` (in bins).

    bw -> 0 is a delta (a sparse spike — the pathological case).
    bw >= ~2 is what any real receiver produces (a smooth continuum).
    """
    x = _bins(F) - centre
    return np.exp(-0.5 * (x / max(bw, 1e-6)) ** 2)


def white_noise(T=T_DEFAULT, F=F_DEFAULT, *, rng) -> tuple[np.ndarray, int]:
    """No structure. Ground truth K = 0."""
    return rng.standard_normal((T, F)), 0


def carriers(T=T_DEFAULT, F=F_DEFAULT, *, k=3, bw=0.4, snr=8.0, rng) -> tuple[np.ndarray, int]:
    """`k` steady narrowband tones. Ground truth K = k.

    At bw<1 this is the sparse spike the locality criterion leaves unfolded — the continuous
    analogue of a sparse ontology coordinate.
    """
    W = rng.standard_normal((T, F))
    centres = np.linspace(F * 0.2, F * 0.8, k)
    for i, c in enumerate(centres):
        amp = snr * (1.0 + 0.15 * i)
        env = np.sin(2 * np.pi * (0.01 + 0.004 * i) * np.arange(T)) + 1.5
        W += amp * np.outer(env, _kernel(F, c, bw))
    return W, k


def chirp(T=T_DEFAULT, F=F_DEFAULT, *, bw=3.0, snr=8.0, rng) -> tuple[np.ndarray, int]:
    """A linearly sweeping tone — the canonical RF waterfall. Ground truth K is not 1:
    a sweep is not a stationary mode, so we report it without a hard expectation."""
    W = rng.standard_normal((T, F))
    centres = np.linspace(F * 0.15, F * 0.85, T)
    for t in range(T):
        W[t] += snr * _kernel(F, centres[t], bw)
    return W, -1          # -1 = no clean stationary ground truth


def smooth_band(T=T_DEFAULT, F=F_DEFAULT, *, k=3, bw=14.0, snr=6.0, rng) -> tuple[np.ndarray, int]:
    """`k` wide overlapping bands — an unambiguously smooth ordered continuum.
    Ground truth K = k. This is the case the fold was designed for."""
    W = rng.standard_normal((T, F))
    centres = np.linspace(F * 0.25, F * 0.75, k)
    for i, c in enumerate(centres):
        env = np.cos(2 * np.pi * (0.007 + 0.003 * i) * np.arange(T)) + 1.5
        W += snr * np.outer(env, _kernel(F, c, bw))
    return W, k


def iq_carriers(T=T_DEFAULT, F=F_DEFAULT, *, k=2, bw=3.0, snr=8.0, rng) -> tuple[np.ndarray, int]:
    """Complex I/Q — phase is real information. Ground truth K = k.

    entroptics accepts complex natively (`instrument.py:17`), and `ember/optics.py` materialises at
    the frame's own precision, so a complex frame keeps its phase through the seam. The comparison
    below shows what a plain `dtype=np.float64` coercion does to the same frame."""
    W = (rng.standard_normal((T, F)) + 1j * rng.standard_normal((T, F))) / math.sqrt(2.0)
    centres = np.linspace(F * 0.3, F * 0.7, k)
    for i, c in enumerate(centres):
        phase = np.exp(1j * 2 * np.pi * (0.02 + 0.01 * i) * np.arange(T))
        W += snr * np.outer(phase, _kernel(F, c, bw))
    return W, k


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scoring
# ─────────────────────────────────────────────────────────────────────────────
def verdict(fd: dict) -> str:
    e, l = fd["entropy_folds"], fd["locality_folds"]
    if e == l:
        return "AGREE   (%s)" % ("fold" if e else "no-fold")
    return "DISAGREE (entropy=%s locality=%s)" % ("fold" if e else "no-fold",
                                                  "fold" if l else "no-fold")


def run_case(name: str, W: np.ndarray, truth: int) -> dict:
    print(f"\n{'-'*78}\n{name}   shape={W.shape} dtype={W.dtype}"
          + (f"   ground-truth K={truth}" if truth >= 0 else "   (no stationary truth)"))
    try:
        fd = fold_diagnostics(W)
    except Exception as exc:
        print(f"  fold_diagnostics FAILED: {type(exc).__name__}: {exc}")
        return {}
    print(f"  H_F={fd['H_F']:.3f} log2F={fd['log2F']:.3f} band={fd['band_F']:.3f} "
          f"occ={fd['occupancy']:.3f}")
    print(f"  -> {verdict(fd)}   F_eff(entropy)={fd['F_eff_entropy']}")
    try:
        rd = reads(W)
    except Exception as exc:
        print(f"  reads FAILED: {type(exc).__name__}: {exc}")
        return {"fold": fd}
    floors = []
    for kname in ("A_read", "B_instrument_screen", "C_auto", "D_native",
                  "E_forced_fold", "F_spectral"):
        v = rd.get(kname)
        if not v:
            continue
        floors.append(v["floor"])
        hit = ""
        if truth >= 0:
            hit = "  <== matches truth" if v["K"] == truth else f"  (off by {v['K']-truth:+d})"
        print(f"    {kname:20s} K={v['K']:4d} floor={fmt(v['floor']):>12s}{hit}")
    pos = [f for f in floors if f > 0]
    if len(pos) > 1:
        print(f"    floor spread across paths: {max(pos)/min(pos):.3g}x   "
              f"(symbolic frames measured ~1e13x)")
    return {"fold": fd, "reads": rd}


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("UNIT AB — fold criteria on CONTINUOUS input")
    print("PREDICTION: on a smooth ordered continuum the entropy and locality guards AGREE.")
    print("=" * 78)

    run_case("white noise (no structure)", *white_noise(rng=rng))
    run_case("3 WIDE bands  bw=14  (smooth continuum — the fold's design case)",
             *smooth_band(rng=rng))
    run_case("chirp  bw=3  (canonical RF waterfall)", *chirp(rng=rng))

    print(f"\n{'='*78}\n### THE SWEEP — resolution bandwidth, 3 carriers, everything else fixed"
          f"\n### delta-thin = sparse spike; wide = smooth continuum\n{'='*78}")
    rows = []
    for bw in (0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 8.0, 14.0, 24.0):
        W, truth = carriers(bw=bw, rng=np.random.default_rng(SEED))
        fd = fold_diagnostics(W)
        try:
            rd = reads(W)
            k_spec = rd["F_spectral"]["K"]
            k_auto = rd["C_auto"]["K"]
            fl = rd["A_read"]["floor"]
        except Exception:
            k_spec = k_auto = -1
            fl = float("nan")
        rows.append((bw, fd, k_spec, k_auto, fl, truth))
        print(f"  bw={bw:5.1f}  H_F={fd['H_F']:6.3f}  entropy={'fold ' if fd['entropy_folds'] else 'keep '}"
              f"  locality={'fold ' if fd['locality_folds'] else 'keep '}"
              f"  {'AGREE' if fd['entropy_folds']==fd['locality_folds'] else 'DISAGREE'}"
              f"   K_spectral={k_spec:3d} K_auto={k_auto:3d} (truth {truth})  floor={fmt(fl)}")

    agree_bw = [r[0] for r in rows if r[1]["entropy_folds"] == r[1]["locality_folds"]]
    print(f"\n  criteria AGREE at bw = {agree_bw}")
    print(f"  criteria DISAGREE at bw = {[r[0] for r in rows if r[0] not in agree_bw]}")

    print(f"\n{'='*78}\n### COMPLEX I/Q — does the analog path survive?\n{'='*78}")
    Wc, truth = iq_carriers(rng=np.random.default_rng(SEED))
    run_case("complex I/Q, 2 carriers", Wc, truth)
    print("\n  now the beam/optics.py coercion, applied to the SAME frame:")
    try:
        coerced = np.asarray(Wc, dtype=np.float64)
        print(f"    np.asarray(W, dtype=np.float64) -> {coerced.dtype} (phase SILENTLY DROPPED)")
    except Exception as exc:
        print(f"    np.asarray(W, dtype=np.float64) -> {type(exc).__name__}: {exc}")
        print("    (raises rather than corrupts — the seam fails closed on I/Q)")

    print(f"\n{'='*78}\nInterpretation: if the criteria agree on wide/smooth frames and disagree")
    print("only at delta-thin bandwidths, the fold door is an INPUT-MISMATCH problem and the")
    print("analog direction is sound. If they disagree everywhere, it is a LIBRARY problem.")
    print("=" * 78)


if __name__ == "__main__":
    main()
