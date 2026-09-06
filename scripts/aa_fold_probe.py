"""Unit AA — is the `oov="skip"` noise-floor explosion a property of Jiang-Conrath
coordinates, or an artifact of entering entroptics through the wrong door?

§5.3 of LATTICE-CONTRACT records a hard prerequisite ("a channel-liveness screen is needed
before oov='skip'") on the strength of one measurement: with pure sparse JC coordinates the
entroptics noise floor moves from ~1.5 to 1.13e9. That measurement went through
`entroptics.read()` -> `screen.Screen`, whose fold decision is the entropy guard
(screen.py:619, `fold = H_F < log2F - band_F`). The library's own locality criterion
(batch.py:280-306) states that the entropy guard folds a sparse carrier and that averaging
destroys it. This script measures both paths on the same real frames.

No trained models. nltk is used only to rebuild, offline, the WordNet structure and Resnik IC that
`scripts/enrich_wordnet.py` writes into the store — same source, same values — so the probe runs
without touching the live store. The production runtime stays nltk-free.

Run:  python scripts/aa_fold_probe.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The sample is an exported data file, not part of the checkout, so it cannot be derived. Name
# it in AA_FOLD_CONTENT; unset is refused rather than defaulted, because a probe that silently
# reads nothing still prints numbers.
_content_env = (os.environ.get("AA_FOLD_CONTENT") or "").strip()
if not _content_env:
    raise SystemExit(
        "AA_FOLD_CONTENT is unset. Point it at the exported content sample "
        "(content_sample.jsonl) this probe folds."
    )
CONTENT = Path(_content_env)
DIMS = (256, 512, 2048)
SEED = 20260720


# ─────────────────────────────────────────────────────────────────────────────
# 1. Offline WordNet index — the same fields enrich_wordnet.py wrote to the store
# ─────────────────────────────────────────────────────────────────────────────
def install_offline_wordnet() -> dict:
    """Populate the ontology driver's module singleton from the local nltk WordNet and ic-brown,
    the same source `enrich_wordnet.py` draws from. Keeps the probe off the live store."""
    from nltk.corpus import wordnet as nwn, wordnet_ic
    from nltk.corpus.reader.wordnet import information_content

    from ember.ontology import wn_store as wn

    icd = wordnet_ic.ic("ic-brown.dat")
    idx: dict = {}
    word: dict = {}
    n_missing = 0
    for s in nwn.all_synsets():
        pos = s.pos()
        try:
            v = information_content(s, icd)
            v = float(v) if math.isfinite(v) else None
        except Exception:
            v = None
        counts = {l.name(): int(l.count()) for l in s.lemmas()}
        node = wn.Synset(s.name(), pos,
                         [h.name() for h in s.hypernyms()],
                         [h.name() for h in s.instance_hypernyms()],
                         v, counts)
        if v is None:
            n_missing += 1
        idx[s.name()] = node
        for lm in counts:
            word.setdefault((lm.lower(), pos), []).append(s.name())
    for names in word.values():
        names.sort()
    wn._INDEX = (idx, word)
    wn._IC_STATS = {"synsets": len(idx), "with_ic": len(idx) - n_missing,
                    "without_ic": n_missing}
    return wn._IC_STATS


# ─────────────────────────────────────────────────────────────────────────────
# 2. Real documents
# ─────────────────────────────────────────────────────────────────────────────
def load_docs(n: int, *, min_chars: int = 200) -> list[str]:
    out: list[str] = []
    with CONTENT.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = (d.get("text") or "").strip()
            if len(t) >= min_chars:
                out.append(t)
            if len(out) >= n:
                break
    return out


def pooled_frame(docs, D: int, oov: str) -> np.ndarray:
    """The ordered pooled screen: each document's token rows in order, documents concatenated in a
    fixed order. The Screen is ordered — row order is load-bearing, coherence flips sign on it — so
    this is a deterministic concatenation rather than a set.

    There is no token cap, matching `geometry.text_to_signal`. `T` is the sample count the band is
    computed from, so a probe measuring the noise floor measures it over the whole frame."""
    from ember.ontology import geometry as g
    parts = [g.text_to_signal(t, D=D, oov=oov) for t in docs]
    parts = [p for p in parts if p.shape[0]]
    return np.vstack(parts) if parts else np.zeros((0, D))


# ─────────────────────────────────────────────────────────────────────────────
# 3. The reads
# ─────────────────────────────────────────────────────────────────────────────
def fold_diagnostics(W: np.ndarray) -> dict:
    """What each fold criterion decides on this frame, and the resulting width."""
    from entroptics import screen as _screen
    from entroptics.batch import _foldable
    from entroptics import environment as _env

    T, F = W.shape
    F_eff = int(_screen._fold_target_batch(np, W[None])[0])       # entropy guard (screen.py:619)
    P = np.abs(W) ** 2
    marg = P.sum(axis=0)
    tot = marg.sum()
    p = marg / max(tot, 1e-30)
    H_F = float(-(np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1.0)), 0.0)).sum())
    log2F = math.log2(F)
    band_F = min((F - 1) / (2.0 * max(1, T) * math.log(2.0)), 0.5 * log2F)
    whit = _screen._normalize_batch(np, W[None])
    loc = bool(_foldable(np, whit, T=T)[0])                       # locality guard (batch.py:280)
    occ = float((np.abs(W) > 0).any(axis=0).mean())               # channel occupancy
    nnz = float((np.abs(W) > 0).sum(axis=1).mean()) if T else 0.0
    return {"T": T, "F": F, "H_F": H_F, "log2F": log2F, "band_F": band_F,
            "entropy_folds": bool(H_F < log2F - band_F), "F_eff_entropy": F_eff,
            "locality_folds": loc, "occupancy": occ, "nnz_per_row": nnz}


def reads(W: np.ndarray) -> dict:
    """Every read path, on identical input."""
    import entroptics
    from entroptics import Aperture
    from entroptics.batch import resolved_batch

    out: dict = {}

    # (A) Unit G's path: entroptics.read() -> Screen. Entropy fold.
    r = entroptics.read(W)
    out["A_read"] = {"K": int(r.K_signal), "floor": float(r.noise_floor),
                     "sigma_top": float(r.sigma_top)}

    # (B) The front door, whole window. Aperture.screen() is still a Screen -> entropy fold.
    ap = Aperture(W, window=None)
    sc = ap.screen()
    out["B_instrument_screen"] = {"K": int(sc.K_signal), "floor": float(sc.noise_floor),
                                "sigma_top": float(sc.sigma_top)}

    # (C) The locality auto-fold — the library's default and its stated criterion.
    for tag, fold in (("C_auto", "auto"), ("D_native", False), ("E_forced_fold", True)):
        rb = resolved_batch(W[None], fold=fold)
        out[tag] = {"K": int(np.asarray(rb.K_signal).ravel()[0]),
                    "floor": float(np.asarray(rb.noise_floor).ravel()[0]),
                    "sigma_top": float(np.asarray(rb.sigma_top).ravel()[0])}

    # (F) The instrument's spectral read: correlation eigenspectrum + Tracy-Widom edge. No fold.
    sp = ap.spectral
    out["F_spectral"] = {"K": int(sp.resolved_modes), "floor": float(sp.noise_floor),
                         "contrast": float(sp.contrast)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. §17.3's named-anchor basis, rebuilt for the same panel
# ─────────────────────────────────────────────────────────────────────────────
def anchor_frame(docs, anchors, ic, max_tokens: int = 400) -> np.ndarray:
    """§17.3's proposal: coordinate_k(s) = IC(LCS(s, a_k)) over named anchor synsets.
    Sparse and near-constant by construction."""
    from ember.ontology import geometry as g
    from ember.ontology import wn_store as wn

    A = len(anchors)
    cache: dict = {}
    rows = []
    for text in docs:
        n = 0
        for m in g._WORD.finditer(text):
            tok = m.group(0).lower()
            n += 1
            if n > max_tokens:
                break
            senses = wn.synsets(tok, pos=wn.NOUN)
            if not senses:
                continue
            s = senses[0]
            v = cache.get(s.name())
            if v is None:
                v = np.array([g.tree_lcs_ic(s, a, ic) for a in anchors], dtype=np.float64)
                cache[s.name()] = v
            rows.append(v)
    return np.vstack(rows) if rows else np.zeros((0, A))


def pick_anchors(n: int, ic):
    """An antichain-free, deterministic named-anchor set: the n highest-IC noun synsets that
    appear as a canonical parent. Reproduces §17.3's 'named anchors' shape, not its exact set."""
    from ember.ontology import geometry as g
    from ember.ontology import wn_store as wn

    idx = wn._INDEX[0]
    cands = [s for s in idx.values() if s.pos() == wn.NOUN and s.hypernyms()]
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(cands), size=min(4000, len(cands)), replace=False)
    pool = [cands[i] for i in sel]
    pool.sort(key=lambda s: (-g.ic_of(s, ic), s.name()))
    return pool[:n]


# ─────────────────────────────────────────────────────────────────────────────
def fmt(v: float) -> str:
    return f"{v:.4g}"


def main() -> None:
    stats = install_offline_wordnet()
    print(f"[wordnet] {stats}")

    from ember.ontology import geometry as g
    ic = g.load_ic()

    for n_docs in (4, 32):
        docs = load_docs(n_docs)
        print(f"\n{'='*78}\n### {n_docs} real documents from content_sample.jsonl\n{'='*78}")
        for D in DIMS:
            for oov in ("surface", "skip"):
                W = pooled_frame(docs, D, oov)
                if W.shape[0] < 4:
                    print(f"D={D:5d} oov={oov:8s}  (too few rows: {W.shape})")
                    continue
                fd = fold_diagnostics(W)
                rd = reads(W)
                print(f"\nD={D:<5d} oov={oov:8s} shape={W.shape} "
                      f"occ={fd['occupancy']:.3f} nnz/row={fd['nnz_per_row']:.1f}")
                print(f"  fold: H_F={fd['H_F']:.3f} log2F={fd['log2F']:.3f} "
                      f"band={fd['band_F']:.3f} -> entropy_folds={fd['entropy_folds']} "
                      f"(F_eff={fd['F_eff_entropy']}) locality_folds={fd['locality_folds']}")
                for k in ("A_read", "B_instrument_screen", "C_auto", "D_native",
                          "E_forced_fold", "F_spectral"):
                    v = rd[k]
                    extra = f" sigma_top={fmt(v['sigma_top'])}" if "sigma_top" in v else \
                            f" contrast={fmt(v['contrast'])}"
                    print(f"    {k:20s} K={v['K']:4d} floor={fmt(v['floor']):>12s}{extra}")

    # ── §17.3 re-examination ────────────────────────────────────────────────
    print(f"\n{'='*78}\n### §17.3 named-anchor basis, same panel\n{'='*78}")
    docs = load_docs(32)
    for A in (256,):
        anchors = pick_anchors(A, ic)
        W = anchor_frame(docs, anchors, ic)
        if W.shape[0] < 4:
            print(f"anchors={A}: too few rows {W.shape}")
            continue
        fd = fold_diagnostics(W)
        rd = reads(W)
        print(f"\nanchors={A} shape={W.shape} occ={fd['occupancy']:.3f} "
              f"nnz/row={fd['nnz_per_row']:.1f}")
        print(f"  fold: H_F={fd['H_F']:.3f} log2F={fd['log2F']:.3f} band={fd['band_F']:.3f} "
              f"-> entropy_folds={fd['entropy_folds']} (F_eff={fd['F_eff_entropy']}) "
              f"locality_folds={fd['locality_folds']}")
        for k in ("A_read", "B_instrument_screen", "C_auto", "D_native",
                  "E_forced_fold", "F_spectral"):
            v = rd[k]
            print(f"    {k:20s} K={v['K']:4d} floor={fmt(v['floor']):>12s}")


if __name__ == "__main__":
    main()
