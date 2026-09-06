#!/usr/bin/env python3
"""LATTICE Phase 0.A -- the four retrieval arms, over ONE evaluation corpus.

All arms search `eval.sqlite` (built by `lattice_evalcorpus.py`). That is the point: four arms
over four corpora would measure the corpora.

    arm 1  keyed              token -> point get -> edge walk        (undiscriminated)
    arm 1t keyed_typed        the same, with a content-type filter   (the Phase 3.2 target)
    arm 2  keyed_bm25         arm 1 + FTS5 BM25
    arm 3  keyed_bm25_onto    arm 2, reranked by the ontology coordinate
    arm 4  control            Arcade + SSE blind-token + bge-m3      -- see `ControlArm`

Arm 1 is run twice. `lookup_by_lemma` has no type discrimination and the 6M-row `world` collection
shares one `lemmas` index with the 117,659-row lexicon, so `define spaceship` returns Wikipedia
disambiguation pages as dictionary senses -- a live, user-facing wrong-answer defect (contract
5.1.1). `keyed` reproduces that faithfully; `keyed_typed` adds the discriminator. The gap between
them is the measured value of Phase 3.2, which 0.C was scheduled to estimate. The typed variant
alone flatters the arm; the undiscriminated one alone understates what the design proposes.

The ontology coordinate, and why mean-pooled cosine
---------------------------------------------------
`geometry.text_to_signal` returns `[T, D]` -- one row per token, rather than a document vector.
LATTICE 2.2 specifies storage as "a `D x float32` BLOB per vertex" searched by "a numpy matmul",
i.e. a single pooled vector per document, so this pools the token rows (mean, then L2-normalize)
and scores by cosine.

That also makes it the right instrument for the centering question. Unit G showed centering is a
provable no-op for `feature_covariance`/`fingerprint`/`e_distance`, which already subtract a
per-document weighted mean, so measuring centering through those reports "no effect" for a reason
unrelated to retrieval. Mean-pooled cosine over stored per-vertex coordinates is where 17.4's
+0.505 -> -0.332 was measured.

WordNet is loaded from a local file, not the live store
-------------------------------------------------------
The ontology driver's `_load_index()` reads 117,659 artifacts through `open_store()`. This module
populates the same module singleton from a local keyset export instead, using the driver's own
`Synset` class and the real geometry code. Nothing is reimplemented and node 71 is not scanned.

Nothing here is a default: not FTS5, not centering, not `oov="skip"`. This is a measuring
instrument; enabling any of those is a separate decision that follows from the table it produces.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_EMBER_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
_MANTLE_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "agience-mantle", "src"))
for _p in (_HERE, _EMBER_SRC, _MANTLE_SRC,
           os.path.join(_MANTLE_SRC, "mantle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402


# ==========================================================================================
# WordNet, loaded offline into the real wn_store singleton
# ==========================================================================================
def load_wn_offline(jsonl_path: str) -> dict:
    """Populate the ontology driver's module singleton from a local keyset export.

    Uses the driver's own `Synset` class and mirrors `_load_index`'s field handling exactly,
    including the three-state IC rule (absent stays None so `has_ic()` can tell absence from a
    stored 0.0). Any deviation here would change every IC-weighted coordinate."""
    from ember.ontology import wn_store as wn

    idx: Dict[str, "wn.Synset"] = {}
    word: Dict[Tuple[str, str], List[str]] = {}
    n_missing_ic = 0
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            aid = a.get("id") or ""
            name = aid[3:] if aid.startswith("wn-") else aid
            if not name:
                continue
            pos = a.get("pos") or (name.rsplit(".", 2)[-2] if name.count(".") >= 2 else wn.NOUN)
            counts = a.get("lemma_counts") or {}
            if not isinstance(counts, dict):
                counts = {}
            counts = {str(k): int(v) for k, v in counts.items()}
            raw = a.get("ic")
            ic_val = None if raw is None else float(raw)
            if ic_val is None:
                n_missing_ic += 1
            node = wn.Synset(name, pos, list(a.get("hypernyms") or []),
                             list(a.get("instance_hypernyms") or []), ic_val, counts)
            idx[name] = node
            for lm in (list(counts.keys()) or list(a.get("lemmas") or [])):
                word.setdefault((str(lm).lower(), pos), []).append(name)
    for names in word.values():
        names.sort()
    wn._INDEX = (idx, word)
    wn._IC_STATS = {"synsets": len(idx), "with_ic": len(idx) - n_missing_ic,
                    "without_ic": n_missing_ic}
    # IC coverage is reported rather than assumed: every coordinate is IC-weighted, and a corpus
    # caught mid-enrichment looks the same as one of genuinely zero-IC synsets.
    return wn.ic_coverage()


# ==========================================================================================
# The evaluation corpus
# ==========================================================================================
CT_FOR_CLASS = {
    # what a type-discriminating keyed lookup should admit, per query class
    "dictionary": ("text/x-wordnet",),
    "paraphrase": ("text/x-wordnet",),
    "topical": ("application/x-concept", "text/x-wordnet", "text/markdown"),
    "code_nav": ("text/x-python-symbol",),
}


class EvalCorpus:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect("file:%s?mode=ro" % path, uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row

    def stats(self) -> dict:
        out = {"docs": self.db.execute("SELECT count(*) FROM doc").fetchone()[0],
               "by_text_source": {}, "by_collection": {}}
        for r in self.db.execute("SELECT text_source, count(*), cast(avg(length(text)) as int) "
                                 "FROM doc GROUP BY 1"):
            out["by_text_source"][r[0]] = {"n": r[1], "avg_len": r[2]}
        for r in self.db.execute("SELECT collection, count(*) FROM doc GROUP BY 1"):
            out["by_collection"][r[0]] = r[1]
        out["lemma_postings"] = self.db.execute("SELECT count(*) FROM lemma").fetchone()[0]
        return out

    def lemma_lookup(self, token: str, limit: int,
                     cts: Optional[Sequence[str]] = None) -> List[str]:
        if cts:
            qm = ",".join("?" * len(cts))
            sql = ("SELECT l.id FROM lemma l JOIN doc d ON d.id = l.id "
                   "WHERE l.lemma = ? AND d.ct IN (%s) LIMIT ?" % qm)
            return [r[0] for r in self.db.execute(sql, (token, *cts, limit))]
        # No ORDER BY. The production `lookup_by_lemma` truncates an unordered index scan at
        # LIMIT, which is why the synset is unreachable at any practical k rather than merely
        # ranked low. An ORDER BY here would repair the defect inside the instrument and hide the
        # thing being measured.
        return [r[0] for r in self.db.execute(
            "SELECT id FROM lemma WHERE lemma = ? LIMIT ?", (token, limit))]

    def callers_of(self, token: str, limit: int) -> List[str]:
        return [r[0] for r in self.db.execute(
            "SELECT id FROM calls WHERE callee = ? LIMIT ?", (token, limit))]

    def text_of(self, ids: Sequence[str]) -> Dict[str, str]:
        if not ids:
            return {}
        qm = ",".join("?" * len(ids))
        return {r[0]: (r[1] or "") for r in self.db.execute(
            "SELECT id, context || ' ' || text FROM doc WHERE id IN (%s)" % qm, list(ids))}


# ==========================================================================================
# Query -> tokens
# ==========================================================================================
import re  # noqa: E402

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
# There is no stop-list. A stop-list inside the A/B harness biases the measurement the arms are
# judged by: it drops `define`, `where`, `calls`, `what`, which are the subject of the dev-copilot
# queries being evaluated. Deciding which words matter is what the arms are competing to do.


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(text)


# ==========================================================================================
# Arm 1 -- keyed
# ==========================================================================================
class KeyedArm:
    """token -> point get -> edge walk. `typed=True` adds the content-type discriminator."""

    def __init__(self, corpus: EvalCorpus, typed: bool = False):
        self.corpus = corpus
        self.typed = typed
        self.name = "keyed_typed" if typed else "keyed"

    def available(self) -> Tuple[bool, str]:
        try:
            self.corpus.lemma_lookup("dog", 1)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, "keyed lookup failed: %s" % e

    def retrieve(self, text: str, k: int, cls: Optional[str] = None) -> List[str]:
        cts = CT_FOR_CLASS.get(cls or "") if self.typed else None
        out: List[str] = []
        seen = set()
        toks = tokens(text)
        # Call-site queries are an edge walk rather than a lemma lookup: `what calls X` asks the
        # `calls` edge, the same primitive as a hypernym walk (2 3.1: code navigation and
        # dictionary lookup unify once a symbol is a token).
        if cls == "code_nav" and "calls" in text.lower():
            for t in toks:
                for aid in self.corpus.callers_of(t.lower(), k):
                    if aid not in seen:
                        seen.add(aid)
                        out.append(aid)
                if len(out) >= k:
                    return out[:k]
        for t in toks:
            for aid in self.corpus.lemma_lookup(t.lower(), k, cts):
                if aid not in seen:
                    seen.add(aid)
                    out.append(aid)
            if len(out) >= k:
                break
        return out[:k]


# ==========================================================================================
# Arm 2 -- keyed + BM25 (FTS5)
# ==========================================================================================
class Bm25Arm:
    def __init__(self, corpus: EvalCorpus, fts_db: str, keyed: KeyedArm):
        from ember.corpus.fts import FtsIndex
        self.corpus = corpus
        self.keyed = keyed
        self.name = "keyed_bm25"
        self._err: Optional[str] = None
        self._idx = None
        if not Path(fts_db).exists():
            self._err = ("FTS5 index %s does not exist. Nothing in the tree builds one -- "
                         "`fts.py` is imported only by its own test. Run "
                         "`lattice_fts_build.py` first." % fts_db)
            return
        conn = sqlite3.connect("file:%s?mode=ro" % fts_db, uri=True, check_same_thread=False)
        # resolver=None: recall@k needs ranked ids rather than spans. With a resolver the
        # 7,000-char total budget truncates the hit list, which is right for Lumen and wrong here.
        self._idx = FtsIndex(conn, resolver=None)

    def available(self) -> Tuple[bool, str]:
        return (False, self._err) if self._err else (True, "")

    def retrieve(self, text: str, k: int, cls: Optional[str] = None) -> List[str]:
        out: List[str] = []
        seen = set()
        for aid in self.keyed.retrieve(text, k, cls):
            if aid not in seen:
                seen.add(aid)
                out.append(aid)
        try:
            for h in self._idx.search(text, limit=k):
                if h.vertex_id not in seen:
                    seen.add(h.vertex_id)
                    out.append(h.vertex_id)
        except Exception:  # noqa: BLE001
            return out[:k]
        return out[:k]


# ==========================================================================================
# Arm 3 -- keyed + BM25 + ontology coordinate
# ==========================================================================================
class OntologyArm:
    """Reranks arm 2's candidate pool with the computed Jiang-Conrath coordinate.

    The coordinate is computed rather than trained: GENESIS 12 permits vectors that are computed
    and forbids vectors that are trained, which is why bge-m3 is the incumbent being replaced.

    Reranking rather than exhaustive search is a stated limitation. Per-vertex coordinate storage
    is Phase 2.2 work, so an exhaustive matmul would mean embedding the corpus on the fly.
    `rerank_pool` is recorded in the output, so the number reads as a rerank rather than a full
    semantic search."""

    def __init__(self, corpus: EvalCorpus, base: Bm25Arm, pool: int = 100,
                 center: Optional[np.ndarray] = None, label: str = "",
                 keyed_first: bool = False,
                 cache: Optional[Dict[str, Optional[np.ndarray]]] = None):
        from ember.ontology import geometry
        self.corpus = corpus
        self.base = base
        self.pool = pool
        self.geo = geometry
        self.center = center
        # `keyed_first` separates two questions that a single stacked arm conflates: "is the
        # coordinate a bad ranker?" and "is naive stacking a bad combiner?". With it set, exact
        # keyed hits keep their positions and the coordinate reranks only the lexical tail below
        # them, so a drop against arm 2 is attributable to the coordinate itself rather than to
        # the coordinate overwriting exact matches.
        self.keyed_first = keyed_first
        self.name = "keyed_bm25_onto" + (label or "")
        # The document-coordinate cache may be shared between arms that use the same centering. A
        # centered and an uncentered vector have identical shape and will matmul happily while
        # meaning nothing (Unit G's `centering_id` exists for this), so the caller keeps one cache
        # per centering.
        self._cache: Dict[str, Optional[np.ndarray]] = cache if cache is not None else {}
        self._err: Optional[str] = None
        if base is None or not base.available()[0]:
            self._err = "arm 2 unavailable: %s" % (base.available()[1] if base else "not built")

    def available(self) -> Tuple[bool, str]:
        return (False, self._err) if self._err else (True, "")

    def _vec(self, text: str) -> Optional[np.ndarray]:
        """Mean-pooled, L2-normalized document coordinate -- the `D x float32` of 2.2."""
        rows = self.geo.text_to_signal(text[:4000], center=self.center)
        if rows.shape[0] == 0:
            return None                      # no noun carried a coordinate; there is nothing to read
        v = rows.mean(axis=0)
        n = float(np.linalg.norm(v))
        return (v / n) if n else None

    def retrieve(self, text: str, k: int, cls: Optional[str] = None) -> List[str]:
        cands = self.base.retrieve(text, self.pool, cls)
        if not cands:
            return []
        head: List[str] = []
        if self.keyed_first:
            head = self.base.keyed.retrieve(text, k, cls)
            hs = set(head)
            cands = [c for c in cands if c not in hs]
            if len(head) >= k:
                return head[:k]
        qv = self._vec(text)
        if qv is None:
            return (head + cands)[:k]        # no coordinate for the query -> keep arm 2's order
        bodies = self.corpus.text_of(cands)
        scored: List[Tuple[float, int, str]] = []
        for i, aid in enumerate(cands):
            if aid not in self._cache:
                self._cache[aid] = self._vec(bodies.get(aid, ""))
            dv = self._cache[aid]
            # A document with no coordinate keeps its arm-2 rank rather than being scored 0 and
            # sinking: a missing coordinate is an absence of evidence (adjectives have no hypernym
            # tree at all), which is a different fact from evidence of irrelevance.
            scored.append((float(qv @ dv) if dv is not None else -2.0, -i, aid))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return (head + [aid for _s, _i, aid in scored])[:k]


# ==========================================================================================
# Arm 4 -- the control
# ==========================================================================================
class ControlArm:
    """Arcade + SSE blind-token index + bge-m3 (Prism).

    This is driven rather than simulated: a reimplementation measures our model of the control,
    which is how every retrieval claim in the design became prose. When it cannot be driven the
    answer is `not_run` with the reason, rather than a zero — a control scored 0 would make every
    other arm look like a win."""

    name = "control"

    def __init__(self, url: Optional[str], token: Optional[str] = None):
        self.url = url
        self.token = token or os.getenv("LATTICE_CONTROL_TOKEN")
        self._err: Optional[str] = None
        if not url:
            self._err = ("no --control-url given, and the control cannot be simulated. "
                         "MEASURED 2026-07-20: `prism.agience.ai` does not resolve (NXDOMAIN), "
                         "no prism/embedding process runs on the MANTLE box, no SSE index or "
                         "indexer is deployed there, and the only search route the deployed "
                         "mantle-api exposes (`GET /api/search`) is a SQL `LIKE '%q%'` substring "
                         "scan over id+context -- not BM25, not blind-token SSE, not bge-m3. "
                         "The Arcade+SSE+bge-m3 control has been DECOMMISSIONED.")
            return
        ok, why = self._probe()
        if not ok:
            self._err = why

    def _probe(self) -> Tuple[bool, str]:
        import urllib.request
        try:
            req = urllib.request.Request(self.url, method="GET")
            if self.token:
                req.add_header("Authorization", "Bearer %s" % self.token)
            urllib.request.urlopen(req, timeout=20).read(1)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, "control endpoint unreachable: %s" % str(e)[:200]

    def available(self) -> Tuple[bool, str]:
        return (False, self._err) if self._err else (True, "")

    def retrieve(self, text: str, k: int, cls: Optional[str] = None) -> List[str]:
        import urllib.parse
        import urllib.request
        q = urllib.parse.urlencode({"q": text, "limit": k})
        req = urllib.request.Request("%s?%s" % (self.url, q), method="GET")
        if self.token:
            req.add_header("Authorization", "Bearer %s" % self.token)
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception:  # noqa: BLE001
            return []
        rows = data.get("results") or data.get("hits") or []
        return [str(r.get("id")) for r in rows[:k] if isinstance(r, dict) and r.get("id")]
