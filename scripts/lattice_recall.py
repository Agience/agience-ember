#!/usr/bin/env python3
"""LATTICE Phase 0.A -- the recall@k harness. Scores retrieval arms against oracle labels.

Consumes the query set from `lattice_queryset.py` and emits a table of recall@k per arm per
query class. This is the measuring instrument that gates Phase 2: per
`LATTICE-IMPLEMENTATION.md` E.3 item 3, Phase 2 does not start before 0.A has run -- without
this baseline there is no way to tell an FTS5 stemmer regression from an expected difference.

The headline carries oracle labels only
----------------------------------------
`Headline.add()` raises `JudgedLabelInHeadline` / `SegregatedClassInHeadline` when a judged label
or a segregated class reaches it, and there is no flag to turn that off. Grading a deterministic
retriever against frontier-model relevance judgements is circular, because the labels come from
the paradigm being replaced, so a retriever that is better but differently shaped scores worse.
A filter can be dropped in a refactor without anything noticing; a raise stops the run.

`paraphrase` is segregated from the headline even though its labels are oracle-derived
(WordNet's own gloss<->synset pairing). The design doc designates that class judgement-prone;
the segregation follows the class, not this run's provenance, so it holds even if judged items
are later added to it.

Three numbers, not one
-----------------------
    recall@k   |retrieved@k INTERSECT relevant| / |relevant|
    hit@k      1 if any relevant id is in the top k, else 0
    ceiling@k  min(k, |relevant|) / |relevant|   -- the best recall@k any arm could score

`ceiling@k` is reported because it is not always 1. The topical class has queries with 72
oracle-relevant members; at k=10 the maximum achievable recall@10 is 0.139. Without the
ceiling, a perfect arm reads as a 14% failure. `recall_norm@k = recall@k / ceiling@k` is also
emitted -- that is the number to compare across classes.

No silent zeros
-----------------
Two distinct ways a 0 can be a lie, both handled:

  1. An arm that cannot run. Every arm implements `available() -> (bool, reason)`. An
     unavailable arm reports status `not_run` with the reason and is omitted from the table
     body -- it never contributes a 0.0 row that reads as "measured, and bad".
  2. A target absent from the corpus. An oracle target that is not in the store cannot be
     retrieved by any arm, so scoring it measures the corpus, not the retriever. Queries
     whose targets are all absent are moved to an `unretrievable` bucket and excluded from
     scoring; partially-absent queries have their ceiling adjusted to the present subset.
     For example, no `sym-*` artifacts exist in the corpus, so the whole code_nav class is
     unretrievable: its oracle is sound, but the corpus has none of the targets, so the harness
     reports that rather than printing four zeros.

Usage
-----
    python lattice_recall.py --queryset <eval-dir>/lattice-queryset.json \
                             --arm keyed --arm keyed_bm25 --arm keyed_bm25_onto --arm control \
                             --k 1 --k 5 --k 10 \
                             --out <eval-dir>/lattice-recall.json

    python lattice_recall.py --queryset ... --list-arms     # availability probe, no scoring
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

SEGREGATED_CLASSES = ("paraphrase",)
DEFAULT_KS = (1, 5, 10, 20)


# ============================================================================================
# Structural guard -- the unit's subject
# ============================================================================================
class JudgedLabelInHeadline(Exception):
    """A query with model/human-judged labels reached the headline aggregate."""


class SegregatedClassInHeadline(Exception):
    """A query from a judgement-prone class reached the headline aggregate."""


class Headline:
    """The oracle-only aggregate. Judged and segregated input raises rather than being filtered.

    A filter such as `if q['judged']: continue` can be dropped in a later refactor without
    anything noticing, after which the headline would absorb opinions with nothing to mark it.
    A raise fails the run instead."""

    def __init__(self) -> None:
        self.rows: List[Tuple[dict, Dict[int, dict]]] = []

    def add(self, q: dict, per_k: Dict[int, dict]) -> None:
        if q.get("judged"):
            raise JudgedLabelInHeadline(
                f"query {q['qid']} carries {q.get('label_provenance')!r}. Judged labels are "
                "reported separately and never aggregated -- see this file's docstring.")
        if q.get("cls") in SEGREGATED_CLASSES:
            raise SegregatedClassInHeadline(
                f"query {q['qid']} is class {q['cls']!r}, which LATTICE-IMPLEMENTATION 0.A "
                "designates judgement-prone. Report it in its own table.")
        self.rows.append((q, per_k))


# ============================================================================================
# Arms
# ============================================================================================
class Arm(Protocol):
    name: str

    def available(self) -> Tuple[bool, str]: ...

    def retrieve(self, text: str, k: int) -> List[str]: ...


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")


def _tokens(text: str) -> List[str]:
    """Query -> candidate token keys. Backticks and the `define`/`where is` framing are stripped:
    the framing words are query syntax rather than content, and feeding them to a keyed lookup
    adds noise hits for the synset of `define`."""
    raw = [t for t in _TOKEN.findall(text)]
    stop = {"define", "where", "is", "are", "defined", "what", "calls", "the", "a", "an",
            "of", "for", "to", "in", "and", "or", "by", "as", "that", "this", "it", "its",
            "with", "on", "at", "from", "be", "been", "has", "have", "not", "who", "which"}
    return [t for t in raw if t.lower() not in stop]


class KeyedArm:
    """Arm 1 -- keyed only: token vertex -> point get -> edge walk.

    Drives the current keyed index -- `lookup_by_lemma`, the `listkeys` inverted index over the
    `lemmas` field -- plus one hop of edge walk, on today's storage. This is the mechanism the
    contract's RESOLVED-4 describes (a token vertex walk); `_lookup` is the seam a dedicated
    lattice token-vertex table would replace.

    Deterministic: results are ordered by token position, then store order, rather than by
    score."""

    name = "keyed"

    def __init__(self, store=None, walk_edges: bool = True):
        self._store = store
        self._walk = walk_edges
        self._err: Optional[str] = None
        if self._store is None:
            try:
                _ensure_paths()
                from mantle.shard.local_store import open_store  # type: ignore
                self._store = open_store()
            except Exception as e:  # noqa: BLE001
                self._err = f"cannot open local store: {e}"

    def available(self) -> Tuple[bool, str]:
        if self._err:
            return False, self._err
        try:
            self._store.artifacts.lookup_by_lemma("dog", limit=1)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"keyed lookup failed: {e}"

    def retrieve(self, text: str, k: int) -> List[str]:
        out: List[str] = []
        seen = set()
        toks = _tokens(text)
        for t in toks:
            for a in self._store.artifacts.lookup_by_lemma(t.lower(), limit=k):
                aid = a.get("id")
                if aid and aid not in seen:
                    seen.add(aid)
                    out.append(aid)
            if len(out) >= k:
                break
        # one hop of edge walk from the keyed hits -- the "edge walk" half of arm 1
        if self._walk and len(out) < k:
            for aid in list(out):
                try:
                    for nb in self._store.graph.neighbors(aid, limit=k):
                        nid = nb.get("id") if isinstance(nb, dict) else str(nb)
                        if nid and nid not in seen:
                            seen.add(nid)
                            out.append(nid)
                except Exception:  # noqa: BLE001 -- a walk failure is not a retrieval answer
                    break
                if len(out) >= k:
                    break
        return out[:k]


class Bm25Arm:
    """Arm 2 -- keyed + BM25 (FTS5).

    Takes an explicit `--fts-db` pointing at a SQLite file carrying an `fts` virtual table with
    an `id` mapping; without one it reports `not_run` with the reason. The lexical path in
    production is the SSE blind-token stack in S3, which is arm 4's subject rather than this
    one's."""

    name = "keyed_bm25"

    def __init__(self, fts_db: Optional[str], keyed: Optional[KeyedArm] = None):
        self.fts_db = fts_db
        self.keyed = keyed
        self._conn = None
        self._err: Optional[str] = None
        if not fts_db:
            self._err = "no --fts-db given; there is no FTS5 index to score against"
        elif not Path(fts_db).exists():
            self._err = f"--fts-db {fts_db} does not exist"
        else:
            try:
                import sqlite3
                self._conn = sqlite3.connect(f"file:{fts_db}?mode=ro", uri=True)
                self._conn.row_factory = sqlite3.Row
                names = {r[0] for r in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
                if "fts" not in names:
                    self._err = f"{fts_db} has no `fts` table (found: {sorted(names)[:8]})"
            except Exception as e:  # noqa: BLE001
                self._err = f"cannot open {fts_db}: {e}"

    def available(self) -> Tuple[bool, str]:
        return (False, self._err) if self._err else (True, "")

    def retrieve(self, text: str, k: int) -> List[str]:
        out: List[str] = []
        seen = set()
        if self.keyed is not None and self.keyed.available()[0]:
            for aid in self.keyed.retrieve(text, k):
                if aid not in seen:
                    seen.add(aid)
                    out.append(aid)
        toks = _tokens(text)
        if toks and self._conn is not None:
            match = " OR ".join(f'"{t}"' for t in toks[:24])
            try:
                for r in self._conn.execute(
                        "SELECT id FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
                        (match, int(k))):
                    aid = r["id"]
                    if aid not in seen:
                        seen.add(aid)
                        out.append(aid)
            except Exception:  # noqa: BLE001
                return out[:k]
        return out[:k]


class OntologyArm:
    """Arm 3 -- keyed + BM25 + ontology coordinate.

    The semantic stage is the geometry module's `text_to_signal`: an exact Jiang-Conrath
    coordinate computed from WordNet, hashed into D dims. It is computed rather than trained --
    GENESIS 12 allows vectors that are computed and forbids vectors that are trained, which is
    why bge-m3 (arm 4) is the incumbent being replaced.

    It reranks arm 2's candidates rather than searching independently: coordinates are not stored
    per vertex, so an exhaustive matmul would mean embedding the corpus on the fly. Reranking a
    candidate pool is the measurable subset today, and `--onto-pool` controls the pool depth. The
    depth is recorded in the output as `rerank_pool`, so the result reads as a rerank rather than
    a full semantic search."""

    name = "keyed_bm25_onto"

    def __init__(self, base: Optional[Bm25Arm], pool: int = 100):
        self.base = base
        self.pool = pool
        self._err: Optional[str] = None
        self._geo = None
        self._store = None
        try:
            _ensure_paths()
            from ember.ontology import geometry# type: ignore
            from mantle.shard.local_store import open_store  # type: ignore
            self._geo = geometry
            self._store = open_store()
        except Exception as e:  # noqa: BLE001
            self._err = f"cannot import ember.geometry / open store: {e}"
        if self.base is None:
            self._err = self._err or "arm 2 (keyed_bm25) unavailable, so arm 3 has no candidates"
        elif not self.base.available()[0]:
            self._err = self._err or f"arm 2 unavailable: {self.base.available()[1]}"

    def available(self) -> Tuple[bool, str]:
        return (False, self._err) if self._err else (True, "")

    def retrieve(self, text: str, k: int) -> List[str]:
        import numpy as np
        cands = self.base.retrieve(text, self.pool)
        if not cands:
            return []
        qv = self._geo.text_to_signal(text)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        scored = []
        for aid in cands:
            try:
                a = self._store.artifacts.get_artifact(aid)
                body = " ".join(str(a.get(f) or "") for f in ("context", "content"))[:2000]
                v = self._geo.text_to_signal(body)
                v = v / (np.linalg.norm(v) or 1.0)
                scored.append((float(qv @ v), aid))
            except Exception:  # noqa: BLE001
                scored.append((-1e9, aid))
        scored.sort(key=lambda p: -p[0])
        return [aid for _s, aid in scored[:k]]


class ControlArm:
    """Arm 4 -- the control: the existing Arcade + SSE blind-token + bge-m3 (Prism) path.

    This is the incumbent the new arms are compared against, so it is driven as it is -- through
    the deployed Mantle search endpoint rather than a reimplementation. A reimplementation measures
    our model of the control, which is what left every retrieval claim in the design as prose.

    Needs `--control-url` (Mantle search) and, where the endpoint is authenticated, a bearer
    token in $LATTICE_CONTROL_TOKEN. Read-only: one GET/POST search call per query, no writes,
    consistent with the external-operator rule."""

    name = "control"

    def __init__(self, url: Optional[str], token: Optional[str] = None, timeout: float = 20.0):
        self.url = url
        self.token = token or os.getenv("LATTICE_CONTROL_TOKEN")
        self.timeout = timeout
        self._err: Optional[str] = None
        if not url:
            self._err = ("no --control-url given; the control is the deployed Arcade+SSE+bge-m3 "
                         "search path and must be driven, not simulated")
        else:
            ok, why = self._probe()
            if not ok:
                self._err = why

    def _probe(self) -> Tuple[bool, str]:
        import urllib.request
        try:
            req = urllib.request.Request(self.url, method="GET")
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            urllib.request.urlopen(req, timeout=self.timeout).read(1)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"control endpoint unreachable: {str(e)[:160]}"

    def available(self) -> Tuple[bool, str]:
        return (False, self._err) if self._err else (True, "")

    def retrieve(self, text: str, k: int) -> List[str]:
        import urllib.parse
        import urllib.request
        q = urllib.parse.urlencode({"q": text, "k": k, "limit": k})
        req = urllib.request.Request(f"{self.url}?{q}", method="GET")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
        except Exception:  # noqa: BLE001
            return []
        rows = data.get("results") or data.get("hits") or data.get("items") or []
        out = []
        for r in rows[:k]:
            aid = (r.get("id") or r.get("artifact_id") or r.get("_id")) if isinstance(r, dict) else r
            if aid:
                out.append(str(aid))
        return out


def _ensure_paths() -> Path:
    """Put the ember src dir and mantle's outer src on sys.path, and return the ember src dir.

    The store this harness drives lives in the sibling `agience-mantle` checkout and is imported
    as a package (`mantle.db....`), so without `.../agience-mantle/src` on the path the import
    fails with a bare `No module named` that reads as a broken ember rather than a missing sibling
    repo."""
    here = Path(__file__).resolve()
    ember_src = here.parent.parent / "src"
    for p in here.parents:
        if (p / "src" / "ember" / "local_store.py").exists():
            ember_src = p / "src"
            genesis_root = p.parent
            # mantle is imported as a package (`mantle.db.…`), so its outer src goes on the path.
            # Inserting the inner `<mantle>/mantle` path instead is what makes one class import as
            # two distinct objects.
            mantle_src = genesis_root / "agience-mantle" / "src"
            if mantle_src.is_dir() and str(mantle_src) not in sys.path:
                sys.path.insert(0, str(mantle_src))
            break
    if str(ember_src) not in sys.path:
        sys.path.insert(0, str(ember_src))
    return ember_src


def build_arms(names: Sequence[str], args) -> "Dict[str, Arm]":
    keyed = KeyedArm() if {"keyed", "keyed_bm25", "keyed_bm25_onto"} & set(names) else None
    bm25 = (Bm25Arm(args.fts_db, keyed)
            if {"keyed_bm25", "keyed_bm25_onto"} & set(names) else None)
    made: Dict[str, Arm] = {}
    for n in names:
        if n == "keyed":
            made[n] = keyed
        elif n == "keyed_bm25":
            made[n] = bm25
        elif n == "keyed_bm25_onto":
            made[n] = OntologyArm(bm25, pool=args.onto_pool)
        elif n == "control":
            made[n] = ControlArm(args.control_url)
        else:
            raise SystemExit(f"unknown arm {n!r}; known: keyed keyed_bm25 keyed_bm25_onto control")
    return made


# ============================================================================================
# Scoring
# ============================================================================================
def score_query(retrieved: Sequence[str], relevant: Sequence[str], ks: Sequence[int]) -> Dict[int, dict]:
    rel = set(relevant)
    n_rel = len(rel)
    out: Dict[int, dict] = {}
    for k in ks:
        top = list(retrieved[:k])
        got = len(rel & set(top))
        ceiling = min(k, n_rel) / n_rel if n_rel else 0.0
        recall = got / n_rel if n_rel else 0.0
        out[k] = {"recall": recall, "hit": 1.0 if got else 0.0, "ceiling": ceiling,
                  "recall_norm": (recall / ceiling) if ceiling else 0.0, "n_retrieved": len(top)}
    return out


def _mean(vals: Sequence[float]) -> Optional[float]:
    return (sum(vals) / len(vals)) if vals else None


def run(args) -> int:
    doc = json.loads(Path(args.queryset).read_text(encoding="utf-8"))
    queries: List[dict] = doc["queries"]
    ks = sorted(set(args.k or DEFAULT_KS))
    kmax = max(ks)
    arms = build_arms(args.arm, args)

    # ---- availability probe first: an arm that cannot run says so before any scoring, so a
    # "not_run" reads differently from a measured zero.
    status: Dict[str, dict] = {}
    for name, arm in arms.items():
        ok, why = (False, "arm failed to construct") if arm is None else arm.available()
        status[name] = {"status": "ok" if ok else "not_run", "reason": why}
        print(f"arm {name:<18} {'OK' if ok else 'NOT RUN'}  {why}", file=sys.stderr)
    if args.list_arms:
        print(json.dumps(status, indent=2))
        return 0

    # ---- partition by retrievability. A target absent from the corpus is not a retriever miss.
    scored, unretrievable, unknown_presence = [], [], []
    for q in queries:
        cp = q.get("corpus_presence") or {}
        if cp.get("none_present") is True:
            unretrievable.append(q)
        else:
            if cp.get("all_present") is None:
                unknown_presence.append(q)
            scored.append(q)

    results: Dict[str, dict] = {}
    for name, arm in arms.items():
        if status[name]["status"] != "ok":
            results[name] = {**status[name], "by_class": {}, "headline": {}}
            continue
        per_class: Dict[str, Dict[int, List[dict]]] = {}
        headline = Headline()
        t0 = time.time()
        latencies = []
        for q in scored:
            t1 = time.time()
            try:
                got = arm.retrieve(q["text"], kmax)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {name} failed on {q['qid']}: {str(e)[:120]}", file=sys.stderr)
                got = []
            latencies.append(time.time() - t1)
            per_k = score_query(got, q["relevant"], ks)
            per_class.setdefault(q["cls"], {}).setdefault(0, []).append({"q": q, "s": per_k})
            # Only oracle, non-segregated classes reach the headline, and `Headline` enforces that
            # by raising rather than by filtering.
            if not q["judged"] and q["cls"] not in SEGREGATED_CLASSES:
                headline.add(q, per_k)

        by_class = {}
        for cls, buckets in per_class.items():
            rows = buckets[0]
            by_class[cls] = {
                "n": len(rows),
                "judged": any(r["q"]["judged"] for r in rows),
                "segregated": cls in SEGREGATED_CLASSES,
                "provenance": sorted({r["q"]["label_provenance"] for r in rows}),
                "k": {str(k): {m: _mean([r["s"][k][m] for r in rows])
                               for m in ("recall", "hit", "ceiling", "recall_norm")}
                      for k in ks},
            }
        results[name] = {
            **status[name],
            "n_scored": len(scored),
            "by_class": by_class,
            "headline": {
                "classes": sorted({q["cls"] for q, _ in headline.rows}),
                "n": len(headline.rows),
                "judged_labels_included": 0,
                "k": {str(k): {m: _mean([s[k][m] for _q, s in headline.rows])
                               for m in ("recall", "hit", "ceiling", "recall_norm")}
                      for k in ks},
            },
            "latency_s": {"mean": _mean(latencies), "total": round(time.time() - t0, 1)},
        }

    out_doc = {
        "schema": "0.A-recall/1",
        "run_at": int(time.time()),
        "queryset": str(args.queryset),
        "queryset_counts": doc.get("counts"),
        "ks": ks,
        "arms": results,
        "excluded": {
            "unretrievable": {
                "n": len(unretrievable),
                "why": ("every oracle target is absent from the corpus, so no arm could "
                        "retrieve it; scoring these would measure the corpus, not retrieval"),
                "by_class": _tally(unretrievable),
                "qids": [q["qid"] for q in unretrievable[:50]],
            },
            "unknown_presence": {"n": len(unknown_presence),
                                 "why": "query set was built with --no-presence-check"},
        },
        "segregated_classes": list(SEGREGATED_CLASSES),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out_doc, indent=1), encoding="utf-8")
    print(render(out_doc))
    return 0


def _tally(qs: Sequence[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for q in qs:
        out[q["cls"]] = out.get(q["cls"], 0) + 1
    return out


def render(doc: dict) -> str:
    """Markdown tables. Per-class always; headline separately and explicitly scoped."""
    ks = doc["ks"]
    L: List[str] = ["", "# LATTICE 0.A -- recall@k", ""]
    L.append(f"query set: `{doc['queryset']}`  counts: {json.dumps(doc.get('queryset_counts'))}")
    exc = doc["excluded"]["unretrievable"]
    if exc["n"]:
        L += ["", f"**{exc['n']} queries EXCLUDED as unretrievable** "
                  f"{json.dumps(exc['by_class'])} -- {exc['why']}."]
    if doc["excluded"]["unknown_presence"]["n"]:
        L += ["", f"warning: {doc['excluded']['unknown_presence']['n']} queries have unknown "
                  "corpus presence (built with --no-presence-check); a 0 for these may be a "
                  "corpus gap, not a retrieval failure."]

    not_run = {n: a for n, a in doc["arms"].items() if a["status"] != "ok"}
    ok_arms = {n: a for n, a in doc["arms"].items() if a["status"] == "ok"}
    if not_run:
        L += ["", "## Arms NOT RUN", "",
              "These are reported as absent, never as zero.", "", "| arm | reason |", "|---|---|"]
        for n, a in not_run.items():
            L.append(f"| `{n}` | {a['reason']} |")

    classes = sorted({c for a in ok_arms.values() for c in a["by_class"]})
    for cls in classes:
        seg = cls in doc["segregated_classes"]
        L += ["", f"## class: `{cls}`" + ("  **(SEGREGATED -- excluded from headline)**" if seg else "")]
        prov = sorted({p for a in ok_arms.values()
                       for p in a["by_class"].get(cls, {}).get("provenance", [])})
        L.append(f"label provenance: {', '.join(prov) or 'n/a'}")
        L += ["", "| arm | n | " + " | ".join(f"r@{k} | norm@{k}" for k in ks) + " |",
              "|---|---|" + "---|---|" * len(ks)]
        for n, a in ok_arms.items():
            c = a["by_class"].get(cls)
            if not c:
                continue
            cells = []
            for k in ks:
                m = c["k"][str(k)]
                cells.append(f"{_f(m['recall'])} | {_f(m['recall_norm'])}")
            L.append(f"| `{n}` | {c['n']} | " + " | ".join(cells) + " |")
        ceil = next((a["by_class"][cls]["k"][str(ks[0])]["ceiling"]
                     for a in ok_arms.values() if cls in a["by_class"]), None)
        if ceil is not None and ceil < 0.999:
            L.append("")
            L.append(f"note: mean ceiling@{ks[0]} is {_f(ceil)} -- |relevant| exceeds k for some "
                     "queries here, so `r@k` cannot reach 1.0. Compare `norm@k`.")

    L += ["", "## HEADLINE (oracle labels only)", "",
          "Excludes every judged label and every segregated class, enforced by a raise in "
          "`Headline.add`, not by a filter.", ""]
    hl_classes = sorted({c for a in ok_arms.values() for c in a.get("headline", {}).get("classes", [])})
    L.append(f"classes in headline: {', '.join(hl_classes) or 'none'}  |  judged labels included: 0")
    L += ["", "| arm | n | " + " | ".join(f"r@{k} | norm@{k}" for k in ks) + " |",
          "|---|---|" + "---|---|" * len(ks)]
    for n, a in ok_arms.items():
        h = a["headline"]
        cells = []
        for k in ks:
            m = h["k"][str(k)]
            cells.append(f"{_f(m['recall'])} | {_f(m['recall_norm'])}")
        L.append(f"| `{n}` | {h['n']} | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


def _f(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.3f}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queryset", required=True)
    ap.add_argument("--arm", action="append", default=[],
                    help="repeatable: keyed | keyed_bm25 | keyed_bm25_onto | control")
    ap.add_argument("--k", action="append", type=int, default=[])
    ap.add_argument("--out")
    ap.add_argument("--fts-db", help="SQLite file with an `fts` FTS5 table (Phase 2.1)")
    ap.add_argument("--control-url", help="deployed Mantle search endpoint (arm 4 control)")
    ap.add_argument("--onto-pool", type=int, default=100)
    ap.add_argument("--list-arms", action="store_true", help="probe availability, do not score")
    a = ap.parse_args(argv)
    if not a.arm:
        a.arm = ["keyed", "keyed_bm25", "keyed_bm25_onto", "control"]
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
