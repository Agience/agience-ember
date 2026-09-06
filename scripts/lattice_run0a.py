#!/usr/bin/env python3
"""LATTICE Phase 0.A -- RUN the retrieval A/B.

Drives `lattice_arms.py`'s arms over the evaluation corpus and scores them with Unit Q's harness.
`Headline`, `score_query`, `render` and `_tally` are imported from `lattice_recall.py` rather than
reimplemented: `Headline.add` is the anti-circularity guard — it raises on a judged label or a
segregated class rather than filtering one out — and a second copy of the scoring code would be a
second place for that guard to be softened.

What this run measures
-----------------------
Corpus presence is re-stamped against this corpus. A target absent from the searched corpus is
unretrievable by every arm, so scoring it measures the corpus rather than the retriever.

Variants run as separate arms so each appears in the per-class table on its own row:

    keyed              undiscriminated lemma lookup (reproduces the live 5.1.1 defect)
    keyed_typed        + content-type discriminator (the Phase 3.2 target)
    keyed_bm25         + FTS5 BM25 (no field weighting — bm25() takes no weights)
    keyed_bm25_onto    + ontology coordinate, centering off        (the default)
    keyed_bm25_onto_centered  + ontology coordinate, centering on  (opt-in, Unit G)
    control            not run -- see `lattice_arms.ControlArm`

This script enables nothing as a default: not FTS5, not centering, not `oov="skip"`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lattice_recall import (Headline, SEGREGATED_CLASSES, _mean, _tally,  # noqa: E402
                            render, score_query)
import lattice_arms as A  # noqa: E402


def stamp_presence(queries: List[dict], corpus) -> Dict[str, int]:
    present = {r[0] for r in corpus.db.execute("SELECT id FROM doc")}
    tally = {"all": 0, "partial": 0, "none": 0}
    for q in queries:
        have = [i for i in q["relevant"] if i in present]
        q["corpus_presence"] = {"present": len(have), "absent": q["n_relevant"] - len(have),
                                "all_present": len(have) == q["n_relevant"],
                                "none_present": len(have) == 0}
        # Score against the PRESENT subset so a partially-absent oracle does not read as a
        # retrieval failure; `n_relevant` stays as authored so the shortfall stays visible.
        q["relevant_present"] = have
        tally["none" if not have else "all" if len(have) == q["n_relevant"] else "partial"] += 1
    return tally


def run(args) -> int:
    doc = json.loads(Path(args.queryset).read_text(encoding="utf-8"))
    queries: List[dict] = doc["queries"]
    ks = sorted(set(args.k or (1, 5, 10, 20)))
    kmax = max(ks)

    corpus = A.EvalCorpus(args.corpus)
    cstats = corpus.stats()
    print("evaluation corpus: %s" % json.dumps(cstats, indent=1), file=sys.stderr)

    ic_cov = A.load_wn_offline(args.wn)
    print("wordnet loaded offline: %s" % ic_cov, file=sys.stderr)

    pres = stamp_presence(queries, corpus)
    print("corpus presence: %s" % pres, file=sys.stderr)

    keyed = A.KeyedArm(corpus, typed=False)
    keyed_typed = A.KeyedArm(corpus, typed=True)
    bm25 = A.Bm25Arm(corpus, args.fts_db, keyed)

    center = None
    center_err = None
    if args.centering_mean and Path(args.centering_mean).exists():
        from ember.ontology import geometry
        center = geometry.load_centering_mean(args.centering_mean)
    elif args.centering_mean:
        center_err = "centering mean %s not found" % args.centering_mean

    uncentered_cache: dict = {}          # shared ONLY between the two center=None arms
    onto = A.OntologyArm(corpus, bm25, pool=args.onto_pool, center=None, label="",
                         cache=uncentered_cache)
    onto_c = A.OntologyArm(corpus, bm25, pool=args.onto_pool, center=center, label="_centered")
    onto_kf = A.OntologyArm(corpus, bm25, pool=args.onto_pool, center=None,
                            label="_keyedfirst", keyed_first=True, cache=uncentered_cache)
    if center is None:
        onto_c._err = center_err or ("no centering mean supplied; centering ON cannot be "
                                     "measured without one -- a recomputed-per-call mean is a "
                                     "different translation on every node (Unit G)")
    control = A.ControlArm(args.control_url)

    arms = [keyed, keyed_typed, bm25, onto, onto_c, onto_kf, control]
    status: Dict[str, dict] = {}
    for arm in arms:
        ok, why = arm.available()
        status[arm.name] = {"status": "ok" if ok else "not_run", "reason": why}
        print("arm %-28s %s  %s" % (arm.name, "OK" if ok else "NOT RUN", why[:150]),
              file=sys.stderr)
    if args.list_arms:
        print(json.dumps(status, indent=2))
        return 0

    scored, unretrievable = [], []
    for q in queries:
        (unretrievable if q["corpus_presence"]["none_present"] else scored).append(q)

    results: Dict[str, dict] = {}
    for arm in arms:
        if status[arm.name]["status"] != "ok":
            results[arm.name] = {**status[arm.name], "by_class": {}, "headline": {}}
            continue
        per_class: Dict[str, List[dict]] = {}
        headline = Headline()
        lat, t0 = [], time.time()
        for i, q in enumerate(scored):
            t1 = time.time()
            try:
                got = arm.retrieve(q["text"], kmax, q["cls"])
            except Exception as e:  # noqa: BLE001
                print("  ! %s failed on %s: %s" % (arm.name, q["qid"], str(e)[:140]),
                      file=sys.stderr)
                got = []
            lat.append(time.time() - t1)
            per_k = score_query(got, q["relevant_present"], ks)
            per_class.setdefault(q["cls"], []).append({"q": q, "s": per_k})
            if not q["judged"] and q["cls"] not in SEGREGATED_CLASSES:
                headline.add(q, per_k)
            if args.progress and i and i % args.progress == 0:
                print("    %s %d/%d" % (arm.name, i, len(scored)), file=sys.stderr)
        by_class = {}
        for cls, rws in per_class.items():
            by_class[cls] = {
                "n": len(rws),
                "judged": any(r["q"]["judged"] for r in rws),
                "segregated": cls in SEGREGATED_CLASSES,
                "provenance": sorted({r["q"]["label_provenance"] for r in rws}),
                "k": {str(k): {m: _mean([r["s"][k][m] for r in rws])
                               for m in ("recall", "hit", "ceiling", "recall_norm")}
                      for k in ks},
            }
        results[arm.name] = {
            **status[arm.name], "n_scored": len(scored), "by_class": by_class,
            "headline": {"classes": sorted({q["cls"] for q, _ in headline.rows}),
                         "n": len(headline.rows), "judged_labels_included": 0,
                         "k": {str(k): {m: _mean([s[k][m] for _q, s in headline.rows])
                                        for m in ("recall", "hit", "ceiling", "recall_norm")}
                               for k in ks}},
            "latency_s": {"mean": _mean(lat), "total": round(time.time() - t0, 1)},
        }
        print("  %-28s done in %.0fs" % (arm.name, time.time() - t0), file=sys.stderr)

    out_doc = {
        "schema": "0.A-recall/1",
        "run_at": int(time.time()),
        "queryset": str(args.queryset),
        "queryset_counts": doc.get("counts"),
        "corpus": {"path": args.corpus, **cstats},
        "wordnet_ic_coverage": ic_cov,
        "rerank_pool": args.onto_pool,
        "ks": ks,
        "arms": results,
        "excluded": {
            "unretrievable": {
                "n": len(unretrievable),
                "why": ("every oracle target is absent from the corpus, so no arm could "
                        "retrieve it; scoring these would measure the corpus, not retrieval"),
                "by_class": _tally(unretrievable),
                "qids": [q["qid"] for q in unretrievable[:50]]},
            "unknown_presence": {"n": 0, "why": "presence re-stamped against this corpus"},
        },
        "presence_tally": pres,
        "segregated_classes": list(SEGREGATED_CLASSES),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out_doc, indent=1), encoding="utf-8")
    print(render(out_doc))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queryset", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fts-db", required=True)
    ap.add_argument("--wn", required=True, help="local wn-* keyset export (offline WordNet)")
    ap.add_argument("--centering-mean", help="saved centering mean (Unit G format)")
    ap.add_argument("--control-url")
    ap.add_argument("--onto-pool", type=int, default=100)
    ap.add_argument("--k", action="append", type=int, default=[])
    ap.add_argument("--out")
    ap.add_argument("--progress", type=int, default=50)
    ap.add_argument("--list-arms", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
