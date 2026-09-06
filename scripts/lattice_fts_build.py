#!/usr/bin/env python3
"""LATTICE Phase 0.A -- build the FTS5 index arms 2 and 3 need.

`src/ember/corpus/fts.py` builds and queries FTS5 indexes; it is covered by its own test suite
(51 tests). This script is the one thing in the tree that builds an FTS5 index from the corpus,
and `lattice_arms.py`'s arm 2 (`Bm25Arm`) reads the index this script writes, via `--fts-db`. The
index is a measurement artifact: a standalone file for the A/B that changes no default and is
imported by nothing in the runtime.

What is indexed
-----------------------------------------------------------------------------------
Documents come from the evaluation corpus, and each carries `text_source`:

    inline   wn-* / concept-* / sym-*  -- complete text (these rows never use CAS)
    cas      wiki-*                    -- the real plaintext, fetched and sha256-verified
    preview  wiki-*                    -- a 300-char preview; CAS was unavailable for this row

`--refuse-preview` (default on) aborts if any `preview` row would be indexed. An index built over
previews looks healthy at every observable point -- BM25 ranks sensibly, snippets render, no error
is raised -- while lexical recall past the first ~50 words is gone, so 0.A is not run on previews.
The counts by `text_source` are printed and stored in the index file so the recall table can be
read alongside them.

No field weighting
------------------------------------------------------------------------------
`bm25()` is called with no weight arguments, so every column contributes equally. Fields do not
carry weights: there is no `FieldWeights` type, no presets, and no loader. A per-field boost is a
hand-picked claim about what matters, which nothing here measures. Weighting is a query-time
parameter, so the index itself is unaffected.

USAGE
    python lattice_fts_build.py --corpus <eval-dir>/eval.sqlite \
                                --out    <eval-dir>/eval-fts.sqlite
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_MANTLE_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "agience-mantle", "src"))
for _p in (_HERE, _MANTLE_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ember.corpus.fts import FtsDocument, FtsIndex  # noqa: E402


def rows(corpus: sqlite3.Connection, batch: int = 5000) -> Iterator[List[tuple]]:
    cur = corpus.execute("SELECT id, title, context, ct, collection, text, text_source FROM doc")
    while True:
        chunk = cur.fetchmany(batch)
        if not chunk:
            return
        yield chunk


def build(args) -> int:
    src = sqlite3.connect("file:%s?mode=ro" % args.corpus, uri=True)
    by_src = {r[0]: r[1] for r in src.execute(
        "SELECT text_source, count(*) FROM doc GROUP BY 1")}
    print("evaluation corpus %s" % args.corpus)
    for k, v in sorted(by_src.items()):
        print("   text_source=%-9s %d" % (k, v))
    n_prev = by_src.get("preview", 0)
    if n_prev and args.refuse_preview:
        raise SystemExit(
            "REFUSING to build: %d documents would be indexed from a 300-char PREVIEW rather "
            "than their real CAS text. Such an index looks healthy at every observable point -- "
            "BM25 ranks sensibly, no error is raised -- while recall past ~50 words is gone, so "
            "a recall number measured on it measures the wrong system. Fix the CAS fetch, or "
            "pass --allow-preview and report the count prominently." % n_prev)

    out = Path(args.out)
    if out.exists():
        out.unlink()
    dst = sqlite3.connect(str(out))
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=NORMAL")
    idx = FtsIndex(dst, resolver=None)
    idx.ensure_schema()
    dst.execute("CREATE TABLE IF NOT EXISTS index_provenance (k TEXT PRIMARY KEY, v TEXT)")

    n = 0
    stat = Counter()
    t0 = time.time()
    for chunk in rows(src):
        docs = []
        for aid, title, context, ct, coll, text, tsrc in chunk:
            stat[tsrc] += 1
            stat["ct:" + (ct or "?")] += 1
            docs.append(FtsDocument(
                vertex_id=aid,
                title=title or "",
                # `context` is the offer field -- what the artifact advertises, and what retrieval
                # indexes on. It maps to the description column.
                description=context or "",
                tags=" ".join(x for x in (ct, coll) if x),
                content=text or ""))
        n += idx.index_many(docs)
        if n % 25000 < len(chunk):
            print("   indexed %d (%.0fs)" % (n, time.time() - t0), file=sys.stderr)
    dst.commit()

    prov = {"built_at": int(time.time()), "corpus": str(args.corpus), "documents": n,
            "by_text_source": {k: v for k, v in stat.items() if not k.startswith("ct:")},
            "by_content_type": {k[3:]: v for k, v in stat.items() if k.startswith("ct:")},
            "tokenizer": "porter unicode61", "contentless_delete": True,
            "weights_note": "no field weighting -- bm25() is called with no weight arguments",
            "stemmer_note": "FTS5 porter != sse/tokenizer.py porter -- this is a REBUILD, not a "
                            "migration; every lexical result shifts (Unit M DECISION 1)"}
    for k, v in prov.items():
        dst.execute("INSERT OR REPLACE INTO index_provenance (k,v) VALUES (?,?)",
                    (k, json.dumps(v)))
    dst.commit()

    vocab = dst.execute("SELECT count(*) FROM (SELECT DISTINCT term FROM fts_vocab)").fetchone() \
        if _has_vocab(dst) else None
    print("\nindexed %d documents in %.0fs (%.0f docs/s)"
          % (n, time.time() - t0, n / max(time.time() - t0, 1e-9)))
    print("index file %s (%.1f MB)" % (out, out.stat().st_size / 1e6))
    if vocab:
        print("distinct stems %d" % vocab[0])
    print("by text_source: %s" % {k: v for k, v in stat.items() if not k.startswith("ct:")})
    dst.close()
    return 0


def _has_vocab(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_vocab USING fts5vocab(fts,'row')")
        return True
    except Exception:  # noqa: BLE001
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-preview", dest="refuse_preview", action="store_false",
                    help="index preview-only rows anyway (the count is reported prominently)")
    ap.set_defaults(refuse_preview=True)
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
