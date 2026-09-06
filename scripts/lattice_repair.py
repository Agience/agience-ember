#!/usr/bin/env python3
"""Post-`vertices` repairs. Every step is idempotent and reports what it measured.

Run this after `creation` rather than after `vertices`. `consolidate` and `creation` may still
write vertices, so `listkey` is final only once `creation` completes; rebuilding the index earlier
indexes a moving table.

  1. Rebuild `ix_lk_lookup`, dropped during bulk load so 6.25M x ~27 postings could be inserted
     without maintaining a random-valued B-tree. Without it `lookup_by_lemma` degrades to a full
     scan: correct answers, unusable latency. Measured on the live store: `define` for four
     unrelated words returned in 9.8–211.7 ms without the index — fast because the scan stops at
     the LIMIT rather than finding the best matches.

  2. Stamp WHEN on system artifacts that lack it. Measured: exactly 40 rows (14 typedefs +
     2 collections + 20 grants + 4 identities). A pipeline re-run leaves these alone — `_fp(doc)`
     is computed before `put_vertex` stamps, so they match their stored fingerprint and skip — so
     this script is the path that repairs them. It uses the pipeline's constants: epoch
     1970-01-01, claimant `genesis` rather than a node id, because "node 45 read its clock as
     1970" is a claim node 45 never made.

  3. Confirm the ~2,000 `wn-*` rows whose enrichment-ledger `content_type` was None carry their
     type. `_project` recovers `ct` from the extract's column (verified populated for all
     117,659). A NULL `ct` is invisible to every typed query and to `ix_v_ct`, and it is the
     discriminator the `define` fix rests on.

  4. Verify TOP coupling: WHO and WHEN on every vertex. Three code paths write WHEN, so the proof
     is a count on the finished store.

    python3 repair.py --dry-run     # measure everything, change nothing
    python3 repair.py               # apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time

DB_DEFAULT = "/home/builder/genesis/lattice/work/corpus.db"
GENESIS_EPOCH = "1970-01-01T00:00:00+00:00"
GENESIS_TIME_CLAIMANT = "genesis"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report; write nothing")
    args = ap.parse_args()

    mode = "ro" if args.dry_run else "rwc"
    con = sqlite3.connect("file:%s?mode=%s" % (args.db, "ro" if args.dry_run else "rw"), uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=900000")
    tag = "[DRY RUN] " if args.dry_run else ""

    # ── 1. the keyed-arm index ───────────────────────────────────────────────────────────────
    print("=== 1. ix_lk_lookup ===")
    idx = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='listkey'")]
    print("  indexes on listkey : %s" % idx)
    # The posting count is informational, and counting a 20M-row table can block for minutes while
    # anything else is writing it — this statement stalled a dry-run during the migration. A number
    # nobody acts on does not gate a repair, so it is skipped unless it comes back cheaply. SQLite
    # will not zombie the node the way an ArcadeDB count(*) can, but "counting a large hot table to
    # print a number" is avoided either way.
    postings = None
    try:
        con.execute("PRAGMA busy_timeout=5000")
        postings = con.execute("SELECT count(*) FROM listkey").fetchone()[0]
    except Exception:
        pass
    finally:
        con.execute("PRAGMA busy_timeout=900000")
    print("  postings           : %s" % ("%d" % postings if postings is not None
                                         else "(not counted — table busy)"))
    if "ix_lk_lookup" in idx:
        print("  ✅ already present — nothing to do (idempotent)")
    elif args.dry_run:
        print("  %sWOULD build ix_lk_lookup(field, value, ct)" % tag)
    else:
        t = time.time()
        con.execute("CREATE INDEX ix_lk_lookup ON listkey(field, value, ct)")
        con.commit()
        print("  built in %.1fs" % (time.time() - t))
    # ix_lk_aid is kept through the bulk load: `_index_lists`' DELETE path needs it, and `aid`
    # arrives in sorted order so maintaining it stays cheap. Verify it is present.
    if "ix_lk_aid" not in idx:
        print("  ⚠ ix_lk_aid MISSING — the _index_lists DELETE path would full-scan")

    # ── 2. WHEN on the system artifacts ──────────────────────────────────────────────────────
    print()
    print("=== 2. WHEN on rows missing it ===")
    missing = [r["id"] for r in con.execute(
        "SELECT id FROM vertex WHERE created_time IS NULL")]
    print("  rows without WHEN  : %d" % len(missing))
    for aid in missing[:8]:
        print("     %s" % aid)
    if missing and not args.dry_run:
        fixed = 0
        for aid in missing:
            row = con.execute("SELECT doc FROM vertex WHERE id=?", (aid,)).fetchone()
            if row is None:
                continue
            doc = json.loads(row["doc"])
            doc["created_time"] = GENESIS_EPOCH
            doc["created_time_origin"] = GENESIS_TIME_CLAIMANT
            con.execute("UPDATE vertex SET created_time=?, doc=? WHERE id=?",
                        (GENESIS_EPOCH, json.dumps(doc, sort_keys=True), aid))
            fixed += 1
        con.commit()
        print("  stamped %d row(s), claimant=%s" % (fixed, GENESIS_TIME_CLAIMANT))
    elif missing:
        print("  %sWOULD stamp %d row(s) with %s / claimant=%s"
              % (tag, len(missing), GENESIS_EPOCH, GENESIS_TIME_CLAIMANT))

    # ── 3. the wn rows that had no content_type in the ledger ────────────────────────────────
    print()
    print("=== 3. wn-* content_type recovery ===")
    tot = con.execute("SELECT count(*) FROM vertex WHERE id LIKE 'wn-%'").fetchone()[0]
    typed = con.execute(
        "SELECT count(*) FROM vertex WHERE id LIKE 'wn-%' AND ct = 'text/x-wordnet'").fetchone()[0]
    nullct = con.execute("SELECT count(*) FROM vertex WHERE ct IS NULL").fetchone()[0]
    print("  wn-* rows          : %d" % tot)
    print("  typed x-wordnet    : %d" % typed)
    print("  untyped wn-*       : %d   <- MUST be 0 (a NULL ct is invisible to every typed query)"
          % (tot - typed))
    print("  NULL ct anywhere   : %d   <- MUST be 0" % nullct)

    # ── 4. TOP coupling ──────────────────────────────────────────────────────────────────────
    print()
    print("=== 4. TOP coupling (WHO + WHEN on every vertex) ===")
    r = con.execute("SELECT count(*) n, sum(created_time IS NOT NULL) w, "
                    "sum(created_by IS NOT NULL) o FROM vertex").fetchone()
    n, w, o = r["n"], r["w"] or 0, r["o"] or 0
    print("  rows               : %d" % n)
    print("  WHO  (created_by)  : %d (%.4f%%)" % (o, 100.0 * o / max(n, 1)))
    print("  WHEN (created_time): %d (%.4f%%)" % (w, 100.0 * w / max(n, 1)))
    print("  UNCOUPLED          : %d   <- MUST be 0" % (n - min(w, o)))
    print()
    print("  claimant split:")
    for row in con.execute(
            "SELECT json_extract(doc,'$.created_time_origin') c, count(*) k "
            "FROM vertex GROUP BY c ORDER BY k DESC"):
        print("    %-12s %d" % (str(row["c"]), row["k"]))

    con.close()
    print()
    print("%sdone" % tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
