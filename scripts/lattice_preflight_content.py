#!/usr/bin/env python3
"""Pre-flight: does the local blob corpus hold every `content_ref` the store references?

This is the comparison the other metrics do not make, run before the expensive work rather than
discovered inside it. Measured: the `content` stage failed after scanning all 6,114,746 refs
because 15 were absent, and identifying them then cost a 25-minute anti-join, after ~4 hours of
migration had already run.

Both existing metrics are correct on their own terms:

  · `pull.py` reports `coverage: 100.000% of manifest` — true. The manifest is what `manifest.py`
    listed from S3, and those 15 objects are not in S3 at all (verified: HTTP 404 `NoSuchKey`
    against working credentials, with a known-good ref returning 200 as a control).
  · `enrich` reports per-origin coverage and warns on an origin at 0% — also true, and itemised.

The manifest describes what the source has; the store describes what the corpus needs. The gap
lives between two honest metrics, which is harder to catch than a wrong number because there is
nothing wrong to find in either one.

## How it asks

A temp-table sieve: insert every needed ref, then DELETE the ones each blob shard holds. What
survives is absent from all of them.

No `count(*)` over the corpus, and no shard arithmetic in SQL. On this store `count(*)`
dereferences every record; and SQLite's `CAST('0x1a' AS INTEGER)` is 0, not 26 — it does not parse
hex — so computing a ref's shard inside a query mis-assigns every row while still returning a
confident answer. The sieve needs neither.

    python3 lattice_preflight_content.py --target work/corpus.db --local-corpus ../local-corpus
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys


def missing_refs(target: str, local_corpus: str, shards: int = 16):
    """Refs the store references that no blob shard holds. Returns (needed_count, [missing])."""
    con = sqlite3.connect("file:%s?mode=ro" % target, uri=True)
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("CREATE TEMP TABLE need(ref TEXT PRIMARY KEY) WITHOUT ROWID")
    con.execute("INSERT INTO need(ref) SELECT DISTINCT content_ref FROM vertex "
                "WHERE content_ref IS NOT NULL")
    needed = con.execute("SELECT count(*) FROM need").fetchone()[0]   # temp table, not the corpus
    absent_shards = []
    for s in range(shards):
        f = os.path.join(local_corpus, "blobs-%02d.sqlite" % s)
        if not os.path.exists(f):
            absent_shards.append(s)
            continue
        con.execute("ATTACH DATABASE ? AS b", ("file:%s?mode=ro" % f,))
        con.execute("DELETE FROM need WHERE ref IN (SELECT ref FROM b.blob)")
        # Commit before DETACH. The DELETE holds a read lock on the attached shard, and `DETACH`
        # fails with "database b is locked" while it is held.
        con.commit()
        con.execute("DETACH DATABASE b")
    out = [r[0] for r in con.execute("SELECT ref FROM need ORDER BY ref")]
    con.close()
    return needed, out, absent_shards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--local-corpus", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--out", help="write the missing refs here, one per line")
    args = ap.parse_args()

    needed, missing, absent_shards = missing_refs(args.target, args.local_corpus, args.shards)
    print("refs the corpus NEEDS      : %d" % needed)
    print("blob shard files missing   : %s" % (absent_shards or "none"))
    print("refs NOT in local-corpus   : %d" % len(missing))
    if absent_shards:
        # A missing shard file is a different finding from a missing ref, and is reported as its
        # own: every ref that shard would have held reads as absent.
        print("⚠ %d shard file(s) are missing entirely, so the 'not in local-corpus' count above "
              "is inflated by whatever they held. Fix the shards before trusting it."
              % len(absent_shards), file=sys.stderr)
    for ref in missing[:50]:
        print("   %s" % ref)
    if len(missing) > 50:
        print("   ... and %d more (use --out for the full list)" % (len(missing) - 50))
    if args.out and missing:
        with open(args.out, "w") as fh:
            fh.write("\n".join(missing) + "\n")
        print("full list -> %s" % args.out)

    if missing:
        print()
        print("PRE-FLIGHT FAILED: the content stage would fail on these. Recover them first "
              "(lattice_recover_blobs.py from the origin, or lattice_regen_from_dataset.py if the "
              "source dataset is PINNED), then re-run this check.")
        return 1
    print()
    print("PRE-FLIGHT OK: every referenced blob is present locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
