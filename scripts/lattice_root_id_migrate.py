#!/usr/bin/env python3
"""One-time §6B migration: give 6.25M existing rows their version-lineage handle.

The step order carries the cost, and it is the `ix_lk_lookup` lesson applied again:

    1. ALTER TABLE ADD COLUMN root_id     -- metadata only, instant even on 6.25M rows
    2. UPDATE every row to fill it        -- one bulk write, no index to maintain
    3. CREATE INDEX ix_v_root_id          -- a single ordered pass at the end

`ensure_schema()` creates the index first (it is in `ALL_DDL`), which makes step 2 maintain a
random-valued B-tree across 6.25M row updates — the same reason the listkey index is dropped for
the bulk load. A store opened by any other tool in the meantime creates the index early, so the
order here holds only if this runs first.

The backfill is part of the feature, and its absence is quiet. A row written without `root_id`
holds NULL, and a NULL discriminator is invisible to every query that filters on it — the way
`wn-*` rows with `ct IS NULL` are absent from every typed query while present in the totals. A
lineage feature over a column that is NULL for 6.25M rows is nominal.

Run after the pipeline has finished. This rewrites every row, so racing the migration's own writers
means lock contention on a store that is already the critical path.

    python3 lattice_root_id_migrate.py --db work/corpus.db --dry-run
    python3 lattice_root_id_migrate.py --db work/corpus.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/home/builder/genesis/lattice/work/corpus.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=%s" % (args.db, "ro" if args.dry_run else "rw"), uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=900000")
    tag = "[DRY RUN] " if args.dry_run else ""

    cols = {r[1] for r in con.execute("PRAGMA table_info(vertex)")}
    idx = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='vertex'")}
    print("=== state ===")
    print("  root_id column  : %s" % ("present" if "root_id" in cols else "ABSENT"))
    print("  ix_v_root_id    : %s" % ("present" if "ix_v_root_id" in idx else "absent"))

    # EXISTS rather than count(*). On this store count(*) dereferences every record — a full-table
    # cost this check does not need — and the answer to "is there work left?" is a boolean.
    def _has_col() -> bool:
        # The schema is re-read here rather than taken from `cols`, which is captured once at the
        # top and cannot see the ALTER below. Closing over that snapshot leaves pending()
        # permanently True, so the script does the work and then reports FAILED. Pinned by
        # test_root_id_migration_adds_backfills_and_indexes_in_that_order; a dry-run returns
        # before this path, so it cannot cover it.
        return "root_id" in {r[1] for r in con.execute("PRAGMA table_info(vertex)")}

    def pending() -> bool:
        if not _has_col():
            return True
        return bool(con.execute(
            "SELECT EXISTS(SELECT 1 FROM vertex WHERE root_id IS NULL)").fetchone()[0])

    print("  rows still NULL : %s" % ("YES" if pending() else "no"))
    if args.dry_run:
        print()
        print("%sWOULD: add column (if absent) -> backfill root_id=COALESCE(doc.root_id, id) "
              "-> create ix_v_root_id" % tag)
        print("%sorder matters: the index is built LAST so the bulk UPDATE maintains no B-tree"
              % tag)
        return 0

    # ── 1. the column ────────────────────────────────────────────────────────────────────────
    if not _has_col():
        t = time.time()
        con.execute("ALTER TABLE vertex ADD COLUMN root_id TEXT")
        con.commit()
        print("  + column added in %.2fs" % (time.time() - t))

    # ── 2. the backfill, before the index ────────────────────────────────────────────────────
    if pending():
        t = time.time()
        cur = con.execute(
            "UPDATE vertex SET root_id = COALESCE(json_extract(doc, '$.root_id'), id) "
            "WHERE root_id IS NULL")
        con.commit()
        print("  + backfilled %d row(s) in %.1fs" % (cur.rowcount, time.time() - t))
    else:
        print("  + backfill: nothing to do (idempotent)")

    # ── 3. the index, last ───────────────────────────────────────────────────────────────────
    if "ix_v_root_id" not in idx:
        t = time.time()
        con.execute("CREATE INDEX ix_v_root_id ON vertex(root_id)")
        con.commit()
        print("  + ix_v_root_id built in %.1fs" % (time.time() - t))

    # ── verify ───────────────────────────────────────────────────────────────────────────────
    still = pending()
    print()
    print("=== verify ===")
    print("  rows still NULL : %s   <- MUST be no" % ("YES" if still else "no"))
    sample = con.execute(
        "SELECT id, root_id FROM vertex WHERE root_id IS NOT NULL LIMIT 3").fetchall()
    for r in sample:
        print("    %s -> root_id=%s%s" % (r["id"], r["root_id"],
                                          "  (own root: first version)" if r["id"] == r["root_id"]
                                          else ""))
    con.close()
    if still:
        print("FAILED: rows remain without a lineage handle", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
