#!/usr/bin/env python3
"""Apply precomputed MinHash signatures from the side tables onto the corpus rows.

`lattice_signatures.py` computes signatures off `local-corpus` into per-worker side DBs, keyed by
`content_ref`. This pass joins them onto `vertex` by that ref and writes `doc.minhash` — a cheap
keyed update instead of a second read-and-hash over 24.5 GB of content.

What this disarms: where `minhash` is unwritten, every signature the consolidation operator sees is
all-zero, and `estimated_jaccard(zeros, zeros)` is 1.0. With `apply=True` that archives every
unsigned row as a duplicate of one row, with only the unsigned-artifact guard standing between that
and the corpus. Signatures make consolidation safe rather than merely blocked.

## Degenerate blobs get no signature

A blob that yielded no shingles is recorded `degenerate=1` with a NULL signature, and this pass
writes `minhash_status = "degenerate_no_shingles"` in place of a value. Writing zeros would rebuild
the hazard one layer down: a reader would find "a signature" and take 1.0 similarity as evidence.
Same rule as `ic_status`, `Page.truncated`, `K_signal == 0` and `fit_error = NaN` — an absent
measurement stays absent rather than becoming an extreme value that passes every threshold.

Run after `creation`. This writes vertices, so it is kept clear of the pipeline's own writers.

    python3 lattice_apply_signatures.py --target work/corpus.db --dry-run
    python3 lattice_apply_signatures.py --target work/corpus.db
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--sigs", default="/home/builder/genesis/lattice/work/signatures.sqlite")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--report", type=float, default=30.0)
    args = ap.parse_args()

    sys.path.insert(0, "/home/builder/genesis/lattice/pysrc")
    from mantle.db.vertex import LatticeArtifactStore

    # ── the side tables. One per worker; ATTACH them all and read as one. ────────────────────
    # Match only the numbered shards. A bare `.*` glob also matches SQLite's `-wal` and `-shm`
    # sidecars, and ATTACHing a write-ahead log reports "file is not a database" — which reads as
    # corruption rather than as the glob being wrong.
    import re as _re
    parts = sorted(p for p in glob.glob(args.sigs + ".*")
                   if _re.fullmatch(_re.escape(args.sigs) + r"\.\d+", p))
    if not parts and os.path.exists(args.sigs):
        parts = [args.sigs]
    if not parts:
        print("FATAL: no signature side-tables at %s[.N] — run lattice_signatures.py first. "
              "Refusing to proceed: writing no signatures is not the same as writing none "
              "successfully." % args.sigs, file=sys.stderr)
        return 2
    # uri=True is what lets ATTACH accept a "file:...?mode=ro" filename; without it SQLite treats
    # the whole URI as a literal path and reports "file is not a database", which reads like
    # corruption rather than a typo.
    sig = sqlite3.connect(":memory:", uri=True)
    for i, p in enumerate(parts):
        sig.execute("ATTACH DATABASE ? AS s%d" % i, ("file:%s?mode=ro" % p,))
    union = " UNION ALL ".join("SELECT ref, sig, degenerate FROM s%d.sig" % i
                               for i in range(len(parts)))
    print("side tables: %d (%s)" % (len(parts), ", ".join(os.path.basename(p) for p in parts)))

    # Held in memory: a per-ref query against N attached DBs would be N lookups per row over 6.11M
    # rows. At ~1 KB/entry this is a few hundred MB and the box has 23 GB.
    print("loading signatures...")
    t0 = time.time()
    table = {}
    for ref, blob, degen in sig.execute(union):
        table[ref] = (blob, degen)
    print("  %d signature record(s) in %.1fs" % (len(table), time.time() - t0))
    ndegen = sum(1 for _, d in table.values() if d)
    print("  degenerate: %d  (will get minhash_status, NOT zeros)" % ndegen)

    store = LatticeArtifactStore(args.target, origin=os.environ.get("EMBER_NODE_ID", "45"))
    con = store.db.read()
    con.execute("PRAGMA busy_timeout=600000")

    applied = degen_marked = missing = skipped = scanned = 0
    cursor, t0, last = "", time.time(), 0.0

    while True:
        rows = con.execute(
            "SELECT id, content_ref, doc FROM vertex "
            "WHERE id > ? AND content_ref IS NOT NULL ORDER BY id LIMIT ?",
            (cursor, args.chunk)).fetchall()
        if not rows:
            break
        batch = []
        for r in rows:
            cursor = r["id"]
            scanned += 1
            got = table.get(r["content_ref"])
            if got is None:
                missing += 1                      # not signed yet; a later pass picks it up
                continue
            blob, degen = got
            doc = json.loads(r["doc"])
            if degen:
                if doc.get("minhash_status") == "degenerate_no_shingles":
                    skipped += 1
                    continue
                doc.pop("minhash", None)
                doc["minhash_status"] = "degenerate_no_shingles"
                batch.append((r["id"], doc))
                degen_marked += 1
            else:
                vals = [int.from_bytes(blob[i:i + 8], "big") for i in range(0, len(blob), 8)]
                if doc.get("minhash") == vals:
                    skipped += 1
                    continue
                doc["minhash"] = vals
                doc.pop("minhash_status", None)
                batch.append((r["id"], doc))
                applied += 1
        if batch and not args.dry_run:
            with store.db.write():
                for aid, doc in batch:
                    store.put_artifact(doc, stamp_rev=True)
        now = time.time()
        if now - last >= args.report:
            last = now
            el = now - t0
            print("  %8d scanned  %8d applied  %6d degenerate  %7d unsigned  %6d unchanged  "
                  "%5.0f/s" % (scanned, applied, degen_marked, missing, skipped,
                               scanned / max(el, 1e-9)))

    el = time.time() - t0
    print()
    print("=" * 78)
    print("%s scanned=%d applied=%d degenerate_marked=%d unsigned=%d unchanged=%d (%.1fs)"
          % ("DRY RUN" if args.dry_run else "DONE", scanned, applied, degen_marked,
             missing, skipped, el))
    # Reported even at zero. "unsigned" is the number the consolidation gate reads: an operator
    # runs only over a scope where it is 0, because an unsigned row is one the comparison cannot
    # see rather than one it found dissimilar.
    print("UNSIGNED ROWS: %d   <- consolidation must refuse any scope where this is > 0" % missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
