#!/usr/bin/env python3
"""Compute near-duplicate signatures for every blob in `local-corpus`, into a side table.

This never opens `corpus.db`. Signatures are keyed by `content_ref` and `local-corpus` is keyed by
`content_ref`, so the whole computation runs off the local blob shards with no contention against a
live migration. The B2 pass later applies them as a cheap keyed update instead of a second
read-and-hash over 24.5 GB.

What signing buys. Where `minhash` is unwritten, every signature reads back all-zero, and
`estimated_jaccard(zeros, zeros)` is 1.0 — so `consolidate_nearvdup` with `apply=True` archives the
entire corpus as duplicates of one row, with only the unsigned-artifact guard standing between it
and the corpus. Signatures are what disarm it.

Computed from full plaintext rather than from `doc["content"]`: that field is a 300-char preview on
older rows and absent on new ones (§6A rules it out). `local-corpus` holds the real bytes,
sha256-verified by `pull.py` on the way in, which is what makes it the right source.

## Degeneracy is recorded rather than stored as a value

A blob that yields no shingles gets `degenerate = 1` and no signature is written. An all-zero
signature would rebuild the hazard one layer down: a later reader would find "a signature" and take
1.0 similarity as evidence. Absence stays absence — the same rule `ic_status`, `Page.truncated` and
`K_signal == 0` all encode.

    python3 lattice_signatures.py --shard 0 --shards 8
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS sig (
  ref        TEXT PRIMARY KEY,     -- 'cas/<sha256>' — the same key local-corpus and vertex use
  sig        BLOB,                 -- 128 x 8-byte big-endian minhash values; NULL if degenerate
  nbytes     INTEGER NOT NULL,     -- plaintext size, so a later pass need not re-read the blob
  degenerate INTEGER NOT NULL      -- 1 = no shingles. NOT a zero signature. See the docstring.
) WITHOUT ROWID;
"""


def _load_minhash():
    """The near-duplicate estimator, `prism.minhash`, loaded as a plain import.

    `ember/genesis.py` reads the same estimator by the same name — `from prism import minhash`.
    One distribution, one module, one name at both call sites, so the signatures written here and
    the merge boundary read there stay in step. That agreement is the reason signing exists: an
    all-zero signature makes `estimated_jaccard` return 1.0 and near-dup consolidation archive the
    corpus.
    """
    from prism import minhash
    return minhash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-corpus", default="/home/builder/genesis/local-corpus")
    ap.add_argument("--out", default="/home/builder/genesis/lattice/work/signatures.sqlite")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--report", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    M = _load_minhash()

    # One output DB per worker — no writer contention, no lock waits. Merge after.
    out_path = args.out if args.shards == 1 else "%s.%d" % (args.out, args.shard)
    out = sqlite3.connect(out_path)
    out.execute("PRAGMA journal_mode=WAL")
    out.execute("PRAGMA synchronous=NORMAL")
    out.executescript(SCHEMA)
    out.commit()

    # ── resume from every sibling shard, not just our own ────────────────────────────────────
    # Workers claim whole blob files, so changing --shards re-maps files to workers. A worker that
    # read only its own DB would recompute refs a differently-numbered worker already signed. The
    # values are deterministic, so that is wasteful rather than wrong — but the union is two lines
    # and makes the shard count a free parameter instead of a choice locked in at first launch.
    done = {r[0] for r in out.execute("SELECT ref FROM sig")}
    mine_n = len(done)
    for sib in sorted(glob.glob(args.out + ".*")):
        if not sib.rsplit(".", 1)[-1].isdigit() or sib == out_path:
            continue                          # skips -wal / -shm sidecars and our own DB
        try:
            cx = sqlite3.connect("file:%s?mode=ro" % sib, uri=True)
            done.update(r[0] for r in cx.execute("SELECT ref FROM sig"))
            cx.close()
        except Exception as e:
            # Reported rather than swallowed. An unreadable sibling means its work may be redone,
            # which is safe; treating it as empty without saying so would hide a broken shard.
            print("  warn: sibling %s unreadable (%s) — its refs will be recomputed"
                  % (os.path.basename(sib), e))
    if done:
        print("resuming: %d signature(s) already computed (%d ours, %d from siblings)"
              % (len(done), mine_n, len(done) - mine_n))

    files = sorted(glob.glob(os.path.join(args.local_corpus, "blobs-*.sqlite")))
    # Each worker takes whole shard files. The blob shards are already a partition of the ref
    # space (`int(ref[4:6],16) % 16`), so this needs no hashing and guarantees disjointness.
    mine = [f for i, f in enumerate(files) if i % args.shards == args.shard]
    print("worker %d/%d: %d blob file(s)" % (args.shard, args.shards, len(mine)))
    # A worker with no files exits non-zero and names the real ceiling. Whole files are the unit of
    # work, so a `--shards` greater than the file count leaves the tail workers with nothing, and
    # they would otherwise print the same "DONE scanned=0 signed=0" line a finished worker prints —
    # no work done wearing the shape of work completed.
    if not mine:
        print("FATAL: worker %d of %d got 0 of %d blob file(s) — --shards exceeds the file count. "
              "Whole files are the unit of work, so at most %d workers can do anything. This is a "
              "refusal, not a completed run of zero work."
              % (args.shard, args.shards, len(files), len(files)), file=sys.stderr)
        return 2

    n = wrote = degen = 0
    t0, last = time.time(), 0.0
    batch = []
    for f in mine:
        cx = sqlite3.connect("file:%s?mode=ro" % f, uri=True)
        for ref, data in cx.execute("SELECT ref, data FROM blob"):
            n += 1
            if ref in done:
                continue
            try:
                text = data.decode("utf-8", "replace")
            except Exception:
                text = ""
            sig = M.signature(text)
            if M.is_degenerate(sig):
                # NULL signature plus degenerate=1. An all-zero signature would read as "similar
                # to everything" to any consumer that does not check the flag.
                batch.append((ref, None, len(data), 1))
                degen += 1
            else:
                blob = b"".join(v.to_bytes(8, "big") for v in sig)
                batch.append((ref, blob, len(data), 0))
                wrote += 1
            if len(batch) >= 2000:
                out.executemany("INSERT OR REPLACE INTO sig VALUES (?,?,?,?)", batch)
                out.commit()
                batch.clear()
            now = time.time()
            if now - last >= args.report:
                last = now
                el = now - t0
                print("  %8d scanned  %8d signed  %6d degenerate  %6.0f/s"
                      % (n, wrote, degen, n / max(el, 1e-9)))
            if args.limit and n >= args.limit:
                break
        cx.close()
        if args.limit and n >= args.limit:
            break
    if batch:
        out.executemany("INSERT OR REPLACE INTO sig VALUES (?,?,?,?)", batch)
        out.commit()

    el = time.time() - t0
    print()
    print("DONE  scanned=%d  signed=%d  degenerate=%d  (%.1fs, %.0f/s)"
          % (n, wrote, degen, el, n / max(el, 1e-9)))
    # Printed even when zero: a run silent about degeneracy looks the same as one that had none,
    # and degeneracy is what an all-zero signature would be built from.
    print("DEGENERATE: %d  (signature withheld, NOT stored as zeros)" % degen)
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
