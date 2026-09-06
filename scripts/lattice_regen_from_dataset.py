#!/usr/bin/env python3
"""Regenerate missing blobs from the pinned source dataset, reproducing the original bytes exactly.

This recovers rather than replaces. `wiki-simple-*` was ingested from `wikimedia/wikipedia` config
`20231101.simple` — a frozen snapshot, not live Wikipedia. So re-deriving a row yields the same
bytes, the same sha256, and satisfies the same `content_ref` already recorded on the artifact.
Nothing is versioned, nothing is rewritten, and provenance is unchanged: this is recovery of the
identical object, not a new observation of a changed one.

For example, `wiki-simple-178` ("Cuba") re-derives to
`a05293154dd0ad1e06ae65f6e8243033dace1468242a209883f4c49c7fd037b0` — exactly the missing ref.

The pinned source is what makes this work. Against a live source the bytes drift, the hash differs,
and the correct handling is a new version under the same `root_id` (§6B) rather than a rewrite of
the old one. A config changed to an unpinned or moving snapshot puts this script out of scope.

## The recipe is reproduced verbatim from the ingest

`genesis.ingest_stage1_simplewiki._to_record`:

    title = (row["title"] or "").strip()
    text  = (row["text"]  or "").strip()
    body  = f"{title}\\n\\n{text}" if title else text

Any deviation — a different separator, an unstripped field, a normalised newline — changes the
hash, and the hash is the whole verification. The sha256 check below proves the recipe was
reproduced, so a mistake here surfaces as a mismatch rather than as a plausible-looking body.

    python3 lattice_regen_from_dataset.py --idref /tmp/idref.txt --dry-run
    python3 lattice_regen_from_dataset.py --idref /tmp/idref.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.simple"          # Pinned. See the module docstring before changing this.
SPLIT = "train"
API = "https://datasets-server.huggingface.co/filter"
ID_PREFIX = "wiki-simple-"


def shard_of(ref: str, shards: int = 16) -> int:
    """Computed in Python: SQLite's `CAST('0x1a' AS INTEGER)` is 0, not 26 — it does not parse hex."""
    return int(ref[4:6], 16) % shards


def fetch_row(row_id: str, *, retries: int = 5, timeout: int = 60):
    """One row from the pinned dataset, by its own `id` column.

    Retried, because the datasets-server returns intermittent 500s. Treating a transient 500 as
    "row not found" reports a recoverable object as unrecoverable. A row is absent only when the
    API answers successfully with no rows.
    """
    q = urllib.parse.urlencode({"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                                "where": '"id"=\'%s\'' % row_id, "limit": "1"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(API + "?" + q, timeout=timeout) as r:
                d = json.load(r)
            rows = d.get("rows") or []
            return (rows[0]["row"] if rows else None), None
        except Exception as e:                      # noqa: BLE001 - transport/HTTP both retried
            last = e
            time.sleep(2 * (attempt + 1))
    return None, last                                # could not look; not proof of absence


def body_of(row) -> str:
    """The ingest recipe, verbatim. Reproduced exactly or the hash will not match."""
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return ("%s\n\n%s" % (title, text)) if title else text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idref", required=True,
                    help="file of '<artifact_id> <content_ref>' pairs, one per line")
    ap.add_argument("--local-corpus", default="/home/builder/genesis/local-corpus")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs = []
    for line in open(args.idref):
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    if not pairs:
        print("FATAL: %s lists no '<id> <ref>' pairs. Refusing to report a successful "
              "regeneration of nothing." % args.idref, file=sys.stderr)
        return 2
    print("blobs to regenerate: %d   (source: %s config=%s PINNED)" % (len(pairs), DATASET, CONFIG))

    recovered = mismatch = absent = unreachable = 0
    failures = []

    for aid, ref in pairs:
        if not aid.startswith(ID_PREFIX):
            print("  SKIP (not a %s artifact): %s" % (ID_PREFIX, aid))
            failures.append(ref)
            continue
        row_id = aid[len(ID_PREFIX):]
        row, err = fetch_row(row_id)
        if row is None:
            failures.append(ref)
            if err is None:
                absent += 1
                print("  NOT IN DATASET      %s (row id %s)" % (aid, row_id))
            else:
                unreachable += 1
                print("  API UNREACHABLE     %s (%s) — NOT proof of absence"
                      % (aid, type(err).__name__))
            continue
        body = body_of(row)
        plain = body.encode("utf-8")
        got = hashlib.sha256(plain).hexdigest()
        want = ref[4:]
        if got != want:
            # Left unwritten. A body stored under an address it does not hash to poisons every
            # later read of that address. A mismatch here means the recipe drifted or the snapshot
            # moved; both are findings to investigate.
            mismatch += 1
            failures.append(ref)
            print("  HASH MISMATCH       %s  want %s got %s (REFUSED)"
                  % (aid, want[:16], got[:16]))
            continue
        if args.dry_run:
            recovered += 1
            print("  [DRY RUN] %s -> %s (%d bytes, hash VERIFIED)" % (aid, want[:16], len(plain)))
            continue
        path = os.path.join(args.local_corpus, "blobs-%02d.sqlite" % shard_of(ref))
        cx = sqlite3.connect(path)
        # The schema is the store's own, read from it rather than assumed. `blob` is
        #     (ref TEXT PRIMARY KEY, data BLOB NOT NULL, nbytes INTEGER NOT NULL) WITHOUT ROWID
        # and an INSERT omitting `nbytes` fails its NOT NULL. The CREATE-IF-NOT-EXISTS below is a
        # no-op against an existing table and reconciles nothing: it is a fallback for a fresh
        # store, not a description of an existing one.
        cx.execute("CREATE TABLE IF NOT EXISTS blob (ref TEXT PRIMARY KEY, data BLOB NOT NULL, "
                   "nbytes INTEGER NOT NULL) WITHOUT ROWID")
        cx.execute("INSERT OR REPLACE INTO blob(ref, data, nbytes) VALUES(?,?,?)",
                   (ref, plain, len(plain)))
        cx.commit()
        cx.close()
        recovered += 1
        print("  recovered %s -> shard %02d (%d bytes, hash verified)"
              % (aid, shard_of(ref), len(plain)))

    print()
    print("=" * 70)
    print("%srecovered=%d  not_in_dataset=%d  api_unreachable=%d  hash_mismatch=%d"
          % ("DRY RUN " if args.dry_run else "", recovered, absent, unreachable, mismatch))
    # Named rather than counted: recovering an object requires its address.
    print("STILL MISSING: %d%s" % (len(failures), ("  " + " ".join(failures)) if failures else ""))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
