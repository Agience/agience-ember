"""Backfill the sense-level relations the OEWN parser used to drop. Additive, idempotent, seconds.

    python scripts/backfill_sense_relations.py --report        # count, write nothing
    python scripts/backfill_sense_relations.py                 # insert

## What it adds, and why this is not a rebuild

`parse_oewn_lmf` takes every SYNSET-level relation the source names, and until `_SENSE_RELATIONS_KEPT`
existed it filtered SENSE-level relations down to `antonym` alone. Two of the dropped ones are what
place an adjective or an adverb, neither of which has a hypernym parent:

    derivation   74,646    the noun a modifier derives from — `beautiful` -> `beauty`
    pertainym     8,072    an adverb to its adjective — `quickly` -> `quick`

82,718 edges. They are pure INSERTS over the `edge` table, and that is what makes this cheap:

  * **No artifact is written, so no `_rev` is stamped.** This is the trap `backfill_leaf.py`
    documents at length: `put_artifact` mints a fresh `_rev`, which pushes every touched row into
    the change feed and re-ships the corpus to every peer. Fleet-wide write amplification to add a
    local edge. This script only ever calls `store.graph.add_edges`.
  * **No information content is recomputed.** IC is a property of the hypernym tree and none of
    these labels is a hypernym. `jc_tree` still travels IS-A only; these edges are read by the
    projection in `crystal.ontology.lookup`, and by `related`, which excludes IS-A precisely so the
    tree is not counted twice.
  * **Neither search arm is reindexed.** Blind tokens and FTS terms are over text, not over edges.

## Idempotence, and why there is no checkpoint

`add_edges` is an upsert on `(src, dst, label)`, so re-running converges rather than duplicating.
That is also why this needs none of `backfill_leaf.py`'s resume machinery: that script walked the
whole corpus with an UPDATE per row and had to be resumable because it could not afford to start
over. This one derives its whole input from a local file in one pass and writes in batches, so
starting over IS the recovery. Being interruptible without a checkpoint is a property of being
additive, not an omission.

It still batches, for the reason that script gives: a maintenance job with no brakes competes with
production for the one resource neither can do without. `--batch` and `--pause` are there for a
loaded node.
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from collections import Counter
from pathlib import Path

#: The labels this script is responsible for. Deliberately not `_SENSE_RELATIONS_KEPT` itself:
#: `antonym` was always written, so a store that has been ingested already holds it, and including
#: it here would report thousands of "added" edges that were only ever re-upserted. This is the
#: DIFFERENCE between the old parser and the new one, which is what a backfill is.
BACKFILLED = ("derivation", "pertainym")


def _source_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    from ember.corpus.stage0_sources import _downloads_dir
    return _downloads_dir() / "english-wordnet-2024.xml.gz"


def _open_maybe_gz(path: Path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true",
                    help="parse and count, write nothing")
    ap.add_argument("--source", default=None,
                    help="path to english-wordnet-2024.xml(.gz); defaults to the download cache")
    ap.add_argument("--batch", type=int, default=2000,
                    help="edges per add_edges call (default 2000)")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to sleep between batches; raise it on a loaded node")
    args = ap.parse_args(argv)

    path = _source_path(args.source)
    if not path.is_file():
        print("source not found: %s\n"
              "Run the OEWN ingest once, or pass --source. This script does not download."
              % path, file=sys.stderr)
        return 2

    from ember.corpus.stage0_sources import parse_oewn_lmf

    t0 = time.time()
    with _open_maybe_gz(path) as fh:
        parsed = parse_oewn_lmf(fh)
    wanted = [(a, b, lbl) for a, b, lbl in parsed["relations"] if lbl in BACKFILLED]
    counts = Counter(lbl for _a, _b, lbl in wanted)
    print("parsed %s in %.1fs" % (path.name, time.time() - t0))
    for lbl in BACKFILLED:
        print("   %-12s %8d" % (lbl, counts.get(lbl, 0)))
    print("   %-12s %8d" % ("TOTAL", len(wanted)))

    if args.report:
        print("\n--report: nothing written.")
        return 0

    if not wanted:
        print("\nnothing to add — the source names none of %s" % (BACKFILLED,))
        return 0

    # Resolved lazily: `--report` parses the source and needs no store at all.
    from mantle.shard.local_store import open_store
    store = open_store()
    from ember.corpus import genesis as g

    # An edge whose endpoints are not both stored would dangle, and the synset pass applied the same
    # rule ("only where BOTH endpoints were stored"). Asking the store is what keeps this honest on a
    # partially-ingested corpus rather than inserting edges into nothing.
    t1 = time.time()
    added = skipped = 0
    batch: list = []

    def _flush():
        nonlocal added, batch
        if not batch:
            return
        added += store.graph.add_edges(iter(batch), batch=len(batch))
        batch = []
        if args.pause:
            time.sleep(args.pause)

    for src, dst, lbl in wanted:
        a, b = "wn-" + src, "wn-" + dst
        if not (_stored(store, a) and _stored(store, b)):
            skipped += 1
            continue
        batch.append((a, b, lbl, {"via": "op.source.oewn", "rung": g.P_OBSERVED}))
        if len(batch) >= args.batch:
            _flush()
    _flush()

    print("\nadded %d edges in %.1fs (%d skipped: an endpoint is not in this store)"
          % (added, time.time() - t1, skipped))
    print("Re-running is safe: `add_edges` upserts on (src, dst, label).")
    return 0


_STORED_CACHE: dict = {}


def _stored(store, artifact_id: str) -> bool:
    """Is this synset artifact present? Cached, because the two endpoints of 82,718 edges are drawn
    from a set of ~118,000 ids and every one recurs."""
    hit = _STORED_CACHE.get(artifact_id)
    if hit is None:
        try:
            hit = store.artifacts.get_artifact(artifact_id) is not None
        except Exception:
            hit = False
        _STORED_CACHE[artifact_id] = hit
    return hit


if __name__ == "__main__":
    raise SystemExit(main())
