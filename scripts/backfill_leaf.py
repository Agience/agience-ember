"""Backfill `_leaf` — throttled, resumable, and self-aborting. Local only.

Why this is not a loop over the corpus
An unindexed full-scan UPDATE loop, run as fast as possible against a store that is concurrently
building the `_leaf` index, saturates ArcadeDB's IO layer: measured, the node stalled for 13
minutes and the aggregator stalled with it. Nothing about the SQL itself is at fault — the failure
mode is a maintenance job with no budget and no brakes competing with production for the one
resource neither can do without.

So this version is built from four rules, each of which is load-bearing:

  (a) Bounded and resumable. It processes at most `--rows` per run and checkpoints its @rid position
      durably, so finishing the backfill is a sequence of small, individually harmless runs rather
      than one long one that must be babysat. A run that is killed mid-way loses at most one batch.
  (b) It sleeps. `--pause` between batches yields IO back to the aggregator. The backfill has no
      deadline; the node does. Being slow is free, and being fast is what strains the node.
  (c) It aborts on its own. Before each checkpoint it asks whether the local aggregator is falling
      behind (mesh lag, and its own batch latency). If it is, the script stops rather than
      throttling harder and hoping — a maintenance job has to be able to notice it is the problem.
  (d) It never touches `_rev`. `UPDATE ... SET _leaf`, never put_artifact. put_artifact stamps a
      fresh `_rev`, which would push every backfilled row into the `_rev` UPDATE change-feed and
      re-ship the whole corpus to every peer — fleet-wide write amplification to fix a local
      column. This is the same trap that made @rid, not `_rev`, the scan key in the first place.

It also skips `_OP_EXCLUDE` content types: those keep a NULL `_leaf` permanently, which is the
mesh's definition of "not replicated", so stamping them would reintroduce the defect the column
was rescoped to fix.

  python scripts/backfill_leaf.py --rows 50000 --pause 0.25    # one polite chunk
  python scripts/backfill_leaf.py --status                     # coverage + where the cursor sits
  python scripts/backfill_leaf.py --reset                      # start the sweep over
"""
import argparse
import sys
import time

from mantle.shard.local_store import open_store
from mantle.mesh import merkle, sync

CURSOR_ID = "backfill.leaf.cursor"
# Per-box operational state, so it is excluded from every feed and never replicates. A backfill
# checkpoint is meaningless on any other node — resuming from a PEER's position would skip rows.
CURSOR_CT = "application/vnd.agience.mesh-cursor+json"


def _load(A):
    return (A.get_artifact(CURSOR_ID) or {}).get("rid") or "#-1:-1"


def _save(A, rid, done, note=""):
    A.put_artifact({"id": CURSOR_ID, "content_type": CURSOR_CT, "state": "committed",
                    "rid": rid, "rows_done": done, "note": note, "ts": time.time()},
                   stamp_rev=False)


def _lag_ok(b, max_behind):
    """Is the aggregator keeping up? Returns (ok, why). Unknown/no-S3 counts as OK: this guard
    exists to stop the backfill from making a bad situation worse, not to require a healthy mesh."""
    try:
        lag = sync.mesh_lag(b)
    except Exception:
        return True, ""
    if lag.get("reason") == "no-s3":
        return True, ""
    if lag.get("stuck"):
        return False, "a peer stream is stuck: %s" % (lag["stuck"][0],)
    behind = int(lag.get("segments_behind", 0) or 0)
    if behind > max_behind:
        return False, "aggregator is %d segments behind (limit %d)" % (behind, max_behind)
    return True, ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=50000, help="max rows to stamp this run")
    p.add_argument("--batch", type=int, default=200, help="rows per UPDATE transaction")
    p.add_argument("--pause", type=float, default=0.25, help="seconds to sleep between batches")
    p.add_argument("--budget", type=float, default=600.0, help="wall-clock seconds for this run")
    p.add_argument("--max-behind", type=int, default=25, help="abort if the mesh is further behind")
    p.add_argument("--max-batch-secs", type=float, default=5.0,
                   help="abort if a single batch takes longer than this (the store is under strain)")
    p.add_argument("--status", action="store_true")
    p.add_argument("--reset", action="store_true")
    a = p.parse_args()

    b = open_store()
    A = b.artifacts

    if a.reset:
        _save(A, "#-1:-1", 0, "reset")
        print("cursor reset", flush=True)
        return 0
    if a.status:
        print("coverage:", sync.merkle_coverage(b), flush=True)
        print("cursor:  ", A.get_artifact(CURSOR_ID) or {}, flush=True)
        return 0

    ok, why = _lag_ok(b, a.max_behind)
    if not ok:
        print("REFUSING TO START: %s" % why, flush=True)
        return 2

    rid = _load(A)
    done = int((A.get_artifact(CURSOR_ID) or {}).get("rows_done", 0) or 0)
    t0, stamped, scanned, stop = time.time(), 0, 0, ""

    while stamped < a.rows and not stop:
        if time.time() - t0 >= a.budget:
            stop = "budget"
            break
        # Keyset on @rid — the same key `_scan_rows` uses, and for the same reasons (always present,
        # ordered, no O(offset) SKIP, no dependence on the `_rev` this script must not write).
        try:
            rows = A.c.query(
                f"SELECT @rid AS _r, id, content_type, _leaf FROM Artifact "
                f"WHERE @rid > {rid} ORDER BY @rid LIMIT {int(a.batch)}")
        except Exception as e:
            print("scan failed: %s" % str(e)[:160], flush=True)
            stop = "scan-error"
            break
        if not rows:
            stop = "corpus-exhausted"
            break
        scanned += len(rows)
        nxt = rows[-1].get("_r")

        todo = [(r.get("id"), merkle.leaf_of(r["id"])) for r in rows
                if r.get("id") and r.get("_leaf") is None
                and r.get("content_type") not in sync._OP_EXCLUDE]
        t_batch = time.time()
        if todo:
            # UPDATE ... SET — rule (d). Never CONTENT, never put_artifact: those rewrite the whole
            # record and would take `_rev` with them.
            ops = [("UPDATE Artifact SET _leaf = %d WHERE id = :id" % li, {"id": aid})
                   for aid, li in todo]
            try:
                A.c.run_txn(ops)                    # one HTTP round-trip per batch
                stamped += len(todo)
            except Exception as e:
                print("batch failed at %s: %s" % (rid, str(e)[:160]), flush=True)
                stop = "batch-error"                # cursor is NOT advanced -> this batch is retried
                break
        batch_secs = time.time() - t_batch

        rid = nxt or rid
        done += len(todo)
        _save(A, rid, done, "running")

        # (c) Self-abort. A batch that suddenly takes seconds means the store is straining, and the
        # backfill is the only optional workload in the room. The checkpoint makes resuming free,
        # so stopping costs nothing.
        if batch_secs > a.max_batch_secs:
            stop = "store-under-strain (batch took %.1fs)" % batch_secs
            break
        if not nxt:
            stop = "no-progress"
            break
        if (scanned // max(1, a.batch)) % 20 == 0:
            ok, why = _lag_ok(b, a.max_behind)
            if not ok:
                stop = why
                break
        time.sleep(a.pause)                         # (b) yield IO — the whole point

    dt = time.time() - t0
    _save(A, rid, done, stop or "run-complete")
    print("stamped %d rows (scanned %d) in %.0fs — stopped: %s" %
          (stamped, scanned, dt, stop or "row budget reached"), flush=True)
    print("cursor:  ", rid, flush=True)
    print("coverage:", sync.merkle_coverage(b), flush=True)
    if stop not in ("", "corpus-exhausted", "budget"):
        print("NOTE: run again to resume from the cursor above.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
