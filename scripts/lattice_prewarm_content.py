#!/usr/bin/env python3
"""Pre-warm the file-content-cache from `local-corpus`, concurrently with the migration.

The pull writes decrypted plaintext into `local-corpus/blobs-NN.sqlite`. The pipeline's `content`
stage encrypts it at rest into the cache under each collection's key, but that stage sits behind
`vertices -> edges -> colimit -> consolidate -> creation`, ~90 minutes away. Doing the work now
turns `content` into mostly verification instead of ~6.11M encrypt-and-write operations.

Three properties make this safe to run against a live migration:

  * `corpus.db` is opened read-only (`mode=ro`). It reads `content_ref` and `collection_id` from
    rows the migration has already written, and writes no rows.
  * It writes only files, into the cas directory. No counters, no leaf digests, no `_seq`.
  * The `content` stage is resumable: present-and-verifying is a no-op, present-and-corrupt is
    evicted and refetched. So anything warmed here is verified and skipped later, and anything
    warmed wrong is caught and redone.

The key is per collection. `collection_key(root_secret, origin_root)` — P9.3 roots at the
collection's immutable origin root rather than at `created_by`/'owner', because rooting content
keys at `created_by` leaves every blob underivable and unauthenticatable once identities are folded
(§4.3.X). So the collection is read per row from `doc.collection_id`; a guessed collection produces
blobs the content stage later evicts.

Verify-on-read still happens: `cache.put` re-hashes the plaintext against the ref before storing,
exactly as the content stage does, so a corrupt local blob fails here in the same place it would
fail there. Only `mf.decrypt` is skipped, because `local-corpus` already holds plaintext —
`pull.py` decrypted and sha256-verified it on the way in.

    python3 lattice_prewarm_content.py --target work/corpus.db --cas work/cas \\
        --local-corpus /home/builder/genesis/local-corpus
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
import time


def hash_id(aid: str) -> int:
    """Stable partition key. `hash()` is salted per process in Python 3, so it would give each
    worker a different partition of the same rows — some refs done N times, others never."""
    import hashlib
    return int(hashlib.blake2b(str(aid).encode(), digest_size=8).hexdigest(), 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--cas", required=True)
    ap.add_argument("--local-corpus", default="/home/builder/genesis/local-corpus")
    ap.add_argument("--keys-dir", default="/home/builder/genesis/keys")
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--report", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    # ── fan-out. AES-256-GCM is CPU-bound and per-blob independent, so workers over disjoint id
    # ranges touch disjoint refs and contend on no cache entry. Every worker opens `corpus.db`
    # read-only, so N workers add N readers and zero writers to a live migration.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    import hashlib
    from mantle.db.content_cache import (CacheCorrupt, CacheMiss,   # noqa: F401
                                                 FileContentCache, collection_key,
                                                 shared_content_key)

    os.makedirs(args.cas, exist_ok=True)

    # ── the root secret, exactly as the content stage derives it ─────────────────────────────
    primary = os.path.join(args.keys_dir, "content.key")
    if not os.path.exists(primary):
        print("FATAL: no %s — refusing to invent a key." % primary, file=sys.stderr)
        return 2
    with open(primary, "rb") as fh:
        root_secret = hashlib.blake2b(fh.read().strip(), digest_size=32).digest()

    con = sqlite3.connect("file:%s?mode=ro" % args.target, uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=600000")

    # ── each collection's immutable origin root (P9.3) ───────────────────────────────────────
    roots = {}
    for r in con.execute("SELECT id, doc FROM vertex WHERE ct = ?",
                         ("application/vnd.agience.collection+json",)):
        d = json.loads(r["doc"])
        if d.get("origin_root"):
            roots[str(r["id"])] = str(d["origin_root"])
    # An empty `roots` map is fine (mantle §1): the at-rest key is one node-wide key that does not
    # depend on the collection, so there is nothing left to scope. The map is read because it is
    # the decrypt fallback for objects written under the earlier per-collection scheme; those open
    # through it and are rewritten under the shared key as they are read.
    if roots:
        print("legacy collections (decrypt-only fallback): %s"
              % ", ".join("%s->%s..." % (k, v[:8]) for k, v in sorted(roots.items())))
    else:
        print("no collection origin_roots — nothing to migrate, writing under the shared key")

    cache = FileContentCache(
        args.cas, key=shared_content_key(root_secret),
        legacy_key_for_collection=(lambda c: collection_key(root_secret, roots[c])) if roots
        else None)

    # ── local blob shards, opened once (README: reopening per lookup is the wrong shape) ─────
    shards = {}

    def local(ref):
        try:
            s = int(ref[4:6], 16) % 16
        except Exception:
            return None
        cx = shards.get(s)
        if cx is None:
            p = os.path.join(args.local_corpus, "blobs-%02d.sqlite" % s)
            if not os.path.exists(p):
                return None
            cx = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
            shards[s] = cx
        row = cx.execute("SELECT data FROM blob WHERE ref = ?", (ref,)).fetchone()
        return row[0] if row else None

    if args.shards > 1:
        print("worker %d of %d" % (args.shard, args.shards))
    warmed = cached = missing = failed = scanned = 0
    unknown_coll = collections.Counter()
    cursor, t0, last = "", time.time(), 0.0

    while True:
        rows = con.execute(
            "SELECT id, content_ref, doc FROM vertex "
            "WHERE id > ? AND content_ref IS NOT NULL ORDER BY id LIMIT ?",
            (cursor, args.chunk)).fetchall()
        if not rows:
            break
        for r in rows:
            cursor = r["id"]
            # Partition by a stable hash of the id rather than by row position: positions shift as
            # the migration writes, which would hand the same ref to two workers.
            if args.shards > 1 and (hash_id(r["id"]) % args.shards) != args.shard:
                continue
            scanned += 1
            ref = r["content_ref"]
            coll = json.loads(r["doc"]).get("collection_id")
            if coll not in roots:
                # The collection has no default here. A wrong key writes a blob the content stage
                # later evicts as corrupt — wasted work that looks like progress.
                unknown_coll[str(coll)] += 1
                failed += 1
                continue
            if ref in cache:
                cached += 1
                continue
            plain = local(ref)
            if plain is None:
                missing += 1          # the pull has not reached it yet; `content` will fetch it
                continue
            try:
                cache.put(ref, plain, collection=coll, tier="prewarm")
                warmed += 1
            except CacheCorrupt:
                # sha256(plaintext) != ref. Fails here exactly as it would in the content stage.
                failed += 1
            except Exception:
                failed += 1
        now = time.time()
        if now - last >= args.report:
            last = now
            el = now - t0
            print("  %8d scanned  %8d warmed  %8d already  %7d not-yet-pulled  %6d failed  "
                  "%6.0f/s" % (scanned, warmed, cached, missing, failed, scanned / max(el, 1e-9)))
        if args.limit and scanned >= args.limit:
            break

    el = time.time() - t0
    print()
    print("=" * 78)
    print("DONE  scanned=%d  warmed=%d  already_cached=%d  not_yet_pulled=%d  failed=%d  (%.1fs)"
          % (scanned, warmed, cached, missing, failed, el))
    # Printed even when zero: a run that says nothing about failures looks the same as one that
    # had none.
    print("FAILED: %d" % failed)
    if unknown_coll:
        print("rows whose collection has no origin_root (NOT keyed, left for `content`):")
        for k, n in unknown_coll.most_common(8):
            print("   %-40s %d" % (str(k)[:40], n))
    print("cache report: %s" % json.dumps(cache.report(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
