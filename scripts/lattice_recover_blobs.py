#!/usr/bin/env python3
"""Recover specific blobs into `local-corpus` from the durable origin. Hash-verified, per object.

This is a separate script rather than a pipeline stage. The `content` stage completes with the
remote sources unreachable (acceptance S3.8) — that is the decommission criterion. Recovery is the
opposite act: it reaches out. Keeping them apart means the pipeline run that populates the cache
stays purely local, and any remote fetch is an explicit, logged decision rather than a fallback
inside a stage.

Measured: `content` scanned all 6,114,746 refs and 15 did not resolve because `local-corpus` did
not hold them, while the pull reported "6,115,252 blobs, 100.000%, 0 failed". The pull's
denominator is the manifest, not the set the corpus needs.

## No boto3

Node 45 has no pip, no ensurepip, and boto3 nowhere on disk. `curl` 8.18 signs SigV4 natively
(`--aws-sigv4`), a published feature rather than hand-rolled request signing — and a hand-rolled
signer has to be right on the first try, with no way to tell when it is subtly wrong.

## The hash is checked before anything is written

`content_ref` is `cas/<sha256-of-plaintext>`, so every object is self-verifying: fetch, decrypt,
sha256 the plaintext, compare to the address it was requested by. A mismatch is reported and left
unstored. Writing unverified bytes into the corpus under a content address is worse than the gap it
would close, because every later reader trusts that address.

    python3 lattice_recover_blobs.py --refs /tmp/missing_refs.txt --dry-run
    python3 lattice_recover_blobs.py --refs /tmp/missing_refs.txt
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import subprocess
import sqlite3
import sys

# The object store this recovers from is the operator's, so it is named by the operator rather
# than defaulted here. A hardcoded endpoint and bucket published in a public repository is one
# deployment's infrastructure map, and it is also the wrong default for everyone else: a script
# that silently points at a bucket the caller does not own fails in a way that reads as "the blobs
# are gone" rather than "you are looking in the wrong place".
DEFAULT_ENDPOINT = os.environ.get("EMBER_OVH_ENDPOINT", "")
DEFAULT_BUCKET = os.environ.get("EMBER_OVH_BUCKET", "")
DEFAULT_REGION = os.environ.get("EMBER_OVH_REGION", "")


def shard_of(ref: str, shards: int = 16) -> int:
    """Which blob file holds a ref. The partition `pull.py` used: first byte of the sha256.

    Computed in Python rather than in SQL. SQLite's `CAST('0x1a' AS INTEGER)` is 0, not 26 — it
    does not parse hex — so shard arithmetic pushed into a query mis-assigns every ref while still
    returning a confident-looking answer.
    """
    return int(ref[4:6], 16) % shards


def build_fernet(keys_dir: str):
    """MultiFernet in contract order: `content.key` primary, every `content.key.*` decrypt-only."""
    from cryptography.fernet import Fernet, MultiFernet

    primary = os.path.join(keys_dir, "content.key")
    if not os.path.exists(primary):
        raise SystemExit("FATAL: no %s. A missing content key is a SILENT PARTITION — MultiFernet "
                         "simply fails to decrypt and the caller sees an empty string. Refusing "
                         "to run rather than write empty blobs." % primary)
    names = ["content.key"] + sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(keys_dir, "content.key.*")))
    keys, fps = [], []
    for n in names:
        raw = open(os.path.join(keys_dir, n), "rb").read().strip()
        keys.append(Fernet(raw))
        # Fingerprints, not key material.
        fps.append("%s=%s" % (n, hashlib.blake2b(raw, digest_size=8).hexdigest()))
    print("content keys (blake2b-8 fingerprints, never material): %s" % "; ".join(fps))
    return MultiFernet(keys)


def fetch(ref: str, *, endpoint: str, bucket: str, region: str, keys_dir: str,
          timeout: int = 60):
    """One GET via curl's native SigV4. Returns `(body, http_code)`.

    The HTTP code travels with the body rather than being collapsed into an empty result. 404 (the
    object does not exist), 403 (these credentials cannot see it) and 5xx (the origin is unwell)
    call for opposite responses — accept the loss, fix the grant, or retry later — and a single
    "absent at origin" reading would record a recoverable outage as permanent data loss."""
    ak = open(os.path.join(keys_dir, "ovh.access_key")).read().strip()
    sk = open(os.path.join(keys_dir, "ovh.secret_key")).read().strip()
    out = "/tmp/.recover.%d.bin" % os.getpid()
    cmd = ["curl", "-s", "-o", out, "-w", "%{http_code}", "--max-time", str(timeout),
           "--aws-sigv4", "aws:amz:%s:s3" % region, "--user", "%s:%s" % (ak, sk),
           "%s/%s/%s" % (endpoint.rstrip("/"), bucket, ref)]
    try:
        code = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        if code != "200":
            return b"", code
        with open(out, "rb") as fh:
            return fh.read(), code
    finally:
        if os.path.exists(out):
            os.remove(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="file with one content_ref per line")
    ap.add_argument("--local-corpus", default="/home/builder/genesis/local-corpus")
    ap.add_argument("--keys-dir", default="/home/builder/genesis/keys")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help="object-store endpoint (or $EMBER_OVH_ENDPOINT)")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET,
                    help="bucket holding the content blobs (or $EMBER_OVH_BUCKET)")
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--checked", help="ISO date recorded as WHEN absence was confirmed")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mark-absent", metavar="CORPUS_DB",
                    help="record `content_status=absent_at_origin` on artifacts whose ref "
                         "returned a CONFIRMED 404. Requires --target-store.")
    args = ap.parse_args()

    with open(args.refs) as fh:
        refs = [l.strip() for l in fh if l.strip()]
    if not refs:
        print("FATAL: %s lists no refs. Refusing to report a successful recovery of nothing."
              % args.refs, file=sys.stderr)
        return 2
    print("refs to recover: %d" % len(refs))

    bad = [r for r in refs if not r.startswith("cas/") or len(r) != 4 + 64]
    if bad:
        print("FATAL: %d ref(s) are not `cas/<64 hex>`: %s" % (len(bad), bad[:3]), file=sys.stderr)
        return 2

    mf = build_fernet(args.keys_dir)
    recovered = absent = mismatch = unreachable = 0
    still_missing = []
    confirmed_absent = []          # 404 only; a 403/5xx means "could not look"

    for ref in refs:
        want = ref[4:]
        ct, code = fetch(ref, endpoint=args.endpoint, bucket=args.bucket, region=args.region,
                         keys_dir=args.keys_dir)
        if not ct:
            still_missing.append(ref)
            if code == "404":
                absent += 1
                confirmed_absent.append(ref)
                print("  ABSENT AT ORIGIN (404 NoSuchKey)  %s" % ref)
            else:
                # Counted separately from absence. A 403/5xx says the origin could not be read,
                # which is a different fact from the object not existing, and only one is permanent.
                unreachable += 1
                print("  FETCH FAILED HTTP %-4s (NOT proof of absence)  %s" % (code, ref))
            continue
        try:
            plain = mf.decrypt(ct)
        except Exception as e:
            mismatch += 1
            still_missing.append(ref)
            print("  DECRYPT FAILED    %s (%s)" % (ref, type(e).__name__))
            continue
        got = hashlib.sha256(plain).hexdigest()
        if got != want:
            # Left unstored. A blob written under an address it does not hash to poisons every
            # later read of that address, and content-addressed storage has no way to notice.
            mismatch += 1
            still_missing.append(ref)
            print("  HASH MISMATCH     %s -> %s (REFUSED, not written)" % (ref, got[:16]))
            continue
        if args.dry_run:
            recovered += 1
            print("  [DRY RUN] would write %s (%d bytes, hash verified)" % (ref, len(plain)))
            continue
        shard = shard_of(ref)
        path = os.path.join(args.local_corpus, "blobs-%02d.sqlite" % shard)
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
        print("  recovered %s -> shard %02d (%d bytes)" % (ref, shard, len(plain)))

    print()
    print("=" * 70)
    print("%srecovered=%d  absent_at_origin_404=%d  unreachable_non404=%d  "
          "hash_or_decrypt_failed=%d"
          % ("DRY RUN " if args.dry_run else "", recovered, absent, unreachable, mismatch))
    if unreachable:
        print("⚠ %d ref(s) returned a non-404 error. That is NOT evidence of absence — do not "
              "record them as lost until they return 404 against working credentials."
              % unreachable)
    # Printed even at zero, with the refs named. "0 still missing" and "nothing was checked" read
    # differently, and a recovery that leaves objects behind says which ones.
    print("STILL MISSING: %d%s" % (len(still_missing),
                                   ("  " + " ".join(still_missing)) if still_missing else ""))

    # ── record the loss on the data, with its evidence ───────────────────────────────────────
    # Only confirmed 404s are marked. `confirmed_absent` excludes anything that merely failed to
    # fetch: marking a 403 or a 5xx as absent-at-origin would write "this content is gone" into
    # the corpus on the strength of an outage, and that claim outlives the outage.
    #
    # The marker goes on the artifact rather than into a migration-local allowlist because the
    # allowlist disappears with the migration, while a reader of this artifact in a year still
    # needs to learn that the content is unavailable rather than receiving silence.
    if args.mark_absent and confirmed_absent and not args.dry_run:
        import json as _json
        con = sqlite3.connect(args.mark_absent)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=600000")
        marked = 0
        q = ("SELECT id, doc FROM vertex WHERE content_ref IN (%s)"
             % ",".join("?" * len(confirmed_absent)))
        for r in con.execute(q, confirmed_absent).fetchall():
            d = _json.loads(r["doc"])
            if d.get("content_status") == "absent_at_origin":
                continue
            d["content_status"] = "absent_at_origin"
            d["content_absent_evidence"] = {
                "http": 404, "code": "NoSuchKey", "endpoint": args.endpoint,
                "bucket": args.bucket, "checked": args.checked or "unspecified",
                "note": "also absent from local-corpus and Garage",
            }
            con.execute("UPDATE vertex SET doc=? WHERE id=?",
                        (_json.dumps(d, sort_keys=True), r["id"]))
            marked += 1
        con.commit()
        con.close()
        print("marked %d artifact(s) content_status=absent_at_origin" % marked)
    return 0 if not still_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
