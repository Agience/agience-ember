#!/usr/bin/env python3
"""B2a — generate the offer for every artifact whose required fields are already present.

An offer is what an artifact advertises: the `context` a query resonates against. This pass fills
it by projection from fields the row already carries — never by composition, never by a model.

This runs without the content pass. Measured on node 45 (stratified over 4 leaf buckets, n=2,798
`text/markdown`): `title` 100%, `lemmas` 100%, `content_ref` 100%, `context` 0%. And
`type.text/markdown`'s `context_schema.required` is `["kind", "title"]` — `summary` is optional.
So the offer's required fields are fully available with no CAS fetch, and 6.11M rows can go from
0% to ~100% `context` without waiting on the 24.5 GB content pull.

B2b (`summary` + `minhash`) needs the content and is one pass over it. This script touches no
content, so the cheap half does not wait for the expensive one.

## Rows with no offer are reported by reason, never counted as zero

Some rows yield no offer — measured: `mem-b9a46f052ccd55e5` is `text/markdown` with no title. It
is a different data type wearing the same format (`text/markdown` is an encoding; the data has its
own type), rather than a row missing a field. That is a finding about type resolution, so this
script prints every distinct reason with counts. A run reporting "0 offers written" and a run
reporting "N with no offer, here is why" are different facts.

## Idempotent and resumable

Keyset walk over `id`, never OFFSET. A row whose `context` already equals what we would write is
skipped, so a re-run converges to `0 written` rather than churning `_seq`.

    python3 lattice_offers.py --target work/corpus.db            # write
    python3 lattice_offers.py --target work/corpus.db --dry-run  # measure only
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _load_offer_module():
    """Load the offer renderer by path from `agience-chorus/src/sage/offer.py`.

    The renderer lives in chorus: it renders what an artifact advertises, and sage owns
    `op.describe.*`.

    Loading by path rather than `import sage.offer`. Ember does not import chorus (enforced by
    `tests/test_defect_closure_wiring.py` and `tests/test_reach_wiring.py`), and chorus is not an
    installed distribution here — `sage/` has no `__init__.py`. This script reaches
    `agience-mantle/src` the same way, a few lines below. The module depends only on `re` and
    `typing`, which is what makes loading it in isolation work.

    If the file is in none of the candidate places the script stops. Offers are a projection from
    the type artifact, so a run that could not load the renderer and reported "0 offers written"
    would look identical to a clean run over an undescribed corpus.
    """
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    # The repo layout and node 45's deploy layout differ, so both are tried. On 45 the source
    # ships flattened; a path that resolves in only one of the two places this script runs is a
    # latent break either way.
    cands = [os.path.join(here, "..", "..", "agience-chorus", "src", "sage", "offer.py"),  # repo
             "/home/builder/genesis/chorus-src/sage/offer.py",                    # 45 deploy
             os.path.join(here, "..", "..", "chorus-src", "sage", "offer.py")]
    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
        raise SystemExit(
            "FATAL: sage/offer.py not found in any of: " + "; ".join(cands) +
            "  (it MOVED out of ember on 2026-07-29 — this script needs the agience-chorus "
            "checkout beside agience-ember)")
    spec = importlib.util.spec_from_file_location("offer", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="measure coverage and refusals; write nothing")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (smoke)")
    args = ap.parse_args()

    O = _load_offer_module()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "agience-mantle", "src"))
    from mantle.db.vertex import LatticeArtifactStore

    store = LatticeArtifactStore(args.target, origin=os.environ.get("EMBER_NODE_ID", "45"))
    con = store.db.read()
    con.execute("PRAGMA busy_timeout=600000")

    # ── the type artifacts: one keyed read, held for the whole pass ──────────────────────────
    types: Dict[str, Dict[str, Any]] = {}
    for r in con.execute("SELECT doc FROM vertex WHERE ct = ?",
                         ("application/vnd.agience.content-type+json",)):
        d = json.loads(r["doc"])
        if d.get("declares"):
            types[str(d["declares"])] = d
    if not types:
        print("FATAL: no content-type artifacts in the store. Run the `typedefs` stage first — "
              "without them there is no offer_template and no schema, so nothing can be "
              "described. Refusing to guess a shape.", file=sys.stderr)
        return 2
    print("type artifacts: %d (%s)" % (len(types), ", ".join(sorted(types)[:4]) + " ..."))

    written = skipped = refused = scanned = 0
    reasons: collections.Counter = collections.Counter()
    no_type: collections.Counter = collections.Counter()
    examples: Dict[str, str] = {}
    cursor, t0 = "", time.time()

    while True:
        rows = con.execute(
            "SELECT id, ct, doc FROM vertex WHERE id > ? ORDER BY id LIMIT ?",
            (cursor, args.chunk)).fetchall()
        if not rows:
            break
        batch = []
        for r in rows:
            cursor = r["id"]
            scanned += 1
            doc = json.loads(r["doc"])

            # Format is not type: prefer a resolved data type, fall back to the content type.
            key = O.describer_key(doc)
            ta = types.get(key)
            if ta is None:
                no_type[key] += 1
                refused += 1
                continue

            text, err = O.offer_or_none(ta, doc)
            if err is not None:
                refused += 1
                reasons["%s: missing %s" % (key, ",".join(err.missing))] += 1
                examples.setdefault("%s: missing %s" % (key, ",".join(err.missing)), str(r["id"]))
                continue
            if doc.get("context") == text:
                skipped += 1
                continue
            doc["context"] = text
            batch.append((r["id"], json.dumps(doc)))

        if batch and not args.dry_run:
            # Direct doc UPDATE with no `_seq` bump — 10-50x faster than put_artifact, and the
            # right shape. An offer is derived metadata (a projection of existing fields into
            # `context`), not a new version of the artifact. `put_artifact(stamp_rev=True)`
            # allocates a fresh `_seq`, re-XORs the merkle leaf and re-indexes listkey for every
            # row; on 6.11M wn rows with many lemmas the listkey rebuild dominates and the pass
            # runs for 30+ minutes with no visible progress.
            #
            # Raw SQL is safe here because everything the store maintains inside its write path is
            # unaffected by a context-only change:
            #   · merkle: `row_hash = blake2b(id + NUL + str(_seq))` (constants.py) covers (id,seq)
            #     only, never doc content. `_seq` is unchanged, so every leaf digest is unchanged
            #     and node-repair's merkle check still passes. Verified against node-repair.
            #   · counters: count rows and specific fields; `context` is neither, and no row is added.
            #   · listkey: indexes named fields (lemma, title, ...); `context` is not one.
            # So there is nothing to backfill — the write changes only the JSON the store does not
            # track. Leaving `_seq` alone keeps an offer from minting 6.11M versions that differ
            # only in a derived field.
            with store.db.write() as cur:
                cur.executemany("UPDATE vertex SET doc = ? WHERE id = ?",
                                [(d, aid) for aid, d in batch])
            written += len(batch)
        elif batch:
            written += len(batch)          # dry-run: count what WOULD be written

        if scanned % 100000 < args.chunk:
            el = time.time() - t0
            print("  %8d scanned  %8d offered  %7d skipped  %7d refused  %6.0f rows/s"
                  % (scanned, written, skipped, refused, scanned / max(el, 1e-9)))
        if args.limit and scanned >= args.limit:
            break

    el = time.time() - t0
    print()
    print("=" * 78)
    print("%s  scanned=%d  offered=%d  unchanged=%d  refused=%d  (%.1fs)"
          % ("DRY RUN" if args.dry_run else "DONE", scanned, written, skipped, refused, el))

    # The count is printed even when zero: a pass that says nothing about rows without an offer
    # looks the same as one that had none.
    print("REFUSED: %d" % refused)
    for why, n in reasons.most_common(12):
        print("   %-52s %8d   e.g. %s" % (why[:52], n, examples.get(why, "")))
    if no_type:
        print("NO TYPE ARTIFACT for these keys (format/type unresolved):")
        for k, n in no_type.most_common(8):
            print("   %-52s %8d" % (str(k)[:52], n))
    if not reasons and not no_type:
        print("   (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
