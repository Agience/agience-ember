"""Decoupled stats: a refresher computes the heavy full-corpus metrics once and writes them to a
small on-disk snapshot (stats.json); every read path (/, /status, /health, the mesh view) reads the
snapshot, so no page blocks on a scan that grows O(corpus).

The snapshot is as old as the last refresh — "never far behind real state" in exchange for "exact
but 60s to render". Writes are atomic (temp + rename) so a reader never sees a half-written file.
Per box: <keys_dir>/../stats.json. The mesh reads each peer's snapshot over a cheap /stats endpoint,
so aggregating the whole universe is N tiny file reads rather than N scans.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

#: Without a module logger here, an outage like this stays invisible: `MESH_STATS_PREFIX` was
#: deleted as collateral, both uses raised `NameError`, and the handler below had nowhere to say
#: so — the mesh-stats outage this exposed lasted 26 days. A swallow with no logger is not
#: fail-soft, it is fail-invisible.
logger = logging.getLogger(__name__)


def stats_path(keys_dir) -> Path:
    return Path(keys_dir).parent / "stats.json"


def tail_jsonl(path, last: int) -> list:
    """The last `last` parsed records of a jsonl file — bad lines skipped, `[]` when absent.

    The one home for the tail-of-a-jsonl-log read, shared by `genesis.metrics_trend`, genesis's
    worker-log status and `improve.trend`. It reads the whole file, which suits the bounded
    trend/log files it serves. For a file that can grow unbounded, use a seek-from-end tail like
    `_activity`'s 64KB read below: `read_text()` on a 500MB log allocates the whole log."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines()[-int(last):]:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


# ── Store capability, not store type ──────────────────────────────────────────────────────────
# Every count and fetch in this module goes through a typed store method (LATTICE-CONTRACT §5).
# A query layer whose answer to "I did not understand you" is a value rather than an error publishes
# wrong numbers instead of empty ones — a `count(*)` that falls through to a bare-total branch
# reports the whole corpus as the mesh backlog, and a `SELECT` that matches no pattern returns `[]`,
# which then satisfies `len([]) <= cap` and serves as authoritative. A typed method has no such
# fallthrough: a renamed method is an AttributeError at the call site.




# ── The allowlist ─────────────────────────────────────────────────────────────────────────────
# The question this asks is "is this connection known to execute SQL faithfully?", which is an
# allowlist and fails closed. A new backend is untrusted until it is named here, and the cost of
# leaving it unnamed is "not measured" — an em-dash, this module's convention for an unmeasured
# value — rather than a confident wrong number.
#
# A denylist keyed on the one class known to answer badly has the opposite property: deleting that
# class makes it answer True for everything, so the guard stops guarding without failing, and the
# next partially-implemented connection object walks through the hole.
#
# Named rather than `isinstance` so that importing a backend module and its dependencies is not
# a prerequisite for a SQLite-only node.
#
# A store with typed methods never reaches this predicate — `_typed` is tried first at every call
# site. This is the legacy raw-SQL fallback, and the lattice store has no `.c` at all, so it is
# `False` there.
_FAITHFUL_CONNS = frozenset({
    # The legacy graph store's connection: it answered, or it raised. No production object carries
    # this name; it is a sentinel, which keeps the allowlist mechanism under its pinned tests
    # (test_drain_invariants unit F) and lets a fake opt into the raw-SQL path by taking the name.
    # The sentinel is arbitrary; the mechanism is not.
    "RawSqlConnection",
})


def _raw_ok(artifacts) -> bool:
    """True exactly when `artifacts.c` is a connection known to either answer or raise.

    An allowlist — see the block above. A connection not named in `_FAITHFUL_CONNS` is treated as
    having no reading to give, and every caller in this module and in `content_tier.py` then reports
    "not measured" or takes an exhaustive fallback that can itself fail. The one thing none of them
    produce is a value that looks healthy."""
    c = getattr(artifacts, "c", None)
    if c is None:
        return False
    return type(c).__name__ in _FAITHFUL_CONNS


def write_stats(store) -> Dict[str, Any]:
    """Persist a compact snapshot, using server-side COUNT queries rather than an O(corpus) fetch.

    The dashboard's headline numbers (artifacts, per-type counts, shard convergence, queue,
    provenance) are all cheap counts. ρ, coverage and dark-matter need a full byte scan, so they
    ride along from the metrics cache, refreshed on its own slower cadence, and this write proceeds
    without waiting for it. Safe to call every 20s: it does not scan the corpus."""
    from ember import genesis
    from ember.runtime import improve
    from ember.runtime import pool
    st = genesis.status(store)                          # COUNT-based: artifacts, curriculum, provenance
    heavy = genesis._METRICS_CACHE.get("val") or {}     # last full scan (rho/coverage/dark/bytes)
    # Per-type counts ride the metrics scan that already happened (improve.metrics, cached 90s and
    # refreshed off the request path) and fall back to a live COUNT only when that cache is cold.
    # `count(*) WHERE content_type = :ct` is cheap only on a selective value: measured on node 71
    # (5,657,386 rows), one index, one query shape — a 48-row type 0.12s, a 1-row type 0.00s,
    # text/markdown (~6M rows) timed out at 120s, over 1700x apart. wordnet is ~10^5 rows and the
    # concept count is unbounded, so counting them live on this 20s publish loop costs far more than
    # the query shape suggests. `operators` (~50 rows) is selective and costs nothing either way.
    heavy_bt = (improve._METRICS_CACHE.get("val") or {}).get("by_content_type") or {}

    def _ct(content_type: str) -> int:
        v = heavy_bt.get(content_type)
        return int(v) if v is not None else _count_ct(store, content_type)

    snap = {
        "ts": time.time(),
        "artifacts": st.get("artifacts"),
        "wordnet": _ct("text/x-wordnet"),
        "operators": _ct(genesis.OPERATOR_CONTENT_TYPE),
        "concepts": _ct("application/x-concept"),
        # `content_docs` comes from improve.metrics (`heavy_bt`'s cache). `genesis.all_metrics`'
        # return dict has no such key, so reading it off `heavy` yields None on every snapshot.
        "content_docs": (heavy_bt or {}).get("content_docs") if isinstance(heavy_bt, dict) else None,
        "rho": st.get("rho"), "keyed_coverage": st.get("keyed_coverage"),
        "dark_matter": heavy.get("dark_matter"),
        "bytes": heavy.get("bytes"), "generator_bytes": heavy.get("generator_bytes"),
        # The sampling labels travel with the figures. `genesis.all_metrics` defaults to a
        # 20,000-row sample and emits `sampled`/`sample_n` so an estimate reads as an estimate.
        # `dark_matter` is the sharp case: it is an absolute count rather than a ratio, so on a
        # 5.6M-row node it tops out at 20,000 however much dark matter exists, and the dashboard
        # renders it as the mesh total. These labels let a reader tell an estimate from a census.
        "metrics_sampled": heavy.get("sampled"),
        "metrics_sample_n": heavy.get("sample_n"),
        "rho_as_of": st.get("rho_as_of"),
        "collections": st.get("collections", {}),
        "ingest": genesis.ingest_progress(store),       # shard convergence (landing signal)
        "queue": pool.queue_stats(store),               # work-pool depth
        "provenance": st.get("provenance", {}),
    }
    # Disk — the ingest boxes are capped at 100 GB and a full disk crashes them. Cheap local stat.
    try:
        import shutil
        du = shutil.disk_usage(str(store.keys_dir) if store.keys_dir else ".")
        snap["disk_pct"] = round(100.0 * du.used / du.total, 1)
        snap["disk_free_gb"] = round(du.free / 1e9, 1)
    except Exception:
        pass

    # ── What it is doing, how fast, and how far behind ─────────────────────────────────────────
    # A level is not a signal: artifact counts alone make a node that gained 300,000 rows in an hour
    # look identical to one doing nothing. The three readings added here — current work, throughput
    # and backlog — carry the difference, and each is a cheap indexed query or a local file read.
    snap.update(_activity(store))
    return _finish(store, snap)


# ── How long a differencing interval may be, read from the publisher ──────────────────────────
#
# The node that publishes the snapshots declares its own cadence, so the admissible interval is
# derived from that declaration. It is the same rule `genesis.published_gate` states for staleness:
# derive it from the publisher's schedule rather than from a constant chosen at the reader.
#
# One stated tolerance: how many publish cycles may be missed before two snapshots stop being
# adjacent samples of the same regime. Both ends of the window follow it, symmetrically in ratio,
# because a gap far shorter than the cadence is the same kind of evidence as one far longer — it
# means something other than the declared loop wrote a snapshot (the two-publishers case recorded
# in serve.py).
_RATE_MAX_MISSED_CYCLES = 3

# Used when no gate has been published, i.e. when there is no declared cadence to read. It tracks
# `_fleet/peers/71/ember/health-loop.py`'s own `--fast` default, because that loop is the sole publisher of these
# snapshots. If health-loop's default changes, this changes with it: it restates the publisher's
# cadence rather than making an independent judgement about sampling.
_UNDECLARED_PUBLISH_CADENCE_S = 30.0


def _rate_window_s(store) -> tuple:
    """(min_dt, max_dt) an adjacent-snapshot interval may span, from the publisher's own cadence."""
    cadence = None
    try:
        from ember import genesis
        gate = genesis.published_gate(store) or {}
        cadence = float(gate.get("fast_s") or 0) or None
    except Exception:
        cadence = None
    if cadence is None:
        cadence = _UNDECLARED_PUBLISH_CADENCE_S
    slack = 1.0 + _RATE_MAX_MISSED_CYCLES
    return cadence / slack, cadence * slack


def _rate_from_history(store, snap):
    """rows/min since the previous snapshot, derived here rather than asked of the reader.

    Kept on the node because only the node holds its own last sample; the dashboard sees one
    snapshot per box and cannot difference them itself."""
    prev = store.artifacts.get_artifact("stats.prev") or {}
    # `status()` returns None for `artifacts` when the count was not taken — it is cached, and only
    # the slow path measures. An unmeasured count skips the rate entirely and leaves the previous
    # sample in place. Coercing it to 0 on a 5.6M-row node would give rows_per_min = -11,200,000 and
    # persist artifacts:0 into stats.prev, so the next cycle reports +11,200,000, which the renderer
    # colours green for being positive.
    now_n, now_t = snap.get("artifacts"), snap.get("ts") or time.time()
    out = {}
    if now_n is None:
        return out
    try:
        p_n_raw = prev.get("artifacts")
        if p_n_raw is None:
            raise ValueError("no previous artifact count")
        p_n, p_t = int(p_n_raw), float(prev.get("ts") or 0)
        dt = now_t - p_t
        _lo, _hi = _rate_window_s(store)                 # derived from the publisher's cadence
        if p_t and _lo <= dt <= _hi:
            out["rows_per_min"] = round((int(now_n) - p_n) * 60.0 / dt, 1)
            out["sample_s"] = round(dt)
            # Drain throughput, by the same differencing. Derived on the node because only the node
            # holds its previous sample; the dashboard sees one snapshot per box.
            try:
                cur = store.artifacts.get_artifact("content.promote.cursor") or {}
                now_s = int(cur.get("scanned_total") or 0)
                p_s = int(prev.get("scanned_total") or 0)
                if p_s and now_s >= p_s:
                    out["drain_per_min"] = round((now_s - p_s) * 60.0 / dt, 1)
                out["_scanned_total"] = now_s
            except Exception:
                pass
    except Exception:
        pass
    try:
        store.artifacts.put_artifact({"id": "stats.prev", "content_type": _S3SYNC_CT_PREV,
                                      "state": "committed", "artifacts": now_n, "ts": now_t,
                                      "scanned_total": out.get("_scanned_total", 0)})
    except Exception:
        pass
    return out


_S3SYNC_CT_PREV = "application/vnd.agience.s3sync-cursor+json"   # operational: never replicated




def _activity(store):
    """Backlogs and current work. The publish backlog is Merkle-native (unpublished changed leaves)."""
    out = {}
    try:
        from mantle.mesh import sync as _sync
        # Merkle anti-entropy is the one publish path. The backlog is how many of this node's leaves
        # have changed but are not yet published (live differs from published), and
        # `publish_backlog_unset` before any tree exists. It is exact by construction — a leaf-state
        # compare rather than a seq subtraction — so update churn leaves it unchanged.
        out.update(_sync.publish_backlog_now(store))
    except Exception:
        pass
    # What it is working on — the ingest loop's current claim and the content drain's position,
    # read from the loops' own logs so the page reflects the running process.
    try:
        import json as _json, os as _os
        base = _os.path.dirname(str(store.keys_dir or "")) or "."
        for key, fname, want in (("working_on", "svc-ingest.log", ("claimed", "yield_to_publish")),
                                 ("content_at", "svc-content.log", ("drain",))):
            path = _os.path.join(base, fname)
            try:
                # Seek from the end and read a bounded tail. `readlines()[-40:]` materialises the
                # whole file to keep 40 lines — two files, every fast cycle, in both the serve and
                # health loops, on the most loaded box: a 500MB svc-ingest.log is ~1GB of transient
                # allocation every 20-30s.
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 65536))
                    tail = f.read().decode("utf-8", "replace").splitlines()[-40:]
                for line in reversed(tail):
                    d = _json.loads(line)
                    kind = d.get("ingest") or d.get("content")
                    if kind in want:
                        out[key] = {"state": kind, "shard": d.get("shard"),
                                    "backlog": d.get("backlog"), "promoted": d.get("promoted"),
                                    "at": d.get("last_id"), "ts": d.get("ts")}
                        break
            except Exception:
                continue
    except Exception:
        pass
    return out


def _finish(store, snap):
    snap.update(_rate_from_history(store, snap))
    snap.pop("_scanned_total", None)
    # Content → S3 — the drain, reported as a state a human can act on rather than one number, so
    # "is it done?", "how far?" and "is it stuck?" are answerable from the monitor:
    #   promoted_total  cumulative refs pushed to the durable origin
    #   drain_cursor    where the keyset walk has reached
    #   drain_done      the retirement gate: true only when the cursor reports exhausted. It is the
    #                   one signal a retire decision rests on — cursor position and promoted_total
    #                   each say something narrower.
    #   drain_swept_at  when the last full sweep completed (re-sweep runs on a cadence)
    try:
        cur = store.artifacts.get_artifact("content.promote.cursor") or {}
        last_id = str(cur.get("last_id") or "")
        snap["promoted_total"] = cur.get("promoted_total")
        # The position reported is the walk that is actually running: the `(_origin, _seq)` keyset,
        # whose cursor field is `seq` (content_tier.promote_local_content). `rev` is read as the
        # legacy fallback, so a box mid-migration whose cursor doc was last written by the Arcade
        # path still reports a position.
        #
        # `last_id` belongs to the one-time id backfill, which runs to exhaustion once, sets
        # `last_id = ""` and is never re-armed. Past its backfill a box would publish either a dead
        # id or None from it while the drain ran normally, which is why it is used only for the
        # pre-backfill case below.
        _pos = int(cur.get("seq") or cur.get("rev") or 0)
        snap["drain_cursor"] = (str(_pos) if _pos else None) if cur.get("id_backfill_done") \
            else (last_id or None)
        snap["drain_rev"] = _pos or None
        # Backlog — refs still to walk. Measured on the health slow path (see health-loop) because
        # the query dereferences records and costs ~25s, far more than this 20s path allows.
        # Published as None when not measured, so an unmeasured backlog reads differently from an
        # empty one.
        snap["drain_backlog"] = cur.get("backlog")
        snap["drain_backlog_at"] = cur.get("backlog_at")
        snap["drain_swept_at"] = cur.get("swept_at")
        # The gate is the walk's own `exhausted`: it reached the horizon with nothing left, it costs
        # nothing to read, and it self-clears when new content arrives. A measured backlog, when
        # present, has to agree; an absent measurement leaves the answer to `exhausted`, because a
        # gate that requires a number nothing produces can never open.
        #
        # Two shapes it deliberately does not use. `last_id == ""` means both "finished and wrapped"
        # and "a fresh sweep has not advanced yet", and `swept_at` is sticky once set, so a box that
        # completed a sweep and then ingested 500k artifacts would read as safe to retire for content
        # that was never promoted. And a required measured backlog of zero is unreachable: health-loop
        # counts `backlog` as
        #     count(*) WHERE id > last_id AND content_ref IS NOT NULL
        # against that same `last_id`, which is "" past the backfill, so the predicate degenerates to
        # every row in the corpus that has content (2.19M on 45, 1.49M on TU) and stays above zero
        # while the box holds any content. That shape is also the un-index-servable full scan this
        # module's comments rule out, so it reads "backlog not measured" on every box.
        _bl = cur.get("backlog")
        _ex = cur.get("exhausted")
        if _ex is None:
            snap["drain_done"] = None                      # older cursor doc: unmeasured, not False
        else:
            snap["drain_done"] = bool(_ex) and bool(cur.get("id_backfill_done")) and _bl in (None, 0)
    except Exception:
        pass
    # Ingest — shard progress for this box, so "is it actually ingesting?" is answerable from the
    # status page rather than by tailing a log over ssh. Cheap: shard-done markers are a handful of
    # rows fetched by an indexed content_type lookup rather than a corpus scan.
    try:
        import os as _os
        rng = _os.getenv("EMBER_SHARDS", "")
        if rng:
            done = len(list_by_content_type(store.artifacts,
                                            "application/vnd.agience.shard-done+json"))
            # EMBER_SHARDS is not always a numeric range — 71 uses "all". Reports what is known,
            # the range string and the done count, and omits the comparison when there is no numeric
            # range to compare against: `assigned: None` beside `done: 0` renders as "0/None shards",
            # which reads like a box that has ingested nothing.
            lo, _, hi = rng.partition("-")
            assigned = None
            try:
                if hi:
                    assigned = int(hi) - int(lo) + 1
            except ValueError:
                assigned = None
            # Fills in only what is missing, leaving the narrower value in place. `write_stats`
            # already set `ingest` from `genesis.ingest_progress`, whose `done` counts shard-done
            # markers for one dataset; the count here spans every marker on the box, so on a node
            # holding markers from a second corpus it would give done > assigned and
            # `converged: true` with real shards unfinished — and convergence gates teardown.
            prior = snap.get("ingest") if isinstance(snap.get("ingest"), dict) else {}
            if not prior:
                snap["ingest"] = {"range": rng, "markers": done}
                if assigned is not None:
                    snap["ingest"]["assigned"] = assigned
            else:
                prior.setdefault("range", rng)
    except Exception:
        pass
    # Role and S3-mesh sync progress: what this box is for, and how far its S3 publish/consume has got.
    import os
    snap["role"] = os.getenv("EMBER_ROLE", "ingest" if os.getenv("EMBER_SHARDS") else "full")
    snap["purpose"] = os.getenv("EMBER_PURPOSE", "")
    try:
        pub = store.artifacts.get_artifact("s3.pub.cursor") or {}
        subs = {}
        # An indexed fetch rather than a corpus stream. `list_artifacts(content_type=…)` streams
        # every row and filters in Python — ~500 keyset pages, about 370s at 5M rows — which this
        # function's "does not scan the corpus" contract rules out on a 20s publish loop. There are
        # a handful of s3sync-cursor docs, and one indexed query returns exactly the same set.
        for c in list_by_content_type(store.artifacts, "application/vnd.agience.s3sync-cursor+json"):
            if str(c.get("id", "")).startswith("s3.sub.cursor."):
                subs[str(c.get("node"))] = str(c.get("last_key", "")).rsplit("/", 1)[-1]
        snap["s3sync"] = {"published_segments": int(pub.get("seq", 0) or 0),
                          "consumed_from": subs}                # {peer_node: last-segment-consumed}
    except Exception:
        pass
    snap["node"] = _node_id()
    p = stats_path(store.keys_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap), encoding="utf-8")
    tmp.replace(p)                                       # atomic swap — readers never see a partial
    _publish_to_mesh(store, snap)                        # gossip: one file per host in the shared dir
    return snap


def _node_id() -> str:
    from prism.envelope import node_id
    return node_id()


#: Key prefix for the shared mesh-stats directory each host publishes its snapshot into.
#:
#: Restored 2026-08-25. This line was deleted as collateral in `8716d55`, a commit whose subject
#: was replacing `_node_id`'s body with `prism.envelope.node_id()` — the constant simply sat in the
#: same hunk. Its two uses survived (`_publish_to_mesh` and `read_mesh_stats`), so both raised
#: `NameError`, and both sit inside handlers that swallow: mesh stats publishing and reading have
#: silently done nothing since 2026-07-30.
#:
#: The value is not a guess — `git log -S` recovers it from `10a4a93`, the commit that added it.
MESH_STATS_PREFIX = "mesh-stats/"


def _mesh_remote(store):
    """The shared origin (OVH S3) each host publishes its snapshot to — the 'synchronized directory'.
    Reached through the tiered content store's remote tier (or a direct OVH store)."""
    r = getattr(store.content, "remote", None)
    if r is not None:
        return r
    try:
        from mantle.shard.content_tier import open_ovh_store
        return open_ovh_store(store.keys_dir)
    except Exception:
        return None


def _publish_to_mesh(store, snap) -> None:
    """Write this host's snapshot to mesh-stats/<node>.json in the shared bucket. Small JSON, one per
    host — every host reads the whole folder to see the mesh. Best-effort: a failure here leaves the
    local snapshot written."""
    r = _mesh_remote(store)
    if r is None:
        return
    try:
        r.put(MESH_STATS_PREFIX + _node_id() + ".json", json.dumps(snap).encode("utf-8"),
              "application/json")
    except Exception:
        # Fail-soft stays: a mesh publish must not fail the refresh that already wrote the local
        # snapshot. Fail-silent does not. This handler swallowed a `NameError` on every call from
        # 2026-07-30 to 2026-08-25 and nothing anywhere said a word — `exc_info` is what makes the
        # difference between a feature that is off and a feature that looks on.
        logger.warning("mesh-stats publish failed; the local snapshot is still written",
                       exc_info=True)


def read_mesh_stats(store) -> list:
    """Read every host's snapshot from the shared dir (mesh-stats/*.json) — the whole mesh, from one
    synchronized location instead of N live HTTP calls, so a host whose serve is down still
    appears."""
    r = _mesh_remote(store)
    if r is None:
        return []
    out = []
    try:
        s3 = getattr(r, "_s3", None)
        bucket = getattr(r, "bucket", None)
        if s3 is None or bucket is None:
            return []
        pag = s3.get_paginator("list_objects_v2")
        keys = [o["Key"] for pg in pag.paginate(Bucket=bucket, Prefix=MESH_STATS_PREFIX)
                for o in pg.get("Contents", [])]
        now = time.time()
        # Read failures are counted and carried, so the renderer can say "4 of 5 peers read". A
        # shorter list on its own makes a dropped peer look like a peer that does not exist: one
        # transient S3 error moves `whole_art = max(...)` from 5.6M to 1.7M, switches `lead` to a
        # different box (taking rho/coverage/wordnet/concepts with it) and drops a `content_s3`
        # summand, with nothing on the page to show for it.
        _failed = 0
        for k in keys:
            try:
                d = json.loads(r.get(k))
                d["age_s"] = round(now - d.get("ts", 0), 1)
                out.append(d)
            except Exception:
                _failed += 1
        if _failed and out:
            out[0] = dict(out[0]); out[0]["_peers_unreadable"] = _failed
            out[0]["_peers_expected"] = len(keys)
    except Exception:
        # The OUTER handler, and it is a different animal from the per-key one above. That one
        # COUNTS and surfaces `_peers_unreadable`, which is why a dropped peer is visible. This one
        # wraps the listing itself, so anything it catches makes the mesh look like a mesh of one —
        # and an empty list is this function's normal answer, so nothing downstream can tell the
        # difference. It swallowed a `NameError` for 26 days on exactly that basis.
        logger.warning("mesh-stats listing failed; reporting a mesh of %d", len(out), exc_info=True)
    return out


import os as _os

PUBLIC_STATUS_KEY = "index.html"
# Dedicated public bucket, holding the anonymized status page alone. The content/mesh-stats bucket
# stays fully private, with no public objects on it. Private topology — node ids, hostnames, cursors,
# peer targets — stays out of here: publish_public_status renders with public=True, opaque labels only.
STATUS_BUCKET = _os.getenv("EMBER_STATUS_BUCKET", "agience-genesis-status")



# ── The dashboard facet seam ─────────────────────────────────────────────────────────────────────
# Rendering is a persona's view, so the dashboard renderer lives in `aria.facets.dashboard_render`
# and ember imports no chorus module. It arrives here by injection: a host that serves both wires it,
# the way `host_lumen.py` wires lumen's responder into aria's bff.
#
# Unwired, publishing stops before the PUT, leaving the last good public status object in place —
# see `publish_public_status`, which distinguishes "no renderer" from "a renderer that produced
# nothing" and treats both as nothing to publish.
_DASHBOARD_RENDERER = None


def _render_dashboard(*args, **kwargs):
    """The injected renderer's output, or None when nothing is wired. Imports no chorus module."""
    fn = _DASHBOARD_RENDERER
    if fn is None:
        return None
    return fn(*args, **kwargs)


def publish_public_status(store) -> bool:
    """Render the anonymized whole-mesh dashboard from the gossiped snapshots and push it to the public
    bucket as a public-read object. status.agience.ai (Caddy) serves that object directly, so the public
    VPS holds no credentials and runs no code that can touch the corpus. Every box publishes
    independently — last-writer-wins over identical anonymized data — so status stays fresh while any
    box is up. Best-effort: returns False when there is nothing to publish."""
    r = _mesh_remote(store)
    if r is None:
        return False
    try:
        boxes = read_mesh_stats(store)
        # An empty mesh publishes nothing. `read_mesh_stats` swallows every exception at both levels
        # and returns `[]`, so an S3 listing failure, expired credentials or a wrong bucket read the
        # same as "no peers". Rendering that list gives a page where every metric is zero or None,
        # and PUTting it over the last good public object would report success to `health-loop.py`.
        # The stale page keeps its own visible age, which is the honest signal.
        if not boxes:
            return False
        html = _render_dashboard(boxes, None, public=True)  # public=True: no host/ip/ids
        # No renderer stands on the same footing as no boxes: there is nothing to publish, and the
        # last good page stays in place with its own visible age. A blank or partial page put over
        # it would be the empty-mesh case above in a new form.
        if not html:
            return False
        r._s3.put_object(Bucket=STATUS_BUCKET, Key=PUBLIC_STATUS_KEY, Body=html.encode("utf-8"),
                         ACL="public-read", ContentType="text/html; charset=utf-8",
                         CacheControl="no-cache, max-age=5")
        return True
    except Exception:
        return False


def _count_ct(store, content_type: str):
    """Count rows of one content type, or None when the count could not be taken.

    None is the house convention for "not measured" (`invariant_holds`, `converged`,
    `spearman_vs_tree_JC`, `has_ic`) and the renderer prints an em-dash for it. Returning 0 instead
    would make a query timeout, a dropped connection or a store with no counter read as a genuinely
    empty corpus, and the dashboard renders these three as headline totals with no null check.

    The typed path is `count_by_content_type`, a counter lookup: O(1), maintained in the write
    transaction. A counter has no selectivity, so `text/x-wordnet` and `application/x-concept` cost
    exactly what `operators` costs — which is what makes counting them here cheap, where
    `count(*) WHERE content_type = :ct` would not be (see the module comment above). The
    cache-vs-live fallback in `write_stats` still applies: a cold cache wants the live counter, and
    a store with no typed counter reports not measured."""
    counter = _typed(store.artifacts, "count_by_content_type")
    if counter is not None:
        try:
            return int(counter(content_type))
        except Exception:
            return None
    return None                        # no typed counter: not measured, which is not the same as 0


# `CT_FETCH_CAP` is the largest matched-row count for which the indexed fetch is still the cheap
# path. An index is cheap only on a selective value — measured on node 71 (5,657,386 rows), same
# index, same query shape: content_type='application/vnd.agience.task+json' (48 rows) 0.12s,
# content-cursor (1 row) 0.00s, text/markdown (~6M rows) timed out at 120s, over 1700x apart.
# Selectivity cannot be indexed away, so `list_by_content_type` serves small operational types and
# hands anything larger back to the streaming keyset. The cap sits well under the ~20k silent result
# truncation of the legacy backend, so a full probe page reads as a real overflow.
#
# Both live in mantle: the cap is a store fetch bound (`mantle.db.constants`) and the typed
# fetch is a store query (`mantle.db.typed_fetch`), so every caller reaches them downward.
# Re-exported here for this surface's callers.
from mantle.db.constants import CT_FETCH_CAP as _CT_FETCH_CAP  # noqa: F401,E402
from mantle.db.typed_fetch import _typed, list_by_content_type  # noqa: F401,E402




def read_stats(keys_dir) -> Optional[Dict[str, Any]]:
    """Instant read of the last snapshot, or None when none has been written yet."""
    try:
        d = json.loads(stats_path(keys_dir).read_text(encoding="utf-8"))
        d["age_s"] = round(time.time() - d.get("ts", 0), 1)
        return d
    except Exception:
        return None