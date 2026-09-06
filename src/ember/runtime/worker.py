"""Ember's autonomous worker — the loop that runs unattended.

A plain OS process. Launch it detached and it runs the deterministic self-improvement cycle
forever — illuminate dark matter, apply selection (retire the proven-unfit), measure — writing a
heartbeat every tick. It needs no model, no network, and no session, so it keeps running when
nothing is attached to it.

Verifiable: tail the heartbeat log and watch `tick` climb on its own.

  python -m ember.runtime.worker --interval 60                # run forever, one cycle/min
  python -m ember.runtime.worker --interval 5 --max-ticks 3   # bounded (for a quick proof)

Each tick is idempotent and gated: describe illuminates real dark matter only, retire sheds only
operators reality has already condemned. It runs alongside `serve` (the store retries boundedly on
write contention); normally one canonical worker runs per node.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path
from typing import Optional



# The meter's previous-tick snapshot (per-process): dark pool + write counter, so each
# tick's FlowWindow is a delta over published numbers. See the "feed the meter" block in `_tick`.
_METER_PREV: dict = {}

def _log_path(bundle) -> Path:
    base = bundle.keys_dir.parent if bundle.keys_dir else Path(".")
    return Path(base) / "worker.log"


# ── liveness: one ping cadence, and every window derived from it ─────────────────────────────
#
# Both pingers below (`run`'s and `run_aggregator`'s) sleep this long, and `genesis.health`
# derives `alive` from it. The two numbers below are the only inputs: for a different liveness
# window to be right, either the cadence or the number of consecutive pings a node may miss before
# it is called dead would have to change.
PING_CADENCE_S = 20.0
DEAD_AFTER_MISSED_PINGS = 4


def liveness_window_s(interval: float | None = None) -> float:
    """Seconds of silence after which a node reads as not alive.

    The pinger sleeps `min(interval, PING_CADENCE_S)`, so a faster loop pings faster and its window
    tightens with it: the window follows the cadence actually in force rather than the ceiling."""
    cadence = PING_CADENCE_S if interval is None else min(float(interval), PING_CADENCE_S)
    return DEAD_AFTER_MISSED_PINGS * cadence


# ── the throughput sample admission rule ─────────────────────────────────────────────────────
#
# `sync_s` is persisted as `round(elapsed, _SYNC_S_DECIMALS)`, so a cycle's duration is known to
# half a quantum. A rate computed from it inherits that error, and only samples whose error the
# EWMA can distinguish from rounding enter the EWMA. The floor is therefore derived from the
# recorded precision and one stated tolerance.
#
# For a different floor to be right, either the persisted precision or the accepted error would
# have to change; both are here, and `min_sync_s` follows either.
_SYNC_S_DECIMALS = 1
_RATE_QUANTISATION_TOL = 0.05      # max relative error a timing sample may contribute to `rate`


def min_sync_s() -> float:
    """The shortest cycle whose *recorded* duration is precise enough to learn a rate from."""
    half_quantum = (10.0 ** -_SYNC_S_DECIMALS) / 2.0
    return half_quantum / _RATE_QUANTISATION_TOL


# A stated value, about representativeness rather than precision: a cycle that moved almost nothing
# spends its time on fixed per-cycle overhead (claim, connect, commit), so its rate measures the
# overhead rather than the throughput. The quantisation of an exact integer count says nothing about
# that, so it is not the source. Measuring the fixed per-cycle overhead — the intercept of `sync_s`
# against `_moved` — would make this `overhead_s * rate / _RATE_QUANTISATION_TOL`; nothing measures
# that intercept today.
MIN_SAMPLE_MOVED = 1000


def consolidation_due(tick: int, every: int) -> bool:
    """Is this tick due for the cross-source colimit sweep?

    `every=0` disables the cadence entirely, which is the pre-2026-08-25 behaviour: the sweep then
    runs only when a stage promotes. Named rather than inlined so the rule can be asserted without
    standing up a store.
    """
    return bool(every) and tick % every == 0


def _tick(bundle, *, ingest: bool = False, consolidate: bool = False) -> dict:
    """One autonomous, deterministic iteration. Returns a compact heartbeat record.

    With `ingest=True` the worker also drives the curriculum: it advances the current stage's
    ingestion by one bounded increment, or promotes it, unattended from an OS process. The ingest
    step runs first so new dark matter is illuminated and measured within the same tick.

    `consolidate=True` runs the cross-source colimit sweep on THIS tick. A promotion still triggers
    it — that is a good moment — but it is no longer the only one; see the note below."""
    from ember.runtime import improve
    ingested = consolidated = None
    if ingest:
        try:
            from ember import genesis
            ingested = genesis.advance_curriculum(bundle)
            # Full deterministic pipeline: once a stage is drained/promoted (or nothing left to
            # ingest), COMPRESS — run the cross-source colimit sweep so ρ falls autonomously.
            # This is the worker doing more than ingest: ingest -> consolidate -> measure.
            if ingested.get("promoted") or ingested.get("curriculum") == "complete":
                consolidate = True
        except Exception as e:
            ingested = {"error": str(e)[:200]}
    # The sweep reads only what is ALREADY in the store — it indexes synsets by keyed word and
    # matches article titles against them. It has no dependency on an ingest having just happened.
    # Gating it only on a promotion would couple it to remote source reachability: a stage promotes
    # only when every source in it is drained, and a source that cannot be fetched (OEWN 503, an
    # OMW licence still clearing) defers rather than failing. Measured 2026-08-25 on 71/home: the
    # sweep had not run since 2026-07-25 and `consolidates` edges were frozen at 11,231, while the
    # store held far more unconsolidated alignments than that.
    # The gate itself is not wrong to exist — this is a full-corpus scan and running it every tick
    # would be waste. So it keeps a bound; what changes is that the bound is its own cadence rather
    # than someone else's milestone.
    if consolidate:
        try:
            from ember import genesis
            consolidated = genesis.consolidate_crosswalk(bundle, apply=True, limit=3000)
        except Exception as e:
            consolidated = {"error": str(e)[:200]}
    snap = improve.improve_cycle(bundle)          # illuminate -> sweep_retire -> measure -> record
    rec = {"ok": True,
           "illuminated": snap.get("illuminated_this_cycle"),
           "retired": snap.get("retired_this_cycle"),
           "dark_matter": snap.get("dark_matter"),
           "keyed_coverage": snap.get("keyed_coverage"),
           "rho": snap.get("rho"),
           "operators": snap.get("operators"),
           "avg_operator_fitness": snap.get("avg_operator_fitness")}
    if ingest:
        rec["ingest"] = ingested
    # Reported whenever it ran, not only on ingest ticks — the heartbeat is the only record that
    # the sweep fired at all, and a silent sweep is how the 2026-07 stall stayed invisible.
    if consolidated is not None:
        rec["consolidated"] = consolidated
    # ---- feed the meter ----
    # The window is this tick, tallied from numbers the cycle already publishes, so no new probe is
    # taken: promoted = illuminated, evicted = retired, and minted_dark comes by balance from the
    # standing pool:
    #   new_dark = (pool_now − pool_prev) + cleared   (clearing shrank the pool; add it back).
    # The first tick has no prev, so there is no window and "flow" is absent — the visible
    # unmeasured state. A failure here leaves "flow" absent as well, and the loop continues.
    try:
        from prism.error_threshold import FlowWindow
        dark_now = snap.get("dark_matter")
        writes_now = None
        try:
            writes_now = int(bundle.artifacts.count())      # maintained counter, never count(*)
        except Exception:
            pass
        prev = _METER_PREV
        if dark_now is not None and prev.get("dark") is not None:
            promoted = int(rec.get("illuminated") or 0)
            evicted = int(rec.get("retired") or 0)
            cleared = promoted + evicted
            minted = max(0, (int(dark_now) - int(prev["dark"])) + cleared)
            tw = (max(0, writes_now - prev["writes"])
                  if writes_now is not None and prev.get("writes") is not None else 0)
            w = FlowWindow(minted_dark=minted, promoted=promoted,
                           evicted=evicted, total_writes=tw)
            rec["flow"] = {"minted_dark": minted, "cleared": cleared,
                           "validation_ratio": round(w.validation_ratio, 4) if minted else None,
                           "health": w.health.value}
        _METER_PREV["dark"] = dark_now if dark_now is not None else prev.get("dark")
        _METER_PREV["writes"] = writes_now if writes_now is not None else prev.get("writes")
    except Exception:
        pass
    # Checks and balances every tick: surface anomalies (ρ∉[0,1], mass vanished, …) on the record
    # so the Monitor sees them and the loop can investigate.
    try:
        from ember import genesis
        cons = genesis.consistency(bundle)
        if cons["anomalies"]:
            rec["ANOMALY"] = [c["check"] for c in cons["anomalies"]]
        if cons.get("watch"):
            rec["watch"] = [c["check"] for c in cons["watch"]]
    except Exception:
        pass
    return rec


def run(*, interval: float = 60.0, max_ticks: Optional[int] = None, ingest: bool = False,
        consolidate_every: int = 0, clock=time.time) -> int:
    from mantle.shard.local_store import open_store
    from ember.runtime import runner
    bundle = open_store()
    runner.attach(bundle)                          # store bundles outrank shipped (first load pins)
    from ember import genesis                          # AFTER attach — genesis loads bundles at import
    arithmetic = runner.load("arithmetic")
    operators = runner.load("operators")
    dev_ops = runner.load("dev_ops")
    fetch = runner.load("fetch")
    genesis.bootstrap(bundle)                      # schema is always present before a tick
    # Register every operator family (fitness-preserving) so metrics and selection see them all.
    # The author is the resolved principal, via `_author_ref` (contract §2.1), matching what
    # genesis's own ops do. The register fns default to a process author, which on a fresh store
    # writes rows whose `created_by` resolves to nothing and fails the gate's `created_by resolves`
    # check at the first row.
    _author = genesis._author_ref(bundle, os.getenv("EMBER_PRINCIPAL") or "ember-local")
    # These four are the writers of the 25. `artifacts_holding_inline_plaintext` reads
    # 25 on 71/home and every row is one of theirs — `op.math.*` 13 (arithmetic), `op.dev.*` 8
    # (dev_ops), `op.describe.*` 3 (sage operators), `op.fetch.*` 1 (fetch). They write inline
    # `content` as raw dicts, and the store now roots and seals such a write — which needs an acting
    # principal in scope, and this loop had none.
    #
    # Wrapped here, which is an ember file, so the bundles constraint does not apply: the
    # registrars themselves live in chorus and ship as bundles — the runtime reads the payload, not
    # the source — so editing them would be inert until `build_bundles.py` rebuilt them. Their
    # caller is here, in ember, and is not bundled.
    from ember.custody import system_custody_if_unowned
    with system_custody_if_unowned("ember.register_persona_operators"):
        for reg in (arithmetic.register_transform_operators, operators.register_operators,
                    dev_ops.register_dev_operators, fetch.register_fetch_operators):
            try:
                # The full store, not `.artifacts` — `evolution.put_operator` asks it for
                # `address_inline` so the operator body reaches the CAS instead of sitting
                # inline in the row. The bare artifacts face has no content tier, and these
                # four registrars wrote every one of the 25 rows the invariant reports.
                reg(bundle, author=_author)
            except Exception as exc:
                # The swallow stays — a registrar for a persona this node does not carry is a
                # normal miss — but it is no longer silent. A seal that refuses would otherwise
                # make the operator catalogue stop being written with nothing said, which is an
                # invisible capability loss in place of a visible plaintext one.
                # One-line JSON to stdout — this module has no logger and every other report in
                # it is a `print(json.dumps(...))`. Adding one for a single line would split its
                # output across two streams the supervisor does not capture the same way.
                print(json.dumps({
                    "event": "register_operators_failed",
                    "registrar": getattr(reg, "__module__", "?"),
                    "error": "%s: %s" % (type(exc).__name__, str(exc)[:200]),
                }), flush=True)
    try:
        genesis.register_consolidate_operators(bundle)
    except Exception:
        pass
    genesis.register_control_operators(bundle)
    try:
        genesis.backfill_provenance(bundle, apply=True)     # §12: every artifact stays cited
    except Exception:
        pass
    # Publish this host — probed capabilities plus measured limits, as an artifact.
    # `opsign.admit` then has real offers to check, and a peer can read what this environment can
    # do without asking the host.
    #
    # Publishing runs through `capability.publish_host` rather than inline here, because the serve
    # path never published at all: `agience up ember` runs `python -m ember.cli serve`, a different
    # process from this one, so a node that served and never ingested never said what it was. Both
    # paths call the one function — including its grant warning, which is the only place a published
    # -but-unreachable host is ever visible.
    from ember.runtime import capability
    capability.publish_host(bundle, who="ember.worker")
    logp = _log_path(bundle)
    # Liveness: a single ingest tick can run for many minutes (a whole parquet shard). A side
    # thread pings the heartbeat log every `PING_CADENCE_S` so health reads the process as alive and
    # working mid-tick. The main loop still writes the detailed per-tick record.
    import threading
    _state = {"tick": 0}

    def _pinger():
        while True:
            time.sleep(min(interval, PING_CADENCE_S))
            try:
                with open(logp, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ping": True, "ts": time.time(), "tick": _state["tick"]}) + "\n")
            except Exception:
                pass
    threading.Thread(target=_pinger, daemon=True).start()

    tick = 0
    while max_ticks is None or tick < max_ticks:
        tick += 1
        _state["tick"] = tick
        rec = {"tick": tick, "ts": clock()}
        try:
            due = consolidation_due(tick, consolidate_every)
            rec.update(_tick(bundle, ingest=ingest, consolidate=due))
        except Exception as e:                    # a bad cycle never kills the loop
            rec.update({"ok": False, "error": str(e)[:200]})
        try:
            with open(logp, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        print(json.dumps(rec), flush=True)        # heartbeat to stdout too
        if max_ticks is not None and tick >= max_ticks:
            break
        time.sleep(interval)
    return 0


def _healthy_progress(rec: dict, is_consumer: bool) -> bool:
    """Did this cycle make the progress this node is asked to make?

    Health is judged per role, against the work the node actually does:
      - a CONSUMER is healthy if it applied something, or if it is caught up;
      - a PUBLISH-ONLY node is healthy if it published, or has nothing to publish.

    Being caught up counts as progress, so a converged fleet stays up. Judging every node on
    `applied or published` instead would clear the stall counter on a node that publishes fine
    while consuming nothing — the `applied:0 / behind:1837` shape this detector exists for — and
    would also read a shard-only node, where `applied` is structurally 0, as permanently stalled."""
    behind = rec.get("segments_behind")
    caught_up = isinstance(behind, int) and behind == 0
    if is_consumer:
        # Prefer the genuinely-new count over `applied`. `applied` counts upserts — new rows and
        # no-op re-applies of rows already held, identically — so a node rewriting what it already
        # has would clear this counter every cycle while falling further behind (the
        # `applied:723k / net-gain:0 / behind rising` shape). `new` is the split: rows actually
        # gained. `applied` is the fallback when `new` is absent — a pre-split rec, or a non-lattice
        # store where the split is not computable.
        new = rec.get("new")
        if new is not None:
            return new > 0 or caught_up
        return bool(rec.get("applied")) or caught_up
    return bool(rec.get("published")) or caught_up


def run_aggregator(*, interval: float = 20.0) -> int:
    """The aggregator daemon — Ember-owned, one process, driving the mesh operators on a loop.

    Everything is an operator; this drives them. Each round runs `op.mesh.reconcile`, the Merkle
    anti-entropy pass that publishes this node's tree and pulls the leaves that differ. The stats
    snapshot is written by serve, so this daemon does no heavy scan and reports no `rho`. It
    heartbeats like a worker, on the same `PING_CADENCE_S`, so /health reads it alive. Each step is
    guarded, so a transient failure leaves the daemon running."""
    import json
    import threading
    import time
    from pathlib import Path
    from mantle.shard.local_store import open_store
    from ember import genesis
    store = open_store()
    hbp = (Path(store.keys_dir).parent / "worker.log") if store.keys_dir else Path("worker.log")
    # Sync and promote are independent jobs. This daemon runs sync on every box:
    #   • sync (all boxes) — S3 is the mesh plane. Each box publishes its new graph as an encrypted
    #     segment log to S3 (authoritative + backup) and consumes every other node's segments. No
    #     box contacts another box (NAT-irrelevant); the union of segments is the whole graph, so a
    #     consuming box converges to full. Bounded and cursor-resumable.
    #   • promote — draining the local content cache up to the S3 origin — is owned by
    #     `content-loop.py`, on its own cadence. `EMBER_SHARDS` marks an ingest box, which is what
    #     produces content, and is reported in the banner below.
    # Ember derives each node's mesh behaviour from its role (`EMBER_ROLE`), set once at provisioning:
    #   • publish — every node publishes its own shard (graph/updates/edges) to S3.
    #   • consume — consumers pull the union and go full: role=full, or purpose in {tests,compress}.
    #     Shard-only (ingest) nodes publish without consuming.
    # Identity and role fail closed: a missing or mistyped env var stops the process here. A default
    # would make the node a publish-only participant that consumes nothing while logging "up" every
    # cycle, and from outside a non-consumer reads exactly like a caught-up one.
    _VALID_ROLES = ("full", "ingest")
    role = os.getenv("EMBER_ROLE", "").strip()
    purpose = os.getenv("EMBER_PURPOSE", "").strip()
    node_env = os.getenv("EMBER_NODE_ID", "").strip()
    if not node_env:
        raise SystemExit("EMBER_NODE_ID is not set — refusing to start. An unidentified node "
                         "publishes under whatever the OS calls the box, so the same shard "
                         "reappears in the mesh under a second identity and neither copy ever "
                         "converges with the other.")
    if role not in _VALID_ROLES:
        raise SystemExit(f"EMBER_ROLE must be one of {_VALID_ROLES}, got {role!r} — refusing to "
                         "start. Defaulting this silently makes a node a non-consumer forever.")
    is_ingest = bool(os.getenv("EMBER_SHARDS", "").strip())
    # `full` always consumes; a tests or compress purpose consumes as well.
    is_consumer = (role == "full") or (purpose in ("tests", "compress"))
    from mantle.mesh import sync as _sync0
    print(json.dumps({"aggregator": "up", "transport": "s3", "node": _sync0._node_id(),
                      "role": role, "purpose": purpose, "consumer": is_consumer, "ingest": is_ingest,
                      "env_node": os.getenv("EMBER_NODE_ID"), "ts": time.time()}), flush=True)
    # A cycle is time-boxed rather than count-bounded. Wall clock is the one budget that means the
    # same thing on a 4-core VPS and a 32-core pod, so every cycle completes, reports, and lets the
    # next cycle re-evaluate; a slow box does less per cycle. A count budget sizes a cycle in hours
    # on a real corpus, which leaves the daemon silent between its startup banner and its first
    # report, with lag invisible for the whole span.
    cycle_budget = float(os.getenv("EMBER_CYCLE_SECONDS", "120"))
    sync_share = float(os.getenv("EMBER_SYNC_SHARE", "0.6"))   # of the cycle, leaving room to promote
    rate = None                       # measured docs/sec, learned from this node's own last cycle
    # Self-healing. A node that is behind and applying nothing already holds both numbers, so it
    # acts on them here rather than waiting to be noticed.
    #
    # Escalation, cheapest first, because a slow recovery costs less than a wrong self-restart:
    #   1-2 stalls -> back off. A store that is timing out is asked for less, so the next cycle is
    #                 small enough to complete.
    # 3+ stalls -> exit nonzero. supervisord (Linux) and the.cmd launcher already restart this
    #                 process, so crash-only recovery reuses the supervisor that exists. A fresh
    #                 process drops every wedged HTTP connection and pool, which is what a
    #                 timed-out store needs.
    stall_limit = int(os.getenv("EMBER_STALL_LIMIT", "3"))
    stalls = 0

    # Three things make this daemon's health signal readable, and each is a separate mechanism:
    #
    #  (a) The pinger below writes every `min(interval, PING_CADENCE_S)`, the same cadence
    # `liveness_window_s` is derived from. One line per cycle would not do: a cycle is
    #      cycle_budget (120s default) plus interval (20s), so the gap of ~95-140s sits past the
    #      window and a working node would read `healthy: false`. The pinger matches the window
    #      rather than the window being widened to match the cycle.
    #
    #  (b) The per-cycle `rec` — `sync_err`, STALL, CRAWL, DEGRADED, segments_behind — is persisted
    # to worker.log as well as printed, because `health` reads that file. Pings alone leave
    #      `recent_errors` and `recent_ticks` at 0 whatever the store is doing.
    #
    # (c) The sync failure path clears `ok`, since `health` counts an error as
    #      `r.get("ok") is False`. See the failure site below.
    #
    # `last_rho` stays None on an aggregator box: this daemon does no heavy scan (serve writes the
    # stats snapshot), so there is no rho to report.
    def _agg_pinger():
        while True:
            time.sleep(min(interval, PING_CADENCE_S))
            try:
                with open(hbp, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ping": True, "ts": time.time(),
                                        "worker": "aggregator"}) + "\n")
            except Exception:
                pass
    threading.Thread(target=_agg_pinger, daemon=True).start()

    cycle = 0
    while True:
        t0 = time.time()
        cycle += 1
        deadline = t0 + cycle_budget
        # `tick` is what `health` keys `recent_ticks` on (`tick is not None and not ping`), so it
        # is carried here to make each aggregator cycle visible to that accounting.
        rec = {"aggregator": True, "tick": cycle, "ts": t0}
        try:
            # Per-cycle slices rather than backlog-sized ones. Catch-up happens across many cycles,
            # each visible, and the cursor makes that resumable.
            #
            # The budget is measured rather than configured: the next cycle is sized from what this
            # node achieved last cycle (budget = rate x time-slice), so a slow box asks for less and
            # a fast box for more, with no per-box constant to hand-tune across a fleet. A fixed
            # count leaves a fast node idle inside its budget while segments_behind rises. Env still
            # overrides for a deliberate one-off.
            #
            # One path: Merkle anti-entropy (op.mesh.reconcile) — publish this tree incrementally
            # and pull only the differing leaves (vertices and edges). `max_leaves=0` on a
            # non-consumer makes the round publish-only, which is the shard-node role. No `_rev`, no
            # segment feed, no cursor.
            ml = 256 if is_consumer else 0
            # Convergence is the priority task: keep reconciling while this node is still pulling
            # differing leaves and there is time in the box, so behind-and-idle is a state the loop
            # leaves as soon as it can.
            applied = fetched = pub_leaves = 0
            peers: dict = {}
            passes = 0
            while True:
                left = (t0 + cycle_budget * sync_share) - time.time()
                if left <= 0:
                    break
                r = genesis.invoke(store, "op.mesh.reconcile",
                                   {"max_leaves": ml, "max_seconds": left})
                res = r.get("result", {}) or {}
                fetched_now = res.get("leaves_fetched", 0) or 0
                applied += res.get("applied", 0) or 0
                fetched += fetched_now
                pub_leaves += res.get("published_leaves", 0) or 0
                peers = res.get("peers", {}) or peers
                passes += 1
                # stop when converged (this pass pulled nothing) or the sync slice is exhausted
                if fetched_now == 0 or time.time() >= t0 + cycle_budget * sync_share:
                    break
            rec["applied"] = applied
            rec["leaves_fetched"] = fetched
            rec["published_leaves"] = pub_leaves
            rec["peers"] = peers
            rec["passes"] = passes
        except Exception as e:
            # Kept long. The error text is the only record of which query failed; a short cap cuts
            # the message inside the store's own JSON envelope and leaves the cause unreadable.
            rec["sync_err"] = str(e)[:4000]
            # `health` counts an error as `ok is False`, so the flag is cleared here; a record
            # carrying a failure string beside `"ok": True` would count as a clean cycle.
            rec["ok"] = False
        rec["sync_s"] = round(time.time() - t0, _SYNC_S_DECIMALS)
        # Learn this node's real throughput (EWMA, so one slow cycle does not collapse the budget
        # and one fast cycle does not overshoot it). Only cycles that moved enough work to be a
        # signal enter the average — see MIN_SAMPLE_MOVED and min_sync_s above.
        _moved = (rec.get("applied") or 0) + (rec.get("published") or 0)
        if _moved >= MIN_SAMPLE_MOVED and rec["sync_s"] > min_sync_s():
            _obs = _moved / rec["sync_s"]
            rate = _obs if rate is None else (0.7 * rate + 0.3 * _obs)
            rec["rate"] = round(rate, 1)
        # `mp`/`mc` are assigned inside the try above, so they are read through `locals` with a
        # default: the block can raise before either exists (a non-numeric EMBER_SYNC_* env var),
        # and a bare reference would take the daemon down inside its own error handler.
        rec["budget"] = locals().get("mc") or locals().get("mp") or 0
        # Content promotion is `content-loop.py`'s, with its own cadence, for two reasons:
        #   1. A job scheduled in another job's leftover time is not scheduled. Run here it was
        # gated on `time.time < deadline` after sync, and every observed cycle reported
        #      `over_budget: true`, so the body executed zero times.
        #   2. One owner per cursor. Two processes driving `op.content.promote` scan the same page,
        #      re-PUT the same refs, and race an unguarded read-modify-write on
        #      `content.promote.cursor`, which can move it backward or past a page neither
        #      finished.
        rec["cycle_s"] = round(time.time() - t0, 1)
        rec["over_budget"] = rec["cycle_s"] > cycle_budget
        # Self-observation: every cycle reports how far behind this node is, per peer per feed.
        # This is the number the 5-minute online SLO is measured against, and without it a fleet can
        # drift while every log line reads healthy.
        try:
            from mantle.mesh import sync as _sync
            lag = _sync.mesh_lag(store)
            rec["segments_behind"] = lag.get("segments_behind")
            rec["converged"] = lag.get("converged")
            if lag.get("stuck"):
                rec["STUCK"] = lag["stuck"]        # a poisoned segment, named on the record
        except Exception as e:
            rec["lag_err"] = str(e)[:120]
        # --- self-heal: the trigger is a wedged local store ---
        # A store that cannot answer its own queries is the condition a restart fixes, because a
        # fresh process is what drops the wedged HTTP connections and pools. "Behind and idle" is
        # not that condition:
        #   - a held cursor on a poison segment leaves that peer behind by design (sync.py), so the
        #     condition would be permanently true in a state the code creates on purpose;
        #   - one idle cycle would then be stall -> exit -> restart -> identical state, and
        #     supervisord's default startretries=3 marks the program FATAL and stops restarting it;
        #   - `lag_err` covers both "my store is wedged" (a restart helps) and "S3 returned 503" (a
        #     restart adds load), so at fleet scale one throttling incident would restart every node
        #     against the service already failing.
        store_err = str(rec.get("sync_err") or "") + " " + str(rec.get("lag_err") or "")
        wedged = any(s in store_err.lower() for s in
                     ("read timed out", "connection refused", "connection reset",
                      "connection aborted", "max retries exceeded", "transaction commit"))
        # Also wedged: a store that answers but crawls, so a single segment apply outruns the whole
        # sync slice. Such a cycle reports `applied: 0` with no exception, so `wedged` above stays
        # false and the escalation never fires on it.
        #
        # The extra term is gross over-budget, which is what separates this from "behind and idle":
        #   - a held cursor on a poison segment returns fast (list keys, break) -> small sync_s, so
        #     the state sync.py creates on purpose stays stable;
        #   - a caught-up node has behind == 0;
        #   - only a store too slow to finish one segment inside 1.5x its entire slice trips.
        # Read against the real cycle log this fires on the stalled cycles and stays quiet on the
        # healthy ones (74s / 60k applied) and on a 98s partial.
        sync_slice = cycle_budget * sync_share
        crawling = (rec["sync_s"] > max(30.0, 1.5 * sync_slice)
                    and not (rec.get("applied") or 0)
                    and (rec.get("segments_behind") or 0) > 0)
        if crawling:
            rec["CRAWL"] = {"sync_s": rec["sync_s"], "slice": round(sync_slice, 1),
                            "behind": rec.get("segments_behind")}
            wedged = True
        if wedged:
            stalls += 1
            rec["STALL"] = stalls
            if stalls >= stall_limit:
                # Restart budget. Crash-only software (Candea & Fox) needs a bounded restart rate,
                # or the loop burns the supervisor's retry allowance and ends in FATAL. The budget
                # lives in the store, because a process-local counter resets on the very restart it
                # is counting.
                import random
                now = time.time()
                b = {}
                try:
                    b = store.artifacts.get_artifact("ember.restart.budget") or {}
                except Exception:
                    pass
                win_start = float(b.get("window_start") or 0) or now
                count = int(b.get("count") or 0)
                if now - win_start > 3600:              # hourly window
                    win_start, count = now, 0
                if count >= int(os.getenv("EMBER_RESTART_BUDGET", "4")):
                    # Out of budget: stay up and degraded rather than flapping, and say so on the
                    # record, so a node that cannot fix itself stays visible.
                    rec["DEGRADED"] = "restart budget exhausted; running at minimum rate"
                    rate = 50.0
                    stalls = 0
                else:
                    try:
                        store.artifacts.put_artifact(
                            {"id": "ember.restart.budget", "content_type": "application/x-ember-state",
                             "state": "committed", "window_start": win_start, "count": count + 1},
                            stamp_rev=False)
                    except Exception:
                        pass
                    rec["SELF_RESTART"] = {"count": count + 1}
                    print(json.dumps(rec), flush=True)
                    try:
                        with open(hbp, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                    except Exception:
                        pass
                    # Decorrelated jitter before exiting, so a fleet-wide cause (S3 hiccup) does not
                    # restart 500 nodes in lockstep against an already-struggling dependency.
                    time.sleep(random.uniform(1.0, 5.0) * (count + 1))
                    return 3
            # Back off: a store that is timing out gets smaller asks rather than retries at size.
            rate = max(50.0, (rate or 200.0) * 0.5)
            rec["backoff_rate"] = round(rate, 1)
        elif _healthy_progress(rec, is_consumer):
            stalls = 0                          # real progress — the node is healthy, forget it
        else:
            # No progress, and not diagnosably wedged either. The counter decays rather than
            # resetting: a one-off idle cycle still clears, while a sustained run of zero-progress
            # cycles keeps climbing. A hard reset lets one borderline cycle between two crawling
            # ones zero the count, so an outage never accumulates enough stalls to escalate.
            stalls = max(0, stalls - 1)
        try:
            with open(hbp, "a", encoding="utf-8") as f:
                # Persist the full record. `sync_err` / STALL / CRAWL / DEGRADED /
                # segments_behind are what `health` reads this file for; liveness is the pinger
                # thread's job rather than this line's.
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        print(json.dumps(rec), flush=True)
        time.sleep(interval)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="ember-worker", description="Ember autonomous self-improvement loop")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between cycles")
    ap.add_argument("--max-ticks", type=int, default=None, help="stop after N cycles (default: forever)")
    ap.add_argument("--ingest", action="store_true",
                    help="ALSO drive the GENESIS curriculum each tick (Ember self-ingests)")
    # Operator work-pool: a self-scaling fleet running operators from a queue (see `pool.py`).
    ap.add_argument("--consolidate-every", type=int, default=0, metavar="N",
                    help="run the cross-source colimit sweep every N ticks (0 = only on a stage "
                         "promotion, the pre-2026-08-25 behaviour). The sweep is a full-corpus "
                         "scan, so keep N well above 1 for a long-running worker.")
    ap.add_argument("--supervise", action="store_true",
                    help="run the AUTOSCALER: schedule operator-tasks + size the pool to the backlog")
    ap.add_argument("--pool", action="store_true",
                    help="run one POOL WORKER: claim a task -> invoke its operator -> complete")
    ap.add_argument("--id", default="w0", help="pool worker id (for --pool)")
    ap.add_argument("--max", type=int, default=None, help="max pool workers (for --supervise)")
    ap.add_argument("--aggregator", action="store_true",
                    help="run the MESH AGGREGATOR daemon: pull + promote + stats (the authoritative box)")
    a = ap.parse_args(argv)
    if a.aggregator:
        return run_aggregator(interval=(a.interval if a.interval != 60.0 else 20.0))
    if a.supervise:
        from ember.runtime import pool
        return pool.supervise(max_workers=a.max, interval=(a.interval if a.interval != 60.0 else 15.0))
    if a.pool:
        from ember.runtime import pool
        return pool.pool_worker(a.id, interval=(a.interval if a.interval != 60.0 else 5.0),
                                max_ticks=a.max_ticks)
    return run(interval=a.interval, max_ticks=a.max_ticks, ingest=a.ingest,
               consolidate_every=a.consolidate_every)


if __name__ == "__main__":
    sys.exit(main())
