"""The operator work-pool — the system's built-in automation.

Everything the universe does is an operator, and the system schedules and runs its own operators
across a dynamically-scaled fleet. Work enters as tasks (operator-invocations) placed on a queue,
and a pool of workers pulls them, so different workers run different operators at the same time —
some download/ingest (op.source.*), some index (op.consolidate.crosswalk), some compress
(op.consolidate.nearvdup), some run hygiene (op.provenance.audit).

Three roles, all deterministic, no models:
  • schedule  genesis state -> the operator-tasks the universe needs run (idempotent).
  • claim     a worker atomically takes one pending task (the store is the coordinator).
  • run       invoke the task's operator; mark done/failed; fitness and provenance flow as usual.

The supervisor scales the fleet to the backlog: a deep queue spawns workers up to a cap, and a
drained queue retires them.

    python -m ember.runtime.worker --supervise [--max N]   # autoscaler: schedule + size the pool
    python -m ember.runtime.worker --pool --id K           # one worker: claim -> invoke -> complete
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# One definition of "is this connection known to execute SQL faithfully?", shared with `stats.py`
# and `content_tier.py`, so every call site reads the same answer about what an empty result means
# (LATTICE-CONTRACT §5).
from ember.surface.stats import _FAITHFUL_CONNS, _raw_ok

TASK_CT = "application/vnd.agience.task+json"
MAX_ATTEMPTS = 5          # after this many failures a task is dead (dead-letter), not retried forever
_CLAIM_WINDOW = 24        # workers shuffle within this many head-of-queue rows to avoid claim contention


def _now() -> str:
    # ONE home for the ISO-UTC timestamp: genesis._now (signal.py delegates the same way).
    from ember import genesis
    return genesis._now()


# ── the typed task seam ──────────────────────────────────────────────────────
# Every queue operation in this module goes through one of eight typed methods.
# No function below builds SQL.
#
# A typed method binds the call site to the backend: rename one and the caller
# raises AttributeError. SQL passed through a pattern-dispatching shim binds
# nothing — a query the shim does not recognise returns `[]`, and an empty queue
# reads exactly like a drained one, so `claim` answers None while
# `queue_stats` reports a healthy pending count (LATTICE-CONTRACT §5).
#
# Two backends satisfy the seam:
#   * `LatticeArtifactStore` implements all eight natively (mantle.db).
#     This is the production backend.
#   * An allowlisted raw connection gets `_ArcadeTaskPool`, which holds ArcadeDB
#     SQL. `test_pool.py` exercises both sides, which is what keeps the allowlist
#     mechanism exercised.

_TASK_API = ("pending_window", "try_claim", "renew_lease", "claimed",
             "active_claims", "recent_terminal", "release", "count_by_status")


def _tasks(store):
    """Resolve the task-pool backend for `store`. What it returns answers a query
    it understands, or raises.

    The guard is an allowlist — `stats._raw_ok`, shared so the rule has one
    definition — and it fails closed: a backend that is neither typed nor a known
    ArcadeDB connection raises, naming what it got. An allowlist holds as new
    backends appear, where a denylist of known-bad classes stops holding as soon
    as one is deleted and every unknown `.query` object falls through to
    ArcadeDB SQL. An unusable queue raises, because an empty queue reads exactly
    like a drained one."""
    a = store.artifacts
    if all(hasattr(a, m) for m in _TASK_API):
        return a                                    # lattice: typed, native
    if not _raw_ok(a):
        c = getattr(a, "c", None)
        raise RuntimeError(
            "the work pool has no usable backend on %s (connection: %s). It "
            "implements neither the typed task API (%s) nor a connection known to "
            "execute SQL faithfully (%s). Refusing rather than returning an empty "
            "queue: [] is indistinguishable from a drained queue, which is exactly "
            "how this defect survived in production (LATTICE-CONTRACT §5). Wire the "
            "backend to mantle.db.LatticeArtifactStore, which implements the "
            "typed task API natively."
            % (type(a).__name__, type(c).__name__ if c is not None else "none",
               ", ".join(_TASK_API), ", ".join(sorted(_FAITHFUL_CONNS))))
    return _ArcadeTaskPool(a)


class _ArcadeTaskPool:
    """The eight typed methods, over ArcadeDB SQL.

    This class is the one place in the module that touches `store.artifacts.c`.
    Two shapes in the SQL below are load-bearing:

      * **status first, and no ORDER BY, on the claim scan.** `content_type`
        matches every task ever created (the queue is never pruned: pending +
        claimed + done + failed + dead) while `status = 'pending'` is the
        selective, indexed predicate. `priority`/`created_time` are not indexed
        on ArcadeDB, so an ORDER BY would materialise that whole set and sort it
        before LIMIT — a sort per worker every 5s, which `claim` then discards
        with `random.shuffle`. (On the lattice backend the ordering is free —
        `ix_t_pending` is `(ct, status, priority DESC, id)` — so `pending_window`
        returns priority-ordered rows there. Callers treat the order as
        unspecified either way.)

      * **bounded, then sorted in Python, for the terminal window.** `ORDER BY
        completed_at DESC` would sort the entire never-pruned done+failed set to
        display 8 rows.
    """

    def __init__(self, artifacts):
        self.a = artifacts

    @staticmethod
    def _count(res) -> int:
        """ArcadeDB reports a modified-row count under one of several keys."""
        if res and isinstance(res[0], dict):
            r = res[0]
            return int(r.get("count", r.get("modified", r.get("total", 0))) or 0)
        return 0

    def pending_window(self, content_type, *, limit=24, now_iso=None):
        rows = self.a.c.query(
            "SELECT id, priority, next_retry_at FROM Artifact "
            "WHERE status = 'pending' AND content_type = :ct "
            f"LIMIT {int(limit)}", {"ct": content_type})
        out = [dict(r) for r in rows if r.get("id")]
        if now_iso is not None:
            # Retry-backoff is applied here so both backends hand the caller an
            # already-eligible window. The lattice store does it in SQL.
            out = [r for r in out
                   if not r.get("next_retry_at") or str(r["next_retry_at"]) <= now_iso]
        return out

    def try_claim(self, task_id, *, worker_id, now_iso):
        won = self._count(self.a.c.command(
            "UPDATE Artifact SET status = 'claimed', claimed_by = :w, claimed_at = :t "
            "WHERE id = :id AND status = 'pending'",
            {"w": worker_id, "t": now_iso, "id": task_id})) >= 1
        if not won:
            return False
        # ArcadeDB reports its modified-count under one of three keys, so the
        # re-read that resolves which happens here, inside the backend that has
        # the ambiguity. `claim` gets one unambiguous boolean.
        t = self.a.get_artifact(task_id)
        return bool(t and t.get("claimed_by") == worker_id)

    def claimed(self, content_type, *, limit=1000):
        rows = self.a.c.query(
            "SELECT id, claimed_at, claimed_by, operator, task_key FROM Artifact "
            "WHERE status = 'claimed' AND content_type = :ct", {"ct": content_type})
        return [dict(r) for r in rows][:int(limit)]

    def active_claims(self, content_type, *, limit=1000):
        rows = self.a.c.query(
            "SELECT claimed_by, operator, task_key, claimed_at FROM Artifact "
            "WHERE status = 'claimed' AND content_type = :ct", {"ct": content_type})
        # claimed_by is not indexed on ArcadeDB, so the small claimed set is sorted here.
        out = [dict(r) for r in rows]
        out.sort(key=lambda r: str(r.get("claimed_by") or ""))
        return out[:int(limit)]

    def recent_terminal(self, content_type, *, limit=8, statuses=("done", "failed")):
        rows = []
        for st in statuses:
            rows += [dict(r) for r in self.a.c.query(
                "SELECT operator, task_key, status, completed_at, result FROM Artifact "
                "WHERE status = :s AND content_type = :ct LIMIT 500",
                {"ct": content_type, "s": st})]
        rows.sort(key=lambda r: str(r.get("completed_at") or ""), reverse=True)
        return rows[:int(limit)]

    def release(self, task_id, *, to_status="pending", next_retry_at=None):
        return self._count(self.a.c.command(
            "UPDATE Artifact SET status = :st, claimed_by = null, next_retry_at = :nr "
            "WHERE id = :id AND status = 'claimed'",
            {"id": task_id, "st": to_status, "nr": next_retry_at})) >= 1

    def renew_lease(self, task_id, *, now_iso):
        return self._count(self.a.c.command(
            "UPDATE Artifact SET claimed_at = :t WHERE id = :id AND status = 'claimed'",
            {"t": now_iso, "id": task_id})) >= 1

    def count_by_status(self, content_type, statuses=("pending", "claimed", "done",
                                                      "failed", "dead")):
        out = {}
        for st in statuses:
            rows = self.a.c.query(
                "SELECT count(*) AS n FROM Artifact WHERE status = :s AND content_type = :ct",
                {"ct": content_type, "s": st})
            out[st] = int(rows[0]["n"]) if rows else 0
        return out


# ── the task queue (the store is the coordinator) ────────────────────────────
def enqueue(store, operator: str, arguments: Optional[Dict[str, Any]] = None, *,
            key: Optional[str] = None, priority: int = 0, cooldown_s: float = 0.0) -> Dict[str, Any]:
    """Place an operator-invocation on the queue. Idempotent by `key`: never enqueues a duplicate
    of an already-active task; with `cooldown_s`, won't re-enqueue a recently-done recurring task."""
    from ember import genesis
    import hashlib
    key = key or (operator + ":" + hashlib.sha256(
        json.dumps(arguments or {}, sort_keys=True).encode()).hexdigest()[:12])
    tid = "task-" + key
    ex = store.artifacts.get_artifact(tid)
    if ex and ex.get("status") in ("pending", "claimed", "running", "dead"):
        # dead = dead-lettered (exhausted retries) — not recreated here; an operator requeues it.
        return {"enqueued": False, "reason": ex.get("status"), "id": tid}
    if ex and ex.get("status") == "done" and cooldown_s > 0:
        try:
            done_ts = datetime.fromisoformat(ex.get("completed_at", "")).timestamp()
            if time.time() - done_ts < cooldown_s:
                return {"enqueued": False, "reason": "cooldown", "id": tid}
        except Exception:
            pass
    genesis._mint(store, {
        "id": tid, "content_type": TASK_CT, "operator": operator, "arguments": arguments or {},
        "status": "pending", "priority": int(priority), "task_key": key,
        "lemmas": [operator], "content": f"task {operator} [{key}]",
        "provenance": genesis.P_HUMAN, "cited_from": genesis.CITE_GENESIS})
    return {"enqueued": True, "id": tid}


def claim(store, worker_id: str) -> Optional[Dict[str, Any]]:
    """Atomically take one pending task. `try_claim` is a compare-and-set on status='pending',
    so exactly one worker wins each task and the rest see False.

    The window comes from `pending_window` on the typed seam (LATTICE-CONTRACT §5), which is what
    guarantees three properties this function depends on: the query either runs or raises, the row
    carries `next_retry_at` so the backoff below has something to filter on, and the window honours
    `_CLAIM_WINDOW`.
    """
    tasks = _tasks(store)
    now_iso = _now()
    # The window comes back with retry-backoff already applied (both backends).
    cands = [r for r in tasks.pending_window(TASK_CT, limit=_CLAIM_WINDOW, now_iso=now_iso)
             if r.get("id")]
    # Shuffle so a big fleet does not race the same head-of-queue rows: each worker tries a
    # different subset, and far fewer UPDATEs are lost. This discards whatever ordering the backend
    # returned, because priority is already honoured by which rows enter the window at all.
    random.shuffle(cands)
    for r in cands:
        tid = r["id"]
        if tasks.try_claim(tid, worker_id=worker_id, now_iso=now_iso):
            t = store.artifacts.get_artifact(tid)
            if t is None:
                # try_claim won the CAS but the artifact is gone: the task index and the
                # document store disagree. That divergence orphans claimed tasks, so it is
                # raised rather than presented as "no work available".
                raise RuntimeError(
                    "claimed task %r has no artifact — the task index and the document "
                    "store disagree" % tid)
            return t
    return None


def renew_lease(store, task_id: str) -> bool:
    """Heartbeat a claimed task's lease so stale-recovery reads the worker as alive and working
    (a shard ingest legitimately runs for many minutes).

    Returns True iff the task was still claimed. A False is information: the lease was already
    reclaimed and this worker is now running a task someone else owns. It is reported rather than
    swallowed, so it stays distinguishable from a healthy heartbeat and from a store fault."""
    return bool(_tasks(store).renew_lease(task_id, now_iso=_now()))


def reclaim_stale(store, *, lease_s: float = 120.0) -> int:
    """Return tasks whose lease has expired (a worker died mid-task, no renewal) to the pending queue
    — self-healing: no task is lost when a worker crashes or is retired. Deterministic."""
    import datetime as _dt
    tasks = _tasks(store)
    now = time.time()
    reset = 0
    for r in tasks.claimed(TASK_CT):
        ca = r.get("claimed_at")
        try:
            age = now - _dt.datetime.fromisoformat(ca).timestamp() if ca else 1e9
        except Exception:
            age = 1e9                      # unparseable/absent lease -> treat as expired
        if age > lease_s:
            # `release` is a compare-and-set on status='claimed', so a worker that renewed
            # its lease in the gap between the read above and this write keeps its task —
            # the release simply reports False and is not counted.
            if tasks.release(r["id"], to_status="pending"):
                reset += 1
    return reset


def complete(store, task_id: str, *, ok: bool = True, result: Any = None) -> None:
    t = store.artifacts.get_artifact(task_id)
    if not t:
        return
    t["status"] = "done" if ok else "failed"
    t["result"] = result
    t["completed_at"] = _now()
    store.artifacts.put_artifact(t)


def _fail(store, task_id: str, error: str) -> bool:
    """Record a failure: increment `attempts`; retry with backoff (status back to pending, gated by
    `next_retry_at`) until MAX_ATTEMPTS, then dead-letter (status='dead', never re-run) so a poison
    task can't spin the fleet forever. Returns True if the task was dead-lettered."""
    t = store.artifacts.get_artifact(task_id)
    if not t:
        return False
    attempts = int(t.get("attempts", 0)) + 1
    t["attempts"] = attempts
    t["last_error"] = error
    if attempts >= MAX_ATTEMPTS:
        t["status"] = "dead"
        t["completed_at"] = _now()
        store.artifacts.put_artifact(t)
        return True
    t["status"] = "pending"
    t["claimed_by"] = None
    t["next_retry_at"] = (datetime.now(timezone.utc) + timedelta(seconds=min(300, 15 * attempts))).isoformat()
    store.artifacts.put_artifact(t)
    return False


def run_task(store, task: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the task's operator — the one way work runs — and mark the task done or failed.

    An `{"error":...}` envelope is a failure, not a result. `genesis.invoke` returns
    `{"error": "operator 'x' is not invokable",...}` rather than raising when an operator is
    missing or carries an unknown `kind`, so a task naming a typo'd, renamed or removed operator
    reaches the failure path here. That is what keeps the retry and dead-letter mechanism reachable
    for the most likely failure in the system: `attempts` increments, backoff applies,
    MAX_ATTEMPTS is hit, and `queue_stats` counts the task where it belongs."""
    from ember import genesis
    op = task.get("operator")
    args = task.get("arguments") or {}
    try:
        res = genesis.invoke(store, op, args)
        # invoke's success envelope is {"operator":..., "result":...}; its failure envelope
        # carries a top-level "error" and no "operator". Do not sniff nested results.
        if isinstance(res, dict) and res.get("error") and not res.get("operator"):
            err = str(res["error"])[:300]
            dead = _fail(store, task["id"], err)
            return {"ok": False, "operator": op, "error": err, "dead": dead}
        complete(store, task["id"], ok=True, result=(res.get("result") if isinstance(res, dict) else res))
        return {"ok": True, "operator": op, "result": res}
    except Exception as e:
        dead = _fail(store, task["id"], str(e)[:300])   # retry-with-backoff, then dead-letter
        return {"ok": False, "operator": op, "error": str(e)[:300], "dead": dead}


def queue_stats(store) -> Dict[str, int]:
    """Queue depth per status. Runs per /status load and per supervise cycle.

    On the lattice backend these are five counter lookups — no scan, no `count(*)`.
    That matters beyond speed: `count(*)` dereferences every record (EXPLAIN shows
    it loading millions of rows to produce one integer), which exhausts memory on
    the acceptor thread."""
    return dict(_tasks(store).count_by_status(TASK_CT))


def _short(op: str) -> str:
    return (op or "").replace("op.source.", "src:").replace("op.consolidate.", "cons:").replace("op.", "")


def pool_status(store, *, recent: int = 8) -> Dict[str, Any]:
    """The fleet view for /status: queue counts, who's working on what (active claims), and the
    most recently completed/failed tasks. Read-only."""
    import datetime as _dt
    tasks = _tasks(store)
    now = time.time()

    def _age(iso):
        try:
            return round(now - _dt.datetime.fromisoformat(iso).timestamp())
        except Exception:
            return None

    def _task_label(tk):
        return (tk or "").replace("ingest:", "").split("/")[-1]

    active = [{"worker": r.get("claimed_by"), "op": _short(r.get("operator")),
               "task": _task_label(r.get("task_key")),
               "age_s": _age(r.get("claimed_at"))}
              for r in tasks.active_claims(TASK_CT)]
    # Bounded, then sorted. `ORDER BY completed_at DESC` would sort the entire done+failed set,
    # which is never pruned and grows without bound, to return 8 rows. On the lattice backend
    # `ix_t_terminal` is (ct, status, completed_at DESC), so each status bucket yields its top-N
    # as an index walk with no sort at all.
    done = [{"op": _short(r.get("operator")), "task": _task_label(r.get("task_key")),
             "status": r.get("status"), "age_s": _age(r.get("completed_at"))}
            for r in tasks.recent_terminal(TASK_CT, limit=int(recent))]
    # peering: which machine each worker is on (encoded in its id — lt2-* is the 22-core peer).
    from collections import Counter
    def _machine(wid):
        wid = wid or ""
        if wid.startswith("lt2"):
            return "peer .45 (22-core)"
        if wid.startswith("w"):
            return "this box .71"
        return "other"
    by_machine = dict(Counter(_machine(a["worker"]) for a in active))
    return {"queue": queue_stats(store), "active": active, "recent": done,
            "workers_active": len(active), "by_machine": by_machine,
            "machines": len(by_machine)}


# ── schedule — genesis state -> the operator-tasks to run (the automation) ────
def schedule(store) -> Dict[str, Any]:
    """Enqueue the operators the universe needs run. Deterministic + idempotent. This is where new
    work is discovered: shards to ingest (parallel), an index to refresh, hygiene to run."""
    from ember import genesis
    enq = 0
    reclaim_stale(store)          # first: return any dead worker's task to the queue (self-heal)
    # 1) ingest — one task per not-done en-Wikipedia shard. EMBER_SHARDS="lo-hi" partitions the
    #    shard set across boxes: this box enqueues only its slice, so a peer ingests a different
    #    range and the two add write capacity without overlapping.
    try:
        shards = genesis.list_shards("wikimedia/wikipedia", "20231101.en")
        done = set(genesis._shards_done(store, "wikipedia-en"))
        rng = os.getenv("EMBER_SHARDS", "")
        lo, hi = (0, len(shards) - 1)
        if "-" in rng:
            try:
                lo, hi = (int(x) for x in rng.split("-", 1))
            except Exception:
                pass
        in_range = set()
        for idx, f in enumerate(shards):
            if lo <= idx <= hi:
                in_range.add(f)
            if f in done or not (lo <= idx <= hi):
                continue
            if enqueue(store, "op.source.wikipedia-en", {"shards": [f], "describe_workers": 1},
                       key="ingest:" + f, priority=5)["enqueued"]:
                enq += 1
        # Self-heal the partition: retire any still-pending shard task outside this box's range,
        # left over from before EMBER_SHARDS was set or from an earlier split. Otherwise a worker
        # claims the other box's half and duplicates its work.
        if "-" in rng:
            for t in store.artifacts.list_artifacts(content_type=TASK_CT):
                if t.get("status") != "pending":
                    continue
                tk = t.get("task_key", "")
                if tk.startswith("ingest:") and tk.replace("ingest:", "") not in in_range:
                    t["status"] = "done"
                    t["result"] = {"retired": "shard outside this box's EMBER_SHARDS range"}
                    t["completed_at"] = _now()
                    store.artifacts.put_artifact(t)
    except Exception:
        pass
    # 2) index — cross-source concept alignment (recurring; cheap re-run finds new matches)
    if enqueue(store, "op.consolidate.crosswalk", {"apply": True}, key="index:crosswalk",
               priority=2, cooldown_s=900)["enqueued"]:
        enq += 1
    # 3) compress — genuine near-dup consolidation (templated stubs) -> ρ actually falls.
    #    Recurring: re-runs find newly-ingested duplicates as shards land.
    #
    # No `threshold` is passed below. This decides the single most consequential thing this
    # system does — when two things are one thing — and a wrong merge leaves no trace, because
    # the evidence that would reveal it is exactly what got merged away.
    #
    # `minhash.py` records the derivation and `genesis.consolidate_nearvdup` honours it: with no
    # threshold stated the boundary is `minhash.merge_boundary` = `1 - 1/(k+1)` = 0.99225 at
    # k=128 hashes — the Jaccard at which the estimator's own one-sigma band still touches 1.0,
    # i.e. the point where the read cannot resolve the pair apart from identical. Nothing is
    # chosen: it is a function of the signature width declared, and widening the instrument
    # tightens it.
    #
    # `genesis.py` honours an explicit caller threshold over the derivation, so passing one here
    # would re-impose a fixed cut on every recurring compress job, across the system's two
    # largest collections, permanently. `minhash.py` records the estimator's standard error at
    # J=0.85 as ±0.0316 — coarser than the gaps between the constants once considered here — and
    # the candidate-score distribution has no valley (a smooth decay 0.2 -> 0.94 measured over
    # 1,661 pairs, none reaching 0.95): there is no cut to find, so none is passed.
    #
    # Stating no threshold is not "the default"; it is the instrument's own resolution, which is
    # the only boundary the evidence supports ([[compression-threshold-entroptics]]).
    for col in ("stage.2.world", "stage.1.grammar"):
        if enqueue(store, "op.consolidate.nearvdup",
                   {"collection_id": col, "apply": True},
                   key="compress:" + col, priority=3, cooldown_s=1200)["enqueued"]:
            enq += 1
    # 4) hygiene — provenance backfill (recurring)
    if enqueue(store, "op.provenance.audit", {"backfill": True}, key="hygiene:provenance",
               priority=1, cooldown_s=1800)["enqueued"]:
        enq += 1
    # 4b) content promote — copy local content-cache ciphertext up to the OVH origin (durable shared store).
    #     A no-op if OVH isn't configured. Its own async task so ingest workers never block on the WAN.
    if enqueue(store, "op.content.promote", {"max_refs": 2000, "page": 200}, key="content:promote",
               priority=1, cooldown_s=30)["enqueued"]:
        enq += 1
    # 5) mesh pull — if this box has peers (EMBER_PEERS), replicate their artifacts into here so the
    #    local store stays authoritative. Bounded + cursor-resumable, so re-running drains the peers.
    #    A no-op on boxes without peers (the ingest shards). High priority, so this box's authority
    #    does not lag its peers.
    if os.getenv("EMBER_PEERS", "").strip():
        if enqueue(store, "op.mesh.pull", {"max_pages": 20, "page": 200}, key="mesh:pull",
                   priority=4, cooldown_s=20)["enqueued"]:
            enq += 1
    return {"enqueued": enq}


# ── a pool worker — claim -> invoke operator -> complete ─────────────────────
def _hb_path(store) -> Optional[Path]:
    # One derivation for the node directory, in genesis, rather than a second copy of the
    # same reasoning here. Imported lazily because genesis imports this module in turn.
    # None ⇒ this store has no node layout, so there is nowhere to beat a heart: the callers
    # below simply do not log one, which is what they did when the old hardcoded path was
    # unwritable on every box but one.
    from ember.genesis import _node_dir
    nd = _node_dir(store)
    return (nd / "worker.log") if nd else None


def pool_worker(worker_id: str, *, interval: float = 5.0, max_ticks: Optional[int] = None) -> int:
    from mantle.shard.local_store import open_store
    from ember import genesis
    store = open_store()
    # Light startup: the supervisor already froze the schema + registered operators. Registering
    # is best-effort here — a fleet (esp. remote workers over the LAN) racing to upsert the same
    # operator rows causes ConcurrentModification; the ops already exist, so a lost race is fine.
    if store.artifacts.get_artifact("op.status.universe") is None:
        try:
            genesis.register_control_operators(store)
            genesis.register_consolidate_operators(store)
        except Exception as exc:
            # The swallow stays — the race above is real and a lost one is fine — but it is no
            # longer silent. Measured 2026-08-26 while scoping ruling 2 (seal operator content at
            # write time): a seal that raised here would not have failed loudly, it would have made
            # operator registration **stop happening**, turning a visible plaintext defect into an
            # invisible capability loss. A bare `pass` cannot tell a lost upsert race from a node
            # that has quietly stopped registering its operators, and those need different answers.
            #
            # `warning`, not `debug`: `debug` is below the default level, so the record would
            # exist and be INVISIBLE — the distinction PUBLISHING §4.4 draws, and the reason four
            # sibling swallows were promoted.
            # One-line JSON to stdout, because that is what every other report in this file is
            # (`supervisor: up`, `cycle_error`, the per-cycle records). This module has no logger,
            # and adding one for a single line would split its output across two streams the
            # supervisor does not capture the same way.
            print(json.dumps({
                "worker": worker_id, "event": "register_operators_failed",
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:200]),
                "note": "a lost upsert race between workers is expected and harmless; anything "
                        "else means this node is running without its operator catalogue",
            }), flush=True)
    hb = _hb_path(store)
    state = {"task": None, "task_id": None}

    def _ping():
        while True:
            time.sleep(min(interval, 20))
            rec = {"ping": True, "ts": time.time(), "worker": worker_id, "task": state["task"]}
            if state["task_id"]:
                # `renew_lease` reports rather than swallowing a lost lease. Record what it said: a
                # False means this worker's lease was reclaimed while it was still working,
                # which is the signal that `lease_s` is shorter than this operator's real
                # runtime — the condition that makes two workers run the same task. The
                # heartbeat thread must still never die, so a store fault is caught here and
                # recorded as `lease_error` rather than silently ending the heartbeat.
                try:
                    rec["lease"] = renew_lease(store, state["task_id"])
                except Exception as e:
                    rec["lease"] = False
                    rec["lease_error"] = str(e)[:200]
            try:
                with open(hb, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
    threading.Thread(target=_ping, daemon=True).start()

    tick = 0
    idle = 0
    idle_limit = 48                            # ~4 min of finding no work -> self-exit (orphan safety
                                               # net: if the supervisor died, drained workers don't linger)
    while max_ticks is None or tick < max_ticks:
        tick += 1
        task = claim(store, worker_id)
        if task is None:
            idle += 1
            if idle >= idle_limit:
                break
            schedule(store)                    # nothing to do -> discover + enqueue more work
            time.sleep(interval)
            continue
        idle = 0
        state["task"] = {"op": task.get("operator"), "key": task.get("task_key")}
        state["task_id"] = task["id"]
        t0 = time.time()
        res = run_task(store, task)
        state["task_id"] = None
        rec = {"ts": time.time(), "worker": worker_id, "tick": tick,
               "task": task.get("task_key"), "operator": task.get("operator"),
               "ok": res.get("ok"), "secs": round(time.time() - t0, 1)}
        if not res.get("ok"):
            rec["error"] = res.get("error")
        try:
            with open(hb, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        print(json.dumps(rec), flush=True)
        state["task"] = None
    return 0


# ── the supervisor — schedule and dynamically scale the pool to the backlog ──
def _spawn(worker_id: str) -> subprocess.Popen:
    env = dict(os.environ)
    return subprocess.Popen([sys.executable, "-u", "-m", "ember.runtime.worker", "--pool", "--id", worker_id],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def supervise(*, max_workers: Optional[int] = None, interval: float = 15.0,
              tasks_per_worker: int = 1) -> int:
    """The autoscaler. Each cycle: schedule (enqueue needed tasks), then size the fleet to the
    backlog — spawn pool workers when tasks are pending (up to `max_workers`), retire idle ones
    when the queue drains. Dynamic scaling, within the system."""
    from mantle.shard.local_store import open_store
    from ember import genesis
    # Startup must be resilient too: under load the initial bootstrap can hit a store read
    # timeout (observed on ArcadeDB) before the loop guard begins — orphaning nothing yet, but
    # leaving no fleet running. Retry a few times, then proceed (bootstrap is idempotent; the
    # loop's schedule will re-run it if needed).
    store = open_store()
    for _attempt in range(5):
        try:
            genesis.bootstrap(store)
            break
        except Exception as e:
            print(json.dumps({"supervisor": "bootstrap_retry", "attempt": _attempt,
                              "error": str(e)[:150], "ts": time.time()}), flush=True)
            time.sleep(3 * (_attempt + 1))
    # Ember self-sizes the fleet to the box's actual resources (cgroup-aware — cpu_count lies
    # inside a container). resource.pool_workers returns the right value (32 on a 32-core
    # runpod); use it unless an explicit override is passed.
    if max_workers:
        cap = max_workers
    else:
        try:
            from prism import envelope as resource
            cap = resource.pool_workers()
        except Exception:
            cap = max(2, (os.cpu_count() or 4) - 1)
    hb = _hb_path(store)
    children: Dict[str, subprocess.Popen] = {}
    seq = 0
    pid = os.getpid()                               # unique worker-id prefix -> no cross-supervisor collision

    def _shutdown():
        for proc in children.values():
            try:
                proc.terminate()
            except Exception:
                pass
    import atexit
    import signal
    atexit.register(_shutdown)
    for _sig in ("SIGTERM", "SIGINT", "SIGBREAK"):   # terminate children when the supervisor dies
        s = getattr(signal, _sig, None)
        if s is not None:
            try:
                signal.signal(s, lambda *a: (_shutdown(), os._exit(0)))
            except Exception:
                pass

    print(json.dumps({"supervisor": "up", "max_workers": cap, "pid": pid, "ts": time.time()}), flush=True)
    try:
        while True:
            # A single cycle must not crash the supervisor — a transient store read-timeout or
            # ConcurrentModification under heavy load (both observed on ArcadeDB) would otherwise take the whole fleet down with
            # it (the finally reaps every child). Guard the body; a bad cycle just skips to the next.
            try:
                schedule(store)
                stats = queue_stats(store)
                for wid in list(children):               # reap dead children
                    if children[wid].poll() is not None:
                        children.pop(wid, None)
                backlog = stats["pending"] + stats["claimed"]
                target = 0 if backlog == 0 else min(cap, max(1, -(-backlog // tasks_per_worker)))  # ceil
                while len(children) < target:
                    seq += 1
                    wid = f"w{pid}-{seq}"
                    children[wid] = _spawn(wid)
                while len(children) > target:            # scale down (queue drained)
                    wid, proc = children.popitem()
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                rec = {"supervisor": True, "ts": time.time(), "target_workers": target,
                       "active_workers": len(children), "queue": stats}
                try:
                    with open(hb, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
                print(json.dumps(rec), flush=True)
            except Exception as e:
                print(json.dumps({"supervisor": "cycle_error", "error": str(e)[:200],
                                  "ts": time.time()}), flush=True)
            time.sleep(interval)
    finally:
        _shutdown()
