"""The work pool, exercised against both backends of the typed task seam.

`pool.py` reaches the queue through eight typed store methods (`_TASK_API`). The
lattice backend (`LatticeArtifactStore`) implements them natively; an ArcadeDB
connection that offers none of them gets `_ArcadeTaskPool`, which holds the
equivalent raw SQL. This file runs the same assertions against both, so a change
that passes on one backend and not the other is a fork, not a fix.

`node-repair.py` checks the store, not the pool — its queue probes skip when the
`task` or `counter` table is absent from the store under test. A store can be
healthy while `claim()` returns `None` forever (contract §5: a queue can go
silently dead while `queue_stats` still reports healthy pending counts), which is
exactly what these tests cover and `node-repair.py` alone does not.

The central regression test is `test_claim_actually_claims`: it pins that
`claim()` returns a task from a queue that holds pending, eligible work.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ember.runtime import pool


# ── locating mantle (ember.cmd puts it on PYTHONPATH in production) ───────────
def _mantle_src() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "agience-mantle" / "src"
        if cand.is_dir():
            return cand
    raise RuntimeError("agience-mantle/src not found above %s" % here)


try:
    sys.path.insert(0, str(_mantle_src()))
    from mantle.db import open_lattice
    _HAVE_LATTICE = True
except Exception:                                            # pragma: no cover
    _HAVE_LATTICE = False


TASK_CT = pool.TASK_CT


def _iso(delta_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()


def _tasks_fixture():
    """The task population every backend is seeded with. One list, two backends —
    so a difference in results is a difference in the CODE, not in the data."""
    out = []
    for i in range(30):
        out.append({"id": "task-pending-%02d" % i, "content_type": TASK_CT,
                    "operator": "op.noop", "arguments": {}, "status": "pending",
                    "priority": i % 6, "task_key": "k-pending-%02d" % i})
    out.append({"id": "task-backoff-future", "content_type": TASK_CT,
                "operator": "op.noop", "arguments": {}, "status": "pending",
                "priority": 9, "task_key": "k-future", "attempts": 1,
                "next_retry_at": _iso(3600)})
    out.append({"id": "task-backoff-past", "content_type": TASK_CT,
                "operator": "op.noop", "arguments": {}, "status": "pending",
                "priority": 9, "task_key": "k-past", "attempts": 1,
                "next_retry_at": _iso(-3600)})
    out.append({"id": "task-claimed-fresh", "content_type": TASK_CT,
                "operator": "op.noop", "arguments": {}, "status": "claimed",
                "priority": 0, "task_key": "k-fresh", "claimed_by": "w1",
                "claimed_at": _iso(-5)})
    out.append({"id": "task-claimed-stale", "content_type": TASK_CT,
                "operator": "op.noop", "arguments": {}, "status": "claimed",
                "priority": 0, "task_key": "k-stale", "claimed_by": "w-dead",
                "claimed_at": _iso(-9999)})
    for i in range(4):
        out.append({"id": "task-done-%02d" % i, "content_type": TASK_CT,
                    "operator": "op.noop", "arguments": {}, "status": "done",
                    "priority": 0, "task_key": "k-done-%02d" % i,
                    "completed_at": _iso(-100 - i), "result": {"n": i}})
    out.append({"id": "task-failed-00", "content_type": TASK_CT,
                "operator": "op.noop", "arguments": {}, "status": "failed",
                "priority": 0, "task_key": "k-failed", "completed_at": _iso(-50),
                "last_error": "boom"})
    out.append({"id": "task-dead-00", "content_type": TASK_CT,
                "operator": "op.noop", "arguments": {}, "status": "dead",
                "priority": 0, "task_key": "k-dead", "completed_at": _iso(-500),
                "attempts": 5})
    return out


EXPECTED_COUNTS = {"pending": 32, "claimed": 2, "done": 4, "failed": 1, "dead": 1}


# ── backend 1: the real lattice store ────────────────────────────────────────
class _Store:
    """Minimal stand-in for `LocalStore` — pool.py only ever touches `.artifacts`."""

    def __init__(self, artifacts, graph=None):
        self.artifacts = artifacts
        self.graph = graph
        self.content = None
        self.keys_dir = None


def _lattice_store(tmp_path):
    L = open_lattice(str(tmp_path / ("pool-%s.db" % uuid.uuid4().hex[:8])),
                     origin="test-origin")
    for doc in _tasks_fixture():
        L.artifacts.put_artifact(doc)
    return _Store(L.artifacts, L.graph)


# ── backend 2: a fake ArcadeDB connection ────────────────────────────────────
class RawSqlConnection:
    """Answers the exact SQL `_ArcadeTaskPool` issues, the way ArcadeDB would.

    An unrecognised statement raises. A fake that returned `[]` for what it does
    not understand would reproduce the defect under test and let a broken
    conversion pass — that is how the original bug hid.

    The class name is load-bearing. `pool._tasks` asks an allowlist question — "is
    this a connection known to execute SQL faithfully?" — rather than a denylist
    question about which classes are known-bad: a denylist stops guarding the
    moment its one entry is deleted, where an allowlist fails closed instead. A
    double that wants the ArcadeDB path exercised must be a `RawSqlConnection`;
    renaming this class makes 26 tests fail loudly, which is the correct
    consequence of falling off the allowlist."""

    def __init__(self, docs):
        self.docs = {d["id"]: dict(d) for d in docs}
        self.seen = []

    def _tasks(self, ct):
        return [d for d in self.docs.values() if d.get("content_type") == ct]

    def query(self, q, params=None, **kw):
        params = params or {}
        ql = " ".join(q.lower().split())
        self.seen.append(ql)
        ct = params.get("ct")
        if "count(*)" in ql and "status = :s" in ql:
            n = sum(1 for d in self._tasks(ct) if d.get("status") == params.get("s"))
            return [{"n": n}]
        if "status = 'pending'" in ql:
            rows = [d for d in self._tasks(ct) if d.get("status") == "pending"]
            lim = int(ql.rsplit("limit", 1)[1].strip()) if "limit" in ql else len(rows)
            return [{"id": d["id"], "priority": d.get("priority"),
                     "next_retry_at": d.get("next_retry_at")} for d in rows[:lim]]
        if "status = 'claimed'" in ql and "select id" in ql:
            return [{"id": d["id"], "claimed_by": d.get("claimed_by"),
                     "claimed_at": d.get("claimed_at"), "operator": d.get("operator"),
                     "task_key": d.get("task_key")}
                    for d in self._tasks(ct) if d.get("status") == "claimed"]
        if "select claimed_by, operator, task_key, claimed_at" in ql:
            return [{"id": d["id"], "claimed_by": d.get("claimed_by"),
                     "operator": d.get("operator"), "task_key": d.get("task_key"),
                     "claimed_at": d.get("claimed_at")}
                    for d in self._tasks(ct) if d.get("status") == "claimed"]
        if "operator, task_key, status, completed_at" in ql:
            rows = [d for d in self._tasks(ct) if d.get("status") == params.get("s")]
            lim = int(ql.rsplit("limit", 1)[1].strip()) if "limit" in ql else len(rows)
            return [{"id": d["id"], "operator": d.get("operator"),
                     "task_key": d.get("task_key"), "status": d.get("status"),
                     "completed_at": d.get("completed_at")} for d in rows[:lim]]
        raise AssertionError("fake ArcadeDB got an unrecognised query: %s" % ql)

    def command(self, q, params=None, **kw):
        params = params or {}
        ql = " ".join(q.lower().split())
        self.seen.append(ql)
        d = self.docs.get(params.get("id"))
        if "set status = 'claimed'" in ql:
            if not d or d.get("status") != "pending":
                return [{"count": 0}]
            d.update(status="claimed", claimed_by=params.get("w"),
                     claimed_at=params.get("t"))
            return [{"count": 1}]
        if "set claimed_at" in ql:
            if not d or d.get("status") != "claimed":
                return [{"count": 0}]
            d["claimed_at"] = params.get("t")
            return [{"count": 1}]
        if "set status = :st" in ql or "set status = 'pending'" in ql:
            if not d or d.get("status") != "claimed":
                return [{"count": 0}]
            d.update(status=params.get("st", "pending"), claimed_by=None,
                     next_retry_at=params.get("nr"))
            return [{"count": 1}]
        raise AssertionError("fake ArcadeDB got an unrecognised command: %s" % ql)


class _FakeArcadeArtifacts:
    """An artifact store with NO typed pool methods — i.e. what `ArcadeArtifactStore`
    looks like to `pool._tasks()`. Forces the `_ArcadeTaskPool` branch."""

    def __init__(self, docs):
        self.c = RawSqlConnection(docs)

    def get_artifact(self, artifact_id):
        d = self.c.docs.get(artifact_id)
        return json.loads(json.dumps(d)) if d else None

    def put_artifact(self, doc, *, stamp_rev=True):
        self.c.docs[doc["id"]] = dict(doc)
        return doc

    def list_artifacts(self, *, content_type=None, **kw):
        for d in self.c.docs.values():
            if content_type is None or d.get("content_type") == content_type:
                yield dict(d)


def _arcade_store():
    return _Store(_FakeArcadeArtifacts(_tasks_fixture()))


# ── the parametrisation: every assertion runs on both backends ───────────────
def _backends():
    out = [pytest.param("arcade", id="arcade")]
    out.append(pytest.param("lattice", id="lattice",
                            marks=pytest.mark.skipif(not _HAVE_LATTICE,
                                                     reason="mantle lattice not importable")))
    return out


@pytest.fixture(params=_backends())
def store(request, tmp_path):
    if request.param == "lattice":
        return _lattice_store(tmp_path)
    return _arcade_store()


# ── site 1+2: claim() ────────────────────────────────────────────────────────
def test_claim_actually_claims(store):
    """Pins that `claim()` returns a task when the queue holds pending, eligible
    work — the queue must never read as drained when it is not."""
    t = pool.claim(store, "w-test")
    assert t is not None, "claim() returned None with 32 pending tasks on the queue"
    assert t["status"] == "claimed"
    assert t["claimed_by"] == "w-test"
    assert t["content_type"] == TASK_CT


def test_claim_is_exclusive(store):
    """Two workers, one task each — never the same task twice."""
    claimed = []
    for i in range(6):
        t = pool.claim(store, "w%d" % i)
        assert t is not None
        claimed.append(t["id"])
    assert len(set(claimed)) == len(claimed), "the same task was claimed twice"


def test_claim_respects_retry_backoff(store):
    """`task-backoff-future` is pending but its `next_retry_at` is an hour out.

    The claim window must carry `next_retry_at` for every row so the backoff
    filter below has something to test — a window that omits it makes every row
    look eligible and defeats retry-with-backoff entirely."""
    for i in range(40):
        t = pool.claim(store, "w%d" % i)
        if t is None:
            break
        assert t["id"] != "task-backoff-future", "claimed a task still in backoff"


def test_claim_returns_none_on_an_empty_queue(store):
    """Draining must terminate, and it must terminate by returning None — not by
    running out of a hardcoded window."""
    seen = 0
    while pool.claim(store, "w-drain") is not None:
        seen += 1
        assert seen <= 100, "claim() never drained"
    assert seen == 31, "expected the 31 eligible pending tasks, drained %d" % seen


def test_claim_window_is_not_hardcoded_to_eight(store):
    """Draining more than 8 tasks proves the claim window honours `_CLAIM_WINDOW`
    rather than a hardcoded limit."""
    n = 0
    while pool.claim(store, "w-drain") is not None:
        n += 1
    assert n > 8, "only %d tasks were ever claimable — the LIMIT 8 bug is back" % n


# ── site 3: renew_lease() ────────────────────────────────────────────────────
def test_renew_lease_reports_whether_it_renewed(store):
    """A False here is information: the lease was already reclaimed and this
    worker is running a task someone else now owns, rather than a fault to
    swallow."""
    assert pool.renew_lease(store, "task-claimed-fresh") is True
    assert pool.renew_lease(store, "task-pending-00") is False, \
        "renewed a lease on a task that was never claimed"
    assert pool.renew_lease(store, "task-does-not-exist") is False


def test_renew_lease_moves_the_lease_forward(store):
    before = store.artifacts.get_artifact("task-claimed-stale")["claimed_at"]
    assert pool.renew_lease(store, "task-claimed-stale") is True
    after = store.artifacts.get_artifact("task-claimed-stale")["claimed_at"]
    assert after > before


# ── site 4+5: reclaim_stale() ────────────────────────────────────────────────
def test_reclaim_stale_returns_only_expired_leases(store):
    n = pool.reclaim_stale(store, lease_s=120.0)
    assert n == 1, "expected exactly the one expired lease, got %d" % n
    assert store.artifacts.get_artifact("task-claimed-stale")["status"] == "pending"
    assert store.artifacts.get_artifact("task-claimed-stale")["claimed_by"] is None
    assert store.artifacts.get_artifact("task-claimed-fresh")["status"] == "claimed"


def test_reclaim_stale_is_idempotent(store):
    assert pool.reclaim_stale(store, lease_s=120.0) == 1
    assert pool.reclaim_stale(store, lease_s=120.0) == 0


def test_a_reclaimed_task_is_claimable_again(store):
    """Self-healing end to end: a dead worker's task returns to the queue AND can
    actually be picked up. Reclaiming into a status the claim scan cannot see
    would look identical in `queue_stats` and lose the task forever."""
    pool.reclaim_stale(store, lease_s=120.0)
    seen = set()
    while True:
        t = pool.claim(store, "w-x")
        if t is None:
            break
        seen.add(t["id"])
    assert "task-claimed-stale" in seen


# ── site 6: queue_stats() ────────────────────────────────────────────────────
def test_queue_stats_counts_each_status(store):
    assert pool.queue_stats(store) == EXPECTED_COUNTS


def test_queue_stats_follows_a_claim(store):
    t = pool.claim(store, "w-1")
    assert t is not None
    after = pool.queue_stats(store)
    assert after["pending"] == EXPECTED_COUNTS["pending"] - 1
    assert after["claimed"] == EXPECTED_COUNTS["claimed"] + 1


def test_queue_stats_and_claim_agree(store):
    """The contract §5 pathology, asserted directly: `queue_stats` reporting
    healthy pending counts while `claim` returns nothing is unrepresentable."""
    stats = pool.queue_stats(store)
    if stats["pending"] > 0:
        assert pool.claim(store, "w-agree") is not None, (
            "queue_stats says %d pending but claim() found nothing — this is the "
            "silent-dead-queue defect" % stats["pending"])


# ── site 7+8: pool_status() ──────────────────────────────────────────────────
def test_pool_status_shows_active_claims(store):
    st = pool.pool_status(store)
    workers = {a["worker"] for a in st["active"]}
    assert workers == {"w1", "w-dead"}
    assert st["workers_active"] == 2
    assert all(a["age_s"] is not None for a in st["active"])


def test_pool_status_shows_recent_terminal_tasks(store):
    st = pool.pool_status(store, recent=8)
    assert st["recent"], "no recent done/failed tasks reported"
    assert {r["status"] for r in st["recent"]} <= {"done", "failed"}
    assert len(st["recent"]) <= 8
    ages = [r["age_s"] for r in st["recent"] if r["age_s"] is not None]
    assert ages == sorted(ages), "recent tasks are not most-recent-first"


def test_pool_status_recent_honours_its_limit(store):
    assert len(pool.pool_status(store, recent=2)["recent"]) == 2


def test_pool_status_carries_the_queue(store):
    assert pool.pool_status(store)["queue"] == EXPECTED_COUNTS


# ── run_task: an error envelope is a failure ─────────────────────────────────
def test_an_uninvokable_operator_is_a_failure_not_a_success(store, monkeypatch):
    """`genesis.invoke` returns `{"error": ...}` for a missing operator rather than
    raising, so a task naming an uninvokable operator must reach the failure path
    here — the alternative leaves retry, backoff and dead-lettering unreachable
    for the most likely failure in the system."""
    from ember import genesis

    monkeypatch.setattr(genesis, "invoke", lambda s, op, a: {
        "error": "operator '%s' is not invokable" % op, "invokable": []})

    # Claim a known-fresh task by id rather than via pool.claim: the seed's
    # `task-backoff-past` carries attempts=1 and pool.claim's window can return it,
    # which would make `attempts == 1` below depend on which task got claimed. This
    # test is about the error envelope, not task selection.
    tid = "task-pending-01"
    assert pool._tasks(store).try_claim(tid, worker_id="w-bad", now_iso=_iso())
    t = store.artifacts.get_artifact(tid)
    res = pool.run_task(store, t)

    assert res["ok"] is False, "an error envelope was recorded as a successful run"
    after = store.artifacts.get_artifact(t["id"])
    assert after["status"] == "pending", "a failed task must go back for retry"
    assert after["attempts"] == 1, "attempts did not increment — backoff is unreachable"
    assert after["next_retry_at"], "no backoff was applied"


def test_a_poison_task_dead_letters(store, monkeypatch):
    """MAX_ATTEMPTS failures -> status 'dead', never re-run."""
    from ember import genesis
    monkeypatch.setattr(genesis, "invoke", lambda s, op, a: {"error": "nope"})

    # Claim a known-fresh task by id, not via pool.claim: the seed's `task-backoff-past`
    # (priority 9, sorts first) is claim-eligible with attempts=1 already, and claim()
    # shuffles within the top band, so a shuffled claim can start this test at attempts=1
    # and the final count land short of MAX_ATTEMPTS. This test is about the dead-letter
    # path, not claim selection; selection has its own tests.
    tid = "task-pending-00"
    assert pool._tasks(store).try_claim(tid, worker_id="w-poison", now_iso=_iso())
    for _ in range(pool.MAX_ATTEMPTS):
        d = store.artifacts.get_artifact(tid)
        d["next_retry_at"] = None                     # elapse the backoff
        d["status"] = "pending"
        store.artifacts.put_artifact(d)
        assert pool._tasks(store).try_claim(tid, worker_id="w-poison", now_iso=_iso())
        pool.run_task(store, store.artifacts.get_artifact(tid))

    final = store.artifacts.get_artifact(tid)
    assert final["status"] == "dead"
    assert final["attempts"] == pool.MAX_ATTEMPTS
    assert pool.queue_stats(store)["dead"] == EXPECTED_COUNTS["dead"] + 1


def test_a_successful_result_containing_the_word_error_is_still_a_success(store,
                                                                          monkeypatch):
    """Only invoke()'s top-level error envelope is a failure. An operator whose
    result happens to report errors it counted (an audit, say) still counts as
    having run fine."""
    from ember import genesis
    monkeypatch.setattr(genesis, "invoke", lambda s, op, a: {
        "operator": op, "result": {"errors": 3, "error": "3 rows failed validation"}})

    t = pool.claim(store, "w-ok")
    assert pool.run_task(store, t)["ok"] is True
    assert store.artifacts.get_artifact(t["id"])["status"] == "done"


# ── the seam itself ──────────────────────────────────────────────────────────
def test_an_unrecognised_backend_is_refused_loudly():
    """Anything that is neither typed nor a connection known to execute SQL
    faithfully has no usable backend, and `pool._tasks` raises rather than
    returning one anyway.

    An unknown connection handed ArcadeDB SQL would answer with whatever it
    liked, and a queue that answers `[]` is indistinguishable from one that is
    genuinely drained — that is how the queue can go silently dead while
    `queue_stats` still reports healthy pending counts (contract §5). So the
    guard is an allowlist: only a typed store or a connection vouched for as
    faithful is treated as a usable backend."""

    class _Bogus:
        """A connection whose `.query()` answers `[]` no matter what it is asked."""
        def query(self, sql, params=None):
            return []

    class _SeedArtifacts:
        """No typed pool methods, and a connection nobody vouched for."""
        c = _Bogus()

    with pytest.raises(RuntimeError, match="no usable backend"):
        pool._tasks(_Store(_SeedArtifacts()))

    # Partial implementations are the realistic case and must not pass: having some of the
    # typed API is not having it. `_TASK_API` is checked with `all`, so a store that grew
    # `pending_window` but not `try_claim` has no usable backend rather than being half-served.
    class _PartiallyTyped:
        c = _Bogus()

        def pending_window(self, ct, **kw):
            return []

    with pytest.raises(RuntimeError, match="no usable backend"):
        pool._tasks(_Store(_PartiallyTyped()))

    # No connection at all leaves no usable backend either — `None` must not read as "fine".
    class _NoConn:
        pass

    with pytest.raises(RuntimeError, match="no usable backend"):
        pool._tasks(_Store(_NoConn()))


def test_the_shim_is_deleted_and_the_guard_does_not_depend_on_it():
    """`_SqliteConnShim` is gone from the tree, and `pool.py` does not name it
    either — a guard that names one class is a guard that expires when that
    class does."""
    import ast
    import pathlib

    import mantle.shard.sqlite_store as S

    assert not hasattr(S, "_SqliteConnShim")

    # AST, not grep. The comments in `pool.py` still name the shim — the reason the guard
    # has its present shape is the most valuable thing on the page, and a test that forbade the
    # word would delete the explanation to protect the code. So this asserts on executable
    # constants only: docstrings and comments are exempt, string literals in live code are not.
    tree = ast.parse(pathlib.Path(pool.__file__).read_text(encoding="utf-8"))
    docstrings = {id(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                  and isinstance(n.value.value, str)}
    live = [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and n.value == "_SqliteConnShim"
            and id(n) not in docstrings]
    assert not live, ("pool.py compares against the literal '_SqliteConnShim' at line(s) %s — "
                      "the denylist was reinstated, and it is a guard with an expiry date"
                      % [n.lineno for n in live])


@pytest.mark.skipif(not _HAVE_LATTICE, reason="mantle lattice not importable")
def test_lattice_store_uses_its_typed_methods_directly(tmp_path):
    """No adapter in the way — the typed store is the seam."""
    s = _lattice_store(tmp_path)
    assert pool._tasks(s) is s.artifacts


def test_arcade_store_gets_the_sql_adapter():
    s = _arcade_store()
    assert isinstance(pool._tasks(s), pool._ArcadeTaskPool)


def test_no_raw_sql_left_in_the_call_sites():
    """The 8 converted sites must not reach `store.artifacts.c` any more. The only
    permitted `.c` access in the module is inside `_ArcadeTaskPool`, which exists
    precisely to contain it."""
    import ast
    import inspect

    src = inspect.getsource(pool)
    tree = ast.parse(src)
    adapter = {n.name for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "_ArcadeTaskPool"}
    assert adapter, "_ArcadeTaskPool went missing"

    contained = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == "_ArcadeTaskPool")
    contained_lines = set(range(contained.lineno, (contained.end_lineno or 0) + 1))

    offenders = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr == "c"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "artifacts"
                and node.lineno not in contained_lines):
            offenders.append(node.lineno)
    assert not offenders, (
        "store.artifacts.c is still reached outside _ArcadeTaskPool at line(s) %s"
        % offenders)
