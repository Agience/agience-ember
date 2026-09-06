"""Invariants for the content drain and the metrics it publishes.

Tests are grouped by the invariant they protect rather than by the function they call, because the
same invariant has been broken in several different places.

Half of them protect one rule: an unmeasured or failed state stays distinguishable from a healthy
one. `0` standing in for None, green for a negative rate, "safe to retire" for an empty cursor, a
bold "ingesting" for a dead process — each is a reading nobody took, rendered as one that was.
"""
import time

import pytest

import _paths

from mantle.shard import content_tier as CT


# ── the walk must not skip rows ────────────────────────────────────────────────────────────────
def _seq_rows(n=500):
    """A batch under proper time: `_seq` is allocated one per write, gap-free and injective, so a
    strict `>` cursor is correct. `test_the_seq_walk_visits_every_row_exactly_once` pins the
    outcome that depends on it."""
    return [{"id": f"a{i:04d}", "content_ref": f"cas/{i:04d}", "_seq": i + 1} for i in range(n)]


def _drain_store(rows, cursor, remote_ok=True, seen=None):
    """A lattice-shaped store for the drain: typed walks plus the mantle-tier surface.

    Returns `(store, arts)`, so a test can assert on the persisted cursor at `arts.cursor`."""
    arts = _TypedArts(rows)
    arts.cursor = dict(cursor)
    return _typed_store(arts, remote_ok=remote_ok, seen=seen), arts


class RawSqlConnection:
    """A faithful stand-in for a live ArcadeDB connection, used here as the allowlist fixture in
    `test_a_bogus_object_exposing_query_is_rejected`.

    The class name is load-bearing (unit F). `stats._raw_ok` is an allowlist of connections known to
    execute SQL faithfully, and `RawSqlConnection` is its only entry, so a double that wants the
    raw-SQL path exercised has to be one by name — the same convention as
    `test_lattice_genesis.test_only_a_real_arcade_connection_is_given_sql`.

    An unrecognised statement raises. A double whose answer to "I did not understand you" is an
    empty list lets a broken conversion pass, because that empty list is the defect under test. A
    faithful connection answers or fails."""

    def __init__(self, rows):
        self.rows = rows

    def query(self, sql, params=None):
        p = params or {}
        if "_rev = :r" in sql:
            return [r for r in self.rows if r["_rev"] == p["r"]]
        if "_rev > :r" in sql:
            out = [r for r in self.rows if r["_rev"] > p["r"] and r["_rev"] <= p["ceil"]]
            return sorted(out, key=lambda r: (r["_rev"], r["id"]))[: p["n"]]
        raise AssertionError("unrecognised statement reached the Arcade double: %r" % sql)


def test_the_cursor_is_not_advanced_past_refs_that_failed():
    """Counting errors is not retrying them. A cursor advanced over a failed ref abandons it, and on
    a box whose local cache holds the only copy that is the copy."""
    st, arts = _drain_store(_seq_rows(50), {"id": "content.promote.cursor",
                                            "id_backfill_done": True, "seq": 0},
                            remote_ok=False)
    res = CT.promote_local_content(st, max_refs=50, page=50, workers=4)
    assert res["errors"] > 0, "precondition: this page must fail"
    assert int(arts.cursor.get("seq") or 0) == 0, "cursor advanced past refs that errored"


def test_a_permanently_failing_ref_does_not_pin_the_cursor_forever():
    """Holding the cursor on any error is right for a transient failure and a livelock for a
    permanent one. Measured on TU and T5, four consecutive cycles each, byte-identical:
        {"promoted":0,"scanned":20000,"already":19995,"errors":5,"last_id":"wiki-simple-41172"}
    Five refs failing every time pins the cursor, so the same 20,000 rows are re-walked forever and
    a box "drains" for hours while promoting zero.

    A retry that has failed identically N times is a loop. After `_STUCK_MAX` rounds the drain
    advances past the poison refs and quarantines them, so the content is accounted for rather than
    lost in a counter."""
    st, arts = _drain_store(_seq_rows(6), {"id": "content.promote.cursor",
                                           "id_backfill_done": True, "seq": 0},
                            remote_ok=False)            # every ref fails, every cycle

    for _ in range(CT._STUCK_MAX - 1):                  # still retrying — the cursor holds
        CT.promote_local_content(st, max_refs=6, page=6, workers=2)
        assert int(arts.cursor.get("seq") or 0) == 0, "cursor advanced while still retrying"
        assert not arts.cursor.get("quarantine"), "quarantined before exhausting the retries"

    res = CT.promote_local_content(st, max_refs=6, page=6, workers=2)   # the giving-up cycle
    assert res["forced_past_errors"] is True
    assert arts.cursor.get("quarantined_total", 0) > 0, "advanced past failures without recording them"
    assert arts.cursor["quarantine"][0].get("why"), "a quarantined ref carries no reason"
    assert int(arts.cursor.get("seq") or 0) > 0, "still pinned after exhausting the retries — livelock"


def test_a_failed_ref_records_why_it_failed():
    """A failure is recorded with its ref and its reason. The cursor-hold treats a permanently
    broken ref and a transient 503 differently, so a count that discards which one occurred leaves
    nothing published that says what stopped a box."""
    st, _ = _drain_store(_seq_rows(3), {"id": "content.promote.cursor",
                                        "id_backfill_done": True, "seq": 0},
                         remote_ok=False)
    res = CT.promote_local_content(st, max_refs=3, page=3, workers=2)
    assert res["last_errors"], "errors were counted but not described"
    assert "origin unavailable" in res["last_errors"][0]["why"]
    assert res["last_errors"][0]["ref"], "a failure was recorded without its ref"


def test_promoted_total_survives_a_present_but_null_counter():
    """`dict.get(k, 0)` returns None when the key is present and null, and `None + int` raises. That
    aborts the cursor write, so a completed page is discarded and re-done every cycle."""
    st, arts = _drain_store(_seq_rows(4), {"id": "content.promote.cursor",
                                           "id_backfill_done": True, "seq": 0,
                                           "promoted_total": None, "scanned_total": None})
    CT.promote_local_content(st, max_refs=4, page=4, workers=2)   # must not raise
    assert isinstance(arts.cursor.get("promoted_total"), int)


# ── a failed measurement must never render as a healthy one ────────────────────────────────────
def test_a_failed_artifact_count_does_not_become_a_rate():
    """An unmeasured artifact count produces no rate and poisons no differencing record.

    Reading it as `snap.get("artifacts") or 0` gives rows_per_min = -11,200,000 on a 5.6M-row node
    and persists 0, so the next cycle reports +11,200,000 — which the renderer colours green. A
    measurement that failed then reads as the healthiest number on the page."""
    from ember.surface import stats

    class Arts:
        def __init__(self):
            self.written = []

        def get_artifact(self, _id):
            return {"artifacts": 5_600_000, "ts": time.time() - 30}

        def put_artifact(self, doc, **kw):
            self.written.append(doc)

    class S:
        artifacts = Arts()

    out = stats._rate_from_history(S(), {"artifacts": None, "ts": time.time()})
    assert "rows_per_min" not in out, "an unmeasured count produced a rate"
    assert not S.artifacts.written, "an unmeasured count poisoned the differencing record"


def test_an_unmeasured_memory_ceiling_is_not_reported_as_measured():
    """`snapshot()` names where its memory ceiling came from, `unmeasured-default` included.

    On Windows both `os.sysconf` and the cgroup paths are absent, so `mem_limit_bytes()` returns its
    8GB literal. Reported as `mem_source: "host"` that would make every derived tunable — heap,
    workers, pool, batch — rest on a number nobody measured."""
    from prism import envelope as R

    src = R.snapshot(".")["mem_source"]
    assert src in ("cgroup", "host", "unmeasured-default")
    if not R._cgroup_mem_bytes() and not R._host_mem_bytes():
        assert src == "unmeasured-default"


# ── the retirement gate must be reachable, and must report the walk that is running ────────────
class _CursorOnly:
    """A store that exposes just the promote cursor — enough for stats._finish."""
    def __init__(self, cur, keys_dir):
        outer = self

        class Arts:
            def get_artifact(self, _id):
                return dict(outer.cur) if _id == "content.promote.cursor" else None

            def put_artifact(self, doc, **kw):
                pass

        self.cur = cur
        self.artifacts = Arts()
        # `_finish` has a long unguarded tail after the drain block: it paths and writes to
        # keys_dir (stats.tmp) and reaches into `store.content`. This double satisfies all of it —
        # keys_dir is a real writable directory (tests pass pytest's tmp_path, which keeps it off
        # any key material), and `content` is present even when it is None.
        self.keys_dir = str(keys_dir)
        self.content = None


def test_the_walk_persists_its_own_exhausted_signal():
    """`exhausted` is written to the cursor, not only returned to the caller. It is the free, exact
    statement that the walk caught up and nothing is left to promote, and persisting it gives
    `stats` something cheap to publish instead of differencing a count that may not be takeable."""
    st, arts = _drain_store([], {"id": "content.promote.cursor",
                                 "id_backfill_done": True, "seq": 0})
    res = CT.promote_local_content(st, max_refs=10, page=10, workers=2)
    assert res["exhausted"] is True
    assert arts.cursor.get("exhausted") is True, "the drain-complete signal was not persisted"
    assert isinstance(arts.cursor.get("exhausted_at"), float)


def test_the_drain_does_not_wipe_the_backlog_the_health_loop_measured():
    """Two writers, one document. `backlog`/`backlog_at` are written by health-loop's 900s pass, and
    this cursor is rewritten by the drain every ~90s with a full `put_artifact`. A write that drops
    them leaves the field alive for ~90 of every 900 seconds, so a box publishes "backlog not
    measured" while its own health log shows the count succeeding in 1.6-3.9s.

    Carrying the fields from the `cur` read at the top of the function does not close it: a drain
    cycle runs ~75s and health writes inside that window, so the stale copy carries an absence and
    wipes the measurement the same way. The write therefore takes a fresh read immediately before
    it. This test injects the backlog after the cycle has started, which is the case that separates
    the two."""
    st, arts = _drain_store([], {"id": "content.promote.cursor",
                                 "id_backfill_done": True, "seq": 0})

    # health lands its 900s measurement mid-cycle, after `cur` was captured
    calls = {"n": 0}

    def _get(_id):
        calls["n"] += 1
        if calls["n"] == 1:                     # the read at the top of promote_local_content
            return dict(arts.cursor)
        arts.cursor.update({"backlog": 436315, "backlog_at": 1784559669.7,
                            "backlog_phase": "backfill"})
        return dict(arts.cursor)

    arts.get_artifact = _get
    CT.promote_local_content(st, max_refs=10, page=10, workers=2)
    assert arts.cursor.get("backlog") == 436315, "the drain wiped a backlog written mid-cycle"
    assert arts.cursor.get("backlog_at") == 1784559669.7
    assert arts.cursor.get("backlog_phase") == "backfill"


def test_exhausted_is_never_set_on_a_page_where_promotion_failed():
    """A page where every promotion errored also ends the pass. Recording that as `exhausted` is
    how a box gets retired holding the only copy of its content."""
    st, arts = _drain_store(_seq_rows(5), {"id": "content.promote.cursor",
                                           "id_backfill_done": True, "seq": 0},
                            remote_ok=False)
    CT.promote_local_content(st, max_refs=5, page=5, workers=2)
    assert not arts.cursor.get("exhausted"), "a fully-failed page was recorded as drained"


def test_the_retirement_gate_is_reachable_at_all(tmp_path):
    """A genuinely drained box reports `drain_done`, and the gate reaches that state.

    Requiring a measured `backlog == 0` counted as `id > last_id AND content_ref IS NOT NULL`
    against the backfill cursor makes it unreachable: that cursor is "" once the backfill finishes,
    so the predicate reads "every row that has content" — 2.19M rows on 45, and never zero."""
    from ember.surface import stats

    snap = {}
    stats._finish(_CursorOnly({"id_backfill_done": True, "exhausted": True, "rev": 123,
                               "backlog": 0}, tmp_path), snap)
    assert snap["drain_done"] is True, "a genuinely drained box still cannot report drained"

    # An unmeasured backlog leaves the answer available: requiring a number that is never produced
    # is what makes a gate unreachable.
    snap2 = {}
    stats._finish(_CursorOnly({"id_backfill_done": True, "exhausted": True, "rev": 9}, tmp_path), snap2)
    assert snap2["drain_done"] is True

    # ... but a walk that has not caught up is not done, whatever the backlog says.
    snap3 = {}
    stats._finish(_CursorOnly({"id_backfill_done": True, "exhausted": False, "rev": 9}, tmp_path), snap3)
    assert snap3["drain_done"] is False

    # A cursor carrying no `exhausted` key is unknown, which is None rather than True.
    snap4 = {}
    stats._finish(_CursorOnly({"id_backfill_done": True, "rev": 9}, tmp_path), snap4)
    assert snap4["drain_done"] is None


def test_the_published_cursor_tracks_the_walk_that_is_actually_running(tmp_path):
    """`drain_cursor` publishes the position of the walk that is running.

    The one-time backfill's `last_id` stops advancing once that pass completes, so publishing it on
    a box past its backfill shows a dead id or None while the drain runs normally — a number that
    never moves, in the field added to show movement."""
    from ember.surface import stats

    snap = {}
    stats._finish(_CursorOnly({"id_backfill_done": True, "last_id": "",
                               "rev": 1700000000000000000}, tmp_path), snap)
    assert snap["drain_cursor"], "a running drain published no cursor at all"
    assert snap["drain_rev"] == 1700000000000000000


# ── a row with no `_rev` is invisible to the drain — close it where rows enter ─────────────────
def test_a_consumed_row_with_no_rev_is_NOT_stamped_on_apply():
    """Apply leaves a rev-less consumed doc rev-less, and preserves an origin revision it carries.

    Stamping a `_rev` on the way in disables the anti-downgrade guard in `arcade.py::_keep`, which
    holds an incoming doc with no `_rev` against a local row that has one — an unversioned copy
    cannot be shown to be newer. ~97.6% of rows are rev-less and the mesh/graph backlog is made of
    exactly those, so a stamp lets ancient copies beat current rows fleet-wide and consume writes
    with `stamp_rev=False`, which means the downgraded row is never re-shipped."""
    from mantle.mesh import sync

    class Arts:
        def __init__(self):
            self.seen = None

        def put_many(self, docs, **kw):
            self.seen = [dict(d) for d in docs]
            return len(self.seen)

    class S:
        artifacts = Arts()

    st = S()
    sync._apply_artifacts(st, [{"id": "a", "content_type": "text/markdown"},
                               {"id": "b", "content_type": "text/markdown", "_rev": 42}])
    got = {d["id"]: d.get("_rev") for d in st.artifacts.seen}
    assert got["b"] == 42, "an origin revision was overwritten — rows will echo around the mesh"
    assert not got.get("a"), (
        "a rev-less consumed doc was stamped: this defeats arcade._keep and lets ancient "
        "backlog copies overwrite current rows fleet-wide")


# ── an unmeasured disk is not a full one ──────────────────────────────────────────────────────
def test_an_unreadable_data_volume_does_not_collapse_the_cache_cap():
    """A failed stat yields None where a genuinely full disk yields 0, because those are different
    facts and the number flows onward. `disk_free_bytes` feeds `content_cache_cap_gb`, and a 0
    collapses it to the 2GB floor, so an eviction pass targeting that clears essentially the whole
    local content cache on the strength of a directory being missing for a moment."""
    from prism import envelope as R

    missing = "Q:/no/such/volume/definitely-not-here"
    assert R.disk_free_bytes(missing) is None
    assert R.content_cache_cap_gb(missing) is None, "unmeasured free space became the 2GB floor"
    assert R.snapshot(missing)["disk_free_gb"] is None      # and snapshot() must not raise
    assert isinstance(R.content_cache_cap_gb("."), int)     # the measurable case still measures


# ── exactly one publisher ─────────────────────────────────────────────────────────────────────
def test_serve_does_not_run_a_second_stats_publisher():
    """`_fleet/peers/71/ember/health-loop.py` owns publishing, and serve runs no second publisher.

    Both would call `write_stats`, which read-modify-writes the shared `stats.prev` differencing
    record, so two unsynchronized loops difference against samples the other just overwrote. A copy
    inside serve also wrapped every publish in `except Exception: pass`, so its failures were
    invisible."""
    import pathlib

    src = _paths.read_ember_src("serve.py")
    assert "_publish_loop" not in src, "a second stats publisher was reinstated in serve.py"
    assert "publish_public_status" not in src, "serve is publishing the public status page again"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# LATTICE units C + D — the typed-store conversion of stats.py and content_tier.py.
#
# Each test below covers a call site where a partially-implemented connection can answer with a
# value instead of an error (LATTICE-CONTRACT §5). The through-line is the one this file opens
# with: an unmeasured or failed state stays distinguishable from a healthy one. Here the
# healthy-looking value is `[]` or `count(corpus)` rather than `0` or `green`.
# ══════════════════════════════════════════════════════════════════════════════════════════════


class _UnknownConn:
    """Any object exposing `.query()` that is not a known-faithful connection.

    Named for what it is rather than for one instance of it, because `stats._raw_ok` is an
    allowlist. The property under test is "only what is trusted is recognised", which holds for
    every partially-implemented connection rather than for one named class — a denylist naming a
    single class passes everything the moment that class is deleted.

    Its behaviour is the one that makes such a connection dangerous: `[]` for everything, so "no
    rows" and "I did not understand the question" are the same value."""

    def query(self, sql, params=None):
        return []                    # its answer to EVERYTHING it does not recognise


class _TypedArts:
    """A store exposing the typed lattice surface, to the signatures in `vertex.py`.

    A fake, deliberately: unit L already tests the real store, and what needs testing here is that
    these call sites use the contract correctly — that `exhaustive=False` is honoured, that
    `publish_backlog` is fed the seq cursor and not the id one. `test_the_real_lattice_store_drains`
    runs the same paths against the genuine store whenever mantle is importable, so the fake
    cannot drift unnoticed."""

    def __init__(self, rows=(), origin="node-test", by_ct=None):
        self.rows = list(rows)                    # [{"id","content_ref","_seq"}]
        self.origin = origin
        self.by_ct = by_ct or {}
        self.cursor = {"id": "content.promote.cursor"}
        # No `high_water`: its only reader was `publish_backlog`'s subtraction, and that method no
        # longer exists on the real store.

    # --- the typed methods the two units call ---
    def page_by_id(self, *, after="", limit=200):
        return [r for r in sorted(self.rows, key=lambda r: r["id"]) if r["id"] > after][:limit]

    def page_by_origin(self, *, origin=None, after_seq=0, limit=200):
        return [r for r in sorted(self.rows, key=lambda r: r["_seq"])
                if int(r["_seq"]) > int(after_seq)][:limit]

    # No `publish_backlog`. The real store has none (contract §5.4), and a fake implementing
    # `high_water - cursor` would be a working reference copy of a formula nothing should use.

    def pending_publish(self, *, vertex_cursor=0, edge_cursor=0, origin=None, graph=None,
                        cap=100_000):
        """Counts rows above each feed's own cursor, per `vertex.py`'s signature.

        This fake holds vertices only, so the edge term is a real 0 rather than an omission."""
        v = [r for r in self.rows if int(r.get("_seq") or 0) > int(vertex_cursor)]
        return {"vertex": min(len(v), cap), "edge": 0, "total": min(len(v), cap),
                "exact": len(v) <= cap}

    def count_by_content_type(self, ct):
        return len(self.by_ct.get(ct, []))

    def list_by_content_type(self, ct, *, cap=2000):
        docs = self.by_ct.get(ct, [])
        return (docs[:cap], len(docs) <= cap)

    def list_artifacts(self, *, content_type=None, **kw):
        return iter(self.by_ct.get(content_type, []))

    # --- artifact store basics ---
    def get_artifact(self, _id):
        return dict(self.cursor)

    def put_artifact(self, doc, **kw):
        self.cursor = dict(doc)
        return doc


class _RawArts:
    """A store with no typed methods, only `.c`. That is the shape of a live node, and it is what
    the two backfill tests below need to reach the raw-SQL path."""

    def __init__(self, conn, cursor=None):
        self.c = conn
        self.cursor = dict(cursor or {"id": "content.promote.cursor"})

    def get_artifact(self, _id):
        return dict(self.cursor)

    def put_artifact(self, doc, **kw):
        self.cursor = dict(doc)
        return doc


def _typed_store(arts, remote_ok=True, seen=None):
    seen = seen if seen is not None else set()

    class Remote:
        def exists(self, ref):
            if not remote_ok:
                raise RuntimeError("origin unavailable")
            seen.add(ref)
            return True

    class MTier:
        """The mantle-tier surface the drain drives: `promote_one` / `evict_local`, plus `.remote`
        for the gate. Failure injection lives in `Remote.exists`, so what fails is the origin
        lookup. The unit under test is the drain's cursor discipline; the tier has its own suite in
        mantle."""
        remote = Remote()

        def promote_one(self, ref, *, collection=None):
            return "skip" if self.remote.exists(ref) else "put"

        def evict_local(self, ref):
            return False

    class Store:
        artifacts = arts
        content = None                  # writers' handle — nothing here reads it
        content_tier = MTier()
        keys_dir = None

    return Store()


# ── unit D / content_tier.py:307 — cursor discipline on the one-shot backfill ──────────────────
def test_an_unreadable_backfill_page_does_not_retire_the_backfill():
    """A page that did not load leaves `id_backfill_done` alone.

    `id_backfill_done` is a one-way latch: nothing in the tree writes it back to False, as the
    file's own comment records. Treating an unanswered query as `if not rows: backfill_done = True`
    therefore retires permanently the only walk that can see rows the primary walk cannot, while the
    drain reports a clean cycle. LATTICE-IMPLEMENTATION §1.4 holds a consume cursor at any segment
    that did not apply, and retiring the backfill advances one past the entire corpus."""
    class RawSqlConnection:              # allowlisted, and therefore actually reached (unit F)
        def query(self, sql, params=None):
            raise RuntimeError("ArcadeDB unreachable")

    arts = _RawArts(RawSqlConnection(), {"id": "content.promote.cursor", "id_backfill_done": False})
    res = CT.promote_local_content(_typed_store(arts), max_refs=100, page=10, workers=2)

    assert res["walk_error"], "an unreadable page was not reported"
    assert arts.cursor.get("id_backfill_done") is False, \
        "the backfill was retired by a page that never loaded"
    assert res["exhausted"] is False, "an unreadable walk was reported as caught up"


def test_an_unrecognised_connection_can_never_retire_the_backfill():
    """The live version of the above, stated as the property rather than as one class:

        a connection this system does not know to execute SQL faithfully may not
        produce a value that advances a cursor.

    A partially-implemented backend answers `[]` to a statement it does not recognise, and on the
    first drain cycle that empty page would complete the backfill having read zero rows.
    `_UnknownConn` stands for the whole class of such connections, which is what the allowlist
    catches before one reaches a one-way latch like `id_backfill_done`."""
    arts = _RawArts(_UnknownConn(), {"id": "content.promote.cursor", "id_backfill_done": False})
    res = CT.promote_local_content(_typed_store(arts), max_refs=100, page=10, workers=2)

    assert res["walk_error"], "an unrecognised connection's [] was accepted as an answer"
    assert arts.cursor.get("id_backfill_done") is False
    assert res["exhausted"] is False


# ── unit F / the allowlist itself ─────────────────────────────────────────────────────────────
def test_a_bogus_object_exposing_query_is_rejected():
    """`_raw_ok` admits only what it recognises, so an unknown `.query()` holder is not trusted.

    A predicate of the form `type(c).__name__ != "<one class>"` returns True for everything as soon
    as that class stops existing, and the next object with a `.query()` method walks through. An
    allowlist answers the question the guard is asked. Three objects, none of them recognised:"""
    from ember.surface import stats

    class _Bogus:                                   # answers [] to everything
        def query(self, sql, params=None):
            return []

    class LatticeArtifactStore:                     # a plausible near-miss name
        def query(self, sql, params=None):
            return []

    class _NotEvenAConnection:                      # `.query` is not a method at all
        query = None

    for conn in (_Bogus(), LatticeArtifactStore(), _NotEvenAConnection(), object()):
        arts = _RawArts(conn)
        assert stats._raw_ok(arts) is False, \
            "%s was trusted with raw SQL — the allowlist has a hole" % type(conn).__name__

    # ...and the allowlist admits what it recognises. A guard that blocked everything would quietly
    # retire the ArcadeDB path the live fleet runs on.
    assert stats._raw_ok(_RawArts(RawSqlConnection([]))) is True
    assert stats._raw_ok(_RawArts(None)) is False       # no connection at all


def test_the_deleted_shim_is_gone_from_the_tree():
    """Phase 1.5 exit criterion: no `_SqliteConnShim`. Asserted against the module rather than a
    grep, because an import is what a caller would actually do."""
    import mantle.shard.sqlite_store as S

    assert not hasattr(S, "_SqliteConnShim"), "the shim class is back"
    assert not hasattr(S.SqliteArtifactStore, "c"), "the artifact store's `.c` property is back"
    assert not hasattr(S.SqliteGraphStore, "c"), "the graph store's `.c` property is back"


def test_an_empty_backfill_page_that_DID_load_still_retires_it():
    """The other half, and the reason the walk returns a load status alongside its rows.
    Empty-because-finished is an expected state and still retires the backfill; without it the
    backfill never completes and the id walk runs forever."""
    arts = _TypedArts(rows=[])                       # typed, genuinely empty
    arts.cursor = {"id": "content.promote.cursor", "id_backfill_done": False}
    res = CT.promote_local_content(_typed_store(arts), max_refs=100, page=10, workers=2)
    assert arts.cursor.get("id_backfill_done") is True
    assert not res["walk_error"]


# ── unit D / content_tier.py:331 + :339 — the group drain that stops existing ──────────────────
def test_the_seq_walk_visits_every_row_exactly_once():
    """Every ref is visited exactly once across repeated cycles.

    `_seq` is allocated from the store, gap-free and injective, so a strict `>` cursor is correct
    and the batch below shares no cursor value at all. A timestamp-based revision does not have that
    property — 2000 successive `time.time_ns()` calls on this hardware yield one distinct value, so
    a 500-row batch shares a revision and a bare `_rev > R` cursor drops whatever the first page did
    not return.

    The assertion is the outcome rather than the mechanism, so it keeps its meaning if the walk
    changes."""
    rows = [{"id": "a%04d" % i, "content_ref": "cas/%04d" % i, "_seq": i + 1} for i in range(500)]
    arts = _TypedArts(rows)
    arts.cursor = {"id": "content.promote.cursor", "id_backfill_done": True, "seq": 0}
    seen = set()
    st = _typed_store(arts, seen=seen)
    for _ in range(8):
        CT.promote_local_content(st, max_refs=200, page=200, workers=4)
    assert len(seen) == 500, f"{500 - len(seen)} refs never visited — content loss"


def test_the_seq_cursor_is_persisted_and_no_dead_rev_fields_are():
    """`rev` / `rev_id` / `grp_done` describe a duplicate-revision-group state that gap-free `_seq`
    cannot produce. Persisted, they leave `stats.drain_cursor` rendering a position nothing is
    walking, which is the frozen-`last_id` shape one field over."""
    rows = [{"id": "a%03d" % i, "content_ref": "cas/%03d" % i, "_seq": i + 1} for i in range(5)]
    arts = _TypedArts(rows)
    arts.cursor = {"id": "content.promote.cursor", "id_backfill_done": True, "seq": 0}
    res = CT.promote_local_content(_typed_store(arts), max_refs=10, page=10, workers=2)

    assert res["walk"] == "seq"
    assert arts.cursor["seq"] == 5, "the seq cursor did not advance over a page that applied"
    for dead in ("rev", "rev_id", "grp_done"):
        assert dead not in arts.cursor, "dead `_rev` field %r written by the seq walk" % dead


def test_the_rev_machinery_is_gone_from_the_tree():
    """One walk, one cursor: none of the `_rev`-era machinery is present on the module.

    Asserted against the module rather than by grep, because resurrecting any of it puts two walks
    on one cursor and they disagree about where it is."""
    for dead in ("_rev_page", "_rev_ceiling", "_REV_SLACK_NS", "_RECACHED", "evict_recached",
                 "TieredContentStore"):
        assert not hasattr(CT, dead), "deleted `_rev`-era machinery is back: %s" % dead


# ── where the publish backlog is covered ─────────────────────────────────────────────────────────
# The publish backlog is Merkle-native: `sync.publish_backlog_now(store)` compares two leaf arrays,
# live against published. So it is a leaf count rather than a corpus total, it returns
# `publish_backlog_unset` before any tree exists, it reads live fresh on each call, and it drains to
# zero on a converged node. Its unset case is asserted in `test_the_real_lattice_store_drains`
# below; its drains-to-zero case in `test_sync_lattice::test_steady_state_pulls_zero_leaves`.
#
# The store publishes no `publish_backlog` method (contract §5.4), and mantle's
# `test_publish_backlog_is_gone_and_must_not_come_back` asserts that, so a cursor subtraction coming
# back fails a test rather than a review.


# ── unit C / stats.py:515 — [] served as authoritative ────────────────────────────────────────
def test_an_unfinished_probe_is_not_served_as_the_answer():
    """The typed method returns `(docs, exhaustive)`, so a short list and an unanswered query are
    different values.

    Deciding from the length alone — `if len(rows) <= _CT_FETCH_CAP: return rows` — accepts `[]` as
    the fast answer, because `len([]) <= cap` is true, and the exhaustive `list_artifacts` fallback
    beside it never fires."""
    from ember.surface import stats

    big = [{"id": "d%d" % i} for i in range(stats._CT_FETCH_CAP + 5)]
    arts = _TypedArts(by_ct={"text/markdown": big, "application/x-tiny": [{"id": "t"}]})

    # not exhaustive -> must fall back to the full stream, not serve the truncated probe
    assert len(stats.list_by_content_type(arts, "text/markdown")) == len(big)
    # exhaustive -> the fast answer stands
    assert len(stats.list_by_content_type(arts, "application/x-tiny")) == 1
    # a type with genuinely nothing is still empty, without a fallback lie
    assert stats.list_by_content_type(arts, "application/x-absent") == []


# ── unit C / stats.py:479 — a failed count is not zero ────────────────────────────────────────
def test_a_content_type_count_is_none_when_it_could_not_be_taken():
    """A count that could not be taken is None, and the renderer prints an em-dash for it.
    `wordnet: 0, operators: 0, concepts: 0` as headline totals reads as a measured emptiness.

    The input is any connection off the allowlist, which keeps the domain from shrinking to nothing
    when one class is deleted."""
    from ember.surface import stats

    class Arts:
        c = _UnknownConn()

    class S:
        artifacts = Arts()

    assert stats._count_ct(S(), "text/x-wordnet") is None, "a failed count became 0"

    typed = _TypedArts(by_ct={"text/x-wordnet": [{"id": "w%d" % i} for i in range(7)]})

    class S2:
        artifacts = typed

    assert stats._count_ct(S2(), "text/x-wordnet") == 7
    assert stats._count_ct(S2(), "application/x-absent") == 0     # genuinely zero, and measured


# ── the fake must not drift from the real store ───────────────────────────────────────────────
def test_the_real_lattice_store_drains(tmp_path):
    """The same two paths against the genuine `LatticeArtifactStore`, so `_TypedArts` stays in step
    with the contract it imitates. Skipped rather than passed when mantle is not importable: a check
    that could not run reports as a skip, never as a pass (LATTICE-CONTRACT §7)."""
    import pathlib as _pl
    import sys as _sys

    try:
        from mantle.db import open_lattice
    except Exception:
        p = _paths.repo("agience-mantle") / "src"
        if not p.exists():
            pytest.skip("mantle not importable and not a sibling checkout")
        _sys.path.insert(0, str(p))
        try:
            from mantle.db import open_lattice
        except Exception as exc:
            pytest.skip("mantle not importable: %s" % exc)

    L = open_lattice(str(tmp_path / "lat.db"), origin="node-test")
    # one put_many batch: a timestamp revision would give all 500 rows the same value
    L.artifacts.put_many([{"id": "a%04d" % i, "content_type": "text/markdown",
                           "content_ref": "cas/%04d" % i} for i in range(500)], batch=500)

    seqs = [r["_seq"] for r in L.artifacts.page_by_origin(after_seq=0, limit=1000)]
    assert len(set(seqs)) == len(seqs) == 500, "proper time is not injective"

    # The publish backlog is Merkle-native. Before any tree is published it reads as visibly unset:
    # a 0 would read as caught-up and a corpus total would be the wrong quantity entirely. Its
    # drains-to-zero behaviour is covered by the mesh suite
    # (test_sync_lattice::test_steady_state_pulls_zero_leaves).
    from mantle.mesh import sync
    b = sync.publish_backlog_now(L)
    assert b.get("publish_backlog_unset") is True and "publish_backlog" not in b

    seen = set()
    st = _typed_store(L.artifacts, seen=seen)
    for _ in range(8):
        res = CT.promote_local_content(st, max_refs=200, page=200, workers=4)
        assert not res["walk_error"], res["walk_error"]
    assert res["walk"] == "seq"
    assert len(seen) == 500, "%d refs never visited against the real store" % (500 - len(seen))

    # The fixture has to have churned, or the equality above is vacuous — on an insert-only store a
    # seq subtraction would pass it too. Read from the accounting identity: proper time has run far
    # past the surviving rows.
    acct = L.artifacts.seq_accounting(scan=True)
    rows0 = len(L.artifacts.page_by_origin(after_seq=0, limit=10_000))
    assert acct["vacated"] > 0 and acct["last_seq"] > rows0, (
        "no seqs were vacated, so this test is not exercising the case it exists for: %r" % (acct,))

    # The Merkle-native publish backlog drains to exactly 0 on a converged node, which is what a
    # convergence signal needs. A seq subtraction floors at the number of superseded versions and
    # stays there. The drains-to-zero case is asserted end-to-end in the mesh suite
    # (test_sync_lattice::test_steady_state_pulls_zero_leaves), against a published tree over S3.
