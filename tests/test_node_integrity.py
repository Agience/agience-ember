"""Node integrity suite — `node-repair.py`'s golden invariants, as a named pytest suite.

`agience-cloud/node/node-repair.py` is the node's test suite (standing rule 1): it asserts the lattice store's
integrity invariants and is run after every DB change. It is a hyphenated standalone `main()`
program, which suits an operator CLI but leaves it unimportable, uncollectable by pytest, and named
unlike a test. This file is the same checks repackaged so `pytest` runs them in the gate. It adds
no logic and softens no check: every assertion delegates to a `node-repair` check function and
fails exactly when that check records FAIL, so a failure here is the DB to fix.

`agience-cloud/node/node-repair.py` remains the manual entry point (`python node-repair.py [--fix]
[--sqlite PATH] ...`) and the program `agience-cloud/node/health-loop.py` shells out to. The node's
environment file names it as `node/node-repair.py`, relative to that directory, so the two travel
together. This suite is the collectable twin beside it.

Two layers:
  * Adversarial (always runs, no substrate needed): builds tiny throwaway stores and shows each
    check has teeth — it passes a clean store and fails the specific corruption it targets. A check
    that cannot fail proves nothing, so each of these exercises a real corruption and asserts the
    FAIL is recorded.
  * Live (skips cleanly when no store is present): runs the full golden suite against a real node
    store, exactly as an operator would, and asserts zero FAIL. Gated behind an explicit env var
    (`NODE_REPAIR_TEST_STORE`), so the gate and any serving node are reached only on purpose; the
    live layer performs the same write probes (canary / UPSERT / edge replay) node-repair does.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tempfile

import pytest

# ---------------------------------------------------------------- locate the standalone programs
_HERE = os.path.dirname(os.path.abspath(__file__))
import _paths                      # noqa: E402  (puts src/ on the path, and locates the node tooling)

_NODE = str(_paths.NODE_TOOLING)  # node 71's operator tooling, in the `agience-cloud` sibling
# node-repair's mantle-optional checks (seq accounting cross-check, counter drift, lexical index,
# tail verdict) import `mantle.db`; make it resolvable the way the deployed PYTHONPATH does
# so those checks reach their real implementation instead of taking the honest-SKIP fallback.
# Through `_paths`, not counted from `__file__`. A hand-counted `../..` is right in a developer
# workspace and wrong in CI, where `actions/checkout` puts the siblings under `.siblings/` because
# it cannot write above the workspace. `sys.path.insert` of a directory that does not exist is a
# NO-OP, so the wrong spelling left every mantle-backed check below silently taking its absent-
# dependency fallback while the module reported a pass — which is the condition
# `test_no_test_module_skips_on_a_wrong_path.py` exists to catch, and did.
# A plain call, with no `or ""` fallback: `test_no_test_module_skips_on_a_wrong_path.py` evaluates
# these expressions statically and a BoolOp is not something it can resolve, so the fallback would
# have bought nothing and cost the guard its ability to check this line. `None` is handled below,
# where it is a value rather than part of the path expression.
_MANTLE = _paths.repo_src("agience-mantle")
for _p in (_NODE, _MANTLE):
    _p = str(_p) if _p is not None else ""
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _load(path, modname):
    """Load a standalone `main()` program by path (the `test_lattice_scripts.py` convention).

    node-repair.py is hyphenated and cannot be imported as a package module; loading it by path is
    how its check functions become callable. Its module body only defines functions and stdlib
    imports (the CLI lives under `if __name__ == '__main__'`), so importing it has no side effects.
    """
    if not os.path.exists(path):
        pytest.skip("%s not present" % os.path.basename(path))
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod                       # so intra-module references resolve
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def nr():
    """The node-repair check module."""
    return _load(os.path.join(_NODE, "node-repair.py"), "node_repair")


@pytest.fixture(scope="session")
def sdb():
    """The SQLite transport + contract schema constants (`SqliteDb`, `init_schema`, ...)."""
    return _load(os.path.join(_NODE, "_sqlite_db.py"), "_sqlite_db")


# ---------------------------------------------------------------- helpers
def _run(nr, fn, *args, **kw):
    """Run one check in isolation and return its recorded rows: [(name, status, detail), ...]."""
    nr.QUIET[0] = True                              # keep the suite quiet; we read `results` directly
    nr.results.clear()
    fn(*args, **kw)
    return list(nr.results)


def _fails(nr, rows, name_substr=None):
    """True iff any recorded row is a FAIL (optionally restricted to names containing a substring)."""
    return any(st == nr.BAD and (name_substr is None or name_substr in n)
               for n, st, _ in rows)


def _clean_store(sdb, tmp_path, name="c.db"):
    """A store carrying the contract §2 schema and nothing else."""
    path = str(tmp_path / name)
    sdb.init_schema(path)
    return path


def _leaf_helpers(nr):
    """`(leaf_of, leaves)` for the leaf-placement test — from mantle, never re-derived here.

    This raises rather than skipping. `check_leaf_placement` takes an honest SKIP when
    `mantle.db` is absent, so a helper that skipped alongside it would leave the test green
    with both the clean case and the seeded corruption recorded as SKIP. mantle is on `sys.path`
    here (see the header), so an ImportError means a broken harness.

    `leaf_of` is imported so the test and the check share one derivation; a hand-rolled
    `blake2b(id) % leaves` would only show that two implementations agree with each other.
    """
    try:
        from mantle.db import constants as K
    except Exception as exc:            # pragma: no cover — a broken harness, not a valid state
        raise AssertionError(
            "mantle.db must be importable for the leaf-placement test: check_leaf_placement "
            "SKIPs without it, so skipping here would make this test pass over BOTH the clean store "
            "and the seeded corruption. Fix sys.path, do not skip. (%s: %s)"
            % (type(exc).__name__, exc))
    # The resolution a scratch store actually operates at: `init_schema` records none, so
    # `open_lattice` falls back to the module default — the same value the check will read.
    return K.leaf_of, int(K.DEFAULT_LEAVES)


def _exec(path, *statements):
    con = sqlite3.connect(path)
    for s in statements:
        con.execute(s)
    con.commit()
    con.close()


def _one(path, query):
    """The first column of the first row, or None. For reading a counter back after a repair.

    Opened read-write like `_exec` rather than read-only: these stores are built in `tmp_path` for
    one test, and a second connection mode is a second thing that can differ from what the check
    under test sees."""
    con = sqlite3.connect(path)
    try:
        row = con.execute(query).fetchone()
        return row[0] if row else None
    finally:
        con.close()


# ================================================================ Adversarial layer (always runs)
# Each test shows the packaged check passes a clean store and fails the exact corruption it exists
# to catch. These need no built substrate, so the gate exercises real checks even with no node.

def test_removed_column_reintroduction_is_caught(nr, sdb, tmp_path):
    """§2.2 — a deliberately-removed time-derived column (`_rev`) stays removed.

    Re-adding `_rev` makes a time-derived value a query predicate again, so
    `check_removed_columns` records FAIL on it while a clean schema stays quiet."""
    path = _clean_store(sdb, tmp_path)
    db = sdb.SqliteDb(path, origin="test-node")
    assert not _fails(nr, _run(nr, nr.check_removed_columns, db)), \
        "a clean schema must not report a removed-column violation"

    _exec(path, "ALTER TABLE vertex ADD COLUMN _rev INTEGER")
    db2 = sdb.SqliteDb(path, origin="test-node")
    assert _fails(nr, _run(nr, nr.check_removed_columns, db2)), \
        "reintroducing the removed `_rev` column MUST FAIL §2.2 — it is how a time-derived value " \
        "becomes a query predicate again; the check must not pass over it"
    db.close(); db2.close()


def test_unsanctioned_column_is_caught_by_allowlist(nr, sdb, tmp_path):
    """§2.3, the column test — the schema is an allowlist, so an unrecognised column records FAIL.

    Without it, a frame-local column can be added and go unremarked until it is load-bearing. The
    allowlist turns that from a review question into a suite failure."""
    path = _clean_store(sdb, tmp_path)
    # A clean scratch store carries exactly the sanctioned set with no compensation: SCHEMA_SQL agrees
    # with SANCTIONED_COLUMNS, so init_schema creates vertex.origin_root / vertex.root_id / edge._leaf
    # directly. That agreement is what this baseline measures.
    db = sdb.SqliteDb(path, origin="test-node")
    assert not _fails(nr, _run(nr, nr.check_column_allowlist, db)), \
        "the fully-sanctioned schema (as built by SCHEMA_SQL) must pass the §2.3 allowlist"

    _exec(path, "ALTER TABLE vertex ADD COLUMN sneaky TEXT")
    db2 = sdb.SqliteDb(path, origin="test-node")
    assert _fails(nr, _run(nr, nr.check_column_allowlist, db2)), \
        "an unsanctioned column MUST FAIL the §2.3 COLUMN TEST allowlist, never pass silently"
    db.close(); db2.close()


def test_dangling_created_by_is_caught(nr, sdb, tmp_path):
    """§2.1 — every non-NULL `created_by` must resolve to an existing vertex.

    A `created_by` pointing at nothing raises no error of its own; it breaks grant propagation, so
    the store reads healthy while authorization fails."""
    path = _clean_store(sdb, tmp_path)
    _exec(path,
          "INSERT INTO vertex(id,_origin,_seq) VALUES('person-1','test-node',1)",
          "INSERT INTO vertex(id,created_by,_origin,_seq) VALUES('a1','person-1','test-node',2)")
    db = sdb.SqliteDb(path, origin="test-node")
    assert not _fails(nr, _run(nr, nr.check_created_by_resolves, db)), \
        "a created_by that resolves to a person vertex must pass"

    _exec(path, "INSERT INTO vertex(id,created_by,_origin,_seq) "
                "VALUES('a2','ghost-nowhere','test-node',3)")
    db2 = sdb.SqliteDb(path, origin="test-node")
    assert _fails(nr, _run(nr, nr.check_created_by_resolves, db2)), \
        "a dangling created_by MUST FAIL §2.1 — a reference resolving to nothing silently breaks " \
        "grant propagation for every row citing it"
    db.close(); db2.close()


def test_missing_proper_time_is_caught(nr, sdb, tmp_path):
    """A row with NULL `(_origin, _seq)` has no proper time — it can never publish or merkle-verify.

    Such a row is invisible to the publish scan and to the merkle row_hash: it replicates never
    and verifies never, while still counting toward the row total."""
    path = _clean_store(sdb, tmp_path)
    _exec(path, "INSERT INTO vertex(id,_origin,_seq) VALUES('ok1','test-node',1)")
    db = sdb.SqliteDb(path, origin="test-node")
    assert not _fails(nr, _run(nr, nr.check_proper_time, db, "vertex")), \
        "a row carrying (_origin,_seq) must pass proper-time"

    _exec(path, "INSERT INTO vertex(id,_origin,_seq) VALUES('bad1',NULL,NULL)")
    db2 = sdb.SqliteDb(path, origin="test-node")
    assert _fails(nr, _run(nr, nr.check_proper_time, db2, "vertex")), \
        "a NULL _origin/_seq row MUST FAIL proper-time — it can never publish or merkle-verify"
    db.close(); db2.close()


def test_null_edge_key_is_caught(nr, sdb, tmp_path):
    """§3 / §1.5 — `edge_key` must be non-NULL (SQLite admits NULLs into a non-INTEGER PRIMARY KEY,
    so NULL keys sit OUTSIDE the UNIQUE constraint — unlimited, and duplicable).

    A migration that carries NULL-keyed edges forward duplicates them on both engines."""
    path = _clean_store(sdb, tmp_path)
    _exec(path, "INSERT INTO edge(edge_key,src,dst,label) VALUES(NULL,'s','d','l')")
    db = sdb.SqliteDb(path, origin="test-node")
    rows = _run(nr, nr.check_edge_keys, db)
    assert _fails(nr, rows, "edge_key non-NULL"), \
        "a NULL edge_key MUST FAIL — NULL keys evade the UNIQUE constraint and may already be " \
        "duplicated: %r" % [(n, s) for n, s, _ in rows]
    db.close()


def test_duplicate_origin_seq_is_caught(nr, sdb, tmp_path):
    """`(_origin, _seq)` must be unique across vertex ∪ edge — a repeat is a double-allocated
    proper-time stamp (two SeqAllocators writing over one counter) or a double-applied row.

    Neither cause is benign, so uniqueness records FAIL for every origin rather than WARN."""
    path = _clean_store(sdb, tmp_path)
    _exec(path,
          "INSERT INTO vertex(id,_origin,_seq) VALUES('d1','test-node',1)",
          "INSERT INTO vertex(id,_origin,_seq) VALUES('d2','test-node',1)")   # same (_origin,_seq)
    db = sdb.SqliteDb(path, origin="test-node")
    rows = _run(nr, nr.check_seq_gapfree, db, "test-node", False)
    assert _fails(nr, rows, "(_origin,_seq) unique"), \
        "a duplicate (_origin,_seq) MUST FAIL uniqueness — it is a double-allocated proper-time " \
        "stamp: %r" % [(n, s) for n, s, _ in rows]
    db.close()


def test_misplaced_merkle_leaf_is_caught(nr, sdb, tmp_path):
    """`_leaf` must equal `leaf_of(id, leaves)`.

    This is the invariant an id migration moves. `_leaf` and `row_hash` are both derived from the
    artifact id, so renaming an id moves the row to a different Merkle leaf. Through
    `put_artifact`/`delete_artifact` the write path XORs both leaves and stays exact; in raw SQL
    the row keeps its old `_leaf` and both leaf digests are wrong with nothing else objecting. The
    node then reads fully green while peers compare digests that can never converge, which presents
    as a slow sync rather than as corruption.

    The two ways the check could be wrong are asserted as well: a NULL `_leaf` is legacy-unstamped
    and is not corruption, and a correctly-placed row raises no alarm. A check that flags
    everything is as useless as one that flags nothing.
    """
    leaf_of, leaves = _leaf_helpers(nr)

    path = _clean_store(sdb, tmp_path)
    good, other = "op.gauge.language.en", "wn-dog.n.01"
    _exec(path,
          "INSERT INTO vertex(id,_origin,_seq,_leaf) VALUES('%s','test-node',1,%d)"
          % (good, leaf_of(good, leaves)),
          "INSERT INTO vertex(id,_origin,_seq,_leaf) VALUES('%s','test-node',2,%d)"
          % (other, leaf_of(other, leaves)),
          # legacy, never stamped — computed on read in vertex.py, not corruption
          "INSERT INTO vertex(id,_origin,_seq,_leaf) VALUES('legacy-1','test-node',3,NULL)")
    db = sdb.SqliteDb(path, origin="test-node")
    rows = _run(nr, nr.check_leaf_placement, db, True)
    assert not _fails(nr, rows), (
        "correctly-placed rows (plus one legacy NULL `_leaf`) must PASS — a NULL is unstamped, not "
        "misplaced, and flagging it would make the check unusable on a real store: %r"
        % [(n, s, d) for n, s, d in rows])
    db.close()

    # Now the exact corruption: the shape a raw-SQL id rename leaves behind.
    wrong = (leaf_of(good, leaves) + 1) % leaves
    _exec(path, "UPDATE vertex SET _leaf = %d WHERE id = '%s'" % (wrong, good))
    db2 = sdb.SqliteDb(path, origin="test-node")
    rows2 = _run(nr, nr.check_leaf_placement, db2, True)
    assert _fails(nr, rows2), (
        "a row whose stored `_leaf` disagrees with `leaf_of(id)` MUST FAIL — this is precisely what a "
        "raw-SQL id rename produces, and it makes both leaf digests wrong with nothing else objecting: %r"
        % [(n, s, d) for n, s, d in rows2])
    db2.close()


# ================================================================ Live layer (skips without a store)
# The operator's `node-repair.py` run, as pytest. Gated behind an explicit env var so both the gate
# and a serving node are reached only on purpose.

def _live_store_path():
    p = os.environ.get("NODE_REPAIR_TEST_STORE")
    return p if (p and os.path.exists(p)) else None


@pytest.fixture(scope="session")
def live_results(nr, sdb):
    """Run the full golden suite once against a real store, or skip cleanly if none is declared."""
    path = _live_store_path()
    if not path:
        pytest.skip("no live store: set NODE_REPAIR_TEST_STORE=/path/to/lattice.db to run the "
                    "golden suite against a real node store (unset in the gate → clean skip)")
    origin = (os.environ.get("NODE_REPAIR_TEST_ORIGIN")
              or os.environ.get("EMBER_NODE_ID"))
    if not origin:
        pytest.skip("no observer identity: set NODE_REPAIR_TEST_ORIGIN or EMBER_NODE_ID — _seq "
                    "checks are scoped `WHERE _origin = :me` and cannot run without one")
    deep = os.environ.get("NODE_REPAIR_TEST_DEEP") == "1"
    # keep the live node's real `.node-repair-state.json` untouched: point the state dir at a temp.
    base = tempfile.mkdtemp(prefix="node-integrity-state-")
    db = sdb.SqliteDb(path, origin=origin, timeout=nr.BUDGET_FAST)
    nr.QUIET[0] = True
    nr.results.clear()
    nr.check_all_sqlite(db, {"node": origin, "base": base, "backend": "sqlite"}, deep)
    rows = list(nr.results)
    db.close()
    return rows


def test_live_golden_suite_has_no_fail(nr, live_results):
    """The whole suite is green: no FAIL, and at least one real PASS.

    This mirrors `node-repair.py --json` exit 0, including its rule that a run whose checks all
    skipped is not green — an all-skip run proves nothing."""
    fails = [(n, d) for n, st, d in live_results if st == nr.BAD]
    assert not fails, "the golden node suite reported FAIL(s) — FIX THE DB, never the check: %r" \
        % fails
    passed = [n for n, st, _ in live_results if st == nr.OK]
    assert passed, "no check PASSED (all skipped/failed) — an all-skip run proves nothing and " \
        "must not read as green"


# One named test per invariant group. Each asserts no FAIL among the records it owns; a group with
# no records on this store is legitimately inapplicable (SKIP), not a failure.
_INVARIANTS = [
    ("store_reachable",        lambda n: "store reachable" in n),
    ("lexical_index",          lambda n: "lexical index" in n),
    ("schema_indexes",         lambda n: n.startswith("index ")
                                          and ("exists" in n or "PK" in n)),
    ("seq_cursor_sweep",       lambda n: "cursor sweep" in n),
    ("canary_via_index",       lambda n: "canary via index" in n),
    ("removed_columns",        lambda n: "removed columns" in n),
    ("column_allowlist",       lambda n: "column allowlist" in n),
    ("created_by_resolves",    lambda n: "created_by resolves" in n),
    ("created_by_matches_edge", lambda n: "created_by matches origin edge" in n),
    ("proper_time",            lambda n: "has proper time" in n),
    ("seq_allocation_accounted", lambda n: "allocation accounted" in n),
    ("origin_seq_unique",      lambda n: "(_origin,_seq) unique" in n),
    ("seq_endpoint_verdict",   lambda n: "endpoint verdict" in n),
    ("seq_accounting_crosscheck", lambda n: "accounting cross-check" in n),
    ("index_count_monotonic",  lambda n: "index count monotonic" in n),
    ("id_keyset_pages",        lambda n: "keyset pages" in n or "id index responsive" in n),
    ("upsert_writes",          lambda n: "UPSERT works" in n),
    ("edge_key",               lambda n: "edge_key" in n or "edge write idempotent" in n),
    ("no_fixture_rows",        lambda n: "no fixture rows" in n),
    ("performance_budgets",    lambda n: n.startswith("perf:")),
]


@pytest.mark.parametrize("invariant,pred", _INVARIANTS, ids=[i for i, _ in _INVARIANTS])
def test_live_invariant(nr, live_results, invariant, pred):
    """Each golden invariant, named, asserted FAIL-free against the live store."""
    owned = [(n, st, d) for n, st, d in live_results if pred(n)]
    if not owned:
        pytest.skip("invariant %r was not exercised on this store (inapplicable/--deep only)"
                    % invariant)
    fails = [(n, d) for n, st, d in owned if st == nr.BAD]
    assert not fails, "invariant %r FAILED — FIX THE DB, never the check: %r" % (invariant, fails)

def test_a_row_removed_outside_the_write_path_is_caught(nr, sdb, tmp_path):
    """A raw DELETE removes the row and leaves the bookkeeping believing it is there.

    This is the exact mechanism `node-repair` itself uses. `_sqlite_db.SqliteDb` is a plain
    `sqlite3.connect` on the lattice file, so EVERY statement this suite issues -- `ep="command"`
    included -- goes straight to SQLite and never touches mantle's counters. Three probe sites are
    harmless because they insert raw and delete raw; `purge_fixtures_sqlite` is not, because it
    removes rows the STORE wrote.

    What that costs: `rows:<origin>` is not decremented and `vacated:<origin>` is not incremented,
    so `live_rows + vacated == last_seq` reads short by exactly the number of rows removed --
    `schema.c_vacated` is written around precisely this, so that an accounted removal stays
    distinguishable from an unaccounted one.

    Why this test exists rather than the check being obviously right: on node 71, 2026-08-26, the
    cross-check reported PASS over a store with **6 rows lost** (last_seq 31,686,451, live_rows
    10,405,435 by scan, vacated 21,281,010). It called `seq_accounting(scan=True)` -- the ONE mode
    that can see a lost row -- compared `last_seq`, `live_rows` and `vacated` against its own SQL,
    found them equal, and discarded `balanced`. Two implementations agreeing about a broken store is
    agreement, not health.
    """
    try:
        from mantle.db import open_lattice
    except Exception as exc:            # pragma: no cover -- a broken harness, not a valid state
        raise AssertionError(
            "mantle.db must be importable: check_seq_accounting_agrees SKIPs without it, so "
            "skipping here would leave this test green over BOTH the balanced store and the "
            "seeded loss. Fix sys.path, do not skip. (%s: %s)" % (type(exc).__name__, exc))

    path = str(tmp_path / "lost.db")
    L = open_lattice(path, origin="test-node")
    for i in range(8):
        L.artifacts.put_artifact({"id": "a%04d" % i, "content_type": "text/markdown", "content": "x"})
    L.db.close() if hasattr(L.db, "close") else None

    db = sdb.SqliteDb(path, origin="test-node")
    assert not _fails(nr, _run(nr, nr.check_seq_accounting_agrees, db, "test-node")),         "a store whose rows all went through the write path must BALANCE"
    db.close()

    # The mechanism, in one statement: the row leaves, the bookkeeping does not learn.
    _exec(path, "DELETE FROM vertex WHERE id = 'a0003'")

    db2 = sdb.SqliteDb(path, origin="test-node")
    rows = _run(nr, nr.check_seq_accounting_agrees, db2, "test-node")
    assert _fails(nr, rows), (
        "a row removed OUTSIDE the write path MUST FAIL the cross-check -- this is the node-71 "
        "defect: %r" % rows)
    assert any("LOST" in (d or "") for _n, _s, d in rows),         "the failure must say the rows are lost, not merely that two numbers differ: %r" % rows
    db2.close()


def test_the_repair_reclassifies_lost_rows_rather_than_resetting_the_counter(nr, sdb, tmp_path):
    """`reanchor_seq_counters_sqlite` brings the invariant back by moving rows -> vacated.

    The distinction is the whole point. Setting `rows` to the scan and stopping would make
    `live_rows + vacated == last_seq` fail by exactly as much as before; zeroing a counter or
    lowering `last_seq` would make the arithmetic agree while destroying the evidence of how many
    rows left. Reclassifying the shortfall as VACATED says what actually happened -- a removal
    nobody recorded -- and `schema.c_vacated` exists to carry exactly that.
    """
    try:
        from mantle.db import open_lattice
    except Exception as exc:            # pragma: no cover -- a broken harness, not a valid state
        raise AssertionError("mantle.db must be importable (%s: %s)" % (type(exc).__name__, exc))

    path = str(tmp_path / "reanchor.db")
    L = open_lattice(path, origin="test-node")
    for i in range(8):
        L.artifacts.put_artifact({"id": "a%04d" % i, "content_type": "text/markdown", "content": "x"})
    L.db.close() if hasattr(L.db, "close") else None

    _exec(path, "DELETE FROM vertex WHERE id IN ('a0003', 'a0004')")   # two rows, unaccounted

    db = sdb.SqliteDb(path, origin="test-node")
    assert _fails(nr, _run(nr, nr.check_seq_accounting_agrees, db, "test-node")), (
        "the seeded loss must be visible before the repair, or this test proves nothing")

    # None, not 0, on a store that has never vacated anything: the counter row does not exist
    # until something creates it, which is why the repair UPSERTs rather than UPDATEs.
    before = _one(path, "SELECT n FROM counter WHERE name = 'vacated:test-node'") or 0
    _run(nr, nr.reanchor_seq_counters_sqlite, db, {"base": str(tmp_path)})
    after_rows = _one(path, "SELECT n FROM counter WHERE name = 'rows:test-node'")
    after_vac = _one(path, "SELECT n FROM counter WHERE name = 'vacated:test-node'")
    live = _one(path, "SELECT COUNT(_seq) FROM ("
                      "  SELECT _seq FROM vertex WHERE _origin = 'test-node'"
                      "  UNION SELECT _seq FROM edge WHERE _origin = 'test-node')")

    assert after_rows == live, (
        "`rows` must equal the rows that exist after a re-anchor: counter=%s scan=%s"
        % (after_rows, live))
    assert after_vac == before + 2, (
        "the two removed seqs must be RECLASSIFIED as vacated, not discarded: %s -> %s"
        % (before, after_vac))
    assert not _fails(nr, _run(nr, nr.check_seq_accounting_agrees, db, "test-node")), (
        "the cross-check must pass once the shortfall is accounted for")
    db.close()


def test_the_repair_leaves_a_peer_alone(nr, sdb, tmp_path):
    """A peer's shortfall is replication lag, not loss, and re-anchoring it would erase in-flight state.

    `last_seq` for a peer is a high-water mark learned from whatever arrived while the rows behind it
    are still coming. `is_local` in `seq_counter` is what separates the two, and a repair that
    ignored it would quietly declare every healthy peer's lag to be vacated rows.
    """
    try:
        from mantle.db import open_lattice
    except Exception as exc:            # pragma: no cover
        raise AssertionError("mantle.db must be importable (%s: %s)" % (type(exc).__name__, exc))

    path = str(tmp_path / "peer.db")
    L = open_lattice(path, origin="test-node")
    for i in range(4):
        L.artifacts.put_artifact({"id": "b%04d" % i, "content_type": "text/markdown", "content": "x"})
    L.db.close() if hasattr(L.db, "close") else None

    # A peer that has announced more than it has delivered: the healthy, ordinary case.
    _exec(path, "INSERT INTO seq_counter(origin, last_seq, is_local) VALUES('peer-9', 500, 0)")
    _exec(path, "INSERT INTO counter(name, n) VALUES('rows:peer-9', 400)")

    db = sdb.SqliteDb(path, origin="test-node")
    _run(nr, nr.reanchor_seq_counters_sqlite, db, {"base": str(tmp_path)})
    assert _one(path, "SELECT n FROM counter WHERE name = 'rows:peer-9'") == 400, (
        "a PEER's counter must not be re-anchored -- its shortfall is lag, and rewriting it "
        "destroys the record of what has not arrived yet")
    db.close()
