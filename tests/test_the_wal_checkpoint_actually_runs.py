"""`genesis._wal_checkpoint` runs. For the whole life of this function, it did not.

THE DEFECT. `os` was never imported at module scope in `genesis.py` — the four `import os` lines
in that file are all INSIDE other functions — so `os.path.join(os.getenv(...))` raised `NameError`
on every call. The body was wrapped in `except Exception: pass`, which turned that into a silent
no-op.

This function is called THREE times per ingest (every N bulk batches, after the synset pass, and
again after the edge pass) and had never once executed. Its own docstring carries the measurement
that justifies it: *"an uncheckpointed force=True WordNet re-ingest (117k synset rewrites) grows
`lattice.db-wal` to 18 GB, and every read then scans the whole WAL, so chat takes 20-60s."* That is
what the store has been getting all along, because the fix never ran.

Fail-soft was the right call and is kept — a checkpoint that cannot run must not fail an ingest.
Fail-soft and fail-SILENT are different things, and only one of them is debuggable. The swallow now
logs.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from ember import genesis


def test_the_names_the_function_uses_are_actually_in_its_namespace():
    """The bug, as a one-line assertion. `os` was missing from module scope, and the only symptom
    was a checkpoint that quietly did nothing."""
    g = genesis._wal_checkpoint.__globals__
    assert "os" in g, (
        "`os` is not in this module's namespace — `_wal_checkpoint` will raise NameError and the "
        "swallow will hide it, exactly as it did before 2026-08-25")
    assert "_log" in g, (
        "no module logger — the exception handler has nowhere to report, which is how a NameError "
        "survived here unnoticed")


def _wal_bytes(db_path: str) -> int:
    wal = db_path + "-wal"
    return os.path.getsize(wal) if os.path.exists(wal) else 0


@pytest.fixture
def wal_db(tmp_path, monkeypatch):
    """A real WAL with committed frames, and a SECOND connection held open.

    The holder is not incidental: SQLite checkpoints and deletes the WAL when the LAST connection
    closes, so without it this test would watch SQLite tidy up and pass regardless of what
    `_wal_checkpoint` did."""
    path = str(tmp_path / "lattice.db")
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(x)")
    conn.executemany("INSERT INTO t VALUES(?)", [("y" * 200,) for _ in range(2000)])
    conn.commit()
    holder = sqlite3.connect(path)
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(tmp_path))
    monkeypatch.setenv("EMBER_SQLITE_DB", "lattice.db")
    assert _wal_bytes(path) > 0, "no WAL was produced — the assertions below would be vacuous"
    yield path
    holder.close()
    conn.close()


def test_it_reclaims_the_log(wal_db):
    before = _wal_bytes(wal_db)
    genesis._wal_checkpoint()
    after = _wal_bytes(wal_db)
    assert after < before, (
        "the WAL did not shrink (%d -> %d). Before 2026-08-25 this function raised NameError and "
        "the swallow hid it, so the numbers were identical and nothing said why" % (before, after))
    assert after == 0, "TRUNCATE should reset the log to zero, got %d" % after


def test_a_failure_is_logged_rather_than_swallowed(wal_db, caplog, monkeypatch):
    """The swallow is kept — an ingest must not fail over a checkpoint — but it must speak.
    A silent best-effort is indistinguishable from a best-effort that never ran, which is the
    entire reason this defect lasted."""
    monkeypatch.setattr(genesis.os.path, "join", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level("WARNING"):
        genesis._wal_checkpoint()          # must not raise
    assert any("WAL checkpoint did not run" in r.message for r in caplog.records), (
        "the failure was swallowed without a word: %r" % [r.message for r in caplog.records])
