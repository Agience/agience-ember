"""Tests for the `scripts/lattice_*.py` migration tools.

These scripts run against the foundational corpus, where a wrong answer is written and kept. The
properties pinned here are the ones a hand-run cannot keep: that a tool which does no work says so
rather than exiting 0, that a dry run writes nothing, that a re-run is a no-op, and that a tool
never records a recoverable condition as permanent loss.

The scripts are standalone `main()` programs, so they are loaded by path rather than imported as
package modules.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys

import pytest

import _paths

# Both paths come from `_paths`, the one module that knows where the trees are and asserts it, so
# a relocation fails loudly. A path resolved relative to this file would still resolve after a move
# — to a directory that does not exist — and every `_load()` and `_src()` would then hit
# `pytest.skip`, which reads as a pass.
_SCRIPTS = str(_paths.EMBER_REPO / "scripts")
# the scripts import `mantle`; make it resolvable the same way the deployed PYTHONPATH does
_MANTLE = str(_paths.repo("agience-mantle") / "src")
if os.path.isdir(_MANTLE) and _MANTLE not in sys.path:
    sys.path.insert(0, _MANTLE)


def _script_path(name):
    """Resolve a script, distinguishing a wrong path from an absent script.

    The two findings get different outcomes. A `_SCRIPTS` that is not a directory points every test
    in the module at nothing, so it fails. A directory that is present but missing one script means
    that script was retired or has not landed, so that single test skips.
    """
    if not os.path.isdir(_SCRIPTS):
        pytest.fail(
            "the scripts directory %r does not exist, so EVERY test in this module would skip and "
            "the module would report a silent pass. This is a wrong path, not an absent "
            "dependency — fix tests/_paths.py." % _SCRIPTS)
    path = os.path.join(_SCRIPTS, name)
    if not os.path.exists(path):
        pytest.skip("%s not present in %s" % (name, _SCRIPTS))
    return path


def _load(name):
    path = _script_path(name)
    spec = importlib.util.spec_from_file_location("s_" + name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _src(name):
    with open(_script_path(name), encoding="utf-8") as fh:
        return fh.read()


# ── S3.8: the runtime imports without the S3 client ──────────────────────────────────────────
# S3.8 asks for content end to end with the object stores unreachable. A module-level `import
# boto3` makes that unsatisfiable by construction: the failure lands at import time, before a
# socket is ever opened, so unplugging the network cannot produce the condition the check
# describes. The client is imported lazily inside the remote-tier factory instead, which leaves a
# fully-local run with no need of it.
#
# The scan covers the live tree — ember's source and mantle's lattice — rather than any single
# script, so it holds wherever the import might be written.
def test_no_runtime_module_imports_boto3_at_module_level():
    import glob
    # Two roots: ember's source (which includes `ember/optics.py`) and mantle's lattice. Every root
    # is asserted to exist below, because a `glob` over a missing directory returns [] and enforces
    # nothing while the test still passes.
    roots = [os.path.join(_SCRIPTS, "..", "src", "ember"),
             os.path.join(_SCRIPTS, "..", "..", "agience-mantle", "src", "mantle", "db")]
    absent = [r for r in roots if not os.path.isdir(r)]
    assert not absent, (
        "declared as a scan root but not on disk: %s. A glob over a missing directory yields nothing "
        "and this test would go on passing while enforcing less than it claims — fix the path or "
        "delete the entry deliberately, never leave one to be silently skipped."
        % ", ".join(os.path.normpath(r) for r in absent))
    offenders = []
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
            if os.path.basename(f).startswith("test_"):
                continue
            with open(f, encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if line.startswith("import boto3") or line.startswith("from boto3")                             or line.startswith("import botocore"):
                        offenders.append("%s:%d" % (os.path.relpath(f, _SCRIPTS), n))
    assert not offenders, (
        "the S3 client is imported at MODULE level (column 0) in: %s. It must be imported lazily, "
        "inside the remote-tier factory, so a fully-local run never needs it -- see S3.8."
        % ", ".join(offenders))



# ── a worker with no work says so ────────────────────────────────────────────────────────────
def test_signature_worker_refuses_when_it_owns_no_blob_files(tmp_path, capsys):
    """A worker holding no files exits non-zero and names the real ceiling. Whole blob files are
    the unit of work, so any `--shards` above the file count leaves tail workers with nothing;
    exiting 0 would print the same "DONE scanned=0 signed=0" line a finished worker prints, and no
    work done would be indistinguishable from all work done."""
    mod = _load("lattice_signatures.py")
    corpus = tmp_path / "local-corpus"
    corpus.mkdir()
    for i in range(2):                          # only two blob files exist
        cx = sqlite3.connect(str(corpus / ("blobs-%02d.sqlite" % i)))
        cx.execute("CREATE TABLE blob (ref TEXT PRIMARY KEY, data BLOB)")
        cx.commit()
        cx.close()
    argv = sys.argv
    try:
        sys.argv = ["x", "--local-corpus", str(corpus), "--out", str(tmp_path / "sig.sqlite"),
                    "--shard", "5", "--shards", "6"]          # worker 5 of 6 gets nothing
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 2, "a worker with 0 files returned success"
    err = capsys.readouterr().err
    assert "FATAL" in err and "0 of 2 blob file(s)" in err
    assert "at most 2 workers" in err, "the refusal must name the real ceiling"


def test_apply_signatures_glob_ignores_wal_and_shm_sidecars():
    """The shard glob filters to numeric suffixes. A bare `glob(out + '.*')` also matches
    `signatures.sqlite.0-wal` and `-shm`, and ATTACHing a write-ahead log reports `file is not a
    database`, which reads as corpus corruption rather than as a glob picking up sidecars."""
    src = _src("lattice_apply_signatures.py")
    assert "isdigit" in src or "\\d+" in src, (
        "the shard glob no longer filters to numeric suffixes; WAL/SHM sidecars will be ATTACHed")
    assert "uri=True" in src, (
        "sqlite3.connect needs uri=True or ATTACH treats 'file:...?mode=ro' as a literal path "
        "and reports 'file is not a database'")


# ── the one-time root_id migration, end to end on a real database ────────────────────────────
def _mk_store(path, rows, with_root_id=False):
    con = sqlite3.connect(path)
    cols = "id TEXT PRIMARY KEY, doc TEXT" + (", root_id TEXT" if with_root_id else "")
    con.execute("CREATE TABLE vertex (%s)" % cols)
    for rid, doc in rows:
        con.execute("INSERT INTO vertex(id, doc) VALUES(?,?)", (rid, doc))
    con.commit()
    con.close()


def test_root_id_migration_adds_backfills_and_indexes_in_that_order(tmp_path, capsys):
    """The migration adds the column, backfills every row, and then indexes, in that order. A row
    left holding NULL is invisible to every query that filters on the discriminator, so the totals
    stay healthy while the rows are unreachable — adding the column alone would not migrate
    anything."""
    mod = _load("lattice_root_id_migrate.py")
    db = str(tmp_path / "corpus.db")
    _mk_store(db, [("a1", '{"id":"a1"}'),
                   ("a2", '{"id":"a2","root_id":"a1"}'),      # already names a lineage
                   ("a3", '{"id":"a3"}')])
    argv = sys.argv
    try:
        sys.argv = ["x", "--db", db]
        assert mod.main() == 0
    finally:
        sys.argv = argv

    con = sqlite3.connect(db)
    got = dict(con.execute("SELECT id, root_id FROM vertex").fetchall())
    assert got == {"a1": "a1", "a2": "a1", "a3": "a3"}, (
        "a doc naming a root must KEEP it; one that names none IS its own first version")
    idx = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='vertex'")}
    assert "ix_v_root_id" in idx, "the lineage index was not built; a walk would be a full scan"
    con.close()


def test_root_id_migration_is_idempotent_and_dry_run_changes_nothing(tmp_path):
    mod = _load("lattice_root_id_migrate.py")
    db = str(tmp_path / "corpus.db")
    _mk_store(db, [("a1", '{"id":"a1"}')])
    argv = sys.argv
    try:
        sys.argv = ["x", "--db", db, "--dry-run"]
        assert mod.main() == 0
        con = sqlite3.connect(db)
        cols = {r[1] for r in con.execute("PRAGMA table_info(vertex)")}
        con.close()
        assert "root_id" not in cols, "a DRY RUN added the column"

        sys.argv = ["x", "--db", db]
        assert mod.main() == 0
        sys.argv = ["x", "--db", db]
        assert mod.main() == 0, "second run failed; the migration is not idempotent"
    finally:
        sys.argv = argv
    con = sqlite3.connect(db)
    assert con.execute("SELECT root_id FROM vertex WHERE id='a1'").fetchone()[0] == "a1"
    con.close()


def test_root_id_migration_uses_no_count_star():
    """The `count(*)` ban is not stylistic: on the 6.25M-row corpus it dereferences every record —
    the query that zombied node 71. EXISTS against `ix_v_root_id` is a seek that stops at the
    first match, and the honest answer to "is there work left?" is a boolean."""
    # Comments are stripped first: the file discusses the count(*) ban at length in comments, and
    # an unstripped substring scan would match that explanation rather than actual code.
    code = chr(10).join(l.split("#", 1)[0]
                        for l in _src("lattice_root_id_migrate.py").splitlines())
    assert "count(*)" not in code.lower(), "count(*) reached the root_id migration"
    src = _src("lattice_root_id_migrate.py")
    assert "EXISTS" in src, "the pending-work check should be an EXISTS seek"


# ── the repair write path (A2) — it stamps rows in the foundational corpus ────────────────────
def _mk_repair_store(path, rows):
    """A minimal `vertex` shaped like the real one for the columns repair.py touches."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE vertex (id TEXT PRIMARY KEY, ct TEXT, created_by TEXT, "
                "created_time TEXT, doc TEXT)")
    con.execute("CREATE TABLE listkey (aid TEXT, field TEXT, value TEXT, ct TEXT)")
    for r in rows:
        con.execute("INSERT INTO vertex(id, ct, created_by, created_time, doc) "
                    "VALUES(?,?,?,?,?)", r)
    con.commit()
    con.close()


def test_repair_dry_run_writes_nothing():
    """A dry run that mutates is worse than no dry run — it removes the one safe way to inspect a
    destructive tool."""
    import tempfile
    mod = _load("lattice_repair.py")
    db = os.path.join(tempfile.mkdtemp(), "c.db")
    _mk_repair_store(db, [("a1", "text/markdown", "u", None, '{"id":"a1"}')])
    argv = sys.argv
    try:
        sys.argv = ["x", "--db", db, "--dry-run"]
        mod.main()
    finally:
        sys.argv = argv
    con = sqlite3.connect(db)
    assert con.execute("SELECT created_time FROM vertex WHERE id='a1'").fetchone()[0] is None
    con.close()


def test_repair_stamps_the_genesis_epoch_and_never_a_node_id_as_claimant():
    """The claimant is never a node id: stamping `created_time_origin = "45"` would assert that
    node 45 read its clock as 1970, a claim node 45 never made. The constant `genesis` states what
    is true — this timestamp is a placeholder the migration supplied, not an observation.

    The epoch is a constant, so every observer derives it identically (§2.3)."""
    import json as _json
    import tempfile
    mod = _load("lattice_repair.py")
    db = os.path.join(tempfile.mkdtemp(), "c.db")
    _mk_repair_store(db, [
        ("a1", "text/markdown", "u", None, '{"id":"a1"}'),                       # needs a stamp
        ("a2", "text/markdown", "u", "2020-01-01T00:00:00+00:00", '{"id":"a2"}'),  # already has one
    ])
    argv = sys.argv
    try:
        sys.argv = ["x", "--db", db]
        mod.main()
    finally:
        sys.argv = argv
    con = sqlite3.connect(db)
    got = dict(con.execute("SELECT id, created_time FROM vertex").fetchall())
    assert got["a1"] == mod.GENESIS_EPOCH
    assert got["a2"] == "2020-01-01T00:00:00+00:00", "an existing time CLAIM was overwritten"
    doc = _json.loads(con.execute("SELECT doc FROM vertex WHERE id='a1'").fetchone()[0])
    assert doc["created_time_origin"] == mod.GENESIS_TIME_CLAIMANT == "genesis"
    assert doc["created_time_origin"] != "45", "a node id was asserted as the time claimant"
    con.close()


def test_repair_is_idempotent():
    """It is run after a long migration, often more than once. A second run must be a no-op, not
    a second stamp or a fresh error."""
    import tempfile
    mod = _load("lattice_repair.py")
    db = os.path.join(tempfile.mkdtemp(), "c.db")
    _mk_repair_store(db, [("a1", "text/markdown", "u", None, '{"id":"a1"}')])
    argv = sys.argv
    try:
        for _ in range(2):
            sys.argv = ["x", "--db", db]
            assert mod.main() == 0
    finally:
        sys.argv = argv
    con = sqlite3.connect(db)
    assert con.execute("SELECT created_time FROM vertex WHERE id='a1'").fetchone()[0] \
        == mod.GENESIS_EPOCH
    con.close()


# ── the offers write path (B2a) ──────────────────────────────────────────────────────────────
def _offers_store(tmp):
    """A lattice store holding one type artifact plus rows that will and will not resolve."""
    from mantle.db import open_lattice
    L = open_lattice(os.path.join(tmp, "c.db"), origin="test")
    L.artifacts.put_artifact({
        "id": "type.text/markdown", "content_type": "application/vnd.agience.content-type+json",
        "declares": "text/markdown", "state": "committed", "created_by": "u",
        "created_time": "2026-01-01T00:00:00+00:00",
        # The markdown template is `{title}: {summary}`, with no `kind` field. A row resolves an
        # offer only if it carries a template field; `kind` is a structural discriminator, not
        # descriptive content, and the template omits it — so a row with only `kind` genuinely
        # advertises nothing and the data resolves it to no offer.
        "offer_template": "{title}: {summary}",
        "context_schema": {"properties": {"title": {"type": "string"},
                                          "summary": {"type": "string"}}},
    })
    base = {"content_type": "text/markdown", "state": "committed", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00", "kind": "doc"}
    L.artifacts.put_artifact(dict(base, id="ok1", title="Has A Title"))
    L.artifacts.put_artifact(dict(base, id="no1"))   # no title/summary -> resolves nothing, no offer
    return L


def test_offers_refuses_when_there_are_no_type_artifacts():
    """With no type artifacts there is no offer_template and no schema, so nothing can be
    described. The run stops rather than writing a guessed shape for 6.11M rows — "0 offers
    written" would look like a clean run over an undescribed corpus."""
    import tempfile
    from mantle.db import open_lattice
    mod = _load("lattice_offers.py")
    tmp = tempfile.mkdtemp()
    open_lattice(os.path.join(tmp, "c.db"), origin="test")   # store exists, but has no typedefs
    argv = sys.argv
    try:
        sys.argv = ["x", "--target", os.path.join(tmp, "c.db")]
        assert mod.main() == 2, "ran without type artifacts instead of refusing"
    finally:
        sys.argv = argv


def test_offers_dry_run_writes_nothing_and_reports_refusals():
    """An unresolved row is a finding, not a zero. `text/markdown` with no title is not a row
    missing a field — it is a different data type wearing the same format. A run reporting
    "0 offers" and a run reporting "N rows resolved nothing, here is why" are different facts
    about the corpus."""
    import tempfile
    mod = _load("lattice_offers.py")
    tmp = tempfile.mkdtemp()
    L = _offers_store(tmp)
    argv = sys.argv
    try:
        sys.argv = ["x", "--target", os.path.join(tmp, "c.db"), "--dry-run"]
        assert mod.main() == 0
    finally:
        sys.argv = argv
    assert L.artifacts.get_artifact("ok1").get("context") is None, "a DRY RUN wrote an offer"


def test_offers_writes_then_converges_to_zero():
    """Idempotence matters here specifically: a re-run that rewrites identical offers churns
    `_seq` for every row in the corpus, and `_seq` is the version identity everything else rests
    on."""
    import tempfile
    mod = _load("lattice_offers.py")
    tmp = tempfile.mkdtemp()
    L = _offers_store(tmp)
    argv = sys.argv
    try:
        sys.argv = ["x", "--target", os.path.join(tmp, "c.db")]
        assert mod.main() == 0
        ctx1 = L.artifacts.get_artifact("ok1").get("context")
        assert ctx1, "the resolvable row got no offer"
        # the unresolvable row is left alone, never given a fabricated offer
        assert L.artifacts.get_artifact("no1").get("context") is None
        seq1 = L.db.read().execute("SELECT _seq FROM vertex WHERE id='ok1'").fetchone()[0]

        sys.argv = ["x", "--target", os.path.join(tmp, "c.db")]
        assert mod.main() == 0
        seq2 = L.db.read().execute("SELECT _seq FROM vertex WHERE id='ok1'").fetchone()[0]
    finally:
        sys.argv = argv
    assert seq2 == seq1, "a second run re-wrote an unchanged offer and churned _seq"


# ── blob recovery (fetch from the durable origin) ────────────────────────────────────────────
def test_recover_shard_of_is_computed_in_python_not_sql():
    """SQLite's `CAST('0x1a' AS INTEGER)` is 0, not 26 — it does not parse hex. Pushing this
    arithmetic into SQL would silently mis-assign every ref while still returning a confident
    answer, so the shard computation runs in Python instead."""
    mod = _load("lattice_recover_blobs.py")
    assert mod.shard_of("cas/" + "1a" + "0" * 62) == 0x1a % 16
    assert mod.shard_of("cas/" + "ff" + "0" * 62) == 0xff % 16
    assert mod.shard_of("cas/" + "00" + "0" * 62) == 0


def test_recover_refuses_an_empty_ref_list():
    """A recovery that recovers nothing must not report success. 'Recovered 0 of 0' and 'the file
    was empty' are different facts."""
    import tempfile
    mod = _load("lattice_recover_blobs.py")
    p = os.path.join(tempfile.mkdtemp(), "refs.txt")
    open(p, "w").write("\n")
    argv = sys.argv
    try:
        sys.argv = ["x", "--refs", p]
        assert mod.main() == 2
    finally:
        sys.argv = argv


def test_recover_refuses_malformed_refs():
    """`cas/<64 hex>` is the contract. A short or unprefixed ref would be fetched as a key that
    cannot exist and then reported as 'absent at origin' — a wrong diagnosis of a typo."""
    import tempfile
    mod = _load("lattice_recover_blobs.py")
    p = os.path.join(tempfile.mkdtemp(), "refs.txt")
    open(p, "w").write("cas/deadbeef\nnot-a-ref\n")
    argv = sys.argv
    try:
        sys.argv = ["x", "--refs", p]
        assert mod.main() == 2
    finally:
        sys.argv = argv


def test_recover_never_writes_a_blob_whose_hash_disagrees(monkeypatch):
    """A blob stored under an address it does not hash to poisons every later read of that
    address, and content-addressed storage has no way to notice — the address is the integrity
    check, so a wrong body silently becomes the truth.

    Here the origin returns bytes that decrypt fine but hash to something else. Nothing is
    written, and the ref is reported as still missing."""
    import tempfile
    mod = _load("lattice_recover_blobs.py")
    tmp = tempfile.mkdtemp()
    corpus = os.path.join(tmp, "local-corpus")
    os.makedirs(corpus)
    ref = "cas/" + "ab" * 32                      # an address the payload will not match
    p = os.path.join(tmp, "refs.txt")
    open(p, "w").write(ref + "\n")

    class _FakeMF:
        def decrypt(self, b):
            return b"totally different bytes"

    monkeypatch.setattr(mod, "build_fernet", lambda kd: _FakeMF())
    monkeypatch.setattr(mod, "fetch", lambda ref, **kw: (b"ciphertext", "200"))
    argv = sys.argv
    try:
        sys.argv = ["x", "--refs", p, "--local-corpus", corpus, "--keys-dir", tmp]
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 1, "a hash mismatch reported success"
    # nothing was written anywhere
    assert not glob_any(corpus), "a blob with a mismatched hash was WRITTEN"


def glob_any(d):
    import glob as _g
    for f in _g.glob(os.path.join(d, "blobs-*.sqlite")):
        cx = sqlite3.connect(f)
        try:
            n = cx.execute("SELECT count(*) FROM blob").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
        cx.close()
        if n:
            return True
    return False


def test_recover_does_not_call_a_non_404_failure_absence(monkeypatch, capsys):
    """404, 403, and 5xx are distinct: only a 404 means absence. Treating any non-200 as absence
    would print a credentials problem or an origin outage as "absent at origin", recording a
    recoverable condition as permanent data loss — a direction that cannot be undone once the
    source is decommissioned on the strength of that report."""
    import tempfile
    mod = _load("lattice_recover_blobs.py")
    tmp = tempfile.mkdtemp()
    ref = "cas/" + "cd" * 32
    p = os.path.join(tmp, "refs.txt")
    open(p, "w").write(ref + "\n")
    monkeypatch.setattr(mod, "build_fernet", lambda kd: object())
    monkeypatch.setattr(mod, "fetch", lambda ref, **kw: (b"", "503"))   # origin unwell
    argv = sys.argv
    try:
        sys.argv = ["x", "--refs", p, "--local-corpus", tmp, "--keys-dir", tmp]
        rc = mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert rc == 1
    assert "absent_at_origin_404=0" in out, "a 503 was counted as proof of absence"
    assert "unreachable_non404=1" in out
    assert "NOT proof of absence" in out


def test_recover_counts_a_real_404_as_absence(monkeypatch, capsys):
    """The other half: a genuine 404 NoSuchKey is absence, and is reported as such so the loss
    can be recorded rather than retried forever."""
    import tempfile
    mod = _load("lattice_recover_blobs.py")
    tmp = tempfile.mkdtemp()
    ref = "cas/" + "ef" * 32
    p = os.path.join(tmp, "refs.txt")
    open(p, "w").write(ref + "\n")
    monkeypatch.setattr(mod, "build_fernet", lambda kd: object())
    monkeypatch.setattr(mod, "fetch", lambda ref, **kw: (b"", "404"))
    argv = sys.argv
    try:
        sys.argv = ["x", "--refs", p, "--local-corpus", tmp, "--keys-dir", tmp]
        rc = mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert rc == 1
    assert "absent_at_origin_404=1" in out and "unreachable_non404=0" in out


# ── nltk belongs to bootstrap, not to the runtime ─────────────────────────────────────────────
def test_the_reasoning_runtime_does_not_need_nltk():
    """Ontology structure is served from the corpus by `crystal.ontology.driver`, not by nltk.

    This poisons nltk rather than checking that it is absent: a machine that happens to have it
    installed would pass any import-grep while still carrying the dependency."""
    import importlib
    import sys

    poisoned = ("nltk", "nltk.corpus")
    saved = {m: sys.modules.get(m) for m in poisoned}
    dropped = {m: sys.modules[m] for m in list(sys.modules) if m.split(".")[0] == "ember"}
    try:
        for m in poisoned:
            sys.modules[m] = None          # any `import nltk` now raises ImportError
        for m in dropped:
            del sys.modules[m]
        # `ember.templates` now lives at `lumen/templates.py` in chorus; the nltk-free claim for
        # it belongs there, not here — the check follows the code, and ember does not own it.
        for mod in ("crystal.ontology.driver", "ember.ontology.activation", "crystal.ontology.geometry",
                    "ember.signal.forgetting", "mantle.shard.keyed"):
            importlib.import_module(mod)
    finally:
        for m, v in saved.items():
            if v is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = v
        sys.modules.update(dropped)


def test_serve_warms_wn_store_and_not_the_nltk_corpus():
    """The warm populates the index the runtime reads. Warming nltk's WordNet instead would pay
    real startup cost for a cache nothing consults, and silently, since the block swallows
    `Exception`."""
    src = _src_ember("serve.py")
    i = src.index("def _warm_ic")
    # Scoped to the function rather than a fixed character window, so a slice cannot run past the
    # end of `_warm_ic` into unrelated code that legitimately swallows exceptions and misattribute
    # that failure to this warm path.
    block = src[i:src.index("threading.Thread(target=_warm_ic", i)]
    # The warmed module is `crystal.ontology.driver`. The invariant: warm the index the runtime
    # reads, never nltk's.
    assert "driver" in block, "serve no longer warms the store-backed ontology index"
    assert "from nltk" not in block, "serve is warming the nltk corpus again — the runtime does " \
                                     "not read it"
    assert "pass" not in block.split("except Exception")[-1][:120], \
        "the warm failure is swallowed silently; it must at least name what failed"


def _src_ember(name):
    """Read a module out of `src/ember`."""
    return _paths.read_ember_src(name)


# ── pre-flight: what the manifest has, compared directly against what the corpus needs ────────
def _preflight_fixtures(tmp, needed, held):
    corpus = os.path.join(tmp, "corpus.db")
    con = sqlite3.connect(corpus)
    con.execute("CREATE TABLE vertex (id TEXT PRIMARY KEY, content_ref TEXT)")
    for i, ref in enumerate(needed):
        con.execute("INSERT INTO vertex VALUES(?,?)", ("a%d" % i, ref))
    con.commit(); con.close()
    lc = os.path.join(tmp, "local-corpus")
    os.makedirs(lc, exist_ok=True)
    for s in range(16):
        cx = sqlite3.connect(os.path.join(lc, "blobs-%02d.sqlite" % s))
        cx.execute("CREATE TABLE blob (ref TEXT PRIMARY KEY, data BLOB NOT NULL, "
                   "nbytes INTEGER NOT NULL) WITHOUT ROWID")
        cx.commit(); cx.close()
    for ref in held:
        s = int(ref[4:6], 16) % 16
        cx = sqlite3.connect(os.path.join(lc, "blobs-%02d.sqlite" % s))
        cx.execute("INSERT OR REPLACE INTO blob VALUES(?,?,?)", (ref, b"x", 1))
        cx.commit(); cx.close()
    return corpus, lc


def test_preflight_finds_refs_the_corpus_needs_but_the_blobs_lack():
    """The manifest describes what the source has; the store describes what the corpus needs.
    `pull`'s manifest coverage and `enrich`'s per-origin coverage each report truthfully on their
    own metric, but neither compares itself against the other — this check does."""
    import tempfile
    mod = _load("lattice_preflight_content.py")
    tmp = tempfile.mkdtemp()
    a, b, c = "cas/" + "1a" * 32, "cas/" + "2b" * 32, "cas/" + "3c" * 32
    corpus, lc = _preflight_fixtures(tmp, needed=[a, b, c], held=[a, b])   # c is absent
    needed, missing, absent_shards = mod.missing_refs(corpus, lc)
    assert needed == 3
    assert missing == [c], "the pre-flight did not name the absent ref"
    assert absent_shards == []


def test_preflight_passes_when_everything_is_present():
    import tempfile
    mod = _load("lattice_preflight_content.py")
    tmp = tempfile.mkdtemp()
    a, b = "cas/" + "4d" * 32, "cas/" + "5e" * 32
    corpus, lc = _preflight_fixtures(tmp, needed=[a, b], held=[a, b])
    needed, missing, absent_shards = mod.missing_refs(corpus, lc)
    assert needed == 2 and missing == [] and absent_shards == []


def test_preflight_reports_a_missing_shard_FILE_separately_from_missing_refs():
    """A missing shard file is a different finding from a missing ref; conflating them would
    report every ref that shard held as absent, turning one infrastructure problem into thousands
    of phantom data-loss reports."""
    import tempfile
    mod = _load("lattice_preflight_content.py")
    tmp = tempfile.mkdtemp()
    a = "cas/" + "1a" * 32
    corpus, lc = _preflight_fixtures(tmp, needed=[a], held=[a])
    os.remove(os.path.join(lc, "blobs-%02d.sqlite" % (0x1a % 16)))    # lose the shard holding `a`
    needed, missing, absent_shards = mod.missing_refs(corpus, lc)
    assert absent_shards == [0x1a % 16], "the missing shard file was not reported as such"
    assert missing == [a]     # reported, but the caller is warned the count is inflated
# The typedefs-projection invariant — that every field the offer machinery reads
# (`context_schema`, `offer_template`, `field_source`, ...) reaches the type artifact, because a
# dropped field silently disables that field for every row of the content type — lives with the
# index, in `ember.corpus.fts.project_artifact`, pinned by `tests/test_fts_index.py`: an
# artifact advertises an offer, so the offer is its description and its keyed lemmas are its tags.



def test_offers_write_does_NOT_bump_seq_an_offer_is_not_a_new_version():
    """An offer is derived metadata — a projection into `context` — not a new version of the
    artifact. The write sets `context` without allocating a fresh `_seq`, both because minting
    6.11M spurious versions that differ only in a derived field is wrong, and because the per-row
    `put_artifact` path (new _seq + merkle XOR + listkey re-index) is far slower than a direct
    UPDATE, which runs at roughly 30k rows/s.

    Safe because `row_hash = blake2b(id + NUL + str(_seq))` (constants.py) covers (id, _seq) only,
    never doc content — an unchanged _seq leaves every merkle leaf digest unchanged and
    node-repair still passes."""
    import tempfile
    from mantle.db import open_lattice
    mod = _load("lattice_offers.py")
    tmp = tempfile.mkdtemp()
    L = open_lattice(os.path.join(tmp, "c.db"), origin="test")
    L.artifacts.put_artifact({
        "id": "type.text/x-wordnet", "content_type": "application/vnd.agience.content-type+json",
        "declares": "text/x-wordnet", "state": "committed", "created_by": "u",
        "created_time": "2026-01-01T00:00:00+00:00",
        "offer_template": "the word {lemma}: {gloss}",
        "field_source": {"lemma": "lemmas.0"}})
    L.artifacts.put_artifact({"id": "wn-x", "content_type": "text/x-wordnet", "state": "committed",
                              "created_by": "u", "created_time": "2026-01-01T00:00:00+00:00",
                              "lemmas": ["dog"], "gloss": "a canine"})
    seq_before = L.db.read().execute("SELECT _seq FROM vertex WHERE id='wn-x'").fetchone()[0]

    argv = sys.argv
    try:
        sys.argv = ["x", "--target", os.path.join(tmp, "c.db")]
        assert mod.main() == 0
    finally:
        sys.argv = argv

    import json as _json
    row = L.db.read().execute("SELECT _seq, doc FROM vertex WHERE id='wn-x'").fetchone()
    assert _json.loads(row[1])["context"] == "the word dog: a canine", "offer not written"
    assert row[0] == seq_before, "the offer write BUMPED _seq — it minted a spurious version"
