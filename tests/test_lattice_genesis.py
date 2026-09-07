"""genesis.py's four store call sites, on typed methods rather than raw SQL.

Each test pins an answer, not the presence of a feature: a counter that reports the global corpus
count in place of a per-collection one, or a checkpoint read that answers `[]`, is wrong rather than
missing, and an empty result is otherwise indistinguishable from a broken query.

The suite runs against a real lattice store (`mantle.db`) on a temp file, because what these
tests measure is the fit between what the call site asks for and what the backend answers — a mock
of the backend would encode an assumption about that boundary instead of testing it. Where the
lattice package is not importable the whole module skips rather than asserting less.
"""
from __future__ import annotations

import pytest

from ember import genesis as g

lattice = pytest.importorskip("mantle.db",
                              reason="lattice store (unit L) not importable on this path")

SHARD_DONE_CT = g.SHARD_DONE_CT
ORIGIN = "test-observer"


class _Store:
    """The store faces genesis.py touches, shaped like `mantle.shard.local_store.LocalStore`."""

    def __init__(self, L):
        self.artifacts = L.artifacts
        self.graph = L.graph
        self.content = None
        self.keys_dir = None


@pytest.fixture()
def store(tmp_path):
    L = lattice.open_lattice(str(tmp_path / "lattice.db"), origin=ORIGIN)
    L.artifacts.ensure_schema()
    L.graph.ensure_schema()
    return _Store(L)


def _seed(store, *, grammar=40, world_committed=120, world_draft=30, loose=25):
    """Deliberately asymmetric: two collections of different sizes plus artifacts in no collection,
    so a per-collection count and the global count cannot coincide by accident. A symmetric fixture
    would let a global-count answer pass as a per-collection one."""
    docs = []
    for i in range(grammar):
        docs.append({"id": "g%04d" % i, "content_type": "text/markdown",
                     "collection_id": "stage.1.grammar", "state": "committed",
                     "content": "grammar", "size": 100, "lemmas": ["g%d" % i],
                     "cited_from": g.CITE_GENESIS, "provenance": g.P_HUMAN})
    for i in range(world_committed + world_draft):
        docs.append({"id": "w%04d" % i, "content_type": "text/markdown",
                     "collection_id": "stage.2.world",
                     "state": "committed" if i < world_committed else "draft",
                     "content": "world", "size": 200, "lemmas": ["w%d" % i],
                     "cited_from": g.CITE_GENESIS, "provenance": g.P_HUMAN})
    for i in range(loose):
        docs.append({"id": "op.free.%02d" % i, "content_type": g.OPERATOR_CONTENT_TYPE,
                     "state": "committed", "content": "op",
                     "cited_from": g.CITE_GENESIS, "provenance": g.P_HUMAN})
    store.artifacts.put_many(docs, batch=200)
    return len(docs)


# ── the collection counter ───────────────────────────────────────────────────
def test_collection_count_is_per_collection_not_the_global_corpus(store):
    """`_count_collection` answers for the named collection. A backend that does not recognise the
    caller's parameter name falls through to a bare total and returns the global corpus count."""
    total = _seed(store)
    assert store.artifacts.count() == total

    grammar = g._count_collection(store, "stage.1.grammar", committed_only=False)
    world = g._count_collection(store, "stage.2.world", committed_only=False)

    assert grammar == 40
    assert world == 150
    # The two collections must not agree with each other OR with the corpus. Asserting only
    # `!= total` would still pass if both returned some other single constant.
    assert grammar != world != total != grammar


def test_a_collection_with_no_members_counts_zero_not_the_corpus(store):
    """An empty collection counts zero. The error a global count introduces is `global - real`, so it
    is largest for the emptiest stage — a brand-new curriculum stage would read as already
    complete."""
    _seed(store)
    assert g._count_collection(store, "stage.9.never.ingested", committed_only=False) == 0


def test_committed_only_selects_a_different_counter(store):
    """`committed_only` selects a different counter, so the two arms give different numbers.
    `advance_curriculum` depends on `False` counting every state: a committed-only answer there
    under-counts, hands the ingester a low resume offset, and re-ingests held records."""
    _seed(store, world_committed=120, world_draft=30)
    assert g._count_collection(store, "stage.2.world", committed_only=False) == 150
    assert g._count_collection(store, "stage.2.world", committed_only=True) == 120


def test_advance_curriculum_resume_offset_is_the_stage_count(monkeypatch, store):
    """The same property end to end. `have` is the resume offset and both sides of the exhaustion
    test `drained = (now <= have)`; fed the global count it would skip un-ingested records and
    promote on corpus size rather than stage size."""
    _seed(store, grammar=0, world_committed=0, world_draft=0, loose=200)
    seen = {}

    def fake_ingest(s, skip, limit):
        seen["skip"] = skip
        s.artifacts.put_many([
            {"id": "n%04d" % i, "content_type": "text/markdown", "state": "committed",
             "collection_id": "stage.test", "content": "x", "size": 1,
             "cited_from": g.CITE_GENESIS, "provenance": g.P_HUMAN} for i in range(5)])
        return {"ingested": 5}

    monkeypatch.setattr(g, "CURRICULUM", [
        {"stage": "stage.test", "ingest": fake_ingest, "target": 25, "per_tick": 5}])
    g.bootstrap(store)

    r = g.advance_curriculum(store)
    # 200 loose artifacts exist; the stage holds none. The offset is the stage's count.
    assert seen["skip"] == 0
    assert r["had"] == 0 and r["now"] == 5
    assert r["promoted"] is False           # 5 < 25, despite a corpus far past the target


# ── the shard checkpoints ────────────────────────────────────────────────────
def test_shards_done_is_not_silently_empty_and_is_scoped_by_source(store):
    for i in range(7):
        store.artifacts.put_artifact({
            "id": g._shard_done_id("wikipedia-en", "s%d" % i), "content_type": SHARD_DONE_CT,
            "state": "committed", "source_name": "wikipedia-en", "shard": "s%d" % i,
            "cited_from": g.CITE_GENESIS, "provenance": g.P_HUMAN})
    for i in range(3):
        store.artifacts.put_artifact({
            "id": g._shard_done_id("simplewiki", "t%d" % i), "content_type": SHARD_DONE_CT,
            "state": "committed", "source_name": "simplewiki", "shard": "t%d" % i,
            "cited_from": g.CITE_GENESIS, "provenance": g.P_HUMAN})

    assert g._shards_done(store, "wikipedia-en") == ["s%d" % i for i in range(7)]
    assert len(g._shards_done(store, "simplewiki")) == 3
    # a source with no checkpoints is genuinely empty, and reads as empty
    assert g._shards_done(store, "nosuchsource") == []


def test_mark_shard_done_round_trips(store):
    """The checkpoint write and the checkpoint read agree. Where they do not — the write lands and
    the read answers `[]` — the fleet re-ingests finished shards indefinitely."""
    g._mark_shard_done(store, "wikipedia-en", "data/train-00007.parquet")
    assert g._shards_done(store, "wikipedia-en") == ["data/train-00007.parquet"]


# ── the rho ledger ───────────────────────────────────────────────────────────
def test_consolidated_members_reads_the_edges(store):
    _seed(store, grammar=10, world_committed=0, world_draft=0, loose=0)
    store.graph.add_edges([("g0000", "g%04d" % i, "consolidates", {"force": "derivation"})
                           for i in range(1, 4)])
    store.graph.add_edges([("g0000", "g0009", "lineage", {"force": "derivation"})])

    members = g._consolidated_members(store)
    assert members == {"g0001", "g0002", "g0003"}
    assert "g0009" not in members            # a different label does not leak in


def test_rho_falls_once_members_are_consolidated(store):
    """ρ is the entropy gauge the curriculum is steered by. With `_consolidated_members` answering
    `set()`, every artifact counts as its own generator and ρ reads exactly 1.0 — the "nothing
    consolidated yet" baseline — on a corpus that has been consolidated."""
    _seed(store, grammar=10, world_committed=0, world_draft=0, loose=0)
    before = g.collection_metrics(store, "stage.1.grammar")
    assert before["rho"] == 1.0

    store.graph.add_edges([("g0000", "g%04d" % i, "consolidates", {"force": "derivation"})
                           for i in range(1, 6)])
    after = g.collection_metrics(store, "stage.1.grammar")
    assert after["consolidated"] == 5
    assert after["rho"] == pytest.approx(0.5, abs=1e-6)   # 5 generators of 10 equal-size
    assert after["rho"] < before["rho"]


# ── the null-field audit ─────────────────────────────────────────────────────
def test_count_null_rejects_a_field_outside_the_audited_set(store):
    """The audited field set is the single place the list of scannable fields lives, and a name
    outside it raises. An interpolated field name would otherwise reach the query text directly."""
    with pytest.raises(ValueError):
        g._count_null(store, "state", allow_scan=True)
    with pytest.raises(ValueError):
        g._count_null(store, "cited_from; DROP TABLE vertex", allow_scan=True)


def test_count_null_is_none_without_allow_scan(store):
    """status() is on the 30s path and does not pay for this scan. `None` means not measured; a 0
    would read as a clean audit that never ran."""
    _seed(store)
    assert g._count_null(store, "cited_from") is None
    assert g._count_null(store, "provenance") is None


def test_count_null_counts_the_holes_under_allow_scan(store):
    _seed(store, grammar=10, world_committed=0, world_draft=0, loose=0)
    for i in range(4):
        store.artifacts.put_artifact({"id": "hole%d" % i, "content_type": "text/markdown",
                                      "collection_id": "stage.1.grammar", "state": "committed",
                                      "content": "x", "provenance": g.P_HUMAN})
    assert g._count_null(store, "cited_from", allow_scan=True) == 4
    assert g._count_null(store, "provenance", allow_scan=True) == 0


def test_count_null_is_a_superset_when_state_is_gone(store):
    """`_count_null` counts every state, because contract §2 has no `state` column for the predicate
    to scope on.

    The count is therefore a superset of the committed-only one, and the direction is what makes it
    usable: the only consumer is `invariant_holds = (miss == 0)`, so a superset can raise a false
    alarm but cannot report a clean audit over a dirty corpus."""
    _seed(store, grammar=0, world_committed=0, world_draft=0, loose=0)
    store.artifacts.put_artifact({"id": "committed-hole", "content_type": "text/markdown",
                                  "state": "committed", "content": "x", "provenance": g.P_HUMAN})
    store.artifacts.put_artifact({"id": "draft-hole", "content_type": "text/markdown",
                                  "state": "draft", "content": "x", "provenance": g.P_HUMAN})
    # committed-only truth is 1; the state-free count is 2. Superset, never subset.
    assert g._count_null(store, "cited_from", allow_scan=True) == 2


def test_count_null_does_not_cache_a_failed_scan(store, monkeypatch):
    """A scan that could not run leaves nothing cached, so a later working scan is not shadowed by
    it. Caching a 0 from a failed query would store a clean audit nobody took."""
    monkeypatch.setattr(g, "_scan_missing_field", lambda s, f: None)
    assert g._count_null(store, "cited_from", allow_scan=True) is None
    # nothing was cached, so a later working scan is not shadowed by the failure
    monkeypatch.setattr(g, "_scan_missing_field", lambda s, f: 7)
    assert g._count_null(store, "cited_from", allow_scan=True) == 7


# ── the seam itself ──────────────────────────────────────────────────────────
class _Exploding:
    """A connection that raises on any use, standing in for a non-lattice connection.

    The assertions below are on the returned value rather than on the raise. Every legacy branch
    wraps its query in `except Exception`, so `_Exploding` is absorbed the same way an `[]` answer
    would be, and a `pytest.raises` here would pass while measuring nothing. Routing to a non-lattice
    connection therefore shows up as the empty or fallback answer, which is what these assertions
    pin. The store is always the SQLite lattice, so this is the guard that raw SQL never reaches
    anything else."""

    def query(self, *a, **kw):
        raise AssertionError("raw SQL reached a non-Arcade connection")

    def command(self, *a, **kw):
        raise AssertionError("raw SQL reached a non-Arcade connection")


def test_no_converted_site_hands_sql_to_a_non_arcade_connection(store, monkeypatch):
    _seed(store, grammar=5, world_committed=5, world_draft=0, loose=5)
    store.graph.add_edges([("g0000", "g0001", "consolidates", {})])
    monkeypatch.setattr(type(store.artifacts), "c", _Exploding(), raising=False)
    monkeypatch.setattr(type(store.graph), "c", _Exploding(), raising=False)

    assert g._shards_done(store, "wikipedia-en") == []
    assert g._count_collection(store, "stage.1.grammar", committed_only=False) == 5
    assert g._count_null(store, "cited_from", allow_scan=True) == 0
    assert g._consolidated_members(store) == {"g0001"}



def test_typed_returns_none_for_a_backend_without_the_method():
    class Bare:
        pass

    assert g._typed(Bare(), "count_in_collection") is None
    assert g._typed(None, "count_in_collection") is None


# ── the guard on consolidate_nearvdup ────────────────────────────────────────


# ── operator registration writes only on change ──────────────────────────────
# `register_control_operators` runs on every `invoke()`, so on every task the pool executes.
# Rewriting all 14 operator artifacts unconditionally would cost 14 `_seq` allocations, 14 vacated
# seqs, 14 churned merkle leaves and 14 mesh republishes per task, including for tasks that failed.
# `put_artifact` is idempotent by id, so the rows would stay correct throughout: the cost is in the
# events, not the rows, which is why these tests measure allocations.

def _last_seq(store):
    r = store.artifacts.db.read().execute(
        "SELECT last_seq FROM seq_counter WHERE origin = ?", (ORIGIN,)).fetchone()
    return int(r["last_seq"]) if r else 0










def test_same_as_stored_ignores_only_what_the_store_itself_added(store):
    """`_same_as_stored` ignores exactly the fields the store itself added. Ignoring more would stop
    it repairing drift; ignoring less would stop it skipping anything. Both edges are pinned."""
    doc = {"id": "x", "content_type": "text/markdown", "context": "a"}
    stored = dict(doc, _origin=ORIGIN, _seq=7, created_time_origin=ORIGIN)
    assert g._same_as_stored(stored, doc) is True
    assert g._same_as_stored(dict(stored, context="b"), doc) is False
    assert g._same_as_stored(dict(stored, extra="dropped-by-the-write"), doc) is False
    assert g._same_as_stored(None, doc) is False


