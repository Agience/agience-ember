"""Unit E — `mantle.mesh.sync` and `mantle.mesh.federation` against a real lattice store.

Every test here states an invariant of the mesh surface: what may enter a published leaf, what an
apply counts as handled, what a consumed edge is stamped with, and what two converged nodes must
agree on. Invariants rather than behavioural comparisons, because the properties are what the mesh
is for — a segment feed and an anti-entropy round have to reach the same state by different routes.

Everything runs against a real `open_lattice()` file and a fake S3, so no test touches a live
database.
"""
# The imports below name `mantle.mesh` explicitly. This file lives in `tests/`, so a relative
# `from . import merkle` would resolve to `tests.merkle`.
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import _paths

# `_paths.repo` computes the workspace depth once and asserts it, rather than each test file
# counting `parents[n]` for itself. A hand-counted depth that is wrong lands on a directory that
# does not exist, and the `is_dir()` guard below turns that into a silent no-op — the module then
# imports from wherever `mantle` happens to be installed, or skips.
_MANTLE = _paths.repo("agience-mantle") / "src"
if _MANTLE.is_dir() and str(_MANTLE) not in sys.path:
    sys.path.insert(0, str(_MANTLE))

lattice = pytest.importorskip("mantle.db", reason="lattice store not on the path")

from mantle.mesh import federation, sync  # noqa: E402


# ── fakes ────────────────────────────────────────────────────────────────────────────────────────
class _FakePaginator:
    """Emulates `list_objects_v2` including `Delimiter` and `StartAfter`.

    Both are load-bearing here. `StartAfter` is lexicographic, so a `tail-*` key sorts after every
    numeric backfill key; and `Delimiter` is what makes `_s3_prefixes` see nothing under the flat
    `mesh/merkle/<node>.json` objects, which leaves reconcile enumerating zero peers while it
    returns `{"applied": 0}` and reads as healthy. Emulating them is what lets these tests observe
    either case."""

    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket=None, Prefix="", Delimiter=None, StartAfter=None, **kw):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        if StartAfter:
            keys = [k for k in keys if k > StartAfter]
        if Delimiter:
            commons, contents = [], []
            for k in keys:
                rest = k[len(Prefix):]
                if Delimiter in rest:
                    cp = Prefix + rest.split(Delimiter, 1)[0] + Delimiter
                    if cp not in commons:
                        commons.append(cp)
                else:
                    contents.append({"Key": k})
            yield {"CommonPrefixes": [{"Prefix": p} for p in commons], "Contents": contents}
            return
        yield {"Contents": [{"Key": k} for k in keys]}


class _FakeClient:
    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self.objects)


class FakeS3:
    """Enough of the GarageContentStore surface for the publish/consume paths."""

    def __init__(self):
        self.objects = {}
        self.fail_on = set()
        self.bucket = "test-bucket"
        self._s3 = _FakeClient(self.objects)

    def put(self, key, data, content_type="application/octet-stream"):
        self.objects[key] = data

    def get(self, key):
        if key in self.fail_on:
            raise RuntimeError("simulated S3 read failure")
        return self.objects[key]

    def exists(self, key):
        return key in self.objects


class FakeFernet:
    """Identity 'encryption'. The crypto is not what these tests are about."""

    @staticmethod
    def encrypt(b):
        return b

    @staticmethod
    def decrypt(b):
        return b


class Store:
    """The `store` duck-type sync.py expects: `.artifacts`, `.graph`, `.content`, `.keys_dir`."""

    def __init__(self, path, origin):
        L = lattice.open_lattice(str(path), origin=origin)
        self.artifacts = L.artifacts
        self.graph = L.graph
        self.content = None
        self.keys_dir = None


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "lat.db", origin="node-A")


@pytest.fixture()
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(sync, "_mesh_s3", lambda store: fake)
    monkeypatch.setattr(sync, "_fernet", lambda store: FakeFernet())
    monkeypatch.setattr(sync, "_node_id", lambda: "node-A")
    return fake


def _seg_docs(s3, prefix):
    """Every doc across every segment under `prefix`, in key order."""
    out = []
    for k in sorted(s3.objects):
        if k.startswith(prefix) and k.endswith(".ndjson.enc"):
            for line in s3.objects[k].decode("utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
    return out


# ── the store seam ───────────────────────────────────────────────────────────────────────────────
def test_seam_detects_the_lattice_store(store):
    assert sync._vertices(store) is not None
    assert sync._edges(store) is not None


def test_unservable_is_not_empty(tmp_path):
    """An unrecognised query raises rather than answering `[]`.

    An empty list is a measurement — there are no such rows. A query that cannot be served here is
    the absence of a measurement, and the two must stay distinguishable: a global count and a
    per-collection count read from the same accessor family, so collapsing "unservable" into "0"
    would make one indistinguishable from the other."""
    class Bare:
        pass

    class BareStore:
        artifacts = Bare()
        graph = Bare()

    # The accessors resting on `_require_lattice` are the typed counts. Handed a store they cannot
    # read, each raises — a 0 here would be a reading nobody took.
    for fn, args in ((sync._total_count, ()), (sync._count_of_type, ("text/markdown",))):
        with pytest.raises(RuntimeError, match="Refusing to return an empty result"):
            fn(BareStore(), *args)


# ── Invariant 4: `_OP_EXCLUDE` and its prefix bans keep operational rows out of the mesh ──────────
def test_operational_and_probe_rows_never_enter_a_leaf(store, s3):
    """A published leaf carries replicated rows only — no probes, no operational cursors.

    A probe fixture carrying a `_rev` far in the future replicates to every node and pins their
    publish cursors beyond every real row, which mutes the feed. On the Merkle path the exclusion
    holds on the leaf objects themselves: a leaf containing per-node state gives two converged nodes
    different roots."""
    store.artifacts.put_artifact({"id": "real", "content_type": "text/markdown"})
    store.artifacts.put_artifact({"id": "p1", "content_type": "application/x-probe-thing"})
    store.artifacts.put_artifact({"id": "p2", "content_type": "application/vnd.agience.probe+json"})
    store.artifacts.put_artifact({"id": "p3", "content_type": "application/x-throwaway-fixture"})
    store.artifacts.put_artifact({"id": "t1", "content_type": "application/vnd.agience.task+json"})

    sync.publish_merkle_incremental(store)
    ids = {d["id"] for d in _seg_docs(s3, sync._MESH_LEAF_PREFIX) if "id" in d}
    assert ids == {"real"}, "an excluded type reached a leaf object: %r" % (ids,)


def test_merkle_tree_excludes_operational_rows(store, s3):
    """The lattice store stamps `_leaf` on every row, including per-box cursors, so the incremental
    tree carries operational state. `publish_merkle_incremental` (via `refresh_leaves`) XORs those
    rows back out. Left in, two converged nodes hash their own cursors into the same leaves, their
    roots stay unequal, and reconcile chases a difference it is itself creating."""
    store.artifacts.put_artifact({"id": "keep", "content_type": "text/markdown"})
    sync.publish_merkle_incremental(store)
    root_clean = (store.artifacts.get_artifact(sync._S3_MERKLE_CURSOR) or {}).get("root")

    sync._put_op(store, {"id": "s3.some.cursor", "content_type": sync._S3SYNC_CT, "last_id": "x"})
    store.artifacts.put_artifact({"id": "probe", "content_type": "application/x-probe-y"})
    sync.publish_merkle_incremental(store)
    # the probe is prefix-banned rather than subtracted incrementally, so it may move the tree; the
    # operational cursor may not. Assert no leaf object carries an operational or probe id.
    ids = {d["id"] for d in _seg_docs(s3, sync._MESH_LEAF_PREFIX) if "id" in d}
    assert "keep" in ids and "s3.some.cursor" not in ids and "probe" not in ids


def test_replicated_count_subtracts_operational(store):
    store.artifacts.put_artifact({"id": "a", "content_type": "text/markdown"})
    store.artifacts.put_artifact({"id": "b", "content_type": "text/markdown"})
    sync._put_op(store, {"id": "cur", "content_type": sync._S3SYNC_CT})
    assert sync._replicated_count(store) == 2


# ── operational rows consume no proper time (the live-lock this pins) ────────────────────────────
def test_cursor_writes_do_not_consume_proper_time(store):
    """A cursor write leaves the publish feed exactly as it found it.

    Under `(_origin,_seq)`, stamping a cursor allocates a `_seq`, which guarantees the next publish
    page is non-empty — and every cycle writes a cursor, so the feed live-locks on its own
    bookkeeping. `_put_op` writes them outside the sequence instead.

    Measured on `pending_publish`, which counts rows in the publish feed. Counting allocations
    would show only that the write consumed no proper time; a row that allocated nothing and still
    landed in the feed live-locks it just the same."""
    before = store.artifacts.pending_publish(vertex_cursor=0, edge_cursor=0)
    for _ in range(5):
        sync._put_op(store, {"id": "s3.pub.cursor", "content_type": sync._S3SYNC_CT, "seq": 1})
    assert store.artifacts.pending_publish(vertex_cursor=0, edge_cursor=0) == before


def test_operational_version_identity_is_unique_and_gap_free(store):
    """Node-repair enforces both properties, so an operational version identity has to satisfy them
    too: unique per row, and trivially contiguous. A shared `("_local", 0)` breaks
    `(_origin,_seq) unique`; a hashed seq breaks `_seq contiguity (peers)` and raises a WARN that
    stands forever on a healthy node. A per-row `_local:<id>` origin at seq 1 satisfies both."""
    ids = ["s3.pub.cursor", "s3.sub.cursor.node-B", "s3.merkle.cursor"]
    for i in ids:
        sync._put_op(store, {"id": i, "content_type": sync._S3SYNC_CT, "v": 1})
    vers = [store.artifacts.version_of(i) for i in ids]
    assert len(set(vers)) == len(ids), "duplicate operational version identity"
    assert all(sq == 1 for _o, sq in vers), "operational seqs must be trivially gap-free"
    assert all(o.startswith(sync._LOCAL_ORIGIN + ":") for o, _s in vers)


def test_operational_rewrite_is_version_stable(store):
    """A cursor rewritten every cycle keeps the same version identity, so its merkle leaf stays put
    across publish rounds."""
    sync._put_op(store, {"id": "s3.pub.cursor", "content_type": sync._S3SYNC_CT, "seq": 1})
    first = store.artifacts.version_of("s3.pub.cursor")
    for i in range(5):
        sync._put_op(store, {"id": "s3.pub.cursor", "content_type": sync._S3SYNC_CT, "seq": i})
    assert store.artifacts.version_of("s3.pub.cursor") == first


def test_put_op_refuses_a_replicated_type(store):
    """`_put_op` accepts operational content types only. A replicated type written as operational
    sits outside every observer's sequence: unpublishable, and invisible to every peer."""
    with pytest.raises(ValueError, match="REPLICATES"):
        sync._put_op(store, {"id": "x", "content_type": "text/markdown"})


# ── Invariant 2: put_many returns the handled count; an errored row is not handled ───────────────
def test_partial_apply_raises_so_the_cursor_is_held(store, monkeypatch):
    """The put_many shortfall is `_apply_artifacts`' only evidence of data loss, so a shortfall
    raises and the cursor holds. A segment recorded as applied advances `last_key` behind a
    monotone StartAfter marker, which puts anything unwritten permanently out of reach."""
    monkeypatch.setattr(store.artifacts, "put_many", lambda *a, **k: 1)
    items = [{"id": "x", "content_type": "text/markdown", "_origin": "node-B", "_seq": 1},
             {"id": "y", "content_type": "text/markdown", "_origin": "node-B", "_seq": 2}]
    with pytest.raises(RuntimeError, match="partial apply"):
        sync._apply_artifacts(store, items)


def test_lww_rejected_counts_as_handled(store):
    """Declining to overwrite a newer local row counts as handled: the row was considered and a
    decision was reached. Counting it as a shortfall would fire the data-loss guard on correct
    behaviour and wedge the cursor."""
    store.artifacts.put_artifact({"id": "v", "content_type": "text/markdown",
                                  "_origin": "node-B", "_seq": 9}, stamp_rev=False)
    older = [{"id": "v", "content_type": "text/markdown", "_origin": "node-B", "_seq": 3}]
    assert sync._apply_artifacts(store, older) == 1


def test_consume_preserves_peer_authorship(store):
    """`stamp_rev=False` on the apply path preserves `(_origin,_seq)`, so a peer's event keeps its
    author. That pair is the system's only causal record; re-stamping it would claim this node
    authored the event. This is the apply the Merkle leaf transfer runs on peer rows."""
    d = {"id": "z", "content_type": "text/markdown", "_origin": "node-B", "_seq": 42}
    sync._apply_artifacts(store, [d])
    assert store.artifacts.version_of("z") == ("node-B", 42)


# ── Invariant 3: _MERKLE_LIVE / _MERKLE_PUBLISHED stay distinct ──────────────────────────────────
def test_merkle_live_and_published_are_distinct_keys(store, s3):
    """The live tree and the published tree are tracked under separate keys, because the publish
    decision is "what have I uploaded", not "what have I computed". Computing `changed` against the
    live tree reads a leaf that `refresh_leaves` already recomputed as unchanged, so no leaf file is
    uploaded while the root advertises it — peers then fetch a key that 404s and stay divergent."""
    assert sync._MERKLE_LIVE != sync._MERKLE_PUBLISHED
    for i in range(5):
        store.artifacts.put_artifact({"id": "m%d" % i, "content_type": "text/markdown"})
    sync.publish_merkle_incremental(store)
    cur = store.artifacts.get_artifact(sync._S3_MERKLE_CURSOR) or {}
    assert sync._MERKLE_LIVE in cur and sync._MERKLE_PUBLISHED in cur

    # refresh_leaves recomputes LIVE and uploads nothing, so PUBLISHED stays where it was.
    before = list(cur[sync._MERKLE_PUBLISHED])
    sync.refresh_leaves(store, [0, 1, 2])
    after = store.artifacts.get_artifact(sync._S3_MERKLE_CURSOR) or {}
    assert list(after[sync._MERKLE_PUBLISHED]) == before


def test_refresh_leaves_subtracts_operational_rows(store, s3):
    """The lattice store XORs every row into its incremental tree, including per-box cursors, so
    `refresh_leaves` subtracts them back out. The unmodified tree gives two converged nodes
    different roots, because their cursors differ."""
    for i in range(5):
        store.artifacts.put_artifact({"id": "r%d" % i, "content_type": "text/markdown"})
    sync.publish_merkle_incremental(store)
    clean = list((store.artifacts.get_artifact(sync._S3_MERKLE_CURSOR) or {})[sync._MERKLE_LIVE])

    # A cursor write mutates the store's incremental tree; the mesh tree stays where it was.
    sync._put_op(store, {"id": "s3.some.cursor", "content_type": sync._S3SYNC_CT, "v": 1})
    out = sync.refresh_leaves(store, range(len(clean)))
    assert out.get("excluded_op_rows", 0) >= 1
    refreshed = list((store.artifacts.get_artifact(sync._S3_MERKLE_CURSOR) or {})[sync._MERKLE_LIVE])
    assert refreshed == clean, "an operational row leaked into the replication tree"


def test_merkle_row_hash_uses_seq(store):
    """Contract RESOLVED-1: a row's merkle hash is keyed on `_seq`, so two versions of one id hash
    differently. Roots therefore compare equal only between nodes that key rows the same way."""
    from mantle.mesh import merkle
    store.artifacts.put_artifact({"id": "h", "content_type": "text/markdown"})
    sq = store.artifacts.version_of("h")[1]
    assert isinstance(sq, int)
    assert merkle.row_hash("h", sq) != merkle.row_hash("h", sq + 1)


def test_edge_apply_is_idempotent(store):
    store.artifacts.put_artifact({"id": "i1", "content_type": "text/markdown"})
    store.artifacts.put_artifact({"id": "i2", "content_type": "text/markdown"})
    items = [{"f": "i1", "t": "i2", "label": "lineage", "props": {}}]
    for _ in range(5):
        sync._apply_edges(store, items)
    assert store.graph.count_edges() == 1


# ── Unit R / contract §5.8.2 — consumed edges carry a reserved origin ─────────────────────────────
def _last_seq(store, origin):
    r = store.artifacts.db.read().execute(
        "SELECT last_seq FROM seq_counter WHERE origin = ?", (origin,)).fetchone()
    return int(r["last_seq"]) if r else 0


def _feed(store):
    """The local publish feed, as `publish_edges_to_s3` scans it: `(src, dst, label)` triples."""
    me = sync._origin_of(store)
    return {(r["src"], r["dst"], r["label"])
            for r in store.graph.page_by_origin(origin=me, after_seq=0, limit=100000)}


def _endpoints(store, *ids):
    for i in ids:
        store.artifacts.put_artifact({"id": i, "content_type": "text/markdown"})


def test_a_consumed_edge_is_not_stamped_as_locally_authored(store):
    """A consumed edge is stamped with an origin other than this node's.

    Contract §5.8.2 makes no accuracy claim about edge `_origin` — the wire format does not carry
    provenance, so there is nothing accurate to stamp. The property it does carry is "not mine",
    which is what keeps the edge out of the publish feed."""
    _endpoints(store, "r1", "r2")
    sync._apply_edges(store, [{"f": "r1", "t": "r2", "label": "lineage", "props": {}}])

    row = store.graph.db.read().execute(
        "SELECT _origin, _seq FROM edge WHERE src = 'r1'").fetchone()
    assert row["_origin"] != sync._origin_of(store)
    assert row["_origin"].startswith(sync._CONSUMED_EDGE_NS)
    assert row["_seq"] == 1          # its own degenerate origin -> sequence trivially 1..1


def test_consumed_edges_never_enter_the_local_publish_feed(store):
    """The no-echo property, stated against the scan rather than against a flag.

    `page_by_origin(origin=me)` is `publish_edges_to_s3`'s query, so asserting on it measures the
    exclusion where it happens. The reserved origin makes the exclusion structural — an index range
    that does not reach the row — rather than a filter each caller applies."""
    _endpoints(store, "k1", "k2", "k3", "k4")
    store.graph.add_edges([("k1", "k2", "lineage", {})])              # locally authored
    sync._apply_edges(store, [{"f": "k3", "t": "k4", "label": "lineage", "props": {}}])  # consumed

    feed = _feed(store)
    assert ("k1", "k2", "lineage") in feed, "a locally authored edge must still publish"
    assert ("k3", "k4", "lineage") not in feed, "a consumed edge echoed back into the publish feed"
    assert store.graph.count_edges() == 2, "both edges are STORED; only one is PUBLISHED"


def test_a_second_consume_round_is_a_no_op_not_a_republish(store):
    """Replay consumes no proper time at all.

    The reserved origin is deterministic in the edge triple, so a replayed segment presents the
    identical `(_origin, _seq)`; `add_edges(stamp_rev=False)` compares versions, reads them as the
    same, and writes nothing. Allocating a fresh `_seq` per replayed edge would re-XOR the row into
    its merkle leaf, and anti-entropy would chase a difference it is itself creating."""
    _endpoints(store, "z1", "z2")
    seg = [{"f": "z1", "t": "z2", "label": "lineage", "props": {"force": "semantic"}}]

    before = _last_seq(store, store.artifacts.origin)
    sync._apply_edges(store, seg)
    after_first = _last_seq(store, store.artifacts.origin)
    ver_first = store.graph.db.read().execute(
        "SELECT _origin, _seq FROM edge WHERE src = 'z1'").fetchone()

    for _ in range(4):
        sync._apply_edges(store, seg)
    after_replays = _last_seq(store, store.artifacts.origin)
    ver_after = store.graph.db.read().execute(
        "SELECT _origin, _seq FROM edge WHERE src = 'z1'").fetchone()

    assert after_first == before, "consuming an edge must allocate NO local proper time"
    assert after_replays == before, "a replayed segment must allocate no proper time either"
    assert (ver_after["_origin"], ver_after["_seq"]) == (ver_first["_origin"], ver_first["_seq"])
    assert store.graph.count_edges() == 1
    assert _feed(store) == set()


def test_the_reserved_edge_origin_is_deterministic_and_nul_separated(store):
    """Determinism is what makes replay free. The NUL separator keeps ("ab","c") and ("a","bc")
    distinct — contract §3 calls it load-bearing for `edge_key`, and it is load-bearing here for
    the same reason."""
    o = sync._consumed_edge_origin
    assert o("a", "b", "l") == o("a", "b", "l")
    assert o("ab", "c", "l") != o("a", "bc", "l")
    assert o("a", "b", "l") != o("a", "b", "m")
    assert o("a", "b", "l") != o("b", "a", "l")          # direction is not symmetric


def test_a_peer_claiming_our_origin_is_re_stamped(store):
    """A segment asserting `_origin = me` is re-stamped reserved, so the row stays out of this
    node's publish feed. The wire format does not carry `(_origin,_seq)`, so the case is one the
    consume path handles by construction rather than one it meets today."""
    me = sync._origin_of(store)
    _endpoints(store, "c1", "c2")
    sync._apply_edges(store, [{"f": "c1", "t": "c2", "label": "lineage",
                               "props": {"_origin": me, "_seq": 5}}])
    row = store.graph.db.read().execute(
        "SELECT _origin FROM edge WHERE src = 'c1'").fetchone()
    assert row["_origin"] != me
    assert row["_origin"].startswith(sync._CONSUMED_EDGE_NS)
    assert _feed(store) == set()


def test_real_peer_provenance_is_preserved_when_the_wire_ever_carries_it(store):
    """The reserved origin is a fallback for when the segment carries no provenance. A segment that
    does carry a genuine `(_origin,_seq)` keeps it: that is the real measurement, and the reserved
    origin exists only to stand in for it."""
    _endpoints(store, "p1", "p2")
    sync._apply_edges(store, [{"f": "p1", "t": "p2", "label": "lineage",
                               "props": {"_origin": "node-Z", "_seq": 77}}])
    row = store.graph.db.read().execute(
        "SELECT _origin, _seq FROM edge WHERE src = 'p1'").fetchone()
    assert (row["_origin"], row["_seq"]) == ("node-Z", 77)
    assert _feed(store) == set()


def test_a_locally_authored_edge_is_untouched_by_the_consume_path(store):
    """The consume path leaves this node's own authorship alone. `compare_version` reads a reserved
    origin against a local one as unordered and keeps the local row; the two are genuinely
    incomparable, so no tiebreak is synthesized (RESOLVED-3)."""
    _endpoints(store, "m1", "m2")
    store.graph.add_edges([("m1", "m2", "lineage", {})])
    mine = store.graph.db.read().execute(
        "SELECT _origin, _seq FROM edge WHERE src = 'm1'").fetchone()

    sync._apply_edges(store, [{"f": "m1", "t": "m2", "label": "lineage", "props": {}}])
    after = store.graph.db.read().execute(
        "SELECT _origin, _seq FROM edge WHERE src = 'm1'").fetchone()
    assert (after["_origin"], after["_seq"]) == (mine["_origin"], mine["_seq"])
    assert ("m1", "m2", "lineage") in _feed(store)


def test_a_partial_edge_apply_raises_so_the_cursor_is_held(store, monkeypatch):
    """`_add_chunk` rolls a failing edge back per savepoint and carries on, so a partial apply
    raises nothing by itself. The shortfall is the only evidence available, and `_consume_stream`
    writes a later `last_key` whenever no exception reached it — which would leave those edges
    behind a monotone StartAfter marker. Same guard `_apply_artifacts` puts on `put_many`."""
    _endpoints(store, "s1", "s2")
    monkeypatch.setattr(store.graph, "add_edges", lambda edges, **kw: 0)
    with pytest.raises(RuntimeError, match="partially applied"):
        sync._apply_edges(store, [{"f": "s1", "t": "s2", "label": "lineage", "props": {}}])


# ── the unordered-conflict decision ──────────────────────────────────────────────────────────────
def test_unordered_is_declined_not_tiebroken(store):
    """Two rows from different `_origin`s are genuinely unordered, so there is no later version to
    pick. The local row stands and the conflict is recorded (RESOLVED-3, §C.7)."""
    store.artifacts.put_artifact({"id": "u", "content_type": "text/markdown", "body": "local"})
    local_ver = store.artifacts.version_of("u")

    incoming = [{"id": "u", "content_type": "text/markdown", "body": "remote",
                 "_origin": "node-Z", "_seq": 999}]
    handled = sync._apply_artifacts(store, incoming)

    assert handled == 1, "a declined row is HANDLED — it must not look like a partial apply"
    assert store.artifacts.version_of("u") == local_ver, "an order was synthesized"
    assert store.artifacts.get_artifact("u")["body"] == "local"


def test_declination_is_recorded_and_reported(store):
    """A declined version is counted and sampled, so permanent divergence is a number a reader can
    look at rather than churn in the reconcile loop."""
    store.artifacts.put_artifact({"id": "u2", "content_type": "text/markdown"})
    incoming = [{"id": "u2", "content_type": "text/markdown",
                 "_origin": "node-Z", "_seq": 5}]
    sync._apply_artifacts(store, incoming)
    rep = sync.unordered_report(store)
    assert rep["declined"] == 1
    assert "u2" in rep["sample"]
    assert "no tiebreak" in rep["basis"]


def test_declination_is_stable_across_rounds(store):
    """Re-offering the same version every round leaves the count where it was, so the metric
    measures divergence rather than round frequency."""
    store.artifacts.put_artifact({"id": "u3", "content_type": "text/markdown"})
    incoming = [{"id": "u3", "content_type": "text/markdown", "_origin": "node-Z", "_seq": 5}]
    for _ in range(4):
        sync._apply_artifacts(store, incoming)
    assert sync.unordered_report(store)["declined"] == 1


def test_declination_markers_never_replicate(store, s3):
    """A declination marker records this node's local decision, so it stays local. It is a fact
    about this observer, not about the vertex, and the shared graph carries facts about vertices."""
    assert not sync._is_replicated(sync._DECLINED_CT)
    store.artifacts.put_artifact({"id": "u4", "content_type": "text/markdown"})
    sync._apply_artifacts(store, [{"id": "u4", "content_type": "text/markdown",
                                   "_origin": "node-Z", "_seq": 5}])
    sync.publish_merkle_incremental(store)
    ids = {d["id"] for d in _seg_docs(s3, sync._MESH_LEAF_PREFIX) if "id" in d}
    assert sync._DECLINED_ID not in ids


def test_a_newer_version_from_the_same_origin_still_applies(store):
    """The marker suppresses one declined version, scoped to that version rather than to the vertex.
    A genuinely newer version from the same origin is still adopted, so declining one offer leaves
    the vertex reachable."""
    store.artifacts.put_artifact({"id": "u5", "content_type": "text/markdown",
                                  "_origin": "node-Z", "_seq": 1}, stamp_rev=False)
    sync._apply_artifacts(store, [{"id": "u5", "content_type": "text/markdown", "body": "newer",
                                   "_origin": "node-Z", "_seq": 2}])
    assert store.artifacts.get_artifact("u5")["body"] == "newer"


# ── consumed peer rows are queued for the content drain (authorship is preserved) ────────────────
def test_consumed_peer_rows_are_queued_for_the_content_drain(store):
    """`page_by_origin` walks the local origin, so a consumed row is outside
    `promote_local_content`'s reach and its content would live only on this box. A separate queue
    carries those rows to the drain, which keeps authorship intact: stamping a local version instead
    would disable the anti-downgrade guard and let an old backlog copy overwrite a current row."""
    items = [{"id": "pc1", "content_type": "text/markdown", "_origin": "node-B", "_seq": 1,
              "content_ref": "cas/" + "a" * 64},
             {"id": "pc2", "content_type": "text/markdown", "_origin": "node-B", "_seq": 2}]
    sync._apply_artifacts(store, items)
    q = sync.consumed_pending(store)
    assert q["pending"] == ["pc1"], "only rows carrying content belong in the drain queue"
    # authorship is untouched — nothing was forged
    assert store.artifacts.version_of("pc1") == ("node-B", 1)


def test_locally_authored_rows_are_not_queued(store):
    store.artifacts.put_artifact({"id": "own", "content_type": "text/markdown",
                                  "content_ref": "cas/" + "b" * 64})
    assert sync.consumed_pending(store)["pending"] == []


def test_drain_consumed_removes_only_what_was_promoted(store):
    items = [{"id": "d%d" % i, "content_type": "text/markdown", "_origin": "node-B", "_seq": i,
              "content_ref": "cas/%064d" % i} for i in range(1, 4)]
    sync._apply_artifacts(store, items)
    sync.drain_consumed(store, ["d1"])
    assert sync.consumed_pending(store)["pending"] == ["d2", "d3"]


def test_consume_queue_is_idempotent_across_replays(store):
    """Segments are replayed by design: consume is retried on any held cursor, so the queue counts
    a row once however many times it arrives."""
    items = [{"id": "rp", "content_type": "text/markdown", "_origin": "node-B", "_seq": 1,
              "content_ref": "cas/" + "c" * 64}]
    for _ in range(5):
        sync._apply_artifacts(store, items)
    assert sync.consumed_pending(store)["pending"] == ["rp"]


# ── federation ───────────────────────────────────────────────────────────────────────────────────
def test_export_page_pages_by_keyset(store):
    for i in range(10):
        store.artifacts.put_artifact({"id": "f%02d" % i, "content_type": "text/markdown"})
    seen, after = [], ""
    while True:
        pg = federation.export_page(store, 0, 3, after=after)
        if not pg["artifacts"]:
            break
        seen += [a["doc"]["id"] for a in pg["artifacts"]]
        if pg["next_after"] == after:
            break
        after = pg["next_after"]
    assert seen == sorted(seen) and len(set(seen)) == 10


def test_export_page_refuses_a_dropped_cursor(store):
    """`export_page` raises when handed an offset with no cursor. `genesis.py`'s op.mesh.export
    handler does not forward `after`, and serving page 1 for every offset would be a plausible
    wrong answer: the caller receives rows, so nothing looks broken."""
    store.artifacts.put_artifact({"id": "q", "content_type": "text/markdown"})
    with pytest.raises(RuntimeError, match="after"):
        federation.export_page(store, 25, 25)


def test_export_page_still_excludes_operational(store):
    store.artifacts.put_artifact({"id": "ok", "content_type": "text/markdown"})
    sync._put_op(store, {"id": "cur", "content_type": sync._S3SYNC_CT})
    pg = federation.export_page(store, 0, 50)
    assert [a["doc"]["id"] for a in pg["artifacts"]] == ["ok"]


# ── the lattice path answers through the lattice API ─────────────────────────────────────────────
def test_no_skip_reaches_the_lattice_path(store, s3):
    """No mesh entrypoint issues raw SQL against a lattice store. `SKIP` degrades badly with table
    depth against a keyset page, and contract §0.6 excludes it. The tripwire covers every alive
    entrypoint at once, so the property is asserted over the surface rather than per call site."""
    calls = []

    class Tripwire:
        def query(self, q, params=None, **kw):
            calls.append(q)
            raise AssertionError("raw SQL reached a lattice store: %r" % q)

    store.artifacts.c = Tripwire()
    store.graph.c = Tripwire()
    for i in range(5):
        store.artifacts.put_artifact({"id": "n%d" % i, "content_type": "text/markdown"})
    store.graph.add_edges([("n0", "n1", "lineage", {})])

    # Every alive mesh entrypoint, end to end: each answers from the lattice API.
    sync.publish_merkle_incremental(store)
    sync.reconcile_via_s3(store)
    store.artifacts.list_by_leaf(0)
    store.graph.list_by_leaf(0)
    sync.refresh_leaves(store, [0, 1])
    sync.reconcile_merkle(store)
    sync.merkle_coverage(store)
    sync._replicated_count(store)
    sync.mesh_lag(store)
    sync.publish_backlog_now(store)
    federation.export_page(store, 0, 5)
    assert calls == []


# ── S3 substrate resiliency: Merkle anti-entropy (S3-SUBSTRATE-RESILIENCY.md §4) ───────────────────
# `publish_merkle_incremental` and `reconcile_via_s3` are the steady-state tail. These pin the
# convergence property end to end through S3 with the segment feeds off: Merkle alone carries a node
# from empty to converged, so recovery is the same operation as steady state.
class _Ident:
    @staticmethod
    def encrypt(b):
        return b

    @staticmethod
    def decrypt(b):
        return b


def _merkle_fleet(monkeypatch, tmp_path, node_ids):
    """N lattice stores over one shared FakeS3 with a settable 'current node'. The segment feeds are
    left uninvoked, so any convergence observed is anti-entropy alone."""
    shared = FakeS3()
    cur = {"id": node_ids[0]}
    monkeypatch.setattr(sync, "_mesh_s3", lambda store: shared)
    monkeypatch.setattr(sync, "_fernet", lambda store: _Ident())
    monkeypatch.setattr(sync, "_node_id", lambda: cur["id"])
    stores = {n: Store(tmp_path / ("%s.db" % n), origin=n) for n in node_ids}

    def as_node(n):
        cur["id"] = n

    return shared, stores, as_node


def _live_root(store):
    return (store.artifacts.get_artifact(sync._S3_MERKLE_CURSOR) or {}).get("root")


def test_merkle_only_convergence_end_to_end(monkeypatch, tmp_path):
    """T1. Two lattice nodes, segment feeds off. A holds 200 vertices and 200 edges; B is empty. B
    converges to A's exact state — vertices and edges — by Merkle anti-entropy alone: the tree diff
    localizes the difference, only differing leaves transfer (each a mixed vertex+edge object), peer
    authorship survives, and a further round moves nothing. Recovery is steady state."""
    shared, S, as_node = _merkle_fleet(monkeypatch, tmp_path, ["node-A", "node-B"])
    A, B = S["node-A"], S["node-B"]
    ids = ["wn-%04d" % i for i in range(200)]          # skewed, real-shaped ids: spread across leaves
    as_node("node-A")
    for i in ids:
        A.artifacts.put_artifact({"id": i, "content_type": "text/markdown"})
    A.graph.add_edges([(ids[i], ids[(i + 1) % len(ids)], "lineage", {}) for i in range(len(ids))])
    pub = sync.publish_merkle_incremental(A)
    assert pub.get("uploaded", 0) > 0 and pub.get("root"), pub

    as_node("node-B")
    for _ in range(16):                                # ~400 diff leaves / 64-per-round -> ~7 rounds
        sync.reconcile_via_s3(B)
        if _live_root(B) == pub["root"]:
            break
    assert _live_root(B) == pub["root"], "B did not converge to A's root by Merkle alone"
    for i in ids:
        assert B.artifacts.get_artifact(i) is not None, "missing vertex after convergence: %s" % i
    assert B.graph.count_edges() == A.graph.count_edges() == len(ids), \
        "edges did not converge by Merkle alone"
    assert B.artifacts.version_of(ids[0])[0] == "node-A", "peer authorship lost in leaf transfer"
    # Provably converged: a further round transfers nothing (also the no-livelock guarantee, M8).
    final = sync.reconcile_via_s3(B)
    assert final["leaves_fetched"] == 0, "converged node still pulling leaves (livelock)"


def test_edge_merkle_identity_is_node_invariant(monkeypatch, tmp_path):
    """T5 / contract M2e — an edge authored on A (real `(_origin,_seq)`) and the same edge consumed
    on B (reserved `_local:edge:` origin, seq 1) XOR the same value into their leaf. An edge keyed
    on `_seq` like a vertex leaves the two nodes permanently different; keying on node-invariant
    content is what lets edges converge at all."""
    shared, S, as_node = _merkle_fleet(monkeypatch, tmp_path, ["node-A", "node-B"])
    A, B = S["node-A"], S["node-B"]
    as_node("node-A")
    A.graph.add_edge("u", "v", "lineage", {})                       # authored: (node-A, seq)
    as_node("node-B")
    sync._apply_edges(B, [{"f": "u", "t": "v", "label": "lineage", "props": {}}])  # consumed: reserved
    assert A.artifacts.merkle_leaves() == B.artifacts.merkle_leaves(), \
        "the same edge hashed to different leaves on author vs consumer"


def test_edge_hash_is_node_invariant_and_deterministic():
    """M2e at the unit level: `edge_hash` ignores the per-node fields (`_origin`/`_seq`) and the
    endpoints already folded into `edge_key`, is order-insensitive in props, and treats an explicit
    NULL as equivalent to an absent value — so two nodes holding the same edge always agree."""
    from mantle.db import constants as K
    key = K.edge_key("u", "v", "lineage")
    base = {"is_origin": 1, "force": "grant"}
    assert K.edge_hash(key, base) == K.edge_hash(key, {"force": "grant", "is_origin": 1})   # order
    assert K.edge_hash(key, base) == K.edge_hash(key, {**base, "_origin": "node-A", "_seq": 5})
    assert K.edge_hash(key, base) == K.edge_hash(key, {**base, "_origin": "_local:edge:x", "_seq": 1})
    assert K.edge_hash(key, {}) == K.edge_hash(key, {"force": None, "order_key": None})       # NULL==absent
    assert K.edge_hash(key, base) != K.edge_hash(key, {"is_origin": 0, "force": "grant"})     # content matters


def test_steady_state_pulls_zero_leaves(monkeypatch, tmp_path):
    """T3 — converged peers exchange one 32 KB tree and transfer nothing. The flat steady-state cost
    is the whole reason Merkle scales where the log feed cannot. Roots match despite different
    `_origin`, because row identity is `(id, _seq)`, not `(id, _origin, _seq)`."""
    shared, S, as_node = _merkle_fleet(monkeypatch, tmp_path, ["node-A", "node-B"])
    A, B = S["node-A"], S["node-B"]
    for i in range(20):
        as_node("node-A"); A.artifacts.put_artifact({"id": "s%d" % i, "content_type": "text/markdown"})
        as_node("node-B"); B.artifacts.put_artifact({"id": "s%d" % i, "content_type": "text/markdown"})
    as_node("node-A"); sync.publish_merkle_incremental(A)
    as_node("node-B"); sync.publish_merkle_incremental(B)
    as_node("node-B")
    out = sync.reconcile_merkle(B)
    assert out["applied"] == 0 and out["leaves_fetched"] == 0
    assert out["peers"].get("node-A") == 0, "identical corpora were not recognised as converged"


def test_operational_churn_does_not_move_published_root(store, s3):
    """T4 / contract M7 — writing only operational rows (the cursor writes every consume round makes)
    between two incremental publishes leaves the published summary byte-identical. Operational state
    stays out of the replication tree, because it differs per node and two converged nodes need one
    root."""
    for i in range(10):
        store.artifacts.put_artifact({"id": "o%d" % i, "content_type": "text/markdown"})
    sync.publish_merkle_incremental(store)
    key = "%snode-A.json" % sync._MESH_MERKLE_PREFIX
    before = bytes(s3.objects[key])
    for i in range(5):
        sync._put_op(store, {"id": "s3.churn.%d" % i, "content_type": sync._S3SYNC_CT, "v": i})
    sync.publish_merkle_incremental(store)
    assert bytes(s3.objects[key]) == before, "operational churn moved the published Merkle root"


def test_row_hash_matches_between_store_and_mesh():
    """T6 / contract M2 — the store's incremental leaves (`constants.row_hash`) and the mesh's tree
    (`mesh.merkle.row_hash`) hash a row identically, so a node's own live tree matches its published
    tree. Two hashes would put reconcile on the trail of a difference that exists only between the
    node and itself. The `rev` parameter in `mesh.merkle` is fed `_seq` by every caller."""
    from mantle.mesh import merkle
    from mantle.db import constants as K
    for aid, sq in [("wn-1", 0), ("x", 7), ("y", 123456), ("z", None)]:
        assert merkle.row_hash(aid, sq) == K.row_hash(aid, sq), (aid, sq)
    assert merkle.leaf_of("wn-1") == K.leaf_of("wn-1")
    assert merkle.DEFAULT_LEAVES == K.DEFAULT_LEAVES


def test_incremental_publish_declines_on_non_lattice(monkeypatch):
    """`publish_merkle_incremental` rests on the incremental leaf tree and `list_by_leaf`. Handed a
    store that has neither, it returns `published: 0` with the reason named, so the caller learns
    that nothing was published rather than paying for a full scan under an 'incremental' label."""
    fake = FakeS3()
    monkeypatch.setattr(sync, "_mesh_s3", lambda s: fake)

    class _NotLattice:
        pass

    s = type("S", (), {"artifacts": _NotLattice(), "graph": None,
                       "content": None, "keys_dir": None})()
    out = sync.publish_merkle_incremental(s)
    assert out["published"] == 0 and out["reason"] == "not-a-lattice-store"


def test_reach_answers_a_miss_without_holding_the_whole_graph(monkeypatch, tmp_path):
    """The reach primitive — a limited ember. B holds nothing, yet answers for an id by reaching the
    substrate: it fetches the one leaf that contains the id from a publisher's tree, rather than the
    whole graph. Authorship is preserved. An id nobody holds reads as a miss (`reached=False`,
    `resolve -> None`), which is the absence of a row, not a row of absence."""
    shared, S, as_node = _merkle_fleet(monkeypatch, tmp_path, ["node-A", "node-B"])
    A, B = S["node-A"], S["node-B"]
    ids = ["wn-%04d" % i for i in range(200)]
    as_node("node-A")
    for i in ids:
        A.artifacts.put_artifact({"id": i, "content_type": "text/markdown"})
    sync.publish_merkle_incremental(A)                         # A publishes its tree + leaves to S3

    as_node("node-B")
    assert B.artifacts.get_artifact("wn-0050") is None         # B holds nothing
    r = sync.reach_index(B, "wn-0050")
    assert r["reached"] is True and r["from"] == "node-A", r
    assert B.artifacts.get_artifact("wn-0050") is not None     # now held, purely by reaching
    assert B.artifacts.version_of("wn-0050")[0] == "node-A"    # authorship preserved through the reach
    # limited: a reach pulls one leaf, so B holds a fraction of the 200 rows.
    assert B.artifacts.count() < len(ids)
    # resolve() front door: held -> returns; a fresh miss -> reaches and returns
    assert sync.resolve(B, "wn-0050")["id"] == "wn-0050"
    assert sync.resolve(B, "wn-0120")["id"] == "wn-0120"
    # an id nobody holds reads as a miss
    assert sync.reach_index(B, "does-not-exist")["reached"] is False
    assert sync.resolve(B, "does-not-exist") is None


def test_peers_are_artifacts_with_cas_addresses(monkeypatch, tmp_path):
    """A peer is an artifact too. A publishes itself as a peer-artifact whose content is a
    CAS-addressed manifest of its measured state (Merkle root, leaf count, envelope); B, reconciling
    the mesh, receives that peer-artifact and so holds the CAS address of peer A. Peers are
    discovered through the same path as everything else, so there is no separate directory file to
    keep current."""
    shared, S, as_node = _merkle_fleet(monkeypatch, tmp_path, ["node-A", "node-B"])
    A, B = S["node-A"], S["node-B"]
    as_node("node-A")
    for i in range(50):
        A.artifacts.put_artifact({"id": "wn-%04d" % i, "content_type": "text/markdown"})
    m = sync.publish_manifest(A)                       # A publishes itself as a peer-artifact
    sync.publish_merkle_incremental(A)
    assert m["published"] and m["content_ref"].startswith("cas/")
    pa = A.artifacts.get_artifact("node-A")
    assert pa["content_type"] == sync._OBSERVER_CT and pa["content_ref"] == m["content_ref"]
    assert shared.exists(m["content_ref"])             # the manifest lives in CAS at that address

    as_node("node-B")
    for _ in range(16):
        sync.reconcile_via_s3(B)                       # B reconciles -> receives A's peer-artifact
        if B.artifacts.get_artifact("node-A") is not None:
            break
    ps = sync.peers(B)
    assert any(p["node"] == "node-A" and str(p.get("content_ref", "")).startswith("cas/")
               for p in ps), ps                        # B now holds peer A's CAS address
    assert all(p["node"] != "node-B" for p in ps)      # peers() excludes self


def test_reach_prefers_the_measured_useful_peer(monkeypatch, tmp_path):
    """Attention is measured: a peer that has answered carries demand mass on its peer-artifact, so
    `_reach_candidates` ranks it first for the next miss. The ordering comes from the measurement,
    so a peer earns its place by answering."""
    shared, S, as_node = _merkle_fleet(monkeypatch, tmp_path, ["node-A", "node-B"])
    B = S["node-B"]
    v = B.artifacts
    # B knows two peers (artifacts), and has reached through node-A more than node-C
    v.put_artifact({"id": "node-A", "content_type": sync._OBSERVER_CT, "node": "node-A",
                    "content_ref": "cas/a"})
    v.put_artifact({"id": "node-C", "content_type": sync._OBSERVER_CT, "node": "node-C",
                    "content_ref": "cas/c"})
    for _ in range(5):
        sync._demand_touch(B, "node-A")                # node-A has been useful
    sync._demand_touch(B, "node-C")                    # node-C, once
    as_node("node-B")
    order = sync._reach_candidates(B)
    assert order[0] == "node-A" and "node-C" in order  # the measured-useful peer is tried first
    """Demurrage (the 2nd law) through the one prism.law kernel: same mass, older last-touch -> lower
    current mass. This is what makes the cold end of the cache cold."""
    from mantle.mesh import demand as D
    import time
    now = time.time()
    tau = D._tau()
    fresh = D.current_mass({"mass": 5.0, "ts": now}, now=now, tau=tau)
    stale = D.current_mass({"mass": 5.0, "ts": now - tau}, now=now, tau=tau)
    assert fresh == 5.0 and 0.0 < stale < fresh          # decayed by exactly exp(-1) after one tau


def test_demand_cache_evicts_coldest_keeps_hot_and_own(monkeypatch, tmp_path):
    """The demand cache is what keeps a limited ember limited: reached rows carry a mass, and when
    the cache exceeds its budget the coldest (lowest decayed mass) are evicted first. Hot rows
    survive; own-authored rows carry no demand entry, so they are outside the candidate set; and the
    authoritative copy stays in the substrate, so an evicted row can be re-reached."""
    from mantle.mesh import demand as D
    shared, S, as_node = _merkle_fleet(monkeypatch, tmp_path, ["node-A", "node-B"])
    A, B = S["node-A"], S["node-B"]
    ids = ["wn-%04d" % i for i in range(60)]
    as_node("node-A")
    for i in ids:
        A.artifacts.put_artifact({"id": i, "content_type": "text/markdown"})
    sync.publish_merkle_incremental(A)

    as_node("node-B")
    B.artifacts.put_artifact({"id": "B-own", "content_type": "text/markdown"})   # own -> no demand row
    for i in ids:
        sync.reach_index(B, i)                            # each reached -> demand mass 1
    hot = ids[:5]
    for _ in range(10):
        for h in hot:
            D.touch(B, h)                                 # heat the hot set well above the cold
    before = B.artifacts.demand_count()
    assert before == 60 and B.artifacts.demand_get("B-own") is None   # own carries no demand

    out = D.evict(B, budget=10)
    assert out["evicted"] == before - 10 and B.artifacts.demand_count() == 10
    for h in hot:
        assert B.artifacts.get_artifact(h) is not None    # hot survived
    assert B.artifacts.get_artifact("B-own") is not None  # own-authored never evicted
    # an evicted (cold) row is re-reachable from the substrate, so eviction frees space rather than
    # discarding the row
    gone = next(i for i in ids if B.artifacts.get_artifact(i) is None)
    assert sync.reach_index(B, gone)["reached"] is True and B.artifacts.get_artifact(gone) is not None


# ── mesh egress is the ungated public set; a grant-gated collection stays on the node ─────────────
def test_grant_gated_collection_is_withheld_from_the_published_tree(store, s3):
    """What leaves the node is the ungated public top. A collection is made private by minting a
    grant on it — the owner's Read grant, a row rather than a flag — and its rows are then held and
    queried locally while being subtracted from the tree the node advertises. Public rows publish,
    gated rows stay local, and `_replicated_count` counts only what the mesh carries."""
    from mantle.db import access
    try:
        access._api()                                # the grant subsystem must be importable here
    except Exception:
        pytest.skip("grant subsystem (mantle lattice_api) not on the path")
    store.artifacts.put_artifact({"id": "pub-1", "content_type": "text/markdown",
                                  "collection_id": "universe"})
    store.artifacts.put_artifact({"id": "pub-2", "content_type": "text/markdown"})
    access.mint_owner_read_grant(store, "private.alice", "alice")     # a grant gates the collection
    for i in range(4):
        store.artifacts.put_artifact({"id": "sec-%d" % i, "content_type": "text/markdown",
                                      "collection_id": "private.alice", "created_by": "alice"})

    sync.publish_merkle_incremental(store)
    ids = {d["id"] for d in _seg_docs(s3, sync._MESH_LEAF_PREFIX) if "id" in d}
    assert "pub-1" in ids and "pub-2" in ids
    assert not any(i.startswith("sec-") for i in ids), "a grant-gated row reached a published leaf"
    assert store.artifacts.get_artifact("sec-0") is not None          # still held + locally queryable
    # the grant artifact itself is not a replicated content type either
    assert sync._replicated_count(store) == 2                         # only the two public rows


def test_a_made_public_row_meshes_out_of_a_gated_collection(store, s3):
    """Making a private row public is a grant to the public entity, so the row itself is untouched.
    It then meshes out even though it sits in a gated collection, while its still-private neighbours
    stay local."""
    from mantle.db import access
    try:
        access._api()
    except Exception:
        pytest.skip("grant subsystem not on the path")
    access.mint_owner_read_grant(store, "private.alice", "alice")
    for i in range(3):
        store.artifacts.put_artifact({"id": "sec-%d" % i, "content_type": "text/markdown",
                                      "collection_id": "private.alice"})
    access.grant_read(store, "sec-1", access.PUBLIC_PRINCIPAL, "alice")   # make sec-1 public

    sync.publish_merkle_incremental(store)
    ids = {d["id"] for d in _seg_docs(s3, sync._MESH_LEAF_PREFIX) if "id" in d}
    assert "sec-1" in ids, "a made-public row did not mesh out"
    assert "sec-0" not in ids and "sec-2" not in ids, "a still-private row leaked"


def test_no_grant_means_byte_identical_publish(store, s3):
    """With no grant present nothing is gated, and the publish carries every row. Public is the
    default: the access mechanism costs nothing until a grant exists."""
    for i in range(5):
        store.artifacts.put_artifact({"id": "p%d" % i, "content_type": "text/markdown"})
    out = sync.publish_merkle_incremental(store)
    ids = {d["id"] for d in _seg_docs(s3, sync._MESH_LEAF_PREFIX) if "id" in d}
    assert ids == {"p0", "p1", "p2", "p3", "p4"} and out.get("root")


def test_an_unmeasurable_envelope_is_omitted_not_published_as_zero(monkeypatch, tmp_path):
    """`_envelope_bytes` returns None when nothing can be measured, and `publish_manifest` then
    leaves the key out of the manifest entirely.

    `envelope: 0` is a number meaning "I can carry nothing" — the reading a genuinely full disk
    gives. A node that could not measure has no reading to give, and an absent key is how a peer
    learns that. A measured zero still publishes as 0.
    """
    shared, S, as_node = _merkle_fleet(monkeypatch, tmp_path, ["node-A"])
    A = S["node-A"]
    as_node("node-A")
    A.artifacts.put_artifact({"id": "wn-0001", "content_type": "text/markdown"})

    from prism import envelope as _env

    # measurable: the key is present and carries the measured number
    monkeypatch.setattr(_env, "disk_free_bytes", lambda _p: 4096)
    m = sync.publish_manifest(A)
    assert m["published"]
    assert sync._envelope_bytes(A) == 4096

    # a real measured zero still publishes 0 — that is a measurement
    monkeypatch.setattr(_env, "disk_free_bytes", lambda _p: 0)
    monkeypatch.setattr(_env, "mem_limit_bytes", lambda: None)
    assert sync._envelope_bytes(A) == 0

    # unmeasurable: None, and the key is gone from the manifest entirely
    monkeypatch.setattr(_env, "disk_free_bytes", lambda _p: None)
    monkeypatch.setattr(_env, "mem_limit_bytes", lambda: None)
    assert sync._envelope_bytes(A) is None

    import json
    m2 = sync.publish_manifest(A)
    assert m2["published"]

    # Read the manifest back off the wire: what a peer receives is the property under test, and a
    # local dict is one serialization step short of it.
    published = json.loads(sync._fernet(A).decrypt(shared.get(m2["content_ref"])).decode("utf-8"))
    assert "envelope" not in published, f"unmeasurable envelope still published: {published}"

    # The positive direction: with `envelope` never published at all, "not in published" would also
    # hold. Assert the key is present and carries the measured number when a measurement exists —
    # including a measured zero, which an `if _env:` truthiness guard would drop.
    monkeypatch.setattr(_env, "disk_free_bytes", lambda _p: 4096)
    m3 = sync.publish_manifest(A)
    pub3 = json.loads(sync._fernet(A).decrypt(shared.get(m3["content_ref"])).decode("utf-8"))
    assert pub3.get("envelope") == 4096, f"measured envelope not published: {pub3}"

    monkeypatch.setattr(_env, "disk_free_bytes", lambda _p: 0)
    m4 = sync.publish_manifest(A)
    pub4 = json.loads(sync._fernet(A).decrypt(shared.get(m4["content_ref"])).decode("utf-8"))
    assert "envelope" in pub4 and pub4["envelope"] == 0, (
        f"a REAL measured zero must be published, not treated as absent: {pub4}")
    assert published["node"] == "node-A"          # the rest of the manifest is intact
    assert "root" in published and "leaves" in published
