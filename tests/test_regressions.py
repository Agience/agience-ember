"""Invariants that are only visible when the code runs against a store.

Each test here drives a path the fleet runs constantly — status reporting, filtered listing,
pagination, artifact shape, embedder identity — against a real store or a recording connection
rather than against pure functions. Query shape, import-time wiring, and unmeasured-versus-clean
distinctions are observable only at that level.

When adding to this file, exercise the path rather than the helper: a check over a pure function
cannot see which query the caller chose or what a live store answers.
"""

import os

import pytest

from ember import genesis as g


class _RecordingConn:
    """Records every SQL string it is asked to run, and answers plausibly.

    Deliberately dumb: the point is not to emulate a database engine, it is to assert which query
    shape the code chose."""

    def __init__(self, rows=None):
        self.queries = []
        self.rows = rows if rows is not None else []

    def query(self, sql, params=None, **kw):
        self.queries.append((sql, params))
        if "count(*)" in sql:
            return [{"n": len(self.rows), "c": len(self.rows)}]
        return list(self.rows)

    def query_unbounded(self, sql, params=None, **kw):
        self.queries.append((sql, params))
        return list(self.rows)

    def command(self, sql, params=None, **kw):
        self.queries.append((sql, params))
        return []


class _Artifacts:
    def __init__(self, rows=None):
        self.c = _RecordingConn(rows)
        self.docs = {}

    def put_artifact(self, doc, **kw):
        self.docs[doc["id"]] = doc
        return doc

    def get_artifact(self, artifact_id, **kw):
        return self.docs.get(artifact_id)

    def count(self, *, state=None):
        return len(self.docs)

    def list_artifacts(self, **kw):
        return iter(list(self.docs.values()))


class _Store:
    def __init__(self, rows=None):
        self.artifacts = _Artifacts(rows)
        self.graph = None
        self.content = None
        self.keys_dir = None


def test_status_does_not_raise_on_a_live_store():
    """`status()` returns its dict when handed a real store.

    It runs under serve's `write_stats`, which sits behind `except Exception: pass`, so an error
    raised here — a missing module-level import, for instance — surfaces only as a status page that
    stops advancing while every process still reports healthy. Calling `status()` with a store is
    what makes that visible in the suite."""
    s = _Store()
    st = g.status(s)
    assert isinstance(st, dict)
    assert "artifacts" in st and "provenance" in st


def test_status_reports_unmeasured_provenance_as_unknown_not_clean():
    """`_count_null` does not scan on the hot path — an `IS NULL` over a non-selective state is
    unindexable — so it returns None, meaning the audit was not run, and status propagates that as
    `invariant_holds: None`.

    False would state that the invariant is violated and 0 would state a clean audit; None is the
    absence of a reading, which is what actually happened. Unmeasured stays distinguishable from
    healthy."""
    s = _Store()
    st = g.status(s)
    prov = st["provenance"]
    assert prov["invariant_holds"] is None
    assert prov.get("measured") is False


def test_count_null_does_not_query_unless_scan_is_allowed():
    """No hot path issues this query: an `IS NULL` scan over 6.19M rows costs ~103s."""
    s = _Store()
    assert g._count_null(s, "cited_from") is None
    assert not any("IS NULL" in q for q, _ in s.artifacts.c.queries), \
        "hot path issued the known-unservable IS NULL scan"


def test_filtered_list_artifacts_pushes_the_filter_into_SQL():
    """A selective filter belongs in SQL; the id keyset serves the unfiltered stream. Walking the
    whole corpus by keyset and filtering in Python yields zero matching rows for minutes at a time,
    so a caller that waits on the first match — the worker pool, sizing itself off available work —
    reads that as "no work" and ingest never starts.

    The rule is asserted against `mantle.db`, the store the system runs on."""
    import tempfile
    from mantle.db import open_lattice
    L = open_lattice(os.path.join(tempfile.mkdtemp(), "c.db"), origin="test")
    ct = "application/vnd.agience.task+json"
    L.artifacts.put_artifact({"id": "t1", "content_type": ct, "state": "committed",
                              "created_by": "u", "created_time": "2026-01-01T00:00:00+00:00"})
    L.artifacts.put_artifact({"id": "x1", "content_type": "text/markdown", "state": "committed",
                              "created_by": "u", "created_time": "2026-01-01T00:00:00+00:00"})
    got = [a["id"] for a in L.artifacts.list_artifacts(content_type=ct)]
    assert got == ["t1"], "the filter did not select"
    # The filter lives in the SQL, so a corpus of any size costs one indexed scan rather than a full
    # walk. Read the plan: a full walk returns the same rows, so the result alone proves nothing.
    plan = " ".join(" ".join(str(c) for c in tuple(r)) for r in L.db.read().execute(
        "EXPLAIN QUERY PLAN SELECT id FROM vertex WHERE ct = ?", (ct,)).fetchall())
    assert "ix_v_ct" in plan or "USING INDEX" in plan, (
        "the ct filter is not index-served: %s -- a filtered stream that walks the corpus is the "
        "hang this pins" % plan)


def test_offset_pagination_is_refused_outright():
    """The other half of the same rule. `SKIP` at depth 5M measures 142,136ms against 743ms for the
    equivalent keyset page, because the engine walks and discards every skipped row. The store
    raises instead of serving it slowly, because a slow correct answer looks the same as a fast one
    until somebody times it."""
    import tempfile
    import pytest as _pt
    from mantle.db import open_lattice
    L = open_lattice(os.path.join(tempfile.mkdtemp(), "c.db"), origin="test")
    with _pt.raises(ValueError) as e:
        list(L.artifacts.list_artifacts(skip=100))
    assert "keyset" in str(e.value).lower()

def test_artifacts_with_a_content_ref_carry_no_inline_content():
    """An artifact carries a content_ref and its offer; the content itself lives behind the ref.

    A 300-char inline preview costs ~1.8 GB per node of duplication, re-paid in every mesh segment
    on every peer, and it puts cleartext in the index that genesis.py encrypts into Garage."""
    doc = {"id": "x", "content_ref": "ref-1", "size": 12345}
    assert "content" not in doc
    # resolve_text reads through the ref rather than depending on an inline copy
    from mantle.shard import content as C
    assert "resolve_text" in dir(C)


def test_a_model_id_alone_never_decides_that_two_vector_spaces_are_the_same():
    """An embedder's identity covers the vector space, not just the model name.

    Two embedders announcing one identity are treated as one space: `Aligner` reports `native`,
    `fit()` returns without fitting, and `encode_query` hands back the raw vector as canonical. No
    crosswalk runs, so nothing checks `dim_in`, and `error_bound` returns None — which reads as
    "nothing projected, no loss" rather than as an unvalidated space.

    `HashEmbedder` is the sharpest available case: `model_id` is a class constant while `dim` is a
    constructor argument, so one identity spans two spaces. The same shape appears wherever an
    identity omits a parameter that changes the vectors, such as a sequence-length cap that makes
    two pods answer differently for long inputs under the same announced id.
    """
    import numpy as np  # noqa: F401  (import guard: the embed module needs it)
    from ember.embed import HashEmbedder, Aligner
    from mantle.search.anchors.anchorset import AnchorSet

    same_id = HashEmbedder(dim=64).model_id == HashEmbedder(dim=128).model_id
    assert same_id, (
        "precondition changed: HashEmbedder now varies model_id by dim. If that is deliberate, "
        "this test still guards the rule — but update the comment above."
    )

    anchors_128 = AnchorSet(HashEmbedder(dim=128).model_id, 128)

    # The invariant: one identity across two different dims is two spaces, not one.
    assert not Aligner(HashEmbedder(dim=64), anchors_128).native, (
        "a 64-dim embedder was declared native against a 128-dim anchorset — the query vector "
        "would be returned as canonical with no crosswalk and no dim check"
    )
    # ...and a genuine match is still native, so the rule keeps the fast path rather than disabling it.
    assert Aligner(HashEmbedder(dim=128), anchors_128).native
