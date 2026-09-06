"""The genuinely-new-rows split: new knowledge separated from no-op re-applies, by a read.

`mesh.sync._new_docs` / `_apply_artifacts(stats=...)` report how many of a consumed batch are
genuinely new and how many are already-held re-applies. `applied` is an upsert count, so it cannot
give that split: a large `applied` with no net row gain looks identical to real progress.

The split is a keyed `version_of` read — the same one `_split_unordered` already does — so it
touches neither the write path, `put_many`, nor `_rev`. Invariants:

  SPLITS      — an id already present locally is a re-apply; an absent id is new. Counting a
                re-apply as new is the false-progress signal this measurement exists to remove.
  HONEST-NULL — on a non-lattice store (no `version_of`) the split is None: the measurement was
                never taken, and a 0 would be a reading nobody took.
  ADDITIVE    — `stats=None` leaves `_apply_artifacts` byte-identical (the load-bearing `handled`
                return and the cursor guard are untouched); the metric is a pure read on the side.
"""
from __future__ import annotations

from mantle.mesh import sync as S
from mantle.db import open_lattice


def _store(tmp_path):
    L = open_lattice(str(tmp_path / "n.db"), origin="node-71")
    L.artifacts.ensure_schema()
    return L


def test_new_docs_splits_present_from_absent(tmp_path):
    store = _store(tmp_path)
    store.artifacts.put_artifact({"id": "known", "content_type": "application/x-thing"})
    new = S._new_docs(store, [{"id": "known"}, {"id": "fresh-1"}, {"id": "fresh-2"}])
    assert [d["id"] for d in new] == ["fresh-1", "fresh-2"]   # present → re-apply (not new); absent → new


def test_new_docs_all_new_on_empty_store(tmp_path):
    store = _store(tmp_path)
    new = S._new_docs(store, [{"id": "a"}, {"id": "b"}])
    assert len(new) == 2                                     # nothing held yet → both genuinely new


def test_new_docs_is_none_on_a_non_lattice_store():
    class _Bare:                                            # no artifacts.page_by_origin → not a lattice node
        artifacts = object()
    assert S._new_docs(_Bare(), [{"id": "x"}]) is None      # unknown: the split was never measured


def test_healthy_progress_prefers_new_over_applied():
    """The consumer of the split: `worker._healthy_progress` judges on `new`, not `applied`, so a
    large upsert count with no net gain while still behind reads as what it is — no progress."""
    from ember.runtime.worker import _healthy_progress
    # applied is large, but nothing is new and the node is still behind → no progress → unhealthy
    assert _healthy_progress({"applied": 723000, "new": 0, "segments_behind": 1837}, True) is False
    # genuinely gaining rows → healthy
    assert _healthy_progress({"applied": 500, "new": 12, "segments_behind": 1837}, True) is True
    # caught up is progress regardless of new, so a converged fleet stays up
    assert _healthy_progress({"applied": 0, "new": 0, "segments_behind": 0}, True) is True
    # a record with no `new` key (non-lattice store) falls back to `applied`
    assert _healthy_progress({"applied": 5, "segments_behind": 3}, True) is True
    assert _healthy_progress({"applied": 0, "segments_behind": 3}, True) is False
