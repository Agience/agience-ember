"""`consolidate_nearvdup` reports the bound its scan ran under.

The operator archives rows, so a caller reads "consolidated N groups" as a statement about the
store. A capped run examined part of it, and a run that reports only `scope: len(arts)` reads the
same either way: ρ looks improved while the unexamined remainder still holds duplicates.

The cap stays, because it bounds real work on a 5.8 GB store — the live shard holds more than
60,000 `text/markdown` rows (316,421 in total) against a default `limit=60000`, so a run covers
roughly a fifth of it. Asking the store for `limit + 1` is what makes the bound observable:
receiving that extra row is the evidence there was more.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from ember import genesis


class _Arts:
    def __init__(self, n):
        self._rows = [{"id": "a-%d" % i, "content_type": "text/markdown", "state": "committed",
                       "content_ref": "cas/same" if i % 2 else "cas/other-%d" % i}
                      for i in range(n)]

    def list_artifacts(self, **kw):
        return iter(self._rows)

    def put_artifact(self, doc):        # registration writes; irrelevant here
        return doc


class _Store:
    def __init__(self, n):
        self.artifacts = _Arts(n)

    def put_artifact(self, doc):
        return doc


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch):
    """Keep the test on the bound rather than on consolidation's machinery."""
    monkeypatch.setattr(genesis, "register_consolidate_operators", lambda *a, **k: 0)
    import mantle.db.access as access
    monkeypatch.setattr(access, "is_public", lambda store, a: True)


def _run(n, limit):
    return genesis.consolidate_nearvdup(_Store(n), limit=limit, apply=False)


def test_a_TRUNCATED_scan_says_so():
    """More rows than the cap: the answer carries `truncated: True`, and `scope` reports what was
    examined. Slicing the list to `limit` and reporting its length makes this run read exactly like
    a complete one.
    """
    got = _run(50, limit=10)
    assert got["truncated"] is True
    assert got["limit"] == 10
    assert got["scope"] == 10, "scope must report what was EXAMINED, not what exists"


def test_a_COMPLETE_scan_says_so_TOO():
    """The other direction, which is what makes the flag worth reading: a `truncated` that is always
    True carries no information.
    """
    got = _run(8, limit=10)
    assert got["truncated"] is False
    assert got["scope"] == 8


def test_EXACTLY_at_the_limit_is_NOT_truncated():
    """The boundary, and the reason the scan asks for `limit + 1`. A corpus of exactly `limit` rows
    was examined completely; a corpus of `limit + 1` was not. Comparing `len(arts) >= limit` reads
    both as truncated, so the extra row is what separates them.
    """
    assert _run(10, limit=10)["truncated"] is False
    assert _run(11, limit=10)["truncated"] is True


def test_the_bound_rides_with_the_COUNTS_on_the_full_path():
    """A scope too small to consolidate returns early; a full run returns the counts. Both carry the
    bound, so a caller reading the counts learns what they cover.
    """
    early = _run(1, limit=10)
    full = _run(50, limit=10)
    for got in (early, full):
        assert {"scope", "limit", "truncated"} <= set(got), got
    assert "groups" in full and "consolidated" in full
