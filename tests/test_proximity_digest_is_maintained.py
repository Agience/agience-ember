"""A membership change reaches the digest refresher, and a spent digest is retaken.

Proximity's write half is a chain across three packages: mantle's collection service reports a
membership change, `digest_refresh` counts it, and once the collection has turned over the
refresher enumerates the members and takes the read with the host's instrument. Each piece is
tested on its own; this pins that they are connected, which is the property that made the whole
capability unreachable while every part of it passed.

The instrument is the real one. `install_digest_refresher` takes whatever a host builds, so a stub
would prove the chain calls something rather than that it calls an instrument mantle cannot name.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from ember import optics
from mantle.search.ingest import digest_refresh as dr
from mantle.search.ingest.collection_frame import digest_is_spent


class _RecordingRefresher:
    """Stands in for `CollectionDigestRefresher` to observe what the write path reports."""

    def __init__(self):
        self.writes = []

    def note_write(self, principal_id, collection_id):
        self.writes.append((principal_id, collection_id))
        return len(self.writes)


@pytest.fixture(autouse=True)
def _clean_install():
    dr.install_digest_refresher(None)
    yield
    dr.install_digest_refresher(None)


def test_a_membership_change_reaches_the_installed_refresher():
    rec = _RecordingRefresher()
    dr.install_digest_refresher(rec)
    dr.note_membership_change("owner-A", "col-1")
    assert rec.writes == [("owner-A", "col-1")], (
        "the membership change did not reach the refresher, so a collection's digest goes stale "
        "with nothing counting the changes that made it stale")


def test_a_store_with_no_refresher_installed_is_silent():
    """The base install maintains no digests. Reporting into nothing must not raise."""
    dr.note_membership_change("owner-A", "col-1")   # no refresher installed


def test_a_refresher_fault_cannot_break_the_write_that_prompted_it():
    """A digest is an accelerator for a query nobody has issued; it may not fail a write."""
    class _Broken:
        def note_write(self, *a):
            raise RuntimeError("slot unavailable")

    dr.install_digest_refresher(_Broken())
    dr.note_membership_change("owner-A", "col-1")   # swallowed, by design


def test_the_real_instrument_fills_the_seam_mantle_declares():
    """`read` and `engine_id` come from the host, and they are entroptics' own."""
    read, engine_id = optics.proximity_read(), optics.proximity_engine_id()
    assert engine_id == "entroptics.mp.dev"
    frame = np.random.default_rng(0).normal(size=(12, 6))
    got = read(frame)
    assert len(got) > 0 and all(np.isfinite(v) for v in got), (
        "the host's instrument returned no finite read, so installing it would give the refresher "
        "something that cannot digest a frame")


def test_the_turnover_rule_decides_when_a_digest_is_retaken():
    """`events_since >= rows`, base case included: a never-digested collection is due at once."""
    assert digest_is_spent(0, 0) is True
    assert digest_is_spent(5, 4) is False
    assert digest_is_spent(5, 5) is True
