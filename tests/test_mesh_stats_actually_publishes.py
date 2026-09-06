"""Mesh stats must actually reach the shared bucket — asserted by EFFECT, never by absence of error.

Why this file exists. `MESH_STATS_PREFIX` was deleted as COLLATERAL in `8716d55`, a commit whose
subject was replacing `_node_id`'s body. Its two uses survived, so both raised `NameError` — and
both sit inside handlers that swallow (`_publish_to_mesh` has a bare `except Exception: pass`).
**Mesh stats published nothing and read nothing from 2026-07-30 to 2026-08-25**, while
`read_mesh_stats`' docstring promised "the whole mesh … so a host whose serve is down still
appears". Restored in `34b1958`, with the value recovered from `10a4a93` by `git log -S` rather than
inferred from the prose.

**The shape of the test is the whole point.** A test that calls these functions and asserts they
do not raise passes over the dead feature — that is exactly what the swallow guarantees. So each
test below asserts the call reached the remote with the expected key. Under the bug the remote is
never touched, the assertion fails, and it names the missing constant.

Nothing here touches S3: the remote is a stub, and `_mesh_remote` is redirected onto it.
"""
from __future__ import annotations

import json

import pytest

from ember.surface import stats


class _Remote:
    """The two shapes `stats` reaches for: `.put(key, body, ctype)` and `._s3` / `.bucket`."""

    def __init__(self):
        self.puts = []
        self.prefixes = []
        self.bucket = "test-bucket"
        self._s3 = self

    def put(self, key, body, ctype):
        self.puts.append((key, body, ctype))

    def get_paginator(self, _op):
        return self

    def paginate(self, Bucket=None, Prefix=None):   # noqa: N803 — boto3's own casing
        self.prefixes.append(Prefix)
        return [{"Contents": []}]


@pytest.fixture
def remote(monkeypatch):
    r = _Remote()
    monkeypatch.setattr(stats, "_mesh_remote", lambda _store: r)
    monkeypatch.setattr(stats, "_node_id", lambda: "71")
    return r


def test_the_prefix_is_defined_and_is_the_recovered_value():
    """The one-line form of the bug. `git log -S` recovers `mesh-stats/` from the commit that
    introduced it, so this is the ORIGINAL value and not a reading of the docstrings."""
    assert stats.MESH_STATS_PREFIX == "mesh-stats/"


def test_publishing_reaches_the_remote_with_the_documented_key(remote):
    """`mesh-stats/<node>.json` — the layout both docstrings name.

    Asserts the put HAPPENED. `_publish_to_mesh` swallows every exception, so 'it did not raise'
    is precisely the assertion that stayed green for 26 days while nothing was published."""
    stats._publish_to_mesh(object(), {"whole_art": 5_600_000})

    assert remote.puts, (
        "nothing reached the remote — this is the 2026-07-30 defect: a NameError inside "
        "`except Exception: pass` makes publishing a no-op that looks healthy")
    key, body, ctype = remote.puts[0]
    assert key == "mesh-stats/71.json", key
    assert json.loads(body.decode("utf-8")) == {"whole_art": 5_600_000}
    assert ctype == "application/json"


def test_reading_lists_the_shared_prefix(remote):
    """The read half failed the same way, and an empty list is its normal answer — so the
    assertion has to be that the LISTING was attempted under the right prefix."""
    assert stats.read_mesh_stats(object()) == []
    assert remote.prefixes == ["mesh-stats/"], (
        "the mesh was never listed — under the bug `read_mesh_stats` returns [] without ever "
        "reaching S3, which is indistinguishable from a mesh of one host")


def test_a_missing_remote_is_still_a_quiet_no_op(monkeypatch):
    """The fail-soft half is deliberate and must stay: a host with no shared bucket configured
    publishes nothing and reads nothing, without failing the surface that called it."""
    monkeypatch.setattr(stats, "_mesh_remote", lambda _store: None)
    stats._publish_to_mesh(object(), {"a": 1})
    assert stats.read_mesh_stats(object()) == []
