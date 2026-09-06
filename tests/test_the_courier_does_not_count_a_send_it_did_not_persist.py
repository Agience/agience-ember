"""A message whose `sent` state does not persist is not counted, and does not go quiet.

THE DEFECT. `comms-loop.tick` publishes in this order: write the wire file -> set the artifact's
state to `sent` -> persist -> count. The persist was a bare `except Exception: pass`. So a failure
left the artifact `outbound` in the store while its file was already on the plane, and the next tick
published it again — every tick, for as long as the store kept refusing, with `published` counting a
send that was never durably recorded and nothing said anywhere.

AND THE DUPLICATE SEND ITSELF IS HARMLESS — MEASURED, NOT ASSUMED, WHICH IS WHY NO DEDUPE WAS
ADDED. The question put to John was whether receivers are idempotent. Reading the other half of this
same file answers it: the wire is idempotent end to end, keyed on the message id.

  · `wire_filename` is `<ts>-<from>-<id>.json` — deterministic per message, so a republish
    OVERWRITES one file rather than queueing a second copy;
  · ingest skips any id that already has a `seen-<node>/<id>` marker, and writes that marker only
    AFTER `put_artifact` succeeds — so a failed ingest retries and a successful one never repeats;
  · `put_artifact` upserts on id, so even a lost marker re-applies the same row.

So building a second dedupe would have been two mechanisms answering one question — the shape this
workspace has already paid for elsewhere (`oracle.LightConeGrantVerifier` against `check_access`).
What was missing is not deduplication. It is that a failure to persist was INVISIBLE and was counted
as a success. `published` now reports what the tick durably sent.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

from _paths import FLEET_EMBER                                    # noqa: E402

# Node 71's operator tooling lives in the `_fleet` sibling, not in this repository.
_LOOP = str(FLEET_EMBER / "comms-loop.py")


def _load(tmp_root):
    """Load the hyphenated node script by path — the `test_node_integrity.py` convention.

    `COMMS_ROOT` must be set BEFORE exec: the module refuses to load without it, deliberately, so
    that "a courier silently writing to a path nobody reads" cannot happen. That refusal is a
    module-level `raise SystemExit`, so it is an import-time contract, not a runtime one.
    """
    if not os.path.exists(_LOOP):
        pytest.skip("comms-loop.py not present")
    os.environ["COMMS_ROOT"] = str(tmp_root)
    os.environ["EMBER_NODE_ID"] = "45"
    spec = importlib.util.spec_from_file_location("comms_loop_under_test", _LOOP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comms_loop_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Arts:
    """The artifact face, with a persist that can be made to fail."""

    def __init__(self, rows, fail_put=False):
        self._rows = rows
        self.fail_put = fail_put
        self.puts = []

    def list_artifacts(self, content_type=None, limit=None):
        return list(self._rows)

    def put_artifact(self, a):
        if self.fail_put:
            raise RuntimeError("store refused the write")
        self.puts.append(dict(a))
        return a


class _Store:
    def __init__(self, arts):
        self.artifacts = arts


def _outbound(mod, mid="m1", to="71"):
    return {"id": "msg.45.%s" % mid, "content_type": mod.MESSAGE_CT, "state": "outbound",
            "msg_id": mid, "from": "45", "to": to, "ts": "20260826T120000Z",
            "kind": "note", "subject": "s", "content": "b", "ref": None}


def test_a_successful_send_is_counted_and_written(tmp_path):
    mod = _load(tmp_path)
    arts = _Arts([_outbound(mod)])
    r = mod.tick(_Store(arts))
    assert r["published"] == 1, r
    assert arts.puts and arts.puts[0]["state"] == "sent"
    written = os.listdir(os.path.join(str(tmp_path), "to-71"))
    assert len(written) == 1 and written[0].endswith(".json"), written


def test_a_send_whose_state_does_not_persist_is_NOT_counted(tmp_path, capsys):
    """THE DEFECT, ASSERTED DIRECTLY. It used to return `published: 1` here."""
    mod = _load(tmp_path)
    arts = _Arts([_outbound(mod)], fail_put=True)
    r = mod.tick(_Store(arts))
    assert r["published"] == 0, (
        "the tick counted a send whose 'sent' state never persisted — `published` must report what "
        "was durably sent, or a courier republishing forever reports healthy ticks: %r" % r)


def test_the_failure_is_audible(tmp_path, capsys):
    """The swallow is what made this invisible. A store refusing the write must reach stderr."""
    mod = _load(tmp_path)
    mod.tick(_Store(_Arts([_outbound(mod)], fail_put=True)))
    err = capsys.readouterr().err
    assert "comms tick WARN" in err, err
    assert "did not persist" in err, err
    assert "msg.45.m1" in err, "the warning does not name the message: %r" % err


def test_the_file_is_still_on_the_plane_after_a_failed_persist(tmp_path):
    """The write happens FIRST and is not rolled back — deliberately. The message really is on
    the plane; what failed is this node's record of having sent it. Republishing is therefore the
    correct behaviour, not a bug to be suppressed."""
    mod = _load(tmp_path)
    mod.tick(_Store(_Arts([_outbound(mod)], fail_put=True)))
    assert os.listdir(os.path.join(str(tmp_path), "to-71")), (
        "the wire file was not written, so the failure is not the one this test is about")


def test_a_republish_overwrites_rather_than_duplicating(tmp_path):
    """WHY NO DEDUPE WAS ADDED. Two ticks with a failing persist leave ONE file, because
    `wire_filename` is deterministic per message id."""
    mod = _load(tmp_path)
    arts = _Arts([_outbound(mod)], fail_put=True)
    for _ in range(3):
        mod.tick(_Store(arts))
    files = os.listdir(os.path.join(str(tmp_path), "to-71"))
    assert len(files) == 1, (
        "three republishes produced %d files; the wire is no longer keyed on the message id, so a "
        "retry now duplicates instead of overwriting: %r" % (len(files), files))


def test_ingest_skips_an_id_it_has_already_seen(tmp_path):
    """The receiving half of the same idempotence, asserted so nobody 'fixes' the sender for a
    duplicate the receiver already refuses."""
    mod = _load(tmp_path)
    inbox = os.path.join(str(tmp_path), "to-45")
    os.makedirs(inbox, exist_ok=True)
    wire = {"id": "w1", "from": "71", "to": "45", "ts": "20260826T120000Z",
            "kind": "note", "subject": "s", "body": "b", "ref": None}
    with open(os.path.join(inbox, mod.wire_filename(wire)), "w", encoding="utf-8") as f:
        json.dump(wire, f)

    arts = _Arts([])
    first = mod.tick(_Store(arts))
    second = mod.tick(_Store(arts))
    assert first["ingested"] == 1, first
    assert second["ingested"] == 0, (
        "the same wire file was ingested twice — the `seen-<node>/<id>` marker is not deduping, "
        "which is the property the sender's republish relies on: %r" % second)


def test_the_marker_is_written_only_after_a_successful_ingest(tmp_path):
    """Order matters on this side too: a marker written first would drop a message whose
    `put_artifact` failed, silently and permanently."""
    mod = _load(tmp_path)
    inbox = os.path.join(str(tmp_path), "to-45")
    os.makedirs(inbox, exist_ok=True)
    wire = {"id": "w2", "from": "71", "to": "45", "ts": "20260826T120000Z",
            "kind": "note", "subject": "s", "body": "b", "ref": None}
    with open(os.path.join(inbox, mod.wire_filename(wire)), "w", encoding="utf-8") as f:
        json.dump(wire, f)

    failing = _Arts([], fail_put=True)
    assert mod.tick(_Store(failing))["ingested"] == 0
    # The retry must still be possible: no marker was left behind.
    ok = _Arts([])
    assert mod.tick(_Store(ok))["ingested"] == 1, (
        "a failed ingest left a `seen` marker, so the message can never be retried and is lost")
