"""Peer-apply reaches cognition, on genuinely-new rows only and without moving the cursor.

Peer-apply is `signal.deliver(forward=…)`: an arriving peer signal is written as a row and then
heard. The two constraints are what keep that affordable — delivery fires on the new split rather
than on the applied batch, and a delivery error leaves the caller's `handled` count alone.
"""
from __future__ import annotations

import logging

import pytest

from mantle.mesh import sync


def test_delivery_is_off_by_default():
    """A plain node grounds locally, and the rows still land. Delivery is opt-in because switching
    it on by default fires cognition on every consuming node in the fleet at once."""
    assert sync._DELIVER_PEER_SIGNALS is False


def test_it_fires_on_NEW_docs_only_not_on_reapplies(monkeypatch):
    """Delivery fires on the new split, because `applied` counts upserts rather than new knowledge.
    A converged fleet applies large batches that change nothing — measured, ~723,000 docs across 12
    consume cycles for a net gain of 0 rows — so delivering on the batch would re-fire cognition
    for three quarters of a million no-op re-applies every cycle."""
    seen = {}

    def _fake_deliver(delegate, docs, *, store=None, forward=None):
        seen["docs"] = list(docs)
        seen["forward"] = forward
        return {}

    # Patch the module attribute rather than the `sys.modules` entry. `_deliver_new` does
    # `from .. import signal`, which reads the attribute already bound on the `ember` package, so a
    # `sys.modules` replacement takes effect only while the module is still unimported — which
    # depends on what else the run imported first.
    monkeypatch.setattr(sync, "_DELIVER_PEER_SIGNALS", True)
    monkeypatch.setattr("ember.signal.signal.deliver", _fake_deliver)
    monkeypatch.setattr("ember.runtime.delegate.Delegate.get", staticmethod(lambda store: object()))

    new_only = [{"id": "a"}, {"id": "b"}]
    sync._deliver_new(object(), new_only)

    assert [d["id"] for d in seen["docs"]] == ["a", "b"]
    # `forward` is left unset: the onward persona hop needs a live carrier and reactor, so the
    # parameter is wired when there is one to reach.
    assert seen["forward"] is None


def test_nothing_fires_when_there_are_no_new_docs(monkeypatch):
    """The common case on a converged node: every consumed row is a re-apply."""
    called = []
    monkeypatch.setattr(sync, "_DELIVER_PEER_SIGNALS", True)
    monkeypatch.setattr("ember.signal.signal.deliver", lambda *a, **k: called.append(1))
    for empty in (None, []):
        sync._deliver_new(object(), empty)
    assert called == []


def test_a_delivery_failure_is_LOUD_and_never_fatal(monkeypatch, caplog):
    """A delivery error is logged with its cause and leaves the return value alone.

    The caller's `written < len(batch)` guard is what holds the mesh cursor, so a cognition error
    that changed `handled` would read as a partial write and strand a segment behind a monotone
    marker. Logging is what keeps the error visible: the row is written either way, and without the
    log nothing records that cognition did not run on it."""
    monkeypatch.setattr(sync, "_DELIVER_PEER_SIGNALS", True)

    def _boom(*a, **k):
        raise RuntimeError("switch exploded")

    monkeypatch.setattr("ember.signal.signal.deliver", _boom)
    monkeypatch.setattr("ember.runtime.delegate.Delegate.get", staticmethod(lambda s: object()))

    with caplog.at_level(logging.ERROR):
        assert sync._deliver_new(object(), [{"id": "a"}]) is None   # returns nothing, raises nothing

    # `getMessage()` renders the record with its args — `r.message` is the unformatted template and
    # `r.message % r.args` blows up on a record that carries none.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("peer-apply delivery failed" in m for m in msgs), msgs
    assert any("switch exploded" in m for m in msgs), "the CAUSE must survive, not just the fact"
