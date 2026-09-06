"""`invoke` dispatches on the surface an operator declares, rather than on a `kind` field.

An operator served by a registered prism carries no `kind`: its row records `dispatch` and
`endpoint` instead, which is what says where it runs. Gating dispatch on `op.get("kind")` answers
`"not invokable"` for such an operator — the same shape as a typo'd operator id, and a different
fact from the truth, which is that the operator is invokable but not on this box.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    d = tmp_path / "store"
    (d / "keys").mkdir(parents=True)
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(d))
    monkeypatch.setenv("EMBER_SQLITE_DB", "s.db")
    monkeypatch.setenv("EMBER_STORE_KEYS_DIR", str(d / "keys"))
    monkeypatch.setenv("EMBER_SQLITE_CREATE", "1")
    monkeypatch.delenv("EMBER_HOST_TOKEN", raising=False)


def _store_with_host(endpoint="http://127.0.0.1:9"):
    """Port 9 is discard: nothing serves it, so delivery fails at the transport and the test does
    not depend on a listener existing."""
    from ember.runtime import capability
    from mantle.shard import local_store
    s = local_store.open_store()
    capability.register_remote_host(
        s, name="gw", operators=["analyze"], endpoint=endpoint,
        capabilities=["compute.local"])
    return s


def test_a_prism_hosted_operator_dispatches_by_signal_not_by_kind():
    """`register_remote_host` records `dispatch` and `endpoint` so the row says which surface
    serves the operator, and `invoke` reads them: the dispatch is `signal` and the resolved target
    is remote."""
    from ember import genesis
    s = _store_with_host()

    r = genesis.invoke(s, "op.host.gw.analyze")

    assert r.get("error") is None, r
    assert r.get("dispatch") == "signal"
    assert (r["result"].get("target") or {}).get("kind") == "remote"


def test_the_operator_carries_no_kind_and_that_is_deliberate():
    """Registration records a routing surface, not executable code: the row carries `dispatch` and
    `endpoint`, and no `kind`/`spec` is minted from a POST. `invoke` runs composition steps
    recursively, which bypasses serve.py's `op.dev.*` HTTP ban, so a `spec` arriving from the wire
    could compose a step onto `op.dev.run_tests` — the operator that shells out."""
    s = _store_with_host()
    op = s.artifacts.get_artifact("op.host.gw.analyze")
    assert op is not None
    assert not op.get("kind") and not op.get("spec")
    assert op.get("dispatch") and op.get("endpoint")


def test_a_failed_delivery_is_reported_not_swallowed():
    """The delivery result carries the failure, so an undelivered signal is distinguishable from a
    delivered one. A transport that hides its own failures makes every unreachable host look like a
    success."""
    from ember import genesis
    s = _store_with_host()

    delivery = genesis.invoke(s, "op.host.gw.analyze")["result"]["delivery"]

    assert delivery["shipped"] is False
    assert delivery.get("reason")          # says why, not just that it failed
    assert "9" in delivery.get("url", "")  # and names where it tried


def test_a_transport_failure_is_not_evidence_against_the_operator():
    """Fitness is evidence about a behaviour. A signal that never reached the host carries no
    information about whether the operator works, so it is not recorded as a refutation."""
    from ember import genesis
    s = _store_with_host()
    genesis.invoke(s, "op.host.gw.analyze")

    op = s.artifacts.get_artifact("op.host.gw.analyze")
    assert not op.get("refuted"), "a host being unreachable was counted against the operator"


def test_an_unknown_operator_is_still_refused():
    """The negative control: an id with no operator behind it still answers as unknown. Widening
    dispatch keeps unknown ids local instead of turning each one into a remote attempt."""
    from ember import genesis
    s = _store_with_host()

    r = genesis.invoke(s, "op.does.not.exist")

    assert r.get("error")
    assert "not invokable" in r["error"] or "no operator" in r["error"]
    assert r.get("dispatch") != "signal"
