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






