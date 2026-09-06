"""The commons reader resolves to the commons, and no thread starts on a name that does not exist.

Both properties are invisible in normal operation. An over-broad delegate grants more
access than intended, so no request is ever denied and nothing reports it; a failure
inside a bare daemon thread dies with the thread. These checks make each observable at
test time.
"""
from __future__ import annotations

import os

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# 1. The commons reader
# ──────────────────────────────────────────────────────────────────────────────
class _Headers:
    def __init__(self, **h):
        self._h = {k.replace("_", "-").lower(): v for k, v in h.items()}

    def get(self, k, default=None):
        return self._h.get(str(k).lower(), default)


class _Handler:
    def __init__(self, **h):
        self.headers = _Headers(**h)


def test_unauthenticated_request_reads_as_the_commons_not_as_a_person(monkeypatch):
    """An unauthenticated request resolves to the commons principal, not to whoever the
    node runs as. `Delegate.get` falls back to `EMBER_PRINCIPAL` when called without
    `person=`, and a node deployed as a person sets that variable — node 71's own
    `run-serve.ps1` sets it to `author@example.com` — so `_resolve_delegate` passes the
    commons principal explicitly.

    The fault this pins grants more access than intended rather than less, so it produces
    no user complaint and no error; the test is the only place it becomes visible."""
    from mantle.shard import curate
    from ember.surface import serve

    # The node is deployed as a person, which is legitimate: `EMBER_PRINCIPAL` names who
    # the node is. Who a reader is comes from the request.
    monkeypatch.setenv("EMBER_PRINCIPAL", "author@example.com")
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)

    captured = {}

    class _FakeDelegate:
        @classmethod
        def get(cls, store, *, person=None, id=None, **kw):
            # Reproduce the real fallback: with no `person=`, `EMBER_PRINCIPAL` decides.
            captured["person"] = (person or os.getenv("EMBER_PRINCIPAL") or "").strip()
            captured["id"] = id
            return cls()

    import ember.runtime.delegate as _d
    monkeypatch.setattr(_d, "Delegate", _FakeDelegate)

    serve._resolve_delegate(_Handler(), store=object())

    assert captured["person"] == curate.COMMONS_PRINCIPAL, (
        f"an unauthenticated request resolved to {captured['person']!r}; it must read as "
        f"the commons ({curate.COMMONS_PRINCIPAL!r}), never as whoever the node runs as"
    )
    assert captured["person"] != "author@example.com"


def test_the_commons_delegate_id_is_pinned_so_the_cache_cannot_hand_back_a_person(monkeypatch):
    """The commons resolution pins the delegate id as well as the person, because the
    delegate cache is keyed on the id. `Delegate.get` derives its id from
    `EMBER_DELEGATE_ID` when none is given, so on a node that sets both env vars a
    `person=` argument alone would return the delegate already built for
    `$EMBER_DELEGATE_ID`, carrying `EMBER_PRINCIPAL` and that person's screen."""
    from mantle.shard import curate
    from ember.surface import serve

    monkeypatch.setenv("EMBER_PRINCIPAL", "author@example.com")
    monkeypatch.setenv("EMBER_DELEGATE_ID", "d.author@example.com")
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)

    captured = {}

    class _FakeDelegate:
        @classmethod
        def get(cls, store, *, person=None, id=None, **kw):
            captured["id"] = (id or os.getenv("EMBER_DELEGATE_ID") or "").strip()
            return cls()

    import ember.runtime.delegate as _d
    monkeypatch.setattr(_d, "Delegate", _FakeDelegate)

    serve._resolve_delegate(_Handler(), store=object())

    assert captured["id"] != "d.author@example.com", (
        "the commons resolved onto the node's own delegate id; the cache is keyed on id, "
        "so this hands back that person's Delegate with `person=` discarded"
    )
    assert curate.COMMONS_PRINCIPAL in captured["id"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. No thread starts on a name that does not exist
# ──────────────────────────────────────────────────────────────────────────────
def test_serve_thread_targets_resolve():
    """Every attribute `serve.py` reaches for on `mesh.sync` resolves.

    A bare `threading.Thread(target=…)` keeps any failure in the target inside the thread:
    an `AttributeError` on a name that does not exist kills the thread on its first tick
    and takes the error with it, leaving the loop's comment as the only description of
    what it does. Asserting the rule for every `sync.<name>(...)` call site catches that
    class of fault here rather than in a silently dead thread."""
    import ast
    import inspect

    from ember.surface import serve
    from mantle.mesh import sync

    src = inspect.getsource(serve)
    tree = ast.parse(src)

    # Every `_sync.<name>(...)` / `sync.<name>(...)` call in serve.py must resolve.
    wanted = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"_sync", "sync"}
    }
    missing = sorted(n for n in wanted if not hasattr(sync, n))
    assert not missing, (
        f"serve.py calls mesh.sync.{missing} which do not exist. If this is inside a "
        f"threading.Thread target the AttributeError dies with the thread and nothing reports it."
    )


def test_digest_loop_is_gone_and_stays_gone():
    """`digest_loop` stays out of `serve.py`'s executable code, as a name and as an
    implementation. The digest is maintained incrementally on write (`sync._MERKLE_LIVE`;
    mantle XORs the leaf inside the same transaction as the row), so a periodic recompute
    is a full corpus scan on a timer — the shape that pinned T5 at ~713% CPU. A cache
    refresh for a value that is always current buys nothing."""
    import inspect

    from ember.surface import serve

    src = inspect.getsource(serve)
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "digest_loop" not in code, (
        "digest_loop is back in executable code. The digest is maintained on write; "
        "a timer that rebuilds it is a periodic full-corpus scan."
    )
