"""Serve identity: origin is the IdP, and it fails closed (§13.3, §13.11.1).

Three states:
  * no token          -> the commons principal, the public-commons port
  * valid origin JWT  -> that caller's identity, their own scope
  * forged/expired    -> the commons principal, with none of the caller's claims

Each assertion names the principal that comes out, rather than checking that no `person=` argument
was passed. `Delegate.get` with no `person=` falls through to `EMBER_PRINCIPAL`, a real person on
every deployed node, so an absent argument cannot distinguish "resolved as the commons" from
"resolved as whoever the node runs as".
"""
import sys
import types

import pytest

from ember.surface import serve
from mantle.shard.curate import COMMONS_PRINCIPAL

# What an unauthenticated request resolves to. Both fields matter: the id is pinned from the
# principal so `EMBER_DELEGATE_ID` cannot route the commons onto the process delegate's cache entry.
_COMMONS = {"person": COMMONS_PRINCIPAL, "id": "d." + COMMONS_PRINCIPAL}


class _H:
    """The minimum of an http.server handler that `_resolve_delegate` touches."""
    def __init__(self, headers=None):
        self.headers = dict(headers or {})


class _Delegate:
    """Records how Delegate.get was called, so the resolved identity is visible."""
    last = None

    @classmethod
    def get(cls, store, person=None, id=None):
        cls.last = {"person": person, "id": id}
        return "delegate(%s)" % (person or "process")


@pytest.fixture(autouse=True)
def _stub_delegate(monkeypatch):
    mod = types.ModuleType("ember.runtime.delegate")
    mod.Delegate = _Delegate
    monkeypatch.setitem(sys.modules, "ember.runtime.delegate", mod)
    _Delegate.last = None
    yield


def _stub_verify(monkeypatch, fn):
    """Substitute the verifier `serve.py` itself imports.

    The name is load-bearing. `serve.py` imports the trust floor from `prism.trust`, so the stub
    goes at `sys.modules["prism.trust.authority_trust"]`. `origin.*` re-exports the same module
    object, but `monkeypatch.setitem` replaces a sys.modules key: patching the `origin` key would
    leave the `prism.trust` key pointing at the real verifier, every token would fail to verify,
    and the identity tests would fall through to the commons."""
    mod = types.ModuleType("prism.trust.authority_trust")
    mod.verify_jwt = fn
    monkeypatch.setitem(sys.modules, "prism.trust.authority_trust", mod)


def test_no_token_is_anonymous_not_an_assumed_person(monkeypatch):
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    serve._resolve_delegate(_H(), store=object())
    assert _Delegate.last == _COMMONS                          # the commons, not the node's person


def test_valid_origin_jwt_resolves_to_that_caller(monkeypatch):
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    _stub_verify(monkeypatch, lambda tok, **kw: {"sub": "alice@example.com"}
                 if tok == "good" else (_ for _ in ()).throw(ValueError("bad")))
    serve._resolve_delegate(_H({"Authorization": "Bearer good"}), store=object())
    assert _Delegate.last["person"] == "alice@example.com"      # their scope, not the node's


def test_forged_token_yields_NO_identity_not_the_assumed_one(monkeypatch):
    """The security property: a bad signature authenticates nobody, and resolves to the commons
    rather than inheriting the node's principal."""
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    def _boom(tok, **kw):
        raise ValueError("signature invalid / unknown key / expired")
    _stub_verify(monkeypatch, _boom)
    serve._resolve_delegate(_H({"Authorization": "Bearer forged"}), store=object())
    assert _Delegate.last == _COMMONS                           # the commons, never 'alice'


def test_a_token_without_sub_is_not_an_identity(monkeypatch):
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    _stub_verify(monkeypatch, lambda tok, **kw: {"aud": "ember"})   # verified but subject-less
    serve._resolve_delegate(_H({"Authorization": "Bearer nosub"}), store=object())
    assert _Delegate.last == _COMMONS


def test_origin_is_checked_before_the_shared_secret_header(monkeypatch):
    """A verified IdP identity outranks a header-asserted one: the header path serves internal
    callers, and a real token wins over it."""
    monkeypatch.setenv("EMBER_INVOKE_TOKEN", "shared")
    _stub_verify(monkeypatch, lambda tok, **kw: {"sub": "alice@example.com"})
    serve._resolve_delegate(_H({"Authorization": "Bearer good",
                                "X-Ember-Token": "shared",
                                "X-Ember-Principal": "mallory@evil"}), store=object())
    assert _Delegate.last["person"] == "alice@example.com"


def test_header_identity_still_requires_the_shared_secret(monkeypatch):
    """An unauthenticated header naming a delegate is a spoofing vector, so it carries no identity
    without the matching shared secret."""
    monkeypatch.setenv("EMBER_INVOKE_TOKEN", "shared")
    serve._resolve_delegate(_H({"X-Ember-Principal": "mallory@evil"}), store=object())
    assert _Delegate.last == _COMMONS                           # no matching X-Ember-Token
