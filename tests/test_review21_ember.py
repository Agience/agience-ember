"""Review 21 at Ember.

F3: `EMBER_INVOKE_TOKEN` is the node operator's credential for the node's own operators
(`/v1/invoke`, the console, host registration, advance and ingest). Its holder never chooses whose
scope a request reads or writes: `X-Ember-Principal` and `X-Ember-Delegate` name nobody, and a
person is reached only on Origin's delegation for them.
"""
from __future__ import annotations

import sys
import types

import pytest

from ember.surface import serve
from mantle.shard.curate import COMMONS_PRINCIPAL

_COMMONS = {"person": COMMONS_PRINCIPAL, "id": "d." + COMMONS_PRINCIPAL}


class _H:
    def __init__(self, headers=None):
        self.headers = dict(headers or {})


class _Delegate:
    last = None

    @classmethod
    def get(cls, store, person=None, id=None):
        cls.last = {"person": person, "id": id}
        return "delegate"


@pytest.fixture(autouse=True)
def _stub_delegate(monkeypatch):
    mod = types.ModuleType("ember.runtime.delegate")
    mod.Delegate = _Delegate
    monkeypatch.setitem(sys.modules, "ember.runtime.delegate", mod)
    monkeypatch.setenv("EMBER_INVOKE_TOKEN", "shared")
    _Delegate.last = None


def test_the_shared_token_names_no_delegate():
    serve._resolve_delegate(_H({"X-Ember-Token": "shared", "X-Ember-Principal": "alice@x",
                                "X-Ember-Delegate": "d.alice"}), store=object())
    assert _Delegate.last == _COMMONS


def test_the_shared_token_is_the_operator_on_the_node_s_own_principal():
    caller = serve._verified_caller(_H({"X-Ember-Token": "shared",
                                        "X-Ember-Principal": "alice@x"}))
    assert caller is not None and caller.operator is True
    assert caller.principal is None and caller.token is None


@pytest.mark.parametrize("op", ["op.remember", "op.share"])
def test_an_invoked_operator_acts_for_the_verified_caller_only(op):
    """`/v1/invoke` on the shared token runs as the node's own principal: an argument naming a
    principal is replaced by the verified caller's (None, the node's own)."""
    caller = serve._verified_caller(_H({"X-Ember-Token": "shared"}))
    args = serve._invoke_arguments(op, {"principal": "alice@x", "text": "t"}, caller)
    assert args["principal"] is None and args["text"] == "t"


def test_another_operator_s_arguments_pass_as_given():
    caller = serve._verified_caller(_H({"X-Ember-Token": "shared"}))
    assert serve._invoke_arguments("op.status.universe", {"x": 1}, caller) == {"x": 1}


# F6: Ember calls Mantle on a delegation Origin addressed to Mantle, from its own exchange, never
# on the delegation it was itself sent (`aud` `ember`), which Mantle could replay at Ember.

def _origin_and_mantle(monkeypatch, *, refuse=False):
    import httpx
    from ember import mantle_api
    from test_review10_ember import _token

    seen = []

    def handler(req):
        form = dict(httpx.QueryParams(req.content.decode())) if req.method == "POST" and \
            req.url.path == "/auth/token" else {}
        seen.append((req.url.host, req.url.path, form, req.headers.get("authorization")))
        if req.url.path == "/auth/token":
            if refuse:
                return httpx.Response(403, json={"detail": "no"})
            return httpx.Response(200, json={"access_token": _token(aud="mantle"),
                                             "expires_in": 300})
        return httpx.Response(201, json={"id": "mem-1"})

    monkeypatch.setenv("EMBER_MANTLE_URI", "http://mantle.test")
    monkeypatch.setenv("EMBER_ORIGIN_URI", "http://origin.test")
    monkeypatch.setenv("EMBER_MANTLE_TOKEN", "embers-own")
    monkeypatch.delenv("EMBER_MANTLE_REFRESH_TOKEN", raising=False)
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(handler))
    monkeypatch.setattr(mantle_api, "_EXCHANGED", {})
    return seen, _token


def test_ember_exchanges_the_caller_s_delegation_for_one_addressed_to_mantle(monkeypatch):
    from ember import mantle_api

    seen, _token = _origin_and_mantle(monkeypatch)
    inbound = _token(aud="ember")
    api = mantle_api.MantleAPI.for_caller(inbound)
    api._call("POST", "/artifacts", body={"x": 1}, ok=(201,))
    host, path, form, _auth = seen[0]
    assert (host, path) == ("origin.test", "/auth/token")
    assert form["subject_token"] == inbound and form["audience"] == "mantle"
    assert form["actor_token"] == "embers-own"
    mantle_calls = [s for s in seen if s[0] == "mantle.test"]
    assert mantle_calls and all(inbound not in (a or "") for *_x, a in mantle_calls)


def test_no_exchange_no_mantle_call(monkeypatch):
    from ember import mantle_api

    seen, _token = _origin_and_mantle(monkeypatch, refuse=True)
    assert mantle_api.MantleAPI.for_caller(_token(aud="ember")) is None
    assert not [s for s in seen if s[0] == "mantle.test"]


def test_a_delegation_addressed_to_ember_is_never_sent_to_mantle(monkeypatch):
    from ember import mantle_api

    _seen, _token = _origin_and_mantle(monkeypatch)
    assert mantle_api.MantleAPI.from_env(caller_token=_token(aud="ember")) is None


# Review 22, F5: Ember's own credential is addressed to Ember too (its account's token, narrowed
# to it); on its own reads Ember exchanges it for one addressed to Mantle, never sending it on.

def test_ember_s_own_credential_is_exchanged_before_mantle_sees_it(monkeypatch):
    from ember import mantle_api

    seen, _token = _origin_and_mantle(monkeypatch)
    own = _token(aud="ember")
    monkeypatch.setenv("EMBER_MANTLE_TOKEN", own)
    api = mantle_api.MantleAPI.from_env()
    api._call("POST", "/artifacts", body={"x": 1}, ok=(201,))
    origin_calls = [s for s in seen if s[0] == "origin.test"]
    assert origin_calls and origin_calls[0][2]["subject_token"] == own
    assert all(own not in (a or "") for h, *_x, a in seen if h == "mantle.test")


def test_no_exchange_of_ember_s_own_credential_no_mantle_call(monkeypatch):
    import pytest

    from ember import mantle_api

    seen, _token = _origin_and_mantle(monkeypatch, refuse=True)
    monkeypatch.setenv("EMBER_MANTLE_TOKEN", _token(aud="ember"))
    api = mantle_api.MantleAPI.from_env()
    with pytest.raises(mantle_api.MantleUnavailable):
        api._call("POST", "/artifacts", body={"x": 1}, ok=(201,))
    assert not [s for s in seen if s[0] == "mantle.test"]
