"""The serve port's gates, driven over real HTTP with real tokens.

Three properties, each asserted beside a positive control that differs in the one input under test:

  * identity comes only from an Origin *access* token. Origin signs its refresh, email-verification
    and password-reset tokens with the same key, and a delegation minted for a server is not a
    session here, so each of those resolves to the commons;
  * `/v1/invoke` is closed when `EMBER_INVOKE_TOKEN` is unset, as `/hosts/register` and
    `/api/console/run` are;
  * `/v1/chat/completions` refuses a browser origin not named in `EMBER_ALLOWED_ORIGINS`, and its
    writing control phrases (`remember`, `advance`, `ingest`) need an authenticated caller.

The tokens are minted here with `jose` against a manifest this file writes, so nothing under test
helps build its own input. The store is a stand-in and `genesis.invoke` / `genesis.remember` are
recorders: the property is whether the handler reaches them, not what they do.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt as jose_jwt

ISSUER = "https://platform.test"


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


@pytest.fixture()
def origin(tmp_path, monkeypatch):
    """An `origin` trust anchor in a manifest under KEYS_DIR, and a signer for it."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                            serialization.NoEncryption()).decode()
    n = key.public_key().public_numbers()
    jwk = {"kty": "RSA", "alg": "RS256", "use": "sig", "kid": "origin-1",
           "n": _b64u(n.n.to_bytes((n.n.bit_length() + 7) // 8, "big")),
           "e": _b64u(n.e.to_bytes((n.e.bit_length() + 7) // 8, "big"))}
    (tmp_path / "authority.manifest.json").write_text(json.dumps({
        "artifact_id": "test-authority", "issuer": ISSUER,
        "trust_anchors": {"origin": {"jwks": {"keys": [jwk]}}},
    }))
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))
    from prism.trust import authority_trust
    authority_trust.reset_authority_manifest_for_tests()

    def sign(**claims):
        now = int(time.time())
        body = {"iss": ISSUER, "sub": "alice@example.com", "aud": ISSUER,
                "iat": now, "exp": now + 300}
        body.update(claims)
        return jose_jwt.encode(body, pem, algorithm="RS256", headers={"kid": "origin-1"})

    yield sign
    authority_trust.reset_authority_manifest_for_tests()


class _H:
    def __init__(self, headers=None):
        self.headers = dict(headers or {})


# ── identity: only an access token names a caller ──────────────────────────────────────────────────
def test_an_access_token_names_its_subject(origin):
    from ember.surface import serve
    assert serve._origin_subject(_H({"Authorization": "Bearer " + origin()})) == "alice@example.com"


@pytest.mark.parametrize("claims", [
    {"token_type": "refresh"},
    {"token_type": "email_verify"},
    {"token_type": "pwd_reset"},
    {"aud": "agience-server-aria", "principal_type": "delegation",
     "act": {"sub": "agience-server-aria"}},
    {"aud": "mantle", "principal_type": "delegation", "act": {"sub": "platform-mail"}},
    {"aud": "agience", "principal_type": "server"},
    {"aud": "https://another-authority.test"},
], ids=["refresh", "email_verify", "pwd_reset", "delegation", "system-delegation", "server",
        "other-audience"])
def test_a_token_that_is_not_an_access_token_names_nobody(origin, claims, monkeypatch):
    from ember.surface import serve
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    assert serve._origin_subject(_H({"Authorization": "Bearer " + origin()})), \
        "positive control failed FIRST"
    handler = _H({"Authorization": "Bearer " + origin(**claims)})
    assert serve._origin_subject(handler) is None
    assert serve._verified_caller(handler) is None


@pytest.mark.parametrize("token_type", ["refresh", "email_verify", "pwd_reset"])
def test_a_special_purpose_token_resolves_to_the_commons(origin, monkeypatch, token_type):
    """The same property through `_resolve_delegate`, the seam every request's identity crosses."""
    import sys
    import types
    from ember.surface import serve
    from mantle.shard.curate import COMMONS_PRINCIPAL

    seen = []
    mod = types.ModuleType("ember.runtime.delegate")
    mod.Delegate = type("D", (), {"get": classmethod(
        lambda cls, store, person=None, id=None: seen.append(person))})
    monkeypatch.setitem(sys.modules, "ember.runtime.delegate", mod)
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)

    serve._resolve_delegate(_H({"Authorization": "Bearer " + origin()}), store=object())
    serve._resolve_delegate(_H({"Authorization": "Bearer " + origin(token_type=token_type)}),
                            store=object())
    assert seen == ["alice@example.com", COMMONS_PRINCIPAL]


# ── the HTTP surface ────────────────────────────────────────────────────────────────────────────────
class _Store:
    keys_dir = None
    content = None


@pytest.fixture()
def server(monkeypatch):
    """`serve_openai` over a stand-in store, with genesis's two writers replaced by recorders."""
    from ember import genesis
    from ember.runtime import capability, improve, runner
    from mantle.shard import local_store
    from ember.surface import serve

    calls = {"invoke": [], "remember": []}

    class _Bundle:
        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(local_store, "open_store", lambda *a, **k: _Store())
    monkeypatch.setattr(runner, "attach", lambda *a, **k: None)
    monkeypatch.setattr(runner, "load", lambda *a, **k: _Bundle())
    monkeypatch.setattr(genesis, "_author_ref", lambda *a, **k: "author")
    monkeypatch.setattr(capability, "publish_host", lambda *a, **k: None)
    monkeypatch.setattr(improve, "start", lambda *a, **k: None)
    monkeypatch.setattr(genesis, "invoke", lambda store, op, args=None, **k:
                        calls["invoke"].append(op) or {"result": {"op": op}})
    monkeypatch.setattr(genesis, "remember", lambda store, text, principal=None, **k:
                        calls["remember"].append((text, principal))
                        or {"stored": True, "id": "mem-1", "collection": "private.x"})
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    monkeypatch.delenv("EMBER_ALLOWED_ORIGINS", raising=False)

    httpd = serve.serve_openai(port=0, host="127.0.0.1", ember=object())
    base = "http://127.0.0.1:%d" % httpd.server_address[1]

    def post(path, body, headers=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    try:
        yield post, calls
    finally:
        httpd.shutdown()


def _chat(text):
    return {"model": "agience-ember", "messages": [{"role": "user", "content": text}]}


def test_invoke_is_closed_when_no_token_is_configured(server, monkeypatch):
    post, calls = server
    code, _ = post("/v1/invoke", {"operator": "op.status.universe"})
    assert code == 403 and calls["invoke"] == []

    monkeypatch.setenv("EMBER_INVOKE_TOKEN", "shared")
    code, _ = post("/v1/invoke", {"operator": "op.status.universe"}, {"X-Ember-Token": "wrong"})
    assert code == 401 and calls["invoke"] == []
    code, _ = post("/v1/invoke", {"operator": "op.status.universe"}, {"X-Ember-Token": "shared"})
    assert code == 200 and calls["invoke"] == ["op.status.universe"], "positive control"


def test_chat_refuses_a_browser_origin_it_was_not_told_to_allow(server, monkeypatch):
    post, calls = server
    code, _ = post("/v1/chat/completions", _chat("status"), {"Origin": "https://evil.example"})
    assert code == 403 and calls["invoke"] == []

    monkeypatch.setenv("EMBER_ALLOWED_ORIGINS", "https://app.example, https://evil.example")
    code, _ = post("/v1/chat/completions", _chat("status"), {"Origin": "https://evil.example"})
    assert code == 200, "an allowed origin was refused"
    code, _ = post("/v1/chat/completions", _chat("status"))
    assert code == 200, "a request with no Origin header was refused"


def test_an_unauthenticated_remember_writes_nothing(server, origin, monkeypatch):
    post, calls = server
    code, body = post("/v1/chat/completions", _chat("remember that the vault code is 1234"))
    assert code == 200 and calls["remember"] == []
    assert "Not stored" in body["choices"][0]["message"]["content"]

    # a refresh token is not a caller either
    post("/v1/chat/completions", _chat("remember that x"),
         {"Authorization": "Bearer " + origin(token_type="refresh")})
    assert calls["remember"] == []

    # positive controls: an access token writes into its own subject's scope, not the node's
    post("/v1/chat/completions", _chat("remember that my project ships Friday"),
         {"Authorization": "Bearer " + origin()})
    assert calls["remember"] == [("my project ships Friday", "alice@example.com")]

    monkeypatch.setenv("EMBER_INVOKE_TOKEN", "shared")
    post("/v1/chat/completions", _chat("remember that the node note"), {"X-Ember-Token": "shared"})
    assert calls["remember"][-1] == ("the node note", None)


@pytest.mark.parametrize("phrase", ["advance curriculum", "next", "ingest wiki"])
def test_corpus_writing_phrases_need_the_operator_credential(server, origin, monkeypatch, phrase):
    post, calls = server
    post("/v1/chat/completions", _chat(phrase))
    post("/v1/chat/completions", _chat(phrase), {"Authorization": "Bearer " + origin()})
    assert calls["invoke"] == [], "a corpus operator ran without the operator credential"

    monkeypatch.setenv("EMBER_INVOKE_TOKEN", "shared")
    post("/v1/chat/completions", _chat(phrase), {"X-Ember-Token": "shared"})
    assert len(calls["invoke"]) == 1, "positive control: the operator credential was refused"
