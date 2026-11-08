"""Ember reaches Mantle through Mantle's HTTP API, never through Mantle's store (John, 2026-11-02).

Ember keeps its own store, in its own folder, under its own key: its corpus, operators, index and
records. What belongs to a person is Mantle's, and Ember reads and writes it over Mantle's routes:

* a memory (`remember`) is created in the caller's own memory collection in Mantle, on the caller's
  own Origin token (`POST /artifacts`), so it is theirs by their creation grant and Mantle seals it;
* sharing it (`share`) is a grant made on the caller's token (`POST /grants`); Mantle decides
  whether they may make it;
* what Ember reads of Mantle for its index it reads on its own credential (`GET /artifacts/visible`),
  so it sees only what that credential is granted, and under threshold custody Origin decides it
  like any other caller's.

With `EMBER_MANTLE_URI` set, none of these writes a person's data into Ember's own store.
"""
from __future__ import annotations

import json

import httpx
import pytest

from _fakes import _FakeStore

MANTLE = "http://mantle.test"


class _Mantle:
    """Mantle's routes, as far as Ember uses them: records every request."""

    def __init__(self):
        self.seen = []
        self.next = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content or b"{}") if req.content else {}
        self.seen.append((req.method, req.url.path, req.headers.get("authorization"), body,
                          dict(req.url.params)))
        if req.method == "POST" and req.url.path == "/artifacts":
            if body.get("identity"):
                return httpx.Response(201, json={"id": "memory-col", **body})
            self.next += 1
            return httpx.Response(201, json={"id": f"mem-{self.next}", **body})
        if req.method == "POST" and req.url.path == "/grants":
            return httpx.Response(201, json={"id": "g-1", **body})
        if req.method == "GET" and req.url.path == "/artifacts/a-1":
            return httpx.Response(200, json={"id": "a-1", "name": "note", "context": "{}"})
        if req.method == "GET" and req.url.path == "/artifacts/visible":
            return httpx.Response(200, json={"items": [
                {"id": "v-1", "context": json.dumps({"about": "tides"}), "content": "high water",
                 "content_type": "text/plain"}], "total": None, "has_more": False})
        return httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def mantle(monkeypatch):
    from ember import mantle_api
    m = _Mantle()
    monkeypatch.setenv("EMBER_MANTLE_URI", MANTLE)
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(m.handler))
    return m


# A memory goes into the caller's derived memory collection, which Ember never creates, and a share
# is one bounded Read grant to one named person: `test_review10_ember.py` holds both (review 10).


def test_with_mantle_configured_a_memory_needs_the_caller_s_own_token(mantle):
    from ember import genesis

    store = _FakeStore()
    r = genesis.remember(store, "something", principal="alice")
    assert not r["stored"] and "Origin" in r["reason"]
    assert mantle.seen == [] and not list(store.artifacts.list_artifacts())


def test_ember_s_own_credential_is_exchanged_at_origin_and_kept_until_it_expires(monkeypatch):
    from ember import mantle_api

    calls = []

    def origin(req: httpx.Request) -> httpx.Response:
        form = dict(httpx.QueryParams(req.content.decode()))
        if form.get("grant_type") != "refresh_token":
            return httpx.Response(200, json={"access_token": form["subject_token"],
                                             "expires_in": 300})       # narrowed to Ember
        calls.append(form)
        return httpx.Response(200, json={"access_token": f"ember-access-{len(calls)}",
                                         "expires_in": 600})

    monkeypatch.setenv("EMBER_ORIGIN_URI", "http://origin.test")
    monkeypatch.setenv("EMBER_MANTLE_REFRESH_TOKEN", "ember-refresh")
    monkeypatch.delenv("EMBER_MANTLE_TOKEN", raising=False)
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(origin))
    cred = mantle_api.EmberCredential.from_env()
    assert cred() == "ember-access-1" and cred() == "ember-access-1"        # kept
    # Origin's refresh grant authenticates its client (review 11, M1): the refresh names the
    # client its family was issued to, `platform` unless EMBER_ORIGIN_CLIENT_ID says otherwise.
    assert calls == [{"grant_type": "refresh_token", "refresh_token": "ember-refresh",
                      "client_id": "platform"}]


def test_ember_has_no_credential_of_its_own_unless_one_is_set(monkeypatch):
    from ember import mantle_api

    for k in ("EMBER_MANTLE_REFRESH_TOKEN", "EMBER_MANTLE_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert mantle_api.EmberCredential.from_env() is None


def test_the_index_reads_mantle_on_ember_s_own_credential(mantle, monkeypatch):
    from ember import mantle_api

    monkeypatch.setenv("EMBER_MANTLE_TOKEN", "ember-own")
    rows = list(mantle_api.visible_rows(mantle_api.MantleAPI.from_env()))
    assert [tuple(r)[:4] for r in rows] == [("mantle:v-1", b"high water",
                                             json.dumps({"about": "tides"}), "text/plain")]
    assert [(m, p, a) for m, p, a, _b, _q in mantle.seen] == [
        ("GET", "/artifacts/visible", "Bearer ember-own")]


def test_ember_s_store_is_never_mantle_s(monkeypatch, tmp_path):
    """The store Ember opens is its own: it refuses one named as Mantle's lattice."""
    from ember import mantle_api

    monkeypatch.setenv("MANTLE_LATTICE_PATH", str(tmp_path / "brain" / "lattice.db"))
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(tmp_path / "brain"))
    with pytest.raises(RuntimeError, match="Mantle's folder"):
        mantle_api.refuse_mantle_store()
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(tmp_path / "ember"))
    mantle_api.refuse_mantle_store()


def test_a_verified_caller_carries_their_own_token_and_only_they_reach_mantle(monkeypatch):
    from ember.surface import serve

    class H:
        def __init__(self, auth):
            self.headers = {"Authorization": auth} if auth else {}

    from ember import mantle_api

    monkeypatch.setattr(serve, "_origin_subject", lambda h: "alice" if h.headers else None)
    monkeypatch.setenv("EMBER_MANTLE_URI", MANTLE)
    # Ember calls Mantle on Origin's exchange of the caller's delegation, addressed to Mantle
    # (review 21, F6), never on the one it was sent.
    monkeypatch.setattr(mantle_api, "exchange_for_mantle",
                        lambda tok: "alice-for-mantle" if tok == "alice-token" else None)
    caller = serve._verified_caller(H("Bearer alice-token"))
    assert caller.token == "alice-token"
    api = serve._mantle_for(caller)
    assert api is not None and api._token() == "alice-for-mantle"
    assert serve._mantle_for(None) is None
    assert serve._mantle_for(serve._Caller(principal="bob", operator=True)) is None


# -- review 7, F2: Origin's refresh tokens rotate on use ------------------------------------------

def _rotating_origin(calls):
    issued = {"n": 0}

    def origin(req: httpx.Request) -> httpx.Response:
        form = dict(httpx.QueryParams(req.content.decode()))
        if form.get("grant_type") != "refresh_token":
            return httpx.Response(200, json={"access_token": form["subject_token"],
                                             "expires_in": 300})       # narrowed to Ember
        calls.append(form["refresh_token"])
        issued["n"] += 1
        return httpx.Response(200, json={"access_token": f"access-{issued['n']}",
                                         "refresh_token": f"refresh-{issued['n']}",
                                         "expires_in": 0})
    return origin


def test_the_newest_refresh_token_is_kept_in_ember_s_own_keys_and_a_used_one_never_again(
        monkeypatch, tmp_path):
    import os
    import stat

    from ember import mantle_api

    calls = []
    monkeypatch.setenv("EMBER_ORIGIN_URI", "http://origin.test")
    monkeypatch.setenv("EMBER_MANTLE_REFRESH_TOKEN", "seed-refresh")
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))
    monkeypatch.delenv("EMBER_MANTLE_TOKEN", raising=False)
    origin = _rotating_origin(calls)
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(origin))
    cred = mantle_api.EmberCredential.from_env()
    assert cred() == "access-1"
    held = tmp_path / "mantle.refresh"
    assert held.read_text().strip() == "refresh-1"
    if os.name == "posix":
        assert stat.S_IMODE(held.stat().st_mode) == 0o600
    assert cred() == "access-2"                       # expired at once: refreshed again
    assert calls == ["seed-refresh", "refresh-1"]     # each token used once, the newest next
    # A restart reads the kept token first, never the seed it has already spent.
    again = mantle_api.EmberCredential.from_env()
    assert again() == "access-3" and calls[-1] == "refresh-2"


def test_a_refused_refresh_keeps_nothing_and_says_so(monkeypatch, tmp_path):
    from ember import mantle_api

    monkeypatch.setenv("EMBER_ORIGIN_URI", "http://origin.test")
    monkeypatch.setenv("EMBER_MANTLE_REFRESH_TOKEN", "stolen-and-revoked")
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(
        lambda req: httpx.Response(401, json={"detail": "refresh token reused"})))
    with pytest.raises(mantle_api.MantleRefused):
        mantle_api.EmberCredential.from_env()()
    assert not (tmp_path / "mantle.refresh").exists()
