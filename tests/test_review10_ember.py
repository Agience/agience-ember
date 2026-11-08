"""Review 10: Ember acts for a person without ever opening their content.

John, 2026-11-02: under custody Ember does everything except open keys.

* remember: a Create into the person's own memory collection, which Mantle makes at their sign-in.
  Ember names it by its derived id and never creates or reads it, so the memory is sealed to the
  collection's drop key, written without Read.
* share: a Read grant to one named person on one artifact, bounded in time; Ember reads nothing to
  make it. The person's consent is their own `confirm`.
* the read check is a verdict (`GET /grants/my-access`), never a read of content.
* Ember's index never holds a row of Mantle's (R7), so no answer rests on Mantle text.
* Ember's own credential is a narrowed delegation (R13); a revoked family stops it quietly; answers
  are size-capped while they stream; a withheld answer reads like any empty one; the store refusal
  covers Mantle's keys folders and runs on every store open.
"""
from __future__ import annotations

import json
import time
import uuid

import httpx
import pytest

from _fakes import _FakeStore

MANTLE = "http://mantle.test"
ALICE = str(uuid.uuid4())
BOB = str(uuid.uuid4())


def _token(sub=ALICE, aud="mantle"):
    """A person's delegation as Ember holds it for Mantle: Origin's exchange of the one sent to
    Ember (`aud` `mantle`, review 21). Pass `aud="ember"` for the one Ember was sent."""
    from jose import jwt
    return jwt.encode({"sub": sub, "aud": aud, "act": {"sub": "ember"},
                       "principal_type": "delegation", "exp": int(time.time()) + 300}, "x",
                      algorithm="HS256")


class _Mantle:
    def __init__(self):
        self.seen = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content or b"{}") if req.content else {}
        self.seen.append((req.method, req.url.path, body, dict(req.url.params)))
        if req.method == "POST" and req.url.path == "/artifacts":
            return httpx.Response(201, json={"id": "mem-1", **body})
        if req.method == "POST" and req.url.path == "/grants":
            return httpx.Response(201, json={"id": "g-1", **body})
        if req.method == "GET" and req.url.path == "/grants/my-access":
            ok = req.url.params.get("resource_id") == "a-1"
            return httpx.Response(200, json={"allowed": ok})
        return httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def mantle(monkeypatch):
    from ember import mantle_api
    m = _Mantle()
    monkeypatch.setenv("EMBER_MANTLE_URI", MANTLE)
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(m.handler))
    return m


def _api():
    from ember import mantle_api
    return mantle_api.MantleAPI.from_env(caller_token=_token())


# -- remember -------------------------------------------------------------------------------------

def test_remember_writes_into_the_derived_memory_collection_and_creates_none(mantle):
    from ember import genesis
    from mantle.services.artifact_identity import derive_artifact_id

    r = genesis.remember(_FakeStore(), "ships Friday", principal=ALICE, mantle=_api())
    assert r["stored"] and r["collection"] == derive_artifact_id(ALICE, "ember:memory")
    posts = [b for m, p, b, _q in mantle.seen if (m, p) == ("POST", "/artifacts")]
    assert len(posts) == 1 and posts[0]["container_id"] == r["collection"]
    assert "identity" not in posts[0]
    assert [(m, p) for m, p, _b, _q in mantle.seen] == [("POST", "/artifacts")]   # nothing read


def test_a_memory_with_no_collection_yet_says_so(monkeypatch):
    from ember import genesis, mantle_api

    monkeypatch.setenv("EMBER_MANTLE_URI", MANTLE)
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(
        lambda req: httpx.Response(404, json={"detail": "Not Found"})))
    r = genesis.remember(_FakeStore(), "x", principal=ALICE, mantle=_api())
    assert not r["stored"] and "next sign-in" in r["reason"]


# -- share ----------------------------------------------------------------------------------------

def test_share_is_refused_and_asks_mantle_nothing(mantle):
    """Ember shares nothing (review 11); sharing happens only in the person's own session."""
    from ember import genesis

    for kw in ({"to_principal": BOB, "confirm": True}, {"confirm": True, "to_public": True}):
        r = genesis.share(_FakeStore(), "a-1", principal=ALICE, mantle=_api(), **kw)
        assert not r["shared"] and "own session" in r["reason"]
    assert not [p for _m, p, _b, _q in mantle.seen if p == "/grants"]


# -- the read check -------------------------------------------------------------------------------

def test_the_read_check_is_a_verdict_and_never_a_read(mantle):
    from ember import mantle_api

    assert mantle_api.readable_by(["mantle:a-1"], _api()) is True
    assert mantle_api.readable_by(["mantle:a-2"], _api()) is False
    assert all(p == "/grants/my-access" for _m, p, _b, _q in mantle.seen)
    assert {q["action"] for _m, _p, _b, q in mantle.seen} == {"read"}


# -- R7: no Mantle row in Ember's index -----------------------------------------------------------

def test_the_index_never_takes_a_row_of_mantle_s():
    from ember import mantle_api

    with pytest.raises(ValueError, match="never holds"):
        mantle_api.refuse_mantle_rows([("mantle:abc", b"Q3 acquisition", "offer")])
    mantle_api.refuse_mantle_rows([("wn-dog", b"dog", "offer")])


def test_rebuilding_the_index_reads_nothing_of_mantle(monkeypatch):
    import inspect

    from ember.corpus import ingest
    assert "visible_rows" not in inspect.getsource(ingest.rebuild_index_from_store)


# -- R13 ------------------------------------------------------------------------------------------

def test_ember_s_own_credential_is_the_delegation_origin_issues_it(tmp_path, monkeypatch):
    """Origin issues Ember's account only a delegation narrowed to Ember (review 11, M5): Ember
    holds what the refresh grant answers and exchanges nothing to narrow itself."""
    from ember import mantle_api

    calls = []

    def origin(req):
        form = dict(httpx.QueryParams(req.content.decode()))
        calls.append(form)
        return httpx.Response(200, json={"access_token": "ember-delegation",
                                         "refresh_token": "rt-2", "expires_in": 300})

    monkeypatch.setenv("EMBER_ORIGIN_URI", "http://origin.test")
    monkeypatch.setenv("EMBER_MANTLE_REFRESH_TOKEN", "rt-1")
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(origin))
    assert mantle_api.EmberCredential.from_env()() == "ember-delegation"
    assert [c["grant_type"] for c in calls] == ["refresh_token"]


def test_a_revoked_family_stops_ember_s_own_credential_quietly(tmp_path, monkeypatch):
    from ember import mantle_api

    calls = []

    def origin(req):
        calls.append(1)
        return httpx.Response(400, json={"detail": "refresh token reused; family revoked"})

    monkeypatch.setenv("EMBER_ORIGIN_URI", "http://origin.test")
    monkeypatch.setenv("EMBER_MANTLE_REFRESH_TOKEN", "rt-1")
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(origin))
    cred = mantle_api.EmberCredential.from_env()
    for _ in range(3):
        with pytest.raises(mantle_api.MantleRefused):
            cred()
    assert len(calls) == 1, "a refused family is asked once, then not again"


@pytest.mark.parametrize("headers,body", [
    ({"content-length": str(10 * 1024 * 1024)}, b"{}"),         # declared too large
    ({}, b'{"x": "' + b"a" * (5 * 1024 * 1024) + b'"}'),         # streamed too large
], ids=["declared", "streamed"])
def test_answers_are_capped_while_they_stream(monkeypatch, headers, body):
    from ember import mantle_api

    monkeypatch.setenv("EMBER_MANTLE_URI", MANTLE)
    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(
        lambda req: httpx.Response(200, headers=headers, content=body)))
    with pytest.raises(mantle_api.MantleRefused, match="too large"):
        _api().get("a-1")


def test_a_withheld_answer_reads_like_any_empty_one():
    from ember.surface import serve
    assert serve.WITHHELD_TEXT == serve.EMPTY_TEXT


def test_the_store_refusal_covers_mantle_s_keys_folders(tmp_path, monkeypatch):
    from ember import mantle_api

    keys = tmp_path / "brain" / "keys"
    keys.mkdir(parents=True)
    monkeypatch.setenv("EMBER_FORBIDDEN_DIRS", str(tmp_path / "brain"))
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(tmp_path / "ember"))
    monkeypatch.setenv("EMBER_STORE_KEYS_DIR", str(keys))
    with pytest.raises(RuntimeError, match="keys"):
        mantle_api.refuse_mantle_store()
    monkeypatch.setenv("EMBER_STORE_KEYS_DIR", str(tmp_path / "ember-keys"))
    monkeypatch.setenv("KEYS_DIR", str(keys))
    with pytest.raises(RuntimeError, match="keys"):
        mantle_api.refuse_mantle_store()


def test_every_store_open_runs_the_refusal(monkeypatch):
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "ember"
    raw = [str(p.relative_to(src)) for p in src.rglob("*.py")
           if re.search(r"from mantle\.shard\.local_store import open_store\b",
                        p.read_text(encoding="utf-8"))
           and p.name != "mantle_api.py"]
    assert raw == [], f"opens Mantle's open_store without the refusal: {raw}"
