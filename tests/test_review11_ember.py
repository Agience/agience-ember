"""Review 11, Ember.

* Ember shares nothing (John, 2026-11-02): sharing happens only in the person's own session. Ember
  keeps remember (a Create to the drop key) and the read check (a verdict).
* Ember's own credential comes from Origin already narrowed: its refresh answer is an Ember
  delegation, and Ember exchanges nothing to narrow itself (M5).
* The refresh answer is capped while it streams.
* The store refusal sees through `\\\\?\\` paths, UNC `localhost` shares, hard links and the
  content folder, and the opened store is checked through its handle.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

from _fakes import _FakeStore


def test_share_is_refused_in_every_mode(monkeypatch):
    from ember import genesis
    for configured in (True, False):
        if configured:
            monkeypatch.setenv("EMBER_MANTLE_URI", "http://mantle.test")
        else:
            monkeypatch.delenv("EMBER_MANTLE_URI", raising=False)
        r = genesis.share(_FakeStore(), "a-1", to_principal="bob", principal="alice",
                          confirm=True)
        assert r["shared"] is False and "own session" in r["reason"]


def test_op_share_over_ember_is_refused(monkeypatch):
    from ember import genesis
    monkeypatch.setenv("EMBER_MANTLE_URI", "http://mantle.test")
    r = genesis.invoke(_FakeStore(), "op.share", {"artifact_id": "a-1", "to_principal": "bob",
                                                   "confirm": True})
    assert r["result"]["shared"] is False and "own session" in r["result"]["reason"]


def test_ember_s_mantle_client_makes_no_grant():
    from ember import mantle_api
    assert not hasattr(mantle_api.MantleAPI, "grant_read")
    assert "ember:share" not in json.dumps(getattr(mantle_api, "EMBER_SCOPES", ""))


def _keys(tmp_path, monkeypatch):
    keys = tmp_path / "keys"
    keys.mkdir()
    monkeypatch.setenv("KEYS_DIR", str(keys))
    monkeypatch.setenv("EMBER_ORIGIN_URI", "http://origin.test")
    monkeypatch.setenv("EMBER_MANTLE_REFRESH_TOKEN", "seed-rt")
    monkeypatch.delenv("EMBER_MANTLE_TOKEN", raising=False)
    return keys


def test_ember_s_own_credential_is_origin_s_narrowed_delegation_with_no_self_exchange(
        tmp_path, monkeypatch):
    from ember import mantle_api
    _keys(tmp_path, monkeypatch)
    seen = []

    def origin(req: httpx.Request) -> httpx.Response:
        form = dict(httpx.QueryParams(req.content.decode()))
        seen.append(form.get("grant_type"))
        return httpx.Response(200, json={"access_token": "ember-delegation", "expires_in": 300,
                                         "refresh_token": "rt-2"})

    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(origin))
    assert mantle_api.EmberCredential.from_env()() == "ember-delegation"
    assert seen == ["refresh_token"], "no token exchange: Origin narrows Ember's credential itself"


def test_the_refresh_answer_is_capped_while_it_streams(tmp_path, monkeypatch):
    from ember import mantle_api
    _keys(tmp_path, monkeypatch)
    pulled = {"n": 0}

    class Big(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(64):
                pulled["n"] += 1024
                yield b" " * 1024

    def origin(req):
        return httpx.Response(200, stream=Big())

    monkeypatch.setattr(mantle_api, "_transport", lambda: httpx.MockTransport(origin))
    monkeypatch.setattr(mantle_api, "MAX_RESPONSE", 2048)
    with pytest.raises(mantle_api.MantleRefused):
        mantle_api.EmberCredential.from_env()()
    assert pulled["n"] <= 4096, f"pulled {pulled['n']} bytes before refusing"


# -- the store refusal ----------------------------------------------------------------------------

@pytest.fixture
def mantle_dir(tmp_path, monkeypatch):
    fake = tmp_path / "FakeMantleData"
    (fake / "keys").mkdir(parents=True)
    (fake / "lattice.db").write_bytes(b"")
    for k in ("EMBER_FORBIDDEN_DIRS", "MANTLE_DATA_DIR", "MANTLE_KEYS_DIR", "MANTLE_LATTICE_PATH",
              "EMBER_SQLITE_DIR", "EMBER_SQLITE_DB", "KEYS_DIR", "EMBER_STORE_KEYS_DIR",
              "EMBER_MANTLE_URI", "AGIENCE_PLANE", "EMBER_CONTENT_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EMBER_FORBIDDEN_DIRS", str(fake))
    return fake


def _refused(monkeypatch, **env):
    from ember import mantle_api
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    try:
        mantle_api.refuse_mantle_store()
        return False
    except RuntimeError:
        return True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path forms")
def test_a_long_path_prefix_does_not_hide_mantle_s_folder(mantle_dir, monkeypatch):
    assert _refused(monkeypatch, EMBER_SQLITE_DIR="\\\\?\\" + str(mantle_dir))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path forms")
def test_a_long_path_prefix_in_the_forbidden_list_still_refuses(mantle_dir, monkeypatch):
    monkeypatch.setenv("EMBER_FORBIDDEN_DIRS", "\\\\?\\" + str(mantle_dir))
    assert _refused(monkeypatch, EMBER_SQLITE_DIR=str(mantle_dir))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path forms")
def test_a_unc_localhost_share_does_not_hide_mantle_s_folder(mantle_dir, monkeypatch):
    p = str(mantle_dir)
    unc = "\\\\localhost\\" + p[0].lower() + "$" + p[2:]
    assert _refused(monkeypatch, EMBER_SQLITE_DIR=unc)


def test_a_hard_link_to_mantle_s_lattice_is_refused(mantle_dir, tmp_path, monkeypatch):
    out = tmp_path / "outside"
    out.mkdir()
    link = out / "lattice.db"
    try:
        os.link(str(mantle_dir / "lattice.db"), str(link))
    except OSError:
        pytest.skip("hard links not permitted here")
    assert _refused(monkeypatch, EMBER_SQLITE_DIR=str(out), EMBER_SQLITE_DB="lattice.db")


def test_the_content_folder_inside_mantle_s_is_refused(mantle_dir, tmp_path, monkeypatch):
    own = tmp_path / "ember"
    own.mkdir()
    assert _refused(monkeypatch, EMBER_SQLITE_DIR=str(own),
                    EMBER_CONTENT_DIR=str(mantle_dir / "cas"))


def test_the_opened_store_is_checked_through_its_handle(mantle_dir, tmp_path, monkeypatch):
    """The folder passes the check, then is swapped for Mantle's before the store opens: the store
    is checked again on what was actually opened."""
    from ember import mantle_api
    own = tmp_path / "ember"
    own.mkdir()
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(own))
    mantle_api.refuse_mantle_store()
    opened = Path(mantle_dir / "lattice.db")
    with pytest.raises(RuntimeError):
        mantle_api.refuse_opened_store(str(opened))


def test_a_delegation_from_before_its_person_s_sign_in_bound_is_refused(monkeypatch):
    from prism.trust import not_before

    from ember.surface import serve

    claims = {"principal_type": "delegation", "aud": "ember", "act": {"sub": "ember"},
              "scope": "ember:memory ember:read", "sub": "alice", "iat": 100}
    monkeypatch.setattr("prism.trust.authority_trust.verify_access_token",
                        lambda tok, **k: dict(claims))

    class H:
        headers = {"Authorization": "Bearer t"}

    nb = not_before.NotBefore("http://origin.test")
    monkeypatch.setattr(not_before, "_shared", nb)
    assert serve._origin_subject(H()) == "alice"
    nb.apply({"entries": [{"sub": "alice", "not_before": 200}]})
    assert serve._origin_subject(H()) is None
