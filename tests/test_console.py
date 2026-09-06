"""The local control console: it renders, it reports what is registered, and the endpoint that
causes work on another machine is closed unless a token is set.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    """Every test here starts from a known environment and its own store.

    `open_store()` takes no path argument: it resolves the store from `EMBER_SQLITE_DIR` /
    `EMBER_SQLITE_DB`, so isolation is done by pointing the environment at a per-test directory.
    The facet and token variables are cleared here too, so each test asserts a property of the code
    rather than of whatever the developer's shell happens to export."""
    d = tmp_path / "store"
    (d / "keys").mkdir(parents=True)
    for var in ("EMBER_FACET_ROOT", "EMBER_INVOKE_TOKEN", "EMBER_HOST_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EMBER_SQLITE_DIR", str(d))
    monkeypatch.setenv("EMBER_SQLITE_DB", "s.db")
    monkeypatch.setenv("EMBER_STORE_KEYS_DIR", str(d / "keys"))
    monkeypatch.setenv("EMBER_SQLITE_CREATE", "1")     # a per-test store must be created deliberately


def _store(tmp_path):
    from mantle.shard import local_store
    return local_store.open_store()


def test_state_reports_writes_closed_when_no_token_is_set(tmp_path, monkeypatch):
    """A console that renders a `run` button while `EMBER_INVOKE_TOKEN` is unset invites a click
    that returns 403 with no explanation. The endpoint is closed either way; this pins that the
    page says so, so its appearance matches what it will do."""
    from ember.surface import console

    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    st = console.state(_store(tmp_path), probe=False)
    assert st["writes_enabled"] is False
    assert "UNSET" in st["config"]["EMBER_INVOKE_TOKEN"]


def test_the_token_is_never_echoed(tmp_path, monkeypatch):
    """A console that prints a token is a console that leaks one into a screenshot or a support
    thread. Presence is reportable; the value is not."""
    from ember.surface import console

    monkeypatch.setenv("EMBER_INVOKE_TOKEN", "s3cr3t-do-not-print")
    st = console.state(_store(tmp_path), probe=False)
    assert st["config"]["EMBER_INVOKE_TOKEN"] == "set"
    assert "s3cr3t-do-not-print" not in json.dumps(st)
    assert "s3cr3t-do-not-print" not in console.page(_store(tmp_path), probe=False)


def test_page_renders_with_no_hosts_and_says_so(tmp_path, monkeypatch):
    """An empty console explains the emptiness. 'No hosts' and 'the host list failed to load'
    look identical on a page that just renders nothing."""
    from ember.surface import console

    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    html = console.page(_store(tmp_path), probe=False)
    assert "<title>ember console</title>" in html
    assert "No prism host has registered" in html
    assert "prism init" in html


def test_a_registered_host_and_its_operators_appear(tmp_path, monkeypatch):
    """The whole point of the page: a prism that registered is visible, and so are the operators it
    announced — the two artifacts `register_remote_host` writes."""
    from ember.runtime import capability
    from ember.surface import console

    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    store = _store(tmp_path)
    capability.register_remote_host(
        store, name="gw", operators=["analyze"], endpoint="http://127.0.0.1:9911",
        capabilities=["compute.local"])

    st = console.state(store, probe=False)
    assert len(st["hosts"]) == 1, st["hosts"]
    h = st["hosts"][0]
    assert h["endpoint"] == "http://127.0.0.1:9911"
    assert [o["operator_name"] for o in h["operators"]] == ["analyze"]

    html = console.page(store, probe=False)
    assert "gw" in html and "analyze" in html


def test_shipping_to_an_unreachable_host_reports_it_and_does_not_raise(tmp_path):
    """`ship` reports a delivery failure in its return value rather than raising or swallowing it,
    so a caller can tell an unreachable host from a delivered signal. Port 9 is discard; nothing
    serves it, so delivery fails at the transport without depending on a listener."""
    from ember.runtime import capability
    from ember.signal import signal

    store = _store(tmp_path)
    capability.register_remote_host(
        store, name="gw", operators=["analyze"], endpoint="http://127.0.0.1:9",
        capabilities=["compute.local"])

    target = signal.resolve(capability.remote_operator_id("gw", "analyze"), store=store)
    assert target["kind"] == "remote"
    out = signal.ship({"target": target, "signal": {}}, timeout=1.0)
    assert out["shipped"] is False
    assert "reason" in out and out["reason"]


def test_ship_refuses_a_target_with_no_endpoint(tmp_path):
    """Guessing a URL is not delivery. A host that registered without an endpoint is unreachable,
    and saying so beats inventing `http://localhost`."""
    from ember.signal import signal

    out = signal.ship({"target": {"kind": "remote", "operator": "op.host.x.y",
                                  "operator_name": "y", "endpoint": ""}})
    assert out["shipped"] is False
    assert "endpoint" in out["reason"]


def _facet_root(tmp_path, monkeypatch):
    root = tmp_path / "facets"
    (root / "workbench").mkdir(parents=True)
    (root / "workbench" / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    (root / "workbench" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (root / "unbuilt").mkdir()
    (tmp_path / "SECRET.txt").write_text("do not serve me", encoding="utf-8")
    monkeypatch.setenv("EMBER_FACET_ROOT", str(root))
    return root


def test_unset_facet_root_is_not_the_same_as_an_empty_one(tmp_path, monkeypatch):
    from ember.surface import console
    monkeypatch.delenv("EMBER_FACET_ROOT", raising=False)
    assert console.facet_root() is None
    assert console.facets() == []
    assert "Unset is not the same as empty" in console.page(_store(tmp_path), probe=False)


def test_facets_list_includes_one_that_did_not_build(tmp_path, monkeypatch):
    """A facet with no `index.html` is listed as unserveable rather than hidden. A facet that
    failed to build is exactly the case an operator needs to see, and an empty list cannot
    distinguish it from a facet that was never installed."""
    from ember.surface import console
    _facet_root(tmp_path, monkeypatch)
    got = {f["name"]: f["serveable"] for f in console.facets()}
    assert got == {"workbench": True, "unbuilt": False}


def test_a_facet_file_resolves(tmp_path, monkeypatch):
    from ember.surface import console
    _facet_root(tmp_path, monkeypatch)
    assert console.facet_file("workbench", "").name == "index.html"
    assert console.facet_file("workbench", "app.js").read_text() == "console.log(1)"
    assert console.facet_content_type(console.facet_file("workbench", "app.js")) \
        == "text/javascript; charset=utf-8"


@pytest.mark.parametrize("rel", [
    "../SECRET.txt",                     # the obvious one
    "..%2fSECRET.txt",                   # encoded — the server unquotes before we see it
    "sub/../../SECRET.txt",              # traversal that only appears after normalisation
    "/etc/passwd",                       # absolute: contains no ".." at all
    "C:\\Windows\\win.ini",              # Windows drive letter: also no ".."
])
def test_facet_traversal_is_refused(tmp_path, monkeypatch, rel):
    """Containment is proved on the resolved path rather than guessed from the input string: a
    guard written as `".." not in rel` passes every case below except the first, because an
    absolute path and a drive letter escape the root while containing no `..` at all."""
    from ember.surface import console
    _facet_root(tmp_path, monkeypatch)
    assert console.facet_file("workbench", rel) is None


def test_facet_name_itself_cannot_traverse(tmp_path, monkeypatch):
    """The facet name is attacker-controlled too, so it is contained on the same resolved-path
    check; guarding only the relative path would leave the other half of the join open."""
    from ember.surface import console
    _facet_root(tmp_path, monkeypatch)
    assert console.facet_file("..", "SECRET.txt") is None
    assert console.facet_file("../..", "SECRET.txt") is None


def test_an_unknown_suffix_is_never_served_as_html(tmp_path, monkeypatch):
    """A browser that downloads an unrecognised file is a nuisance; one that executes it because
    the server guessed `text/html` is a vulnerability. The suffix map is closed, and anything
    outside it is served as `application/octet-stream`."""
    from ember.surface import console
    root = _facet_root(tmp_path, monkeypatch)
    odd = root / "workbench" / "payload.weird"
    odd.write_text("<script>alert(1)</script>", encoding="utf-8")
    assert console.facet_content_type(console.facet_file("workbench", "payload.weird")) \
        == "application/octet-stream"


def test_ship_will_not_dial_a_local_target(tmp_path):
    """`send` delivers local signals itself. `ship` is the remote transport, and mixing them would
    make a local invocation depend on a network round trip."""
    from ember.signal import signal

    out = signal.ship({"target": {"kind": "local", "operator": "op.math.add"}})
    assert out["shipped"] is False
    assert "not a remote target" in out["reason"]
