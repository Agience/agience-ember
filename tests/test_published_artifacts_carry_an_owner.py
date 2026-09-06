"""Anything ember writes straight to the store must carry an owner.

`put_artifact` is the direct store path: it writes the vertex and nothing else — no owner, no
self-grant. Mantle's authorization is grant-based, and `services/dependencies.py::check_access`
walks the light cone looking for a grant, so a row with `created_by` NULL has none and the walk
finds nothing — an owner-less row is invisible to every API surface while looking healthy in
SQLite.

Measured on node 71/home, 2026-08-25 — all four true at the same moment:

    SELECT id, ct FROM vertex WHERE id='host.71'   ->  1 row, application/vnd.agience.host+json
    SELECT created_by       FROM vertex ...         ->  NULL
    get_artifact  host.71   over /mcp               ->  404
    list_artifacts by host+json                     ->  []

And the 404 cannot tell you which it is: `check_access` says so in its own words — "Nonexistence
and denial return the same 404 throughout this function — no existence oracle." So the symptom of
an unowned row is identical to the host never having been published, and those have opposite fixes.
It cost a live investigation to tell them apart.

This was never a design decision — this file's own sibling already did it right. In one module,
`register_remote_host` sets `created_by` on the operator rows and on the remote host row; the local
host writer was the only one of the three that did not. That asymmetry is what this test freezes:
it checks every writer, so the next one added is held to what the other three do.

Found because `agience-cloud/scripts/sensor_services.py` could not file a reading into `host.71`
as a container. The sensor was correct and the container was unreachable.
"""
from __future__ import annotations

import os
import pathlib
import re

import pytest

from ember.runtime import capability

_SRC = pathlib.Path(capability.__file__)

#: Fields a directly-written row needs to be reachable. `created_by` is the one that was missing and
#: the only one authorization actually walks; the rest travel with it everywhere else in this file,
#: and a row carrying an owner but no provenance is a different kind of half-written.
_REQUIRED = ("created_by", "created_time", "provenance", "cited_from")


@pytest.mark.parametrize("field", _REQUIRED)
def test_the_local_host_artifact_carries(field: str) -> None:
    """The regression itself: `created_by` absent here is a host nothing can read."""
    doc = capability.host_artifact("71")
    assert doc.get(field), (
        "host_artifact() produced no %r.\n"
        "  It is written with put_artifact, which sets no owner and writes no self-grant, so a row\n"
        "  without this is invisible to every API surface while looking correct in SQLite — and the\n"
        "  404 it produces is indistinguishable from 'never published'. Measured on 71/home,\n"
        "  2026-08-25." % field)


def test_the_owner_defaults_to_the_host_rather_than_to_nothing() -> None:
    """An empty string would pass authorization nowhere, and would be easy to write by accident.

    `principal or host_id` is the same fallback `register_remote_host` uses. A caller that knows the
    node's principal supplies it; one that does not still produces a reachable row.
    """
    assert capability.host_artifact("71")["created_by"] == "71"
    assert capability.host_artifact("71", principal="alice")["created_by"] == "alice"
    # The falsy-principal path is the one a caller hits by passing an unset variable through.
    assert capability.host_artifact("71", principal="")["created_by"] == "71"


def test_publish_threads_the_principal_through() -> None:
    """`publish()` is what callers actually use; an owner it drops is an owner nobody set."""
    captured = {}

    class _Store:
        def put_artifact(self, doc):
            captured.update(doc)

    doc = capability.publish(_Store(), "71", principal="node-71")
    assert doc.get("published") is True, "publish reported failure: %r" % doc.get("publish_error")
    assert captured.get("created_by") == "node-71", (
        "publish() accepted a principal and wrote %r — the argument is not reaching the row"
        % captured.get("created_by"))


def test_every_put_artifact_in_this_module_sets_an_owner() -> None:
    """The asymmetry that caused this, frozen — asserted on the source, over all writers.

    The tests above cover the two writers that are cheap to call. `register_remote_host` needs a
    store, a running genesis and a remote endpoint, and it is precisely the writer that was already
    correct — so the property worth holding is the one about the file: every direct write sets an
    owner. A fourth writer added later is caught here, which is the only place it would be.
    """
    text = _SRC.read_text(encoding="utf-8", errors="replace")

    # Each `put_artifact({...})` call, matched to its closing brace-paren. Comments are stripped
    # first: this module argues about `created_by` in prose at length, and prose is not a write.
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    calls = re.findall(r"put_artifact\(\s*\{.*?\n\s*\}\s*\)", code, re.S)
    # Two, not three: there are three direct writes in this module; only two pass a dict literal
    # and can be read this way. The third is `publish()`'s `artifacts.put_artifact(doc)`, which
    # writes whatever `host_artifact()` built — so it is covered by the four field tests above,
    # not by this regex.
    assert len(calls) == 2, (
        "expected the 2 literal-dict writers in %s, found %d — either a writer was added (add it "
        "here) or they were refactored and this assertion has stopped checking anything. The "
        "non-literal `put_artifact(doc)` in publish() is deliberately not counted; it is covered "
        "by the field tests above." % (_SRC.name, len(calls)))

    missing = [c.strip()[:70] for c in calls if not re.search(r'"created_by"\s*:', c)]
    assert not missing, (
        "%d of %d direct put_artifact writes in %s set no `created_by`:\n  %s\n"
        "  A row without an owner has no grant, and check_access finds nothing to walk — it 404s "
        "on every read while looking correct in SQLite."
        % (len(missing), len(calls), _SRC.name, "\n  ".join(missing)))


# ── the grant, which is the half that makes the row readable ─────────────────────────────────────
#
# `created_by` alone does not reach the allow path: `check_access::_check_grants` calls
# `get_active_grants_for_principal_resource(grantee_id, resource_id)`; `created_by` is never read.
# Measured 2026-08-25: zero grants in the live store named `host.71`, which is why it 404'd while
# carrying genuinely probed content. Fixing the owner field was necessary and not sufficient, and
# these tests exist so that is never re-learned.

# A synthesized keyset, not this box's. Pointing these at a real `<home>/keys` directory would
# pass here and fail in CI and on every other machine — a test that asserts a portable property
# against a local path. `instance.uuid` is the only file the derivation reads, so one line of it
# is a complete keyset for this purpose.
#
# Never name the real path, even to warn against it: a scanner cannot tell prose about a leak
# from the leak itself, and `agience-cloud/deploy/test_no_published_path_is_local.py` treats the
# two the same.
#
# The one test that must use the real keyset is the one asserting this node's principal equals
# what the live store's 3,483 grants are held by; it names `_REAL_KEYSET` and skips when absent.
#
# Taken from the environment, never written down: `KEYS_DIR` is the variable every node already
# exports for this — `service_common.sh` sets it and `write_env_file` passes it to every service —
# so the test finds the real keyset on any node and skips on a box that has none.
_REAL_KEYSET = os.environ.get("KEYS_DIR", "")


@pytest.fixture()
def keyset(tmp_path):
    """A keyset with a fixed instance.uuid, so the derived principal is reproducible."""
    (tmp_path / "instance.uuid").write_text("fd65e098-193b-459d-9a23-37c53cad692b",
                                            encoding="utf-8")
    return str(tmp_path)


class _RecordingStore:
    def __init__(self):
        self.written = []

    def put_artifact(self, doc):
        self.written.append(doc)


def _publish(**kw):
    s = _RecordingStore()
    return s, capability.publish(s, "71", **kw)


def test_publish_writes_both_the_host_and_a_grant(keyset) -> None:
    """Two rows, and the second is the one that took a live investigation to discover was missing."""
    s, pub = _publish(keys_dir=keyset)
    types = [d.get("content_type") for d in s.written]
    assert types == ["application/vnd.agience.host+json",
                     "application/vnd.agience.grant+json"], types
    assert pub.get("published") is True and pub.get("granted") is True


def test_the_grant_names_the_artifact_and_not_the_bare_node_id(keyset) -> None:
    """The grant must name the artifact, not the bare node id.

    `host_artifact` writes the row as `host.<host_id>`, so a grant written with the bare `71`
    would be stored, valid, well-formed — and matched by nothing, since `_check_grants` seeks on
    an exact `resource_id` match. It would fail identically to having no grant, which is exactly
    the defect this whole change exists to remove, reintroduced one layer down.
    """
    s, _ = _publish(keys_dir=keyset)
    host, grant = s.written
    assert host["id"] == "host.71"
    assert grant["resource_id"] == host["id"], (
        "the grant names %r but the artifact is %r — an exact-match seek finds nothing, and the "
        "artifact stays as unreachable as it was before the grant was written"
        % (grant["resource_id"], host["id"]))


def test_the_grantee_is_the_principal_a_node_token_actually_carries() -> None:
    """Measured, not chosen. `uuid5(instance.uuid, "mantle/local-user")` is the `sub` a node-signed
    token carries, and of 3,603 grants in the live store on 2026-08-25, 3,483 (96.7%) are held by
    exactly it. A grant to `"71"` would be well-formed and would never match a token."""
    import pathlib
    import uuid
    if not _REAL_KEYSET:
        pytest.skip("KEYS_DIR is unset — this assertion is about a REAL node's identity")
    kd = pathlib.Path(_REAL_KEYSET)
    if not (kd / "instance.uuid").is_file():
        pytest.skip("no keyset at %s — this assertion is about THIS box's identity" % kd)
    expected = str(uuid.uuid5(uuid.UUID((kd / "instance.uuid").read_text(encoding="utf-8").strip()),
                              "mantle/local-user"))
    assert capability.node_principal_id(_REAL_KEYSET) == expected
    _, pub = _publish(keys_dir=_REAL_KEYSET)
    assert pub["grantee_id"] == expected


@pytest.mark.parametrize("field,value", [
    ("effect", "allow"),      # grant_is_allow reads this; flags without it are not an allow
    ("state", "active"),      # _check_grants skips anything else
    ("can_create", True),     # filing into a container is checked as "create" ON THE CONTAINER
    ("can_read", True),
])
def test_the_grant_carries_what_the_reader_filters_on(field, value, keyset) -> None:
    """Every one of these is a filter in `get_active_grants_for_principal_resource` or the allow
    check beside it. None is decoration, and a grant missing any of them is inert."""
    s, _ = _publish(keys_dir=keyset)
    assert s.written[1].get(field) == value


def test_an_unreadable_keyset_refuses_rather_than_guessing_a_grantee() -> None:
    """The dangerous failure is the plausible one. Falling back to `host_id` would write a grant
    to a principal no token carries — inert at read time, indistinguishable from no grant, and
    harder to find because it looks done. The host still publishes; only the grant refuses."""
    s, pub = _publish(keys_dir="C:/definitely-not-a-keyset")
    assert pub.get("published") is True, "the host must still publish; only the grant refuses"
    assert pub.get("granted") is False
    assert "refusing to guess" in (pub.get("grant_error") or "").lower()
    assert [d.get("content_type") for d in s.written] == ["application/vnd.agience.host+json"], (
        "a grant row was written despite having no derivable grantee")


def test_the_host_publish_reports_an_ungrantable_host_instead_of_swallowing_it() -> None:
    """A silent `except: pass` here is how this stayed invisible for weeks, and the 404 could
    never have surfaced it — `check_access` answers the same 404 for absent and for unreachable,
    so this log line is the only place the difference is ever visible.

    Asserted on `capability.publish_host` rather than on one call site, because there are two:
    `worker.py` on the ingest path and `surface/serve.py` on the serve path, a different process
    that never published at all on its own. Asserting on the shared function covers both paths."""
    src = (pathlib.Path(capability.__file__)).read_text(encoding="utf-8", errors="replace")
    i = src.index("def publish_host(")
    window = src[i:i + 2600]
    assert 'pub.get("granted")' in window, (
        "publish_host never checks `granted` — a host that published but cannot be read would "
        "report as a clean start")
    assert not re.search(r"except Exception:\s*\n\s*(pass|return)\s*$", window, re.M), (
        "the host-publish handler is silent again; a bare swallow here is precisely what hid this")


def test_every_path_that_starts_a_node_publishes_what_it_is() -> None:
    """Publishing must happen wherever a node starts, not only where it ingests.

    `capability.publish` had exactly one call site, `runtime/worker.py`, while `agience up ember`
    runs `python -m ember.cli serve` (`agience-cloud/scripts/service_common.sh:169`) — a different
    process. A node that only serves and never ingests would never publish its host artifact, and
    `agience restart ember` could not repair one. Measured 2026-08-25 by a parallel pass:
    restarting a live service to fix exactly that had no effect, because the restart runs the path
    that does not publish.

    Both entry points are asserted by name. A third one added later will not be caught here, which
    is why the check is on the function being reached rather than on a count of call sites."""
    base = pathlib.Path(capability.__file__).parent.parent
    for rel in ("runtime/worker.py", "surface/serve.py"):
        src = (base / rel).read_text(encoding="utf-8", errors="replace")
        assert "publish_host(" in src, (
            "%s starts a node and never publishes what it is — the defect that left `host.71` "
            "unpublishable by restart" % rel)


def test_the_publish_is_best_effort_and_never_takes_the_node_down() -> None:
    """*"A node that cannot describe itself still runs"* is the standing rule for this step, and it
    has to survive being called on a store that cannot take the write — which is exactly the state a
    serve-only node on a busy shard can be in."""
    class _Hostile:
        def __getattr__(self, _name):
            raise RuntimeError("store is unavailable")

    got = capability.publish_host(_Hostile(), node_id="test")
    assert got.get("granted") is False
    assert "RuntimeError" in str(got.get("error") or ""), got
