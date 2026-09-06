"""Wiring invariants across serve, mesh, capability and reach.

  1. No module reaches for a `mesh.sync` attribute that does not exist. A missing name is a static
     fact, so it is checked statically, across every module that touches `mesh.sync`.
  2. Every operator id the mesh daemon invokes is dispatchable, checked against an independent
     oracle: the id set parsed out of `genesis.invoke`'s own dispatch chain. An `{"error": …}`
     envelope from a tick reaches the daemon's record and stderr.
  3. `capability.register_remote_host` mints `op.host.<h>.<o>` rows carrying no `kind`/`spec`, so
     `genesis.invoke` does not dispatch them. `signal.resolve`/`signal.send` is the surface that
     does, and the row and the response say so.
  4. `capability.validate` accepts prism's canonical vocabulary and open-family members, because it
     gates the endpoint that receives a signed host manifest.
  5. `reach.py` states only claims that hold: the repo's real licence, packages it really imports,
     and no guarantee that no test enforces.

The commons-principal path in `serve._resolve_delegate` is covered in
`test_serve_commons_and_threads.py`; the forged-token fall-through is covered here because that path
is not exercised there.
"""
from __future__ import annotations

import ast
import inspect
import os
import pathlib

import _paths
import sys
import types

import pytest

import ember

from _fakes import _FakeStore

_EMBER = pathlib.Path(inspect.getsourcefile(ember)).resolve().parent                 # …/src/ember
_SRC = _EMBER.parent                                                                 # …/src
_REPO = _SRC.parent                                                                  # …/agience-ember


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# 0. The commons reader — the one path the sibling suite does not cover
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class _H:
    def __init__(self, headers=None):
        self.headers = dict(headers or {})


class _EnvDelegate:
    """A Delegate stub reproducing `delegate.Delegate.get`: the env fallback and the id-keyed cache
    (`delegate.py:245-255`).

    The fidelity is what lets the tests below fail. A stub that merely records `person=None` shows
    that serve passed nothing, but not what the store does with that nothing; reproducing the
    fallback makes the value it produces visible, and makes the cache collision below reachable."""

    cache: dict = {}

    @classmethod
    def get(cls, store, person=None, id=None, origin=None, host=None, restore=True):
        person = (person or os.getenv("EMBER_PRINCIPAL") or "local").strip()
        did = (id or os.getenv("EMBER_DELEGATE_ID") or ("d." + person)).strip()
        hit = cls.cache.get(did)
        if hit is not None:
            return hit                     # keyed by id, not by person — the real cache's rule
        d = {"person": person, "id": did}
        cls.cache[did] = d
        return d


@pytest.fixture
def env_delegate(monkeypatch):
    mod = types.ModuleType("ember.runtime.delegate")
    mod.Delegate = _EnvDelegate
    mod.LOCAL_PERSON = "local"
    monkeypatch.setitem(sys.modules, "ember.runtime.delegate", mod)
    _EnvDelegate.cache = {}
    yield _EnvDelegate


def test_a_forged_token_falls_through_to_the_commons_not_to_a_person(env_delegate, monkeypatch):
    """Fails closed to the commons, not to the identity the node runs as.

    A rejected token must not fall through to `Delegate.get(store)`, which reads `EMBER_PRINCIPAL`
    and would hand the request the node's own principal."""
    from mantle.shard import curate
    from ember.surface import serve
    monkeypatch.setenv("EMBER_PRINCIPAL", "author@example.com")
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    monkeypatch.delenv("EMBER_DELEGATE_ID", raising=False)
    # Patch the name serve.py imports (`prism.trust.*`) and prove the stub fired. With an inert stub
    # the real verifier also rejects a forged token, so the assertion below would be satisfied for a
    # reason having nothing to do with the stub — it would stay green with the fail-closed path
    # deleted. The `_fired` flag makes the test depend on the thing it claims to test.
    mod = types.ModuleType("prism.trust.authority_trust")
    _fired = []

    def _boom(tok, **kw):
        _fired.append(tok)
        raise ValueError("signature invalid / unknown key / expired")

    mod.verify_jwt = _boom
    monkeypatch.setitem(sys.modules, "prism.trust.authority_trust", mod)

    d = serve._resolve_delegate(_H({"Authorization": "Bearer forged"}), store=object())
    assert _fired, ("the stubbed verifier was never called — serve.py is importing a different "
                    "module than this test patches, so the fail-closed assertion below would be "
                    "proving nothing")
    assert d["person"] == curate.COMMONS_PRINCIPAL == "common@ground"
    assert d["person"] != "author@example.com"


def test_the_commons_is_not_served_from_the_process_delegate_cache(env_delegate, monkeypatch):
    """Passing `person=` alone is not enough; the call must also pass `id=`.

    `Delegate.get` derives the id from `EMBER_DELEGATE_ID` when none is given, and the delegate cache
    is keyed on the id. On a node setting both env vars, a commons request that does not supply its
    own id is handed the cached process delegate — person, screen and all — and the `person=`
    argument is discarded."""
    from ember.surface import serve
    monkeypatch.setenv("EMBER_PRINCIPAL", "author@example.com")
    monkeypatch.setenv("EMBER_DELEGATE_ID", "d.node71")
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    _EnvDelegate.cache["d.node71"] = {"person": "author@example.com", "id": "d.node71"}

    d = serve._resolve_delegate(_H(), store=object())
    assert d["person"] == "common@ground", (
        "the commons request was served the cached PROCESS delegate (%r): the principal argument was "
        "discarded because the delegate id came from EMBER_DELEGATE_ID." % d["person"])


def test_serve_reaches_the_commons_principal_by_import_not_by_a_literal():
    """A typed-in `"common@ground"` in serve.py would be a second home for the name, free to drift
    from `curate.COMMONS_PRINCIPAL` and from `curate.LEGACY_COMMONS` (which keeps an existing store's
    rows attributable). The name is imported, so there is one home for it."""
    src = (_EMBER / "surface" / "serve.py").read_text(encoding="utf-8")
    assert "COMMONS_PRINCIPAL" in src, "serve.py never mentions the commons principal at all"
    assert '"common@ground"' not in src and "'common@ground'" not in src, \
        "serve.py hardcodes the commons principal instead of importing curate.COMMONS_PRINCIPAL"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# 1. No module may reach for a `mesh.sync` attribute that does not exist
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _sync_aliases(tree: ast.AST) -> set:
    """Local names bound to `mantle.mesh.sync` by any import form this package uses."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                if a.name == "sync" and (mod == "" or mod.endswith("mesh")):
                    out.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("mesh.sync") and a.asname:
                    out.add(a.asname)
    return out


def test_every_mesh_sync_attribute_referenced_in_ember_exists():
    """An attribute named on `mesh.sync` but absent from it is a thread that dies on its first tick,
    inside a body with nothing to catch the `AttributeError` — the module reads as a running loop and
    is not one.

    A missing name is a static fact, so it is checked statically, and across every module that
    touches `mesh.sync` rather than any one call site."""
    from mantle.mesh import sync
    scanned, missing = 0, []
    for path in sorted(_EMBER.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _sync_aliases(tree)
        if not aliases:
            continue
        scanned += 1
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in aliases and not hasattr(sync, node.attr)):
                missing.append("%s:%d: mesh.sync has no attribute %r"
                               % (path.name, node.lineno, node.attr))
    assert scanned, "no module in ember imports mesh.sync — this check has stopped measuring anything"
    assert not missing, "\n".join(missing)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# 2. The mesh daemon may only invoke ids that dispatch, and is loud when one does not
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _dispatchable_operator_ids() -> set:
    """The ids `genesis.invoke` dispatches, parsed out of `genesis.py`'s own dispatch chain.

    An independent oracle: a list maintained beside the daemon would drift with it, and `invoke`'s
    own `"invokable"` error field is hand-maintained and incomplete — it omits `op.mesh.reconcile`,
    `op.mesh.manifest`, `op.mesh.export`, `op.mesh.reach`, `op.recognize`, `op.respond`, `op.learn`,
    `op.thought` and `op.act`, all of which dispatch. This reads the `operator_id == "…"` /
    `operator_id in (…)` comparisons and the `SOURCE_INGESTERS` keys."""
    path = _EMBER / "genesis.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) \
                and node.left.id == "operator_id":
            for comp in node.comparators:
                elts = comp.elts if isinstance(comp, (ast.Tuple, ast.List, ast.Set)) else [comp]
                for e in elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        ids.add(e.value)
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "SOURCE_INGESTERS" for t in node.targets):
            for k in getattr(node.value, "keys", []) or []:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    ids.add(k.value)
    assert len(ids) > 15, ("only %d dispatchable ids parsed out of genesis.invoke — the dispatch "
                           "shape changed and this oracle is now blind" % len(ids))
    return ids


def test_every_operator_the_mesh_daemon_invokes_is_dispatchable():
    """An id the daemon invokes on every tick that `genesis.invoke` does not dispatch produces an
    `{"error": …}` envelope the arm discards, so the daemon's stated invariant — ingest-role boxes
    are always ingesting — reads as held on every box where it is false.

    The scan follows the daemon into mantle: the daemon is the store's replication driver, and the
    invariant is ember's to hold because `genesis.invoke` is ember's dispatcher."""
    path = _paths.repo("agience-mantle") / "src" / "mantle" / "mesh" / "daemon.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dispatchable = _dispatchable_operator_ids()
    invoked = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) \
                else getattr(node.func, "id", "")
            if fname in ("invoke", "invoke_loud"):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                            and a.value.startswith("op."):
                        invoked.append((node.lineno, a.value))
    assert invoked, "the mesh daemon invokes no operator at all — this check measures nothing"
    bad = ["daemon.py:%d invokes %r, which genesis.invoke does not dispatch" % (ln, oid)
           for ln, oid in invoked if oid not in dispatchable]
    assert not bad, "\n".join(bad)


def test_the_daemon_reports_an_error_envelope_instead_of_swallowing_it(monkeypatch, capsys):
    """`genesis.invoke` returns its failures; it does not raise them. A `try/except` around the call
    sees nothing, so the envelope has to be read and reported explicitly — on the daemon's own record
    and on stderr."""
    from ember import genesis
    from mantle.mesh import daemon
    # Wire the seam rather than patching the runner. The daemon lives in mantle, which does not
    # import ember, so it asks `runner_hooks.invoke` rather than calling `genesis.invoke`; patching
    # `genesis` intercepts nothing and the daemon would see an unwired hook. Wiring the hook is what
    # `boot.py` does, so this exercises the real path.
    from mantle.system import runner_hooks
    monkeypatch.setattr(runner_hooks, "_INVOKE",
                        lambda store, op, args=None: {"error": "operator %r is not invokable" % op})
    rec: dict = {}
    daemon.invoke_loud(object(), "op.pool.ensure_ingest", {}, rec)
    assert rec.get("invoke_errors"), "an {'error': …} envelope left no trace on the daemon's record"
    assert "op.pool.ensure_ingest" in rec["invoke_errors"][0]
    assert "op.pool.ensure_ingest" in capsys.readouterr().err, "the failure was not logged"


def test_a_successful_invoke_is_not_reported_as_an_error(monkeypatch):
    """The negative control: without it the check above could pass by flagging everything. The
    success envelope carries `operator`, and a composition step's own nested error is that step's,
    not this loop's — the same discrimination `pool.run_task` makes."""
    from ember import genesis
    from mantle.mesh import daemon
    from mantle.system import runner_hooks
    monkeypatch.setattr(runner_hooks, "_INVOKE", lambda store, op, args=None: {
        "operator": op, "result": {"steps": [{"error": "a nested step failed"}]}})
    rec: dict = {}
    out = daemon.invoke_loud(object(), "op.mesh.reconcile", {}, rec)
    assert "invoke_errors" not in rec
    assert out["operator"] == "op.mesh.reconcile"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# 3. A registered remote operator is dispatchable — through the surface that serves it
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def test_a_registered_remote_operator_resolves_and_carries_its_transport_target():
    """The registration path produces something dispatchable. `genesis.invoke` is not that surface:
    the row carries no `kind`, because a prism host sends a name, so there is nothing to
    content-address, verify or execute, and minting an executable spec from a POST would let a
    registration compose a step onto `op.dev.run_tests`.

    The surface that serves it is `signal.resolve` / `signal.send`: a remote invocation is a sealed
    signal toward the host endpoint rather than an RPC. This exercises that path end to end, so a
    change to the id shape or the `remote` flag shows up as `kind: unknown` with the endpoint
    lost."""
    from ember.runtime import capability as cap
    from ember.signal import signal
    s = _FakeStore()
    r = cap.register_remote_host(s, name="gw", operators=["analyze"],
                                 endpoint="http://10.0.0.7:8083")
    oid = r["operators"][0]
    assert oid == "op.host.gw.analyze"
    target = signal.resolve(oid, store=s)
    assert target["kind"] == "remote"
    assert target["endpoint"] == "http://10.0.0.7:8083"
    assert target["operator_name"] == "analyze"


def test_the_op_prefix_is_load_bearing_for_the_remote_surface():
    """The negative control for the test above, and the reason these ids stay in the `op.` namespace
    even though `genesis.invoke` does not dispatch them. `signal.resolve` keys the remote branch on
    `address.startswith("op.")`; an id outside that namespace resolves as a bare observer, so the
    host and endpoint are never read. Moving these ids out of `op.` would break the only surface that
    dispatches them."""
    from ember.signal import signal
    s = _FakeStore()
    s.artifacts.put_artifact({"id": "host.gw.op.analyze", "remote": True, "host": "host.gw",
                              "endpoint": "http://10.0.0.7:8083", "operator_name": "analyze",
                              "state": "committed", "content": ""})
    assert signal.resolve("host.gw.op.analyze", store=s)["kind"] != "remote"


def test_the_row_and_the_response_say_that_invoke_is_not_its_surface():
    """The row carries no `kind`/`spec`, so `genesis.invoke` answers "is not invokable" — the same
    answer a typo'd id gets. The row and the registration response therefore name the surface that
    does serve it, so a reader who finds `op.host.gw.analyze` in the operator listing can learn that
    `invoke` is the wrong door."""
    from ember.runtime import capability as cap
    s = _FakeStore()
    r = cap.register_remote_host(s, name="gw", operators=["analyze"])
    doc = s.artifacts.get_artifact("op.host.gw.analyze")
    assert doc["invokable_locally"] is False
    assert doc["dispatch"] == cap.REMOTE_DISPATCH == "signal"
    assert "kind" not in doc and "spec" not in doc      # deliberately: nothing to execute
    assert r["invokable_locally"] is False and r["dispatch"] == "signal"
    assert "signal" in r["note"] and "invokable" in r["note"]


def test_a_registration_is_logged(capsys):
    """Both prism legs wrap this POST in a bare `except` logged "non-fatal", so this log line is the
    only place a host learns anything about its registration — including that its operators are not
    locally invokable."""
    from ember.runtime import capability as cap
    cap.register_remote_host(_FakeStore(), name="gw", operators=["analyze"])
    err = capsys.readouterr().err
    assert "op.host.gw.analyze" in err and "invokable" in err


def test_a_host_still_cannot_shadow_a_local_operator():
    """An announced `op.dev.run_tests` lands under `op.host.<name>.`, so it never becomes the local
    operator that shells out to pytest."""
    from ember.runtime import capability as cap
    s = _FakeStore()
    r = cap.register_remote_host(s, name="evil",
                                 operators=["op.dev.run_tests", "../../dev.run_tests"])
    for oid in r["operators"]:
        assert oid.startswith("op.host.evil.") and ".." not in oid
    assert s.artifacts.get_artifact("op.dev.run_tests") is None


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# 4. Prism is the canonical capability vocabulary, and open families work
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def test_the_canonical_vocabulary_comes_from_prism_not_from_ember():
    """One home. `prism.capabilities` owns the names; `crystal.operator_schema` re-exports them and is
    how ember reaches them (that module's own stated rule)."""
    from prism.capabilities import CAPABILITY_KINDS
    from ember.runtime import capability as cap
    assert cap.CANONICAL_KINDS == CAPABILITY_KINDS
    assert cap.OPEN_FAMILIES == ("sensor.", "actuator.")


@pytest.mark.parametrize("name", ["fs.read", "fs.write", "storage.kv", "net.get", "net.request",
                                  "compute.local", "compute.wasm", "store.read", "store.write",
                                  "ui.render", "sensor.capture", "actuator.control"])
def test_a_signed_prism_manifest_capability_is_accepted(name):
    """Each of these is canonical in prism, and `validate()` gates `POST /hosts/register` — the one
    endpoint in the workspace that receives a host manifest. A prism-c host that signs a manifest
    advertising `fs.read` is accepted by the place that receives it."""
    from ember.runtime import capability as cap
    assert cap.validate([name]) == [name]


@pytest.mark.parametrize("name", ["sensor.temperature", "sensor.lidar", "actuator.relay"])
def test_an_open_family_member_is_accepted_without_a_vocabulary_release(name):
    """Physical devices are unbounded, so `sensor.` and `actuator.` are open families accepted by
    prefix. A closed membership list would admit no real device, which is the plug-and-play case."""
    from ember.runtime import capability as cap
    assert cap.validate([name]) == [name]


def test_a_bare_family_prefix_is_not_a_capability():
    """The negative control for the check above — without it, `validate` could pass by accepting
    anything starting with `sensor.`. A bare prefix names no device."""
    from ember.runtime import capability as cap
    for bad in ("sensor.", "actuator.", "sensor", "actuator", "telepathy", "gpu"):
        with pytest.raises(cap.UnknownCapability):
            cap.validate([bad])


def test_an_invented_name_is_still_refused():
    """The vocabulary stays governed: two open families are two prefixes, and everything else is
    still checked against the registry."""
    from ember.runtime import capability as cap
    for bad in ("compute.quantum", "fs.delete", "storage.cloud", "net.post"):
        with pytest.raises(cap.UnknownCapability):
            cap.validate([bad])


def test_the_legacy_local_names_still_resolve_and_the_set_is_closed():
    """Deployments carry `EMBER_HOST_CAPABILITIES=net.egress` and operators are signed with
    `requires: ["compute.cpu"]`, so accepting prism's vocabulary keeps those working. The legacy set
    is closed and is exactly what `agience-cloud/deploy/capability_drift.py` pins as ember-only, so
    a sixth non-canonical probe name would be a third vocabulary and shows up here."""
    from ember.runtime import capability as cap
    assert cap.validate(sorted(cap.LEGACY_LOCAL_NAMES)) == sorted(cap.LEGACY_LOCAL_NAMES)
    assert set(cap.CAPABILITIES) - set(cap.CANONICAL_KINDS) == set(cap.LEGACY_LOCAL_NAMES)


def test_a_registration_accepts_a_prism_capability_and_types_it():
    """End to end through the endpoint's own entry point. A source pushes and an effector pulls, and
    that asymmetry survives the wire for a name this host cannot probe: the kind is carried for
    canonical names as well as local ones, and the declaration is labelled as declared."""
    from ember.runtime import capability as cap
    s = _FakeStore()
    r = cap.register_remote_host(s, name="gw", operators=[],
                                 capabilities=["fs.read", "sensor.temperature"])
    assert r["declared_capabilities"] == ["fs.read", "sensor.temperature"]
    host = s.artifacts.get_artifact("host.gw")
    assert host["capabilities"]["sensor.temperature"]["kind"] == cap.SOURCE
    assert host["capabilities"]["fs.read"]["kind"] == cap.EFFECTOR
    assert host["capabilities"]["fs.read"]["source"] == "declared"      # never read as measured
    assert "fs.read" in host["offers"] and "sensor.temperature" in host["offers"]
    assert "ui.render" in host["unknown"], "a canonical name the host did not declare is UNKNOWN"


def test_a_declared_open_family_capability_reaches_offers(monkeypatch):
    """A declared open-family name travels the whole way: `_declared()` keeps it, `probe_all` carries
    it beyond the local registry, and it reaches `offers()`. A name that is announced and then
    invisible tells the operator nothing about why."""
    from ember.runtime import capability as cap
    monkeypatch.setenv("EMBER_HOST_CAPABILITIES", "sensor.temperature,fs.read=false")
    p = cap.probe_all()
    assert p["sensor.temperature"]["present"] is True
    assert p["sensor.temperature"]["kind"] == cap.SOURCE
    assert p["sensor.temperature"]["source"] == "declared"
    assert "sensor.temperature" in cap.offers(p)
    assert p["fs.read"]["present"] is False and "fs.read" not in cap.offers(p)


def test_an_invalid_declaration_is_dropped_and_does_not_break_boot(monkeypatch):
    """This reads an env var at boot. A typo leaves the node bootable, and it does not become a
    capability."""
    from ember.runtime import capability as cap
    monkeypatch.setenv("EMBER_HOST_CAPABILITIES", "compute.quantum,fs.read")
    p = cap.probe_all()
    assert "compute.quantum" not in p
    assert p["fs.read"]["present"] is True


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# 5. `reach.py` states only what holds
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def test_ember_is_agpl_so_reach_may_not_claim_an_apache_boundary():
    """`agience-ember/LICENSE` is the GNU Affero GPL v3.0, so a licensing rationale naming a
    permissive licence would send the next reader to "fix" the boundary against a file that says the
    opposite. The separation `reach.py` draws is architectural — runner versus persona layer."""
    licence = (_REPO / "LICENSE").read_text(encoding="utf-8", errors="replace")
    assert "Affero" in licence, "ember's LICENSE is no longer AGPL — re-read this whole test"
    src = (_EMBER / "runtime" / "reach.py").read_text(encoding="utf-8")
    assert "Apache" not in src, \
        "reach.py names a permissive licence in an AGPL repo — that is the false rationale, back"
    # …and the positive control, so this cannot pass by the header going silent on licensing: the
    # licence it states has to be the repo's own.
    assert "AGPL" in src, "reach.py dropped the false claim without stating the true licence"


def test_reach_names_the_package_it_actually_imports():
    """Every path a docstring cites is openable. No `comms` subpackage exists in ember or in prism,
    while the import line reads `from prism.reach import GROUND, Reactor`.

    The `import ember` below is load-bearing twice over: it resolves the package whose directory is
    checked, and it registers `ember.optics` as this process's default instrument. It reads as an
    unused import and is not one."""
    import ember
    import prism
    ember_src = pathlib.Path(inspect.getsourcefile(ember)).resolve().parent
    prism_src = pathlib.Path(inspect.getsourcefile(prism)).resolve().parent
    assert not (ember_src / "comms").exists(), "ember.comms now exists — re-read this test"
    assert not (prism_src / "comms").exists(), "prism.comms now exists — re-read this test"
    src = (_EMBER / "runtime" / "reach.py").read_text(encoding="utf-8")
    for dead in ("beam.comms", "beam/comms/", "ember.comms", "ember/comms/",
                 "prism.comms", "prism/comms/"):
        assert dead not in src, "reach.py cites the non-existent `%s`" % dead
    assert "from prism.reach import" in src


def test_reach_does_not_claim_an_import_allow_list_that_no_test_enforces():
    """`test_reach_wiring.py` enforces an exclusion — no iris/chorus/lumen/sage, by source scan and
    by `sys.modules` — and computes no allow-list. A header claiming an allow-list would name a
    guarantee that suite does not give, and a reader who believes it stops looking.

    The second half asserts the exclusion that IS enforced, so the property is pinned and not merely
    the absence of a sentence."""
    src = (_EMBER / "runtime" / "reach.py").read_text(encoding="utf-8")
    assert "imports ONLY" not in src
    wiring = (_REPO / "tests" / "test_reach_wiring.py").read_text(encoding="utf-8")
    assert "test_reach_module_imports_nothing_from_chorus_iris_lumen_or_sage" in wiring
    assert "test_importing_ember_reach_pulls_in_no_chorus_module" in wiring


def test_ember_reach_still_imports_no_chorus():
    """ember stays runner-side: it imports no chorus, iris, lumen or sage. Re-checked here so this
    file stands alone."""
    import ember.runtime.reach as er
    src = inspect.getsource(er)
    for banned in ("import iris", "from iris", "import chorus", "from chorus",
                   "import lumen", "from lumen", "import sage", "from sage"):
        assert banned not in src, "ember/reach.py must stay runner-side — found %r" % banned
