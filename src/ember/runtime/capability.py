"""Host capabilities — what this environment can do, measured and published.

AGENT-HOST-DESIGN.md Phase 5, D8 + D12. `opsign.admit()` checks an operator's `requires` against a
set of host offers; this module produces that set.

## The separation this rests on (D12)

An operator declares what it requires, by name, never by implementation. A host publishes what it
offers. The adapter that makes an offer real is **owned by the host** and installed by whoever
provisioned it, so it is trusted by provisioning rather than remotely, while the operator travels
freely because it can only invoke what the host already published.

    operator ──requires: ["sensor.camera"]──▶  (name, the contract)
    host     ──offers:   ["sensor.camera"]──▶  adapter, local, trusted by provisioning

That indirection is why one declarative operator runs on a watch, a Pi and a GPU box over
completely different code, and why a hardware adapter is **additive** rather than a new interpreter
branch.

## Three-valued presence

*has* / *has not* / *did not probe* stay distinguishable throughout, the same discipline
`prism.envelope` holds to. A probe that cannot determine an answer returns `None`: a reading nobody
took, which is different from a negative one. An unmeasured host read as a definitively-empty one
is this repo's signature defect (`mem_source: "host"` for a number nobody read; `K_signal = 0` for
noise; `ic = 0.0` for absent IC).

`offers()` therefore contains only capabilities probed present. A `None` is reported separately in
`unknown()`, so "this host has no camera" and "nobody asked" read differently.

## The vocabulary is governed, and prism owns it

One shared vocabulary is what keeps an operator portable; per-host names degrade the scheme into
per-host scripting. The governed vocabulary is **`prism.capabilities`**, reached (per the one-home
rule) through `crystal.operator_schema`, and `sensor.*` / `actuator.*` are **open families accepted
by prefix**.

`validate()` checks a name against that vocabulary, which is what `ember/surface/serve.py`'s
`POST /hosts/register` — the one endpoint in the workspace that receives a host manifest — gates
on. A prism-c host signing a manifest that advertises `fs.read`, and a real `sensor.temperature`
device announcing itself, both pass. The measured split between the prism and ember spellings is
published by `agience-cloud/deploy/capability_drift.py`.

`CAPABILITIES` below is this host's **local probe registry** — the capabilities ember can measure
on the box it is running on. Each entry names its **kind**, because source and effector
capabilities are not symmetric:

  * **source** (sensor, radio, mic, clock) — pushes. A reading activates the delegate
    (`activation.activate(d, seeds, source=...)`) rather than returning a value to an operator.
    Needs a salience gate: it can flood you.
  * **effector** (compute, storage, display, actuator) — pulls. An operator invokes it and gets a
    result. Needs an effect contract and a resource bound: it can cost you.

## Probes are safe, fast, and side-effect free

A probe runs at boot and on demand, so it makes no network call, spawns no long process, and writes
nothing. Anything unanswerable under those constraints is `None`, which is a real answer here.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# The canonical capability vocabulary, reached through crystal: `prism.capabilities` is its home,
# `crystal.operator_schema` re-exports it, and "ember reaches it through crystal" is that module's
# own stated rule. Imported at module level with no try/except, so a missing package raises an
# ImportError naming it rather than narrowing an authorization vocabulary in silence.
from crystal.operator_schema import (  # noqa: E402
    CAPABILITY_KINDS as CANONICAL_KINDS,
    OPEN_FAMILIES,
    is_known_capability as _is_canonical,
)

SOURCE = "source"
EFFECTOR = "effector"

HOST_CONTENT_TYPE = "application/vnd.agience.host+json"


@dataclass(frozen=True)
class Capability:
    name: str
    kind: str
    description: str
    probe: Callable[[], Optional[bool]]


# ── probes ───────────────────────────────────────────────────────────────────────────────────────
def _probe_cpu() -> Optional[bool]:
    """True by construction — this code is executing on a CPU."""
    return True


def _probe_clock() -> Optional[bool]:
    """A wall clock exists. Distinct from the delegate's subjective time (`Delegate.tick`), which is
    an observation count rather than a clock (see `ember/signal/forgetting.py`). A capability named
    `clock.wall` is the host's, not the agent's."""
    return True


def _probe_nvidia() -> Optional[bool]:
    """NVIDIA specifically, as distinct from `compute.gpu` in general.

    Reads the driver surface only. Ember carries no trained weights and no model runtime, and a
    probe that imported torch to ask about a GPU would drag one in.

    Returns `False` when a definite negative is available — the driver interface is absent on a
    platform where it would be present — and `None` when the question could not be looked at."""
    try:
        if shutil.which("nvidia-smi"):
            return True
        if sys.platform.startswith("linux"):
            if os.path.exists("/proc/driver/nvidia/version"):
                return True
            # Linux exposes the driver at a known path; its absence IS evidence of absence.
            return False
        if sys.platform == "win32":
            # No nvidia-smi on PATH is suggestive but not conclusive on Windows — the driver can be
            # present with the tool unshipped, so there is no reading to give.
            return None
        return None
    except Exception:
        return None


def _probe_gpu_any() -> Optional[bool]:
    """Any accelerator at all. NVIDIA present ⇒ True; otherwise unknown, because AMD, Intel, Apple
    and the rest need vendor tooling this module does not depend on."""
    return True if _probe_nvidia() is True else None


def _probe_storage() -> Optional[bool]:
    """A writable data volume. Measured through `prism.envelope.disk_free_bytes`, which returns
    `None` when the volume could not be read; that `None` is propagated as-is."""
    try:
        from prism import envelope as resource
        free = resource.disk_free_bytes(os.getenv("EMBER_DATA_PATH", "."))
        if free is None:
            return None
        return free > 0
    except Exception:
        return None


def _probe_egress() -> Optional[bool]:
    """Always `None`: there is no reading to give here.

    Determining egress means making a request, and a capability probe is side-effect free — it runs
    at boot, unprompted, and `boot.py` reaches no network. Dialling out would also cross the
    read-only-external-operators rule.

    Egress is therefore a capability a host declares (via `EMBER_HOST_CAPABILITIES`) rather than one
    this module infers."""
    return None


def _probe_human() -> Optional[bool]:
    """Is a human reachable to answer a question (D13 — a Cuddler questionnaire is a capability an
    adapter provides)? A TTY is weak evidence of an interactive session; anything else is unknown,
    because a host may have a human interface this process cannot see."""
    try:
        return True if sys.stdin is not None and sys.stdin.isatty() else None
    except Exception:
        return None


# ── the local probe registry (the vocabulary is prism's — see the header) ──────────────────────────────────────────────────────────────────────
CAPABILITIES: Dict[str, Capability] = {c.name: c for c in [
    Capability("compute.cpu", EFFECTOR, "general computation on this host", _probe_cpu),
    Capability("compute.gpu", EFFECTOR, "any hardware accelerator", _probe_gpu_any),
    Capability("compute.gpu.nvidia", EFFECTOR, "an NVIDIA accelerator", _probe_nvidia),
    Capability("storage.local", EFFECTOR, "a writable local data volume", _probe_storage),
    Capability("net.egress", EFFECTOR, "outbound network reachability", _probe_egress),
    Capability("clock.wall", SOURCE, "host wall-clock time", _probe_clock),
    Capability("human.ask", SOURCE, "a human who can be asked a question", _probe_human),
]}


# ── adapters: how an offered capability is actually satisfied (D12) ───────────────────────────────
# A capability name is a contract; an adapter is the host-owned code that fulfils it. The adapter is
# local, trusted by provisioning, and stays put — that is the split: an operator declares
# `requires: ["human.ask"]` by name and the host supplies the implementation, so one operator runs
# anywhere the capability is offered, over completely different code.
def _adapter_human_ask(request: dict, *, ask=None):
    """`human.ask` — run a Cuddler Process questionnaire against whatever can answer (D13).

    `request` carries the Process document (`request["process"]`); `ask(question)->answer` is the
    host's answer source (a TTY prompt, an MCP relay, a test stub). A source capability: the human's
    typed answers are an input that can seed the delegate's activation rather than a value returned
    to a remote caller."""
    from ember.runtime import cuddler
    if ask is None:
        raise RuntimeError("human.ask needs an `ask` answer-source; the host has not wired one")
    doc = request.get("process")
    if not isinstance(doc, dict):
        raise ValueError("human.ask request needs a Cuddler Process document under 'process'")
    return cuddler.run_process(doc, ask)


# name -> adapter. Only capabilities the node can fulfil locally appear here; a probed-True
# capability with no adapter is offered for discovery and has no local implementation.
ADAPTERS: Dict[str, Callable] = {
    "human.ask": _adapter_human_ask,
}


def adapter_for(name: str) -> Optional[Callable]:
    """The local adapter that fulfils `name`, or None. `admit()` gates on `offers()`; this is how a
    runtime actually invokes an admitted capability."""
    return ADAPTERS.get(name)


class UnknownCapability(KeyError):
    """A name outside the governed vocabulary. Raised, so a host cannot quietly invent its own names
    and leave its operators portable to nobody."""


# The five non-canonical probe names, pinned. `clock.wall`, `compute.cpu`, `compute.gpu.nvidia`,
# `net.egress` and `storage.local` are ember's own spellings, accepted so that deployments carrying
# `EMBER_HOST_CAPABILITIES=net.egress`, and operators already signed with
# `requires: ["compute.cpu"]`, keep working. A closed set rather than a vocabulary: a new name is
# canonical.
#
# The spellings stay as they are because `agience-cloud/deploy/capability_drift.py` pins the
# measured prism/ember split and asserts on them, so a rename is a coordinated edit across a repo
# this module does not own.
LEGACY_LOCAL_NAMES = frozenset({
    "clock.wall", "compute.cpu", "compute.gpu.nvidia", "net.egress", "storage.local",
})


def is_valid_capability(name: str) -> bool:
    """Is `name` an acceptable capability name here? Canonical (the prism base list), a member of an
    open family (`sensor.*` / `actuator.*` — the plug-and-play case), or a pinned legacy local."""
    n = str(name).strip()
    return bool(n) and (_is_canonical(n) or n in LEGACY_LOCAL_NAMES)


def kind_of(name: str) -> str:
    """SOURCE or EFFECTOR for any acceptable name, including one this host cannot probe.

    A source pushes (a reading activates the delegate; it needs a salience gate); an effector pulls
    (an operator invokes it and gets a result; it needs an effect contract and a resource bound).
    That asymmetry has to survive the wire, so it is answered for canonical and open-family names as
    well as for the local probe registry."""
    n = str(name).strip()
    c = CAPABILITIES.get(n)
    if c is not None:
        return c.kind
    return SOURCE if (n.startswith("sensor.") or n == "human.ask") else EFFECTOR


def describe(name: str) -> str:
    """The local probe's description, the canonical one, or a statement that an open-family member
    has none — a device name appears in no base list, by design."""
    n = str(name).strip()
    c = CAPABILITIES.get(n)
    if c is not None:
        return c.description
    d = CANONICAL_KINDS.get(n)
    if d:
        return d
    fam = next((f for f in OPEN_FAMILIES if n.startswith(f)), "")
    return ("a member of the OPEN %s* family — named by the host, not enumerated in any base list"
            % fam) if fam else ""


def validate(names) -> List[str]:
    """Sorted, de-duplicated capability names. Raises `UnknownCapability` on anything unacceptable.

    The check is the canonical one — `prism.capabilities.is_known_capability`: the base list plus
    the open families by prefix — with ember's five pinned legacy spellings also accepted. That is
    what lets a real device (`sensor.temperature`) and a canonical prism name (`fs.read`) both come
    in through `POST /hosts/register`. A bare family prefix is not a name: `sensor.` identifies no
    device, and `is_known_capability` does not accept it."""
    out = set()
    for n in names or []:
        n = str(n).strip()
        if not n:
            continue
        if not is_valid_capability(n):
            raise UnknownCapability(
                "%r is not a capability name. The vocabulary is PRISM's (%s), plus the OPEN families "
                "%s accepted by prefix (a bare prefix is not a name). Add a base kind in "
                "prism/capabilities.py — never ad-hoc here; an invented name is portable to nobody."
                % (n, ", ".join(sorted(CANONICAL_KINDS)),
                   " ".join("%s*" % f for f in OPEN_FAMILIES)))
        out.add(n)
    return sorted(out)


def _declared() -> Dict[str, bool]:
    """Capabilities the host operator asserts, via `EMBER_HOST_CAPABILITIES`.

    For things a probe may not determine safely — `net.egress` above, or a sensor behind an adapter
    this module knows nothing about. Syntax: `name` or `name=false`, comma-separated. A declaration
    overrides a probe, and `probe_all` records that it did, so an asserted capability stays
    distinguishable from a measured one.

    A declaration is not limited to what this host can probe — the filter is validity — so
    `EMBER_HOST_CAPABILITIES=sensor.temperature` is how a sensor host announces itself. An invalid
    name is dropped rather than raised: this reads an env var at boot, and a typo in it leaves the
    node bootable."""
    out: Dict[str, bool] = {}
    for item in (os.getenv("EMBER_HOST_CAPABILITIES", "") or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, val = item.partition("=")
        name = name.strip()
        if is_valid_capability(name):
            out[name] = (val.strip().lower() not in ("false", "0", "no"))
    return out


def probe_all() -> Dict[str, dict]:
    """Every registered capability, plus anything the host declared:
    `{name: {present, kind, source, description}}`.

    `present` is **True / False / None** and `source` is `"probed"` or `"declared"` — so a caller
    can always tell a measurement from an assertion, and both from silence.

    A declared name outside the local probe registry appears here too, so a host that declares
    `sensor.temperature` reaches `offers()` rather than being accepted by `_declared` and then left
    invisible."""
    declared = _declared()
    out: Dict[str, dict] = {}
    for name, cap in CAPABILITIES.items():
        if name in declared:
            present, src = declared[name], "declared"
        else:
            try:
                present = cap.probe()
            except Exception:
                present = None
            src = "probed"
        out[name] = {"present": present, "kind": cap.kind, "source": src,
                     "description": cap.description}
    for name, present in declared.items():
        if name not in out:
            out[name] = {"present": present, "kind": kind_of(name), "source": "declared",
                         "description": describe(name)}
    return out


def offers(probed: Optional[Dict[str, dict]] = None) -> List[str]:
    """The capability names this host offers — exactly those probed or declared **True**.

    `None` is excluded, which is separate from excluding `False`: an unknown capability is not
    offered, because there is nothing measured to offer, and it is not denied either — see
    `unknown()`. Fed to `opsign.admit(host_offers=...)`, that holds back an operator needing an
    unprobed capability, which is the conservative direction: an operator held back on a host that
    might have run it costs less than one admitted to a host that cannot."""
    p = probed if probed is not None else probe_all()
    return sorted(n for n, v in p.items() if v["present"] is True)


def unknown(probed: Optional[Dict[str, dict]] = None) -> List[str]:
    """Capabilities that were not determined. Reported separately so "this host has no camera" and
    "nobody asked" read differently."""
    p = probed if probed is not None else probe_all()
    return sorted(n for n, v in p.items() if v["present"] is None)


def host_artifact(host_id: str, *, probed: Optional[Dict[str, dict]] = None,
                  data_path: str = ".", principal: str = "") -> dict:
    """The HOST artifact — this environment, published as an artifact like everything else.

    Carries capability presence (three-valued) alongside `resource.snapshot()`'s measured
    quantities, because they answer different questions: *can this host do X* versus *how much of
    it is there*. An operator's admission needs the first; sizing needs the second.

    `principal` names the owner. Defaulted to `host_id` rather than left empty — the same fallback
    `register_remote_host` uses — because a host with no owner is unreachable through the API; see
    the note on `created_by` below.
    """
    p = probed if probed is not None else probe_all()
    # Deferred, matching `register_remote_host`: `ember.genesis` imports back into this package, so
    # the import lives in the function rather than at module scope.
    from ember import genesis
    try:
        from prism import envelope as resource
        limits = resource.snapshot(data_path)
    except Exception as e:
        limits = {"error": "%s: %s" % (type(e).__name__, str(e)[:120])}
    return {
        "id": "host.%s" % host_id,
        "content_type": HOST_CONTENT_TYPE,
        "state": "committed",
        "host": host_id,
        "platform": platform.system().lower() or None,
        "capabilities": p,
        "offers": offers(p),
        "unknown": unknown(p),
        "limits": limits,
        "context": "host %s offering: %s" % (host_id, ", ".join(offers(p)) or "nothing probed"),
        "content": "",
        # Without an owner this row is invisible to every API surface. `put_artifact` is the direct
        # store path: it writes the vertex and nothing else. Mantle's authorization is grant-based
        # and `services/dependencies.py::check_access` walks the light cone looking for one, so a
        # row with `created_by` NULL has no owner, no grant, and nothing for that walk to find. It
        # looks perfectly healthy in SQLite and 404s on every read.
        #
        # Measured on node 71/home, 2026-08-25: `host.71` was in the lattice as exactly one
        # `host+json` row, with `get_artifact host.71` returning 404 and `list_artifacts` by that
        # content type returning `[]` — both true because the row carried no `created_by`.
        #
        # And the 404 cannot tell you that: `check_access` returns the same 404 for "absent" and
        # for "present but you hold no grant" — deliberately, so it is not an existence oracle — so
        # the symptom is indistinguishable from the host never having been published.
        #
        # This matches this file's own sibling: `register_remote_host` below sets `created_by` on
        # both the operator rows and the host row, and `put_artifact` upserts by id, so a publish
        # repairs an existing row in place.
        #
        # This line is necessary but not sufficient, measured rather than assumed: the allow path
        # in `check_access` never reads `created_by` — `_check_grants` calls
        # `get_active_grants_for_principal_resource(grantee_id=auth.user_id, resource_id=...)`, so
        # reachability needs a grant row, and on 2026-08-25 exactly zero grants in the live
        # store named `host.71`. An owner field with no grant beside it makes the row attributable
        # and still unreadable.
        #
        # What remains is a design question, not an omission: existing grants are granted to string
        # principals (`grantee='lumen'`, `'sage'`), while a node-minted token authenticates as a
        # UUID subject derived from `instance.uuid`. So "grant the host to its owner" has to answer
        # which name a node is known by before it can be written, and guessing would mint a grant to
        # a principal that never authenticates.
        "provenance": genesis.P_ASSERTION,   # measured by this box; nothing external verified it
        "cited_from": genesis.CITE_GENESIS,
        "created_by": principal or host_id,
        "created_time": genesis._now(),
    }


#: The label the node's own principal id is derived under, and it is NOT a name this file chose.
#: `mantle/services/peer_signing` and `mantle/scripts/dev_mint_token` both derive
#: `uuid5(instance.uuid, "mantle/local-user")`, and that is the `sub` a node-signed token carries —
#: so a grant written to it is a grant the node's own token can spend. Repeating the string is the
#: coupling; it is written down here rather than imported because ember does not depend on mantle.
_NODE_PRINCIPAL_LABEL = "mantle/local-user"


def node_principal_id(keys_dir: str = "") -> str:
    """`uuid5(instance.uuid, "mantle/local-user")` — who this box IS, to the lattice.

    This is the only name that works, measured rather than chosen. Of 3,603 grants
    in the live store on 2026-08-25, **3,483 (96.7%) are held by exactly this principal**, and it is
    also the `granted_by` on all 3,483. The 38 string-named grantees (`ember-runner`, `local`,
    an email address) are 1.1% and legacy. A grant to `"71"` would be well-formed, would look right,
    and would never match a token — `resolve_auth` returns `user_id = payload["sub"]`, and `sub` is
    this uuid5.

    Returns "" when the keyset cannot be read, and the caller must treat that as a failure rather
    than substituting a placeholder. A grant to the wrong principal fails exactly like no grant, and
    is harder to find.
    """
    import uuid as _uuid
    kd = keys_dir or os.getenv("KEYS_DIR") or ""
    if not kd:
        return ""
    try:
        raw = open(os.path.join(kd, "instance.uuid"), encoding="utf-8").read().strip()
        return str(_uuid.uuid5(_uuid.UUID(raw), _NODE_PRINCIPAL_LABEL))
    except (OSError, ValueError):
        return ""


def host_self_grant(host_id: str, grantee_id: str) -> dict:
    """The grant that makes the host artifact reachable — the row `put_artifact` does not write.

    Without this the host is attributable and still unreadable. `created_by` is never consulted
    by the allow path: `check_access::_check_grants` calls
    `get_active_grants_for_principal_resource(grantee_id=..., resource_id=...)`, which filters on
    `state == "active"`, matching `resource_id`, non-expiry, and then `grant_is_allow` plus the
    per-action flag. Every field below is one of those filters; none is decoration.

    Shaped from a real grant row read out of the live store, not from the entity class — the
    reader is `from_lattice_doc(d, GrantEntity)` over the stored doc, so what matters is what the
    stored docs actually carry.
    """
    from ember import genesis
    import uuid as _uuid
    now = genesis._now()
    return {
        # A fresh id per grant row; the (grantee, resource) pair is what the reader seeks on.
        "id": str(_uuid.uuid4()),
        "_type": "Grant",
        "content_type": "application/vnd.agience.grant+json",
        "resource_id": host_id,
        "grantee_id": grantee_id,
        "grantee_type": "user",
        "granted_by": grantee_id,      # a self-grant: the node grants itself, as every owner row does
        # `effect` and `state` are both load-bearing. `_check_grants` skips anything whose state
        # is not "active", and `grant_is_allow` reads `effect` — a grant with allow-shaped flags and
        # no effect is not an allow, and a deny grant's flags name what it denies.
        "effect": "allow",
        "state": "active",
        "can_read": True, "can_create": True, "can_update": True, "can_add": True,
        "can_delete": True, "can_share": True, "can_admin": True, "can_invoke": True,
        "can_evict": True,
        # `can_create` is the one the sensor needs: filing a reading into this host as a container
        # is checked as "create" on the container (`artifacts_router.py:895`), not as "update".
        "requires_identity": False,
        "requires_nonce": False,
        "claims_count": 0,
        "granted_at": now,
        "created_time": now,
        "modified_time": now,
    }


def publish_host(store, *, node_id: str = "", who: str = "ember") -> dict:
    """Publish this host and say what happened — the one place a node announces what it is.

    This exists because the serve path never published at all: `publish` had exactly one
    call site, `runtime/worker.py`, and `agience up ember` runs `python -m ember.cli serve`
    (`agience-cloud/scripts/service_common.sh:169`) — a different process. So a node that serves
    and never ingests never published what it is, and `agience restart ember` could not repair a
    host artifact. Measured 2026-08-25 by a parallel pass: a live service was restarted and it had
    no effect, because the restart runs the path that does not publish.

    Two callers now share this rather than one copying the other: `worker.py` on the ingest path and
    `surface/serve.py` on the serve path. Announcing is not incidental to publishing — see below —
    so it belongs with it, not beside each call.

    The grant warning is the whole reason this is a function: publishing writes the vertex;
    only the grant makes it readable, and the two fail independently. `host.71` was published
    successfully and unreachable for weeks — `get_artifact` 404, `list_artifacts` empty, one
    healthy-looking row in SQLite.

    And the 404 could never have told anyone: `check_access` returns the same 404 for "absent" and
    for "present but you hold no grant", deliberately, so it is not an existence oracle. **That
    makes this warning the only place the difference is ever visible.** A caller that inlined the
    publish and forgot the warning would reintroduce exactly the silence that hid it.

    Best-effort, and that is the standing rule for this step: a node that cannot describe itself
    still runs. What it must not do is fail silently — a bare `except: pass` around this step is
    what let the underlying defect go unnoticed for as long as it did.
    """
    import os
    import sys
    try:
        node_id = node_id or os.getenv("EMBER_HOST_ID") or os.getenv("EMBER_NODE_ID") or "local"
        pub = publish(store, _slug(node_id))
        print("[%s] host %s offers: %s" % (who, pub.get("host"), pub.get("offers")),
              file=sys.stderr, flush=True)
        if not pub.get("granted"):
            print("[%s] WARNING host %s is published but NOT grant-reachable: %s\n"
                  "               every API read of it will answer 404 while the row looks correct "
                  "in SQLite." % (who, pub.get("host"), pub.get("grant_error") or "unknown"),
                  file=sys.stderr, flush=True)
        return pub
    except Exception as e:  # noqa: BLE001 — the node still runs; it just says why it could not
        print("[%s] host publish failed (node continues): %s: %s"
              % (who, type(e).__name__, str(e)[:200]), file=sys.stderr, flush=True)
        return {"host": None, "granted": False, "error": "%s: %s" % (type(e).__name__, e)}


def publish(store, host_id: str, *, data_path: str = ".", principal: str = "",
            keys_dir: str = "") -> dict:
    """Write the host artifact AND the grant that makes it reachable.

    The grant is not an extra — it is the difference between published and visible. Without it,
    a host row sits in the lattice as a correct-looking row that every API surface answers 404 on,
    with nothing distinguishing that from the row never having been written. See `host_self_grant`
    and `tests/test_published_artifacts_carry_an_owner.py`.

    The host still publishes if the grant fails, and the returned dict says so: `grant_error` is
    set, and `worker.py` prints it. That follows the rule this function already had — "a node that
    cannot describe itself still runs" — while not letting the failure be silent. A node whose
    host is unreachable keeps working and stops being a secret.
    """
    doc = host_artifact(host_id, data_path=data_path, principal=principal)
    artifacts = getattr(store, "artifacts", None) or store
    try:
        artifacts.put_artifact(doc)
    except Exception as e:
        doc["published"] = False
        doc["publish_error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        return doc
    doc["published"] = True

    grantee = node_principal_id(keys_dir)
    if not grantee:
        # Not substituted with `host_id`. A grant to a name no token carries is indistinguishable
        # from no grant at read time, and considerably harder to find — it looks done.
        doc["granted"] = False
        doc["grant_error"] = (
            "no node principal: could not read instance.uuid from KEYS_DIR=%r, so the grantee "
            "cannot be derived. Refusing to guess one." % (keys_dir or os.getenv("KEYS_DIR") or ""))
        return doc
    try:
        # `doc["id"]`, not `host_id` — the two are not the same string: `host_artifact` writes the
        # row as `host.<host_id>`, so a grant whose `resource_id` is the bare `71` would name a
        # resource that does not exist, and `_check_grants` seeks on an exact `resource_id` match —
        # the grant would be written, stored, and matched by nothing. Indistinguishable from no
        # grant at all.
        artifacts.put_artifact(host_self_grant(doc["id"], grantee))
    except Exception as e:
        doc["granted"] = False
        doc["grant_error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        return doc
    doc["granted"] = True
    doc["grantee_id"] = grantee
    return doc


# ── remote hosts announcing themselves ───────────────────────────────────────────────────────────
from ember.runtime.runner import evolution as _evolution   # the single distribution path (ember/runtime/runner.py)
OPERATOR_CONTENT_TYPE = _evolution.OPERATOR_CONTENT_TYPE   # one home for the operator content type

#: The surface that dispatches a registered remote operator: `signal.resolve` / `signal.send`, not
#: `genesis.invoke`. See `register_remote_host`.
REMOTE_DISPATCH = "signal"

_SLUG_OK = set("abcdefghijklmnopqrstuvwxyz0123456789-")


def _slug(s: str) -> str:
    """A safe id component. Anything outside `[a-z0-9-]` becomes `-`, so a host name cannot inject a
    dot and reach a different operator namespace (`op.dev.run_tests`)."""
    out = "".join(c if c in _SLUG_OK else "-" for c in str(s).strip().lower())
    return "-".join(p for p in out.split("-") if p)[:64] or "unnamed"


def remote_operator_id(host_name: str, operator: str) -> str:
    """The remote operator id, namespaced by host.

    The prism hosts announce bare operator names (`{"name": ..., "operators": [...]}`). Written
    as-is, any host that can reach the register endpoint would claim `op.dev.run_tests` — the
    operator that shells out to pytest — and shadow the local one. A registration creates ids only
    under its own `op.host.<host>.` prefix.

    The `op.` prefix is load-bearing: `signal.resolve` keys its `remote` branch on
    `address.startswith("op.")` together with `artifact["remote"]`, so ids under any other prefix
    resolve `unknown`, `signal.send` returns `dispatched: None`, and the transport target is lost.
    That surface is what these ids exist for — see `register_remote_host`."""
    return "op.host.%s.%s" % (_slug(host_name), _slug(operator))


def register_remote_host(store, *, name: str, operators, endpoint: str = "",
                         capabilities=None, principal: str = "") -> dict:
    """Record a host that announced itself, plus the operators it says it serves.

    Everything here is declared rather than measured, and the artifacts say so. A machine this
    process is not running on cannot be probed, and an operator whose spec was never sent cannot be
    verified. So:

    - the host artifact carries `source: "declared"` on every capability and `probed: false`; an
      unprobed capability is `unknown` rather than absent, so the three-valued discipline holds
      across the wire exactly as it does locally;
    - each operator artifact is written unsigned and not admissible — the prism hosts send a
      name, not a spec, so there is nothing to content-address and nothing to verify, and
      `opsign.admit()` holds them back. They are *pointers to a remote endpoint*, recorded so the
      graph knows the host exists and what it claims.

    The row states which surface dispatches it, because `invoke` is not that surface.

    Each row is minted with no `kind` and no `spec`, so `genesis.invoke` — which dispatches on
    `op["kind"]` — answers `"operator ... is not invokable"`, which on its own reads the same as a
    typo'd id. `signal.resolve` is the surface that serves these: it keys its `remote` branch on
    `id.startswith("op.")` together with `artifact["remote"]`, reads the `endpoint`, and
    `signal.send` returns the sealed signal plus its transport target. Invoking a remote operator is
    emitting a signed signal toward it rather than dialing an RPC (`signal.py` §P4). The id shape
    and the `remote` flag are therefore load-bearing, and the two fields below make that routing
    legible on the row itself rather than only in two other modules.

    A locally-runnable operator is never minted here: this endpoint records a claim about a remote
    host, and it neither calls out to that host's endpoint nor accepts a `kind`+`spec` that would let
    a registration author executable code of its own. Dispatch stays confined to emitting a signed
    signal toward the endpoint that was declared.

    Both prism legs POST `{api_uri}/hosts/register` inside a bare `except` logged "non-fatal", so a
    host that is told nothing here is told nothing anywhere. The registration is logged for that
    reason.
    """
    from ember import genesis
    artifacts = getattr(store, "artifacts", None) or store
    host_slug = _slug(name)
    host_id = "host.%s" % host_slug

    declared = {}
    for cname in validate(capabilities):            # raises UnknownCapability on a bad name
        # `kind_of` and `describe` answer for any acceptable name, including canonical and
        # open-family ones. Indexing `CAPABILITIES` directly would only cover the local probe
        # registry, so a canonical name accepted by `validate` would 500 the endpoint.
        declared[cname] = {"present": True, "kind": kind_of(cname),
                           "source": "declared", "description": describe(cname)}

    names = [str(o).strip() for o in (operators or []) if str(o).strip()]
    written = []
    for op_name in names:
        oid = remote_operator_id(name, op_name)
        artifacts.put_artifact({
            "id": oid,
            "content_type": OPERATOR_CONTENT_TYPE,
            "state": "committed",
            # No `kind`/`spec`: we were sent a name. `evolution.spec_hash` returns None for this,
            # so it carries no content address and `opsign.verify_operator` reports it unsigned.
            "remote": True,
            # The two fields that name the serving surface. `genesis.invoke` dispatches on `kind`,
            # which this row has none of, so it answers "not invokable" — the same shape a typo'd id
            # produces. `signal.resolve` / `signal.send` serve it instead: a remote invocation is a
            # sealed signal toward the endpoint rather than an RPC. Stated on the artifact so a
            # reader of the row learns it here.
            "invokable_locally": False,
            "dispatch": REMOTE_DISPATCH,
            "host": host_id,
            "endpoint": endpoint or None,
            "operator_name": op_name,
            "context": "operator %r served remotely by host %s" % (op_name, name),
            "content": "",
            "lemmas": [_slug(op_name).replace("-", " ")],
            "provenance": genesis.P_ASSERTION,   # a claim by the host; nothing has verified it
            "cited_from": genesis.CITE_GENESIS,
            "created_by": principal or host_id,
            "created_time": genesis._now(),
        })
        written.append(oid)

    artifacts.put_artifact({
        "id": host_id,
        "content_type": HOST_CONTENT_TYPE,
        "state": "committed",
        "host": host_slug,
        "remote": True,
        "probed": False,                 # we are not running on it; nothing here was measured
        "endpoint": endpoint or None,
        "capabilities": declared,
        "offers": sorted(declared),
        # Everything nameable that this host did not declare, over the canonical vocabulary as well
        # as the local probe registry, because a remote host is not limited to what this box can
        # measure. Open-family members are unenumerable by construction and do not appear.
        "unknown": sorted((set(CAPABILITIES) | set(CANONICAL_KINDS)) - set(declared)),
        "operators": written,
        "context": "remote host %s serving: %s" % (name, ", ".join(names) or "nothing"),
        "content": "",
        "provenance": genesis.P_ASSERTION,
        "cited_from": genesis.CITE_GENESIS,
        "created_by": principal or host_id,
        "created_time": genesis._now(),
    })
    if written:
        # Once per registration. Both prism legs wrap this POST in a bare `except` logged
        # "non-fatal", so without this line a host would learn only at invoke time that
        # `genesis.invoke` has nothing to dispatch for its operators.
        print("[ember.capability] host %r registered %d remote operator pointer(s) dispatch=%s "
              "(NOT locally invokable: a name carries no kind/spec): %s"
              % (name, len(written), REMOTE_DISPATCH, ", ".join(written)),
              file=sys.stderr, flush=True)
    try:
        from ember.ontology import match
        match.invalidate(artifacts)      # new offers exist; the selection cache must see them
    except Exception:
        pass
    return {"registered": True, "host": host_id, "operators": written,
            "dispatch": REMOTE_DISPATCH, "invokable_locally": False,
            "declared_capabilities": sorted(declared),
            "note": "operators are UNSIGNED remote pointers — recorded, not admissible, and NOT "
                    "locally invokable: carrying no kind/spec, genesis.invoke has nothing to "
                    "dispatch. They are addressed through signal.resolve/signal.send (a sealed "
                    "signal toward the host endpoint), which is the surface they exist for."}


__all__ = ["SOURCE", "EFFECTOR", "HOST_CONTENT_TYPE", "OPERATOR_CONTENT_TYPE", "REMOTE_DISPATCH",
           "Capability", "CAPABILITIES", "CANONICAL_KINDS", "OPEN_FAMILIES", "LEGACY_LOCAL_NAMES",
           "UnknownCapability", "is_valid_capability", "kind_of", "describe", "validate",
           "probe_all", "offers", "unknown", "host_artifact", "publish", "remote_operator_id",
           "register_remote_host"]
