"""The signal — one signed primitive whose grounding selects message vs event.

Builds the envelope and the regime router of PEERING-AND-MESSAGING.md §0b/§3.

## One substrate, two regimes

A signal in flight and an artifact at rest are one shape: the provenance quadruple, content, and a
signature. A signal adds a `to` (address) and a channel, and at the receiver the channel selects the
regime.

  * `MESSAGE` — an ungrounded signal. It propagates unimpeded and carries information while leaving
    its source as it was. It activates the receiver (`ontology.activation.activate`) and changes no
    stored state. Best-effort, salience-gated. Most peer traffic is this.

  * `EVENT` — a grounded signal. Local, and the only regime that lands state: it commits beside
    whatever is already held under the same root. An event also radiates, so it may emit follow-on
    messages.

Grounding is the switch. A signal whose channel has a checkable referent may transform; one without
may only propagate. `prism.mass.has_referent` is that test — a partition of the channels, so the
boundary has no edge to sit on and nothing to tune.

## The channel is derived at the receiver

The envelope carries the sender's own claim, and `route()` re-derives the channel server-side from
the sender's kind (`prism.mass.derive_provenance`), as every write path in this system does. A
client-kind sender that stamps `human_validated` to force an event earns `assertion` — which has no
referent — before the switch reads it. Authority comes from the sender's kind, not from the sender's
own account of itself.

## Authorization is upstream of the switch

Grounding decides which regime; the field decides what a signal may reach at all. A signal acting
for person P reaches only what P may reach (`mantle.db.access.visible_to`), so the switch
runs on content the sender is permitted to touch.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

SIGNAL_CONTENT_TYPE = "application/vnd.agience.signal+json"

# The fields that constitute the signal's meaning and routing — signed as a set, so none can be
# altered in flight (not the `to`, not the `from`, not the claimed rung).
SIGNED_FIELDS = ("id", "to", "from_", "operator", "claimed", "content_ref", "seeds", "ts", "nonce")

# ── the switch ───────────────────────────────────────────────────────────────────────────────────
# Ungrounded, a signal propagates (a message). Grounded, it may transform (an event). What the
# switch tests: only a signal with a checkable referent or a real observation may transform another
# node's state; a hypothesis, unknown or assertion is heard and no more. That is
# `prism.mass.has_referent` — a partition of the channels, with no edge to sit on.
from prism.mass import has_referent as _has_referent

MESSAGE = "message"      # ungrounded — propagate only
EVENT = "event"          # grounded   — may transform


def _now() -> str:
    from ember import genesis
    return genesis._now()


# ── the sender's kind ────────────────────────────────────────────────────────────────────────────
def _principal_kind(from_: Dict[str, Any]) -> str:
    """The sender's kind for `prism.mass`, derived conservatively from `from_`.

    A delegate's signals are `SYSTEM`-kind at most. A delegation carries a user_id, which is a
    statement about whose authority is in play rather than a statement that a human acted; only an
    explicit human stake (`op.remember`) earns `HUMAN`. So an authenticated delegate's signal is
    `SYSTEM` (the observed ceiling) and an anonymous or local one is `CLIENT` (the hypothesis
    ceiling)."""
    from ember.runtime.delegate import LOCAL_ORIGIN, LOCAL_PERSON
    from prism.mass import SYSTEM, CLIENT
    origin = str(from_.get("origin") or "")
    person = str(from_.get("person") or "")
    authenticated = origin and origin != LOCAL_ORIGIN and person and person != LOCAL_PERSON
    return SYSTEM if authenticated else CLIENT


def derive_channel(from_: Dict[str, Any], claimed: str):
    """The channel this signal earns, derived server-side from the sender rather than read off the
    wire.

    `claimed` is what the sender says about the content; `derive_provenance` grants it or, when the
    sender's kind does not carry that grant, returns `ASSERTION` — the true description of an
    unbacked claim. A forged claim is therefore re-described before the switch reads it."""
    from prism.mass import Provenance, derive_provenance
    kind = _principal_kind(from_)
    try:
        claimed_rung = Provenance(claimed) if claimed else Provenance.HYPOTHESIS
    except ValueError:
        claimed_rung = Provenance.HYPOTHESIS
    # `has_verified_span` would let a citation earn SPAN_CITED; a bare wire claim cannot assert it,
    # so it stays False until the content is checked (the event path can re-derive with the span).
    return derive_provenance(claimed_rung, kind, has_verified_span=False)


def regime(channel) -> str:
    """Which regime this channel is: no checkable referent → MESSAGE (propagate only); a referent →
    EVENT (may transform).

    A partition of the channels, so there is no edge and nothing to tune."""
    return EVENT if _has_referent(channel) else MESSAGE


# ── the envelope ─────────────────────────────────────────────────────────────────────────────────
def _canonical(doc: Dict[str, Any]) -> bytes:
    """The exact bytes signed and verified — the `SIGNED_FIELDS`, canonical. Absent fields are
    normalized so `{}`, `[]` and missing agree across senders."""
    payload = {
        "id": doc.get("id") or "",
        "to": doc.get("to") or "",
        "from_": doc.get("from_") or {},
        "operator": doc.get("operator") or "",
        "claimed": doc.get("claimed") or "",
        "content_ref": doc.get("content_ref") or "",
        "seeds": doc.get("seeds") or {},
        "ts": doc.get("ts") or "",
        "nonce": doc.get("nonce") or "",
    }
    return _jcs_string(payload).encode("utf-8")


def _content_address(doc: Dict[str, Any]) -> str:
    """sha256 over every signed field except the id — the signal's own content address. Two
    identical signals dedup, a replay is detectable, and the id agrees with the payload by
    construction."""
    body = {k: doc.get(k) for k in SIGNED_FIELDS if k != "id"}
    return hashlib.sha256(
        _jcs_string(body).encode("utf-8")
    ).hexdigest()


def seal(delegate, to: str, *, content_ref: str = "", seeds: Optional[Dict[str, float]] = None,
         operator: str = "", claimed: str = "hypothesis", priv=None,
         keys_dir=None) -> Dict[str, Any]:
    """Author a signed signal from this delegate to an address.

    Stamps the provenance quadruple (`from_` = origin/person/host/observer), derives the channel,
    content-addresses the id, adds a timestamp and nonce for replay detection, and signs with the
    delegate's key. `priv` may be passed directly; otherwise it is loaded from `keys_dir` (or the
    delegate's store's keys dir). This is a write path, so key creation is allowed here; the verify
    path reads only."""
    from prism.trust import opsign
    prov = delegate.provenance()                     # {origin, person, host}
    doc: Dict[str, Any] = {
        "content_type": SIGNAL_CONTENT_TYPE,
        "to": to,
        "from_": {**prov, "observer": delegate.id},  # the four provenance terms
        "operator": operator or "op.signal",
        "claimed": claimed,
        "content_ref": content_ref or "",
        "seeds": dict(seeds or {}),
        "ts": _now(),
        "nonce": os.urandom(8).hex(),
    }
    doc["channel"] = derive_channel(doc["from_"], claimed).value   # informational; receiver re-derives
    doc["id"] = _content_address(doc)
    if priv is None:
        kd = keys_dir if keys_dir is not None else getattr(delegate.store, "keys_dir", None)
        if kd is not None:
            priv, _pub = opsign.authority_key(kd, create=True)
    if priv is not None:
        doc["signature"] = opsign.sign_bytes(_canonical(doc), priv)
        doc["signed_by"] = opsign.public_key_hex(priv.public_key())
    return doc


def verify(signal: Dict[str, Any], *, pub=None) -> Tuple[bool, str]:
    """`(ok, reason)`. Each outcome carries its own reason, so unsigned, forged, tampered and
    stale-id stay distinguishable to the caller (the same discipline as
    `opsign.verify_operator`)."""
    from prism.trust import opsign
    sig = signal.get("signature")
    if not sig:
        return False, "unsigned"
    key = pub or opsign.load_public(str(signal.get("signed_by") or ""))
    if key is None:
        return False, "no verifying key (none supplied and signed_by missing/malformed)"
    if not opsign.verify_bytes(_canonical(signal), sig, key):
        return False, "signature does not verify (tampered, or signed by a different key)"
    if signal.get("id") != _content_address(signal):
        return False, "id does not match the signal's content (forged or replayed)"
    return True, ("verified against a supplied key" if pub is not None
                  else "self-consistent (embedded key — authorship unattested)")


# ── the regime router ────────────────────────────────────────────────────────────────────────────
def route(delegate, signal: Dict[str, Any], *, store=None, pub=None) -> Dict[str, Any]:
    """Deliver a signal to this delegate: verify → authorize → re-derive channel → switch.

    Returns `{regime, ...}`. The switch is the only place message-vs-event is decided, and it reads
    the server-derived channel rather than the wire's `channel` field.

    The order is load-bearing: verify (is it authentic?) → authorize (may it touch this at all?) →
    channel (which regime?). Each stage is a precondition of the next, so only a verified and
    authorized signal ever reaches the switch."""
    st = store if store is not None else getattr(delegate, "store", None)

    ok, why = verify(signal, pub=pub)
    if not ok:
        return {"regime": None, "delivered": False, "reason": "unverifiable: %s" % why}

    # Authorization is the extent of the field, upstream of the switch.
    if not _within_field(delegate, signal, st):
        return {"regime": None, "delivered": False,
                "reason": "outside the delegate's field (D4): sender may not reach this"}

    # The switch — re-derive the channel from the verified `from_`; any wire `channel` is ignored.
    channel = derive_channel(signal.get("from_") or {}, signal.get("claimed") or "hypothesis")
    reg = regime(channel)

    if reg == MESSAGE:
        return _propagate(delegate, signal, channel)
    return _transform(delegate, signal, channel, st)


def _within_field(delegate, signal: Dict[str, Any], store) -> bool:
    """May a signal for this sender reach what it targets? A signal carrying a content artifact is
    checked with `mantle.db.access.visible_to` against the delegate's person; a pure
    activation (seeds only, no content) is always within the field, because hearing is not
    reaching."""
    from mantle.db.access import visible_to
    art = signal.get("content") if isinstance(signal.get("content"), dict) else None
    if art is None:
        return True                                  # seeds-only: a message a delegate may always hear
    return visible_to(art, delegate.person, store=store)   # the person's grant light-cone


def _seeds_of(signal: Dict[str, Any]) -> Dict[str, float]:
    """The activation seeds a signal carries — explicit `seeds` if present, else grounded from its
    content artifact (`seeds_from_artifact`), so an in-wave-form signal skips re-grounding."""
    from ember.ontology import activation
    seeds = signal.get("seeds")
    if isinstance(seeds, dict) and seeds:
        return {str(k): float(v) for k, v in seeds.items()}
    art = signal.get("content")
    if isinstance(art, dict):
        return activation.seeds_from_artifact(art)
    return {}


def _propagate(delegate, signal: Dict[str, Any], channel) -> Dict[str, Any]:
    """MESSAGE / ungrounded: activate the receiver and change no state. Salience decides whether it
    fires; a below-salience arrival is recorded and nothing else happens, which is the backpressure
    the message regime provides."""
    from ember.ontology import activation
    seeds = _seeds_of(signal)
    ch = getattr(channel, "value", channel)
    if not seeds:
        return {"regime": MESSAGE, "delivered": True, "fired": False, "channel": ch,
                "reason": "nothing grounded from the signal"}
    r = activation.activate(delegate, seeds, source="peer")
    return {"regime": MESSAGE, "delivered": True, "fired": bool(r.get("fired")),
            "channel": ch, "lead": r.get("lead"), "salience": r.get("salience")}


def _transform(delegate, signal: Dict[str, Any], channel, store) -> Dict[str, Any]:
    """EVENT / grounded: land state beside what is held. An event also radiates, so it activates
    the receiver as a message would.

    Head is decided at read time, by the reader's own measurement, and nothing here arbitrates it.
    At this seam the evidence is always one signer against one holder, and
    `prism.resolution.separated([1, 1])` is False — as it is for every pair, since at n=2 the
    computed null is exactly 1.0000. A count of origins measures agreement, and a timestamp
    measures a clock; neither reads on which version is right. So the event lands and stands beside
    what is held, and the answer reports `stands_beside` rather than a displacement verdict.
    """
    art = signal.get("content") if isinstance(signal.get("content"), dict) else None

    # What is reported is what was observed: does this delegate already hold something under the
    # same root? A reader resolving versions later needs to know a root now carries more than one.
    # It is a fact about what is here, not a claim about which is right.
    stands_beside = None
    if art is not None and store is not None:
        root = art.get("root_id") or art.get("id") or ""
        try:
            existing = (getattr(store, "artifacts", None) or store).get_artifact(root)
        except Exception:
            existing = None
        stands_beside = bool(existing)

    # An event radiates (it still activates the receiver).
    radiated = _propagate(delegate, signal, channel)
    return {"regime": EVENT, "delivered": True, "channel": getattr(channel, "value", channel),
            # None if the event carried no artifact; otherwise whether a version was already held
            # under this root. An observation, not a displacement verdict.
            "stands_beside": stands_beside,
            "radiated": {"fired": radiated.get("fired"), "lead": radiated.get("lead")}}


# ── delivery — an arriving signal is routed ──────────────────────────────────────────────────────
# The mesh applies a peer's artifacts into the store and returns a count. This is the joint that
# carries an arriving signal onward to cognition: given a batch of applied artifacts, route the ones
# that are signals addressed to this observer through the switch. Pure and in-process, so the caller
# decides when to invoke it; `mantle.mesh.sync._apply_artifacts` does not call it, because a live
# wiring must fire on genuinely-new rows rather than on re-applies of rows already held.

BROADCAST = ("*", "all", "broadcast")


def addressed_to(signal: Dict[str, Any], delegate) -> bool:
    """Is this signal for this observer? Its own id, or a broadcast. Operator addresses
    (`op.host.<h>.<op>`) are resolved to a concrete observer id by `resolve` before they become a
    signal's `to`, so delivery matches an observer id or a broadcast."""
    to = str(signal.get("to") or "")
    return to == delegate.id or to in BROADCAST


def _is_own(signal: Dict[str, Any], delegate) -> bool:
    """A signal this delegate itself emitted. The echo guard keeps it from being delivered back,
    the same rule the mesh consume path holds: a node re-applies only what it did not author."""
    frm = signal.get("from_") or {}
    return str(frm.get("observer") or "") == delegate.id


def _grounded(r: Dict[str, Any]) -> bool:
    """Did a delivered signal do something here — fire an activation, land state, or radiate a
    firing? This is the condition under which peer-apply propagates onward to the persona; a
    below-salience arrival that grounded nothing is heard and kept local.

    `transforms` is read alongside `stands_beside` because a result dict can come from outside this
    module: `route` is substituted by callers and tests that supply `transforms`, and an arriving
    peer signal is a foreign dict. `stands_beside` is the key `_transform` itself reports."""
    return bool(r.get("fired") or r.get("transforms") or r.get("stands_beside")
                or (r.get("radiated") or {}).get("fired"))


def deliver(delegate, artifacts, *, store=None, forward=None) -> Dict[str, Any]:
    """Route every arriving signal addressed to this delegate through the switch. `artifacts` is a
    batch, e.g. what a consume pass just applied; non-signals and signals for other observers are
    counted as skipped. Returns a summary. Best-effort: a bad signal is recorded in the summary.

    The delegate's own signals are skipped (the echo guard), as is anything not addressed to it — a
    signal is a broadcast only when its `to` says so. Everything else goes through `route`, which
    verifies, authorizes, and switches message-vs-event.

    `forward` is the optional onward hop to a persona. Once a peer signal has grounded locally
    (`_grounded` — the runner's recognition, which stays in ember), the same signal propagates
    onward by calling `forward(artifact, result)`. The caller builds it, typically from
    `ember.runtime.reach.reactor(...).reach(need, to="op.learn")` over a carrier such as
    `prism.carriers.StoreCarrier`. With `forward=None` local grounding is the whole story, so a
    plain node needs no carrier. The onward reach is fire-and-forget: it neither breaks nor gates
    local delivery."""
    routed, skipped, forwarded = [], 0, 0
    for art in (artifacts or []):
        if not isinstance(art, dict) or art.get("content_type") != SIGNAL_CONTENT_TYPE:
            skipped += 1
            continue
        if _is_own(art, delegate) or not addressed_to(art, delegate):
            skipped += 1
            continue
        try:
            r = route(delegate, art, store=store)
        except Exception as e:
            r = {"regime": None, "delivered": False, "reason": "%s: %s" % (type(e).__name__, str(e)[:120])}
        routed.append({"id": art.get("id"), **r})
        if forward is not None and r.get("delivered") and _grounded(r):
            try:
                forward(art, r)                          # peer-apply propagates onward — fire-and-forget
                forwarded += 1
            except Exception:
                pass                                     # a persona reach never breaks local delivery
    return {"considered": len(artifacts or []), "skipped": skipped,
            "delivered": sum(1 for r in routed if r.get("delivered")),
            "forwarded": forwarded, "signals": routed}


# ── the resolver — an address becomes a target, and an invocation is a signal ────────────────────
# The address fields `op.host.<host>.<op>`, `host` and `endpoint` are minted by
# `runtime.capability.register_remote_host`, and `resolve` is what reads them. Per
# PEERING-AND-MESSAGING.md §3, resolving an operator wraps the invocation as a signed signal and
# delivers it as one, rather than dialling an RPC endpoint: RPC couples caller to callee and needs a
# broker, where a signal is best-effort and decoupled.

import re as _re                                                     # noqa: E402
from prism.canonical import canonical_string as _jcs_string
_HEX64 = _re.compile(r"^[0-9a-f]{64}$")
_AGI = _re.compile(r"^agi://", _re.I)


def resolve(address: str, *, store=None) -> Dict[str, Any]:
    """`{kind, ...}` for an address. Kinds:
      · `content` — a 64-hex spec/content hash → the operator/artifact with that address (immutable).
      · `remote`  — `op.host.<host>.<op>` → its host, endpoint and operator_name.
      · `local`   — a local `op.*` operator present in this store.
      · `observer`— a bare observer/delegate id (a direct recipient).
      · `agi`     — an `agi://` name, parsed; resolution goes through an Origin, and this reports
                    the parse rather than a guessed target.
      · `unknown` — nothing matched, reported as such.
    """
    a = str(address or "").strip()
    arts = getattr(store, "artifacts", None) or store
    if not a:
        return {"kind": "unknown", "reason": "empty address"}

    if _AGI.match(a):
        return {"kind": "agi", "address": a,
                "reason": "agi:// resolver not built (D6 design-only) — resolve via Origin, not DNS"}

    if _HEX64.match(a):
        return {"kind": "content", "hash": a}      # location-independent; caller looks it up by hash

    if a.startswith("op."):
        art = arts.get_artifact(a) if arts is not None else None
        if art and art.get("remote"):
            return {"kind": "remote", "operator": a, "host": art.get("host"),
                    "endpoint": art.get("endpoint"), "operator_name": art.get("operator_name")}
        if art:
            return {"kind": "local", "operator": a, "runtime": art.get("kind")}
        return {"kind": "unknown", "operator": a,
                "reason": "no operator artifact %r in this store" % a}

    return {"kind": "observer", "observer": a}


def send(delegate, to: str, *, seeds: Optional[Dict[str, float]] = None, content_ref: str = "",
         operator: str = "", claimed: str = "hypothesis", store=None) -> Dict[str, Any]:
    """Address a target, seal a signal, and dispatch it.

    Invoking a remote operator means emitting a signed signal toward it: the recipient's own
    attention decides what happens, and any answer comes back as a new signal. `send` returns the
    sealed signal plus how it was dispatched:
      · local target  → routed through the switch on this node, result inline.
      · remote target → the sealed signal and its `transport` target (host/endpoint), for `ship`.
    """
    st = store if store is not None else getattr(delegate, "store", None)
    target = resolve(to, store=st)

    # If the address resolves to a specific observer, that observer id is the signal's `to`;
    # an operator/content address travels as-is (the resolver on the far side dispatches it).
    sig = seal(delegate, to, seeds=seeds, content_ref=content_ref, operator=operator, claimed=claimed)

    if target["kind"] in ("local", "observer"):
        # deliverable on this node right now
        delivered = route(delegate, sig, store=st) if target["kind"] == "observer" \
            else _dispatch_local_operator(delegate, sig, target, st)
        return {"dispatched": "local", "target": target, "signal": sig, "result": delivered}
    if target["kind"] == "remote":
        return {"dispatched": "remote", "target": target, "signal": sig,
                "transport": {"host": target.get("host"), "endpoint": target.get("endpoint")},
                "note": "sealed — P6 ships it; delivery is a signal, not an RPC call"}
    return {"dispatched": None, "target": target, "signal": sig,
            "reason": "unroutable: %s" % target.get("reason", target["kind"])}


def _dispatch_local_operator(delegate, sig: Dict[str, Any], target: Dict[str, Any], store):
    """A signal addressed to a local operator. `route` verifies, authorizes and derives the channel,
    so the signal propagates (activates) exactly as any message does, and the resolved operator and
    its runtime are reported alongside. Activation is what runs here; the operator itself is not
    invoked, and the answer names the operator so a caller can see that."""
    r = route(delegate, sig, store=store)
    r["operator"] = target.get("operator")
    r["runtime"] = target.get("runtime")
    return r


def ship(sent: Dict[str, Any], *, token: str = "", timeout: float = 10.0) -> Dict[str, Any]:
    """Deliver a sealed signal to the host that serves the operator.

    Shipping is a separate step a caller chooses. `send()` seals the signal and reports the
    transport target; dialling happens here. That split is what keeps a signal best-effort and
    decoupled, where an RPC would couple caller to callee, and
    `test_send_to_a_remote_operator_wraps_a_signal_not_an_rpc` pins it. A sealed signal is a signal
    whether or not it is shipped.

    The address contract is the prism Host's, read from its source: an operator is mounted at
    ``POST {endpoint}/operators/{name}`` (`prism.host.host.Host.operator`), and the route carries a
    per-route auth dependency rather than middleware, so a missing credential returns 401 from that
    route.

    Every outcome is a dict naming what happened — shipped, an HTTP status from the host, an
    unreachable host, or an unexpected exception — so a caller can tell which occurred and act on
    it.
    """
    import json as _json
    import urllib.error as _uerr
    import urllib.request as _ureq

    target = (sent or {}).get("target") or {}
    operator = target.get("operator")
    endpoint = str(target.get("endpoint") or "").rstrip("/")
    op_name = str(target.get("operator_name") or "").strip()

    if target.get("kind") != "remote":
        return {"shipped": False, "operator": operator,
                "reason": "not a remote target (kind=%r); local signals are delivered by `send`"
                          % target.get("kind")}
    if not endpoint:
        return {"shipped": False, "operator": operator,
                "reason": "the operator artifact records no `endpoint`. `register_remote_host` "
                          "stores whatever the host announced — a host that registered without "
                          "one cannot be reached, and guessing a URL is not delivery."}
    if not op_name:
        return {"shipped": False, "operator": operator,
                "reason": "the operator artifact records no `operator_name`, so there is no route "
                          "to mount on the far side"}

    url = "%s/operators/%s" % (endpoint, op_name)
    payload = _json.dumps({"signal": sent.get("signal"), "operator": op_name}).encode("utf-8")
    req = _ureq.Request(url, data=payload, method="POST",
                        headers={"content-type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)

    try:
        with _ureq.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                body = _json.loads(raw) if raw else {}
            except ValueError:
                # Every response body reaches the caller as JSON, so a non-JSON body is wrapped and
                # labelled rather than handed back as a string the caller's `json.loads` would fail
                # on.
                body = {"error": "host returned non-JSON", "raw": raw[:500]}
            return {"shipped": True, "operator": operator, "url": url,
                    "status": resp.status, "result": body}
    except _uerr.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500] if exc.fp else ""
        hint = ("the host's operator routes carry a per-route auth dependency — supply a delegation "
                "token" if exc.code in (401, 403) else "")
        return {"shipped": False, "operator": operator, "url": url,
                "status": exc.code, "reason": "host refused: %s" % (detail or exc.reason), **({"hint": hint} if hint else {})}
    except _uerr.URLError as exc:
        return {"shipped": False, "operator": operator, "url": url,
                "reason": "host unreachable: %s" % exc.reason}
    except Exception as exc:                                    # noqa: BLE001 — reported in the result
        return {"shipped": False, "operator": operator, "url": url,
                "reason": "%s: %s" % (type(exc).__name__, exc)}


def health(endpoint: str, *, timeout: float = 5.0) -> Dict[str, Any]:
    """What a prism host says it is and serves — `GET {endpoint}/health` → `{status, host,
    operators}` (`prism.host.host.Host._health`). Used to show a registered host's live operators
    beside the ones it announced at registration, since a registration is a claim and the two can
    disagree."""
    import json as _json
    import urllib.error as _uerr
    import urllib.request as _ureq

    url = "%s/health" % str(endpoint or "").rstrip("/")
    try:
        with _ureq.urlopen(url, timeout=timeout) as resp:
            return {"reachable": True, "status": resp.status,
                    "health": _json.loads(resp.read().decode("utf-8", "replace") or "{}")}
    except _uerr.URLError as exc:
        return {"reachable": False, "url": url, "reason": str(getattr(exc, "reason", exc))}
    except Exception as exc:                                    # noqa: BLE001
        return {"reachable": False, "url": url, "reason": "%s: %s" % (type(exc).__name__, exc)}


__all__ = ["SIGNAL_CONTENT_TYPE", "MESSAGE", "EVENT", "BROADCAST",
           "derive_channel", "regime", "seal", "verify", "route",
           "addressed_to", "deliver", "resolve", "send", "ship", "health"]
