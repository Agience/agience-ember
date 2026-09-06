"""Finding the mesh from one URL, and not trusting the URL alone.

A new ember needs two things: where the peers are, and proof that this really is the network it
meant to join. `https://origin.agience.ai/.well-known/agience` supplies the first. It supplies the
second only in the sense that a courier supplies a sealed envelope — **the hostname proves nothing.**

This module verifies rather than merely fetching. TLS proves you are talking to whoever holds the
DNS name and a certificate for it. That is a statement about *control of a name*, not about *which
network this is*. If an ember joins because a document arrived from `origin.agience.ai`, then
whoever holds that name decides who is in the mesh and what they connect to — and the peer-to-peer
install story is over before it is built, because every install is really a call to one server.

So: the discovery document is checked against an **anchor the ember already holds** — its authority
manifest, which arrived with its bundle, on a USB stick, or from a peer. Match the anchor and the
document is good no matter who carried it (which is also what lets a peer serve this document when
origin is down). Fail the anchor and it is discarded, however impeccable the TLS.

The one case that is genuinely different is a first boot with no anchor at all. That is
trust-on-first-use, it is a real decision with real consequences, and it is spelled `trust_on_first_use=True`
at the call site rather than being the quiet default — see `discover`.

`role` is carried in the document so the distinction survives the trip instead of being re-guessed
by every caller: `authority` is what you verify against, `lattice` is the shared substrate (mantle —
a plain REST service, with no ember on it), and `peer` is another ember. An ember keeps
its own lattice and reconciles; it is not a thin client of the substrate.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: The path is fixed. A configurable discovery path is a configurable trust root.
WELL_KNOWN_PATH = "/.well-known/agience"

DEFAULT_TIMEOUT = 15


class DiscoveryError(RuntimeError):
    """Discovery failed. Never swallowed — an ember with no mesh must say so, not run alone quietly."""


class AnchorMismatch(DiscoveryError):
    """The document did not match the anchor this ember holds.

    This is the interesting failure: it means the URL answered, TLS was fine, and the network on the
    other end is not the one this ember belongs to. Wrong environment, wrong fleet, or an attempt to
    move it onto someone else's mesh.
    """


@dataclass(frozen=True)
class Service:
    name: str
    uri: str
    role: str                       # "authority" | "lattice" | "peer"
    jwks: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_peer(self) -> bool:
        """Another ember, reconciled with directly. The lattice is not one.

        `mantle` is `lattice`, not `peer`, and the difference is load-bearing: there is no ember
        on the mantle endpoint, so treating it as a peer would attempt ember-to-ember reconciliation
        against a plain REST service. Embers peer with each other and share the lattice as
        substrate."""
        return self.role == "peer"

    @property
    def is_lattice(self) -> bool:
        return self.role == "lattice"


@dataclass(frozen=True)
class Discovery:
    """A verified (or explicitly unverified) view of the mesh."""
    issuer: str
    origin: str
    authority_artifact: str
    services: Dict[str, Service]
    verified: bool
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def peers(self) -> List[Service]:
        """Other embers to reconcile with. Ordered by name so a peer list is reproducible."""
        return [s for _, s in sorted(self.services.items()) if s.is_peer]

    @property
    def lattice(self) -> Optional[Service]:
        """The shared substrate (mantle), or None if this mesh publishes no lattice."""
        for _, s in sorted(self.services.items()):
            if s.is_lattice:
                return s
        return None

    def uri(self, name: str) -> Optional[str]:
        s = self.services.get(name)
        return s.uri if s else None

    @property
    def fingerprint(self) -> str:
        """A stable digest of the trust-bearing content — the authority id and every service anchor.

        Not over the whole document: `origin`/`jwks_uri` are echoes of the request's own
        base URL, so including them would make the same mesh fingerprint differently depending on
        which courier served it.
        """
        material = {
            "authority_artifact": self.authority_artifact,
            "issuer": self.issuer,
            "services": {n: {"uri": s.uri, "jwks": s.jwks} for n, s in sorted(self.services.items())},
        }
        # RFC 8785 requires raw UTF-8, not \uXXXX escapes. Without `ensure_ascii=False` a
        # non-ASCII service URI hashes differently here than in prism-c or
        # ember/runtime/runner.py, which already conform — so the same mesh would fingerprint
        # two ways depending on which implementation looked at it. Caught by the
        # canonical-json gate, not by review.
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _parse(doc: Dict[str, Any], *, verified: bool) -> Discovery:
    for required in ("issuer", "authority_artifact", "services"):
        if required not in doc:
            raise DiscoveryError(
                f"discovery document is missing {required!r} — this is not an Agience discovery "
                f"document, or the server is a different version. Keys present: {sorted(doc)}")
    services = {}
    for name, entry in (doc.get("services") or {}).items():
        uri = (entry or {}).get("uri")
        if not uri:
            continue                       # advertised with nowhere to connect — not a service
        services[name] = Service(name=name, uri=uri,
                                 role=(entry.get("role") or "peer"),
                                 jwks=entry.get("jwks") or {})
    return Discovery(
        issuer=doc["issuer"],
        origin=doc.get("origin") or "",
        authority_artifact=doc["authority_artifact"],
        services=services,
        verified=verified,
        raw=doc,
    )


def fetch(origin_url: str, *, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """GET the discovery document. Unverified by construction — the caller must check the anchor."""
    url = origin_url.rstrip("/") + WELL_KNOWN_PATH
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 - operator-supplied URL
            if r.status != 200:
                raise DiscoveryError(f"{url} returned HTTP {r.status}")
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise DiscoveryError(f"{url} returned HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise DiscoveryError(f"{url} unreachable: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise DiscoveryError(f"{url} did not return JSON: {e}") from e


def verify(doc: Dict[str, Any], anchor: Dict[str, Any]) -> Discovery:
    """Check a discovery document against the authority manifest this ember already holds.

    `anchor` is the manifest shape: `{"artifact_id": ..., "trust_anchors": {name: {"jwks": ...}}}`.

    Two things must hold, and the second is the one that catches a swap:
      1. the authority artifact id matches — this is the same authority;
      2. every service the document advertises that the anchor also knows carries the same public
         JWKS. A document may legitimately advertise a service the anchor has not heard of (the mesh
         grew); it may never redefine the key of one the anchor already pins.
    """
    d = _parse(doc, verified=False)
    expected_id = anchor.get("artifact_id")
    if not expected_id:
        raise DiscoveryError("anchor has no artifact_id — it is not an authority manifest")
    if d.authority_artifact != expected_id:
        raise AnchorMismatch(
            f"discovery advertises authority {d.authority_artifact!r} but this ember is anchored to "
            f"{expected_id!r}. This is a different network — refusing to join.")

    pinned = anchor.get("trust_anchors") or {}
    for name, svc in d.services.items():
        known = pinned.get(name)
        if not known or "jwks" not in known:
            continue                        # new service, not pinned — allowed
        if svc.jwks != known["jwks"]:
            raise AnchorMismatch(
                f"service {name!r} is advertised with a DIFFERENT public key than this ember has "
                f"pinned. Either the service was re-keyed (re-anchor deliberately) or this document "
                f"is not from your network — refusing to join.")
    return Discovery(issuer=d.issuer, origin=d.origin, authority_artifact=d.authority_artifact,
                     services=d.services, verified=True, raw=d.raw)


def discover(origin_url: str, *, anchor: Optional[Dict[str, Any]] = None,
             trust_on_first_use: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Discovery:
    """Fetch and verify in one call — the normal entry point.

    With an `anchor`, the document is verified against it and `verified=True`.
    Without one, this raises unless `trust_on_first_use=True` is passed EXPLICITLY. There is no
    default that quietly joins an unverified network: the returned `fingerprint` is what an operator
    records so the second boot is verified rather than trusted again.
    """
    doc = fetch(origin_url, timeout=timeout)
    if anchor is not None:
        return verify(doc, anchor)
    if not trust_on_first_use:
        raise DiscoveryError(
            "no authority anchor supplied. Pass `anchor=` (the authority manifest that came with "
            "this ember's bundle) to verify, or `trust_on_first_use=True` to accept this document "
            "unverified and record its fingerprint. Joining unverified is a decision, not a default.")
    return _parse(doc, verified=False)
