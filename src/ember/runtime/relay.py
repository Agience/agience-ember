"""The two-way channel — the leaf's link to the cloud.

Ember is two-way: chorus + tools reach local services (inward), and local services reach the
platform (outward). Both directions ride one outbound WebSocket to
``wss://<cloud>/relay/v1/connect`` (chorus) — no inbound ports, NAT/firewall friendly. The same
socket carries ``invoke_tool`` down (cloud -> local, see ``serve()``) and shard refills up (local
-> cloud, see ``fetch_regions``).

    inward   chorus --invoke_tool-->  Ember --> local MCP servers / tools     (serve(), a seam)
    outward  local service --> Ember --(cache miss)--> pull shards from mantle (fetch_regions)

Transport is injected, which makes the license boundary structural
--------------------------------------------------------------------
A Channel never imports chorus/lumen/mantle. It is handed a ``get_shard(peer, region)`` callable
and calls it. In tests that callable is an in-process peer (no socket). In production it is an
HTTPS/WS fetch to mantle. So "reach the AGPL platform over the wire, never link it" is not a rule
to remember — it is the *shape of the type*: the platform is a function argument, not an import.

Outward refill is a leaf -> cloud fetch, which is an ordinary **outbound** request (plain HTTPS to
mantle.agience.ai works behind NAT); the WebSocket is only needed for the cloud-*initiated*
inward direction. So `fetch_regions` is single-peer and simple; `serve()` is the WS-driven seam.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple

from mantle.mesh.manifest import ShardManifest
from mantle.mesh.node import MeshNode, ShardVerifyError

# What the injected transport must provide: (peer, region) -> (manifest, items) or (None, None).
# Exactly the mesh's GetShard shape, so an in-process MeshNode.get_shard fits with no adapter.
GetShard = "Callable[[str, str], Tuple[Optional[ShardManifest], Optional[dict]]]"

# The Refill answer_query wants: missing region ids -> the ones actually obtained + verified.
Refill = "Callable[[Sequence[str]], List[str]]"


class Channel(Protocol):
    """The two-way link. Implemented by the relay client; faked in tests."""

    def fetch_regions(self, regions: Sequence[str]) -> List[str]:
        """Outward: pull shards for these region ids, verify, import. Returns what landed."""
        ...

    def serve(self) -> None:
        """Inward: run the loop that lets the cloud invoke local tools."""
        ...


class Disconnected:
    """The default channel: there isn't one.

    Ember boots with no cloud configured and stays useful, so this is the normal state, not an
    error state. A refill request obtains nothing and the read path answers from what is already
    local (or honestly declines).
    """

    def fetch_regions(self, regions: Sequence[str]) -> List[str]:
        return []

    def serve(self) -> None:
        return None


class MeshChannel:
    """A live outward channel: refill missing regions from the cloud, verified.

    Holds the leaf's node (where verified shards land), the authority's public key (to verify
    them), the cloud peer id, and the injected ``get_shard`` transport. Nothing platform-specific
    is imported — the transport is the only thing that touches the wire, and it is a parameter.

    Verification is not optional and not the transport's job: every pulled shard goes through
    ``node.import_shard`` (signature + content_root + per-item hash), so the cloud is exactly as
    untrusted as a mesh peer or a disk file. A tampered or lying cloud response is dropped; the
    region simply stays a miss.
    """

    def __init__(self, node: MeshNode, authority_pub, get_shard, *, peer: str = "cloud") -> None:
        self._node = node
        self._authority_pub = authority_pub
        self._get_shard = get_shard
        self._peer = peer

    def fetch_regions(self, regions: Sequence[str]) -> List[str]:
        """Pull each region from the cloud, verify, import. A leaf<->cloud refill is single-peer,
        so this does not need the gossip directory — it asks the one cloud for each region and
        keeps what verifies. A region the cloud lacks, or that fails verification, is skipped."""
        landed: List[str] = []
        for region in regions:
            try:
                manifest, items = self._get_shard(self._peer, region)
            except OSError:
                continue                      # unreachable cloud — the leaf stays offline-correct
            if manifest is None or items is None:
                continue                      # cloud does not hold it
            try:
                self._node.import_shard(manifest, items, self._authority_pub)   # raises on tamper
            except ShardVerifyError:
                continue                      # a lying/tampered response is not a cache
            landed.append(region)
        return landed

    def serve(self) -> None:
        """Inward direction: an honest seam, not yet implemented.

        This is the cloud-initiated ``invoke_tool`` loop over the outbound WebSocket: the cloud
        asks the leaf to run a local tool. It needs the real WS client plus a local tool registry
        (which local MCP servers this leaf exposes), which is a larger piece. The outward path
        (`fetch_regions`) is complete and independent — a leaf can refill without ever serving.
        """
        raise NotImplementedError(
            "inward serve() is a seam: needs the WS client + a local tool registry (relay-host)"
        )

    def as_refill(self, on_import=None) -> "Refill":
        """Adapt this channel to the ``Refill`` answer_query expects.

        ``on_import(landed)`` (optional) lets the caller make the newly-imported shards
        *searchable* — pulled shards are opaque bytes in the node; turning them into a retrievable
        view (text + vectors) is the leaf's job (its embedder/describer), not the transport's, and
        blind content legitimately has no view. Without the hook, regions are held and verified
        but not yet answerable, which is a valid intermediate state.
        """
        def _refill(regions: Sequence[str]) -> List[str]:
            landed = self.fetch_regions(regions)
            if landed and on_import is not None:
                on_import(landed)
            return landed
        return _refill


__all__ = ["Channel", "Disconnected", "MeshChannel", "GetShard", "Refill"]
