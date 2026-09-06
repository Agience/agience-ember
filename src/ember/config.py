"""Ember settings — env-driven, with defaults that work with the network unplugged."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mantle.shard.local_collection import local_collection_id


def _default_cache_dir() -> Path:
    """Per-user cache location. Ember is a *cache*: this is throwaway state by design —
    deleting it costs a refill, never data (anything Ember authored is pushed as its own
    region; anything else came from a peer that still has it)."""
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Agience" / "Ember"
    return Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "agience-ember"


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    principal: str        # whose cache this is — scopes the region ids
    # Everything is an artifact. A collection is an artifact with the right edges, so this is
    # its artifact ID — never a display name. Mantle keys cells by `collection_id` for the same
    # reason. A name here would route into a cell that corresponds to no artifact: it would
    # "work", produce plausible region ids, and share with nobody.
    collection_id: str
    engine: str           # "extractive" (default) | "entroptics"
    # The local embedder's model_id. It names a vector space, not a class: the cross-walk is
    # fitted per space and anchor ids are content-addressed over it, so two potion models are
    # two spaces. Empty => HashEmbedder (plumbing only, not semantic).
    embed_model: str
    cloud_uri: str        # chorus, for the 2-way relay. Empty => fully disconnected.
    nprobe: int           # anchor cells a query is routed to

    @property
    def offline(self) -> bool:
        """No cloud configured => never attempt a refill. Ember still answers from shards."""
        return not self.cloud_uri


def load() -> Settings:
    return Settings(
        cache_dir=Path(os.getenv("EMBER_CACHE_DIR") or _default_cache_dir()),
        principal=os.getenv("EMBER_PRINCIPAL", "local"),
        # Defaults to the id of the local collection artifact this principal authors
        # (deterministic, so their devices agree). Override to join a different collection —
        # but it must be a real artifact id, never a name.
        collection_id=(os.getenv("EMBER_COLLECTION_ID")
                       or local_collection_id(os.getenv("EMBER_PRINCIPAL", "local"))),
        engine=os.getenv("EMBER_ENGINE", "distill"),
        embed_model=os.getenv("EMBER_EMBED_MODEL", ""),
        # Deliberately empty by default: Ember boots disconnected and useful. You opt in to
        # the network, not out of it.
        cloud_uri=os.getenv("EMBER_CLOUD_URI", "").rstrip("/"),
        nprobe=int(os.getenv("EMBER_NPROBE", "8")),
    )
