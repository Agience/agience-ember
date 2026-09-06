"""Sources — the dual of an operator, on the ingest side.

An operator is a morphism invoked on a need: input to output, synchronous, pull. A source is a
coalgebra: seeded by the world, it unfolds a stream of observations over time. A folder changes, a
feed ticks, a signal arrives, and the source emits `Observation`s into the store as dark matter for
describe-operators to illuminate.

Ember hosts a runtime of pluggable sources, each declared as a `vnd.agience.source+json` artifact in
the same way an MCP server is an artifact. Sources are local, so a leaf watches local folders and
local signals with no chorus and no network.

A source observes the world and takes no submissions: there is no endpoint a client pushes into.
Provenance is grounded in ember having seen the artifact — its path and mtime — so what a source
reports is what it read from disk.
"""
from __future__ import annotations

from prism.source import Observation, Source  # noqa: F401  the contract, single-homed in prism

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Protocol
from urllib.parse import urlparse

SOURCE_CONTENT_TYPE = "application/vnd.agience.source+json"

try:
    from prism.mass import Provenance
    _OBSERVED = Provenance.OBSERVED
except Exception:
    _OBSERVED = "observed"

_EXT_CT = {".py": "text/x-python", ".md": "text/markdown", ".txt": "text/plain",
           ".json": "application/json", ".toml": "text/x-toml", ".yaml": "text/yaml",
           ".yml": "text/yaml", ".ts": "text/x-typescript", ".js": "text/javascript"}






class FolderSource:
    """Watch a folder; each new-or-changed file, by mtime, is an Observation. Polling rather than an
    OS watcher, so it is deterministic and dependency-free."""

    kind = "folder"

    def __init__(self, name: str, root: str, *, exts: Optional[set] = None,
                 recursive: bool = True, skip: Optional[set] = None):
        self.name = name
        self.root = Path(root)
        self.exts = exts or set(_EXT_CT)
        self.recursive = recursive
        self.skip = skip or {".git", "__pycache__", "node_modules", ".venv", "venv",
                             ".mypy_cache", ".pytest_cache", "dist", "build", "site-packages",
                             "_scratch", "_archive", "_library", "library-cleaned", "_generated"}
        self._seen: Dict[str, float] = {}       # path -> mtime last observed

    def prime(self) -> int:
        """Record current mtimes without reading or emitting, so a live watcher fires on changes
        made after this call rather than on the whole tree at startup. Stat-only, so priming a big
        tree is fast."""
        n = 0
        for dp, dns, fns in os.walk(self.root):
            dns[:] = [d for d in dns if d not in self.skip]
            for fn in fns:
                p = Path(dp) / fn
                if p.suffix.lower() in self.exts:
                    try:
                        self._seen[p.as_posix()] = p.stat().st_mtime
                        n += 1
                    except OSError:
                        pass
        return n

    def poll(self) -> Iterable[Observation]:
        for dp, dns, fns in os.walk(self.root):
            dns[:] = [d for d in dns if d not in self.skip]
            if not self.recursive:
                dns[:] = []
            for fn in fns:
                p = Path(dp) / fn
                suf = p.suffix.lower()
                if suf not in self.exts:
                    continue
                key = p.as_posix()
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if self._seen.get(key) == mtime:
                    continue                     # unchanged -> not a new observation
                # Mark seen only after a successful read. A file that could not be opened — locked
                # by another process, which is routine on Windows, or EACCES, or a transient I/O
                # error — keeps its old `_seen` entry, so the next poll tries it again rather than
                # treating its current mtime as observed.
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue                     # unmarked -> retried on the next poll
                self._seen[key] = mtime
                yield Observation(                              # the full text, untruncated
                    id="file-" + hashlib.sha256(key.encode()).hexdigest()[:20],
                    content=f"{key}\n{text}",
                    content_type=_EXT_CT.get(suf, "text/plain"),
                    meta={"path": key, "mtime": mtime, "source": self.name},
                )










class SourceRuntime:
    """Ember-managed host for sources. Registers them (also as artifacts), polls them, and
    routes each Observation -> the store (dark matter) + an optional sink (e.g. code-reindex).
    `poll_once` is one deterministic tick; `run` loops it on an interval."""

    def __init__(self, bundle, *, sink: Optional[Callable] = None):
        # `bundle` is the LocalStore (artifacts + content store + keys_dir).
        self.bundle = bundle
        self.store = bundle.artifacts
        self.sink = sink
        self.sources: Dict[str, Source] = {}

    def register(self, source: Source, *, config: Optional[dict] = None) -> None:
        self.sources[source.name] = source
        # a source is an artifact too (pluggable, discoverable) — same as operators/MCP servers
        self.store.put_artifact({
            "id": f"src.{source.name}", "content_type": SOURCE_CONTENT_TYPE, "state": "committed",
            "context": f"a {source.kind} source that observes and emits into the store: {source.name}",
            "content": f"source {source.name} kind={source.kind} config={config or {}}",
            "created_by": "ember-local",
        })

    def poll_once(self) -> int:
        """Poll every source once; store each observation with its CONTENT in the content store
        (encrypted, content-addressed) and only a reference + preview in the artifact store; then
        fire the sink (describe). Returns the number of observations ingested this tick."""
        from mantle.shard import content as C
        docs: List[dict] = []
        obs_list: List[Observation] = []
        for src in self.sources.values():
            for obs in src.poll():
                doc = obs.to_doc()
                if self.bundle.content is not None and self.bundle.keys_dir is not None:
                    ref, size = C.put_content(self.bundle.content, self.bundle.keys_dir,
                                              obs.content.encode("utf-8"))
                    doc["content_ref"] = ref
                    doc["size"] = size
                    # The artifact carries the ref and its offer (the describe); `resolve_text()`
                    # reads the content through the ref.
                    doc.pop("content", None)
                docs.append(doc)
                obs_list.append((obs, doc))
        if docs:
            self.store.put_many(docs, batch=500)
            if self.sink:
                for obs, doc in obs_list:
                    try:
                        self.sink(doc)                # describe resolves full text via content_ref
                    except Exception:
                        pass                                  # ingestion continues past a bad sink
        return len(docs)

    def run(self, *, interval: float = 5.0, stop: Optional[Callable[[], bool]] = None) -> None:
        """Blocking poll loop (host under a thread/supervisord). Stops when `stop()` is true."""
        while not (stop and stop()):
            self.poll_once()
            time.sleep(interval)


# ── the sink that closes the ingest loop: describe each observation by its content type ──
def describe_sink(bundle):
    """On every observation, run the content-type's describe handler — code to symbols, prose to
    terms — so dark matter is illuminated (keyed) as it is observed. `bundle` is the LocalStore, so
    describe resolves the full content through `content_ref`."""
    from prism.runner import describe            # the sha-verified single distribution path

    def _sink(doc: dict) -> None:
        describe.describe(bundle, doc)

    return _sink


# alias for callers using the old name
def code_reindex_sink(bundle):
    return describe_sink(bundle)