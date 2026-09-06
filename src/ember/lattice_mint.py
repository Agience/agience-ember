"""Lattice minting — the small writes that put a principal, a grant, and a citation into the store.

`_mint` stamps the invariant fields every artifact carries (state, times, provenance, citation) and
writes it. `_ensure_edge` adds a labelled edge idempotently. `_author_ref` resolves an author claim to
the vertex id of its person artifact, minting that artifact if absent. `_ensure_private` mints a
principal's private collection plus the owner read grant that is the only thing making it private.

`_author_ref` resolves through `mantle.services.principal`. A dangling `created_by` does not error:
it silently breaks grant propagation, because authorization stops flowing through a reference to a
vertex that is not there. That is why the resolution exists at all.

That does not mean identity became storage. A person is an artifact (contract §2.1: no separate
identity table, no special-cased principal type), so the derivation that gives it a stable id belongs
wherever the artifacts are written. `agience-origin` still owns identity as a service — its issuer,
its JWKS, its `models/person.py` row — which is a different thing from the uuid5 that makes
`created_by` a legal column value across observers.

Provenance
----------
Vendored from `agience-mantle`, `src/mantle/lattice_mint.py` (Apache-2.0, Copyright Ikailo Inc.), at
commit `9768c5f`; mantle removed it in `a7cfd0c`. These are mint helpers rather than storage
primitives — they compose `put_artifact` / `add_edge` calls a caller could make itself — and ember is
their only consumer, so they sit beside `ember/genesis.py`, their loudest caller, which re-exports
them. Copied unmodified apart from this note and one import: the person derivation lives in
`mantle.services.principal`, which is where `ember/genesis.py` reads `PERSON_CONTENT_TYPE`, so
`_author_ref` names it directly. Every store write goes through mantle's own API.
"""
from __future__ import annotations

from ember.authorship import DEFAULT_AUTHOR

from typing import Any, Dict, List, Optional, Tuple

from mantle.db.constants import COLLECTION_CONTENT_TYPE
from prism.grounding import CITE_GENESIS, P_HUMAN, _now

#: The root collection every other collection hangs under.
UNIVERSE = "universe"
CITATION_CONTENT_TYPE = "application/x-citation"


def _mint(store, doc: Dict[str, Any]) -> None:
    doc.setdefault("state", "committed")
    doc.setdefault("created_time", _now())
    doc.setdefault("modified_time", doc["created_time"])
    doc.setdefault("provenance", P_HUMAN)
    # invariant (§12): every artifact carries a citation. System artifacts anchor on cite.genesis.
    doc.setdefault("cited_from", CITE_GENESIS)
    store.artifacts.put_artifact(doc)


def _ensure_edge(store, from_id: str, to_id: str, label: str,
                 props: Optional[Dict[str, Any]] = None) -> bool:
    """Idempotent labeled edge: add only if `to_id` is not already an out-neighbor under
    `label`. The leaf graph store's add_edge always creates, so we dedupe here."""
    if store.graph is None:
        return False
    if to_id in store.graph.neighbors(from_id, label, direction="out"):
        return False
    store.graph.add_edge(from_id, to_id, label, props or {})
    return True


def _author_ref(store, author: str) -> str:
    """Resolve an author claim (an email / subject) to the vertex id of its person artifact,
    minting that artifact if it does not exist. Idempotent; safe to call per-row.

    A process author resolves to a foundation entity rather than raising or requiring a person
    owner — the same `uuid5(ns, issuer\nsub)` derivation under a distinct foundation issuer, so
    every observer computes the identical id (what makes `created_by` a legal column) and a
    foundation id can never collide with a person id. It is one entity per process, not one for
    all: `ember-source` and `ember-local` write different rows, and collapsing them would destroy
    the only distinction those 200,000+ rows carry.

    `principal_artifact` is the single dispatch point — it mints a person for a person and a
    foundation for a process, so this function does not need to know which it has."""
    from mantle.services.principal import person_id, principal_artifact
    if not author:
        return author
    pid = person_id(author)          # dispatches: a process author resolves to its foundation id
    if store.artifacts.get_artifact(pid) is None:
        # Not via `_mint`: an identity carries no citation and no provenance rung. Provenance is a
        # claim about where content came from; an identity is not content, and stamping
        # `cited_from: cite.genesis` on it would assert that genesis is where this principal came
        # from. It also breaks the ordering — `cite.genesis` itself needs an author.
        store.artifacts.put_artifact(principal_artifact(author))
    return pid


def _ensure_private(store, principal: str) -> Tuple[str, str]:
    pid = f"private.{principal}"
    if store.artifacts.get_artifact(pid) is None:
        _mint(store, {
            "id": pid, "content_type": COLLECTION_CONTENT_TYPE, "name": f"Private ({principal})",
            "context": {"name": f"Private ({principal})", "kind": "private",
                        "provenance": P_HUMAN, "origin": "genesis"},   # owner = the grant grantee
            "lemmas": ["private", principal.lower()], "content": ""})
        _ensure_edge(store, pid, UNIVERSE, "sub_collection_of")
        # Private is a grant, not a flag: the owner's read grant is the only thing that makes this
        # collection non-public. `access.gated_collections` contains `pid`, so `access.is_public` is
        # False for every member filed here and only the owner's light-cone reaches them. There is no
        # flag fallback, so a private collection that cannot be gated must fail rather than silently
        # leak — not wrapped: a mint failure propagates and aborts the write.
        from mantle.db import access
        access.mint_owner_read_grant(store, pid, principal)
    cite = f"cite.owner.{principal}"
    if store.artifacts.get_artifact(cite) is None:
        _mint(store, {
            "id": cite, "content_type": CITATION_CONTENT_TYPE,
            "context": {"dataset": f"owner-provided ({principal})", "kind": "citation",
                        "provenance": P_HUMAN,
                        "role": "the owner deliberately provided this in conversation"},
            "lemmas": ["owner", "private", principal.lower()],
            "content": f"provided by {principal} in conversation"})
    return pid, cite


# ── source progress: what a dataset ingest has already finished ──────────────────────────────────


SHARD_DONE_CT = "application/vnd.agience.shard-done+json"


def _shard_done_id(name: str, shard: str) -> str:
    import hashlib
    return f"sharddone.{name}." + hashlib.sha256(shard.encode()).hexdigest()[:16]


def _shards_done(store, name: str) -> List[str]:
    """Completed shards are the set of per-shard checkpoint artifacts. One artifact per shard (not
    a list inside one shared doc) so concurrent workers never lose each other's updates. The typed
    path filters on the indexed `ct` first and evaluates `source_name` only within that bucket."""
    out = []
    from mantle.db.typed_fetch import _typed
    listf = _typed(store.artifacts, "list_by_doc_field")
    if listf is not None:
        # `limit` must exceed the largest shard count we will ever checkpoint (full Wikipedia is
        # 41). A truncated list here reads as "that shard is not done" and re-ingests it, so it is
        # set well above the real ceiling rather than left at the method default.
        out = [d.get("shard") for d in listf(content_type=SHARD_DONE_CT, field="source_name",
                                             value=name, limit=100000) if d.get("shard")]
    else:
        # No typed method (the in-memory fakes, and any future backend).
        # Fall back to the store's own typed list rather than to SQL — slower, but it is a real
        # answer. Never hand this query to an untyped `.query`, which returns `[]` for this
        # shape of query.
        try:
            out = [d.get("shard") for d in
                   store.artifacts.list_artifacts(content_type=SHARD_DONE_CT)
                   if d.get("source_name") == name and d.get("shard")]
        except Exception:
            pass
    # migrate/merge any legacy list still parked on the source artifact
    c = store.artifacts.get_artifact(f"source.{name}")
    if c and isinstance(c.get("context"), dict):
        out += list(c["context"].get("shards_done", []))
    return sorted(set(out))


def _mark_shard_done(store, name: str, shard: str) -> None:
    # idempotent, contention-free: each completed shard is its own committed artifact.
    #
    # THE LABEL IS AN `offer`, NOT `content` — corrected 2026-08-26. It read
    # `"content": f"shard done: {name}/{shard}"`, which is a TITLE: it says nothing that
    # `source_name` and `shard` on this same doc do not already say. [John, 2026-08-25:
    # *"context, offer, description, title — all the same field."*] `offer` is the one naming
    # field, and it is the field the lexical arm actually indexes — the same defect the 48
    # operator artifacts carried when none of them was findable.
    #
    # And it is not only tidiness — a body in `content` is a body the lattice must protect.
    # `artifacts_holding_inline_plaintext` is pinned at 0 and counts exactly
    # `content <> '' AND content_encrypted <> 1`, so a marker carrying a readable string is a row
    # of that population waiting to happen. Measured 2026-08-26: the 18 rows of this type already
    # on 71/home read `content: ''` with a `content_ref` set, so this string is not what those rows
    # carry — the literal here had drifted away from what the store actually holds.
    store.artifacts.put_artifact({
        "id": _shard_done_id(name, shard), "content_type": SHARD_DONE_CT, "state": "committed",
        "source_name": name, "shard": shard, "offer": f"shard done: {name}/{shard}",
        "provenance": P_HUMAN, "cited_from": CITE_GENESIS, "created_time": _now()})


def mint_source_triple(store, name: str, *, cite_meta: Dict[str, Any], offer: str,
                       author: str = DEFAULT_AUTHOR) -> None:
    """Mint the citation anchor + source-operator (fitness-selected) + per-source collection
    for a dataset `name`, before any record flows. `name` becomes cite.<name> /
    op.source.<name> / source.<name>. Idempotent."""
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    from prism.runner import evolution      # the sha-verified distribution path
    _mint(store, {
        "id": f"cite.{name}",
        "content_type": CITATION_CONTENT_TYPE,
        "context": {**cite_meta, "kind": "citation", "provenance": P_HUMAN},
        "lemmas": [name.lower(), "source", "dataset"],
        "content": f"{cite_meta.get('dataset', name)} — citation anchor.",
        "provenance": P_HUMAN, "created_by": author,
    })
    store.artifacts.put_artifact(evolution.preserve_fitness(store.artifacts, {
        "id": f"op.source.{name}",
        "content_type": evolution.OPERATOR_CONTENT_TYPE,   # from the bundle; mantle may not import crystal
        "state": "committed",
        "context": offer,
        "content": f"source-operator op.source.{name}: {offer}",
        "provenance": P_HUMAN, "created_by": author, "created_time": _now(),
    }))
    _mint(store, {
        "id": f"source.{name}",
        "content_type": COLLECTION_CONTENT_TYPE,
        "name": f"source: {name}",
        "context": {"name": f"source: {name}", "kind": "source", "of": f"cite.{name}",
                    "provenance": P_HUMAN, "origin": "genesis",
                    "metrics": {"artifacts": 0, "dark_matter": 0, "rho": None}},
        "lemmas": [name.lower(), "source"],
        "content": "", "provenance": P_HUMAN, "created_by": author,
    })
    _ensure_edge(store, f"source.{name}", "sources", "sub_collection_of")
