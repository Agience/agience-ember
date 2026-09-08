"""Genesis P0 — schema freeze and collection scaffolding for the model-free universe.

Mints the seed ontology as first-class artifacts. The type system is open (GENESIS §3/§12):
vertex types, edge types, and collections are all unbounded, and the seed sets below are
*generators*, not a fence. New types are minted (as self-describing artifacts) whenever
observed structure demands them. Encoding the seed ontology as artifacts — rather than as
closed-world DDL — is what keeps the ontology legible and lets it grow by the same
observe -> describe -> select loop as everything else.

What P0 lays down (all idempotent — safe to re-run):
  1. vertex-type registry     (Content, Lexeme, Synset, Concept, Entity, Symbol,
                               Operator, Source, Collection, Citation)  — §3.1
  2. edge-type registry       (describes, via, cited_from, member_of, sub_collection_of,
                               calls, hypernym, hyponym, synonym, antonym, instance_of,
                               subclass_of, defines, references, consolidates, supersedes,
                               composed_of, observed_by, access_event)  — §3.2
  3. provenance-rung registry (the four GENESIS rungs mapped to prism.mass bands) — §3.3
  4. collection scaffolding   (universe root; curriculum stages 0-5; staging; self;
                               subjects root; sources root; ontology)  — §3.4/§5

The lemma inverted index that all keyed retrieval uses is live once the store's
`ensure_schema` has run (the lattice store's lemma index) — nothing to build here.

Everything below is deterministic. No models, no network. See GENESIS.md.
"""
from __future__ import annotations

from ember.authorship import DEFAULT_AUTHOR

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

#: Module logger. Fail-soft handlers here log through it rather than swallowing silently.
_log = logging.getLogger(__name__)

# Content-type discriminators for the seed ontology artifacts. The type system is open,
# so these are just labels on the single Artifact vertex class — not DDL.
VTYPE_CONTENT_TYPE = "application/vnd.agience.vtype+json"
ETYPE_CONTENT_TYPE = "application/vnd.agience.etype+json"
RUNG_CONTENT_TYPE = "application/vnd.agience.rung+json"
# A person is an artifact too (§2.1) — see `mantle/services/principal.py` for the id derivation and
# for what a shared person vertex carries.
from mantle.services.principal import PERSON_CONTENT_TYPE  # noqa: E402  (a discriminator, kept with its peers)
# `_mint`, `_ensure_edge`, `_author_ref` and `_ensure_private` are store writes homed in
# `ember.lattice_mint`, where the personas can reach them directly (lumen mints a private
# collection for every remembered turn). They are imported back here because this module is their
# loudest caller, and re-exported so callers keep saying `genesis._mint`; ember may import mantle,
# so the direction is legal.
from mantle.db.constants import COLLECTION_CONTENT_TYPE  # noqa: E402  (one home, in the store)
from ember.lattice_mint import (  # noqa: F401,E402  — re-exported: callers still say `genesis._mint`
    UNIVERSE, CITATION_CONTENT_TYPE, _author_ref, _ensure_edge, _ensure_private, _mint)
# Minting a source's citation anchor and checkpointing which shards are done are store writes whose
# only caller is the stage-0 ingester. Re-exported so `genesis.mint_source_triple` /
# `genesis._shards_done` keep resolving.
from ember.lattice_mint import (  # noqa: F401,E402
    SHARD_DONE_CT, _mark_shard_done, _shard_done_id, _shards_done, mint_source_triple)

# prism.mass provenance string values (the canonical ladder), reused so genesis artifacts carry the
# same provenance vocabulary as the rest of the store.
# Provenance rungs, the system-citation anchor and `_now` are single-sourced in
# `prism/grounding.py` — the small stable surface personas reach without importing this op-table.
from prism.grounding import (                                                       # noqa: E402
    P_HUMAN, P_OBSERVED, P_SPAN_CITED, P_HYPOTHESIS, P_ASSERTION, CITE_GENESIS, _now, TRANSDUCER_OP)


# ── the published precision of ρ / coverage, and the tolerances derived from it ──────────────
#
# ρ and `keyed_coverage` are published rounded to `_METRIC_DECIMALS` places. That rounding is the
# only thing that makes a tolerance necessary, so both tolerances are derived from it rather than
# typed in beside it:
#
#   * a value recorded to `_METRIC_DECIMALS` places carries an absolute error of at most half a
# quantum, so a range check ([0,1]) admits `_metric_half_quantum`;
#   * a difference of two such values carries at most a full quantum, so a trend check
# (`universe_cooling`: did ρ rise?) admits `_metric_quantum`. A ρ that "rose" by less than
#     one recorded digit did not rise; it was rounded.
#
# For a different tolerance to be right, the published precision would have to change — and it
# changes here, in one place, with the range and trend tolerances following it.
_METRIC_DECIMALS = 4


def _node_dir(store):
    """The node directory — where `metrics.jsonl`, `golden.json` and `worker.log` sit, beside
    `keys/` — or None when there is nothing to derive one from.

    Derived from the store's own keys dir, so this module carries no per-box knowledge.
    `mantle.shard.sqlite_store.open_sqlite_store` always populates `keys_dir` (from
    `EMBER_STORE_KEYS_DIR`, else `<root>/keys`) and refuses to open when that directory is absent,
    so a served node always has one. A store handle built without one — a fixture, an in-memory
    probe — has no node layout at all, and None says exactly that.

    None rather than a raise, because every caller here is a STATUS read whose contract is that
    missing telemetry reads as absent, not as a failure. This replaced a hardcoded absolute path
    that existed on one machine and resolved to nothing everywhere else, so absence is the
    behaviour callers already had; only the box-specific string is gone.
    """
    import os
    from pathlib import Path
    kd = store.keys_dir or (os.getenv("EMBER_STORE_KEYS_DIR") or "").strip()
    return Path(kd).parent if kd else None


def _metric_quantum() -> float:
    """The smallest difference two published ρ / coverage values can express."""
    return 10.0 ** -_METRIC_DECIMALS


def _metric_half_quantum() -> float:
    """The largest absolute error `round(x, _METRIC_DECIMALS)` can introduce."""
    return _metric_quantum() / 2.0


# ── typed-store dispatch (LATTICE Phase 1.3) ─────────────────────────────────
# Store reads here go through typed store methods rather than SQL strings handed to an untyped
# connection. A typed method that gets renamed raises `AttributeError` at the call site; an
# unrecognised SQL string comes back as an empty list that reads exactly like a healthy answer.
#
# `_typed` is the whole seam. It returns the store's typed method when the backend has one (the
# lattice store; see `mantle/db/vertex.py` and `edge.py`) and None otherwise, so each call
# site carries its own fallback for the in-memory fakes.

def _typed(obj, name: str):
    """The typed method `name` on `obj`, or None if this backend does not have it."""
    return getattr(obj, name, None) if obj is not None else None


# The provenance anchor for system-authored artifacts (the ontology, operators, collections,
# citations themselves). GENESIS §12: every artifact carries provenance, and therefore a
# `cited_from`. Ingested artifacts cite their source (cite.<dataset>); system artifacts cite this
# anchor, which is HUMAN_VALIDATED and self-anchored. (`CITE_GENESIS` is single-sourced in
# `prism/grounding.py`, imported at the top of this file.)
# Content types that are system artifacts (authored under CITE_GENESIS, provenance HUMAN_VALIDATED).
_SYSTEM_CTS = {
    "application/vnd.agience.operator+json", "application/vnd.agience.source+json",
    "application/vnd.agience.collection+json", "application/vnd.agience.vtype+json",
    "application/vnd.agience.etype+json", "application/vnd.agience.rung+json",
    "application/x-citation",
}


# ── the seed ontology (§3.1, §3.2, §3.3) ─────────────────────────────────────
# (id-suffix, human name, the content_type(s) it carries, role)
SEED_VTYPES: List[Tuple[str, str, str, str]] = [
    ("Content", "Content", "text/*", "Raw observed material; content in the content store, keyed here"),
    ("Lexeme", "Lexeme", "text/x-wordnet", "A word form — the lexical ground truth"),
    ("Synset", "Synset", "text/x-wordnet", "A word sense and its relations (the IS-A lattice)"),
    ("Concept", "Concept", "application/x-concept", "A ConceptNet/derived concept node (language-neutral)"),
    ("Entity", "Entity", "application/x-entity", "A Wikidata item — named-entity ground truth, interlingual"),
    ("Symbol", "Symbol", "text/x-python-symbol", "A code/math unit — the keyed sub-artifact of a file"),
    ("Operator", "Operator", "application/vnd.agience.operator+json", "A morphism/observer (describe/transform/source/fetch/consolidate)"),
    ("Source", "Source", "application/vnd.agience.source+json", "A coalgebra descriptor (a watched folder, a dataset stream, a URL set)"),
    ("Collection", "Collection", COLLECTION_CONTENT_TYPE, "A named working set (a subject / curriculum stage)"),
    ("Citation", "Citation", "application/x-citation", "A provenance anchor: the dataset/paper/site a body of artifacts came from"),
    # §2.1: a person record IS an artifact — no identity table, no special-cased principal type.
    # `created_by` is a vertex reference to one of these. Universal fields only (identity, public
    # key, issuer binding); trust/reputation are per-observer and live in private state (§7.4).
    ("Person", "Person", PERSON_CONTENT_TYPE, "A human — the WHO that `created_by` references; id is uuid5(issuer, sub)"),
]

# (id-suffix, from->to, meaning). Directional, append-only; each edge carries a `via`
# operator id + provenance rung when written.
SEED_ETYPES: List[Tuple[str, str, str]] = [
    ("describes", "operator -> content", "this operator produced this artifact's context (the triple)"),
    ("via", "artifact -> operator", "the operator edge of the triple (fitness credit flows here)"),
    ("cited_from", "artifact -> citation/source", "provenance: where this came from (mandatory, universal)"),
    ("member_of", "artifact -> collection", "belongs to a curriculum stage / subject (transitive up the tree)"),
    ("sub_collection_of", "collection -> collection", "subject hierarchy (collections form a DAG, unbounded depth)"),
    ("calls", "symbol -> symbol", "code call graph (list-field index)"),
    ("hypernym", "synset -> synset", "WordNet IS-A (broader)"),
    ("hyponym", "synset -> synset", "WordNet IS-A (narrower)"),
    ("synonym", "lexeme -> lexeme", "lexical relation"),
    ("antonym", "lexeme -> lexeme", "lexical relation"),
    ("instance_of", "entity -> entity", "Wikidata/ConceptNet taxonomy"),
    ("subclass_of", "entity -> entity", "Wikidata/ConceptNet taxonomy"),
    ("defines", "doc/symbol -> symbol", "cross-reference"),
    ("references", "doc/symbol -> symbol", "cross-reference"),
    ("consolidates", "canonical -> member", "the compression edge — member is RECONSTRUCTIBLE (near-dup / generative); archived + darkened; credits ρ"),
    ("aligns", "concept -> member", "semantic index — same concept, DISTINCT content across sources (synset↔article); member stays active; does NOT credit ρ"),
    ("supersedes", "new -> old", "temporal replacement (freshness)"),
    ("composed_of", "operator -> operator", "category: this morphism = composition of these generators"),
    ("observed_by", "artifact -> operator", "generic observation (the universal edge)"),
    ("access_event", "principal -> artifact", "append-only audit of every read/authz decision"),
]

# The coupling a semantic relation carries lives in `crystal/ontology/coupling.py`, a stable
# grounding surface the conversation tekton reaches; the op table does not define it. Its `sign` is
# the declared semantics of the relation type (measurement over real antonym edges settled that
# endpoint geometry cannot supply it), and its `names` are readable off the lexicon — see that
# module. [[never-impose-knowledge-derive-it]]

# GENESIS §3.3 rungs mapped to the canonical prism.mass band. (genesis_name, mass_value, meaning)
SEED_RUNGS: List[Tuple[str, str, str]] = [
    ("OBSERVED", P_OBSERVED, "ingested with provenance (path+hash); an instrument/system of record saw it"),
    ("FETCHED", P_SPAN_CITED, "a read-only GET brought it; provenance is the fetched document/URL"),
    ("DERIVED", P_HYPOTHESIS, "a verified operator produced it; re-rungs upward when it earns a citation / passes the gate"),
    ("ASSERTED", P_ASSERTION, "a caller merely claimed it — FORBIDDEN to mint content; the server forces a lower rung"),
]

# Collections (§3.4/§5). (id, name, kind, stage_index, parent_id, description)
ONTOLOGY = "ontology"
SEED_COLLECTIONS: List[Tuple[str, str, str, Optional[int], Optional[str], str]] = [
    (UNIVERSE, "Universe", "root", None, None, "The root of the stage/subject DAG; global metrics aggregate here"),
    (ONTOLOGY, "Ontology", "ontology", None, UNIVERSE, "The self-describing type/edge/rung registries live here"),
    ("stage.0.lexicon", "Stage 0 — Lexicon & relations", "stage", 0, UNIVERSE, "The keyed spine: WordNet/ConceptNet lexemes, synsets, concepts"),
    ("stage.1.grammar", "Stage 1 — Grammar & simple world", "stage", 1, UNIVERSE, "Simple, clean, correct prose; first grammar/collocation stats"),
    ("stage.2.world", "Stage 2 — Concepts & entities", "stage", 2, UNIVERSE, "Wikipedia/Wikidata: the world's furniture and its category lattice"),
    ("stage.3.reason", "Stage 3 — Reasoning & verification", "stage", 3, UNIVERSE, "Lean/math/code with a ground-truth checker — the heart; generate-under-verification"),
    ("stage.4.self", "Stage 4 — Self & metacognition", "stage", 4, UNIVERSE, "The system's own source, operator catalog, fitness record — the self collection"),
    ("stage.5.pragmatics", "Stage 5 — Instructions & pragmatics", "stage", 5, UNIVERSE, "Instruction/response pairs and reasoning traces, admitted last"),
    ("staging", "Staging", "staging", None, UNIVERSE, "Dark-matter parking: data enters here, is described + verified, then promoted"),
    ("subjects", "Subjects", "root", None, UNIVERSE, "Root of the demand-carved subject DAG (subject.math ⊃ subject.math.topology ⊃ …)"),
    ("sources", "Sources", "root", None, UNIVERSE, "Root of the per-source collections (one collection per ingested source)"),
]








def bootstrap(store, *, author: str = DEFAULT_AUTHOR) -> Dict[str, int]:
    """Freeze the seed schema + collection scaffolding into the (clean) store. Idempotent.

    Returns a summary count of what was minted. Run this once on a fresh store before any
    Stage-0 ingest; re-running is harmless (upserts + edge-dedupe)."""
    summary = {"vtypes": 0, "etypes": 0, "rungs": 0, "collections": 0, "edges": 0}

    # 0a. The author, as an artifact — a person is an artifact (§2.1) and `created_by` below is a
    # vertex reference to it. It comes first because every row minted after this point cites
    # `author`, so the who has to exist before the first citation of it. `author` is the identity
    # claim on the way in and the person's vertex id from here down.
    author = _author_ref(store, author)

    # 0. the root provenance anchor — every system artifact cites this (§12). Self-anchored.
    _mint(store, {
        "id": CITE_GENESIS,
        "content_type": CITATION_CONTENT_TYPE,
        "context": {"dataset": "GENESIS (system-authored)", "provider": "Ember program (John + agent)",
                    "kind": "citation", "role": "provenance anchor for the ontology, operators, "
                    "collections, and citations themselves", "provenance": P_HUMAN},
        "lemmas": ["genesis", "system", "provenance"],
        "content": "GENESIS system provenance anchor — the ontology/operators/collections are "
                   "authored deliberately under this citation (HUMAN_VALIDATED).",
        "provenance": P_HUMAN, "cited_from": CITE_GENESIS, "created_by": author,
    })

    # 1. vertex-type registry
    for vid, name, ct, role in SEED_VTYPES:
        _mint(store, {
            "id": f"vtype.{vid}",
            "content_type": VTYPE_CONTENT_TYPE,
            "context": {"name": name, "carries_content_type": ct, "role": role,
                        "kind": "vertex-type", "seed": True, "provenance": P_HUMAN},
            "lemmas": [name.lower(), vid.lower()],
            "content": f"vertex type {name}: {role}",
            "provenance": P_HUMAN,
            "created_by": author,
        })
        summary["vtypes"] += 1

    # 2. edge-type registry
    for eid, direction, meaning in SEED_ETYPES:
        _mint(store, {
            "id": f"etype.{eid}",
            "content_type": ETYPE_CONTENT_TYPE,
            "context": {"name": eid, "direction": direction, "meaning": meaning,
                        "kind": "edge-type", "seed": True, "provenance": P_HUMAN},
            "lemmas": [eid.lower()],
            "content": f"edge type {eid} ({direction}): {meaning}",
            "provenance": P_HUMAN,
            "created_by": author,
        })
        summary["etypes"] += 1

    # 3. provenance-rung registry
    for gname, mass_value, meaning in SEED_RUNGS:
        _mint(store, {
            "id": f"rung.{gname.lower()}",
            "content_type": RUNG_CONTENT_TYPE,
            "context": {"genesis_name": gname, "mass_value": mass_value, "meaning": meaning,
                        "kind": "provenance-rung", "seed": True, "provenance": P_HUMAN},
            "lemmas": [gname.lower(), mass_value],
            "content": f"provenance rung {gname} (mass={mass_value}): {meaning}",
            "provenance": P_HUMAN,
            "created_by": author,
        })
        summary["rungs"] += 1

    # 4. collection scaffolding
    for cid, name, kind, stage_index, parent, desc in SEED_COLLECTIONS:
        ctx: Dict[str, Any] = {"name": name, "kind": kind, "description": desc,
                               "provenance": P_HUMAN, "origin": "genesis-p0",
                               "metrics": {"artifacts": 0, "dark_matter": 0, "rho": None}}
        if stage_index is not None:
            ctx["stage"] = stage_index
        # Preserve runtime state across restarts: re-minting the seed carries a stage's promoted
        # flag, shard checkpoints and accrued metrics forward, as `preserve_fitness` does for
        # operators.
        ex = store.artifacts.get_artifact(cid)
        if ex and isinstance(ex.get("context"), dict):
            for f in ("promoted", "promoted_at", "shards_done", "metrics"):
                if f in ex["context"]:
                    ctx[f] = ex["context"][f]
        _mint(store, {
            "id": cid,
            "content_type": COLLECTION_CONTENT_TYPE,
            "name": name,
            "context": ctx,
            "lemmas": [w for w in name.lower().replace("—", " ").split() if len(w) > 2],
            "content": "",
            "provenance": P_HUMAN,
            "created_by": author,
        })
        summary["collections"] += 1

    # collection DAG edges (sub_collection_of) — do AFTER all collections exist
    for cid, name, kind, stage_index, parent, desc in SEED_COLLECTIONS:
        if parent and _ensure_edge(store, cid, parent, "sub_collection_of"):
            summary["edges"] += 1

    # ontology artifacts are member_of the ontology collection (keeps corpus metrics clean)
    for prefix in ("vtype.", "etype.", "rung."):
        for art in store.artifacts.list_artifacts():
            if art["id"].startswith(prefix):
                if _ensure_edge(store, art["id"], ONTOLOGY, "member_of"):
                    summary["edges"] += 1

    # Grant backfill: mint the owner Read grant for any collection marked private by a flag but
    # holding no grant yet, so access — which is grants alone — covers it. Idempotent; runs before
    # any read is served. See `access.backfill_grants_from_flags`.
    try:
        from mantle.db import access
        summary["grants_backfilled"] = access.backfill_grants_from_flags(store).get("minted", 0)
    except Exception:
        summary["grants_backfilled"] = 0

    return summary


def is_bootstrapped(store) -> bool:
    """True if the seed schema has been laid down (the universe root exists)."""
    return store.artifacts.get_artifact(UNIVERSE) is not None


# ── Stage-0 ingest — Lexicon & relations (§5) ────────────────────────────────
# The keyed spine everything else attaches to. Princeton WordNet 3.0 (nltk) is the
# first, fully-verified, hand-built, low-entropy source; it defines the lemma space that
# all later keyed retrieval indexes into. Each source enters as an OPERATOR + a CITATION
# (§6): op.source.wordnet is fitness-selected; cite.wordnet is the provenance anchor.

WORDNET_CONTENT_TYPE = "text/x-wordnet"
SOURCE_CONTENT_TYPE = "application/vnd.agience.source+json"
# One home for the operator content type, and it is `crystal.operator_schema` — the module that
# defines the operator schema, in a package ember already declares and imports at module scope.
# Taking it from there keeps `import ember.genesis` independent of the chorus bundles: the bundles
# are the distribution path for operator code, and a content type is not code, so importing this
# module does not require a sha-verified payload to be present.
from crystal.operator_schema import OPERATOR_CONTENT_TYPE   # noqa: F401  (re-exported below)


def _mint_wordnet_source(store, *, author: str) -> None:
    """Mint the Stage-0 source triple: the citation anchor, the source-operator (fitness-
    selected), and the per-source collection — before any record flows (§6)."""
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    from ember.runtime.runner import evolution
    _mint(store, {
        "id": "cite.wordnet",
        "content_type": CITATION_CONTENT_TYPE,
        "context": {"dataset": "Princeton WordNet 3.0", "provider": "Princeton University",
                    "license": "WordNet 3.0 License (BSD-like, free)", "version": "3.0",
                    "via_package": "nltk", "kind": "citation", "provenance": P_HUMAN},
        "lemmas": ["wordnet", "princeton", "lexicon"],
        "content": "Princeton WordNet 3.0 — the lexical ground truth for Stage 0.",
        "provenance": P_HUMAN, "created_by": author,
    })
    # source-operator: a coalgebra that unfolds the lexicon into observations; accrues fitness.
    store.artifacts.put_artifact(evolution.preserve_fitness(store.artifacts, {
        "id": "op.source.wordnet",
        "content_type": OPERATOR_CONTENT_TYPE,
        "state": "committed",
        "context": "streams Princeton WordNet 3.0 synsets as keyed lexical artifacts "
                   "(Synset/Lexeme) with hypernym/instance_of/antonym/part_of structure",
        "content": "source-operator op.source.wordnet (Stage 0 lexicon coalgebra)",
        "provenance": P_HUMAN, "cited_from": CITE_GENESIS, "created_by": author,
        "created_time": _now(),
    }))
    _mint(store, {
        "id": "source.wordnet",
        "content_type": COLLECTION_CONTENT_TYPE,
        "name": "source: WordNet",
        "context": {"name": "source: WordNet", "kind": "source", "of": "cite.wordnet",
                    "provenance": P_HUMAN, "origin": "genesis-p0",
                    "metrics": {"artifacts": 0, "dark_matter": 0, "rho": None}},
        "lemmas": ["wordnet", "source"],
        "content": "", "provenance": P_HUMAN, "created_by": author,
    })
    _ensure_edge(store, "source.wordnet", "sources", "sub_collection_of")


def _wal_checkpoint() -> None:
    """Collapse the SQLite WAL back into the main DB so a bulk ingest cannot grow it unbounded.

    [[bulk-import-wal-checkpoint]] Measured: an uncheckpointed `force=True` WordNet re-ingest (117k
    synset rewrites) grows `lattice.db-wal` to 18 GB, and every read then scans the whole WAL, so
    chat takes 20-60s. `TRUNCATE` checkpoints all committed frames and resets the WAL to zero.

    It runs on a separate short-lived connection — the store's WAL is shared, so this collapses the
    frames the store just committed — and is fail-soft: a checkpoint that cannot run leaves the
    ingest alone."""
    try:
        import sqlite3 as _sq
        p = os.path.join(os.getenv("EMBER_SQLITE_DIR", "."),
                         os.getenv("EMBER_SQLITE_DB", "lattice.db"))
        c = _sq.connect(p, timeout=120)
        c.execute("PRAGMA busy_timeout=120000")
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()
    except Exception as exc:
        # This checkpoint runs three times per ingest (every N bulk batches, after the synset
        # pass, after the edge pass). Fail-soft: a checkpoint that cannot run must not fail an
        # ingest. It logs rather than passing, because a checkpoint that never runs leaves the
        # 18 GB WAL the docstring measures, and a swallowed failure gives that nowhere to speak.
        _log.warning("WAL checkpoint did not run: %s: %s", type(exc).__name__, exc)


def ingest_stage0_wordnet(store, *, author: str = DEFAULT_AUTHOR,
                          force: bool = False, edge_batch: int = 1000,
                          limit: Optional[int] = None) -> Dict[str, int]:
    """Ingest Princeton WordNet 3.0 as the Stage-0 lexicon. Idempotent on the artifacts
    (upsert); guarded against duplicate edge creation unless `force=True`.

    Each synset -> a Synset artifact keyed by its lemmas, homed in stage.0.lexicon +
    source.wordnet, cited_from cite.wordnet, via op.source.wordnet, provenance OBSERVED.
    Semantic relations become append-only labeled edges between synset artifacts.
    """
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    try:
        from nltk.corpus import wordnet as wn
        wn.all_synsets  # trigger corpus load / LookupError early
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Stage-0 needs nltk + the wordnet corpus: "
            "python -c \"import nltk; nltk.download('wordnet')\""
        ) from e

    if not is_bootstrapped(store):
        bootstrap(store, author=author)

    already = store.artifacts.get_artifact("op.source.wordnet") is not None
    if already and not force:
        # artifacts upsert cleanly, but re-emitting ~100k edges would duplicate them.
        existing = store.artifacts.count(state="committed")
        return {"skipped": 1, "reason_already_ingested": 1, "committed": existing}

    _mint_wordnet_source(store, author=author)
    # The type comes in WITH the rows that need it: a `text/x-wordnet` row is un-renderable without
    # its describer, and a missing type artifact raises nothing (see stage0_sources._ensure_ctype).
    from ember.corpus import stage0_sources as _s0
    _s0._ensure_ctype(store, WORDNET_CONTENT_TYPE, author=author, **_s0._WN_TYPE)

    summary = {"synsets": 0, "edges": 0}
    stored: set = set()

    # pass 1 — synset artifacts (all endpoints must exist before edges are drawn)
    docs: List[Dict[str, Any]] = []
    _bulk_batches = 0                            # checkpoint the WAL every N batches (see _wal_checkpoint)
    for syn in wn.all_synsets():
        if limit is not None and len(stored) >= limit:
            break
        # Observation only, and in the source's order — the same shape `stage0_sources` writes
        # (see its doc-build comment). The source's own lemma order is preserved rather than
        # re-sorted into our alphabet; `content` is the definition itself rather than a rendered
        # card, which would freeze one day's layout into 117,659 artifacts and feed FTS its own
        # scaffolding ("part of speech: n synonyms:") as meaning; and `context` is left to the
        # describer. Context comes via description, and describers evolve, so this writes fields.
        words, _seen = [], set()
        for _l in syn.lemmas():
            _w = _l.name().replace("_", " ")
            if _w and _w not in _seen:
                _seen.add(_w); words.append(_w)
        if not words:
            continue
        aid = "wn-" + syn.name()
        defn = syn.definition() or ""
        # Which sense of each word this synset is — nltk returns a word's synsets in sense order,
        # so the index is the sense number. Recorded because nothing downstream can recover it:
        # a synset name sorts by offset, which says nothing about meaning (see
        # `crystal/ontology/driver.py`).
        _ranks = {}
        for _w in words:
            try:
                _order = [x.name() for x in wn.synsets(_w.replace(" ", "_"), pos=syn.pos())]
                _ranks[_w.lower()] = _order.index(syn.name())
            except Exception:
                pass
        # SemCor sense-frequency — the absolute count of how often each word is used in this sense
        # (`nltk.Lemma.count`, tagged from SemCor). Complementary to `sense_ranks`: ranks order a
        # word's senses among themselves (both `dog`.n.01 and `me`'s `Maine`.n.01 are rank 0), while
        # the count separates them across words — `dog`=42, `Maine`=0 — which is what lets a real
        # topic out-weigh the rare noun reading of a function word ("me", "does", "is"). Without it
        # the seeder surfaces `me->Maine` / `does->Department of Energy`. It is computed here,
        # through the source tekton, so the stat arrives from Ember's own pipeline.
        _counts: Dict[str, int] = {}
        for _l in syn.lemmas():
            _w = _l.name().replace("_", " ").lower()
            if not _w:
                continue
            try:
                _c = int(_l.count())
            except Exception:
                _c = 0
            _counts[_w] = max(_counts.get(_w, 0), _c)
        docs.append({
            "id": aid,
            "content_type": WORDNET_CONTENT_TYPE,
            "state": "committed",
            "title": words[0],                   # the primary lemma — a field, carrying no prose
            "gloss": defn,                       # the definition, clean and unstyled
            "content": defn,                     # the information itself — a bare definition
            "examples": list(syn.examples()[:3]),  # structured, kept as a list
            "lemmas": [w.lower() for w in words],
            "sense_ranks": _ranks,
            "lemma_counts": _counts,             # SemCor absolute sense-frequency (MFS across words)
            "word": words[0].lower(),
            "pos": syn.pos(),
            "collection_id": "stage.0.lexicon",
            "collections": ["stage.0.lexicon", "source.wordnet"],
            "cited_from": "cite.wordnet",
            "via": "op.source.wordnet",
            "operator": "op.source.wordnet",
            "provenance": P_OBSERVED,
            "created_by": author,
            "created_time": _now(),
        })
        stored.add(aid)
        if len(docs) >= 2000:
            summary["synsets"] += store.artifacts.put_many(docs)
            docs = []
            _bulk_batches += 1
            if _bulk_batches % 20 == 0:          # every ~40k rows, collapse the WAL
                _wal_checkpoint()
    if docs:
        summary["synsets"] += store.artifacts.put_many(docs)
    _wal_checkpoint()                            # after the whole synset pass

    # pass 2 — semantic relation edges (only where BOTH endpoints were stored)
    def _edges():
        for syn in wn.all_synsets():
            cid = "wn-" + syn.name()
            if cid not in stored:
                continue
            for hyper in syn.hypernyms():
                tid = "wn-" + hyper.name()
                if tid in stored:
                    yield (cid, tid, "hypernym", {"via": "op.source.wordnet", "rung": P_OBSERVED})
            for inst in syn.instance_hypernyms():
                tid = "wn-" + inst.name()
                if tid in stored:
                    yield (cid, tid, "instance_of", {"via": "op.source.wordnet", "rung": P_OBSERVED})
            for part in syn.part_holonyms():
                tid = "wn-" + part.name()
                if tid in stored:
                    yield (cid, tid, "part_of", {"via": "op.source.wordnet", "rung": P_OBSERVED})
            # `attribute` and `similar` are what let a MODIFIER be placed. Neither an adjective nor
            # an adverb carries a hypernym parent, so neither has a position `jc_tree` can measure;
            # what each has is a noun it is about, and these three labels plus `derivation` and
            # `pertainym` below are how `crystal.ontology.lookup.projected_nouns_for` reaches it.
            # The OEWN path emits the same set — it takes every relation the source names — and the
            # two seeds must agree, or a node seeded from nltk places no adjectives while one seeded
            # from OEWN does, with nothing to say why.
            for attr in syn.attributes():
                tid = "wn-" + attr.name()
                if tid in stored:
                    yield (cid, tid, "attribute", {"via": "op.source.wordnet", "rung": P_OBSERVED})
            for sim in syn.similar_tos():
                tid = "wn-" + sim.name()
                if tid in stored:
                    yield (cid, tid, "similar", {"via": "op.source.wordnet", "rung": P_OBSERVED})
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    tid = "wn-" + ant.synset().name()
                    if tid in stored:
                        yield (cid, tid, "antonym", {"via": "op.source.wordnet", "rung": P_OBSERVED})
                # Sense-level, so each end is lifted to its synset — the same lift `antonym` above
                # has always used, and the same one `parse_oewn_lmf` applies.
                for der in lemma.derivationally_related_forms():
                    tid = "wn-" + der.synset().name()
                    if tid in stored:
                        yield (cid, tid, "derivation",
                               {"via": "op.source.wordnet", "rung": P_OBSERVED})
                for pert in lemma.pertainyms():
                    tid = "wn-" + pert.synset().name()
                    if tid in stored:
                        yield (cid, tid, "pertainym",
                               {"via": "op.source.wordnet", "rung": P_OBSERVED})

    summary["edges"] = store.graph.add_edges(_edges(), batch=edge_batch)
    _wal_checkpoint()                            # collapse the WAL after the edge pass too
    return summary


# ── generic dataset ingest — the reusable contract for Stages 1-5 (§6) ───────



def ingest_dataset(store, source, *, stage: str, cite_meta: Dict[str, Any], offer: str,
                   describe: bool = True, describe_workers: int = 1,
                   author: str = DEFAULT_AUTHOR) -> Dict[str, int]:
    """Ingest a DatasetSource under the full GENESIS contract: mint its source triple, stamp
    every observation with (collection_id=stage, collections=[stage, source.<name>],
    cited_from, via, provenance OBSERVED), stream content into the content store (encrypted,
    content-addressed) with a keyed reference in the artifact store, and illuminate each record at first
    observation. Returns {ingested}. The source-operator accrues fitness for what it yields."""
    from ember.runtime.runner import evolution
    if not is_bootstrapped(store):
        bootstrap(store, author=author)
    name = source.name
    mint_source_triple(store, name, cite_meta=cite_meta, offer=offer, author=author)
    source.stamp = {
        "collection_id": stage,
        "collections": [stage, f"source.{name}"],
        "cited_from": f"cite.{name}",
        "via": f"op.source.{name}",
        "operator": f"op.source.{name}",
        "provenance": P_OBSERVED,
    }
    # Stream the coalgebra in incremental batches, so the whole dataset is never materialized
    # (Stages 2-5 are far too large). Each record is described in memory — lemmas computed from the
    # text already held — and written once, batched: batched writes are ~14x faster than a
    # per-record describe upsert, which accounted for 59% of ingest time. `describe_workers>1`
    # extracts terms across CPU cores in parallel (term extraction is pure CPU).
    from mantle.shard import content as C
    from ember.runtime.runner import describe as D
    n = 0
    batch: List[tuple] = []                          # (doc, content_type, path, full_text)
    pool = None
    if describe and describe_workers and describe_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=describe_workers)

    def _lemmas_for(items: List[tuple]) -> List[List[str]]:
        if not describe:
            return [[] for _ in items]
        jobs = [(ct, path, text) for (_d, ct, path, text) in items]
        if pool is not None:
            return list(pool.map(_terms_star, jobs, chunksize=64))
        return [D.terms_of(ct, path, text) for (ct, path, text) in jobs]

    def _flush() -> None:
        nonlocal n
        if not batch:
            return
        for (doc, ct, _p, t), lem in zip(batch, _lemmas_for(batch)):
            if lem:
                doc["lemmas"] = lem
                doc["via"] = "op.describe.markdown"
        store.artifacts.put_many([b[0] for b in batch], batch=100)   # small txns: less hot-page contention
        n += len(batch)
        batch.clear()

    try:
        for obs in source.poll():
            doc = obs.to_doc()
            full = obs.content
            path = (obs.meta or {}).get("title") or (doc.get("id") or "")
            if store.content is not None and store.keys_dir is not None:
                ref, size = C.put_content(store.content, store.keys_dir, full.encode("utf-8"))
                doc["content_ref"] = ref
                doc["size"] = size
                # An artifact carries a content_ref and its offer (the describe); the content
                # itself lives in the content store. An inline preview would be duplication —
                # ~300 B x 6.15M rows ≈ 1.8 GB per node — and, being a replicated field, would be
                # paid again in every mesh segment on every peer. It would also contradict this
                # file's confidentiality property: content is encrypted into the content store with
                # no cleartext preview in the index. `resolve_text` reads through content_ref, so
                # the full text stays available.
                doc.pop("content", None)
            batch.append((doc, doc.get("content_type", "text/plain"), path, full))
            if len(batch) >= 2000:
                _flush()
        _flush()
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if n:
        evolution.record_invocation(store.artifacts, f"op.source.{name}", verified=True)
    return {"ingested": n, "source": name, "stage": stage}


def _terms_star(job):
    """Top-level (picklable) worker for the describe process pool. Returns lemmas only, keeping
    ingest CPU-light: ingest is the hot path and consolidation is a rare background task."""
    from ember.runtime.runner import describe as D
    ct, path, text = job
    return D.terms_of(ct, path, text)


# ── parquet-shard ingest — resumable, cached, no skip re-download (§6/§9) ───
# For the big stages (full Wikipedia = 41 parquet shards, ~6.4M rows), streaming with.skip(N)
# re-downloads the discarded prefix on every resume. Instead: download each shard once (hf_hub_
# download caches it), read it columnar with pyarrow, and checkpoint per shard. Resume skips whole
# done shards — O(1), no re-download — and shards are independently claimable, so they can be
# processed in parallel.









# `HfApi.list_repo_files` goes through huggingface_hub's shared session, which is built as
# `httpx.Client(..., timeout=None)` — no connect, read, write or pool timeout, no retry, no cap.
# When the TCP flow to huggingface.co is silently dropped (NAT/conntrack eviction, an LB reaping an
# idle connection, a blackholed route) the read blocks indefinitely and raises nothing: every thread
# parks in futex_wait at ~3% CPU with no log line between `claimed` and the next restart.
#
# `socket.setdefaulttimeout` does not cover this: httpx/httpcore set the socket timeout explicitly
# from the client config, so an explicit `None` overrides the process default. The bound has to be
# imposed around the call, which is what `_list_repo_files_cached` does.
#
# The call is also made once per shard for a list identical across all 41 of them, so memoising it
# removes 40 network round-trips per run as well.
_REPO_FILES_CACHE: dict = {}
_REPO_FILES_TIMEOUT_S = float(__import__("os").getenv("EMBER_HF_LIST_TIMEOUT_S", "120"))


def _list_repo_files_cached(repo: str) -> List[str]:
    """`HfApi.list_repo_files`, memoised per process and bounded by a wall clock.

    Raises TimeoutError once the wall clock is exceeded. The caller (the ingest loop) treats that as
    a failed shard, leaves its claim to expire, and a peer picks the work up — the recovery the
    lease design already assumes, and the one an indefinite block would deny it."""
    import threading
    if repo in _REPO_FILES_CACHE:
        return _REPO_FILES_CACHE[repo]
    from huggingface_hub import HfApi
    box: dict = {}

    def _fetch():
        try:
            box["files"] = list(HfApi().list_repo_files(repo, repo_type="dataset"))
        except BaseException as exc:                       # noqa: BLE001 — re-raised below
            box["exc"] = exc

    # Daemon thread: a wedged socket leaves this thread unkillable, and a daemon thread cannot hold
    # the process open. It leaks one parked thread per timeout, which is why the ingest loop
    # restarts the process rather than retrying in place.
    t = threading.Thread(target=_fetch, daemon=True, name="hf-list-repo-files")
    t.start()
    t.join(_REPO_FILES_TIMEOUT_S)
    if t.is_alive():
        raise TimeoutError(
            "huggingface list_repo_files(%s) exceeded %ss — the hub session has no timeout of its "
            "own, so this is the guard." % (repo, _REPO_FILES_TIMEOUT_S))
    if "exc" in box:
        raise box["exc"]
    _REPO_FILES_CACHE[repo] = box["files"]
    return box["files"]


def ingest_sharded(store, *, repo: str, config: str, name: str, stage: str,
                   cite_meta: Dict[str, Any], offer: str, to_record,
                   columns: Optional[List[str]] = None, describe: bool = True,
                   describe_workers: int = 1, max_shards: Optional[int] = None,
                   only_shards: Optional[List[str]] = None,
                   max_rows: Optional[int] = None,
                   row_batch: int = 2000, author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """Resumable parquet-shard ingest under the full GENESIS contract. Downloads each shard once
    (cached), reads it columnar, describes in-memory (parallel), writes once per batch, and
    checkpoints the shard as done. Returns {ingested, shards_done, shards_total}."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from mantle.shard import content as C
    from ember.runtime.runner import evolution   # re-exported by ember/runtime/runner.py (PEP 562)

    if not is_bootstrapped(store):
        bootstrap(store, author=author)
    mint_source_triple(store, name, cite_meta=cite_meta, offer=offer, author=author)
    stamp = {"collection_id": stage, "collections": [stage, f"source.{name}"],
             "cited_from": f"cite.{name}", "via": f"op.source.{name}",
             "operator": f"op.source.{name}", "provenance": P_OBSERVED}

    files = sorted(f for f in _list_repo_files_cached(repo)
                   if f.startswith(config + "/") and f.endswith(".parquet"))
    done = set(_shards_done(store, name))
    todo = [f for f in files if f not in done]
    if only_shards is not None:            # a task targets ONE (or a few) specific shards
        want = set(only_shards)
        todo = [f for f in todo if f in want]
    if max_shards is not None:
        todo = todo[:max_shards]

    pool = None
    if describe and describe_workers and describe_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=describe_workers)
    n = 0
    try:
        for shard in todo:
            local = hf_hub_download(repo, shard, repo_type="dataset")   # cached across runs
            pf = pq.ParquetFile(local)
            cols = columns or None
            batch: List[tuple] = []

            def _flush():
                nonlocal n
                if not batch:
                    return
                if describe:
                    jobs = [(ct, p, t) for (_d, ct, p, t) in batch]
                    lem = (list(pool.map(_terms_star, jobs, chunksize=64)) if pool
                           else [_terms_star(j) for j in jobs])
                    for (doc, _c, _p, _t), l in zip(batch, lem):
                        if l:
                            doc["lemmas"] = l
                            doc["via"] = "op.describe.markdown"
                store.artifacts.put_many([b[0] for b in batch], batch=500)
                n += len(batch)
                batch.clear()

            stop = False
            for rb in pf.iter_batches(batch_size=row_batch, columns=cols):
                for i, row in enumerate(rb.to_pylist()):
                    if max_rows is not None and n + len(batch) >= max_rows:
                        stop = True
                        break
                    rec = to_record(row, i)
                    if not rec:
                        continue
                    body = rec.get("content") or ""
                    if not body.strip():
                        continue
                    doc = {"id": rec["id"], "content_type": rec.get("content_type", "text/plain"),
                           "state": "committed", "context": "", "created_by": "ember-source",
                           "title": (rec.get("meta") or {}).get("title", ""), **stamp}
                    if store.content is not None and store.keys_dir is not None:
                        ref, size = C.put_content(store.content, store.keys_dir, body.encode("utf-8"))
                        # content_ref ONLY — never the body, never a preview. See the note at the
                        # other offload site: the preview was ~1.8 GB/node of duplication, re-paid
                        # in every mesh segment on every peer.
                        doc["content_ref"] = ref; doc["size"] = size
                    else:
                        # No content store configured — the body has nowhere else to live, so it
                        # stays inline. This is the ONLY branch that may hold content.
                        doc["content"] = body
                    path = (rec.get("meta") or {}).get("title") or rec["id"]
                    batch.append((doc, doc["content_type"], path, body))
                    if len(batch) >= row_batch:
                        _flush()
                _flush()
                if stop:
                    break
            if stop:
                break                          # max_rows hit mid-shard: do NOT checkpoint it
            _mark_shard_done(store, name, shard)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    if n:
        evolution.record_invocation(store.artifacts, f"op.source.{name}", verified=True)
    return {"ingested": n, "shards_done": len(done) + len(todo), "shards_total": len(files)}


# ── Stage-1 — Grammar & simple world ("sentences, easy readers", §5) ─────────
# Simple English Wikipedia: clean, correct, high-signal prose on the lexicon scaffolding.
# The extracted-text config is already plain prose (not wikitext), so no heavy cleaning.

# A chosen number, flagged as one. Below this many characters a Simple-Wikipedia row is taken to be
# a stub or a redirect and is not ingested. It is chosen rather than derived, and derivation is out
# of reach here: the ingester sees one row at a time and has no distribution to read a floor off,
# and the only structural signal available instead — MediaWiki's `#REDIRECT` marker — separates
# redirects from stubs but says nothing about stubs, which are the larger half.
#
# It is stated once rather than buried in the row filter because it shapes the corpus: everything
# below it never becomes an artifact, so it is out of reach of any later measurement.
#
# For a different value to be right, someone would have to measure the length distribution of one
# shard and find where prose separates from redirect boilerplate — at which point this becomes a
# quantile of that distribution and stops being a choice.
_WIKI_STUB_CHARS = 120


def ingest_stage1_simplewiki(store, *, max_shards: Optional[int] = None, describe: bool = True,
                             describe_workers: int = 1,
                             author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """Simple English Wikipedia via the resumable parquet-shard path (1 shard) — checkpointed, so
    once processed it is never re-ingested. Clean prose on the lexicon scaffolding."""
    def _to_record(row, i):
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not text or len(text) < _WIKI_STUB_CHARS:
            return None
        body = f"{title}\n\n{text}" if title else text
        return {"id": "wiki-simple-" + str(row.get("id", i)), "content": body,
                "content_type": "text/markdown", "meta": {"title": title}}

    return ingest_sharded(
        store, repo="wikimedia/wikipedia", config="20231101.simple", name="wikipedia-simple",
        stage="stage.1.grammar",
        cite_meta={"dataset": "Simple English Wikipedia (20231101.simple)",
                   "provider": "Wikimedia", "license": "CC BY-SA 4.0 / GFDL",
                   "hf_path": "wikimedia/wikipedia", "config": "20231101.simple"},
        offer="ingests Simple English Wikipedia (parquet shard) as clean prose Content artifacts "
              "(Stage-1 grammar & simple-world scaffolding)",
        to_record=_to_record, columns=["id", "url", "title", "text"],
        describe=describe, describe_workers=describe_workers, max_shards=max_shards, author=author)


# ── Stage-2 — Concepts & entities ("the world's furniture", §5) ──────────────
# Full English Wikipedia: concrete world knowledge + a category lattice — the scaffolding onto
# which reasoning domains hang. Same coalgebra as Stage-1, the large config (~6.4M articles),
# ingested through the resumable parquet-shard path so a resume re-downloads nothing.

def ingest_stage2_wikipedia(store, *, max_shards: Optional[int] = None,
                            only_shards: Optional[List[str]] = None,
                            describe: bool = True, describe_workers: int = 1,
                            author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """Full English Wikipedia via the resumable parquet-shard path (41 shards). Each worker tick
    processes `max_shards` shards and checkpoints them; resume skips done shards (no re-download)."""
    def _to_record(row, i):
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not text or len(text) < 200:            # world facts: a higher signal floor than simple
            return None
        body = f"{title}\n\n{text}" if title else text
        return {"id": "wiki-en-" + str(row.get("id", i)), "content": body,
                "content_type": "text/markdown", "meta": {"title": title}}

    return ingest_sharded(
        store, repo="wikimedia/wikipedia", config="20231101.en", name="wikipedia-en",
        stage="stage.2.world",
        cite_meta={"dataset": "English Wikipedia (20231101.en)", "provider": "Wikimedia",
                   "license": "CC BY-SA 4.0 / GFDL", "hf_path": "wikimedia/wikipedia",
                   "config": "20231101.en"},
        offer="ingests full English Wikipedia (parquet shards) as Content artifacts — concrete "
              "world knowledge + the category lattice (Stage-2 concepts & entities)",
        to_record=_to_record, columns=["id", "url", "title", "text"],
        describe=describe, describe_workers=describe_workers, max_shards=max_shards,
        only_shards=only_shards, author=author)


def list_shards(repo: str = "wikimedia/wikipedia", config: str = "20231101.en") -> List[str]:
    """The parquet shards of a dataset config (cached by hf). Used by the scheduler to enqueue
    one ingest task per shard so the pool ingests shards in parallel.

    Reads through `_list_repo_files_cached`, so the hub call is memoised and wall-clock bounded.
    The bound matters most here: the ingest loop calls this from `pick_shard`, before anything is
    claimed, so an unbounded call would produce neither a `survey` line nor a `claimed` line and
    the loop would go silent with nothing in the log."""
    return sorted(f for f in _list_repo_files_cached(repo)
                  if f.startswith(config + "/") and f.endswith(".parquet"))


# ── the curriculum pump — how EMBER (worker.py) self-drives ingestion (§5) ───
# The worker calls advance_curriculum each tick. It finds the first not-yet-promoted stage,
# ingests the NEXT bounded increment (resumable via skip=already-have), and PROMOTES the stage
# when its exit target is met (§5: a stage is promoted before the next admits high-entropy
# material). New stages are registered here as their source specs are built — the worker picks
# them up automatically, so "Ember takes over ingestion" is: this table + the worker loop.

def _count_in(store, collection_id: str) -> int:
    return sum(1 for _ in store.artifacts.list_artifacts(collection_id=collection_id))


def _promoted(store, collection_id: str) -> bool:
    c = store.artifacts.get_artifact(collection_id)
    return bool(c and isinstance(c.get("context"), dict) and c["context"].get("promoted"))


def _mark_promoted(store, collection_id: str) -> None:
    c = store.artifacts.get_artifact(collection_id)
    if c and isinstance(c.get("context"), dict):
        c["context"]["promoted"] = True
        c["context"]["promoted_at"] = _now()
        store.artifacts.put_artifact(c)


# ordered curriculum. Each stage: collection id, an ingest(store, skip, limit) callable, the
# promotion `target` (artifact count), and the per-tick increment. Grows as stages are built.
_DESCRIBE_WORKERS = max(1, (__import__("os").cpu_count() or 2) - 2)   # leave 2 cores for I/O


def _stage0():
    """The stage-0 source module, imported lazily — CURRICULUM/SOURCE_INGESTERS entries must
    not import it (and transitively its parsers) until a stage-0 ingest actually runs."""
    from ember.corpus import stage0_sources
    return stage0_sources


CURRICULUM: List[Dict[str, Any]] = [
    # ── Stage 0 — lexicon (runbook §H1; TRAINING-QUEUE 0.1–0.5). Five sources share one
    # collection (stage.0.lexicon), so these specs carry two things the later stages don't:
    #   `done`   — per-source completion (the ingester's own checkpoint marker), because a
    #              shared collection count cannot say which source is finished; exit is
    #              exhaustion (the source drains and writes its marker), never the count;
    #   `target` — the source's expected cumulative yield, for /status progress display only.
    #              A reading rather than a cap: promotion depends on the marker, not on it
    #              (no-arbitrary-caps — the real bound is the source's own extent).
    # The collection is promoted once every stage-0 spec reports done (see
    # advance_curriculum / _stage_done). Sequencing within the stage is this list's order:
    # wordnet spine → OEWN rows (carry the ili pivot keys) → CILI (draws the pivot edges,
    # needs both) → ConceptNet → OMW (attaches vocabulary to the pivots).
    {"stage": "stage.0.lexicon", "source": "wordnet",
     "ingest": lambda s, skip, limit: ingest_stage0_wordnet(s),
     # the ingester's own established guard: the source-operator artifact exists once minted.
     "done": lambda s: s.artifacts.get_artifact("op.source.wordnet") is not None,
     "target": 117_659, "per_tick": 0},        # PWN 3.0's own size — 117,659 synsets
    {"stage": "stage.0.lexicon", "source": "oewn",
     "ingest": lambda s, skip, limit: _stage0().ingest_stage0_oewn(s),
     "done": lambda s: _stage0().oewn_done(s),
     "target": 238_000, "per_tick": 0},        # + ~120k OEWN-2024 synsets
    {"stage": "stage.0.lexicon", "source": "cili",
     "ingest": lambda s, skip, limit: _stage0().ingest_stage0_cili(s),
     "done": lambda s: _stage0().cili_done(s),
     "target": 238_000, "per_tick": 0},        # pivot EDGES only — adds no vertices
    {"stage": "stage.0.lexicon", "source": "conceptnet",
     # max_lines is the tick size (the bounded increment, like stage-2's one-shard tick)
     # rather than a cap — the line cursor carries the remainder into the next tick.
     "ingest": lambda s, skip, limit: _stage0().ingest_stage0_conceptnet(s, max_lines=1_000_000),
     "done": lambda s: _stage0().conceptnet_done(s),
     "target": 1_750_000, "per_tick": 0},      # + ~1.5M /c/en/ concept nodes
    {"stage": "stage.0.lexicon", "source": "omw",
     "ingest": lambda s, skip, limit: _stage0().ingest_stage0_omw(s),
     "done": lambda s: _stage0().omw_done(s),
     "target": 2_000_000, "per_tick": 0},      # allowlisted languages' vocabulary
    # Stages 1-2 are shard-driven: one checkpointed parquet shard per tick, so `per_tick` is unused
    # here and a resume re-downloads nothing.
    {"stage": "stage.1.grammar",
     "ingest": lambda s, skip, limit: ingest_stage1_simplewiki(
         s, max_shards=1, describe_workers=_DESCRIBE_WORKERS),
     "target": 40000, "per_tick": 0},   # 1 shard, checkpointed; promotes on shards-complete
    {"stage": "stage.2.world",
     # sharded: one parquet shard per tick (~156k rows), checkpointed + resumable (no re-download)
     "ingest": lambda s, skip, limit: ingest_stage2_wikipedia(
         s, max_shards=1, describe_workers=_DESCRIBE_WORKERS),
     "target": 300000, "per_tick": 0},   # per_tick unused (shard-driven); promotes at target
    # stage.3.reason, … appended as their source specs are built.
]


def _stage_done(store, col: str) -> bool:
    """True iff every spec of collection `col` carries a done-predicate and all report done.
    This is what lets several sources share one collection (stage 0): the collection reads
    `promoted` only once every source it holds is drained. A collection with any spec that has
    no done-predicate answers False here and promotes through the count path."""
    specs = [sp for sp in CURRICULUM if sp["stage"] == col]
    return bool(specs) and all(sp.get("done") is not None and sp["done"](store) for sp in specs)


def advance_curriculum(store, *, per_tick: Optional[int] = None,
                       author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """One autonomous ingestion step. Returns what it did. Idempotent + resumable: it resumes a
    stage from its current artifact count and promotes on target. When every registered stage is
    promoted it returns {curriculum: 'complete'} (nothing to do until a new stage is registered).

    Specs carrying a `done` predicate (the stage-0 sources) exit on exhaustion — the
    ingester's own completion marker — rather than on the count: several sources share one
    collection there, so a shared count can neither resume nor finish any single source."""
    if not is_bootstrapped(store):
        bootstrap(store, author=author)
    # A source that cannot be fetched right now — a down host (OEWN's en-word.net 503), a pending
    # license clearance (OMW), a transient network fault — defers, so the stage keeps moving.
    # advance runs the first not-done spec each tick, so deferral is what keeps one unreachable
    # source from standing in front of every sibling behind it (ConceptNet, stage 1, stage 2). On
    # an ingest exception the deferral is recorded and the loop continues: the source stays
    # not-done and is retried next tick, while a later, available spec makes progress now.
    deferred: List[Dict[str, Any]] = []
    for spec in CURRICULUM:
        col = spec["stage"]
        donef = spec.get("done")
        if donef is not None:
            if donef(store):
                # this source is drained + checkpointed; the collection promotes once every
                # sibling spec is too, since a shared collection reads promoted only when all
                # of its sources are drained.
                if _stage_done(store, col) and not _promoted(store, col):
                    _mark_promoted(store, col)
                continue
            have = _count_collection(store, col, committed_only=False)
            try:
                r = spec["ingest"](store, have, spec["per_tick"] if per_tick is None else per_tick)
            except Exception as e:
                deferred.append({"stage": col, "source": spec.get("source"),
                                 "error": str(e)[:200]})
                continue
            promote = bool(donef(store))           # exit is the marker, never the count
            if promote and _stage_done(store, col) and not _promoted(store, col):
                _mark_promoted(store, col)
            return {"stage": col, "source": spec.get("source"), "had": have,
                    "ingested": r.get("ingested", 0),
                    "now": _count_collection(store, col, committed_only=False),
                    "target": spec["target"], "exhausted": promote,
                    # `promoted` is the collection's state — for a shared collection that
                    # is strictly later than this one source's exhaustion.
                    "promoted": _promoted(store, col)}
        # `_count_collection` is the indexed count. `advance_curriculum` runs every tick under
        # `--ingest` and reads the count twice, so on a multi-million-row stage the streaming form
        # (`_count_in`) would be two full scans per tick.
        # `committed_only=False` is required, not incidental: `have` is the resume offset passed to
        # `spec["ingest"](store, have, n)` below, so it counts every state exactly as `_count_in`
        # does. Anything smaller re-ingests records we already hold.
        have = _count_collection(store, col, committed_only=False)
        if have >= spec["target"]:
            if not _promoted(store, col):
                _mark_promoted(store, col)
            continue
        n = spec["per_tick"] if per_tick is None else per_tick
        # resume from `have`: ingest the next `n` records of this stage
        try:
            r = spec["ingest"](store, have, n)
        except Exception as e:
            deferred.append({"stage": col, "source": spec.get("source"), "error": str(e)[:200]})
            continue
        # Same call, same semantics as `have` above — `drained = (now <= have)` compares the two,
        # so they must count the same thing or the exhaustion test silently changes meaning.
        now = _count_collection(store, col, committed_only=False)
        # Promote when the target is met or the source is exhausted (a tick added no net-new
        # artifacts). Exhaustion matters because a source can hold fewer items than the target
        # (e.g. Simple Wikipedia ~41k against a 60k target); it is what lets the worker advance
        # instead of re-ingesting duplicates. `shards_done`/`shards_total` also signal drained.
        drained = (now <= have)
        sh = r.get("shards_done"), r.get("shards_total")
        shards_complete = sh[0] is not None and sh[1] is not None and sh[0] >= sh[1]
        promote = now >= spec["target"] or drained or shards_complete
        if promote:
            _mark_promoted(store, col)
        return {"stage": col, "had": have, "ingested": r.get("ingested", 0),
                "now": now, "target": spec["target"], "exhausted": drained,
                "shards": {"done": sh[0], "total": sh[1]} if sh[0] is not None else None,
                "promoted": promote}
    # Reached the end with nothing left to ingest. If some sources deferred (unreachable this
    # pass), the curriculum is blocked on them rather than complete, and the answer names which,
    # so a down host or a missing license stays visible and is retried.
    if deferred:
        return {"curriculum": "blocked", "deferred": deferred}
    return {"curriculum": "complete"}


# ── the operator control plane — drive EVERYTHING by invoking operators (§4) ─
# The thesis: an operator is an artifact, and the universe is acted on by invoking operators
# matched to a need. This is that surface, exposed over the channel
# (`ember/surface/serve.py`, POST /v1/invoke). Ingestion, consolidation and status are all
# operators here — the answer is computed geometrically, through operators.

# op.source.<name> -> the ingester that materializes that source (a bounded increment).
# `args` may carry skip/limit; each returns {ingested,...}.
SOURCE_INGESTERS = {
    "op.source.wordnet": lambda store, a: ingest_stage0_wordnet(
        store, force=bool(a.get("force", False)), limit=a.get("limit")),
    # ── stage-0 completion (runbook §H1; stage0_sources.py) ──
    "op.source.oewn": lambda store, a: _stage0().ingest_stage0_oewn(
        store, force=bool(a.get("force", False)), limit=a.get("limit"),
        path=a.get("path")),
    "op.source.cili": lambda store, a: _stage0().ingest_stage0_cili(
        store, force=bool(a.get("force", False)), limit=a.get("limit"),
        path=a.get("path")),
    "op.source.conceptnet": lambda store, a: _stage0().ingest_stage0_conceptnet(
        store, force=bool(a.get("force", False)), path=a.get("path"),
        max_lines=a.get("max_lines"),
        batch_lines=int(a.get("batch_lines", 20000))),
    "op.source.omw": lambda store, a: _stage0().ingest_stage0_omw(
        store, force=bool(a.get("force", False)), limit=a.get("limit"),
        projects=a.get("projects")),
    "op.source.wikipedia-simple": lambda store, a: ingest_stage1_simplewiki(
        store, max_shards=a.get("max_shards", 1),
        describe_workers=int(a.get("describe_workers", _DESCRIBE_WORKERS))),
    "op.source.wikipedia-en": lambda store, a: ingest_stage2_wikipedia(
        store, max_shards=(None if a.get("shards") else a.get("max_shards", 1)),
        only_shards=a.get("shards"),
        describe_workers=int(a.get("describe_workers", _DESCRIBE_WORKERS))),
}

# meta-operators (status / curriculum) that are invokable like any other.
CONTROL_OPS = [
    ("op.status.universe", "reports the geometric status of the universe: ρ, coverage, "
     "per-collection metrics, and curriculum position — computed, not narrated"),
    ("op.health", "monitoring + health: worker liveness (heartbeat age), error rate, ρ, the "
     "provenance invariant, and curriculum progress — 'is the system alive & well'"),
    ("op.consistency", "checks & balances: the physical/logical invariants the universe must obey "
     "(ρ∈[0,1], generators≤corpus, mass never vanishes, fitness∈[0,1], provenance) — anomalies to "
     "investigate"),
    ("op.curriculum.advance", "advances the developmental curriculum by one bounded increment "
     "(ingest the next stage's records, or promote it) — Ember driving its own ingestion"),
    ("op.mesh.status", "the write-scaling mesh view: aggregates THIS shard + every peer shard "
     "(EMBER_PEERS) into one universe — total artifacts, per-shard reachability, ρ. The two "
     "boxes each hold a disjoint half; this is how they read as one"),
    ("op.mesh.pull", "AUTHORITATIVE replication: pull every peer's artifacts (with full content) "
     "INTO this store so the local universe contains everything. Bounded + cursor-resumable; run "
     "repeatedly to drain the peers. This is how D: becomes the single source of truth"),
    ("op.content.promote", "async content tiering: copy this box's local content-cache ciphertext UP "
     "to the OVH origin (content.agience.ai) so the durable shared store holds every blob. Content-"
     "addressed → idempotent (skips refs already there); bounded + cursor-resumable"),
    ("op.source.wordnet", "ingest/refresh the Stage-0 WordNet lexicon"),
    ("op.source.oewn", "ingest Open English WordNet 2024 as Stage-0 synset rows (wn-row shape, "
     "ILI-carrying) beside the WordNet 3.0 spine — via GET of the published LMF"),
    ("op.source.cili", "ingest the CILI interlingual index: ILI ids become typed `ili` pivot "
     "edges between synsets expressing the same concept — edges only, never new vertices"),
    ("op.source.conceptnet", "stream the ConceptNet 5.7 assertions dump (English), landing "
     "Concept vertices + typed relation edges, cited; line-cursor resumable"),
    ("op.source.omw", "ingest Open Multilingual Wordnet vocabulary for LICENSE-VETTED languages "
     "only (explicit allowlist; every skipped language is logged for QUEUE clearance)"),
    ("op.source.wikipedia-simple", "ingest the Stage-1 Simple English Wikipedia prose"),
    ("op.source.wikipedia-en", "ingest the Stage-2 full English Wikipedia (concepts & entities)"),
    ("op.provenance.audit", "audit (and optionally backfill) the §12 invariant: every artifact "
     "carries a citation + provenance rung"),
    ("op.remember", "store an owner-provided statement as a PRIVATE, owner-scoped, gated artifact "
     "(HUMAN_VALIDATED; no_share + no_promote by default) — conversation as verification"),
    ("op.share", "the CONSENT gate: the owner explicitly makes a private memory shareable — the "
     "only way owner-provided information ever leaves the private scope"),
    ("op.operator.define", "create/update an operator FROM DATA (kind + spec) — live immediately, "
     "no code, no restart; operators create operators"),
    # ── the cache tektons. Their `context` is their offer, and the offer is the whole point ──────
    #
    # These two sentences are measurements. `crystal.ontology.lookup.offer_synsets` grounds the
    # `context` field nouns only, first sense only, single tokens only — that grounding is the
    # tekton's coupling subspace (`ember.ontology.match.tekton_basis_for`), so every word either
    # becomes a basis direction or contributes nothing. Both were verified against a live store:
    #
    #   op.measure -> measurement.n.01 headroom.n.01 occupancy.n.01 capacity.n.01 cache.n.01
    #                 lattice.n.01 disk.n.01 byte.n.01 envelope.n.01 node.n.01      basis (2048, 10)
    #   op.reclaim -> eviction.n.01 cache.n.01 content.n.01 disk.n.01 headroom.n.01
    #                 capacity.n.01                                                 basis (2048,  6)
    #
    # The word "reclamation" is absent by measurement. It grounds — to `oewn-00269862-n` — but that
    # synset's `crystal.ontology.geometry.dense_vec` is all zero, so it contributes no direction and
    # `projection.frame` drops it. An offer whose head noun is a dead coordinate reads as correct
    # while coupling to nothing it names. `eviction` is the same concept with a live coordinate.
    # (`reclaim`/`evict`/`acquire` have no noun senses at all.) Function words cost nothing here —
    # of/the/and/this/by/to have no noun sense — but "free space" and "working set" would, because
    # grounding is single-token: write `headroom`, `capacity`.
    ("op.measure", "measurement of the headroom occupancy capacity cache lattice disk and byte "
     "envelope of this node"),
    # Offer only, deliberately unimplemented. `op.reclaim` decides what to evict and returns a
    # list; the deleting belongs to `store.local`, an organon, and is the most dangerous artifact
    # in the system. Registering the offer first is what lets the coupling be measured before
    # anything exists that can destroy data. `invoke` answers for it explicitly (see below), so its
    # result is distinguishable from the generic "not invokable" a typo would get.
    ("op.reclaim", "eviction of cache content to restore disk headroom and capacity"),
]


# Keys the store adds on write, which therefore appear on the way back out and are excluded when
# asking "would writing this doc change the stored row?". `_same_as_stored` drops these by name and
# every `_`- or `@`-prefixed key by prefix.
#   `_origin`/`_seq`       the version identity, allocated by the write itself
#   `created_time_origin`  the store attributing the clock reading to its claimant (contract §2.2)
#   `@`-prefixed keys      backend row metadata, on rows that carry it
_STORE_ADDED_KEYS = ("created_time_origin",)


def _same_as_stored(existing: Optional[Dict[str, Any]], doc: Dict[str, Any]) -> bool:
    """True when `put_artifact(doc)` would store byte-for-byte what is already there.

    The test is whether the content changed, not whether put was called. `put_artifact` is
    idempotent by id, so re-writing identical bytes leaves the row the same — at a cost. Every put
    allocates a fresh `_seq` from this observer's proper time, vacates the old one, XORs the row's
    merkle leaf twice, and re-enters the publish feed. "Idempotent" describes the resulting row; the
    events consumed to get there are its own quantity."""
    if not existing:
        return False

    def durable(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items()
                if not (k.startswith("_") or k.startswith("@") or k in _STORE_ADDED_KEYS)}

    return durable(existing) == durable(doc)


def register_control_operators(store, *, author: str = DEFAULT_AUTHOR) -> int:
    """Make sure every meta-operator artifact exists and matches its definition in code.

    This runs on every `invoke`, and therefore on every task the work pool executes, so it writes
    only when a definition has actually moved. Two properties make that possible:
      1. `created_time` is set once at first registration and preserved thereafter. A fresh clock
         reading per call would make the doc unequal to the stored row every time, and a creation
         time rewritten on every read is wrong on its own terms.
      2. The write is skipped when the doc would land identical.

    Unconditional rewriting costs one `_seq` allocation, one vacated seq, one churned merkle leaf
    and one mesh republish per operator per task — measured on a scratch lattice store at 15
    allocations per `run_task` against 1 unit of real work, which also inflates every downstream
    metric that counts authored events.

    The return value is how many operators are registered, not how many were written this call;
    `written` is recorded in `_CONTROL_OPS_WRITTEN` so a caller can read the churn directly."""
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    from ember.runtime.runner import evolution
    # Custody is established here rather than at the seven call sites. These writes carry inline
    # `content`, and sealing it needs an acting principal in scope
    # (`content_crypto.encrypt_content`). This function runs from background paths that have none —
    # `runtime/pool.py`, `runtime/worker.py`, node startup — and its own docstring says it also runs
    # "on every `invoke`", which includes request work that already has one.
    #
    # Which is why it is `if_unowned` and not a plain wrap: replacing a caller's identity with the
    # platform system principal would mean work done for a person, authorized as the platform, for
    # the rest of that call. See `ember/custody.py`.
    from ember.custody import system_custody_if_unowned
    with system_custody_if_unowned("ember.register_control_operators"):
        return _register_control_operators_inner(store, author, evolution)


def _register_control_operators_inner(store, author, evolution) -> int:
    written = 0
    for name, offer in CONTROL_OPS:
        existing = store.artifacts.get_artifact(name)
        doc = {
            "id": name, "content_type": OPERATOR_CONTENT_TYPE, "state": "committed",
            "context": offer, "content": f"control operator {name}: {offer}",
            "provenance": P_HUMAN, "cited_from": CITE_GENESIS,
            "created_by": author,
            # Preserved, not re-read: when the operator was first registered here.
            "created_time": (existing or {}).get("created_time") or _now(),
        }
        doc = evolution.preserve_fitness(store.artifacts, doc)
        if _same_as_stored(existing, doc):
            continue
        store.artifacts.put_artifact(doc)
        written += 1
    _CONTROL_OPS_WRITTEN["last"] = written
    return len(CONTROL_OPS)


# Last call's write count, so the churn is observable directly rather than re-measured from `_seq`
# deltas. Read by `test_pool` / `test_lattice_genesis`.
_CONTROL_OPS_WRITTEN: Dict[str, int] = {"last": 0}


# ── owner-scoped memory — chat CAN write to the corpus, but private & gated ───
# When the owner gives Ember information rather than a question, it becomes a private artifact:
# homed in private.<principal>, provenance HUMAN_VALIDATED (the owner staked it), flagged
# visibility=private + no_share + no_promote + owner=<principal>. Those flags are the boundary the
# mesh/grants layer enforces, so a private memory stays inside the owner's light-cone: it does not
# promote into a shared subject and does not mesh out. Explicit "remember …" statements are what
# ingest; questions and banter stay read-only.

def _principal() -> str:
    """The person on whose behalf a private write is being made.

    A process with no configured principal is not a person, so there is no owner to record and this
    raises rather than substituting one. `remember` and `share` write into `private.<person>` and
    stake a claim under HUMAN_VALIDATED, both of which need a real owner to mean anything.

    Callers that need cognition rather than ownership go through `delegate.Delegate`, which resolves
    a reserved non-person identity when none is configured."""
    import os
    from ember.runtime.delegate import LOCAL_PERSON
    who = (os.getenv("EMBER_PRINCIPAL") or "").strip()
    if not who:
        raise RuntimeError(
            "EMBER_PRINCIPAL is not set. This path writes an owner-scoped PRIVATE artifact under "
            "HUMAN_VALIDATED provenance, which requires a real person — it previously defaulted to "
            "a hardcoded personal email and silently attributed every anonymous act to that human. "
            "Set EMBER_PRINCIPAL, or use ember.delegate.Delegate for cognition that does not need "
            "an owner (it resolves the reserved %r identity)." % LOCAL_PERSON)
    return who




def remember(store, text: str, *, principal: Optional[str] = None) -> Dict[str, Any]:
    """Ingest an owner-provided statement as a private, owner-scoped, gated artifact. Content is
    encrypted into the content store, and the index carries only keyed lemmas for recall. The
    artifact stays inside the owner's scope: promotion and sharing both go through `share`, under
    consent. Returns what was stored."""
    principal = principal or _principal()
    if not text or not text.strip():
        return {"stored": False, "reason": "empty"}
    if not is_bootstrapped(store):
        bootstrap(store)
    from mantle.shard import content as C
    from ember.runtime.runner import describe as D, evolution
    import hashlib
    register_control_operators(store)
    pid, cite = _ensure_private(store, principal)
    mid = "mem-" + hashlib.sha256((principal + "|" + text.strip()).encode()).hexdigest()[:20]
    lemmas = D.terms_of("text/markdown", "memory", text) or \
        [w.lower() for w in text.split() if len(w) > 2][:16]
    # Access is the grant on `pid` (minted by `_ensure_private`) rather than a flag: this memory is
    # grounded in the owner's gated private collection, so `access.is_public` is False and only the
    # owner's light-cone reaches it. Authorship is provenance — `created_by` is a vertex reference
    # (§2.1) that resolves, so it is the person artifact's id rather than the raw principal.
    doc = {"id": mid, "content_type": "text/markdown", "state": "committed", "context": "",
           "content": "", "lemmas": lemmas, "collection_id": pid, "collections": [pid],
           "cited_from": cite, "via": "op.remember", "operator": "op.remember",
           "provenance": P_HUMAN,       # ownership is the grant on `pid`; authorship is `created_by`
           "created_by": _author_ref(store, principal), "created_time": _now()}
    if store.content is not None and store.keys_dir is not None:
        ref, size = C.put_content(store.content, store.keys_dir, text.encode("utf-8"))
        doc["content_ref"] = ref; doc["size"] = size        # encrypted; the index holds no cleartext
    else:
        doc["content"] = text                        # no content store -> inline (single-owner)
    store.artifacts.put_artifact(doc)
    evolution.record_invocation(store.artifacts, "op.remember", verified=True)
    return {"stored": True, "id": mid, "collection": pid, "private": True}


def share(store, artifact_id: str, *, to_principal: Optional[str] = None,
          to_collection: str = "subjects",
          principal: Optional[str] = None, confirm: bool = False,
          stake: float = 1.0) -> Dict[str, Any]:
    """The consent gate — the one path by which owner-provided information leaves the private scope.
    Two shapes, both requiring `confirm=True`, so sharing is always an explicit act:

      • `to_principal=<person>` — share with a person (#1): mint them a Read grant on this artifact.
        It stays gated; their light-cone now reaches it. Reversible, and no content migration.
      • otherwise — make public (#2): grant the public entity Read on this same artifact, list it
        under `to_collection` for browsing, and record the consent plus a staked claim on it — truth
        is economic. One artifact, now public: no copy and no re-key.

    Settlement (accrue/slash) rides the same verified/refuted counters that select operators."""
    principal = principal or _principal()
    a = store.artifacts.get_artifact(artifact_id)
    if not a:
        return {"shared": False, "reason": "not found"}
    from mantle.db import access
    # Only the owner may consent, and ownership is grant-derived: the grantee of the grounding
    # collection's grant, rather than an `owner` field. A row nobody owns (public/ungated) has no
    # consent to give here.
    g = a.get("collection_id") or a.get("origin_root")
    holder = access.gated_owner_map(store).get(str(g)) if g else None
    if holder is None or holder != principal:
        return {"shared": False, "reason": "not owner"}
    if not confirm:
        return {"shared": False, "reason": "consent required",
                "preview_lemmas": (a.get("lemmas") or [])[:12],
                "hint": "re-invoke with confirm=true (and an optional stake) to consent to sharing"}

    # ── #1 — share with a person: grant them Read (CRUDEASIO Share). The artifact stays gated and
    #    stays the owner's; the grantee's light-cone now reaches exactly this artifact. Cheap,
    #    reversible (revoke is one edit), no content migration. The grant is on the artifact id, so
    #    it covers this artifact alone and the rest of the owner's collection stays gated.
    if to_principal:
        access.grant_read(store, artifact_id, to_principal, principal)
        from ember.runtime.runner import evolution
        evolution.record_invocation(store.artifacts, "op.share", verified=True)
        return {"shared": True, "id": artifact_id, "with": to_principal, "mode": "grant"}

    # ── #2 — make public: grant the public entity Read on this very artifact, with no copy and no
    #    re-key. The grant makes `access.is_public` true, so the artifact meshes out and any reader
    #    may see it; the node that holds it decrypts and serves it (the key follows the node,
    #    per-collection, rather than the reader). The stake and the consent ride on the artifact
    #    itself. This is the one path by which owner-provided information becomes public.
    access.grant_read(store, artifact_id, access.PUBLIC_PRINCIPAL, principal)
    a["shared_consent"] = {"by": principal, "at": _now()}
    a["staked_claim"] = {"by": principal, "at": _now(), "stake": float(stake),
                         "status": "open", "verified": 0, "refuted": 0}
    a.setdefault("provenance", P_HUMAN)          # a staked human claim, not an unbacked assertion
    cols = set(a.get("collections") or [])
    cols.add(to_collection)                       # also list it under the shared collection for browsing
    a["collections"] = sorted(cols)
    store.artifacts.put_artifact(a)
    _ensure_edge(store, artifact_id, to_collection, "member_of")
    from ember.runtime.runner import evolution
    evolution.record_invocation(store.artifacts, "op.share", verified=True)
    return {"shared": True, "id": artifact_id, "to": to_collection,
            "consented_by": principal, "staked": float(stake), "mode": "public"}


def status(store) -> Dict[str, Any]:
    """The geometric status of the universe — the answer to 'how is it doing', computed from the
    graph itself (ρ, coverage, per-collection), plus where the curriculum stands."""
    # Fast path: live numbers come from server-side COUNT queries (cheap even under heavy write
    # load); ρ + coverage, which need a full byte scan, are served from the background cache so
    # /status stays off the corpus inline. That is what makes it responsive while ingesting.
    #
    # The count is unqualified deliberately. Measured via EXPLAIN, the two shapes are different
    # algorithms:
    # SELECT count(*) FROM Artifact -> CALCULATE USERTYPE SIZE: Artifact
    #                                                       O(1) from type metadata — 0.00s for
    #                                                       6,241,124 rows
    #     SELECT count(*) FROM Artifact WHERE state = ?  -> FETCH FROM INDEX -> EXTRACT VALUE FROM
    #                                                       INDEX ENTRY -> FILTER ITEMS BY TYPE
    #                                                       i.e. dereference every matched row
    # Any `WHERE` takes the count off the O(1) metadata path, and `state='committed'` is the least
    # selective predicate available — 5,670,283 rows / 103.03s for exactly this query — which is a
    # ~100s scan on a 30s timer.
    # So "artifacts" means all artifacts rather than committed-only: that is what the field is
    # called and what /status shows, and the non-committed remainder is a rounding error against
    # 6.2M. A selective count (e.g. `state='archived'`, 2,292 rows / 0.41s) is still cheap and can
    # be added if the committed/draft split is ever needed.
    total = _count_artifacts_cached(store)
    stages = []
    for spec in CURRICULUM:
        col = spec["stage"]
        # Served from a TTL cache, like every other expensive number here. A per-collection count is
        # cheap only for a selective collection. Measured on healthy T5:
        #     stage.1.grammar (   38,994 rows)  ->   0.04s
        #     stage.2.world   (6,063,979 rows)  ->  22.52s
        # On a heap-starved node the same query does not complete: it exhausts the 8 GB heap and
        # throws `OutOfMemoryError` from `XNIO-1 Accept`, taking the acceptor thread with it, after
        # which the container reports "Up" while every connection hangs on ReadTimeout. status is
        # on the per-tick path of both the health loop and the aggregator, and `_count_null` two
        # calls below states the governing rule for this function: status is on the 30s stats path
        # and must stay off a two-minute scan.
        # advance_curriculum deliberately keeps the uncached call: there the count is the resume
        # offset handed to the ingester, and a stale value would re-ingest rows we already hold.
        have = _count_collection_cached(store, col)
        # `have` is None when not measured (see _count_collection_cached) — that propagates as
        # "unknown" rather than as an invented number.
        row = {"stage": col, "have": have, "target": spec["target"],
               "promoted": _promoted(store, col),
               "progress": (round(min(1.0, have / spec["target"]), 3)
                            if (spec["target"] and have is not None) else None)}
        if "source" in spec:                      # stage-0: several sources share one collection
            row["source"] = spec["source"]
            row["source_done"] = bool(spec["done"](store)) if spec.get("done") else None
        stages.append(row)
    # Not measured here. status is on the 30s stats path, and these two are known-unservable full
    # scans (see _count_null). They return the cached value or None, and the slow path refreshes
    # them. `None` propagates as "unknown" below: an unmeasured audit, distinct from a clean one.
    miss_cite = _count_null(store, "cited_from")
    miss_prov = _count_null(store, "provenance")
    measured = miss_cite is not None and miss_prov is not None
    heavy = _METRICS_CACHE.get("val") or {}
    return {
        "artifacts": total,
        "rho": heavy.get("rho"),                 # from background scan (None until first computed)
        "keyed_coverage": heavy.get("keyed_coverage"),
        "rho_as_of": _METRICS_CACHE.get("ts"),
        "collections": heavy.get("collections", {}),
        "curriculum": stages,
        "sources": sorted(SOURCE_INGESTERS.keys()),
        # Unmeasured is None. `False` would report the invariant violated and `0` would report a
        # clean audit, when the scan was deliberately skipped on this path. `measured` states which
        # of the two the caller is holding.
        "provenance": {"invariant_holds": (miss_cite == 0 and miss_prov == 0) if measured else None,
                       "measured": measured,
                       "missing_cited_from": miss_cite, "missing_provenance": miss_prov},
    }


def _read_trend(store, last: int = 20) -> List[Dict[str, Any]]:
    from ember.surface.stats import tail_jsonl
    nd = _node_dir(store)                     # None ⇒ no node layout ⇒ no trend to read
    return tail_jsonl(nd / "metrics.jsonl", last) if nd else []


def published_gate(store) -> Optional[Dict[str, Any]]:
    """The health-loop's published golden-gate verdict (node-repair, the real suite), or None.

    serve reads what the standing gate published — `golden.json`, written atomically by
    _fleet/peers/71/ember/health-loop.py next to stats.json — rather than probing node health itself. The record
    carries its own cadences, so staleness is derived from the publisher's declared schedule rather
    than from a constant chosen here: the loop is single-threaded, so the longest honest gap between
    publishes is one slow interval plus one full golden timeout. Older than that means the gate
    itself is down, and the verdict is reported stale (present, and marked as untrusted).
    """
    import json as _json
    import time as _time
    nd = _node_dir(store)
    if nd is None:                            # no node layout ⇒ no gate has published here
        return None
    try:
        rec = _json.loads((nd / "golden.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    horizon = float(rec.get("slow_s") or 900) + float(rec.get("golden_timeout_s") or 1800)
    rec["age_s"] = round(_time.time() - float(rec.get("ts") or 0), 1)
    rec["fresh"] = rec["age_s"] < horizon
    return rec


def consistency(store, *, recent: int = 20) -> Dict[str, Any]:
    """Checks & balances — the physical and logical invariants the universe obeys. Anomalies are
    things to investigate: a ratio out of bounds (ρ∉[0,1]), generators exceeding the corpus, mass
    vanishing (data loss), fitness out of range, or the provenance invariant broken. Cooling (ρ↓)
    is a watch rather than a hard fault — ρ legitimately rises during ingestion and falls on
    consolidation. Deterministic; one cached scan plus the metrics trend."""
    m = all_metrics(store)
    checks: List[Dict[str, Any]] = []

    def chk(name, ok, detail, hard=True):
        checks.append({"check": name, "ok": bool(ok), "hard": hard, "detail": detail})

    rho = m.get("rho")
    chk("rho_in_[0,1]", rho is None or 0.0 <= rho <= 1.0 + _metric_half_quantum(), {"rho": rho})
    chk("generators_le_corpus", m["generator_bytes"] <= m["bytes"],
        {"generator_bytes": m["generator_bytes"], "bytes": m["bytes"]})
    cov = m.get("keyed_coverage")
    chk("coverage_in_[0,1]", cov is None or 0.0 <= cov <= 1.0 + _metric_half_quantum(),
        {"coverage": cov})
    # Unmeasured is None, as status states above; boolifying it would turn "not measured" into a
    # hard failure whose own detail reads missing=0. When the live count was not taken the verdict
    # belongs to the published gate (node-repair's suite measures the invariant every 900s). With no
    # gate published either, the invariant is genuinely unmeasured, which is a soft watch — "go
    # measure" — rather than a claim that it is broken.
    prov = dict(m["provenance"])
    holds = prov.get("invariant_holds")
    if holds is None:
        gate = published_gate(store)
        if gate is not None and gate.get("fresh") and gate.get("ok") is not None:
            holds = bool(gate["ok"]) or not any(
                "provenance" in str(f).lower() or "cited" in str(f).lower()
                for f in (gate.get("fail") or []))
            prov["source"] = "gate"
            prov["gate_age_s"] = gate.get("age_s")
    chk("provenance_invariant", holds if holds is not None else True,
        prov, hard=holds is not None)
    if holds is None:
        checks[-1]["ok"] = False                     # unmeasured: surface as a WATCH, not a pass
    from ember.runtime.runner import evolution
    # Measured on a 6,241,124-row corpus: this form and `evolution._all_operators` return the same
    # 43 operator ids in 0.00s. `content_type` is pushed down and is selective for this value, so
    # there is no scan here to remove — selectivity cuts both ways, and `list_by_content_type`'s
    # "filters in Python over every row" describes a different path.
    bad = [op["id"] for op in store.artifacts.list_artifacts(content_type=OPERATOR_CONTENT_TYPE)
           if not (0.0 <= evolution.fitness(op) <= 1.0)]
    chk("fitness_in_[0,1]", not bad, {"out_of_range": bad[:5]})
    # trend: mass must not vanish (data loss); cooling is a soft watch
    tr = _read_trend(store, recent)
    arts = [t.get("total_artifacts", t.get("artifacts")) for t in tr
            if t.get("total_artifacts", t.get("artifacts")) is not None]
    rhos = [t["rho"] for t in tr if t.get("rho") is not None]
    chk("mass_monotonic", len(arts) < 2 or arts[-1] >= arts[0] - 1,
        {"first": arts[0] if arts else None, "last": arts[-1] if arts else None})
    chk("universe_cooling", len(rhos) < 2 or rhos[-1] <= rhos[0] + _metric_quantum(),
        {"first": rhos[0] if rhos else None, "last": rhos[-1] if rhos else None}, hard=False)
    anomalies = [c for c in checks if not c["ok"] and c["hard"]]
    watches = [c for c in checks if not c["ok"] and not c["hard"]]
    return {"checks": checks, "anomalies": anomalies, "watch": watches,
            "all_pass": not anomalies}


def ingest_progress(store, name: str = "wikipedia-en") -> Dict[str, Any]:
    """Shard convergence for THIS box: how many of its EMBER_SHARDS-assigned shards are checkpointed
    done, and whether ingest has converged. Cheap (no HF call — just the shard-done count vs range).
    `converged` is the landing signal: an ingest box that has converged AND been drained is safe to
    tear down."""
    import os
    rng = os.getenv("EMBER_SHARDS", "")
    lo = hi = None
    if "-" in rng:
        try:
            lo, hi = (int(x) for x in rng.split("-", 1))
        except Exception:
            pass
    assigned = (hi - lo + 1) if (lo is not None and hi is not None) else None
    done = len(_shards_done(store, name))
    return {"range": rng or "all", "assigned": assigned, "done": done,
            "converged": bool(assigned is not None and done >= assigned)}


def health(store) -> Dict[str, Any]:
    """Monitoring + health, answered from the published gate rather than from hand-rolled probes.

    Two rules:
      1. node-repair is the test suite, so its published verdict (`golden.json`) is node health.
         No verdict published means the gate is not running, which reads as not healthy.
      2. The worker heartbeat is workload telemetry rather than node health: a leaf with no ingest
         campaign has no worker at all. It is reported, and its errors still degrade — a failing
         workload is a real problem — while its absence gates nothing."""
    import time
    from ember.surface.stats import tail_jsonl
    nd = _node_dir(store)                     # None ⇒ no node layout ⇒ no heartbeat to read
    logp = (nd / "worker.log") if nd else None
    worker: Dict[str, Any] = {"heartbeat_log": str(logp) if logp else None,
                              "alive": False, "last_tick": None,
                              "age_s": None, "recent_ticks": 0, "recent_errors": 0,
                              "last_ingest": None, "last_rho": None}
    try:
        recs = tail_jsonl(logp, 25) if logp else []
        if recs:
            # freshest ts across ticks AND liveness pings — a long shard-tick still pings alive
            last_ts = max((float(r.get("ts", 0) or 0) for r in recs), default=0.0)
            age = time.time() - last_ts
            tick_recs = [r for r in recs if r.get("tick") is not None and not r.get("ping")]
            last = tick_recs[-1] if tick_recs else recs[-1]
            errs = sum(1 for r in recs if (r.get("ok") is False)
                       or (isinstance(r.get("ingest"), dict) and r["ingest"].get("error")))
            # The liveness window is the pinger's, and it is read from the pinger.
            # `worker.liveness_window_s` is the number of consecutive pings a node may miss times
            # the cadence it actually pings at. Change either there and this follows.
            from ember.runtime import worker as _worker
            worker.update(last_tick=last.get("tick"), age_s=round(age, 1), recent_ticks=len(tick_recs),
                          recent_errors=errs, alive=age < _worker.liveness_window_s(),
                          last_ingest=last.get("ingest"), last_rho=last.get("rho"))
    except Exception:
        pass
    gate = published_gate(store)                        # the health-loop's golden verdict
    st = status(store)                                  # fast (COUNT-based)
    cons = consistency(store)                           # reads the gate for unmeasured provenance
    # Why degraded — each reason names the specific failing check.
    reasons: List[str] = []
    notes: List[str] = []
    if gate is None:
        reasons.append("gate: no published verdict (health-loop not running on this node)")
    elif not gate.get("fresh"):
        reasons.append(f"gate: verdict stale ({gate.get('age_s')}s old — health-loop down?)")
    elif not gate.get("ok"):
        fails = ", ".join(str(f) for f in (gate.get("fail") or [])[:5]) or gate.get("error") or "?"
        reasons.append(f"gate: FAIL — {fails}")
    if st["provenance"]["measured"] and not st["provenance"]["invariant_holds"]:
        reasons.append(f"provenance: {st['provenance']['missing_cited_from']} uncited artifacts")
    if not cons["all_pass"]:
        reasons.append("anomaly: " + ", ".join(a["check"] for a in cons["anomalies"]))
    if worker.get("recent_errors"):
        reasons.append(f"{worker['recent_errors']} worker errors in last {worker.get('recent_ticks')} ticks")
    ingest = ingest_progress(store)
    if not worker["alive"] and not ingest.get("converged"):
        # Informational, not gating: nothing is ingesting right now and the assignment isn't
        # converged. On a serving leaf that is normal life, not degradation.
        notes.append("no worker running; ingest assignment not converged")
    gate_ok = bool(gate and gate.get("fresh") and gate.get("ok"))
    return {
        "gate": gate,                                   # the published golden verdict, verbatim
        "worker": worker,                               # workload telemetry (informational)
        "artifacts": st["artifacts"],
        "rho": st["rho"], "keyed_coverage": st["keyed_coverage"],
        "provenance": st["provenance"],
        "consistency": cons,                            # checks & balances — anomalies to investigate
        "curriculum": st["curriculum"],
        "ingest": ingest,                               # shard convergence (the 'landing' signal)
        "healthy": bool(gate_ok and cons["all_pass"] and not reasons),
        "degraded_reasons": reasons,
        "notes": notes,
    }


def _count_collection(store, collection_id: str, *, committed_only: bool = True) -> int:
    """Indexed count for one collection. `committed_only=False` matches `_count_in` exactly.

    The `WHERE collection_id` is what makes this tractable — selectivity (§3): an index helps on a
    *selective* value. Measured on T5:

        stage.1.grammar (   38,994 rows)  ->   0.04s
        stage.2.world   (6,063,979 rows)  ->  22.52s

    So "cheap" holds for a selective collection; a 6M-row stage still costs ~22s. It is nonetheless
    far better than `_count_in`, which streams every one of those rows through Python over HTTP (the
    ~370s-per-scan class this codebase documents at `crystal/evolution.py`).

    The unqualified form is a different animal entirely: `SELECT count(*) FROM Artifact` over 6.2M
    rows can run past 900s and then die with `java.lang.OutOfMemoryError` against an 8 GB heap while
    keyed lookups answer instantly throughout. It belongs nowhere near a full node. (On a
    heap-starved node the selective query above also times out — that is the heap, and the same
    schema and a comparable corpus run it in 22s on T5.)

    `committed_only` exists because the two callers need different things, and the difference is
    load-bearing: `advance_curriculum` uses the count as its resume offset, so it counts every
    state. Filtering to `committed` there would under-count, hand the ingester a low offset, and
    re-ingest records it already has.

    `count_in_collection` is a counter maintained inside the write transaction: O(1), and it answers
    exactly the question asked. A per-collection count that silently degraded to the global corpus
    count would be `advance_curriculum`'s resume offset (`have`, then `spec["ingest"](store, have,
    n)`) and both sides of the exhaustion test `drained = (now <= have)` — a stage holding 48 rows
    would resume at the corpus total, skipping records it never ingested, and would promote as soon
    as the corpus exceeded `target` regardless of what the stage itself holds. The error there is
    `global - real`, so it grows with the corpus and is largest for the emptiest stage."""
    cnt = _typed(store.artifacts, "count_in_collection")
    if cnt is not None:
        # `committed_only` maps onto the lattice store's `committed_only` flag rather than onto a
        # `state = 'committed'` predicate. The flag selects a separate counter maintained at write
        # time, so the unselective predicate (5,670,283 rows / 103.03s) is never issued, and the
        # count stays independent of whether `state` remains a queryable column.
        return int(cnt(collection_id, committed_only=committed_only))
    # No typed counter: stream the collection through Python. Honest but O(n) — this is the
    # `_count_in` the indexed form replaced, kept as a last resort for the in-memory fakes.
    # `_count_in` counts every state, so `committed_only=True` is satisfied by filtering here
    # rather than by returning the all-states number under a committed-only name.
    if committed_only:
        return sum(1 for a in store.artifacts.list_artifacts(collection_id=collection_id)
                   if a.get("state") == "committed")
    return _count_in(store, collection_id)


import weakref as _weakref

_COLLECTION_COUNT_CACHE: "_weakref.WeakKeyDictionary" = _weakref.WeakKeyDictionary()
_COLLECTION_COUNT_TTL = 300.0


_ARTIFACT_COUNT_CACHE: dict = {}
_ARTIFACT_COUNT_TTL = 900.0


def _count_artifacts_cached(store, *, allow_scan: bool = False):
    """Total artifacts, served from cache, so the 30s path issues no query.

    `SELECT count(*) FROM Artifact` is O(1) only where the planner can answer it from type metadata.
    On T5 it plans as `CALCULATE USERTYPE SIZE` and returns in 0.003s; on a heap-starved node the
    same statement is a full scan that times out at 60s, and the node's log then shows `Java heap
    space` + `Exception in thread "XNIO-1 Accept"` — the acceptor dies and the box goes silent under
    nothing more than the health loop and the aggregator, i.e. under the 30s health tick itself.

    A client timeout is no protection: the client gives up at 60s while the server keeps executing,
    and it is the server-side scan that exhausts the heap. The safe move is to keep the query off
    the frequent path entirely.

    So: return the cached value, or None meaning not measured. Only the slow path passes
    allow_scan=True. None rather than 0 — a corpus of "0 artifacts" would read as an empty node."""
    now = time.time()
    ent = _ARTIFACT_COUNT_CACHE.get("v")
    if ent and now - ent[0] <= _ARTIFACT_COUNT_TTL:
        return ent[1]
    if not allow_scan:
        return ent[1] if ent else None          # a stale real reading, or None for no reading
    try:
        n = store.artifacts.count()
        _ARTIFACT_COUNT_CACHE["v"] = (now, n)
        return n
    except Exception:
        return ent[1] if ent else None


def _count_collection_cached(store, collection_id: str, *, allow_scan: bool = False):
    """`_count_collection` for the 30s stats path, served from a 300s TTL cache.

    A collection count is servable — unlike `_count_null`'s known-unservable scan — so this caches a
    real value. It is not cheap on a bulk collection: measured on healthy T5, `stage.2.world`
    (6,063,979 rows) takes 22.52s, and on a heap-starved node it never completes, taking the JVM's
    acceptor thread with it. status runs every tick, so the cache is what keeps that off the tick.

    Keyed on the store object via a weak map rather than on `id(store)`: CPython reuses freed
    addresses, so an id-keyed cache can serve one store another store's numbers (see
    `_METRICS_CACHE` above).

    The first call per window still pays full price; the point is that the next ~300s of ticks do
    not. On a store that cannot be weak-referenced this degrades to the uncached call."""
    now = time.time()
    try:
        ent = _COLLECTION_COUNT_CACHE.get(store)
        if ent is not None and now - ent["at"] <= _COLLECTION_COUNT_TTL and collection_id in ent["vals"]:
            return ent["vals"][collection_id]
    except TypeError:                       # store not weak-referenceable -> never cache
        return _count_collection(store, collection_id, committed_only=False) if allow_scan else None
    # The cache alone would still leave the first call per window paying the full scan, and on a
    # heap-starved node that one call is fatal. Measured via EXPLAIN on T5, this count is planned as
    #   FETCH FROM INDEX Artifact[collection_id] -> EXTRACT VALUE FROM INDEX ENTRY -> FILTER BY TYPE
    # i.e. the index locates the rows and then every matched entry is dereferenced into a full
    # record just to satisfy the type filter, before counting. So `count(*)` over stage.2.world
    # loads 6,063,979 records into heap to produce one integer — 22.5s on healthy T5, and on an 8 GB
    # heap it OOMs the XNIO acceptor and zombies the node. It is a property of the query shape, so a
    # bigger heap only moves the wall.
    # So this follows `_count_null`'s rule, stated by this function's neighbour: status is on the
    # 30s stats path and stays off a two-minute scan. Return the cached value, or None meaning not
    # measured. Only an explicit slow-path caller passes allow_scan=True. None rather than 0 is
    # deliberate — 0 would read as "this stage is empty", which is a measurement nobody took.
    if not allow_scan:
        return None
    val = _count_collection(store, collection_id, committed_only=False)
    try:
        ent = _COLLECTION_COUNT_CACHE.get(store)
        if ent is None or now - ent["at"] > _COLLECTION_COUNT_TTL:
            ent = {"at": now, "vals": {}}
            _COLLECTION_COUNT_CACHE[store] = ent
        ent["vals"][collection_id] = val
    except TypeError:
        pass
    return val



# Per-store, and keyed on the store object via a weak map rather than on id(store). CPython reuses
# freed memory addresses, so an id-keyed cache can serve one store another store's metrics for a
# whole TTL. A WeakKeyDictionary holds the real object, so two distinct stores never collide, and
# entries disappear with the store. A single module-level dict would also carry state between tests
# sharing one process.
_NULL_COUNT_CACHE: "_weakref.WeakKeyDictionary" = _weakref.WeakKeyDictionary()
_NULL_COUNT_TTL = 300.0

# `field` is a closed set rather than a free string parameter. The set below is the single place
# the fact lives, so the audited fields have one spelling; an unknown field raises ValueError.
_NULL_AUDIT_FIELDS = ("cited_from", "provenance")

# How many rows the keyset walk visits before giving up. Past this it returns None — no reading
# taken — rather than a truncated integer: a truncated count of "rows missing provenance"
# understates the violation and reads as a cleaner audit than the truth.
_NULL_SCAN_CAP = 250_000


def _count_null(store, field: str, *, allow_scan: bool = False):
    """Count committed rows missing `field` (the cited_from/provenance backfill audit).

    This query is known-unservable, so it is issued only on the slow path. Both halves defeat every
    index available: `IS NULL` cannot use an LSM index at all (nullStrategy SKIP — nulls are not
    stored), and `state = 'committed'` matches nearly all rows, so it prunes nothing (measured:
    5,670,283 rows, 103.03s). Given that, the move is to keep it off the hot path rather than to
    time it or cache it after the fact.

    So: return the cached value, or None meaning "not measured". Only an explicit slow-path caller
    passes allow_scan=True (refresh_metrics / the health loop's 900s cadence). status is on the
    30s stats path, so it reports None and says so.

    Returning None rather than 0 is deliberate — 0 would read as "no rows are missing provenance",
    a clean audit that never ran. An unmeasured value stays distinguishable from a healthy one.

    `field` is constrained to `_NULL_AUDIT_FIELDS` — see the note there — and is passed as a value
    by every path this function reaches."""
    if field not in _NULL_AUDIT_FIELDS:
        raise ValueError(
            "_count_null: %r is not an audited provenance field. Add it to _NULL_AUDIT_FIELDS "
            "deliberately — this used to be an f-string interpolated into SQL, and the SQLite "
            "shim re-derived the field by substring-sniffing the query text, so the two spellings "
            "could disagree with nothing to catch it." % (field,))
    now = time.time()
    ent = None
    try:
        ent = _NULL_COUNT_CACHE.get(store)
        if ent is not None and now - ent["at"] <= _NULL_COUNT_TTL and field in ent["vals"]:
            return ent["vals"][field]
    except TypeError:                       # store not weak-referenceable -> never cache
        ent = None
    if not allow_scan:
        return None
    n = _scan_missing_field(store, field)
    # A failed or abandoned scan is None, and it is never cached. Caching 0 there would store a
    # clean audit assembled out of a query that did not complete.
    if n is None:
        return None
    try:
        if ent is None or now - ent["at"] > _NULL_COUNT_TTL:
            ent = {"at": now, "vals": {}}
            _NULL_COUNT_CACHE[store] = ent
        ent["vals"][field] = n
    except TypeError:
        pass
    return n


def _scan_missing_field(store, field: str):
    """Rows missing `field`. Returns an int, or None meaning not measured.

    The scan counts rows missing `field` across every state rather than committed-only. On a store
    where `state` is not a queryable column there is no way to express "committed rows missing
    `field`", and a substitute — treating every row as committed, or sniffing a `state` key out of
    the doc JSON and calling it a predicate — would rebuild the unselective predicate under a new
    name.

    The all-states count is a superset of the committed-only answer, and the direction matters: the
    only consumer is `invariant_holds = (miss_cite == 0 and miss_prov == 0)`, so a superset can
    raise a false alarm while a clean audit over a dirty corpus stays out of reach. Fail-safe in the
    right direction.

    The walk is keyset-paged (`page_by_id`), never `count(*)`, and never `SKIP`. It gives up at
    `_NULL_SCAN_CAP` and returns None rather than a truncated integer."""
    # 1. A typed method, where the backend has one. Preferred: it can use the store's own indexes
    #    and it takes `field` as a value rather than as SQL. This branch is here so adding one
    #    needs no change at this call site.
    typed = _typed(store.artifacts, "count_missing_field")
    if typed is not None:
        try:
            return int(typed(field))
        except Exception:
            return None
    # 2. Keyset walk. Bounded, honest, no count(*), no OFFSET.
    page = _typed(store.artifacts, "page_by_id")
    n = 0
    seen = 0
    if page is not None:
        after = ""
        while seen < _NULL_SCAN_CAP:
            rows = page(after=after, limit=1000)
            if not rows:
                return n                    # walked the whole corpus
            for r in rows:
                d = r.get("doc") or {}
                if not d.get(field):
                    n += 1
            seen += len(rows)
            after = rows[-1]["id"]
        return None                         # hit the cap -> no reading, rather than a partial count
    # 3. Last resort (the in-memory fakes): the store's own typed listing.
    try:
        for a in store.artifacts.list_artifacts():
            seen += 1
            if seen > _NULL_SCAN_CAP:
                return None
            if not a.get(field):
                n += 1
        return n
    except Exception:
        return None


def refresh_metrics(store) -> None:
    """Recompute the heavy ρ/coverage scan and populate the cache. Call from a background thread
    (serve/worker) so the inline status path is served from the warm cache."""
    try:
        all_metrics(store, cache_ok=False)
    except Exception:
        pass
    # The curriculum counts belong here for the same reason ρ/coverage do: they stay off `status`
    # (measured: `count(*)` on stage.2.world dereferences 6,063,979 records into heap — 22.5s on T5,
    # fatal on a starved heap). This is the one place allowed to scan; status then serves the warm
    # cache and reports None until this has run at least once.
    try:
        _count_artifacts_cached(store, allow_scan=True)
    except Exception:
        pass
    for _spec in CURRICULUM:
        try:
            _count_collection_cached(store, _spec["stage"], allow_scan=True)
        except Exception:
            pass
    # Nothing in-tree calls `refresh_metrics` on a schedule: `serve.py` carries no metrics loop, so
    # wiring a background caller is an explicit deploy decision made outside this module. Until one
    # exists, `status` reports these as not measured (None) rather than 0. `_count_null` has the
    # same property.


# How deep compositions may nest. Chosen rather than measured — flagged like `_WIKI_STUB_CHARS`
# above. It exists to fail loudly well before Python's own recursion limit turns a definable data
# structure into a stack overflow with no attribution.
_MAX_COMPOSITION_DEPTH = 16


# ── the conversation tekton is reached, not run here ─────────────────────────────────────────────────
# `respond / learn / think / act` are lumen's (`lumen/conversation.py`, Rule Zero). Ember is "simply a
# runner" ([[ember-is-a-runner]]): it reaches the tekton over the ground plane. Inactive by default — a
# plain node has no live fabric, so the reach returns None and ember surfaces the computed null: no
# answer, no citation, and nothing supplied in the persona's place. The live cross-process carrier is a
# gated deploy step; `_conversation_carrier` is the injection point (None on a plain node).
def _conversation_carrier(store):
    """The live reach carrier for the conversation tekton, or None on a plain node (the default). The
    gated deploy step wires this to return `{root_secret, fabric, principal?}`; without it the reach is
    dark and op.respond yields the computed null."""
    return None


def _reach_conversation(store, operator_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Reach lumen's conversation capability (`operator_id`) over the ground plane, or return the
    computed null when no live fabric is wired (inactive by default). Ember imports no chorus — the
    reach rides `ember.runtime.reach`, which is built on `prism.reach`, keyed by the fleet root and
    gated by ember's own access light-cone."""
    text = args.get("text", "")
    principal = args.get("principal")
    carrier = _conversation_carrier(store)
    if carrier is not None:
        try:
            # A carrier may supply its own resolver, and for the store-and-forward transport it must.
            # `prism.reach.Reactor.__init__` calls `fabric.subscribe(...)`, so `fabric=` requires a
            # pub/sub transport — while the node's real transport is `prism.carriers.StoreCarrier`,
            # which `reach_host` passes as `fallback=` and drives with a `PumpLoop`. Passing it as
            # `fabric=` raises `AttributeError: 'StoreCarrier' object has no attribute 'subscribe'`,
            # which the `except` below turns into the computed null — so a fully wired host would
            # report no live fabric while `host.respond(...)` in the same process returns activations.
            #
            # Pump cadence belongs to whoever owns the loop, so the launcher hands in a
            # `resolve(need, to) -> evidence` closure over its own loop. The fire-and-collect
            # `reach` stays the path for a genuine pub/sub fabric.
            resolver = carrier.get("resolve")
            if callable(resolver):
                ev = resolver({"text": text, "principal": principal}, operator_id)
                if ev is not None:
                    return ev
            elif carrier.get("fabric") is not None:
                from ember.runtime import reach as _reach
                ev = _reach.reach(store, carrier.get("principal") or principal or "ember",
                                  {"text": text, "principal": principal}, to=operator_id,
                                  root_secret=carrier["root_secret"], fabric=carrier["fabric"])
                if ev is not None:
                    return ev
        except Exception:
            pass
    # The computed null — an absent answer and an absent citation, reported as such
    # ([[state-what-it-is]]).
    return {"answer": None, "activations": [], "cited": [],
            "refused": ("%s is the conversation tekton (moved to lumen, P7) — reached over the ground "
                        "plane, not run on this runner; no live reach fabric is wired on this node"
                        % operator_id)}


def invoke(store, operator_id: str, arguments: Optional[Dict[str, Any]] = None,
           *, _pinned: Optional[Dict[str, Any]] = None,
           _stack: tuple = ()) -> Dict[str, Any]:
    """Invoke a GENESIS operator by id (the InvokeArtifactRequest analogue, §4). Drives ingestion,
    consolidation, curriculum, and status — all as operators, over the same NEED->OFFER mechanism
    used locally and remotely. Records fitness for what it invokes.

    `_pinned` / `_stack` are internal and carry the invocation's own state down a composition:
    `_pinned` holds resolved operator artifacts so one logical invocation runs one version
    throughout, `_stack` is the in-flight operator chain used for cycle detection. Callers leave
    both to their defaults."""
    from ember.runtime.runner import evolution
    if not is_bootstrapped(store):
        bootstrap(store)
    register_control_operators(store)
    args = dict(arguments or {})

    if operator_id in SOURCE_INGESTERS:
        res = SOURCE_INGESTERS[operator_id](store, args)
        evolution.record_invocation(store.artifacts, operator_id, verified=bool(res.get("ingested", 0)))
        return {"operator": operator_id, "result": res}
    if operator_id == "op.remember":
        return {"operator": operator_id,
                "result": remember(store, args.get("text", ""), principal=args.get("principal"))}
    if operator_id == "op.share":
        return {"operator": operator_id,
                "result": share(store, args.get("artifact_id") or args.get("id", ""),
                                 to_collection=args.get("to_collection", "subjects"),
                                 principal=args.get("principal"),
                                 confirm=bool(args.get("confirm", False)),
                                 stake=float(args.get("stake", 1.0)))}
    if operator_id == "op.recognize":         # activation over the ontology (which concepts fire)
        from ember.ontology import activation
        return {"operator": operator_id,
                "result": {"activations": activation.recognize(store, args.get("text", ""))}}
    if operator_id in ("op.respond", "op.learn", "op.thought", "op.act"):
        # The conversation acts are lumen's. Ember reaches the tekton over the ground plane, inactive
        # by default, so without a live fabric the result is the computed null. op.recognize (above)
        # stays a direct ember call — it is recognition measurement (which concepts fire), the
        # grounding the runner keeps, distinct from the persona's conversation act.
        return {"operator": operator_id, "result": _reach_conversation(store, operator_id, args)}
    if operator_id == "op.measure":
        # Places a reading; the verdict and any downstream call belong elsewhere. `args` may carry
        # `root` (which store tree to measure) and `free_bytes` (substitute the one measured input,
        # so the coupling can be exercised across the envelopes this node can be in — there is no
        # threshold to move). See `ember/signal/state.py`.
        from ember.signal import state as _state
        return {"operator": operator_id,
                "result": _state.measure(store, root=args.get("root"),
                                         free_bytes=args.get("free_bytes"))}
    if operator_id == "op.reclaim":
        # Registered for its offer, and the result below says so. The offer is what makes the
        # tekton couple (`ember.ontology.match.tekton_basis_for`), and the coupling had to be
        # measurable before anything existed that could act on it. Its body, once written, decides
        # what to evict and returns a list; the deletion belongs to `store.local`, an organon,
        # grant-gated.
        return {"operator": operator_id,
                "result": {"registered": True, "implemented": False,
                           "refused": "op.reclaim is registered for its OFFER only. It has no body "
                                      "yet, deletes nothing, and must never be invoked by another "
                                      "operator — it activates by coupling to a signal, not by "
                                      "being called."}}
    if operator_id == "op.status.universe":
        return {"operator": operator_id, "result": status(store)}
    if operator_id == "op.health":
        return {"operator": operator_id, "result": health(store)}
    if operator_id == "op.consistency":
        return {"operator": operator_id, "result": consistency(store)}
    if operator_id == "op.mesh.status":
        from mantle.mesh import federation as _meshfed
        _rho = (_METRICS_CACHE.get("val") or {}).get("rho")
        return {"operator": operator_id,
                "result": _meshfed.mesh_status(store.artifacts.count(), local_rho=_rho)}
    if operator_id == "op.mesh.export":       # peer side: hand a page of artifacts (docs; +content opt)
        from mantle.mesh import federation as _meshfed
        return {"operator": operator_id,
                "result": _meshfed.export_page(store, int(args.get("offset", 0)),
                                               int(args.get("limit", 25)),
                                               with_content=bool(args.get("with_content", False)))}
    # ── anti-entropy mesh sync (memory-bounded, content-addressed, delta-only) ──────────────────
    if operator_id == "op.mesh.reconcile":     # the one sync path: Merkle anti-entropy over S3 —
        from mantle.mesh import sync as _sync         # publish my tree incrementally + pull only differing
        return {"operator": operator_id,        # leaves (vertices and edges). O(diff): converged peers
                "result": _sync.reconcile_via_s3(store,   # exchange one 32 KB tree and stop. No feeds.
                                                 max_leaves=int(args.get("max_leaves", 256)),
                                                 max_seconds=float(args.get("max_seconds", 0)))}
    if operator_id == "op.mesh.reach":         # reach: pull a missed index row from the substrate so a
        from mantle.mesh import sync as _sync         # limited ember can answer beyond what it holds (a targeted
        return {"operator": operator_id,        # one-leaf fetch, not a full converge). A miss is a need.
                "result": _sync.reach_index(store, str(args.get("id", "")))}
    if operator_id == "op.mesh.manifest":      # publish this ember as a peer-artifact with a
        from mantle.mesh import sync as _sync         # CAS-addressed manifest (its measured state) — peers
        return {"operator": operator_id,        # are artifacts too, discovered through the mesh.
                "result": _sync.publish_manifest(store)}
    if operator_id == "op.content.promote":   # async: copy local ciphertext up to the OVH origin
        from mantle.shard import content_tier
        return {"operator": operator_id,
                "result": content_tier.promote_local_content(
                    store, max_refs=int(args.get("max_refs", 2000)), page=int(args.get("page", 200)))}
    if operator_id == "op.mesh.pull":         # authoritative side: replicate every peer INTO here
        from mantle.mesh import federation as _meshfed
        return {"operator": operator_id,
                "result": _meshfed.pull_from_peers(store, max_pages=int(args.get("max_pages", 40)),
                                                   page=int(args.get("page", 25)),
                                                   with_content=bool(args.get("with_content", False)))}
    if operator_id == "op.provenance.audit":
        if args.get("backfill") or args.get("apply"):
            bf = backfill_provenance(store, apply=True)
            return {"operator": operator_id, "result": {**audit_provenance(store), "backfill": bf}}
        return {"operator": operator_id, "result": audit_provenance(store)}
    if operator_id == "op.curriculum.advance":
        res = advance_curriculum(store, per_tick=args.get("per_tick"))
        evolution.record_invocation(store.artifacts, operator_id, verified=None)
        return {"operator": operator_id, "result": res}
    if operator_id == "op.consolidate.nearvdup":
        res = consolidate_nearvdup(store, apply=bool(args.get("apply", False)),
                                   content_type=args.get("content_type", "text/markdown"),
                                   collection_id=args.get("collection_id"),
                                   # threshold=None -> minhash.merge_boundary = 1-1/(k+1) = 0.99225,
                                   # derived from the estimator width. An explicit caller threshold
                                   # wins.
                                   threshold=(float(args["threshold"]) if "threshold" in args else None),
                                   limit=int(args.get("limit", 60000)))
        return {"operator": operator_id, "result": res}
    if operator_id == "op.consolidate.colimit":
        # `concept_id` derives the diagram; `diagram_ids` has the caller supply it. Dispatch on
        # which one was given rather than on a mode flag — the two take genuinely different inputs,
        # and a flag would let a caller pass both.
        if args.get("concept_id"):
            res = consolidate_colimit_derived(
                store, str(args["concept_id"]),
                label_keyed=bool(args.get("label_keyed", False)),
                include_unanchored=bool(args.get("include_unanchored", False)),
                apply=bool(args.get("apply", False)))
        else:
            res = consolidate_colimit(store, list(args.get("diagram_ids", [])),
                                      canonical_id=args.get("canonical_id"),
                                      concept_lemmas=args.get("concept_lemmas"),
                                      apply=bool(args.get("apply", False)))
        return {"operator": operator_id, "result": res}
    if operator_id == "op.consolidate.crosswalk":
        res = consolidate_crosswalk(store, apply=bool(args.get("apply", False)),
                                    limit=int(args.get("limit", 20000)))
        return {"operator": operator_id, "result": res}
    if operator_id == TRANSDUCER_OP + "build":  # matches `op.transducer.build`
        # Lay the language:<lang> transducer substrate (stored IC · lex:<lang> entry edges · persisted ξ ·
        # the transducer artifact) so the chat path is keyed and never loads the corpus. One-time per corpus.
        from crystal.ontology import seed_lattice as _seed
        return {"operator": operator_id,
                "result": _seed.build(store, lang=str(args.get("lang", "en")))}
    # Define an operator from data (hot — no restart). op.operator.define carries {id, kind, spec,
    # offer}; the new operator is immediately invokable because invoke reads its artifact live.
    if operator_id == "op.operator.define":
        return {"operator": operator_id,
                "result": define_operator(store, args.get("id", ""), args.get("kind", ""),
                                          args.get("spec") or {}, offer=args.get("offer", ""))}

    # ── DATA-DRIVEN dispatch — the operator's OWN artifact (kind + spec) drives invocation, so a
    #    newly-minted/edited operator works instantly with no code change and no restart (§4). ──
    #
    # Version-pinned for the whole call tree. `_pinned` caches the resolved artifact for the
    # duration of one logical invocation, so a redefinition landing mid-composition leaves the
    # in-flight invocation on the version it started with: step 1 and step 2 run the same operator,
    # and the result is attributable to it. Resolve once, use throughout.
    if _pinned is None:
        _pinned = {}
    op = _pinned.get(operator_id)
    if op is None:
        op = store.artifacts.get_artifact(operator_id)
        if op is not None:
            _pinned[operator_id] = op
    if op and op.get("kind"):
        spec = op.get("spec") or {}
        if op["kind"] == "source":
            res = _run_source_spec(store, operator_id, spec, args)
            evolution.record_invocation(store.artifacts, operator_id, verified=bool(res.get("ingested", 0)))
            return {"operator": operator_id, "result": res}
        if op["kind"] == "composition":
            # Cycle + depth guards. A composition can reference itself, directly or through a cycle
            # a->b->a, and `op.operator.define` is reachable over HTTP, so unbounded recursion is
            # mintable. Both bounds fail loudly: a bounded call that names the cycle or the depth
            # beats a stack overflow that names nothing.
            if operator_id in _stack:
                return {"operator": operator_id,
                        "error": "composition cycle: %s" % " -> ".join(list(_stack) + [operator_id])}
            if len(_stack) >= _MAX_COMPOSITION_DEPTH:
                return {"operator": operator_id,
                        "error": "composition nested deeper than %d" % _MAX_COMPOSITION_DEPTH}
            steps = spec.get("steps") or []
            out = []
            for st in steps:                       # run the composed operators in order
                # Step args win over caller args. The author's fixed values are part of the
                # operator's definition — a composition pinning `limit=5` keeps it against a
                # caller's `limit=999` — and the caller's args fill in what the step left open.
                out.append(invoke(store, st.get("op"), {**args, **(st.get("args") or {})},
                                  _pinned=_pinned, _stack=tuple(_stack) + (operator_id,)))
            evolution.record_invocation(store.artifacts, operator_id, verified=None)
            return {"operator": operator_id, "result": {"steps": out}}
        return {"error": f"operator '{operator_id}' has unknown kind '{op['kind']}'"}

    # ── remote: the artifact names its own surface, so `kind` is not the discriminator ──────────
    #
    # The branch above is gated on `op.get("kind")`, and an operator served by a registered prism
    # carries no `kind` — deliberately, because minting an executable kind/spec from a POST would
    # let a registration compose onto `op.dev.run_tests`. Such an operator is invokable, just on
    # another surface, so dispatch follows what the artifact declares rather than whether it carries
    # an executable body. `register_remote_host` records `dispatch` and `endpoint` precisely so the
    # row states which surface serves it.
    #
    # `signal.resolve` reads the address, `send` seals a signal toward it, `ship` delivers it to the
    # prism Host's `POST {endpoint}/operators/{name}`. Sealing and shipping stay separate: a signal
    # is decoupled and best-effort, an RPC is neither.
    if op and (op.get("remote") or op.get("dispatch")):
        import os as _os

        from ember.signal import signal as _signal
        from ember.runtime.delegate import Delegate
        target = _signal.resolve(operator_id, store=store)
        if target.get("kind") != "remote":
            return {"error": f"operator '{operator_id}' declares a remote dispatch but does not "
                             f"resolve as one: {target.get('reason') or target.get('kind')}"}
        # The runner's own delegate: an invoke on this node is this node acting, which is what the
        # process delegate means. An unauthenticated HTTP read is a different principal — that one
        # is the commons.
        sent = _signal.send(Delegate.get(store), operator_id, operator=operator_id, store=store)
        sent["delivery"] = _signal.ship(sent, token=_os.getenv("EMBER_HOST_TOKEN", ""))
        # Fitness is evidence about a behaviour, and a delivery that never reached the host carries
        # no evidence either way, so `verified` stays None rather than counting a transport failure
        # against the operator.
        evolution.record_invocation(store.artifacts, operator_id, verified=None)
        return {"operator": operator_id, "dispatch": "signal", "result": sent}

    return {"error": f"operator '{operator_id}' is not invokable",
            "invokable": sorted(set(list(SOURCE_INGESTERS) + [c[0] for c in CONTROL_OPS]
                                    + ["op.consolidate.nearvdup", "op.consolidate.colimit",
                                       "op.consolidate.crosswalk", "op.operator.define"]))}


def define_operator(store, op_id: str, kind: str, spec: Dict[str, Any], *, offer: str = "",
                    author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """Create OR update an operator entirely from DATA — no code, no restart. The operator artifact
    carries its `kind` (source | composition) + `spec`; invoke interprets it live. This is
    'operators create operators': you define behavior by INVOKING op.operator.define. Idempotent
    (re-defining updates the spec)."""
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    if not op_id or not op_id.startswith("op."):
        return {"defined": False, "reason": "id must start with 'op.'"}
    if kind not in ("source", "composition"):
        return {"defined": False, "reason": f"unknown kind '{kind}' (source|composition)"}
    from ember.runtime.runner import evolution
    updating = store.artifacts.get_artifact(op_id) is not None
    doc = evolution.preserve_fitness(store.artifacts, {
        "id": op_id, "content_type": OPERATOR_CONTENT_TYPE, "state": "committed",
        "kind": kind, "spec": spec, "context": offer or f"{kind} operator {op_id}",
        "content": f"{kind} operator {op_id} (data-driven): {offer}",
        "lemmas": [op_id.replace("op.", "").replace(".", " ").split()[0] if "." in op_id else op_id],
        "provenance": P_HUMAN, "cited_from": CITE_GENESIS, "created_by": author, "created_time": _now(),
    })
    # ── publish = commit + canonicalize + hash + sign (AGENT-HOST-DESIGN.md D10) ──
    # Bundles are reached in peer mantles and cached rather than installed, so a peer verifies an
    # operator it did not author and did not fetch from its author. An operator that cannot be
    # signed here stays unsigned — a draft, admissible nowhere — and `signed` plus `signature_note`
    # carry that in the result, so the caller reads the signing state rather than assuming it.
    signed, sign_note = False, "no keys_dir on this store"
    keys_dir = getattr(store, "keys_dir", None)
    if keys_dir is not None:
        try:
            from prism.trust import opsign
            priv, _pub = opsign.authority_key(keys_dir, create=True)
            if priv is not None:
                doc = opsign.sign_operator(doc, priv)
                signed, sign_note = True, "signed"
            else:
                sign_note = "no signing key available"
        except Exception as e:                      # never let signing break the definition
            sign_note = "%s: %s" % (type(e).__name__, str(e)[:120])
    store.artifacts.put_artifact(doc)
    # The result carries no `spec_hash`. The artifact does not stamp one (see `crystal/evolution.py`
    # and `opsign.sign_operator`), and a result key that is structurally always null is worse than
    # an absent one: a caller cannot tell "not computed" from "computed as nothing".
    #
    # The question the field answered — "is this the same behaviour?" — is answerable on demand:
    # `evolution.spec_hash(doc)` computes it from the spec the artifact already carries. A value
    # that can always be recomputed needs no returning, storing, or migrating.
    return {"defined": True, "id": op_id, "kind": kind, "updated": updating,
            "invokable_now": True, "note": "live immediately — no restart",
            "signed": signed, "signature_note": sign_note}


def _run_source_spec(store, name: str, spec: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """Generic SOURCE interpreter: ingest any HF parquet dataset from a declarative spec — no
    per-source code. spec: {repo, config, stage, title_field?, text_field?, min_len?, columns?,
    id_prefix?, cite_meta?, offer?}. So a new dataset is a data definition, not a code change."""
    repo = spec.get("repo"); config = spec.get("config"); stage = spec.get("stage", "staging")
    if not repo or not config:
        return {"ingested": 0, "error": "spec needs {repo, config}"}
    tf = spec.get("title_field", "title"); xf = spec.get("text_field", "text")
    minlen = int(spec.get("min_len", 120)); idp = spec.get("id_prefix", name.replace("op.source.", ""))

    def _to_record(row, i):
        title = (row.get(tf) or "").strip()
        text = (row.get(xf) or "").strip()
        if not text or len(text) < minlen:
            return None
        body = f"{title}\n\n{text}" if title else text
        return {"id": f"{idp}-{row.get('id', i)}", "content": body,
                "content_type": "text/markdown", "meta": {"title": title}}

    src_name = name.replace("op.source.", "")
    return ingest_sharded(
        store, repo=repo, config=config, name=src_name, stage=stage,
        cite_meta=spec.get("cite_meta", {"dataset": f"{repo}:{config}", "hf_path": repo, "config": config}),
        offer=spec.get("offer", f"ingests {repo}:{config}"),
        to_record=_to_record, columns=spec.get("columns"),
        describe_workers=int(args.get("describe_workers", _DESCRIBE_WORKERS)),
        max_shards=(None if args.get("shards") else args.get("max_shards", 1)),
        only_shards=args.get("shards"))


# ── provenance enforcement — NO artifact without a citation (§12) ────────────
P_UNKNOWN = "unknown"
CITE_UNKNOWN = "cite.unknown"          # honest anchor: "source was never recorded"


def audit_provenance(store) -> Dict[str, Any]:
    """Scan the committed corpus for the §12 invariant: every artifact carries a `cited_from`
    AND a `provenance`. Returns counts + a sample of any violators. Deterministic."""
    total = missing_cite = missing_prov = 0
    by_type: Dict[str, int] = {}
    samples: List[Dict[str, Any]] = []
    for a in store.artifacts.list_artifacts(state="committed"):
        total += 1
        mc = not a.get("cited_from")
        mp = not a.get("provenance")
        if mc:
            missing_cite += 1
        if mp:
            missing_prov += 1
        if mc or mp:
            ct = a.get("content_type", "?")
            by_type[ct] = by_type.get(ct, 0) + 1
            if len(samples) < 20:
                samples.append({"id": a["id"], "content_type": ct,
                                "missing_cite": mc, "missing_prov": mp})
    return {"total": total, "missing_cited_from": missing_cite, "missing_provenance": missing_prov,
            "by_type": by_type, "samples": samples,
            "invariant_holds": missing_cite == 0 and missing_prov == 0}


def backfill_provenance(store, *, apply: bool = False, author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """Enforce the invariant: give every artifact a citation. SYSTEM artifacts (ontology /
    operators / collections / citations) anchor on cite.genesis (HUMAN_VALIDATED); any content
    artifact missing a source is anchored on cite.unknown at the UNKNOWN rung — honest, not
    fabricated (a real source can later re-rung it). Dry-run by default."""
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    if not is_bootstrapped(store):
        bootstrap(store, author=author)
    if apply and store.artifacts.get_artifact(CITE_UNKNOWN) is None:
        _mint(store, {
            "id": CITE_UNKNOWN, "content_type": CITATION_CONTENT_TYPE,
            "context": {"dataset": "unknown source", "kind": "citation", "provenance": P_UNKNOWN,
                        "role": "anchor for artifacts whose source was never recorded"},
            "lemmas": ["unknown", "source"], "content": "provenance was never recorded",
            "provenance": P_UNKNOWN, "cited_from": CITE_GENESIS, "created_by": author})
    fixed_system = fixed_content = 0
    for a in store.artifacts.list_artifacts(state="committed"):
        if a.get("cited_from") and a.get("provenance"):
            continue
        ct = a.get("content_type", "")
        if ct in _SYSTEM_CTS:
            a.setdefault("cited_from", CITE_GENESIS)
            if not a.get("provenance"):
                a["provenance"] = P_HUMAN
            fixed_system += 1
        else:
            if not a.get("cited_from"):
                a["cited_from"] = CITE_UNKNOWN
            if not a.get("provenance"):
                a["provenance"] = P_UNKNOWN
            fixed_content += 1
        if apply:
            store.artifacts.put_artifact(a)
    res = {"applied": apply, "fixed_system": fixed_system, "fixed_content_as_unknown": fixed_content}
    if apply:
        from ember.runtime.runner import evolution
        evolution.record_invocation(store.artifacts, "op.provenance.audit", verified=True)
    return res


# ── ρ (compression ratio) + per-collection metrics (§3.4, §7) ────────────────
# ρ = bytes(generators + morphisms) / bytes(reconstructible corpus), per collection.
# The universe is *cooling* when ρ falls while query-coverage holds. At P0 nothing is
# consolidated, so every artifact is its own generator and ρ = 1.0 — the baseline. As
# op.consolidate.* draws `consolidates` edges (canonical -> member), the member becomes
# reconstructible (context + operator -> content) and stops counting as a generator, so
# generator_bytes falls and ρ drops below 1. This is the literal, measurable entropy gauge.


def _artifact_bytes(a: Dict[str, Any]) -> int:
    # Prefer `size` — the full plaintext content length recorded at ingest, i.e. the real bytes
    # behind a content_ref. Fall back to the inline content length for artifacts stored inline
    # (WordNet definitions, operators).
    s = a.get("size")
    if isinstance(s, int) and s > 0:
        return s
    c = a.get("content") or ""
    if isinstance(c, (bytes, bytearray)):
        return len(c)
    return len(str(c).encode("utf-8", "ignore"))


def _consolidated_members(store) -> set:
    """Every artifact id that has an incoming `consolidates` edge — i.e. is subsumed by a
    canonical and is therefore reconstructible rather than a generator. One query, not N.

    `dst_ids_by_label` is the typed form: one indexed read of `ix_e_dst`, ids only. It is typed
    because an unrecognised query string would come back empty on a backend that does not
    understand it, and an empty result here is indistinguishable from a consolidated-nothing corpus:
    every artifact would look like a generator, `generator_bytes == total_bytes`, and ρ would read
    exactly 1.0 — the baseline — over a corpus that may be consolidated. ρ is the entropy gauge the
    whole curriculum is steered by."""
    if store.graph is None:
        return set()
    dsts = _typed(store.graph, "dst_ids_by_label")
    if dsts is not None:
        try:
            return set(dsts("consolidates"))
        except Exception:
            return set()
    # Not typed (the in-memory fakes). There is no whole-graph primitive on the GraphStore ABC, so
    # fall back to the raw edge list where one is exposed. The fallback stays typed too: an
    # openCypher string handed to an untyped `.query` would answer with an empty set.
    edges = getattr(store.graph, "edges", None)
    if edges is not None:
        try:
            return {e[1] for e in edges if len(e) > 2 and e[2] == "consolidates"}
        except Exception:
            return set()
    return set()


# structural artifacts are not "content to illuminate": they're pre-keyed or bare operators.
# They still count toward ρ bytes, but not toward keyed_coverage (matches improve.metrics).
_NON_CONTENT_CT = {OPERATOR_CONTENT_TYPE, SOURCE_CONTENT_TYPE}


def _is_keyed(a: Dict[str, Any]) -> bool:
    return bool(a.get("lemmas"))            # lemmas are the retrieval surface (§3.3)


def collection_metrics(store, collection_id: str, *, consolidated: Optional[set] = None) -> Dict[str, Any]:
    """Metrics for one collection's DIRECT members (indexed collection_id filter). Carries
    coverage, dark-matter, and ρ so /status shows entropy per subject falling over time."""
    if consolidated is None:
        consolidated = _consolidated_members(store)
    arts = list(store.artifacts.list_artifacts(collection_id=collection_id))
    n = len(arts)
    total_bytes = sum(_artifact_bytes(a) for a in arts)
    gen_bytes = sum(_artifact_bytes(a) for a in arts if a["id"] not in consolidated)
    content = [a for a in arts if a.get("content_type") not in _NON_CONTENT_CT]
    keyed = sum(1 for a in content if _is_keyed(a))
    dark = len(content) - keyed
    return {
        "collection": collection_id,
        "artifacts": n,
        "bytes": total_bytes,
        "generator_bytes": gen_bytes,
        "consolidated": sum(1 for a in arts if a["id"] in consolidated),
        "dark_matter": dark,
        "keyed_coverage": round(keyed / len(content), _METRIC_DECIMALS) if content else None,
        "rho": round(gen_bytes / total_bytes, _METRIC_DECIMALS) if total_bytes else None,
    }


_METRICS_CACHE: Dict[str, Any] = {"ts": 0.0, "val": None}
_METRICS_TTL = 90.0         # seconds — exceeds the serve refresher interval (45s) so the cache
                            # stays continuously warm: inline /status + /health then hit the cache
                            # rather than cold-scanning the full corpus under write load, which is
                            # what makes /status responsive past a few hundred thousand artifacts.
                            # Freshness holds: the background refresher repopulates every 45s.


_METRICS_SAMPLE = 20000


def all_metrics(store, *, cache_ok: bool = True, sample: Optional[int] = _METRICS_SAMPLE) -> Dict[str, Any]:
    """ρ + coverage per collection + global rollup + provenance health, from a bounded sample.

    `sample=None` requests the exhaustive census. It is O(corpus) — the shape that pinned T5 at 713%
    CPU — so it belongs to a deliberate, operator-invoked audit rather than to any loop, timer,
    heartbeat or request path."""
    import time
    # The cache key is the store object rather than id. CPython reuses freed addresses, so an
    # id-keyed cache can serve one store another store's metrics for the whole TTL (recorded in
    # MESH.md). A weak ref holds the real object and cannot collide.
    if (cache_ok and _METRICS_CACHE["val"] is not None
            and _METRICS_CACHE.get("store") is not None
            and _METRICS_CACHE["store"]() is store.artifacts
            and (time.time() - _METRICS_CACHE["ts"]) < _METRICS_TTL):
        return _METRICS_CACHE["val"]
    consolidated = _consolidated_members(store)
    per: Dict[str, Dict[str, int]] = {}
    total = gen = darK = keyed = n = content_n = 0
    miss_cite = miss_prov = 0
    # Sampled, rather than a full-byte scan. ρ, keyed_coverage and dark_matter are ratios, and a
    # ratio does not need the population: a bounded sample estimates it to within a fraction of a
    # percent at 1/300th of the cost.
    # Walking every committed row (6.19M), hydrating each into a dict and summing its bytes, takes
    # longer than the 300s loop that drove it, so scans stacked on each other. Measured on T5: the
    # store pinned at ~713% CPU with 4 request threads and 4 GC threads saturated while the
    # aggregator managed ~114 docs/sec, and stopping the aggregator changed nothing.
    # The result is labelled `sampled` with its n, because an estimate presented as a census is the
    # same class of error as an unmeasured value presented as zero.
    limit = None if sample is None else int(sample)
    for a in store.artifacts.list_artifacts(state="committed", limit=limit):
        b = _artifact_bytes(a)
        isgen = a["id"] not in consolidated
        iscontent = a.get("content_type") not in _NON_CONTENT_CT
        kf = _is_keyed(a)
        total += b; n += 1
        if isgen:
            gen += b
        if iscontent:
            content_n += 1
            if kf:
                keyed += 1
            else:
                darK += 1
        if not a.get("cited_from"):
            miss_cite += 1
        if not a.get("provenance"):
            miss_prov += 1
        cid = a.get("collection_id")
        if cid:
            d = per.setdefault(cid, {"artifacts": 0, "bytes": 0, "generator_bytes": 0,
                                     "content": 0, "keyed": 0, "consolidated": 0})
            d["artifacts"] += 1; d["bytes"] += b
            if isgen:
                d["generator_bytes"] += b
            else:
                d["consolidated"] += 1
            if iscontent:
                d["content"] += 1
                if kf:
                    d["keyed"] += 1
    cols = {}
    for cid, d in per.items():
        c = d.pop("content")
        cols[cid] = {"collection": cid, "artifacts": d["artifacts"], "bytes": d["bytes"],
                     "generator_bytes": d["generator_bytes"], "consolidated": d["consolidated"],
                     "dark_matter": c - d["keyed"],
                     "keyed_coverage": round(d["keyed"] / c, _METRIC_DECIMALS) if c else None,
                     "rho": round(d["generator_bytes"] / d["bytes"], _METRIC_DECIMALS) if d["bytes"] else None}
    # Truncated, rather than merely "a limit was passed". If the corpus is smaller than the sample
    # the walk covered all of it, and that is a census — a small node (or a test fixture) gets exact
    # answers rather than being told its own totals are estimates.
    truncated = limit is not None and n >= limit
    val = {
        # When truncated, `artifacts`/`bytes` are the sample's rather than the corpus's, and are
        # labelled so, so an estimate reads as one. The ratios below are the meaningful outputs.
        "artifacts": n, "bytes": total, "generator_bytes": gen, "dark_matter": darK,
        "sampled": truncated, "sample_n": n if truncated else None,
        "keyed_coverage": round(keyed / content_n, _METRIC_DECIMALS) if content_n else None,
        "rho": round(gen / total, _METRIC_DECIMALS) if total else None,
        "collections": {k: cols[k] for k in sorted(cols)},
        # A truncated sample carries no verdict over the whole corpus: 0 violations in 20k rows says
        # nothing about the remaining 6.17M. Report unknown, which is what was measured.
        "provenance": {"invariant_holds": None if truncated
                       else (miss_cite == 0 and miss_prov == 0),
                       "sampled": truncated,
                       "missing_cited_from": miss_cite, "missing_provenance": miss_prov},
    }
    import weakref as _wr
    try:
        _METRICS_CACHE.update(ts=time.time(), val=val, store=_wr.ref(store.artifacts))
    except TypeError:
        _METRICS_CACHE.update(ts=time.time(), val=val, store=None)
    return val


# ── consolidation — the compression engine (§7) ──────────────────────────────
# Consolidation is the search for the smallest category equivalent to the corpus on the
# queries we care about. Two operators:
#   op.consolidate.nearvdup  — quotient by near-identity (shingled Jaccard). Emits a
#                              `consolidates` edge canonical -> member; the member is archived
#                              but its content is KEPT (lossless-with-pointer, §12).
#   op.consolidate.colimit   — a diagram of the SAME concept across sources (synset ↔ entity ↔
#                              concept) has a colimit: one canonical Concept with morphisms from
#                              each source. We store the colimit + the `consolidates` morphisms;
#                              the sources become derivable (context + operator -> content).
# Both LOWER ρ (members stop counting as generators) without losing information.

CONSOLIDATE_OPS = [
    ("op.consolidate.nearvdup", "quotients near-identical artifacts (shingled Jaccard) to a "
     "canonical, emitting `consolidates` edges — lossless-with-pointer; lowers ρ"),
    ("op.consolidate.colimit", "collapses a same-concept diagram (synset↔entity↔concept) to its "
     "colimit (a canonical Concept) with `consolidates` morphisms from each source; lowers ρ"),
    ("op.consolidate.crosswalk", "sweeps the corpus for same-concept diagrams across sources "
     "(synset word == article title) and colimits each — the cross-source compression that first "
     "drives ρ below 1.0"),
]


def register_consolidate_operators(store, *, author: str = DEFAULT_AUTHOR) -> int:
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    from ember.runtime.runner import evolution
    # Same custody rule as `register_control_operators` — established only when none is in scope,
    # so a request path keeps its own identity. See `ember/custody.py` for why that guard is not
    # optional on a function reachable from both background and request work.
    from ember.custody import system_custody_if_unowned
    with system_custody_if_unowned("ember.register_consolidate_operators"):
        for name, offer in CONSOLIDATE_OPS:
            store.artifacts.put_artifact(evolution.preserve_fitness(store.artifacts, {
                "id": name, "content_type": OPERATOR_CONTENT_TYPE, "state": "committed",
                "context": offer, "content": f"consolidation operator {name}: {offer}",
                "provenance": P_HUMAN, "cited_from": CITE_GENESIS,
                "created_by": author, "created_time": _now(),
            }))
    return len(CONSOLIDATE_OPS)


def _pick_canonical(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The representative of an equivalence class — chosen arbitrarily, and saying so.

    There is no validity question here. Its one caller groups by `content_ref`, so every member is
    byte-identical: the same bytes, differing only in id and metadata. Nothing about one copy is
    truer than another, so ranking by provenance rung would rank a difference that does not exist.

    So the lowest id wins. Arbitrary, and that is the honest word for it — deterministic and stable
    across runs and nodes, so two nodes consolidating the same class pick the same representative,
    which a metadata-dependent rank could not promise. Nothing is lost either way: the non-canonical
    members are archived with `consolidated_by` and their `content_ref` retained.
    """
    return min(group, key=lambda a: a["id"])


def consolidate_nearvdup(store, *, content_type: str = "text/markdown",
                         collection_id: Optional[str] = None,
                         threshold: Optional[float] = None,
                         apply: bool = False, limit: int = 60000,
                         max_candidate_edges: int = 50000,
                         author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """Quotient byte-identical content to canonicals, and record near-duplicates as observations.

    Two separate paths, because they carry different risk:

      · **Consolidate** — members sharing a `content_ref` (`cas/<sha256-of-plaintext>`) are the same
        bytes. The member is archived + darkened and stops counting as a generator: real
        compression, ρ genuinely falls, and it is reconstructible from its canonical exactly.
      · **Observe** — MinHash-LSH candidates that are merely similar get a `near_dup_candidate` edge
        carrying the measured score. Deterministic, model-free, ~O(n). Full recall, and no authority
        to destroy.

    Similarity is observed rather than archived, and the threshold is derived rather than picked.
    `threshold=None` means the estimator's own resolution (`minhash.merge_boundary` — 0.99225 at
    k=128): it is `1 - 1/(k+1)`, a function of how many hashes the instrument computes, so a better
    instrument tightens it automatically.

    Measured over 1,661 LSH candidate pairs, the score distribution has no valley — it decays
    smoothly 0.2→0.94 — so "near-duplicate" and "merely similar" are one continuum and any cut
    through it is ill-posed. The estimator's own resolution is ±0.0316 at J=0.85, coarser than the
    distance between the constants it would be compared against. Merging only when the read cannot
    resolve two artifacts apart at the noise floor yields Ĵ ≳ 0.97, and zero candidate pairs reached
    even 0.95 — the criterion selects exact identity, which `content_ref` provides for free.

    Private/no_share stays out of consolidation. Reports ρ before/after, and reports
    `skipped_no_signature` so `near_dup_candidate: 0` is distinguishable from "we could not tell"."""
    from prism import minhash
    from ember.runtime.runner import evolution
    register_consolidate_operators(store, author=author)
    from mantle.db import access
    # `limit` bounds how much of the corpus is examined, and this path archives, so the bound is
    # reported alongside the counts: taking `limit + 1` makes reaching it observable, which is what
    # proves there was more. Measured on the live shard, more than 60,000 `text/markdown` rows exist
    # (316,421 in total), so the default cap bites — a run sees at most ~19% — and a caller reading
    # "consolidated N groups" needs `truncated` beside it. The cap stays: it bounds real work on a
    # 5.8 GB store.
    _scan = [a for a in store.artifacts.list_artifacts(content_type=content_type,
                                                       collection_id=collection_id, state="committed")
             if access.is_public(store, a)][:limit + 1]   # never consolidate non-public data up
    truncated = len(_scan) > limit
    arts = _scan[:limit]
    # Reported in the answer rather than logged. This module reports through return values and
    # prints only from `__main__`, and the caller that acts on the counts is the caller that needs
    # the bound beside them.
    _bound = {"scope": len(arts), "limit": limit, "truncated": truncated}
    if len(arts) < 2:
        return dict(_bound, groups=0, consolidated=0, applied=apply)
    rho_before = all_metrics(store)["rho"] if apply else None
    # Signatures come from the artifact's stored `minhash` field, so this path fetches no content.
    #
    # An empty signature is the absence of a similarity measurement rather than a claim of one.
    # `minhash.signature("")` returns 128 zeros, and `estimated_jaccard(zeros, zeros)` is 1.0 — so
    # two artifacts nothing is known about would score as a perfect match. An artifact with no
    # usable signature is therefore excluded from consolidation entirely: an unmerged row costs
    # nothing, while merging on no evidence is unrecoverable at the corpus level. Without that
    # exclusion, LSH would bucket a whole collection into one group and `apply=True` would archive
    # the corpus as duplicates of one row while ρ "improves" and the operator reports compression.
    #
    # Signatures are stored-only, with no preview fallback. `content` is a 300-character preview on
    # legacy rows (§6A rules it a hard out) and absent on new ones, so deriving a signature from it
    # would do two incompatible things at once:
    #   1. compare preview-derived signatures against full-content-derived ones as if they were the
    #      same measurement; and
    #   2. bias toward false merges on exactly the templated stubs this operator targets, because
    #      templated content shares its opening 300 characters by construction.
    # An artifact with no stored signature is simply not comparable, and that is what is reported.
    #
    # Current state: `minhash` is written nowhere in the tree — the read on the line below is its
    # only occurrence — so on a content-store-backed box no artifact carries one and the advisory
    # arm reports that it had no comparison to make. The merge arm is unaffected: `content_ref`
    # identity is measurable on every row.
    def _sig(a):
        s = a.get("minhash")
        return list(s) if s and any(s) else None   # all-zero == no evidence == not comparable
    pairs = [(a, _sig(a)) for a in arts]
    unsigned = sum(1 for _, s in pairs if s is None)

    # ── the merge set is exact content identity, rather than a similarity estimate ────────────
    # Measured over one signature pool:
    #   · the null is a spike at zero — 300k random corpus pairs, 99.82% score exactly 0.0;
    #   · the instrument's own resolution is SE = sqrt(J(1-J)/128) = ±0.0316 at J=0.85, so 0.82,
    #     0.85 and 0.88 are one number to a 128-hash signature;
    #   · the LSH candidate distribution has no valley. Over 1,661 candidate pairs it decays
    #     smoothly and monotonically 0.2 -> 0.94, so "near-duplicate" and "merely similar" are one
    #     continuum and a threshold on this axis is ill-posed rather than merely untuned.
    # The standing criterion — merge only when the read cannot resolve the two apart at the corpus
    # noise floor — therefore evaluates to Ĵ within one resolution unit of 1.0 (≳0.97), and zero of
    # those 1,661 pairs reached even 0.95. The criterion selects exact identity on its own.
    # `content_ref` is `cas/<sha256-of-plaintext>`, so identity is available exactly, free and
    # indexed — no estimator, no error bar, and the confabulation guard (the generator regenerates
    # the member) is satisfied byte-for-byte rather than "within resolution".
    by_ref: Dict[str, List[Dict[str, Any]]] = {}
    for a in arts:
        ref = a.get("content_ref")
        if ref:
            by_ref.setdefault(ref, []).append(a)
    exact_groups = [g for g in by_ref.values() if len(g) > 1]

    n_groups = consolidated = 0
    edges: List[tuple] = []
    archived: List[Dict[str, Any]] = []
    for members in exact_groups:
        n_groups += 1
        canon = _pick_canonical(members)          # safe: every member is BYTE-IDENTICAL to canon
        for a in members:
            if a["id"] == canon["id"]:
                continue
            consolidated += 1
            if apply:
                edges.append((canon["id"], a["id"], "consolidates",
                              {"via": "op.consolidate.nearvdup", "match": "content_ref_exact",
                               "content_ref": a.get("content_ref")}))
                doc = dict(a); doc["state"] = "archived"; doc["consolidated_by"] = canon["id"]
                archived.append(doc)               # lossless: content_ref retained, member kept

    # ── similarity is an observation, and archiving is a separate action ──────────────────────
    # LSH keeps its full recall value; the authority to destroy stays with the exact-identity arm.
    # Candidates that are not byte-identical become `near_dup_candidate` edges carrying the measured
    # score, so the signal is retained and reviewable while a ±0.03 estimate archives nothing.
    # The label is `near_dup_candidate` rather than `similar_to`, which is already taken: it is a
    # WordNet adjective-cluster relation and is symmetric, while these edges are directional
    # (base -> other), so reusing it would fail a symmetry audit and mix migration bookkeeping into
    # a real linguistic relation.
    # The asymmetry is the governing rule: an unmerged pair costs a missed duplicate, which stays in
    # the corpus and can be merged later, while merging on no evidence archives unrelated content
    # and is unrecoverable. `threshold` survives as a recall floor on this non-destructive path,
    # distinct from an acceptance criterion for archiving.
    scored = [(a, s) for a, s in pairs if s is not None]
    similar_edges: List[tuple] = []
    # Candidate pairs scale ~n^1.83 rather than linearly. Measured over one signature pool, doubling
    # n multiplied candidate pairs by 3.2-4.0x (5k->23, 10k->143, 20k->536, 40k->2130, 80k->6771).
    # Extrapolated to the 6.11M-blob corpus that is ~2e7 candidate pairs — a linear estimate gives
    # 1.69e5, low by ~116x.
    #
    # Emitting one edge per candidate would add 2e7 edges to a graph holding 273,000 real ones
    # (154,506 + 118,134 colimit) — a 70x expansion of derived data, to store a relation LSH can
    # regenerate on demand in minutes. That is a second copy of the index, rather than provenance.
    #
    # So the budget below bounds emission and reports what it dropped, which keeps a capped run
    # distinguishable from a complete one.
    edge_budget = int(max_candidate_edges)
    candidates_seen = 0
    if len(scored) >= 2:
        idx_list = [a for a, _ in scored]
        sigs = [s for _, s in scored]
        exact_pairs = {frozenset((x["id"], y["id"]))
                       for g in exact_groups for x in g for y in g if x["id"] != y["id"]}
        for grp in minhash.group_signatures(sigs, threshold=threshold):   # LSH: ~O(n), no I/O
            members = [idx_list[i] for i in grp]
            base = members[0]
            for a in members[1:]:
                if frozenset((base["id"], a["id"])) in exact_pairs:
                    continue                       # already consolidated on exact identity
                candidates_seen += 1
                if len(similar_edges) >= edge_budget:
                    continue                   # counted, not emitted -- reported in the result
                similar_edges.append((base["id"], a["id"], "near_dup_candidate",
                                      {"via": "op.consolidate.nearvdup",
                                       "score": minhash.estimated_jaccard(
                                           _sig(base) or [], _sig(a) or []),
                                       "estimator": "minhash-128",
                                       "standard_error": 0.0316,
                                       "note": "OBSERVATION ONLY — never grounds for archiving"}))
    if apply and similar_edges:
        store.graph.add_edges(similar_edges, batch=1000)
    if apply and consolidated:
        # Provenance first — the invariant is that every archived result carries its `consolidates`
        # edge. Drawing the edges before archiving means a crash after this point leaves the member
        # un-archived: still a live generator, re-archived idempotently next run, and always
        # attached to its canonical. Edge-then-archive.
        store.graph.add_edges(edges, batch=1000)
        store.artifacts.put_many(archived, batch=500)
        evolution.record_invocation(store.artifacts, "op.consolidate.nearvdup", verified=True)
    result = {"groups": n_groups, "consolidated": consolidated, "applied": apply,
              # `truncated` rides with the counts rather than beside them, for the same reason as
              # `signatures` below: a number whose coverage is unstated reads as store-wide.
              # `truncated: True` means these counts describe a prefix of the corpus.
              **_bound,
              # Report what was not measured. If most of the scope carried no signature,
              # `near_dup_candidate: 0` means "we could not tell" rather than "nothing is similar",
              # and the caller reads which. `consolidated` is independent of signatures: exact
              # `content_ref` identity is measurable on every row.
              "merge_basis": "content_ref_exact",
              "compared": len(scored), "skipped_no_signature": unsigned,
              "near_dup_candidate_edges": len(similar_edges),
              # Reported always, including when nothing was dropped, so a caller can distinguish
              # "LSH found this many" from "LSH found more and emission stopped at the budget".
              "near_dup_candidates_found": candidates_seen,
              "near_dup_candidates_dropped": max(0, candidates_seen - len(similar_edges)),
              "near_dup_edge_budget": edge_budget,
              "similarity_is_advisory": True,
              "rho_before": rho_before}
    # Say so when the similarity read could not be taken at all. `consolidated` is independent of
    # signatures, so a scope where nothing was comparable would otherwise return
    # `near_dup_candidate_edges: 0` with no reason, which reads identically to "nothing is similar".
    # The merge arm is unaffected; it is the advisory arm that had nothing to read, and the `reason`
    # is what keeps those two distinguishable.
    if len(scored) < 2:
        result["reason"] = ("no artifact carried a usable minhash signature — nothing comparable "
                            "(similarity not measured; merges, if any, are content_ref-exact)")
    if apply and consolidated:
        result["rho_after"] = all_metrics(store, cache_ok=False)["rho"]
    return result


def consolidate_colimit_derived(store, concept_id: str, *, label_keyed: bool = False,
                                include_unanchored: bool = False, apply: bool = False,
                                author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """The colimit with the diagram derived — pass a concept, and the members are measured.

    `consolidate_colimit` below has the caller supply which artifacts name the same concept, which
    is the question itself; that is why it has no production callers. This derives the diagram from
    the store by measurement (`ember.consolidate.diagram.derive_diagram`: the artifacts whose
    evidence the concept's own band absorbs entirely, conservation-certified, no threshold), takes
    the colimit that carries every member's provenance, position and mass
    (`ember.consolidate.colimit`), and writes only a merge whose certificate balances.

    Dry-run by default. Returns the derived diagram, the built colimit and its certificate."""
    from ember.consolidate import colimit as _col
    from ember.consolidate import diagram as _dia
    author = _author_ref(store, author)
    cache: Dict[str, Any] = {}
    dia = _dia.derive_diagram(store, concept_id, label_keyed=label_keyed,
                              include_unanchored=include_unanchored, anchor_cache=cache)
    dia.pop("reads", None)                       # the per-candidate reads are large; kept out of the result
    if len(dia["members"]) < 2:
        return {"applied": False, "diagram": dia,
                "reason": "the evidence separates this concept from every candidate — a diagram of one"}
    built = _col.colimit(store, dia["members"], label_keyed=label_keyed,
                         include_unanchored=include_unanchored, author=author, anchor_cache=cache)
    out = {"diagram": dia, "id": built.get("id"), "members": built.get("members"),
           "certificate": built.get("certificate"), "acceptable": built.get("acceptable"),
           "position_recovery": built.get("position_recovery"), "applied": False}
    if apply:
        out.update(_col.apply_colimit(store, built))
        if out.get("applied"):
            from ember.runtime.runner import evolution
            evolution.record_invocation(store.artifacts, "op.consolidate.colimit", verified=True)
    return out


def consolidate_colimit(store, diagram_ids: List[str], *, canonical_id: Optional[str] = None,
                        concept_lemmas: Optional[List[str]] = None, apply: bool = False,
                        author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """Collapse a same-concept diagram to its colimit. `diagram_ids` are artifacts naming the
    SAME concept across sources (e.g. wn-dog.n.01, an entity, a concept). If `canonical_id` is
    given it becomes the colimit; otherwise a new Concept artifact is minted. Draws
    `consolidates` canonical -> member morphisms (members remain, become reconstructible).

    The diagram is supplied by the caller rather than measured, and the artifact this mints is a
    pointer stub: no positions, no mass, no `(count, sum)`, and an id built with `_now_hash`, so
    taking the same colimit twice mints two artifacts. `consolidate_colimit_derived` above is the
    one to call; this is kept for its existing (test-only) callers."""
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    register_consolidate_operators(store, author=author)
    from mantle.db import access
    members = [store.artifacts.get_artifact(i) for i in diagram_ids]
    # Isolation: the colimit covers public artifacts only, so a grant-gated one stays out even on an
    # errant manual call. Private data leaves its scope through op.share (consent + stake), which is
    # a separate path from compression.
    members = [m for m in members if m and access.is_public(store, m)]
    if len(members) < 2:
        return {"applied": apply, "members": len(members), "reason": "need >=2"}
    if canonical_id:
        canon_id = canonical_id
    else:
        lem = concept_lemmas or sorted({l for m in members for l in (m.get("lemmas") or [])})[:12]
        canon_id = "concept-" + _now_hash(diagram_ids)
        if apply:
            _mint(store, {
                "id": canon_id, "content_type": "application/x-concept",
                "context": {"name": (lem[0] if lem else canon_id), "kind": "concept",
                            "colimit_of": diagram_ids, "provenance": P_HYPOTHESIS},
                "lemmas": lem, "content": "canonical concept (colimit of %d sources)" % len(members),
                "provenance": P_HYPOTHESIS, "collection_id": "subjects",
                "collections": ["subjects"], "via": "op.consolidate.colimit",
                "created_by": author,
            })
    drawn = 0
    for m in members:
        if m["id"] == canon_id:
            continue
        if apply and _ensure_edge(store, canon_id, m["id"], "consolidates",
                                  {"via": "op.consolidate.colimit"}):
            drawn += 1
    if apply and drawn:
        from ember.runtime.runner import evolution
        evolution.record_invocation(store.artifacts, "op.consolidate.colimit", verified=True)
    return {"applied": apply, "canonical": canon_id, "members": len(members),
            "consolidates_edges": drawn, "rho": all_metrics(store)["rho"] if apply and drawn else None}


def consolidate_crosswalk(store, *, apply: bool = False, limit: int = 20000,
                          author: str = DEFAULT_AUTHOR) -> Dict[str, Any]:
    """Discover same-concept diagrams ACROSS sources by exact keyed match — a WordNet synset
    whose word == a Wikipedia article's title — and collapse each to its colimit (a canonical
    Concept) with `consolidates` morphisms (§7). Deterministic + keyed (no models): the match is
    an exact string equality on the keyed `word`/`title`, not similarity. This is the first place
    ρ falls below 1.0 — the concept is stored once, its per-source representations subsumed.

    Returns diagram count + ρ before/after. Dry-run by default."""
    # The author CLAIM resolves to its PERSON artifact here, so `created_by` below is a
    # vertex reference and not a dangling string (contract 2.1). See `_author_ref`.
    author = _author_ref(store, author)
    register_consolidate_operators(store)
    # index synsets by their primary word (the keyed lemma)
    word_to_syn: Dict[str, List[str]] = {}
    for a in store.artifacts.list_artifacts(content_type=WORDNET_CONTENT_TYPE):
        w = (a.get("word") or "").strip().lower()
        if w:
            word_to_syn.setdefault(w, []).append(a["id"])
    if not word_to_syn:
        return {"diagrams": 0, "applied": apply, "note": "no wordnet synsets"}
    rho_before = all_metrics(store, cache_ok=False)["rho"] if apply else (_METRICS_CACHE.get("val") or {}).get("rho")
    # Scan wiki articles by content type, matching title -> synset word; private/no_share rows stay
    # out. Content type is the invariant a wiki article always carries (text/markdown, in any
    # collection), so the crosswalk is discovered on a flattened corpus as well as a properly-staged
    # one — the lattice migration collapses the whole curriculum into one `foundation` collection,
    # where a scan keyed on stage names ("stage.1.grammar", "stage.2.world") finds nothing while
    # 41,103 title==synset-word alignments sit in `foundation`. The match itself is exact keyed
    # title == synset word.
    from mantle.db import access
    diagrams: List[Tuple[str, str, List[str]]] = []
    seen_articles: set = set()
    for a in store.artifacts.list_artifacts(content_type="text/markdown"):
        if not access.is_public(store, a):           # non-public (grant-gated) never crosswalks up
            continue
        title = (a.get("title") or "").strip().lower()
        if not title or title in seen_articles or title not in word_to_syn:
            continue
        seen_articles.add(title)
        diagrams.append((title, a["id"], word_to_syn[title]))
        if len(diagrams) >= limit:
            break
    if not apply:
        return {"diagrams": len(diagrams), "applied": False, "rho_before": rho_before}

    # Batched apply — one put_many for concepts, one add_edges for morphisms. Per-diagram minting
    # with a per-edge cypher and a per-diagram ρ scan is ~1000x slower. This also instruments the
    # holographic measurement: bytes moved from bulk to boundary (the "radiation" this
    # consolidation sheds).
    from ember.runtime.runner import evolution
    concepts: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
    member_ids: set = set()
    for word, art_id, syn_ids in diagrams:
        members = [art_id] + syn_ids
        canon = "concept-" + _now_hash(members)
        concepts.append({
            "id": canon, "content_type": "application/x-concept", "state": "committed",
            "context": {"name": word, "kind": "concept", "colimit_of": members,
                        "sources": ["wikipedia", "wordnet"], "provenance": P_HYPOTHESIS},
            "lemmas": [word], "content": f"canonical concept '{word}' (colimit of {len(members)} sources)",
            "collection_id": "subjects", "collections": ["subjects"],
            "cited_from": CITE_GENESIS, "via": "op.consolidate.crosswalk",
            "provenance": P_HYPOTHESIS, "created_by": author, "created_time": _now()})
        for mid in members:
            # `aligns` rather than `consolidates`: a synset and an article are the same concept
            # carrying distinct content, so the member stays un-reconstructible from a concept stub.
            # This is a semantic index (an interlingual pivot), and it leaves ρ unchanged. Genuine
            # compression — archive plus a ρ drop — belongs to reconstructible members: near-dup and
            # generative.
            edges.append((canon, mid, "aligns", {"via": "op.consolidate.crosswalk"}))
            member_ids.add(mid)
    store.artifacts.put_many(concepts, batch=500)
    store.graph.add_edges(edges, batch=1000)
    evolution.record_invocation(store.artifacts, "op.consolidate.crosswalk", verified=True)

    after = all_metrics(store, cache_ok=False)
    rho_after = after["rho"]
    return {"diagrams": len(diagrams), "applied": True, "concepts": len(concepts),
            "aligns_edges": len(edges), "members_aligned": len(member_ids),
            "note": "aligns is a semantic index, not compression — ρ intentionally unchanged; "
                    "real ρ drop comes from near-dup + generative reconstruction",
            "rho_before": rho_before, "rho_after": rho_after,
            "delta_rho": (round(rho_after - rho_before, 6) if (rho_before is not None and rho_after is not None) else None)}


def _now_hash(parts: List[str]) -> str:
    import hashlib
    return hashlib.sha256("|".join(sorted(parts)).encode()).hexdigest()[:20]


def _resolve(store, a: Dict[str, Any]) -> str:
    """Full content for consolidation (content store via content_ref, else inline)."""
    from mantle.shard import content as C
    return C.resolve_text(store, a)


def record_metrics(store, *, path: Optional[str] = None) -> Dict[str, Any]:
    """Compute all_metrics and append a snapshot line to the genesis metrics trend
    (`metrics.jsonl` in the node directory, beside `keys/`, unless `path` says otherwise).
    Returns the snapshot."""
    import json
    from pathlib import Path
    snap = all_metrics(store)
    snap["ts"] = _now()
    # A write, not a status read: with no explicit path and no node layout there is nowhere
    # this could honestly go, so say so instead of creating a directory nobody chose.
    if path:
        mp = Path(path)
    else:
        nd = _node_dir(store)
        if nd is None:
            raise RuntimeError(
                "no metrics path: this store has no keys_dir and EMBER_STORE_KEYS_DIR is "
                "unset, so the node directory cannot be located. Pass `path=`, or set "
                "EMBER_STORE_KEYS_DIR to this node's provisioned keys directory.")
        mp = nd / "metrics.jsonl"
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "a", encoding="utf-8") as f:
        f.write(json.dumps(snap) + "\n")
    return snap


if __name__ == "__main__":  # python -m ember.genesis [stage0] [metrics]
    import json
    import sys
    from mantle.shard.local_store import open_store
    s = open_store()
    print("store ready:", s.ready(), "| pre-count:", s.artifacts.count())
    result = bootstrap(s)
    print("bootstrap:", json.dumps(result))
    if "stage0" in sys.argv[1:]:
        t0 = time.time()
        r0 = ingest_stage0_wordnet(s)
        print("stage0:", json.dumps(r0), "| secs=%.1f" % (time.time() - t0))
    if "metrics" in sys.argv[1:]:
        snap = record_metrics(s)
        print("rho (global):", snap["rho"], "| coverage:", snap["keyed_coverage"],
              "| artifacts:", snap["artifacts"])
        for cid, m in snap["collections"].items():
            if m["artifacts"]:
                print(f"  {cid:22s} n={m['artifacts']:>7d} rho={m['rho']} cov={m['keyed_coverage']}")
    print("post-count:", s.artifacts.count())