"""GENESIS Stage-0 sources — OEWN 2024, CILI, ConceptNet 5.7, OMW (runbook §H1).

The four ingesters that complete the Stage-0 lexicon around the Princeton WordNet 3.0 spine
(`genesis.ingest_stage0_wordnet` — the pattern these follow, field for field):

  op.source.oewn        Open English WordNet 2024 — synset rows in the same shape as the
                        WordNet-3.0 `wn-*` rows (ct text/x-wordnet), plus an `ili` field
                        each row carries for interlingual pivoting.
  op.source.cili        the CILI interlingual index — ILI ids become typed `ili` pivot edges
                        between synsets that express the same concept. Edges only: where a
                        synset already exists no new vertex is minted (runbook H1.3).
  op.source.conceptnet  ConceptNet 5.7 assertions (~1 GB gz, ~34M lines) — streamed line by
                        line, filtered to /c/en/ endpoints, landed as Concept vertices plus
                        typed relation edges, cited; resumable via a line-count cursor.
  op.source.omw         Open Multilingual Wordnet — license-vetted per language against an
                        explicit allowlist (below); a language not on it is skipped and
                        logged, for clearing via the TRAINING-QUEUE prep column.

Registration (CURRICULUM stage-0 entries, SOURCE_INGESTERS, CONTROL_OPS) lives in
`genesis.py`; each source enters through the source triple (`genesis.mint_source_triple`:
cite.<name> + op.source.<name> + source.<name>) and every row is stamped with collection_id,
collections, cited_from, via, provenance OBSERVED and created_by.

External access is GET-only, per the read-only external-operator rule. `_download` below reuses
the fetch bundle's own guards (`op.fetch.get`: the verb hard-locked to GET, http/https only,
SSRF-guarded on every redirect hop); it differs only in streaming a dataset to disk rather than
taking the 5 MB in-memory text path. Downloads stage under the store's cache location
(EMBER_CACHE_DIR / downloads).

Checkpointing rides the store's own mechanisms:
  • completion markers are the per-shard checkpoint artifacts (`genesis._mark_shard_done` /
    `_shards_done`) — one marker per drained source, per language for OMW, which is what the
    CURRICULUM `done` predicates read;
  • the ConceptNet line cursor follows the content-cursor discipline (content_tier.py): a
    cursor advances only past lines that applied.
"""
from __future__ import annotations

from ember.authorship import DEFAULT_AUTHOR

import gzip
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ember import genesis as g

#: WN-LMF values that mean "no interlingual id", not an id. `in` = proposed for inclusion.
_ILI_SENTINELS = frozenset({"in", "n/a", "none", "-"})


def _ili_or_blank(value) -> str:
    """The ILI id, or `""` when the source says it has none. See the note at the Synset parse."""
    text = str(value or "").strip()
    return "" if text.lower() in _ILI_SENTINELS else text


def _sealed(store, docs):
    """Move each doc's `content` into the CAS, leaving the ADDRESS behind. Returns the same list.

    **All content goes to the CAS.** This writer did not, and it is the single largest reason the
    store held plaintext: measured 2026-08-25, 1,449,857 of the 1,457,067 inline bodies on 71/home
    were written from here.

    The precedent is in `genesis.ingest_dataset:660`, which has always done this — same store, same
    helper, four lines — while this file built `"content": defn` and called `put_many`. 📄
    `db/doc_boundary.encrypt_artifact_content` cannot close the gap: it is the ENTITY path and goes
    through the key oracle, which needs an acting principal the bulk loader does not have, so
    moving it into `put_artifact`/`put_many` fails every ingest write. `shard.content.put_content`
    keys off the KEYS DIRECTORY instead and therefore works in bulk.

    Applied at the `put_many` boundary rather than at each doc literal, because there are four doc
    shapes across three sources and only one place they all pass through. A doc that already
    carries a `content_ref`, or has no body, is returned untouched.

    Degrades rather than refuses: a store with no content tier keeps the body inline, which is what
    every row already does. A loader that dropped the corpus because the tier was unmounted would
    trade a confidentiality property for a data one.
    """
    content = getattr(store, "content", None)
    keys_dir = getattr(store, "keys_dir", None)
    if content is None or keys_dir is None:
        return docs
    try:
        from mantle.shard import content as C
    except Exception:                            # noqa: BLE001 - no CAS reachable; see above
        return docs
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        body = doc.get("content")
        if not body or doc.get("content_ref"):
            continue
        try:
            ref, size = C.put_content(content, keys_dir, str(body).encode("utf-8"),
                                      collection=doc.get("collection_id"))
        except Exception:                        # noqa: BLE001 - one bad body must not stop a load
            continue
        doc["content_ref"] = ref
        doc["size"] = size
        doc.pop("content", None)
    return docs


# ── staging: the store's established download/cache location ─────────────────

def _downloads_dir() -> Path:
    """`<EMBER_CACHE_DIR>/downloads` — the one place stage-0 raw drops land. Cache semantics
    (config.py): deleting it costs a re-fetch, never data."""
    from ember.config import load
    d = load().cache_dir / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
    """Streaming HTTP GET → `dest`. Read-only by construction: the verb is hard-locked to GET
    and the scheme/host/redirect guards are the fetch bundle's own (`op.fetch.get`), reused
    rather than re-implemented, so there is one SSRF/verb policy in the tree.

    Resumable: bytes land in `dest.part` and a re-run resumes with a Range header. A server that
    ignores Range answers 200 and the download restarts cleanly. `dest` appears on completion, by
    atomic rename, so a present `dest` is a whole file."""
    import urllib.request
    from urllib.parse import urlparse
    from ember.runtime.runner import fetch as F                # the single distribution path
    if dest.exists():
        return dest                               # staged already — a re-fetch is never implicit
    p = urlparse(url)
    if p.scheme not in F._ALLOWED_SCHEMES:
        raise ValueError("scheme %r not allowed (http/https only): %s" % (p.scheme, url))
    if F._blocked_host(p.hostname or ""):
        raise ValueError("refusing internal/loopback host %r" % (p.hostname,))
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": F._UA}
    if have:
        headers["Range"] = "bytes=%d-" % have
    req = urllib.request.Request(url, method="GET", headers=headers)   # GET, hard-locked
    # F._OPENER, not urlopen: the SSRF guards must re-run on every redirect hop (fetch.py).
    with F._OPENER.open(req, timeout=timeout) as r:
        status = getattr(r, "status", 200)
        mode = "ab" if (have and status == 206) else "wb"
        with open(part, mode) as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    part.replace(dest)
    return dest


def _open_maybe_gz(path: Path, mode: str = "rb"):
    opener = gzip.open if str(path).endswith(".gz") else open
    if "b" in mode:
        return opener(path, mode)
    # text mode: the sources are UTF-8 — never the locale codec (cp1252 on a Windows dev box
    # would silently mangle every non-ASCII term).
    return opener(path, mode, encoding="utf-8", errors="replace")


def _ensure_etype(store, eid: str, direction: str, meaning: str, author: str) -> bool:
    """Mint an edge-type artifact the seed registry does not carry — the open type system, where
    new types are minted as self-describing artifacts whenever observed structure demands them (see
    the genesis header). Same shape as bootstrap's seed etype rows. Idempotent."""
    aid = f"etype.{eid}"
    if store.artifacts.get_artifact(aid) is not None:
        return False
    g._mint(store, {
        "id": aid,
        "content_type": g.ETYPE_CONTENT_TYPE,
        "context": {"name": eid, "direction": direction, "meaning": meaning,
                    "kind": "edge-type", "seed": False, "provenance": g.P_HUMAN},
        "lemmas": [eid.lower()],
        "content": f"edge type {eid} ({direction}): {meaning}",
        "provenance": g.P_HUMAN,
        "created_by": author,
    })
    g._ensure_edge(store, aid, g.ONTOLOGY, "member_of")
    return True


CT_CONTENTTYPE = "application/vnd.agience.content-type+json"


def _ensure_ctype(store, ct: str, *, template: str, field_source: Dict[str, str],
                  describer: str, role: str, author: str) -> bool:
    """Mint the `type.<ct>` artifact that says how rows of this type are described. Idempotent, and
    it leaves an existing one as it found it.

    Presentation is bootstrapped with the type. A missing type artifact is a legitimate state —
    `_type_def` returns None and the caller falls back to bare titles — so nothing downstream can
    report its absence. A type that a source introduces is part of what that source introduces, and
    this is the only place that knows to create it.

    Leaving an existing row alone is the other half: the describer evolves (B1.13 — recognition is
    learned), so a re-run of the ingester keeps whatever describer the type has reached."""
    aid = "type." + ct
    if store.artifacts.get_artifact(aid) is not None:
        return False
    g._mint(store, {
        "id": aid,
        "content_type": CT_CONTENTTYPE,
        "declares": ct,
        "context": {"kind": "content-type", "declares": ct, "describer": describer, "role": role},
        "offer_template": template,
        "field_source": dict(field_source),
        "lemmas": [ct.lower()],
        "content": "content type %s: %s" % (ct, role),
        "provenance": g.P_HUMAN,
        "created_by": author,
    })
    g._ensure_edge(store, aid, g.ONTOLOGY, "member_of")
    return True


# The synset type's own describer. The rows carry `title` and `gloss` as fields, so
# `{title}: {gloss}` renders exactly what the ingester observed, with no labels or composed prose
# around the data.
_WN_TYPE = dict(
    template="{title}: {gloss}",
    field_source={"title": "title", "gloss": "gloss"},
    describer="wordnet.v1",
    role="a WordNet synset: ONE sense of one or more words, with its definition",
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. op.source.oewn — Open English WordNet 2024 (TRAINING-QUEUE 0.2)
# ═════════════════════════════════════════════════════════════════════════════

OEWN_URL = "https://en-word.net/static/english-wordnet-2024.xml.gz"
OEWN_MARKER = "english-wordnet-2024"
# The source's own name is the label. Whatever OEWN calls a relation is what the edge is called —
# no mapping table, no allowlist, no translation — because a hand-authored map from the source's
# vocabulary to ours is itself a model of which relations matter, and edge names are models.
#
# The OEWN-2024 file carries 28 distinct `relType` values. Naming them from the source keeps all of
# them, including the two largest that a three-entry map would have dropped: `derivation` (74,646),
# which is the derivational morphology the mesh phase reads as surface-form evidence, and
# `domain_topic` (6,946), which is an associative relation the IS-A tree cannot express.
#
# Two aliases are kept because the existing corpus and the tree reader already speak them.
_LMF_ALIAS = {"instance_hypernym": "instance_of", "holo_part": "part_of"}

#: Which SENSE-level relations become synset edges.
#:
#: Synset-level relations are taken whole — the source names them and `_ensure_etype` mints a type
#: row for whatever arrives. Sense-level ones cannot be, because several are bookkeeping between two
#: word forms rather than a relation between two meanings, and `other` (16,887 of them) is a label
#: that names nothing at all. So this side is a stated set, and each member is here for a reason:
#:
#:   antonym     7,996   opposition. The only one this parser has ever taken.
#:   derivation 74,646   the word a word derives from, ACROSS parts of speech — `beautiful` to
#:                       `beauty`, `quickly` to `quickness`. This is what gives an adjective or an
#:                       adverb a noun to be measured at: neither has a hypernym parent, so neither
#:                       has a position of its own, and 43.2% of adjectives reach a noun this way.
#:   pertainym   8,072   an adverb to the adjective it pertains to — `quickly` to `quick` — which is
#:                       then placed by the line above. 69.0% of adverbs have one.
#:
#: Deliberately excluded: `other` (names nothing), `also` and `exemplifies`/`is_exemplified_by`
#: (already present as SYNSET relations — taking the sense-level copies would double the same edge),
#: `participle` (73, a morphological note), and `similar` (2 at sense level against 23,188 at synset
#: level, so the sense-level pair is noise).
_SENSE_RELATIONS_KEPT = frozenset({"antonym", "derivation", "pertainym"})


def parse_oewn_lmf(fileobj) -> Dict[str, Any]:
    """Stream-parse a WN-LMF 1.x file, against the published OEWN-2024 layout →
    {"lexicon": meta, "synsets": [...], "relations": [(from, to, label)]}.

    LMF orders LexicalEntry elements before Synset elements, so one forward pass suffices: entries
    populate the synset→words and sense→synset maps that the Synset pass then reads. `iterparse`
    with a per-element clear keeps memory flat across the ~120k synsets and ~210k entries.
    """
    import xml.etree.ElementTree as ET
    it = ET.iterparse(fileobj, events=("start", "end"))
    lexicon_el = None
    lexicon_meta: Dict[str, str] = {}
    sense_synset: Dict[str, str] = {}      # sense id -> synset id
    syn_words: Dict[str, List[str]] = {}   # synset id -> writtenForms (entry order)
    sense_rank: Dict[str, Dict[str, int]] = {}   # synset id -> {lemma: its sense number for THAT word}
    # Irregular surface forms are source data. LMF lists a lemma's non-regular inflections as
    # `<Form>` (4,474 of them in OEWN-2024: aardwolves, abaci, abetted). A query word that is not
    # itself a lemma resolves to zero senses — `dogs`, `limits`, `derivatives` all do — so an
    # inflected need contributes no position. Regular plurals are recoverable by the standard
    # detachment rules; the irregulars are recoverable only from here, and only at ingest.
    forms: Dict[str, Dict[str, List[str]]] = {}  # synset id -> {lemma: [surface form,...]}
    sense_relations: List[Tuple[str, str, str]] = []
    synsets: List[Dict[str, Any]] = []
    relations: List[Tuple[str, str, str]] = []
    for ev, el in it:
        tag = el.tag.rsplit("}", 1)[-1]
        if ev == "start":
            if tag == "Lexicon" and lexicon_el is None:
                lexicon_el = el
                lexicon_meta = {k: (el.get(k) or "") for k in ("id", "label", "version", "license")}
            continue
        if tag == "LexicalEntry":
            lemma = el.find("Lemma")
            wf = (lemma.get("writtenForm") or "").strip() if lemma is not None else ""
            # Sense order is source data. LMF lists a LexicalEntry's `Sense` children in WordNet
            # sense order — sense 1 first, the most frequent use of that word — and that is the
            # only statement the corpus makes about which meaning a bare word most likely carries.
            # It is kept because an OEWN synset id is an arbitrary offset with no relation to sense
            # order, so sorting on ids instead would surface a word's least common sense first.
            _forms = [(f.get("writtenForm") or "").strip() for f in el.iter("Form")]
            _forms = [x for x in _forms if x]
            for _n_sense, sense in enumerate(el.iter("Sense")):
                sid, syn = sense.get("id") or "", sense.get("synset") or ""
                if syn:
                    if sid:
                        sense_synset[sid] = syn
                    if wf:
                        syn_words.setdefault(syn, []).append(wf)
                        sense_rank.setdefault(syn, {}).setdefault(wf, _n_sense)
                        if _forms:
                            forms.setdefault(syn, {}).setdefault(wf, list(_forms))
                for rel in sense.iter("SenseRelation"):
                    tgt = rel.get("target") or ""
                    _srt = (rel.get("relType") or "").strip()
                    if _srt in _SENSE_RELATIONS_KEPT and sid and tgt:
                        sense_relations.append((sid, tgt, _srt))
            el.clear()
        elif tag == "Synset":
            sid = el.get("id") or ""
            defs = [(d.text or "").strip() for d in el.iter("Definition")]
            exs = [(x.text or "").strip() for x in el.iter("Example")]
            synsets.append({
                # `ili="in"` is WN-LMF's SENTINEL for a synset proposed for the interlingual
                # index but not yet assigned one — the source's own value, not a parse fault.
                # Measured 2026-08-25 on 71/home: 3,216 OEWN rows carry it, and they are 3,085
                # DISTINCT titles (`barely`, `accustomed to`, `unused to`, `exercise`), so it is
                # emphatically not an identity. Carried through as written, it reads downstream
                # exactly like an ILI id and folds 3,216 unrelated concepts into one — which is the
                # single way an ILI-keyed fold can be worse than no fold at all
                # (`search/ranking._fold_key` shape-checks for that reason).
                #
                # Normalised to "" here, where the sentinel is understood, rather than left for
                # every reader to know the convention.
                "id": sid, "ili": _ili_or_blank(el.get("ili")),
                "pos": el.get("partOfSpeech") or "",
                "words": syn_words.get(sid, []),
                "sense_ranks": sense_rank.get(sid, {}),   # lemma -> its sense number for that word
                "forms": forms.get(sid, {}),             # lemma -> its irregular surface forms
                "definition": next((d for d in defs if d), ""),
                "examples": [x for x in exs if x],
            })
            for rel in el.iter("SynsetRelation"):
                _rt = (rel.get("relType") or "").strip()
                lbl = _LMF_ALIAS.get(_rt, _rt)          # the source names the relation
                tgt = rel.get("target") or ""
                if lbl and sid and tgt:
                    relations.append((sid, tgt, lbl))
            el.clear()
        else:
            continue
        if lexicon_el is not None:
            del lexicon_el[:]   # drop the cleared child — keeps the tree O(1) over the file
    # Sense-level relations relate two WORDS; the graph relates synsets, so each end is mapped
    # through `sense_synset` — the same lift `antonym` has always used. A target whose sense was
    # never seen is skipped rather than guessed.
    for a_sid, b_sid, lbl in sense_relations:
        a, b = sense_synset.get(a_sid), sense_synset.get(b_sid)
        if a and b:
            relations.append((a, b, lbl))
    return {"lexicon": lexicon_meta, "synsets": synsets, "relations": relations}


def oewn_done(store) -> bool:
    return OEWN_MARKER in g._shards_done(store, "oewn")


def ingest_stage0_oewn(store, *, author: str = DEFAULT_AUTHOR, force: bool = False,
                       edge_batch: int = 1000, limit: Optional[int] = None,
                       path: Optional[str] = None) -> Dict[str, Any]:
    """Ingest Open English WordNet 2024 into the Stage-0 lexicon. Same contract and row shape
    as `ingest_stage0_wordnet`: each synset → a text/x-wordnet artifact (id `wn-<synset id>`,
    e.g. wn-oewn-08242255-n) homed in stage.0.lexicon + source.oewn, cited_from cite.oewn,
    via op.source.oewn, provenance OBSERVED — plus an `ili` field (OEWN carries CILI ids
    natively) that op.source.cili / op.source.omw pivot on. Relations land under the names the
    source gives them, aliased through `_LMF_ALIAS` where the corpus already speaks another.

    Guarded by its completion marker: a completed ingest is skipped, so the ordinary path does no
    work twice. A run that stopped part-way re-runs whole — artifact upserts are idempotent, and
    the marker is written after the edge pass lands.

    Re-emitting does not duplicate, and this docstring used to claim otherwise: it said
    "re-emitting ~200k edges would duplicate them", which contradicts `add_edges`'s own contract —
    idempotent on `edge_key`, "exactly one row and one edge counter increment". Measured: three
    identical emissions of the same edges leave one row each and leave every edge counter
    unmoved; `test_edge_label_extent_counter.py` now pins both so the two docstrings cannot drift
    apart again.

    That matters because the data here is older than this function. `_SENSE_RELATIONS_KEPT` was
    widened to keep `derivation` and `pertainym`, and the store still holds 0 of each for OEWN
    while the cached source carries 74,646 and 8,072 — the exact figures in the comment above the
    map. `antonym` is present because it was in the narrower map all along. A modifier is placed
    through precisely those two relations, so the served lexicon can currently place 19% of its own
    adjectives where Princeton places 73%.

    `force=True` re-parses the cached file and closes that gap at the source. It is safe from the
    duplication angle; what it costs is the parse and the write."""
    author = g._author_ref(store, author)
    if not g.is_bootstrapped(store):
        g.bootstrap(store, author=author)
    # The type comes in WITH the rows that need it — see `_ensure_ctype`. Before the completion
    # guard, so a store whose rows landed under an older build still gains the describer.
    _ensure_ctype(store, g.WORDNET_CONTENT_TYPE, author=author, **_WN_TYPE)
    if oewn_done(store) and not force:
        existing = store.artifacts.count(state="committed")
        return {"skipped": 1, "reason_already_ingested": 1, "committed": existing, "ingested": 0}

    g.mint_source_triple(store, "oewn", cite_meta={
        "dataset": "Open English WordNet 2024", "provider": "Global WordNet Association",
        "license": "CC BY 4.0 (per the published LMF Lexicon header)", "version": "2024",
        "url": OEWN_URL,
    }, offer="streams Open English WordNet 2024 synsets as keyed lexical artifacts "
             "(Synset/Lexeme, ILI-carrying) with hypernym/instance_of/part_of/antonym "
             "structure — the maintained successor rows beside the WordNet 3.0 spine",
        author=author)

    local = Path(path) if path else _download(OEWN_URL, _downloads_dir() / "english-wordnet-2024.xml.gz")
    with _open_maybe_gz(local, "rb") as f:
        parsed = parse_oewn_lmf(f)

    summary: Dict[str, int] = {"synsets": 0, "edges": 0}
    stored: set = set()
    docs: List[Dict[str, Any]] = []
    for ss in parsed["synsets"]:
        if limit is not None and len(stored) >= limit:
            break
        # The source lists written forms in entry order, and that order carries primacy: the first
        # form is the primary lemma, which becomes the row's `title`. Dedupe preserves it, so the
        # ingester imposes no alphabet of its own.
        words, _seen = [], set()
        for _w in ss["words"]:
            _w = (_w or "").replace("_", " ")
            if _w and _w not in _seen:
                _seen.add(_w); words.append(_w)
        if not words:
            continue
        # Case is source data, so `sense_ranks` keys keep it. LMF gives `mass` (the physical
        # quantity) and `Mass` (the Eucharist) separate LexicalEntries, and a sense number is
        # per-entry, so both are sense 0; a lowercased key would collapse them to one `mass -> 0`
        # and leave the tie to synset-offset order.
        #
        # Capitalization is the source's own proper-noun marker, and `crystal.ontology.driver`
        # reads it to order a lowercase query's senses: common-noun entries first, proper nouns
        # after. Query tokens are lowercased upstream, so a query typed `Mass` reaches this ordering
        # as `mass`.
        _ranks = {str(k or "").replace("_", " "): int(v)
                  for k, v in (ss.get("sense_ranks") or {}).items()}
        _forms = {str(k or "").replace("_", " "):
                  [str(x).replace("_", " ") for x in (v or [])]
                  for k, v in (ss.get("forms") or {}).items()}
        aid = "wn-" + ss["id"]
        defn = ss["definition"]
        # Observation only: the ingester stores the fields it saw — `title`, `gloss`, `content`,
        # `examples`, `pos` — and writes no `context`. Context comes via description, and describers
        # evolve: the offer is an interpretation, so it is the describer's output. The artifact
        # lands dark and the illuminate step (`describe_dark`) keys it. When the describer improves
        # (B1.13: recognition is learned), every artifact is re-described from these same fields,
        # with no re-fetch and no re-ingest, and the type's `offer_template` governs the layout.
        doc = {
            "id": aid,
            "content_type": g.WORDNET_CONTENT_TYPE,
            "state": "committed",
            "title": words[0],                   # the primary lemma, as a field
            "gloss": defn,                       # the definition, clean and unstyled
            "content": defn,                     # the information itself
            "examples": list(ss["examples"][:3]),  # a list, kept structured
            "lemmas": [w.lower() for w in words],
            # Which sense of each word this synset is (0 = that word's first, most common sense).
            # The source's own statement about likely meaning; the alternative is synset-offset
            # order, which carries none (see parse_oewn_lmf).
            "sense_ranks": _ranks,
            # The source's own irregular inflections, kept so an inflected need can resolve from the
            # corpus. Absent for the ~96% of lemmas that inflect regularly; those are reached by the
            # standard detachment rules.
            **({"forms": _forms} if _forms else {}),
            "word": words[0].lower(),
            "pos": ss["pos"],
            "collection_id": "stage.0.lexicon",
            "collections": ["stage.0.lexicon", "source.oewn"],
            "cited_from": "cite.oewn",
            "via": "op.source.oewn",
            "operator": "op.source.oewn",
            "provenance": g.P_OBSERVED,
            "created_by": author,
            "created_time": g._now(),
        }
        if ss["ili"]:
            doc["ili"] = ss["ili"]               # the interlingual pivot key (CILI id)
        docs.append(doc)
        stored.add(aid)
        if len(docs) >= 2000:
            summary["synsets"] += store.artifacts.put_many(_sealed(store, docs))
            docs = []
    if docs:
        summary["synsets"] += store.artifacts.put_many(_sealed(store, docs))

    # Every relation the source names declares itself. The open type system mints new types as
    # self-describing artifacts whenever observed structure demands them, so a relation nobody
    # anticipated arrives with an `etype.<name>` row of its own. That is what lets the source name
    # the relation: the vocabulary grows from the data rather than being declared ahead of it.
    _seen_labels = {lbl for _s, _d, lbl in parsed["relations"] if lbl}
    for _lbl in sorted(_seen_labels):
        _ensure_etype(store, _lbl, "synset -> synset",
                      "relation named by the source (OEWN LMF relType %r)" % _lbl, author)
    summary["edge_types"] = len(_seen_labels)

    def _edges():
        for src, dst, lbl in parsed["relations"]:
            a, b = "wn-" + src, "wn-" + dst
            if a in stored and b in stored:      # only where BOTH endpoints were stored
                yield (a, b, lbl, {"via": "op.source.oewn", "rung": g.P_OBSERVED})

    summary["edges"] = store.graph.add_edges(_edges(), batch=edge_batch)
    if limit is None:                             # a bounded smoke run is not completion
        g._mark_shard_done(store, "oewn", OEWN_MARKER)
    from ember.runtime.runner import evolution
    if summary["synsets"]:
        evolution.record_invocation(store.artifacts, "op.source.oewn", verified=True)
    return {**summary, "ingested": summary["synsets"]}


# ═════════════════════════════════════════════════════════════════════════════
# 2. op.source.cili — the interlingual index (TRAINING-QUEUE 0.3)
# ═════════════════════════════════════════════════════════════════════════════

CILI_URL = "https://github.com/globalwordnet/cili/raw/master/ili.ttl"
CILI_MARKER = "ili-ttl"

_ILI_SUBJ = re.compile(r"^<(i\d+)>")
_ILI_DEF = re.compile(r'skos:definition\s+"((?:[^"\\]|\\.)*)"')
_ILI_SRC = re.compile(r"dc:source\s+pwn30:(\d{8})-([a-z])")


def parse_ili_ttl(lines: Iterable) -> Iterable[Tuple[str, str, Optional[Tuple[str, str]]]]:
    """Yield (ili_id, definition, (wn30_offset, pos) | None) from CILI's ili.ttl.

    A deterministic line-based state machine over the pinned upstream layout:

        <i1>	a	<Concept>;
        	skos:definition	"..."@en;
        	dc:source	pwn30:00001740-a.

    Deliberately not a general Turtle parser. A format drift yields zero records, which the ingester
    reports as `parsed=0` and no completion marker, so the drift is visible in the result."""
    cur: Optional[str] = None
    defn = ""
    src: Optional[Tuple[str, str]] = None
    for raw in lines:
        line = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else raw
        s = line.strip()
        m = _ILI_SUBJ.match(s)
        if m:
            if cur is not None:                  # missing terminator on the previous statement
                yield (cur, defn, src)
            cur, defn, src = m.group(1), "", None
        if cur is None:
            continue
        dm = _ILI_DEF.search(s)
        if dm:
            defn = dm.group(1)
        sm = _ILI_SRC.search(s)
        if sm:
            src = (sm.group(1), sm.group(2))
        if s.endswith("."):                      # end of statement
            yield (cur, defn, src)
            cur, defn, src = None, "", None
    if cur is not None:
        yield (cur, defn, src)


def _pwn30_resolver(store) -> Callable[[Tuple[str, str]], Optional[str]]:
    """(offset, pos) from a `pwn30:` source ref → the id of the WordNet-3.0 row we hold
    (`wn-<nltk name>`), or None when it does not resolve or is not in the store. `nltk` is the
    `bootstrap` extra, used here for the same one-time acquisition as `ingest_stage0_wordnet`."""
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets                                # trigger corpus load / LookupError early
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "op.source.cili needs nltk + the wordnet corpus to resolve pwn30 offsets: "
            "python -c \"import nltk; nltk.download('wordnet')\""
        ) from e

    def resolve(src: Tuple[str, str]) -> Optional[str]:
        offset, pos = src
        try:
            name = wn.synset_from_pos_and_offset(pos, int(offset)).name()
        except Exception:
            return None
        aid = "wn-" + name
        return aid if store.artifacts.get_artifact(aid) is not None else None

    return resolve


def _ili_pivot_map(store) -> Dict[str, List[str]]:
    """ili id → the synset artifact ids that carry it. One full pass over the text/x-wordnet
    rows (a one-time cost per ingester run, on the ~240k-row fresh stage-0 lattice — not on
    any hot path). OEWN rows carry `ili` natively; OMW rows add theirs as they land."""
    out: Dict[str, List[str]] = {}
    for a in store.artifacts.list_artifacts(content_type=g.WORDNET_CONTENT_TYPE):
        v = a.get("ili")
        if v:
            out.setdefault(v, []).append(a["id"])
    return out


def cili_done(store) -> bool:
    return CILI_MARKER in g._shards_done(store, "cili")


def ingest_stage0_cili(store, *, author: str = DEFAULT_AUTHOR, force: bool = False,
                       edge_batch: int = 1000, path: Optional[str] = None,
                       resolve: Optional[Callable] = None,
                       limit: Optional[int] = None) -> Dict[str, Any]:
    """Ingest the CILI interlingual index as PIVOT EDGES (runbook H1.3: "CILI ids become the
    interlingual pivot edges"). For every ILI concept, the synsets that express it — the
    OEWN/OMW rows carrying that `ili` field, and the WordNet-3.0 row its `pwn30:` source
    resolves to — are linked with a typed `ili` edge onto one canonical endpoint (the PWN-3.0
    row when present: the hand-built spine anchors the pivot). Edges only: where a synset
    already exists no new vertex is minted, and an ILI id with fewer than two local synsets
    contributes nothing and is counted as `single_endpoint` or `no_local_synset`.

    Ordering: run after op.source.oewn. Without ILI-carrying rows in the store every concept has at
    most one endpoint and the result is zero edges, reported as such."""
    author = g._author_ref(store, author)
    if not g.is_bootstrapped(store):
        g.bootstrap(store, author=author)
    if cili_done(store) and not force:
        return {"skipped": 1, "reason_already_ingested": 1, "ingested": 0}

    g.mint_source_triple(store, "cili", cite_meta={
        "dataset": "CILI — the Collaborative Interlingual Index", "provider": "Global WordNet Association",
        "license": "CC BY 4.0", "url": CILI_URL,
    }, offer="links synsets that express the same interlingual concept: CILI ids become typed "
             "`ili` pivot edges between existing synset rows — edges only, never new vertices",
        author=author)
    _ensure_etype(store, "ili", "synset -> synset",
                  "interlingual pivot: both endpoints express the same CILI concept "
                  "(props carry the ili id)", author)

    local = Path(path) if path else _download(CILI_URL, _downloads_dir() / "ili.ttl")
    pivots = _ili_pivot_map(store)
    resolve = resolve or _pwn30_resolver(store)

    counts = {"parsed": 0, "edges": 0, "pwn_unresolved": 0, "no_local_synset": 0, "single_endpoint": 0}

    def _edges():
        with _open_maybe_gz(local, "rb") as f:
            for ili, _defn, src in parse_ili_ttl(f):
                counts["parsed"] += 1
                if limit is not None and counts["parsed"] > limit:
                    break
                members = sorted(pivots.get(ili, []))
                pwn = resolve(src) if src else None
                if src and pwn is None:
                    counts["pwn_unresolved"] += 1
                if pwn and pwn not in members:
                    members.append(pwn)
                if not members:
                    counts["no_local_synset"] += 1
                    continue
                if len(members) < 2:
                    counts["single_endpoint"] += 1   # a pivot needs two ends — nothing to draw
                    continue
                # canonical end = the PWN-3.0 spine row when the index names one, else the
                # first member (deterministic). Everything else points AT it.
                canon = pwn or members[0]
                props = {"ili": ili, "via": "op.source.cili", "rung": g.P_OBSERVED,
                         "cited_from": "cite.cili"}
                for m in members:
                    if m != canon:
                        yield (m, canon, "ili", dict(props))

    counts["edges"] = store.graph.add_edges(_edges(), batch=edge_batch)
    if limit is None and counts["parsed"]:        # zero parsed is a format drift: no marker
        g._mark_shard_done(store, "cili", CILI_MARKER)
    from ember.runtime.runner import evolution
    if counts["edges"]:
        evolution.record_invocation(store.artifacts, "op.source.cili", verified=True)
    return {**counts, "ingested": counts["edges"]}


# ═════════════════════════════════════════════════════════════════════════════
# 3. op.source.conceptnet — ConceptNet 5.7 assertions (TRAINING-QUEUE 0.4)
# ═════════════════════════════════════════════════════════════════════════════

CONCEPTNET_URL = ("https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
                  "conceptnet-assertions-5.7.0.csv.gz")
CONCEPTNET_MARKER = "assertions-5.7.0"
CONCEPT_CONTENT_TYPE = "application/x-concept"   # the seed Concept vtype's content type
CN_CURSOR_ID = "source.conceptnet.cursor"
CN_CURSOR_CT = "application/vnd.agience.source-cursor+json"

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _cn_label(rel: str) -> str:
    """`/r/IsA` → `is_a`; `/r/dbpedia/genre` → `dbpedia_genre`. ConceptNet's own relation
    vocabulary, snake-cased and left there: mapping IsA onto instance_of or subclass_of would
    assert a distinction the source does not make. Where the names coincide with a seed etype
    (antonym, synonym, part_of) they coincide."""
    name = rel[3:] if rel.startswith("/r/") else rel
    return _CAMEL.sub("_", name.replace("/", "_")).lower()


def _cn_term(uri: str) -> str:
    """`/c/en/take_off/v` → `take_off` (the term; the optional pos/sense tail is dropped so
    one English term is one Concept vertex — ConceptNet's own term-level normalization)."""
    seg = uri.split("/")
    return seg[3] if len(seg) > 3 else ""


def parse_conceptnet_line(line: str) -> Optional[Tuple[str, str, str, float]]:
    """One assertions-dump TSV line → (edge_label, start_term, end_term, weight) for
    English↔English rows; None otherwise. Strict on shape: a line with the wrong field count or
    unparsable metadata JSON — which is what a truncated line looks like — parses to None rather
    than to a partial record. The ingester reads a None as a filtered row when the line was
    terminated, and as a truncated tail otherwise, holding the cursor there."""
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) != 5:
        return None
    _uri, rel, start, end, meta = parts
    if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
        return None
    t1, t2 = _cn_term(start), _cn_term(end)
    if not t1 or not t2 or t1 == t2:
        return None
    try:
        weight = float(json.loads(meta).get("weight", 1.0))
    except Exception:
        return None                               # unreadable metadata ⇒ the row is not whole
    return (_cn_label(rel), t1, t2, weight)


def _concept_doc(term: str, author: str) -> Dict[str, Any]:
    word = term.replace("_", " ")
    return {
        "id": "cn-" + term,
        "content_type": CONCEPT_CONTENT_TYPE,
        "state": "committed",
        # ── The offer, and the name ──────────────────────────────────────────────────────
        # A term node's offer is the term. It had none: `context` carried "the concept {word}:
        # a ConceptNet 5.7 English term node", which names the source, and these rows carried
        # no title at all — measured, 0% of 3,000 sampled. Provenance is already recorded by
        # `cited_from` / `via` / `operator` / `provenance` below, and by `content`.
        #
        # `offer` is canonical because it is the indexed column the lexical arm reads — context,
        # offer, description and title are the same field; tags or other properties are
        # collection edges instead. A name in any other field is a name that cannot be found,
        # which is exactly why the 5,484 colimits were invisible until theirs was backfilled.
        # Measured before this change: 0 of 300,000 artifacts sampled carried an offer, while
        # 100% carried a title.
        "offer": word,
        # `title` is retained only until the migration lands — it is the same field as `offer`
        # and 100% of the store still reads it. Removing it here would break readers before the
        # store has caught up; the collapse is scoped, not started.
        "title": word,
        # `description` is gone, not moved. It was `"the concept " + title` — measured across
        # the whole population, 282,222 of 282,222 were that template and none carried
        # information beyond the title. A generated restatement is duplication, not a field.
        "content": f"{word} — ConceptNet 5.7 concept /c/en/{term}",
        "lemmas": [w.lower() for w in word.split() if w],
        "word": word.lower(),
        "collection_id": "stage.0.lexicon",
        "collections": ["stage.0.lexicon", "source.conceptnet"],
        "cited_from": "cite.conceptnet",
        "via": "op.source.conceptnet",
        "operator": "op.source.conceptnet",
        "provenance": g.P_OBSERVED,
        "created_by": author,
        "created_time": g._now(),
    }


def conceptnet_done(store) -> bool:
    return CONCEPTNET_MARKER in g._shards_done(store, "conceptnet")


def ingest_stage0_conceptnet(store, *, author: str = DEFAULT_AUTHOR, force: bool = False,
                             path: Optional[str] = None, max_lines: Optional[int] = None,
                             batch_lines: int = 20000, edge_batch: int = 1000) -> Dict[str, Any]:
    """Stream the ConceptNet 5.7 assertions dump into the Stage-0 lexicon: /c/en/↔/c/en/
    assertions land as Concept vertices plus typed relation edges (runbook H1.3: "ConceptNet
    edges land as typed relations, cited_from → cite.conceptnet"). The ~1 GB gz is streamed
    line by line, so it is neither decompressed whole nor materialized.

    Resumable via a line-count cursor artifact (`source.conceptnet.cursor`), under the
    content-cursor discipline (content_tier.py): the cursor advances only past lines whose
    vertices and edges have been applied, so a resume re-reads nothing it holds and skips nothing
    it lost. `max_lines` bounds one call — a tick size, the same bounded-increment shape as
    stage-2's one-shard tick, with the cursor carrying the rest to the next tick. A truncated tail,
    an unterminated line that does not parse, holds the cursor at that line and is reported, so
    re-staging the file recovers it."""
    author = g._author_ref(store, author)
    if not g.is_bootstrapped(store):
        g.bootstrap(store, author=author)
    if conceptnet_done(store) and not force:
        return {"skipped": 1, "reason_already_ingested": 1, "ingested": 0}

    g.mint_source_triple(store, "conceptnet", cite_meta={
        "dataset": "ConceptNet 5.7 assertions", "provider": "ConceptNet (Speer et al.)",
        "license": "CC BY-SA 4.0 (aggregate; per-edge licenses ride the source metadata)",
        "version": "5.7.0", "url": CONCEPTNET_URL,
    }, offer="streams the ConceptNet 5.7 assertions dump, landing English concepts as keyed "
             "Concept artifacts and assertions as typed relation edges (commonsense "
             "structure over the lexicon spine)",
        author=author)

    local = Path(path) if path else _download(CONCEPTNET_URL,
                                              _downloads_dir() / "conceptnet-assertions-5.7.0.csv.gz")
    cur = store.artifacts.get_artifact(CN_CURSOR_ID) or {}
    start = int(cur.get("lines_done") or 0)

    seen_terms: set = set()
    seen_labels: set = set()
    # buffers carry their line index so a held line's contributions can be dropped exactly.
    docs: List[Tuple[int, Dict[str, Any]]] = []
    edges: List[Tuple[int, tuple]] = []
    counts = {"concepts": 0, "edges": 0, "skipped": 0}
    applied = start

    def _flush(upto: int) -> None:
        """Apply every buffered line < `upto`, then move the cursor to `upto`. Entries at or past
        `upto` are dropped un-applied and re-enter when their line is re-read, so the cursor
        advances only over lines that applied."""
        nonlocal applied
        keep_d = [d for i, d in docs if i < upto]
        keep_e = [e for i, e in edges if i < upto]
        docs.clear(); edges.clear()
        if keep_d:
            counts["concepts"] += store.artifacts.put_many(_sealed(store, keep_d), batch=500)
        if keep_e:
            counts["edges"] += store.graph.add_edges(keep_e, batch=edge_batch)
        if upto > applied:
            applied = upto
            store.artifacts.put_artifact({
                "id": CN_CURSOR_ID, "content_type": CN_CURSOR_CT, "state": "committed",
                "lines_done": applied,
                # `offer`, not `content`: this is a title that restates `lines_done` on the same
                # doc, and the naming field is `offer` — context, offer, description and title
                # are the same field. `content` is also the field `artifacts_holding_inline_plaintext` counts, pinned
                # at 0 — a cursor label sitting there is a row of that population, and it is not a
                # body anyone needs to read back.
                "offer": "conceptnet ingest cursor: %d lines applied" % applied,
                "provenance": g.P_HUMAN, "cited_from": g.CITE_GENESIS,
                "created_time": g._now()})

    processed = 0
    i = start - 1                                  # last line index seen
    truncated_tail = False
    gz_truncated = False
    budget_hit = False
    try:
        with _open_maybe_gz(local, "rt") as f:     # text mode: utf-8 line iterator over the gz
            for i, raw in enumerate(f):
                if i < start:
                    continue                       # decompress-and-skip the resumed prefix
                if max_lines is not None and processed >= max_lines:
                    budget_hit = True              # line i is unprocessed; the cursor stops at i
                    break
                processed += 1
                rec = parse_conceptnet_line(raw)
                if rec is None:
                    if not raw.endswith("\n"):
                        # an UNTERMINATED line that does not parse is a truncated download,
                        # not a filtered row: hold the cursor here and say so.
                        truncated_tail = True
                        break
                    counts["skipped"] += 1         # non-English / malformed-but-terminated
                    continue
                label, t1, t2, weight = rec
                for t in (t1, t2):
                    if t not in seen_terms:
                        seen_terms.add(t)
                        docs.append((i, _concept_doc(t, author)))
                if label not in seen_labels:
                    seen_labels.add(label)
                    _ensure_etype(store, label, "concept -> concept",
                                  f"ConceptNet 5.7 relation ({label}) between English concepts",
                                  author)
                edges.append((i, ("cn-" + t1, "cn-" + t2, label,
                                  {"via": "op.source.conceptnet", "rung": g.P_OBSERVED,
                                   "cited_from": "cite.conceptnet", "weight": weight})))
                if (i + 1) - applied >= batch_lines:
                    _flush(i + 1)
    except (EOFError, OSError):
        # The gz stream itself is truncated (BadGzipFile/EOFError mid-iteration). Lines already
        # yielded are intact; the last one is held anyway so that re-staging the file can re-read
        # it. What applied is kept, and what did not apply is re-read.
        gz_truncated = True

    if truncated_tail or gz_truncated:
        _flush(max(i, applied))                    # everything BEFORE the suspect line
    elif budget_hit:
        _flush(i)                                  # i = the first unprocessed line
    elif i >= start:
        _flush(i + 1)                              # clean EOF: the whole file applied

    drained = not (truncated_tail or gz_truncated or budget_hit) and i >= 0
    if drained:
        g._mark_shard_done(store, "conceptnet", CONCEPTNET_MARKER)
    from ember.runtime.runner import evolution
    if counts["edges"]:
        evolution.record_invocation(store.artifacts, "op.source.conceptnet", verified=True)
    return {**counts, "lines_done": applied, "drained": drained,
            "truncated_tail": truncated_tail, "gz_truncated": gz_truncated,
            "ingested": counts["edges"]}


# ═════════════════════════════════════════════════════════════════════════════
# 4. op.source.omw — Open Multilingual Wordnet, license-vetted (QUEUE 0.5)
# ═════════════════════════════════════════════════════════════════════════════

OMW_VERSION = "1.4"

# ── the license allowlist (runbook H1.1: "license-vetted per language") ──────
# Vetting is an explicit allowlist rather than a heuristic: a language ingests when it is named here
# and the `wn` index metadata still carries the license this entry was vetted against, so a drift in
# that metadata is a skip. The entries below are the OMW 1.4 lexicons whose licenses are clearly
# permissive per the wn package's index metadata:
#   CC BY 3.0    attribution-only          MIT / Apache-2.0   permissive software licenses
#   ODC-BY       attribution-only (data)
# Not listed, for clearing via the TRAINING-QUEUE prep column:
#   • license metadata is the bare string "wordnet" — a pointer rather than a verified license:
#     omw-cmn, omw-da, omw-he, omw-ja, omw-nb, omw-nn, omw-th, and omw-pl (plWordNet, which the
#     runbook excludes unless licensed — its real license is custom);
#   • share-alike (CC BY-SA *): omw-arb, omw-lt, omw-nl, omw-pt, omw-ro, omw-sk, omw-sl —
#     copyleft-adjacent, a policy call rather than "clearly permissive";
#   • CeCILL-C (LGPL-style reciprocal): omw-fr;
#   • omw-en / omw-en31: permissively licensed but redundant — the same PWN 3.0/3.1 content the
#     stage-0 spine carries via op.source.wordnet.
OMW_LICENSE_ALLOWLIST: Dict[str, Tuple[str, str]] = {
    # id: (license substring the vetting verifies against wn metadata, rationale)
    "omw-bg": ("creativecommons.org/licenses/by/3.0", "BulTreeBank Wordnet — CC BY 3.0, attribution-only"),
    "omw-ca": ("creativecommons.org/licenses/by/3.0", "MCR Catalan — CC BY 3.0, attribution-only"),
    "omw-el": ("licenses/Apache-2.0", "Greek Wordnet — Apache-2.0, permissive"),
    "omw-es": ("creativecommons.org/licenses/by/3.0", "MCR Spanish — CC BY 3.0, attribution-only"),
    "omw-eu": ("creativecommons.org/licenses/by/3.0", "MCR Basque — CC BY 3.0, attribution-only"),
    "omw-fi": ("creativecommons.org/licenses/by/3.0", "FinnWordNet — CC BY 3.0, attribution-only"),
    "omw-gl": ("creativecommons.org/licenses/by/3.0", "MCR Galician — CC BY 3.0, attribution-only"),
    "omw-hr": ("creativecommons.org/licenses/by/3.0", "Croatian Wordnet — CC BY 3.0, attribution-only"),
    "omw-id": ("licenses/MIT", "Wordnet Bahasa (Indonesian) — MIT, permissive"),
    "omw-is": ("creativecommons.org/licenses/by/3.0", "IceWordNet — CC BY 3.0, attribution-only"),
    "omw-it": ("creativecommons.org/licenses/by/3.0", "MultiWordNet Italian — CC BY 3.0, attribution-only"),
    "omw-iwn": ("opendefinition.org/licenses/odc-by", "ItalWordNet — ODC-BY, attribution-only"),
    "omw-sq": ("creativecommons.org/licenses/by/3.0", "Albanet — CC BY 3.0, attribution-only"),
    "omw-sv": ("creativecommons.org/licenses/by/3.0", "WordNet-SALDO — CC BY 3.0, attribution-only"),
    "omw-zsm": ("licenses/MIT", "Wordnet Bahasa (Malaysian) — MIT, permissive"),
}


def omw_done(store) -> bool:
    """Done = every allowlisted language has its completion marker. Growing the allowlist, by
    clearing a language via the QUEUE, re-opens the stage-0 slot."""
    return set(OMW_LICENSE_ALLOWLIST) <= set(g._shards_done(store, "omw"))


def _omw_index(wn) -> Dict[str, Dict[str, str]]:
    """{project id: {license, label, language}} for OMW-1.4 lexicons, from the wn package's
    own index metadata (`wn.config.index` — where the per-VERSION license actually lives;
    `wn.projects` reports None for version-scoped licenses)."""
    out: Dict[str, Dict[str, str]] = {}
    idx = getattr(getattr(wn, "config", None), "index", None) or {}
    for pid, proj in idx.items():
        if not str(pid).startswith("omw-"):
            continue
        meta = (proj.get("versions") or {}).get(OMW_VERSION)
        if meta is None:
            continue
        lic = (meta.get("license") if isinstance(meta, dict) else None) or proj.get("license") or ""
        out[str(pid)] = {"license": str(lic), "label": str(proj.get("label", pid)),
                         "language": str(proj.get("language", ""))}
    return out


def ingest_stage0_omw(store, *, author: str = DEFAULT_AUTHOR, force: bool = False,
                      limit: Optional[int] = None,
                      projects: Optional[List[str]] = None) -> Dict[str, Any]:
    """Ingest Open Multilingual Wordnet vocabulary for license-vetted languages.

    OMW lexicons are expand-style: they add vocabulary — each language's lemmas — onto the shared
    concept inventory rather than re-carving it (runbook H1.1). So each foreign synset lands in the
    wn-row shape (ct text/x-wordnet, its language's lemmas keyed for retrieval, `lang` and `ili`
    fields) and attaches to the local pivot, the existing synset carrying the same ILI, with a typed
    `ili` edge. Where there is no pivot, the row lands unlinked and is counted in `no_pivot`.

    Vetting is `OMW_LICENSE_ALLOWLIST` above; a language that is available but unvetted, or whose
    license metadata has drifted, is skipped and logged one line each, for clearing via the
    TRAINING-QUEUE prep column. Per-language completion markers make this resumable and let a grown
    allowlist ingest what is newly cleared."""
    try:
        import wn
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "op.source.omw needs the `wn` package (the bootstrap extra): pip install wn"
        ) from e
    author = g._author_ref(store, author)
    if not g.is_bootstrapped(store):
        g.bootstrap(store, author=author)

    _ensure_ctype(store, g.WORDNET_CONTENT_TYPE, author=author, **_WN_TYPE)
    g.mint_source_triple(store, "omw", cite_meta={
        "dataset": "Open Multilingual Wordnet 1.4 (license-vetted subset)",
        "provider": "Global WordNet Association / per-language projects",
        "license": "per-language — only allowlisted clearly-permissive lexicons ingest "
                   "(CC BY 3.0 / MIT / Apache-2.0 / ODC-BY); see OMW_LICENSE_ALLOWLIST",
        "version": OMW_VERSION, "via_package": "wn",
    }, offer="adds license-vetted multilingual vocabulary onto the lexicon spine: each "
             "language's lemmas as keyed wn-shaped rows, attached to their interlingual "
             "pivot synsets by `ili` edges (expand-style — vocabulary, not carving)",
        author=author)
    _ensure_etype(store, "ili", "synset -> synset",
                  "interlingual pivot: both endpoints express the same CILI concept "
                  "(props carry the ili id)", author)

    # stage wn's own downloads under the store's established cache location
    try:
        wn_dir = _downloads_dir() / "wn"
        wn_dir.mkdir(parents=True, exist_ok=True)
        wn.config.data_directory = str(wn_dir)
    except Exception:
        pass                                       # wn falls back to its default data dir

    avail = _omw_index(wn)
    done = set(g._shards_done(store, "omw"))
    wanted = projects if projects is not None else sorted(avail)
    skipped: List[Tuple[str, str, str]] = []       # (id, license, why)
    summary = {"languages": 0, "synsets": 0, "edges": 0, "no_pivot": 0, "no_words": 0}
    pivots: Optional[Dict[str, List[str]]] = None  # built once, on first admitted language

    for pid in wanted:
        meta = avail.get(pid)
        if meta is None:
            skipped.append((pid, "", "not in the wn index at OMW %s" % OMW_VERSION))
            continue
        lic = meta["license"]
        if pid not in OMW_LICENSE_ALLOWLIST:
            skipped.append((pid, lic, "not on the license allowlist — clear via the "
                                      "TRAINING-QUEUE prep column"))
            continue
        expect = OMW_LICENSE_ALLOWLIST[pid][0]
        if expect not in lic:
            skipped.append((pid, lic, "license metadata drifted from the vetted expectation "
                                      "%r — re-vet before ingesting" % expect))
            continue
        if pid in done and not force:
            continue                               # this language already drained
        spec = f"{pid}:{OMW_VERSION}"
        try:
            have = bool(wn.lexicons(lexicon=spec))
        except Exception:
            have = False
        if not have:
            wn.download(spec)                      # GET-only under the hood (wn fetches over https)
        w = wn.Wordnet(lexicon=spec)
        lex = w.lexicons()
        lang = (lex[0].language if lex else "") or meta["language"] or pid.split("-", 1)[-1]
        if pivots is None:
            pivots = _ili_pivot_map(store)
        docs: List[Dict[str, Any]] = []
        edge_buf: List[tuple] = []
        n_lang = 0
        for ss in w.synsets():
            if limit is not None and n_lang >= limit:
                break
            words, _seen = [], set()          # source order carries primacy; dedupe preserves it
            for _f in ss.lemmas():
                _f = (_f or "").replace("_", " ")
                if _f and _f not in _seen:
                    _seen.add(_f); words.append(_f)
            if not words:
                summary["no_words"] += 1
                continue
            ili_obj = ss.ili
            ili = getattr(ili_obj, "id", ili_obj) if ili_obj is not None else None
            aid = "wn-" + ss.id
            defn = ss.definition() or ""
            # Observation only, the same contract as op.source.oewn: fields as observed, and no
            # `context`. The offer is an interpretation, so the describer supplies it at
            # illuminate, and describers evolve.
            doc = {
                "id": aid,
                "content_type": g.WORDNET_CONTENT_TYPE,
                "state": "committed",
                "title": words[0],
                "gloss": defn,
                "content": defn,
                "lemmas": [x.lower() for x in words],
                "word": words[0].lower(),
                "pos": ss.pos,
                "lang": lang,
                "collection_id": "stage.0.lexicon",
                "collections": ["stage.0.lexicon", "source.omw"],
                "cited_from": "cite.omw",
                "via": "op.source.omw",
                "operator": "op.source.omw",
                "provenance": g.P_OBSERVED,
                "created_by": author,
                "created_time": g._now(),
            }
            if ili:
                doc["ili"] = ili
                cands = [c for c in pivots.get(ili, []) if c != aid]
                if cands:
                    # prefer the OEWN row (the maintained English lattice) as the pivot end;
                    # else deterministic first. CILI's own edges close the rest of the star.
                    pivot = next((c for c in sorted(cands) if c.startswith("wn-oewn-")),
                                 sorted(cands)[0])
                    edge_buf.append((aid, pivot, "ili",
                                     {"ili": ili, "via": "op.source.omw", "rung": g.P_OBSERVED,
                                      "cited_from": "cite.omw"}))
                else:
                    summary["no_pivot"] += 1
            docs.append(doc)
            n_lang += 1
            if len(docs) >= 2000:
                summary["synsets"] += store.artifacts.put_many(_sealed(store, docs))
                docs = []
        if docs:
            summary["synsets"] += store.artifacts.put_many(_sealed(store, docs))
        summary["edges"] += store.graph.add_edges(edge_buf, batch=1000)
        if limit is None:                          # a bounded smoke run is not completion
            g._mark_shard_done(store, "omw", pid)
        summary["languages"] += 1

    # The skip log — one line per language, with its license and the reason (the worker.log idiom:
    # stdout, flushed). This is the list cleared via the QUEUE prep column.
    for pid, lic, why in skipped:
        print("[op.source.omw] SKIPPED %s (license: %s): %s" % (pid, lic or "?", why),
              flush=True)

    from ember.runtime.runner import evolution
    if summary["synsets"]:
        evolution.record_invocation(store.artifacts, "op.source.omw", verified=True)
    return {**summary, "skipped": [{"id": p, "license": l, "why": w} for p, l, w in skipped],
            "ingested": summary["synsets"]}
