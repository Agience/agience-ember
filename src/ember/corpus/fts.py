"""Lexical retrieval — FTS5 BM25. LATTICE-IMPLEMENTATION.md §2.1, Phase 2.1.

This index holds PLAINTEXT terms — read this before indexing anything
────────────────────────────────────────────────────────────────────────
Every word of every document indexed here is written to the SQLite file in the
clear, as FTS5 posting lists. Anyone who can read the `.db` file can enumerate
the corpus vocabulary (`SELECT term FROM fts_vocab`) and can reconstruct a good
deal of each document from its postings. Stemmed terms, their document
frequencies and their per-document counts are all recoverable without a key.

That is the trade this module makes, deliberately: FTS5's Okapi BM25 runs inside
the SQLite core over cleartext postings, which is why ranking costs nothing here
and needs no client-side work.

**It inherits no encryption property from mantle.** Mantle removed this module
(commit `a7cfd0c`) as a privacy decision rather than a cleanup. Mantle's search
path is a searchable-symmetric-encryption index: the server holds HMAC'd tokens
and never words, precisely so that holding the index does not disclose the
corpus. This module is the opposite arrangement. A node that vendors it does not
acquire mantle's property, and nothing in mantle's SSE design covers what gets
written here.

Whether a plaintext index is acceptable is ember's call to make — an ember node's
lattice is node-local, and may hold only content that node already stores in the
clear — but it is a call, made per corpus, not a property inherited by moving the
file. If anything indexed here is not already plaintext-at-rest on this node,
this is the wrong index for it and mantle's SSE index is the right one.
────────────────────────────────────────────────────────────────────────

Provenance
──────────
Vendored from `agience-mantle`, `src/mantle/db/lattice/fts.py` (Apache-2.0,
That source path no longer exists in `agience-mantle` and is LEFT VERBATIM on purpose:
an attribution records where code came from at the time it was taken, not where to find
it now. `db/lattice/` was flattened to `db/`, and no `fts.py` survives anywhere in mantle
— so this vendored copy is the surviving one. Repointing it would turn a dated provenance
record into a false claim about today's tree. (checked 2026-08-26.)
Copyright Ikailo Inc.), at commit `9768c5f`. Ember is its only remaining
consumer: `ember.ontology.corpus_stats`, `scripts/lattice_*.py`, `node/*.py` and
`tests/test_fts_index.py`. Copied unmodified apart from this header, one import
(`coverage`'s relative `from . import schema` became an absolute
`mantle.db.schema` import — that module is still mantle's and still exists) and
the logger name. The schema, tokenizer, stemmer and ranking are byte-identical,
so an index a mantle build wrote is read correctly by this copy.

Stemming is FTS5's `porter`: `quickly → quickli`, `happy → happi`,
`universities → univers`. Stemmer choice is part of the index format — a change
to it alters every document's term profile, so BM25's IDF, so ranking
corpus-wide — which is why the index is rebuilt rather than converted whenever
the tokenizer moves.

There are no field weights. `bm25()` is called with no weight arguments, so
every column contributes equally and there is no knob to set. FTS5 supports
column weights natively; leaving them off keeps the mechanism unconfounded when
it is measured.

No prefix tokens (px3/px4/px5) are written. Nothing reads them, so writing them
would be pure write amplification.

Contentless index and span extraction
─────────────────────────────────────
The schema is `content=''` (contentless): FTS5 stores postings, never text, so
FTS5's `snippet()` returns NULL rather than raising. Measured on SQLite 3.49.1:

    SELECT snippet(fts, 3, '[', ']', '...', 10) FROM fts WHERE fts MATCH 'quick'
    -> [(None,)]          # NULL, not an exception

`highlight()` and plain column reads (`SELECT content FROM fts`) behave the same
way. A NULL snippet becomes empty `content`, Lumen's `if not content and not
title: continue` drops every hit, and grounding goes to zero with no error and no
log entry — §A.1's trap 1, reached by a different route.

So span extraction runs in Python against source text fetched through a
`TextResolver` (below), and `_stem_exact()` uses FTS5's own porter stemmer via a
scratch table. A second Porter implementation would drift from the index and
mis-locate spans.

The Lumen contract — §A.1
─────────────────────────
The lumen tekton's op.retrieve (`agience-crystal/src/crystal/operators/impl/
retrieval.py`) is the consumer. Four load-bearing facts:

1. Snippets are returned in `content`. Lumen sends `"highlight": true` but reads
   only `content` (`retrieval.py:72`). `to_search_hits()` (aliased
   `to_lumen_hits`) is the enforced boundary; `test_fts.py` asserts it.
2. Rank order is the contract. `score` is parsed (`retrieval.py:77`) and never
   used — Lumen walks the array in order.
3. Budget: 1400 chars/doc, 7000 total, top-6 (`retrieval.py:23-29`). Lumen
   truncates at 1400, so returning a whole document wastes the window; ~1400
   chars of the best-matching span carries more.
4. Flat hits, no nesting — `content`, `title`, `description`, `version_id`/`id`,
   `score`.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ── the schema, LATTICE-IMPLEMENTATION.md §2.1 ───────────────────────────────
# Column order is load-bearing: bm25() takes weights POSITIONALLY, so reordering
# these silently re-assigns every weight to the wrong field.
FIELDS: Tuple[str, ...] = ("title", "description", "tags", "content")

FTS_TABLE = "fts"
MAP_TABLE = "fts_map"
# fts5vocab over the index: one row per distinct term with `doc` = the number of documents
# carrying it. That IS document frequency, read off the index — exact, no scan, no cap. It exists
# so nothing has to approximate df by counting matches with a LIMIT (which saturates: every term
# past the limit reports the same number and becomes indistinguishable).
VOCAB_TABLE = "fts_vocab"

# Lumen's budget (§A.1 / retrieval.py:23-29). Ours must not exceed Lumen's, or
# we spend bytes it will throw away.
PER_DOC_CHARS = 1400
TOTAL_CHARS = 7000
TOP_K = 6


@dataclass
class FtsDocument:
    """One indexable unit, keyed on `vertex.id` (TEXT)."""
    vertex_id: str
    title: str = ""
    description: str = ""
    tags: str = ""
    content: str = ""

    def field_values(self) -> Tuple[str, str, str, str]:
        return (self.title or "", self.description or "",
                self.tags or "", self.content or "")


@dataclass
class FtsHit:
    """One result. `rank` is 0-based and IS the contract (§A.1 point 2)."""
    vertex_id: str
    rank: int
    score: float
    title: str = ""
    description: str = ""
    content: str = ""          # the extracted SPAN — never empty when text exists
    matched_terms: Tuple[str, ...] = ()


# A TextResolver maps vertex ids -> their source text. The index is contentless
# by design, so snippets REQUIRE this.
TextResolver = Callable[[Sequence[str]], Mapping[str, FtsDocument]]


# ═══════════════════════════════════════════════════════════════════════════
# Tokenization — approximately unicode61, and safe where it is not
# ═══════════════════════════════════════════════════════════════════════════
# Aligns exactly with FTS5's own token stream across ascii prose, snake_case
# identifiers, apostrophes, hyphenation, accents, CJK, version/date numerals,
# URLs, source code, and mixed whitespace (test_fts.py::test_tokenizer_aligns_with_fts5_offsets).
#
# Not reproducing FTS5's tokenizer exactly is safe because token offsets never cross the boundary
# between the two: `extract_span` tokenizes the query and the document with this function and
# compares stems, and never consumes fts5vocab offsets. A divergence therefore cannot mis-locate a
# span onto the wrong text — the worst case is that a token FTS5 matched has no counterpart here,
# no cluster is found, and the span falls back to head-of-text. Degraded snippet quality, bounded,
# never wrong content and never empty (test_emoji_divergence_degrades_gracefully).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize_spans(text: str) -> List[Tuple[int, int, str]]:
    """-> [(char_start, char_end, token_lowercased)] in document order."""
    return [(m.start(), m.end(), m.group(0).lower())
            for m in _TOKEN_RE.finditer(text or "")]


class _Stemmer:
    """FTS5's own porter stemmer, reached through a scratch table.

    Reimplementing Porter here would drift from the index — the index would
    contain one stem and the span extractor would look for another, so spans
    would silently mis-locate while BM25 still ranked correctly. That failure
    shows up as bad snippets, never as an error. So: ask FTS5.

    One word per rowid, then `fts5vocab(...,'instance')` maps doc->term, giving
    an exact word->stem table in a single batched round-trip. Cached, so a warm
    process pays ~nothing.
    """

    def __init__(self) -> None:
        # SHARED across threads, and safe to be: a dict get/set is atomic under the GIL, and the
        # worst a race can do is stem the same word twice and store the same answer twice.
        self._cache: Dict[str, str] = {}
        # PER THREAD, and it must be. This object is a module-level singleton (`_STEMMER`), and a
        # `sqlite3.Connection` may only be used from the thread that created it. Holding one on
        # the instance meant the first thread to miss the cache created it and every other thread
        # raised `ProgrammingError: SQLite objects created in a thread can only be used in that
        # same thread` — under a server that runs requests on a threadpool, that is almost every
        # request after the first.
        #
        # The cache is what made it survive review: the connection is touched only on a MISS, so a
        # warm word works from any thread and the failure is intermittent and word-dependent
        # rather than immediate. `document_frequency` catches `sqlite3.Error` and returns None,
        # and `corpus_stats._salient` reads None as "unmeasurable" and keeps every term — so a
        # cross-thread call did not fail, it silently stopped weighting. Measured on the live
        # recall path, the first query filtered its stems and every later one did not:
        #
        #     'what is a glacier'      -> ['what', 'glacier']            674 artifacts narrowed
        #     'what is photosynthesis' -> ['what', 'is', 'photosynthesi']  7,655
        #     'what is a volcano'      -> ['what', 'is', 'a', 'volcano']  58,209
        #
        # Same query, different answer depending on what ran before it in the process.
        #
        # A scratch table per thread costs one in-memory FTS5 index per thread and needs no lock.
        self._local = threading.local()

    def _ensure(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(":memory:")
            c.execute("CREATE VIRTUAL TABLE s USING fts5"
                      "(w, content='', tokenize='porter unicode61')")
            c.execute("CREATE VIRTUAL TABLE sv USING fts5vocab(s,'instance')")
            self._local.conn = c
            # Monotonic rowid high-water mark, per thread with the connection it indexes into.
            # The scratch table is contentless, so DELETE is as unavailable here as on the main
            # index. Each batch instead claims a fresh rowid range and reads back only that range.
            self._local.next = 1
        return c

    def stem_many(self, words: Iterable[str]) -> Dict[str, str]:
        want = {w.lower() for w in words if w}
        missing = sorted(want - self._cache.keys())
        if missing:
            c = self._ensure()
            base = self._local.next
            for i, w in enumerate(missing):
                c.execute("INSERT INTO s(rowid, w) VALUES(?,?)", (base + i, w))
            self._local.next = base + len(missing)
            got = dict(c.execute(
                "SELECT doc, term FROM sv WHERE doc >= ? AND doc < ?",
                (base, self._local.next)).fetchall())
            for i, w in enumerate(missing):
                # A word that tokenizes to nothing (pure punctuation) stems to
                # itself rather than vanishing, so callers never get a KeyError.
                self._cache[w] = got.get(base + i, w)
        return {w: self._cache[w.lower()] for w in want}

    def stem(self, word: str) -> str:
        return self.stem_many([word])[word.lower()]


_STEMMER = _Stemmer()


def document_frequency(conn: sqlite3.Connection, term: str) -> Optional[int]:
    """How many documents carry `term` — exact, read off `fts5vocab`.

    The vocab table holds exactly the terms the index wrote, which are stems. Both the term AS
    GIVEN and its stem are looked up, and the LARGER count is returned: the index stored the mass
    under exactly one form, and the larger count is what says which. Which form a caller holds is
    not knowable from the word, so neither lookup can be skipped.

    ## Why "as given, and only stem if that misses" was not enough (§105 -> 2026-08-27)

    That rule is right whenever the surface form is ABSENT — `ontology` misses, stems to
    `ontolog`, reads 409. It is wrong when the surface form is PRESENT with a spurious small
    count, because the real mass under the stem is then never consulted. Measured on 71/home:

        this        as-given  2      stem `thi`      25,706     12,853x understated
        collection  as-given  1      stem `collect`   5,233      5,233x understated
        operator    as-given 11      stem `oper`      8,488        772x understated

    896 of the 904 terms where both forms exist were under-reported — 9.2% of all word-occurrences
    in real workspace prose, and the affected list is this domain's own vocabulary rather than
    only function words. The consequence is the one recorded below for the other direction: one
    hugely-overstated IDF raises `_salient`'s mean above every other term and silences the query.

    Those surface entries are CONTAMINATION and are small because of it: `porter unicode61` stems
    every token it is given (`this` -> `thi`), so a literal `this` cannot be written to this vocab
    at all. They came from writes made under a different tokenizer. Taking the larger of the two
    is correct in both directions at once, so neither repair needs to know which kind of caller it
    is serving.

    ## Stemming is not idempotent, and this stemmed unconditionally (§105)

    Porter's step 1a strips a trailing `s`, and it does not ask whether it has run before. A stem
    ending in one `s` is therefore a DIFFERENT word the second time through:

        univers  ->  univ          vocab[univers] = 20,326      vocab[univ] = 27

    Every caller of this function passes stems — `_salient` is handed the query's analyzed terms —
    so the frequency it read was the frequency of a word nobody asked about, 750x too small. An
    IDF built on it is enormous, and `_salient` keeps terms by comparing IDFs, so ONE such term
    silenced every other word of the query. `universal artifact model` searched for `univers`
    alone and answered `undergrad`, `cosmic time`, `prof`.

    Porter fixed points — `water`, `prism`, `protocol`, `doe` — were never affected, which is why
    this survived: it is wrong only for the terms it is wrong for, and they look like ordinary
    words.

    Returns None when the index has no vocab table: unmeasurable, and the caller
    must say so rather than substitute a number.
    """
    word = (term or "").lower()
    if not word:
        return None
    try:
        row = conn.execute(
            f"SELECT doc FROM {VOCAB_TABLE} WHERE term = ?", (word,)
        ).fetchone()
        surface = int(row[0]) if row else None
        stem = _STEMMER.stem_many([word]).get(word)
        stemmed = None
        if stem and stem != word:
            row = conn.execute(
                f"SELECT doc FROM {VOCAB_TABLE} WHERE term = ?", (stem,)
            ).fetchone()
            stemmed = int(row[0]) if row else None
    except sqlite3.Error:
        return None
    if surface is None and stemmed is None:
        return 0
    # The index stored the mass under exactly ONE form, and the larger count says which. Both
    # lookups always run, because which form a caller holds is not knowable from the word.
    return max(surface or 0, stemmed or 0)


def build_match_query(text: str, *, conjunctive: bool = False) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    OR is the default. Lumen sends natural-language questions; under AND a
    single absent word yields zero hits and therefore zero grounding, whereas
    under OR BM25 does the discriminating — which is its job.
    """
    terms = [t for _, _, t in tokenize_spans(text)]
    if not terms:
        return ""
    joiner = " AND " if conjunctive else " OR "
    return joiner.join('"' + t.replace('"', '""') + '"' for t in terms)


# ═══════════════════════════════════════════════════════════════════════════
# Span extraction — the replacement for the NULL-returning snippet()
# ═══════════════════════════════════════════════════════════════════════════

def extract_span(text: str, query_terms: Sequence[str], *,
                 budget: int = PER_DOC_CHARS,
                 ellipsis: str = "…") -> str:
    """Return <= `budget` chars of `text` centred on the densest match cluster.

    Never returns empty for non-empty input. That is a hard requirement, not a
    nicety: an empty `content` is dropped by Lumen (§A.1 trap 1), so a document
    that matched on `title` alone but has no query term in `content` must still
    come back with its head text rather than "".
    """
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= budget:
        return text

    spans = tokenize_spans(text)
    if not spans or not query_terms:
        return _trim(text, 0, budget, ellipsis)

    stems = _STEMMER.stem_many([t for _, _, t in spans] + list(query_terms))
    wanted = {stems.get(q.lower(), q.lower()) for q in query_terms}
    hits = [(s, e, stems.get(tok, tok)) for s, e, tok in spans
            if stems.get(tok, tok) in wanted]
    if not hits:
        return _trim(text, 0, budget, ellipsis)

    # Densest window: two pointers over match positions, with the distinct-term
    # count maintained incrementally. Scoring prefers distinct terms over raw
    # repetition — a passage covering three query terms once is better evidence
    # than one repeating a single term nine times.
    counts: Dict[str, int] = {}
    distinct = 0
    best_score, best_lo, best_hi = -1, hits[0][0], hits[0][1]
    lo = 0
    for hi in range(len(hits)):
        term = hits[hi][2]
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            distinct += 1
        while hits[hi][1] - hits[lo][0] > budget:
            drop = hits[lo][2]
            counts[drop] -= 1
            if counts[drop] == 0:
                distinct -= 1
            lo += 1

        score = distinct * 1000 + (hi - lo + 1)
        if score > best_score:
            best_score, best_lo, best_hi = score, hits[lo][0], hits[hi][1]

    centre = (best_lo + best_hi) // 2
    start = max(0, centre - budget // 2)
    return _trim(text, start, budget, ellipsis)


def _trim(text: str, start: int, budget: int, ellipsis: str) -> str:
    """Cut `budget` chars from `start`, snapped to word boundaries."""
    end = min(len(text), start + budget)
    if start > 0:
        nxt = text.find(" ", start)
        if 0 <= nxt < start + 40:
            start = nxt + 1
    if end < len(text):
        prev = text.rfind(" ", start, end)
        if prev > start:
            end = prev
    out = text[start:end].strip()
    if start > 0:
        out = ellipsis + " " + out
    if end < len(text):
        out = out + " " + ellipsis
    # Snapping must never push us back over budget.
    return out[:budget].strip()


# ═══════════════════════════════════════════════════════════════════════════
# The index
# ═══════════════════════════════════════════════════════════════════════════

class FtsIndex:
    """Contentless FTS5 index over `vertex`, with an explicit rowid mapping.

    1. **rowid reuse.** A plain INTEGER PRIMARY KEY reissues `max(rowid)+1`, so
       deleting the highest row and inserting a new one reuses its rowid. Any
       posting that outlived the delete now resolves to a different vertex —
       wrong answers, not missing ones. `AUTOINCREMENT` keeps a monotonic
       high-water mark in `sqlite_sequence`; ids are never reused.
    2. **Re-index without delete.** FTS5 will happily hold two posting sets for
       one rowid; term frequencies double and BM25 quietly favours whatever was
       indexed twice. `index()` is always delete-then-insert on the same rowid,
       so the mapping is stable across updates.
    3. **Torn writes.** `fts_map` and `fts` diverging leaves orphan postings
       that resolve to nothing. Both moves happen in one transaction.
    4. **Delete without stored text.** A plain contentless table cannot
       `DELETE`; it requires re-supplying the original column values to the
       special 'delete' command. If the text changed, or the CAS blob was
       evicted, the delete corrupts the index with no error. `contentless_delete=1`
       (SQLite >= 3.45) removes that requirement entirely — see below.
    """

    def __init__(self, conn: sqlite3.Connection, *,
                 resolver: Optional[TextResolver] = None,
                 contentless_delete: bool = True) -> None:
        self.conn = conn
        self.resolver = resolver
        self._contentless_delete = contentless_delete and self._supports_cd()

    @staticmethod
    def _supports_cd() -> bool:
        return sqlite3.sqlite_version_info >= (3, 45, 0)

    # ── schema ───────────────────────────────────────────────────────────────
    def ensure_schema(self) -> None:
        """Idempotent.

        `contentless_delete=1` is appended to the §2.1 DDL. It changes nothing about
        what is indexed, the tokenizer, or ranking — the table stays contentless and
        stores no text. It only makes deletion sound, for two reasons:

        1. A plain contentless table cannot `DELETE` at all; removal requires
           re-supplying the exact original field text to the special 'delete'
           command. Supplying the wrong text does not error — it silently
           corrupts the index.
        2. Eviction under valence is the steady state, not an edge case: on overflow
           a vertex darkens to valence-2, content is dropped and `content_ref` is
           kept, so the FTS index routinely holds postings for text that is no
           longer available to re-supply. Under the literal DDL every such eviction
           would then either refuse its delete or corrupt the index.

        `contentless_delete=False` gives the literal spec DDL and makes
        re-index raise rather than corrupt. Requires SQLite >= 3.45 (detected).
        """
        opt = ", contentless_delete=1" if self._contentless_delete else ""
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5("
            f"  {', '.join(FIELDS)},"
            f"  content=''{opt}, tokenize='porter unicode61')"
        )
        self.conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {MAP_TABLE} (
                  fts_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                  vertex_id TEXT NOT NULL UNIQUE
                )"""
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {VOCAB_TABLE} "
            f"USING fts5vocab({FTS_TABLE}, 'row')"
        )

    # ── mapping ──────────────────────────────────────────────────────────────
    def rowid_for(self, vertex_id: str, *, create: bool = False) -> Optional[int]:
        row = self.conn.execute(
            f"SELECT fts_rowid FROM {MAP_TABLE} WHERE vertex_id = ?", (vertex_id,)
        ).fetchone()
        if row:
            return int(row[0])
        if not create:
            return None
        cur = self.conn.execute(
            f"INSERT INTO {MAP_TABLE}(vertex_id) VALUES(?)", (vertex_id,))
        return int(cur.lastrowid)

    def vertex_for(self, fts_rowid: int) -> Optional[str]:
        row = self.conn.execute(
            f"SELECT vertex_id FROM {MAP_TABLE} WHERE fts_rowid = ?", (fts_rowid,)
        ).fetchone()
        return row[0] if row else None

    # ── writes ───────────────────────────────────────────────────────────────
    def index(self, doc: FtsDocument) -> int:
        """Insert or replace. Returns the stable fts rowid.

        Purge runs only on a re-index (the vertex already has a rowid). A fresh insert has a
        brand-new rowid, and `fts_rowid` is AUTOINCREMENT so an id is never reissued — a new
        rid cannot carry stale postings, so there is nothing to purge. This matters on older
        SQLite (< 3.45, e.g. a lean Pi): `_purge_postings` raises without `contentless_delete`,
        so calling it on every fresh insert would make a whole-corpus ingest impossible there."""
        existing = self.rowid_for(doc.vertex_id, create=False)
        if existing is not None:
            # Re-index. With contentless_delete (SQLite >= 3.45) purge the old posting then
            # reinsert. Without it (older SQLite, e.g. a lean Pi at 3.34) a contentless table
            # cannot delete, so the original posting stays. That is correct for an idempotent
            # re-put (same content — the overwhelming case in a curriculum ingest, e.g.
            # ConceptNet re-touching a concept it already created to hang another edge off it),
            # and leaves a stale posting only if the field text genuinely changed, which a
            # periodic FTS rebuild refreshes.
            if not self._contentless_delete:
                return existing
            rid = existing
            self._purge_postings(rid)
        else:
            rid = self.rowid_for(doc.vertex_id, create=True)   # fresh: nothing to purge
        self.conn.execute(
            f"INSERT INTO {FTS_TABLE}(rowid, {', '.join(FIELDS)}) "
            f"VALUES(?,?,?,?,?)", (rid,) + doc.field_values())
        return rid

    def index_many(self, docs: Iterable[FtsDocument]) -> int:
        n = 0
        for d in docs:
            self.index(d)
            n += 1
        return n

    def _purge_postings(self, rid: int) -> None:
        """Remove any existing postings for `rid`. Failure mode 2 above."""
        if self._contentless_delete:
            self.conn.execute(f"DELETE FROM {FTS_TABLE} WHERE rowid = ?", (rid,))
            return
        # Literal-spec fallback: the 'delete' command needs the original values,
        # which a contentless table cannot supply. Callers on this path must
        # rebuild rather than update in place.
        raise NotImplementedError(
            "re-index requires contentless_delete=1; a plain contentless FTS5 "
            "table cannot delete without the original field text (see "
            "ensure_schema docstring). Rebuild the index instead."
        )

    def delete(self, vertex_id: str) -> bool:
        rid = self.rowid_for(vertex_id)
        if rid is None:
            return False
        self._purge_postings(rid)
        # The map row goes too, but AUTOINCREMENT guarantees the id is never
        # reissued to a different vertex.
        self.conn.execute(f"DELETE FROM {MAP_TABLE} WHERE fts_rowid = ?", (rid,))
        return True

    # ── read ─────────────────────────────────────────────────────────────────
    def search(self, query_text: str, *,
               limit: int = TOP_K,
               per_doc_chars: int = PER_DOC_CHARS,
               total_chars: int = TOTAL_CHARS,
               conjunctive: bool = False,
               resolver: Optional[TextResolver] = None) -> List[FtsHit]:
        """BM25 search. Results are returned in rank order — that is the
        contract (§A.1 point 2); callers walk the list, they do not re-sort.
        """
        match = build_match_query(query_text, conjunctive=conjunctive)
        if not match:
            return []
        rows = self.conn.execute(
            f"SELECT rowid, bm25({FTS_TABLE}) AS s "
            f"FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ? "
            f"ORDER BY s ASC LIMIT ?",
            (match, limit),
        ).fetchall()
        if not rows:
            return []

        ids: List[str] = []
        scored: List[Tuple[str, float]] = []
        for rid, score in rows:
            vid = self.vertex_for(int(rid))
            if vid is None:
                # An orphan posting: indexed, but its map row is gone. Skipping
                # is right (we cannot name the vertex) but it is a real
                # inconsistency, so it must not pass silently.
                continue
            ids.append(vid)
            scored.append((vid, float(score)))

        res = resolver or self.resolver
        texts: Mapping[str, FtsDocument] = res(ids) if res else {}
        terms = [t for _, _, t in tokenize_spans(query_text)]

        hits: List[FtsHit] = []
        spent = 0
        for rank, (vid, score) in enumerate(scored):
            src = texts.get(vid)
            body, title, desc = "", "", ""
            if src is not None:
                title, desc = src.title or "", src.description or ""
                # Prefer a span from `content`; fall back to description then
                # title so `content` is NEVER empty when the doc has any text
                # (§A.1 trap 1).
                for candidate in (src.content, src.description, src.title):
                    if (candidate or "").strip():
                        body = extract_span(candidate, terms, budget=per_doc_chars)
                        break
            if spent and spent + len(body) > total_chars:
                break
            spent += len(body)
            hits.append(FtsHit(vertex_id=vid, rank=rank, score=score,
                               title=title, description=desc, content=body,
                               matched_terms=tuple(terms)))
        return hits


# ═══════════════════════════════════════════════════════════════════════════
# The search-hit boundary — §A.1 (consumer: the lumen tekton's op.retrieve,
# crystal/operators/impl/retrieval.py)
# ═══════════════════════════════════════════════════════════════════════════

def to_search_hits(hits: Sequence[FtsHit]) -> List[dict]:
    """Serialize to the retrieval consumer's flat hit shape.

    Flat, no nesting (§A.1). `score` is emitted for compatibility but Lumen
    never uses it — list order is what carries rank.
    """
    return [{"id": h.vertex_id,
             "version_id": h.vertex_id,
             "title": h.title,
             "description": h.description,
             "content": h.content,
             "score": h.score}
            for h in hits]


# Alias kept for external callers; the boundary is named for what it does
# (`to_search_hits`) rather than for the lumen tekton that consumes it.
to_lumen_hits = to_search_hits


# ═══════════════════════════════════════════════════════════════════════════
# Default resolver
# ═══════════════════════════════════════════════════════════════════════════

ContentLoader = Callable[[str], Optional[str]]


class PreviewOnlyIndex(RuntimeError):
    """Raised when full text was required but only a CAS preview was available."""


def _coerce_context(raw) -> dict:
    """`context` arrives either as a nested object or as a JSON-encoded string.

    Over the Lumen POST path `content` and `context` are posted as JSON-encoded
    strings, and `context` is doubly encoded; other extract paths store it already
    decoded. Both shapes are real, so both are handled — the loop below tolerates
    double encoding rather than assuming a fixed depth.
    """
    import json
    for _ in range(3):
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def make_vertex_resolver(conn: sqlite3.Connection,
                         *, doc_column: str = "doc",
                         content_loader: Optional[ContentLoader] = None,
                         strict: bool = False) -> TextResolver:
    """Resolve source text for a set of `vertex.id`s.

    `doc.content` is a ~300-char preview that typically cuts mid-word; the full
    plaintext lives in CAS under `content_ref` (contract §2: *"cas/<sha256(plaintext)>;
    NULL = inline in doc"*). Indexing `doc.content` directly builds an index over
    previews: FTS5 indexes the 300 chars without error, BM25 scores look sane,
    snippets render, and lexical recall for anything past the first ~50 words of
    every article is simply gone — the same shape as §A.1 trap 1, a failure with
    no signal.

    So `content_loader` (`content_ref -> plaintext`) is required for a real
    index. Without it this resolver runs in preview mode and says so:
      * `strict=True`  -> raises `PreviewOnlyIndex` on the first truncated row.
      * `strict=False` -> logs a warning once and counts the rows
        (`resolve.preview_only`), so a caller can assert on it.
    """
    import json
    import logging

    log = logging.getLogger("ember.corpus.fts")

    def resolve(vertex_ids: Sequence[str]) -> Mapping[str, FtsDocument]:
        if not vertex_ids:
            return {}
        qs = ",".join("?" * len(vertex_ids))
        out: Dict[str, FtsDocument] = {}
        for vid, ref, raw in conn.execute(
            f"SELECT id, content_ref, {doc_column} FROM vertex WHERE id IN ({qs})",
            tuple(vertex_ids),
        ):
            try:
                d = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                d = {}
            ctx = _coerce_context(d.get("context"))

            body = str(d.get("content") or "")
            if ref:
                full = content_loader(ref) if content_loader else None
                if full is not None:
                    body = full
                else:
                    resolve.preview_only += 1        # type: ignore[attr-defined]
                    if strict:
                        raise PreviewOnlyIndex(
                            f"{vid}: content_ref={ref} but no content_loader; "
                            "indexing the preview would silently truncate the "
                            "corpus (see make_vertex_resolver docstring)")
                    if not resolve.warned:           # type: ignore[attr-defined]
                        resolve.warned = True        # type: ignore[attr-defined]
                        log.warning(
                            "fts: indexing CAS previews, not full text (%s has "
                            "content_ref but no content_loader). Lexical recall "
                            "will be silently truncated.", vid)

            tags = ctx.get("tags") or ctx.get("sources") or ctx.get("kind") or ""
            if isinstance(tags, (list, tuple)):
                tags = " ".join(str(t) for t in tags)
            title = ctx.get("name") or ctx.get("title") or ""
            if not title and body:
                title = body.splitlines()[0][:200]   # markdown: first line
            out[vid] = FtsDocument(
                vertex_id=vid,
                title=str(title),
                description=str(ctx.get("description") or ctx.get("role") or ""),
                tags=str(tags),
                content=body)
        return out

    resolve.preview_only = 0     # type: ignore[attr-defined]
    resolve.warned = False       # type: ignore[attr-defined]
    return resolve


# ═══════════════════════════════════════════════════════════════════════════
# The lexical index — artifact projection, maintenance, and coverage
# ═══════════════════════════════════════════════════════════════════════════
# This is the one lexical index the runtime uses, maintained from the write path
# (`index_artifacts`) rather than built by a separate migration script.
#
# `porter` stems the query onto the lemma rather than matching raw surface
# forms, which is what makes queries like "dogs" or "a bank that holds money"
# rank on dog breeding / bank holding company rather than on incidental
# co-occurrences of the literal word.
#

def project_artifact(doc: Mapping) -> FtsDocument:
    """An artifact document -> the indexable four fields.

    `description` IS the offer — the one thing a need is matched against — and it is read
    top-level. A structured `context` is a compatibility fallback for rows written before that was
    true, and `gloss` a last resort so a dark artifact is still findable and can therefore still be
    described.

    A bare string in `context` is provenance rather than an offer, and is not read here. Two
    ingests write exactly such a string:

        sage/canon.py        "canon knowledge: best-practices §intro"
        stage0_sources.py    "the concept 0: a ConceptNet 5.7 English term node"

    Read as `desc = ctx if isinstance(ctx, str) else ""`, that makes the indexed offer of every
    canon document its origin: on the live shard all 6,480 of them then position on the same two
    nodes — `canon.n.01` and `cognition.n.01` — and a field that says the same thing about every
    member of a corpus cannot tell them apart. Provenance belongs in `citation` / `source_path` /
    `via`, which canon already carries in full.
    """
    desc = str(doc.get("description") or "").strip()
    if not desc:
        ctx = doc.get("context")
        if isinstance(ctx, dict):
            desc = str(ctx.get("description") or "").strip()
    if not desc:
        desc = str(doc.get("gloss") or "")
    tags = doc.get("lemmas") or doc.get("tags") or ()
    if isinstance(tags, str):
        tags = [tags]
    return FtsDocument(
        vertex_id=str(doc.get("id") or ""),
        title=str(doc.get("title") or ""),
        description=desc,
        tags=" ".join(str(t) for t in tags),
        content=str(doc.get("content") or ""))


# The index carries its own maintained count, for the same reason the store does: `count(*)`
# dereferences every record and this package bans it outright — `test_no_count_star_reaches_sqlite`
# enforces it. The counter is bumped in the same transaction as the posting, so it cannot drift
# from what it counts.
C_FTS_TOTAL = "fts:total"


def _bump(conn: sqlite3.Connection, delta: int) -> None:
    conn.execute("INSERT INTO counter(name, n) VALUES(?, ?) "
                 "ON CONFLICT(name) DO UPDATE SET n = n + excluded.n", (C_FTS_TOTAL, delta))


def retract_artifact(conn: sqlite3.Connection, vertex_id: str) -> bool:
    """Remove one artifact from the lexical index — postings, map row, and the counted total.

    Without this call the deleted id stays ranked first for its own words, and `fts:total`
    reads one higher than the rows the index holds (`test_delete_artifact_retracts_the_lexical_index`).

    Returns True when something was actually retracted. A store whose index was never built
    (`is_built` False — no `fts_map` table) has nothing to retract and says so rather than raising:
    a delete must not fail because a derived index was absent.
    """
    if not is_built(conn):
        return False
    conn.execute("CREATE TABLE IF NOT EXISTS counter (name TEXT PRIMARY KEY, n INTEGER NOT NULL)")
    if not FtsIndex(conn).delete(str(vertex_id)):
        return False
    _bump(conn, -1)
    return True


def index_artifacts(conn: sqlite3.Connection, docs: Iterable[Mapping]) -> int:
    """Index (or re-index) artifact documents. Returns rows indexed.

    An index is derived data, so the path that writes the rows derives it —
    nothing else knows the rows exist. Archived rows are removed rather than
    indexed: a content-decides snapshot must not surface alongside its own head.
    """
    idx = FtsIndex(conn)
    idx.ensure_schema()
    conn.execute("CREATE TABLE IF NOT EXISTS counter (name TEXT PRIMARY KEY, n INTEGER NOT NULL)")
    n, delta = 0, 0
    for d in docs:
        vid = str(d.get("id") or "")
        if not vid:
            continue
        existed = idx.rowid_for(vid) is not None
        if str(d.get("state") or "") == "archived":
            # `retract_artifact` bumps the counter itself, which is why no `delta` is applied
            # here.
            #
            # It is the only retraction path, and nothing else calls it. `vertex.delete_artifact`
            # maintains the vertex row, its edges, the merkle leaf and the `listkey` postings, but
            # not this index — mantle cannot call into ember. So a delete leaves an entry here for
            # an artifact that is gone. Search does not return it (`content_search.search` joins
            # `fts_map` to `vertex`, and the row drops out), but it keeps counting toward the
            # document frequency `corpus_stats` reads, which is what weights a query's words.
            # `_scratch/prune_fts.py` retracts what a bulk delete left behind.
            retract_artifact(conn, vid)
            continue
        idx.index(project_artifact(d))
        if not existed:
            delta += 1                 # a re-index is not a new row; the count must not double it
        n += 1
    if delta:
        _bump(conn, delta)
    return n


def rebuild_from_vertices(conn: sqlite3.Connection, *, doc_column: str = "doc") -> int:
    """Rebuild the whole index from the vertex table — for a store whose rows
    landed before the writer indexed them."""
    import json
    conn.execute(f"DROP TABLE IF EXISTS {FTS_TABLE}")
    conn.execute(f"DROP TABLE IF EXISTS {MAP_TABLE}")
    conn.execute("CREATE TABLE IF NOT EXISTS counter (name TEXT PRIMARY KEY, n INTEGER NOT NULL)")
    conn.execute("DELETE FROM counter WHERE name = ?", (C_FTS_TOTAL,))   # the rows went with it
    FtsIndex(conn).ensure_schema()
    docs = []
    for row in conn.execute(f"SELECT {doc_column} FROM vertex").fetchall():
        try:
            docs.append(json.loads(row[0]))
        except Exception:
            continue
    return index_artifacts(conn, docs)


def is_built(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (MAP_TABLE,)).fetchone() is not None


def coverage(conn: sqlite3.Connection) -> Dict[str, int]:
    """Indexed rows vs vertices — the number that shows a fresh store
    is unsearchable. Published so health monitoring reads it, rather than
    anyone probing for it by hand."""
    def _counter(name: str) -> int:
        row = conn.execute("SELECT n FROM counter WHERE name = ?", (name,)).fetchone()
        return int(row[0]) if row else 0
    try:
        n_idx = _counter(C_FTS_TOTAL)
        # `c_vertex_total()` is a function so a typo is an AttributeError at import rather than a
        # silently wrong number — a missing counter row reads as 0, which is the wrong-answer class.
        from mantle.db import schema as _schema
        n_v = _counter(_schema.c_vertex_total())
    except Exception:
        n_idx = n_v = 0
    return {"indexed": n_idx, "vertices": n_v,
            "missing": n_v - n_idx, "built": is_built(conn)}


# ── the store-level entry points ─────────────────────────────────────────────

def index_for(db, docs: Iterable[Mapping]) -> int:
    """Index documents through the lattice `db` handle (the thing with `read()`/`write()`).

    Rarely needed: the store maintains this index in its own write path, so a caller that writes
    through `put_artifact`/`put_many` never touches it. This exists for a rebuild and for tests."""
    with db.write() as cur:
        return index_artifacts(cur, docs)


def rebuild_for(db) -> int:
    """Rebuild the whole index from the vertex table — for a store whose rows landed before the
    writer maintained it."""
    with db.write() as cur:
        return rebuild_from_vertices(cur)


def coverage_for(db) -> Dict[str, Any]:
    return coverage(db.read())


def is_built_for(db) -> bool:
    try:
        return is_built(db.read())
    except Exception:
        return False
