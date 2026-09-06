"""In-memory stand-ins shared across the ember suite.

A fixture used by more than one test module lives here, in a plain helper module that `conftest.py`
puts on `sys.path`. `tests/` has no `__init__.py`, so a test module is not importable from another
one; `test_genesis.py` and `test_delegate_isolation.py` import these back by name.

Everything here is deterministic and runs without a live store.
"""
from __future__ import annotations

import json as _json
import sqlite3 as _sqlite3


# ── an in-memory stand-in for the leaf store (artifacts + labeled graph) ──────────────────────
class _FakeArtifacts:
    """An in-memory artifact store that publishes a whole-of-store `write_mark`.

    `mantle.db.vertex.Artifacts` publishes that mark and every cache in
    `crystal.ontology.driver` hangs its validity on it. For a store with no mark to publish,
    `driver._gate` advances the generation on every poll and every cached entry is therefore
    unreachable: caching is a claim that a value is still current, and that claim needs a store that
    can support it ([[absence-is-not-an-affirmative-claim]]). Production is the other case — on 71
    the real lattice answers in 14.1 µs and moved 0 times in 80 samples.

    Publishing the mark here is what keeps the ontology layer in its cached mode across the suite.
    One `signal.deliver` measures 0.003 s with this method and 12.106 s without it: 77M function
    calls collapsing to a warm walk (`tree_path` 164,230 calls -> `canonical_parent` 1,494,184 ->
    `_keyed_ready` 1,379,580, each re-verifying a store that could not answer).

    Memoising the tree walk inside `geometry.canonical_parent` also makes it fast — 4.9x, measured —
    but by serving values from a store that cannot certify them, which is the claim `_gate` keeps a
    cache from making. Speed there costs the guarantee.

    A test that wants a store which publishes no mark uses its own double, and the coverage is
    intact: `test_freshness.py`'s `_NoMark()`, `test_basis_generations.py`'s monkeypatched
    `fr.write_mark`, and `test_pooling_screen.py`'s `node`.
    """

    def __init__(self):
        self.d = {}
        self._writes = 0
        # A per-instance origin, so two live doubles never compare equal by accident. The real
        # mark is "every origin's last_seq"; two different stores are two different origins.
        self._origin = "fake-%d" % id(self)

    # ── the whole-of-store write mark: moves on every write, and only on a write ──────────────
    # Same contract as `mantle.db.seq.write_mark` — a tuple that compares unequal after a
    # write and equal after any number of reads. `test_freshness.py` pins that contract against the
    # real store; this mirrors it so the double is faithful in the one dimension the caches read.
    #
    # It spans edges too, because the real one does. Contract §4 RESOLVED-5 gives one `_seq` counter
    # per observer over `vertex` and `edge`, and `crystal.ontology.driver._synset_from_doc` reads
    # the taxonomy from `edge WHERE src=?` for corpora that carry no `hypernyms` field. A mark
    # moving only on artifact writes would let a cached Synset outlive an edge write that changed
    # its parents, serving a taxonomy that is wrong rather than merely stale. `_FakeGraph` bumps
    # this store's counter for that reason.
    def write_mark(self):
        return ((self._origin, self._writes),)

    def note_write(self):
        """Move the mark. Called by `_FakeGraph` so an edge write counts as a store write."""
        self._writes += 1

    def put_artifact(self, doc):
        self.d[doc["id"]] = dict(doc)
        self._writes += 1
        return doc

    def get_artifact(self, artifact_id):
        a = self.d.get(artifact_id)
        return dict(a) if a else None

    def list_artifacts(self, *, content_type=None, state=None, collection_id=None,
                       created_by=None, limit=None, skip=0, include_archived=False):
        """Head-only by default, exactly as `mantle.db.vertex.list_artifacts` is.

        `include_archived` is not a kwarg to absorb: with it False and no explicit `state`, the
        real store drops rows whose `doc.state` is `archived`, so a fake that accepted the word
        and returned the archived rows anyway would be answering a question the store stopped
        answering — and every caller reading a superseded version as live would still pass.

        The two conditions are both load-bearing. An explicit `state` has already decided the set,
        so `include_archived` governs only the unfiltered read; and the test is `== "archived"`
        rather than a truthiness check, mirroring the store's null-safe `IS NOT` — a doc with no
        `state` at all is committed-by-absence and stays.
        """
        for a in self.d.values():
            if content_type and a.get("content_type") != content_type:
                continue
            if state and a.get("state") != state:
                continue
            if collection_id and a.get("collection_id") != collection_id:
                continue
            if state is None and not include_archived and a.get("state") == "archived":
                continue
            yield dict(a)

    def count(self, *, state=None):
        return sum(1 for a in self.d.values() if state is None or a.get("state") == state)

    def put_many(self, docs, *, batch=500):
        n = 0
        for d in docs:
            self.put_artifact(d)
            n += 1
        return n


class _FakeGraph:
    def __init__(self, marker=None):
        self.edges = []  # (from, to, label)
        # The `_FakeArtifacts` whose write mark this graph shares — see that class's `write_mark`
        # for why an edge write must move it. `None` keeps a standalone graph usable.
        self._marker = marker

        # A second reader of the same edges. `crystal.ontology.driver` reads them with raw SQL
        # through `store.db.read()` — `SELECT dst, label FROM edge WHERE src=? AND label IN (…)` —
        # rather than through `neighbors()`, so a double that speaks only the graph API cannot serve
        # the ontology. What a store must publish is listed in `crystal/ontology/driver.py`'s
        # header — `get_artifact`, `list_artifacts`, `write_mark`, `version_of`, `edge_mark` and
        # `.db.read()` — and `agience-crystal/tests/test_ontology_store_injection.py` exercises a
        # double that publishes all six.
        #
        # The table holds the same edges rather than a parallel copy: `add_edge` writes both. Two
        # stores of one fact would drift, and the drift would read as a driver bug.
        self._sql = _sqlite3.connect(":memory:")
        self._sql.execute("CREATE TABLE edge (src TEXT, dst TEXT, label TEXT, props TEXT)")
        self._sql.execute("CREATE INDEX ix_edge_src ON edge(src, label)")

    def add_edge(self, from_id, to_id, label, props=None):
        self.edges.append((from_id, to_id, label))
        self._sql.execute("INSERT INTO edge (src, dst, label, props) VALUES (?,?,?,?)",
                          (from_id, to_id, label,
                           _json.dumps(props) if props is not None else None))
        if self._marker is not None:
            self._marker.note_write()

    def add_edges(self, edges, *, batch=500):
        n = 0
        for e in edges:
            self.add_edge(e[0], e[1], e[2])
            n += 1
        return n

    def neighbors(self, node_id, label=None, *, direction="out"):
        out = []
        for f, t, l in self.edges:
            if label and l != label:
                continue
            if direction == "out" and f == node_id:
                out.append(t)
            elif direction == "in" and t == node_id:
                out.append(f)
            elif direction == "both" and node_id in (f, t):
                out.append(t if f == node_id else f)
        return list(dict.fromkeys(out))


class _FakeDb:
    """The raw-SQL face of the store. `crystal.ontology.driver._conn` is `store.db.read()`, so a
    store that only offers the graph API cannot serve the ontology at all."""

    def __init__(self, graph):
        self._graph = graph

    def read(self):
        return self._graph._sql

    def write(self):
        return self._graph._sql


class _FakeStore:
    def __init__(self):
        self.artifacts = _FakeArtifacts()
        self.graph = _FakeGraph(marker=self.artifacts)   # one mark over artifacts AND edges
        self.db = _FakeDb(self.graph)                    # the SAME edges, read as SQL
        # And on `artifacts` too. `crystal.ontology.driver._conn` is `_observe(store).db.read()`,
        # and `_observe` returns the artifacts accessor — the real lattice exposes
        # `store.artifacts.db` (see `seed_lattice._relation_signature`), so the ontology path reads
        # `db` from there rather than from the store.
        self.artifacts.db = self.db
        self.content = None       # resolve_text falls back to inline `content`
        self.keys_dir = None


# ── the offline WordNet index (real geometry, no live store) ──────────────────────────────────
def _index_is_usable(wn) -> bool:
    """Whether `wn._INDEX` holds a real WordNet, which is a stronger question than whether it exists.

    A `select` against an empty lattice store leaves `_INDEX` built with 0 synsets — present, and
    useless. A fixture short-circuiting on `is not None` would then hand every geometric test an
    empty index and they all read `basis == "unavailable"`, failing for a reason unrelated to what
    they assert. So the size is what is checked, not the presence."""
    idx = getattr(wn, "_INDEX", None)
    if not idx or not idx[0]:
        return False
    return len(idx[0]) > 1000        # a real WordNet is ~117k synsets; 0 or a handful is a stub


def _wn_cache_key(nwn) -> str:
    """What the extracted table derives from, as one string: the identity of its data files (name,
    size and mtime for both WordNet and ic-brown) plus the source text of the extractor. Either one
    moving is a different table, so a stale entry is unreachable by construction and nobody has to
    remember to bump a version.

    Returns None when the corpus cannot be identified, which disables the cache rather than falling
    back to a weaker key. Same rule the store caches run under: a source that cannot support the
    claim "this is still current" does not get the claim made on its behalf
    ([[absence-is-not-an-affirmative-claim]]). The cost of building from source is 17 s, once.

    The synset count and `get_version()` are the obvious keys and both force a full corpus load:
    `sum(1 for _ in all_synsets())` costs 6.12 s and `nwn.get_version()` 6.75 s, paid on every read
    including the hits — most of the cost this cache exists to avoid. Stat-ing the data files
    answers the same question, has the corpus changed, in under 1 ms, because it never opens them.
    """
    import hashlib
    import inspect
    import os

    import nltk.data

    parts, found = [], False
    # WordNet ships zipped or unpacked depending on how it was fetched; ic-brown resolves either
    # way. Whichever are present identify the corpus — `corpora/wordnet` MISSES on a zipped install,
    # which is why this probes rather than assuming (measured: LookupError in 2.5 ms).
    for res in ("corpora/wordnet.zip", "corpora/wordnet",
                "corpora/wordnet_ic/ic-brown.dat"):
        try:
            p = str(nltk.data.find(res))
            names = ([os.path.join(p, f) for f in sorted(os.listdir(p))]
                     if os.path.isdir(p) else [p])
            for f in names:
                st = os.stat(f)
                parts.append("%s:%d:%d" % (os.path.basename(f), st.st_size, int(st.st_mtime)))
            found = True
        except Exception:
            continue
    if not found:
        return None
    try:
        parts.append(inspect.getsource(_install_offline_wordnet))
    except Exception:
        return None                       # cannot pin the extractor ⇒ cannot claim the table is its
    h = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return "agience-wn-offline-index-%s.pkl" % h


def _wn_cache_read(nwn):
    """`(raw, word, missing)` from the on-disk cache, or None. Any fault reads as a miss, so a cache
    that cannot be read sends the caller to the real source rather than to a partial table."""
    import os
    import pickle
    import tempfile

    key = _wn_cache_key(nwn)
    if key is None:
        return None                       # corpus unidentifiable ⇒ no cache, build from source
    path = os.path.join(tempfile.gettempdir(), key)
    try:
        with open(path, "rb") as fh:
            raw, word, missing = pickle.load(fh)
    except Exception:
        return None
    # A truncated or empty table is a miss, not a table — the same check `_index_is_usable` makes.
    if not isinstance(raw, dict) or len(raw) <= 1000:
        return None
    return raw, word, missing


def _wn_cache_write(nwn, raw, word, missing) -> None:
    """Best-effort. A write that fails costs the next process 17 s and nothing else, so every fault
    is swallowed rather than raised into a test. Written to a temp name and renamed, so a concurrent
    `-n 8` run sees either the old file or the new one."""
    import os
    import pickle
    import tempfile

    key = _wn_cache_key(nwn)
    if key is None:
        return
    path = os.path.join(tempfile.gettempdir(), key)
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump((raw, word, missing), fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)          # atomic: readers see the old file or the new one
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
    except Exception:
        pass


def _install_offline_wordnet() -> bool:
    """Populate `crystal.ontology.driver`'s index from the local nltk WordNet and ic-brown — the
    same source `scripts/enrich_wordnet.py` writes into the store, so the values are identical. That
    lets these tests exercise the real geometry without a live store. The production runtime reads
    ontology structure from the corpus and never loads nltk."""
    from crystal.ontology import driver as wn

    # This fixture establishes its own state rather than inheriting the previous test's.
    #
    # The index is one module-level global; the bound store is another. A test that read through a
    # `_FakeStore` leaves it bound as `wn._SOURCE`, and a bound store outranks the ambient one for
    # every later read — the spec, the sense-frequency prior and the per-synset cache all resolve
    # against it while the big nltk index sits there looking usable. Because the state is
    # module-scoped it is shared by every test in the process, so which tests ran first decides what
    # a later one measures.
    #
    # `bind(None)` releases the store and drops the caches built from it. With nothing foreign bound
    # it costs nothing, so the common path is unchanged.
    if getattr(wn, "_SOURCE", None) is not None:
        wn.bind(None)

    if _index_is_usable(wn):
        return True
    try:
        import math
        from nltk.corpus import wordnet as nwn, wordnet_ic
        from nltk.corpus.reader.wordnet import information_content
    except Exception:
        return False
    wn._INDEX = None                     # drop any stub index before rebuilding

    # ── the extraction is cached across processes, because it is a pure derivation ──────────────
    # This loop costs 17.36 s, once per process, and the warm re-check is free. Serial that is ~17%
    # of the ember suite; under `-n 8` every worker pays it, which is what caps what parallelism can
    # buy here.
    #
    # It is cacheable because it derives from a fixed external corpus and nothing else: the same
    # nltk WordNet and the same `ic-brown.dat` give the same table every time. So the cache is keyed
    # on exactly what it derives from — the data files' identity plus the source text of this
    # function. A change to the extraction logic (the 1e300 sentinel handling, the IC ceiling, the
    # lemma-count shape) changes the key by construction, the same discipline
    # `crystal.ontology.geometry._DENSE_CACHE` runs under.
    #
    # Only primitives are stored, never `wn.Synset` objects. Pickling live objects from the module
    # under test would let a stale class definition load back in wearing the right name. The Synsets
    # are rebuilt from the primitives on load, which is dict construction and costs a fraction of
    # the traversal.
    _cached = _wn_cache_read(nwn)
    if _cached is not None:
        _raw, word, missing = _cached
        idx = {n: wn.Synset(n, pos, hyp, inst, ic, counts)
               for n, (pos, hyp, inst, ic, counts) in _raw.items()}
        wn.install_index(idx, word, ic_stats={"synsets": len(idx),
                                              "with_ic": len(idx) - missing,
                                              "without_ic": missing})
        return True

    icd = wordnet_ic.ic("ic-brown.dat")
    idx, word, missing, raw = {}, {}, 0, {}
    for s in nwn.all_synsets():
        try:
            v = information_content(s, icd)
            # nltk's zero-frequency sentinel is 1e300, which is finite, so `math.isfinite` passes it
            # through to the coordinate where `geometry.ic_of` raises AbsurdIC (real Resnik IC on
            # ic-brown.dat tops out at 14.709437882542113). Per that error and
            # `scripts/enrich_wordnet.py`, a zero-frequency synset leaves `ic` genuinely absent
            # rather than carrying an impossible number.
            v = float(v) if math.isfinite(v) else None
            if v is not None and v > 14.709437882542113:
                v = None
        except Exception:
            v = None
        counts = {l.name(): int(l.count()) for l in s.lemmas()}
        hyp = [h.name() for h in s.hypernyms()]
        inst = [h.name() for h in s.instance_hypernyms()]
        raw[s.name()] = (s.pos(), hyp, inst, v, counts)
        idx[s.name()] = wn.Synset(s.name(), s.pos(), hyp, inst, v, counts)
        if v is None:
            missing += 1
        for lm in counts:
            word.setdefault((lm.lower(), s.pos()), []).append(s.name())
    for names in word.values():
        names.sort()
    _wn_cache_write(nwn, raw, word, missing)
    # Installed through `install_index` rather than by assigning the private. This index came from
    # nltk and not from any store, so the store's write mark says nothing about whether it is
    # current. `install_index` records that it is not store-derived, and the freshness gate then
    # leaves it alone; a bare `wn._INDEX = …` reads as an index the gate owns, and it drops it on
    # the first observation of the ambient store — an unbuilt index gives every pair infinite
    # distance, and rebuilding costs a 25.4 s reload of the live corpus. See
    # `crystal.ontology.driver.install_index`.
    wn.install_index(idx, word, ic_stats={"synsets": len(idx), "with_ic": len(idx) - missing,
                                          "without_ic": missing})
    return True


# ── a minimal answerer, for tests that exercise the RUNNER through `ask()` ─────────────────────
class _StubAnswerer:
    """The smallest thing satisfying what `read_path` calls: `.name` and `.answer(query, evidence)`.

    `boot` constructs no answerer, because composing prose from evidence is a persona act:
    `ember.runtime.engine.build()` raises `NoAnswerer` rather than returning a default. The cache,
    relay and versioning tests exercise the runner but assert through `ask()`, so they inject this
    one. What they measure is routing, refill, offline behaviour and version lineage.

    It returns the evidence it was handed, unaltered and uncomposed. With nothing to compose from, a
    test that passes with it is a statement about the path rather than about the prose.
    """

    name = "stub"

    def answer(self, query, evidence):
        # `Answer` is the operator bundle's shape rather than ember's: the runner declares neither
        # the evidence shape nor the type of an answer, so the class is resolved at call time.
        from ember.runtime.runner import answer as _answer_mod
        answer_cls = _answer_mod.Answer

        # Evidence arrives as rows — `read_path` hands over dicts, not objects. A dict has no
        # `.text`, and `getattr(e, "text", "")` reads empty for every row without saying so, so both
        # shapes are read explicitly: the runner sends dicts, and a caller may still pass shaped
        # objects.
        def _f(e, k):
            return (e.get(k) if isinstance(e, dict) else getattr(e, k, None)) or ""

        ev = list(evidence or [])
        text = "\n".join(str(_f(e, "text")) for e in ev).strip()
        return answer_cls(text=text, grounded=bool(ev),
                          cited=[str(_f(e, "id")) for e in ev])
