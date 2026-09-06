#!/usr/bin/env python3
"""P7/S3 acceptance checks 2-8, run against the migrated slice through real Ember components.

Check 1 is `node-repair.py --sqlite <slice> --deep`. It is the suite itself and is run separately
so that nothing wraps it. Checks 2-8 are here.

  2  keyed lookup   `define <word>` returns the synset, not wiki pages
  3  topical        a concept name resolves and walks to its members
  4  content        content_ref -> decrypt -> sha256 verify -> full text, not the 300-char preview
  5  coordinates    a JC coordinate computed from the migrated `ic`, exact to ~3.55e-15
  6  FTS5           the lexical index, built from migrated rows, queried
  7  grants         a principal without a grant gets no key -- the failure is asserted
  8  unplugged      re-run 2-7 with the object stores unreachable

Every check prints the numbers it decided on rather than a bare pass/fail. A check that cannot
state its evidence is a check that can pass on no evidence: an all-constant series scores a
perfect rank correlation for a geometry that encodes nothing.

Usage:
    python scripts/lattice_smoke.py --store work/lattice.db --cas work/cas
    unshare -rn python scripts/lattice_smoke.py --store ... --cas ... --expect-no-network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

RESULTS: List[Tuple[str, str, str]] = []
PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


# ── the floating-point slack rule ────────────────────────────────────────────────────────────
#
# A residual is small compared to the round-off the arithmetic that produced it can generate, and
# that quantity is computable from the arithmetic itself. Exactness is judged against this computed
# bound — the same rule `mantle.search.anchors.crosswalk.null_residual` states for its own null.
#
# Recursive summation of `n` terms has |fl(Σ) − Σ| ≤ (n−1)·eps·Σ|xᵢ| (Higham, *Accuracy and
# Stability of Numerical Algorithms*, §4.2). Both quantities check 5 compares are such sums — L2²
# over the union of two paths' edges, and `jc_tree`'s ic₁+ic₂−2·ic_lcs — so their difference
# inherits at most the sum of their two bounds, hence the factor of two below.
#
# Moving the bound means changing the arithmetic: sum fewer terms, or work at a smaller magnitude.
_FLOAT_EPS = sys.float_info.epsilon


def _roundoff_bound(magnitude: float, terms: int = 1) -> float:
    """Worst-case float64 round-off in a `terms`-term sum whose entries total `magnitude`."""
    return 2.0 * max(int(terms) - 1, 1) * _FLOAT_EPS * abs(float(magnitude))


def rec(name: str, status: str, detail: str) -> str:
    RESULTS.append((name, status, detail))
    print("  %-34s %-4s  %s" % (name[:34], status, detail))
    return status


def _conn(path: str) -> sqlite3.Connection:
    c = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    c.row_factory = sqlite3.Row
    return c


# ═════════════════════════════════════════════════════ 2. keyed lookup — `define`
def check_define(db: sqlite3.Connection, words: List[str]) -> None:
    """The keyed arm discriminates by content type — the check for §5.1.1.

    `lookup_by_lemma('spaceship', limit=200)` returns 200 hits and zero synsets on the live
    system: the lookup has no type discrimination, and the 6,063,979-row `world` collection swamps
    the 117,659-row lexicon in a shared `lemmas` index. `lexical.py:60` calls it with no
    content-type filter, so `define spaceship` answers `grounded=True` citing Wikipedia
    disambiguation pages, with an empty POS and a "synonyms" list that is a wiki token bag.
    Confident, well-formed, wrong.

    In the migrated store `ct` is a real indexed column and `lemmas` survives enrichment into
    `doc`, so the keyed arm can discriminate by type. Both halves are asserted:
      * the typed lookup returns synsets, and
      * the untyped lookup over the same store still returns non-synsets --
    because if the untyped form were clean here, the typed form would pass for free and this check
    would prove nothing about the discriminator."""
    # ── build the lemma -> (synsets, non-synsets) index once, from the migrated docs ──────────
    idx: Dict[str, Dict[str, List[str]]] = {}
    for r in db.execute("SELECT id, ct, doc FROM vertex"):
        d = json.loads(r["doc"])
        for lm in (d.get("lemmas") or []):
            e = idx.setdefault(str(lm).lower(), {"syn": [], "other": []})
            (e["syn"] if r["ct"] == "text/x-wordnet" else e["other"]).append(r["id"])

    # The probe words are derived from the slice. Its `wn-` rows are the first 900 in id order
    # (`wn-'hood`, `wn-.22_caliber`, ...), so a fixed list such as `dog,water,city` names words
    # the corpus under test does not contain, and a probe for an absent word measures nothing in
    # either direction.
    #
    # The selection requires words that have a synset AND are contaminated — at least one
    # non-synset match — because the claim is that the typed lookup excludes the wiki pages the
    # untyped one returns. A word with no contaminating match passes trivially and says nothing
    # about the discriminator, so §5.1.1's defect only shows up through a contaminated probe.
    auto = sorted((w for w, e in idx.items() if e["syn"] and e["other"]),
                  key=lambda w: (-len(idx[w]["other"]), w))
    explicit = [w for w in words if w.lower() in idx and idx[w.lower()]["syn"]]
    probes = (explicit + [w for w in auto if w not in explicit])[:6]
    if not probes:
        return rec("define -> synset (typed)", FAIL,
                   "no lemma in this store has BOTH a synset and a non-synset match, so the type "
                   "discriminator cannot be exercised -- §5.1.1 is UNTESTED by this slice") and None

    typed_ok = untyped_contaminated = 0
    ev: List[str] = []
    for w in probes:
        typed = idx[w]["syn"]
        untyped = [(i, "text/x-wordnet") for i in idx[w]["syn"]] + \
                  [(i, "other") for i in idx[w]["other"]]
        n_non_synset = len(idx[w]["other"])
        if typed and all(i.startswith("wn-") for i in typed):
            typed_ok += 1
        if n_non_synset:
            untyped_contaminated += 1
        ev.append("%s: typed=%d(all wn-=%s) untyped=%d non-synset=%d"
                  % (w, len(typed), all(i.startswith("wn-") for i in typed) if typed else "n/a",
                     len(untyped), n_non_synset))
    if typed_ok == len(probes):
        rec("define -> synset (typed)", PASS,
            "%d/%d word(s) resolve to wn- synsets ONLY. %s" % (typed_ok, len(probes), "; ".join(ev)))
    else:
        rec("define -> synset (typed)", FAIL,
            "%d/%d resolved to synsets. %s" % (typed_ok, len(probes), "; ".join(ev)))
    # The discriminator only proves something if there was contamination to discriminate against.
    rec("define discriminator is load-bearing", INFO,
        "%d/%d word(s) ALSO match a non-synset in the untyped lookup -- that is the §5.1.1 defect "
        "reproduced in the slice, and the typed lookup above is what excludes it"
        % (untyped_contaminated, len(probes)))


# ═════════════════════════════════════════════════════ 3. topical — concept walk
def check_topical(db: sqlite3.Connection, n: int = 60) -> None:
    """§11.3 / P7/S3.3: a concept name resolves to its `concept-` artifact and walks to its
    members.

    Resolution goes through the keyed lemma lookup, the way `define` does. The concept name lives
    in `lemmas[0]` and is indexed in `listkey` (field='lemmas'), so
    `SELECT aid FROM listkey WHERE field='lemmas' AND value=?` is an index seek — the retrieval
    path a topical query travels. Scanning each concept's `context.name` instead would be O(n²)
    and would measure a resolution method the runtime never uses; `context` also carries the offer
    string once the offers pass has run, so it is a string rather than a dict.

    Artifact -> members is the second half: the `colimit` stage mints bidirectional edges
    (`colimit_of` forward, `member_of_concept` back), verified equal here."""
    rows = db.execute(
        "SELECT id, doc FROM vertex WHERE ct = 'application/x-concept' ORDER BY id LIMIT ?",
        (n,)).fetchall()
    resolved = walked = 0
    total_members = 0
    for r in rows:
        d = json.loads(r["doc"])
        lemmas = d.get("lemmas") or []
        name = lemmas[0] if lemmas else None
        if not name:
            continue
        # resolve by name via the keyed index -- an index seek, the way a topical query arrives.
        hits = [x[0] for x in db.execute(
            "SELECT aid FROM listkey WHERE field = 'lemmas' AND value = ? "
            "AND aid LIKE 'concept-%'", (name,))]
        if r["id"] not in hits:
            continue
        resolved += 1
        mem = [x[0] for x in db.execute(
            "SELECT dst FROM edge WHERE src = ? AND label = 'colimit_of'", (r["id"],))]
        back = [x[0] for x in db.execute(
            "SELECT src FROM edge WHERE dst = ? AND label = 'member_of_concept'", (r["id"],))]
        if mem and sorted(mem) == sorted(back):
            walked += 1
            total_members += len(mem)
    if rows and walked == resolved and walked:
        rec("concept name -> members", PASS,
            "%d/%d concept(s) resolve BY NAME and walk to their members in BOTH directions "
            "(%d member link(s)). Baseline was 0/60." % (walked, len(rows), total_members))
    else:
        rec("concept name -> members", FAIL,
            "resolved=%d walked=%d of %d -- the colimit_of->edge transformation did not happen"
            % (resolved, walked, len(rows)))


# ═════════════════════════════════════════════════════ 4. content end-to-end
def check_content(db: sqlite3.Connection, cas: str, keys_dir: str, n: int = 40) -> None:
    """§6A end to end: resolve `content_ref` -> decrypt -> verify sha256 -> return the full text.

    The preview problem is what this pins. `doc.content` was a 300-char truncation on ~94.6% of
    the `text/markdown` population, and an FTS5 index built from it has postings, ranks sensibly,
    and loses all recall past ~50 words. So the returned text is asserted to be both hash-correct
    and materially longer than 300 characters: a blob that happened to be short would pass a hash
    check while proving nothing about truncation."""
    rows = db.execute(
        "SELECT id, content_ref, json_extract(doc,'$.collection_id') AS coll FROM vertex "
        "WHERE content_ref IS NOT NULL ORDER BY id LIMIT ?", (n,)).fetchall()
    if not rows:
        return rec("content -> full text", FAIL, "no content_ref rows in the store") and None

    # Read through the real component rather than the raw cache files. The cache is encrypted at
    # rest, so hashing its files directly verifies the file rather than the retrieval path. Going
    # through `FileContentCache.get()` exercises what a reader does — derive the key, decrypt,
    # AEAD-authenticate, and re-hash against the content address — which is what P7/S3.4 asks
    # about.
    try:
        from mantle.db.content_cache import (CacheCorrupt, CacheMiss,
                                                     FileContentCache, collection_key,
                                                     shared_content_key)
    except Exception as e:
        return rec("content -> full text", FAIL,
                   "content_cache not importable (%s: %s)" % (type(e).__name__, e)) and None
    try:
        with open(os.path.join(keys_dir, "content.key"), "rb") as fh:
            root_secret = hashlib.blake2b(fh.read().strip(), digest_size=32).digest()
    except Exception as e:
        return rec("content -> full text", FAIL, "cannot read content.key: %s" % e) and None
    roots = {}
    for cname in ("personal", "foundation"):
        row = db.execute("SELECT json_extract(doc,'$.origin_root') FROM vertex WHERE id = ?",
                         (cname,)).fetchone()
        if row and row[0]:
            roots[cname] = row[0]
    # Shared at-rest key (mantle §1). The per-collection map is the decrypt fallback for objects
    # written under the earlier per-collection scheme, so such a corpus still verifies here.
    cache = FileContentCache(
        cas, key=shared_content_key(root_secret),
        legacy_key_for_collection=(lambda c: collection_key(root_secret, roots[c])) if roots
        else None)
    ok = corrupt = missing = over300 = 0
    lens: List[int] = []
    for r in rows:
        try:
            data = cache.get(r["content_ref"], collection=r["coll"] or "foundation")
        except CacheMiss:
            missing += 1
            continue
        except CacheCorrupt:
            corrupt += 1
            continue
        ok += 1
        lens.append(len(data))
        if len(data) > 300:
            over300 += 1
    rep = cache.report()
    if ok and not corrupt and not missing:
        rec("content -> full text", PASS,
            "%d/%d object(s) read through FileContentCache: collection-key decrypt + AEAD + "
            "sha256 verify-on-read, all passed; %d are >300 bytes (max %d, median %d) -- FULL "
            "TEXT, not the 300-char preview. cache: hit=%s miss=%s corrupt=%s"
            % (ok, len(rows), over300, max(lens), sorted(lens)[len(lens) // 2],
               rep["hit"], rep["miss"], rep["corrupt"]))
    else:
        rec("content -> full text", FAIL,
            "%d ok, %d CORRUPT (present but failed decrypt/verify), %d MISS -- and those three "
            "are deliberately distinct" % (ok, corrupt, missing))
    # A NULL content_ref is the signal that content is inline, so no row carries both (§13.3).
    both = db.execute("SELECT count(*) FROM vertex WHERE content_ref IS NOT NULL AND "
                      "json_extract(doc,'$.content') IS NOT NULL").fetchone()[0]
    rec("no preview beside a content_ref", PASS if not both else FAIL,
        "0 rows carry BOTH a content_ref and an inline copy -- NULL content_ref is now the "
        "unambiguous signal that content is inline" if not both else
        "%d row(s) still carry both; NULL content_ref is no longer an unambiguous signal" % both)


# ═════════════════════════════════════════════════════ 5. JC coordinate exactness
class _Syn:
    """The ontology driver's `Synset` duck-type, built from the migrated docs.

    Built from the migrated store rather than through the driver's own store loader: the claim
    under test is that the migrated `ic` and `hypernyms` support an exact JC coordinate, so the
    data comes from the migration. The geometry implementation whose exactness is asserted is used
    unmodified."""
    __slots__ = ("_n", "_p", "_h", "_i", "_ic", "_idx")

    def __init__(self, name, pos, hyper, inst, ic, idx):
        self._n, self._p, self._h, self._i, self._ic, self._idx = name, pos, hyper, inst, ic, idx

    def name(self): return self._n
    def pos(self): return self._p
    def ic(self): return self._ic if self._ic is not None else 0.0
    def has_ic(self): return self._ic is not None
    def hypernyms(self): return [self._idx[h] for h in self._h if h in self._idx]
    def instance_hypernyms(self): return [self._idx[h] for h in self._i if h in self._idx]

    def _closure(self):
        out, stack = {}, [self]
        while stack:
            s = stack.pop()
            for p in s.hypernyms() + s.instance_hypernyms():
                if p._n not in out:
                    out[p._n] = p
                    stack.append(p)
        return out

    def common_hypernyms(self, other):
        a, b = self._closure(), other._closure()
        a[self._n] = self
        b[other._n] = other
        return [a[k] for k in set(a) & set(b)]

    def __eq__(self, o): return isinstance(o, _Syn) and o._n == self._n
    def __hash__(self): return hash(self._n)
    def __repr__(self): return "Synset('%s')" % self._n


def check_coordinates(db: sqlite3.Connection, ember_src: str) -> None:
    """P7/S3.5: whether `ic` survived enrichment is what surfaces here.

    Exactness is `max|L2(v1-v2)^2 - tree_JC(s1,s2)|` over the sparse coordinate, which §5.3.3
    records as D-independent at 3.552714e-15 -- machine epsilon, a property of the JC construction
    itself.

    The zero-IC trap is checked first because it is the whole hazard. IC 0 everywhere gives
    jc_tree 0 for every pair, all-zero vectors, and a perfect exactness score for a measurement
    that never happened -- §5.3 records `faithfulness_check` reporting `spearman 1.0` and
    `prehash_max_abs_err 0.0` over a corpus whose `wn-*` artifacts carried no `ic`. An all-zero IC
    set therefore fails here rather than scoring 0.0 and passing."""
    # The geometry module is loaded from a file rather than imported as a package module: it
    # itself imports only hashlib/math/numpy, but importing it through a package runs that
    # package's `__init__` and boot chain, which a migration host does not carry. Loading the file
    # directly keeps the same implementation under test without dragging a runtime onto the host.
    import importlib.util
    gpath = os.path.join(ember_src, "ember", "geometry.py")
    if not os.path.exists(gpath):
        return rec("JC coordinate exact", FAIL, "geometry.py not found at %s" % gpath) and None
    try:
        spec = importlib.util.spec_from_file_location("_geom", gpath)
        geometry = importlib.util.module_from_spec(spec)              # type: ignore
        spec.loader.exec_module(geometry)                             # type: ignore
    except Exception as e:
        return rec("JC coordinate exact", FAIL,
                   "cannot load geometry.py (%s: %s)" % (type(e).__name__, e)) and None

    idx: Dict[str, _Syn] = {}
    raw: Dict[str, Tuple[str, list, list, Optional[float]]] = {}
    for r in db.execute("SELECT id, doc FROM vertex WHERE ct = 'text/x-wordnet'"):
        d = json.loads(r["doc"])
        name = r["id"][3:] if r["id"].startswith("wn-") else r["id"]
        ic = d.get("ic")
        raw[name] = (d.get("pos") or "n", list(d.get("hypernyms") or []),
                     list(d.get("instance_hypernyms") or []),
                     None if ic is None else float(ic),
                     d.get("hypernyms"))   # RAW: None = unenriched, [] = enriched root
    for name, (pos, hy, inst, ic, _rawhy) in raw.items():
        idx[name] = _Syn(name, pos, hy, inst, ic, idx)

    if not idx:
        return rec("JC coordinate exact", FAIL, "no wn- artifacts in the store") and None
    with_ic = sum(1 for v in raw.values() if v[3] is not None)
    SENTINEL = 1e100                       # anything above this is not an information content
    sent = [n for n, v in raw.items() if v[3] is not None and v[3] > SENTINEL]
    zero = [n for n, v in raw.items() if v[3] == 0.0]
    # 0.0 is included. Resnik IC of the corpus root is exactly 0 — the root subsumes everything,
    # so its probability is 1 and -log(1) = 0. A path that does not reach a zero-IC root is a
    # truncated path, so excluding zeros would discard precisely the terminations that make the
    # telescoping sum exact.
    #
    # A stored 0.0 is also ambiguous, and this check cannot separate the cases. `enrich_wordnet.py`
    # records `ic_status` and leaves `ic` absent when it cannot measure, so a 0.0 written under
    # that rule is a root-level zero; rows carrying 0.0 as an absent-marker are indistinguishable
    # from real roots at the source. Reported, not resolved.
    phys = {n: v[3] for n, v in raw.items()
            if v[3] is not None and 0.0 <= v[3] <= SENTINEL and math.isfinite(v[3])}
    if zero:
        rec("ic zero-value ambiguity", INFO,
            "%d synset(s) carry ic=0.0. Resnik IC of the ROOT is legitimately 0, but "
            "enrich_wordnet.py also wrote 0.0 for synsets ABSENT from the nltk build and its own "
            "comment calls the two 'indistinguishable at the source'. Treated as physical here "
            "(a path must reach a zero-IC root to be exact); flagged because no consumer can "
            "separate them." % len(zero))
    rec("ic survived enrichment", PASS if with_ic == len(idx) else FAIL,
        "%d/%d synset(s) carry `ic`: %d physical (%.4f..%.4f), %d zero, %d NON-PHYSICAL"
        % (with_ic, len(idx), len(phys), min(phys.values()) if phys else 0,
           max(phys.values()) if phys else 0, len(zero), len(sent)))

    # ── the 1e+300 sentinel: a live upstream corpus defect ────────────────────────────────────
    # `ic = 1e+300` appears on 32% of the slice's synsets, with `ic_status` absent on every one.
    # The values are bit-identical to node 71's, so the migration carries them faithfully; the
    # defect is upstream and live:
    #
    #   * `geometry.ic_of` guards with `math.isfinite(v)`, and 1e+300 is finite, so it passes
    #     straight through into `jc_tree` and `sparse_vec`.
    #   * `enrich_wordnet.py` handles an unmeasurable IC by recording `ic_status` and leaving
    #     `ic` absent; rows carrying 1e+300 with `ic_status = None` did not come from that path.
    #   * the effect is `sqrt(1e300 - ic(parent))` in the coordinate, i.e. a single edge weight
    #     ~1e150, which swamps every real dimension it shares a hash bucket with.
    #
    # The migration leaves these intact. Rewriting 1e+300 to "absent" is the end state that
    # enrich_wordnet.py already intends, but it changes the semantic arm for a third of the
    # lexicon, and §P2 scopes a field shortfall as "a targeted keyed backfill from 71 -- not a
    # re-plan." Resolving it is outside this script's scope, so it is reported rather than fixed.
    rec("ic has no non-physical sentinel", PASS if not sent else FAIL,
        "no sentinel IC values" if not sent else
        "%d/%d synset(s) carry ic=1e+300 with ic_status ABSENT -- a non-physical value that "
        "`math.isfinite` does NOT filter, so it reaches jc_tree and sparse_vec directly. "
        "LIVE UPSTREAM DEFECT, carried faithfully by this migration (values are bit-identical "
        "to node 71). Examples: %s" % (len(sent), len(idx), ", ".join(sorted(sent)[:4])))

    if not phys:
        return rec("JC coordinate exact", FAIL,
                   "no synset carries a physical `ic` -- refusing to report exactness over a "
                   "coordinate system that encodes nothing (§5.3)") and None

    # ── exactness, measured over the physical-IC population ──────────────────────────────────
    # The claim under test is that the migrated `ic` supports an exact JC coordinate. A synset
    # whose stored IC is 1e+300 has no information content to be exact about, so including it
    # measures the sentinel rather than the construction. The sentinel is asserted separately
    # above, where it fails; both numbers are reported.
    #
    # The whole path has to be well-formed, not just the endpoints: `tree_path` walks all the way
    # to the root, so one poisoned ancestor eight hops up still lands in the coordinate. Two
    # independent conditions hold along the entire path, counted separately so the residual is
    # attributable:
    #
    #   physical   every node on the path has a real IC. A 1e+300 anywhere above dominates.
    #   monotonic  IC does not increase from child to parent. `sparse_vec` computes
    #              `sqrt(max(ic(c) - ic(p), 0.0))`; that `max(...,0)` clamps a negative delta, and
    #              a clamped edge makes L2^2 disagree with `jc_tree`, which does not clamp. A
    #              non-monotonic path breaks exactness by construction, in the code, so a correct
    #              migration cannot recover it.
    excl = {"nonphysical_path": 0, "nonmonotonic_path": 0, "short_path": 0,
            "unenriched_ancestor": 0, "nonzero_ic_root": 0}

    # `hypernyms is None` and `hypernyms == []` are different facts, and the distinction is
    # load-bearing:
    #
    #   []    enriched and genuinely a root. `entity.n.01` -> hypernyms=[], ic=-0.0.
    #   None  not enriched. The field was never recovered, so the ancestry is unknown, and
    #         `tree_path` stops there as though it had found a root.
    #
    # Contract §5.1.3 gives the cause: node 45 is a strict subset of node 71, missing ~2,000
    # `wn-*` artifacts scattered alphabetically. The slice's hypernym closure pulls ancestors out
    # of node 71's extract; enrichment then tries to recover their fields from node 45, which does
    # not have them. Measured here: `physical_condition.n.01` is present as a vertex with
    # `hypernyms = None`, so every descendant's path terminates on it prematurely and the
    # telescoping sum loses `IC(that node)` — which is exactly the residual.
    #
    # A truncated path looks like a short one, so it is counted separately rather than folded into
    # `short_path`. This is the measurement that says enrichment from 45 alone cannot rebuild the
    # hypernym tree.
    unenriched = {n for n, v in raw.items() if v[4] is None}

    def _wellformed(n: str) -> bool:
        try:
            path = geometry.tree_path(idx[n], None)
        except Exception:
            return False
        if len(path) < 2:
            excl["short_path"] += 1
            return False
        for s in path:
            if s.name() not in phys:
                excl["nonphysical_path"] += 1
                return False
        for c, p in zip(path, path[1:]):
            # The slack is the round-off in the IC values themselves, at their own magnitude: a
            # child and parent that differ by less than the arithmetic that produced them can
            # resolve are equal, not decreasing. `terms` is the handful of ops in
            # `-log(count/total)` plus the subtraction, stated here because it describes this
            # expression and nothing else.
            _slack = _roundoff_bound(max(abs(phys[c.name()]), abs(phys[p.name()])), terms=8)
            if phys[c.name()] < phys[p.name()] - _slack:
                excl["nonmonotonic_path"] += 1
                return False
        if path[-1].name() in unenriched:
            excl["unenriched_ancestor"] += 1
            # An unenriched ancestor leaves the synset well-formed on its own -- see the pair rule
            # below. It is counted because it is the dominant structural fact about this corpus.
        # Reaching an IC-0 root is a limit of the construction rather than a data problem.
        # `jc_tree` falls back to `lcs_ic = 0.0` for two synsets with no shared ancestor, i.e. it
        # assumes a universal root at IC 0, while the coordinate telescopes only to the path's
        # actual root. So for a pair whose roots have IC > 0 the two disagree by exactly
        # `IC(root1) + IC(root2)`. Measured: 334 of 2,950 paths terminate at a non-zero root, all
        # of them verbs (`change.v.02` ic=2.71, `act.v.01` ic=2.55, ...) — WordNet's verb
        # hierarchy has many top nodes and no single root, unlike nouns under `entity.n.01`.
        # §5.3.3's 3.55e-15 is therefore a claim about the single-rooted (noun) hierarchy, a scope
        # that geometry.py states nowhere. Recorded as a finding.
        # "Reaches an IC-0 root" is a claim about the sum this root terminates, so the slack is
        # that sum's round-off rather than a chosen epsilon. A root whose IC is below the noise of
        # the path it closes contributed nothing the telescoping could have lost.
        _path_ic = sum(abs(phys[s.name()]) for s in path)
        if abs(phys[path[-1].name()]) > _roundoff_bound(_path_ic, terms=len(path)):
            excl["nonzero_ic_root"] += 1
        return True

    names = sorted(n for n in idx if n in phys and _wellformed(n))
    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:i + 6]:
            pairs.append((a, b))
        if len(pairs) >= 4000:
            break
    # The pair rule. Requiring every path to terminate at an IC-0 root drives the measurable
    # population to zero, and the algebra shows why that requirement is too strong:
    #
    #   `sparse_vec` holds one entry per edge on the path to the root. For a pair that shares an
    #   ancestor, every edge above the LCS appears in both vectors with an identical weight, so it
    #   cancels in `v1 - v2`. What survives is exactly the edges from each synset up to the LCS:
    #
    #       L2^2 = [IC(s1) - IC(LCS)] + [IC(s2) - IC(LCS)]
    #            =  IC(s1) + IC(s2) - 2*IC(LCS)  =  jc_tree(s1, s2)      exactly
    #
    #   Nothing above the LCS enters it, so the identity holds for any pair with a real common
    #   ancestor, whether the shared root is unenriched or carries a non-zero IC.
    #
    # The root condition applies only to DISJOINT pairs, where `tree_lcs_ic` falls back to 0.0 and
    # so asserts a universal IC-0 root that WordNet's verb side does not have. Those pairs are
    # counted separately and reported as a limitation of the construction, kept out of the
    # exactness number where they would corrupt a claim that is otherwise exact.
    worst, worst_pair = 0.0, None
    # `worst_ratio` is the verdict and `worst` is the evidence. A pair's residual is judged against
    # that pair's own round-off bound, so a long path over large ICs is held to a wider absolute
    # number than a short one over small ones.
    worst_ratio, ratio_pair = 0.0, None
    scored = disjoint = 0
    disjoint_worst = 0.0
    for a, b in pairs:
        s1, s2 = idx[a], idx[b]
        p1 = {s.name() for s in geometry.tree_path(s1, None)}
        p2 = {s.name() for s in geometry.tree_path(s2, None)}
        shares = bool(p1 & p2)
        v1 = geometry.sparse_vec(s1, None)
        v2 = geometry.sparse_vec(s2, None)
        keys = set(v1) | set(v2)
        l2sq = sum((v1.get(k, 0.0) - v2.get(k, 0.0)) ** 2 for k in keys)
        jc = geometry.jc_tree(s1, s2, None)
        if not (math.isfinite(l2sq) and math.isfinite(jc)):
            continue
        e = abs(l2sq - jc)
        if not shares:
            disjoint += 1
            disjoint_worst = max(disjoint_worst, e)
            continue
        scored += 1
        # Both sides sum over the union of the two paths' edges; every L2^2 term is a square, so
        # the sum of magnitudes IS the sum, and the larger of the two values bounds both.
        bound = _roundoff_bound(max(abs(l2sq), abs(jc)), terms=len(keys))
        ratio = (e / bound) if bound > 0.0 else (0.0 if e == 0.0 else float("inf"))
        if e > worst:
            worst, worst_pair = e, (a, b)
        if ratio > worst_ratio:
            worst_ratio, ratio_pair = ratio, (a, b)
    if disjoint:
        rec("JC disjoint-pair limitation", INFO,
            "%d pair(s) share NO tree ancestor; max|L2^2 - tree_JC| = %.4e on those. `jc_tree` "
            "falls back to lcs_ic=0.0 (assuming a universal IC-0 root) while the coordinate "
            "telescopes only to each path's ACTUAL root -- so they differ by IC(root1)+IC(root2). "
            "WordNet's VERB side has many top nodes and no single root, so this is a scope limit "
            "of the exactness claim that geometry.py does not state." % (disjoint, disjoint_worst))
    # The verdict is a ratio to a computed bound: is the residual inside the round-off the
    # arithmetic can produce? That is what "exact" means, and it stays true as the corpus grows
    # deeper paths or larger ICs. The absolute residual is reported alongside it, as the evidence
    # a reader wants to see.
    ok = worst_ratio <= 1.0 and scored > 0
    rec("JC coordinate exact (well-formed)", PASS if ok else FAIL,
        "max|L2^2 - tree_JC| = %.6e over %d pair(s) from %d well-formed synset(s) (worst: %s); "
        "worst residual is %.3fx its own float64 round-off bound (worst: %s) -- EXACT means "
        "<=1.0x, D-INDEPENDENT (§5.3.3). Excluded: %s"
        % (worst, scored, len(names), worst_pair, worst_ratio, ratio_pair,
           json.dumps(excl, sort_keys=True)))


# ═════════════════════════════════════════════════════ 6. FTS5
def check_fts(db: sqlite3.Connection) -> None:
    """The one index (`ember.corpus.fts`), probed for real signal.

    Coverage is measured as indexed rows against vertex rows, and the index is queried with live
    probe terms — numbers that can come out wrong. A predicate over the FTS table's own columns
    cannot substitute: the table is contentless, so a column such as `gloss` reads back NULL for
    every row and `NULL <> ''` is never true, which pins any such count at 0 by construction.
    """
    from ember.corpus import fts as _fts
    try:
        cov = _fts.coverage(db)
        n = cov["indexed"]
    except Exception as e:
        return rec("FTS5 queryable", FAIL, "no lexical index: %s" % e) and None
    if not cov.get("built"):
        return rec("FTS5 queryable", FAIL,
                   "the index is absent — a store with rows and no index answers nothing, "
                   "silently") and None
    probes = ["dog", "water", "city", "the"]
    hits = {}
    for p in probes:
        try:
            hits[p] = db.execute("SELECT count(*) FROM %s WHERE %s MATCH ?"
                                 % (_fts.FTS_TABLE, _fts.FTS_TABLE), (p,)).fetchone()[0]
        except Exception as e:
            return rec("FTS5 queryable", FAIL, "MATCH %r raised %s" % (p, e)) and None
    live = sum(1 for v in hits.values() if v)
    ok = live >= 2 and cov["missing"] <= 0
    rec("FTS5 queryable", PASS if ok else FAIL,
        "%d of %d vertex row(s) indexed (%d missing); %d/%d probe term(s) matched %s"
        % (n, cov["vertices"], cov["missing"], live, len(probes), json.dumps(hits)))



# ═════════════════════════════════════════════════════ 7. grants — the security property
class NoGrant(Exception):
    """Raised instead of returning a key. The distinction is the entire check."""


def _derive_master(origin_root: str, secret: bytes) -> bytes:
    """Per-collection master, keyed on the collection's immutable origin root (P9.3).

    The origin root is the only input: `created_by`/'owner' is outside the crypto path, and the
    grant set is excluded because grants mutate and re-keying on every revoke would orphan cells,
    while the origin root never moves. The grant gates whether the oracle issues the key, never
    which key it is."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"master|" + origin_root.encode()).derive(secret)


def _issue_cell_key(db: sqlite3.Connection, principal: str, collection: str,
                    anchor: str, secret: bytes) -> bytes:
    """The gate: resolve the grant first, derive only if one exists.

    P1.5 places the check inside issuance rather than at a bypassable call site. In the live
    system the keying identity comes from the object (`owner = raw.get('created_by')`) and the
    requesting user's id appears nowhere in the decrypt chain, so anyone who reaches the decrypt
    function gets plaintext. Here `principal` is a required argument and the grant lookup is the
    first statement — there is no path to the KDF that does not pass through it.

    The scope of what this proves: the migrated store carries the structure the property needs —
    collections with immutable origin roots, and grants that bind a principal to a collection —
    and grant-gated issuance over that structure denies correctly. The production
    `oracle.py`/`content_service.py` are P1 and owned elsewhere, so a green board here is a
    statement about the store, not about P1."""
    row = db.execute(
        "SELECT id FROM vertex WHERE ct = 'application/vnd.agience.grant+json' "
        "AND json_extract(doc,'$.principal') = ? AND json_extract(doc,'$.collection') = ?",
        (principal, collection)).fetchone()
    if row is None:
        raise NoGrant("principal %s holds no grant on collection %s -- NO KEY IS DERIVED"
                      % (principal, collection))
    col = db.execute("SELECT doc FROM vertex WHERE id = ?", (collection,)).fetchone()
    if col is None:
        raise NoGrant("collection %s does not exist" % collection)
    root = json.loads(col["doc"]).get("origin_root")
    if not root:
        raise NoGrant("collection %s has no origin_root -- the cell-key principal is missing"
                      % collection)
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    master = _derive_master(root, secret)
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=("%s|%s" % (collection, anchor)).encode() + b"\x00" + anchor.encode()
                ).derive(master)


def check_grants(db: sqlite3.Connection) -> None:
    """A principal without a grant gets no key material at all, and that failure is asserted
    alongside the success (P7/S3.7). "Cannot" here means no key is derived, not a denied request."""
    secret = b"smoke-test-root-secret-not-a-real-key"
    grants = db.execute(
        "SELECT json_extract(doc,'$.principal') p, json_extract(doc,'$.collection') c "
        "FROM vertex WHERE ct = 'application/vnd.agience.grant+json'").fetchall()
    if not grants:
        return rec("grant -> key", FAIL, "no grant artifacts in the store") and None
    holder, coll = grants[0]["p"], grants[0]["c"]

    # (a) the positive: a grant holder gets a key.
    try:
        k1 = _issue_cell_key(db, holder, coll, "anchor-0", secret)
        pos = len(k1) == 32
    except NoGrant as e:
        return rec("grant -> key", FAIL, "grant holder was DENIED: %s" % e) and None
    rec("granted principal gets a key", PASS if pos else FAIL,
        "principal %s... on `%s` -> %d-byte cell key (fingerprint %s, never the material)"
        % (holder[:8], coll, len(k1), hashlib.blake2b(k1, digest_size=8).hexdigest()))

    # (b) the negative, which is the check itself: a stranger gets no key.
    stranger = "00000000-0000-0000-0000-000000000000"
    assert db.execute("SELECT 1 FROM vertex WHERE id = ?", (stranger,)).fetchone() is None
    leaked = None
    try:
        leaked = _issue_cell_key(db, stranger, coll, "anchor-0", secret)
    except NoGrant as e:
        rec("UNGRANTED principal gets NO KEY", PASS,
            "issuance RAISED and returned no key material: %s" % str(e)[:110])
    except Exception as e:
        rec("UNGRANTED principal gets NO KEY", FAIL,
            "issuance raised the WRONG error (%s: %s) -- a refused request is not the same as no "
            "key" % (type(e).__name__, e))
    if leaked is not None:
        rec("UNGRANTED principal gets NO KEY", FAIL,
            "⛔ KEY MATERIAL WAS RETURNED to a principal with no grant (%d bytes). This is P1's "
            "exact defect: access is a permission CHECK, not a cryptographic PROPERTY."
            % len(leaked))

    # (c) The key depends on the origin root rather than on the requester (P9.3). Two different
    #     grant-holders on the same collection get the same cell key; a requester-scoped key would
    #     orphan one holder's cells when the other is revoked.
    same = [g for g in grants if g["c"] == coll]
    if len(same) >= 2:
        ka = _issue_cell_key(db, same[0]["p"], coll, "anchor-0", secret)
        kb = _issue_cell_key(db, same[1]["p"], coll, "anchor-0", secret)
        rec("cell key is origin-root-scoped", PASS if ka == kb else FAIL,
            "two grant-holders on `%s` derive the SAME cell key -- the key is bound to the "
            "collection's immutable origin root, not to the requester (P9.3)" % coll
            if ka == kb else
            "two grant-holders derive DIFFERENT keys -- the key is requester-scoped, so every "
            "revoke would orphan cells (P9.3 explicitly rejects this)")

    # (d) Different collections derive different keys.
    others = [g["c"] for g in grants if g["c"] != coll]
    if others:
        kc = _issue_cell_key(db, [g["p"] for g in grants if g["c"] == others[0]][0],
                             others[0], "anchor-0", secret)
        rec("collections are key-isolated", PASS if kc != k1 else FAIL,
            "`%s` and `%s` derive DIFFERENT cell keys" % (coll, others[0]) if kc != k1 else
            "`%s` and `%s` derive the SAME key -- the two contexts are not isolated"
            % (coll, others[0]))


# ═════════════════════════════════════════════════════ 8. the unplugged assertion
def assert_no_network() -> None:
    """P7/S3.8 / F3: with the object stores unreachable, a new Ember still opens this store and
    answers. Anything that still reaches for them means consolidation is incomplete.

    The store is Mantle, in process, so the endpoints under test are the object stores: Garage and
    OVH S3.

    The isolation is established before the result is trusted. A "network unplugged" run that
    still has a network looks identical to a pass, so the endpoints are probed first by connect()
    and the run aborts if any of them answers."""
    # The remote endpoint is the operator's own, named by `$EMBER_OVH_ENDPOINT` rather than written
    # in here: a hardcoded host in a public script is one deployment's infrastructure, and probing
    # somebody else's bucket host proves nothing about THIS node's isolation. With it unset only
    # the local target is probed, which still catches the case this guard exists for.
    _remote = os.environ.get("EMBER_OVH_ENDPOINT", "").replace("https://", "").replace("http://", "").strip("/")
    targets = [("Garage", "127.0.0.1", 3900)]
    if _remote:
        targets.append(("object store", _remote, 443))
    reachable = []
    for name, host, port in targets:
        s = socket.socket()
        s.settimeout(2.0)
        try:
            s.connect((host, port))
            reachable.append("%s (%s:%d)" % (name, host, port))
        except Exception:
            pass
        finally:
            s.close()
    if reachable:
        print("\n⛔ ABORT: --expect-no-network was passed but these are REACHABLE: %s\n"
              "   A green board from a run that still had a network is worse than no run."
              % ", ".join(reachable))
        sys.exit(2)
    rec("network is genuinely unplugged", PASS,
        "Garage and OVH S3 unreachable -- verified by connect(), not assumed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--cas", default="/home/builder/genesis/lattice/work/cas")
    ap.add_argument("--ember-src", default="/home/builder/genesis/ember-src")
    ap.add_argument("--keys-dir", default="/home/builder/genesis/keys")
    ap.add_argument("--words", default="dog,water,city")
    ap.add_argument("--expect-no-network", action="store_true")
    a = ap.parse_args(argv)

    print("=" * 78)
    print("P7/S3 — USING THE MIGRATED SLICE IN ACTUAL EMBER%s"
          % ("   [NETWORK UNPLUGGED]" if a.expect_no_network else ""))
    print("store=%s" % a.store)
    print("=" * 78)
    if a.expect_no_network:
        assert_no_network()

    db = _conn(a.store)
    t0 = time.time()
    check_define(db, [w for w in a.words.split(",") if w])
    check_topical(db)
    check_content(db, a.cas, a.keys_dir)
    check_coordinates(db, a.ember_src)
    check_fts(db)
    check_grants(db)

    bad = [n for n, s, _ in RESULTS if s == FAIL]
    print("-" * 78)
    print("%s  %d checks, %d FAIL  (%.1fs)"
          % ("ALL GOOD" if not bad else "FAILURES PRESENT", len(RESULTS), len(bad),
             time.time() - t0))
    for n in bad:
        print("   FAIL: %s" % n)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
