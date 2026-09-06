"""enrich_wordnet.py — materialize WordNet's structure into our own corpus, once.

The `wn-*` artifacts already carry WordNet's content (gloss, lemmas, pos). The reasoning layer
(activation and `crystal.ontology.geometry`) reaches into the external nltk package for WordNet's
structure — the hypernym IS-A graph, per-sense information content, and SemCor sense-frequency.
That is the last external dependency. This one-time bootstrap reads that structure out of nltk and
writes it onto our own artifacts, so afterwards the store is self-sufficient and nltk can be
dropped from the runtime.

For every `wn-<synset>` in the store it adds:
  hypernyms          [parent synset name, ...]           the IS-A graph (canonical_parent walks this)
  instance_hypernyms [parent synset name, ...]           instance-of (proper nouns)
  ic                 float                                Resnik information content (Brown corpus)
  lemma_counts       {lemma: SemCor count}               most-frequent-sense weighting + grounding

Run on a publisher node (t5/tu/45) so the _rev-stamped updates propagate to every peer via the mesh
updates feed — no box-to-box copy. Idempotent and resumable: synsets already carrying `ic` are
skipped, so a re-run is cheap and a crash loses no progress.

    PYTHONPATH=.../ember-src:.../mantle:... python scripts/enrich_wordnet.py [--force] [--limit N]
"""
from __future__ import annotations

import math
import sys
import time

# nltk's own "undefined IC" sentinel, which is a finite float rather than `math.inf`.
# `nltk.corpus.reader.wordnet` defines `_INF = 1e+300` as a module-level float literal and
# `information_content` returns it verbatim when the synset's corpus count is zero:
#
#     counts = icpos[synset._offset]
#     if counts == 0:
#         return _INF                      # <- 1e+300, a finite float
#
# `1e300 == float("inf")` is False and `math.isfinite(1e300)` is True, so a guard written against
# `inf` lets it through (see `measure_ic`). Measured over the full 117,659-synset corpus: 50,278
# synsets (42.73%) carry it, all confirmed against nltk as genuine zero-frequency rows. It is
# pinned as a named constant so that a guard does not have to rediscover that nltk's infinity is
# spelled with a literal.
NLTK_IC_INF = 1e300

# The largest real Resnik IC measurable on ic-brown.dat, measured over all 44,393 synsets that
# yield a finite value: 14.709437882542113. `IC_ABSURD` marks where no real measurement can reach.
# It is loose rather than tight: it catches sentinels and unit errors (1e300 clears it by 298
# orders of magnitude), and leaves the top of the legitimate range alone so the guard stays a guard
# rather than becoming a data filter.
IC_MEASURED_MAX = 14.709437882542113

# `IC_ABSURD` is derived from the measurement above, so re-measuring the ceiling moves the guard
# with it and keeps the stated relationship true.
#
# The headroom is the one stated choice here. For a different value to be right, the gap between
# the largest real IC and the smallest sentinel would have to close — and it is 298 orders of
# magnitude wide, so any headroom in [1, 1e297] catches every sentinel and filters no real datum.
# It is stated well above 1 so that a ceiling measured on a smaller corpus than the one in use
# leaves legitimate high-IC rows untouched.
IC_ABSURD_HEADROOM = 7.0
IC_ABSURD = IC_ABSURD_HEADROOM * IC_MEASURED_MAX


def load_ic_table():
    """The Brown IC table, downloading it once if absent. Shared by every caller so that two
    consumers cannot silently derive IC from two different tables."""
    from nltk.corpus import wordnet_ic
    try:
        return wordnet_ic.ic("ic-brown.dat")
    except LookupError:
        import nltk
        nltk.download("wordnet_ic", quiet=True)
        return wordnet_ic.ic("ic-brown.dat")


def measure_ic(s, ic_table):
    """Real IC, or None when it could not be measured — never 0.0 and never a sentinel.

    This is the only IC derivation in the codebase. It is module-level, and imported by
    `lattice_pipeline.py`'s `enrich` stage, so the migration has one implementation to share: two
    IC derivations that disagree do so invisibly — every consumer sees a float — and the
    disagreement lands in the coordinate rather than in an error.

    Two distinct failures share the return value `None`, because they have the same handling and
    only the recorded status separates them:

      * A zero result. 0.0 is a legitimate Resnik value — exactly the IC of the corpus root — so
        writing it on failure makes a failed measurement indistinguishable from a real one at
        every consumer. Measured at corpus scale: of 21,778 rows carrying `ic = 0.0`, 21,777 are
        failed measurements (adjectives and adverbs, for which ic-brown.dat has no part-of-speech
        entry at all) and exactly one — `entity.n.01` — is a real root zero.

      * `NLTK_IC_INF` (1e+300), returned for zero-frequency synsets. A guard written as
        `v not in (float("inf"), float("-inf"))` passes it through as a finite measurement into
        `jc_tree`, `sparse_vec` and `dense_vec`, so the value is tested by name.

    Returning None for both and letting the caller record which is the same handling this file
    uses for the absent-from-nltk case: record the status, leave `ic` absent."""
    return measure_ic_detail(s, ic_table)[0]


def measure_ic_detail(s, ic_table):
    """`(value, status)` — the derivation above, plus why it failed when it did.

    The status is what keeps the three look-alike cases apart once they are written down. They
    are not the same fact and they do not have the same remedy:

      `None`               a real measurement; `ic` is written.
      "ic_zero_frequency"  nltk returned `NLTK_IC_INF`: the synset exists and is in the IC table
                           but its Brown count is 0. 50,278 rows at corpus scale.
      "ic_no_pos_table"    ic-brown.dat has no entries for this part of speech (a/r/s). Resnik IC
                           is not defined for adjectives or adverbs at all. 21,777 rows.
      "ic_out_of_range"    finite but past `IC_ABSURD` — an unrecognised sentinel or unit error;
                           the bucket exists so a new one arrives named rather than unnoticed.

    In every non-None case `ic` is left genuinely absent rather than written as 0.0."""
    from nltk.corpus.reader.wordnet import information_content
    try:
        v = information_content(s, ic_table)
    except Exception:
        return None, "ic_no_pos_table"
    if v != v or v in (float("inf"), float("-inf")):
        return None, "ic_out_of_range"
    if v == NLTK_IC_INF:
        return None, "ic_zero_frequency"
    v = float(v)
    if not (-IC_ABSURD < v < IC_ABSURD):
        return None, "ic_out_of_range"
    return v, None


# ── the smoothed IC producer (Laplace before propagation) + the `ic_se` second channel ─────────
#
# This is structurally separate from the `ic IS NULL` drain below. The drain is per-row,
# incremental and crash-resumable (see the cost note in `main`). Laplace-before-propagation is a
# global operation: `N'` is the total smoothed mass over the entire vocabulary and every synset's
# cumulative count depends on its whole subtree, so no row can be finalized before all of them are
# known. Smoothing therefore runs as a two-phase pass: read the whole vocabulary, derive the table,
# then write. Folding it into the drain would normalize rows against different partial `N'` values,
# giving coordinates that look valid, have plausible magnitudes, and sit in different spaces.
#
# The algorithm is proven over the full 82,115-synset noun vocabulary in `test_geometry_lattice.py`
# (0 monotonicity violations against 39,780 unsmoothed; JC exactness 7.105427e-15). The write path
# has no live-corpus run behind it: check the ontology driver's `ic_coverage()["se_coverage"]`
# after running this before trusting anything downstream of it.
def derive_smoothed_ic(wn, ic_table, pos: str = "n"):
    """`{synset name: (ic, ic_se)}` for one POS, Laplace-smoothed before propagation.

    Fills the zero-frequency hole (measured: 48,861 of 82,115 nouns, 59.5%) with a monotone,
    finite IC and attaches the closed-form standard error `sqrt((1-p)/n)` as a second channel.
    See `crystal.ontology.geometry.smooth_ic` for the derivation and why Good-Turing is not used.
    """
    from ember.ontology import geometry as g

    syns = list(wn.all_synsets(pos))
    tbl = ic_table[pos]

    def cum(s):
        """Brown's descendant-inclusive count. 0.0 = unattested = the hole."""
        return float(tbl.get(s.offset(), 0.0))

    # The canonical spanning tree, over the raw IC: the tree is chosen independently of the numbers
    # derived from it. `_A` is the minimal adapter `geometry.canonical_parent` needs.
    class _A:
        __slots__ = ("s",)

        def __init__(self, s):
            self.s = s

        def name(self):
            return self.s.name()

        def ic(self):
            c, root = cum(self.s), float(tbl.get(0, 0.0))
            return -math.log(c / root) if c and root else 0.0

        def hypernyms(self):
            return [_A(h) for h in self.s.hypernyms()]

        def instance_hypernyms(self):
            return [_A(h) for h in self.s.instance_hypernyms()]

    parents = {}
    for s in syns:
        p = g.canonical_parent(_A(s), None)
        parents[s.name()] = p.name() if p is not None else None

    # De-propagate to own counts over that same tree, so Laplace lands before propagation.
    # WordNet is a DAG and nltk propagates each count up every hypernym path, so this subtraction
    # can in principle go negative from double-counting. Measured on ic-brown.dat's noun table over
    # the canonical tree: 0 of 82,115 nodes go negative. `smooth_ic` clamps at 0 as a guard rather
    # than as a load-bearing correction.
    children = {}
    for n, p in parents.items():
        if p:
            children.setdefault(p, []).append(n)
    by_name = {s.name(): s for s in syns}
    own = {n: cum(by_name[n]) - sum(cum(by_name[c]) for c in children.get(n, ()))
           for n in parents}

    return g.smooth_ic(own, parents, alpha=g.IC_LAPLACE_ALPHA)


def main() -> int:
    force = "--force" in sys.argv
    limit = None
    for i, a in enumerate(sys.argv):
        if a == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    from mantle.shard.local_store import open_store
    from nltk.corpus import wordnet as wn

    ic = load_ic_table()

    bundle = open_store()
    arts = bundle.artifacts

    def _ic(s):
        """Delegates to the module-level `measure_ic` — see its docstring for why there is
        exactly one IC derivation and what its two failure modes are."""
        return measure_ic(s, ic)

    # ── preflight: the corpus is proven usable before anything is written or deleted ─────────
    # The WordNet corpus itself needs the guard as much as `wordnet_ic` does. nltk's
    # LazyCorpusLoader raises LookupError on first access, LookupError subclasses Exception, and
    # the per-synset handler below is a bare `except Exception` — so with `wordnet_ic` present but
    # `wordnet` missing (or a partial nltk data dir), every row takes the "absent" branch. The run
    # then blanks `hypernyms`/`lemma_counts` on ~10^5 rows, prints
    # `DONE enriched=0 absent_in_nltk=117659`, and returns 0. Because `put_many` stamps a fresh
    # `_rev`, those blanked rows propagate to every peer over the mesh updates feed; and because
    # every row then has `ic` set, no row matches `ic IS NULL`, so a re-run is a no-op and the
    # corpus reads as fully enriched. One probe closes that path: if the corpus cannot resolve a
    # synset every build has, nothing else runs.
    try:
        _probe = wn.synset("dog.n.01")
        if not _probe.hypernyms():
            raise RuntimeError("wordnet resolved 'dog.n.01' but it has no hypernyms")
    except Exception as _e:
        print("ABORT: the nltk WordNet corpus is not usable (%s: %s).\n"
              "       Nothing was written. Run `python -m nltk.downloader wordnet` first.\n"
              "       Refusing to proceed: every synset would be marked absent, the blanked rows\n"
              "       would replicate to every peer, and the run would look like a success."
              % (type(_e).__name__, str(_e)[:160]), flush=True)
        return 2

    CT = "text/x-wordnet"
    page = 2000
    enriched = missing = 0
    t0 = time.time()

    # Cost, stated plainly: this query is not index-served, and that is accepted here.
    #   * `ic IS NULL` cannot ride an index — the LSM nullStrategy is SKIP, so null rows are not in
    #     the index at all. Negations and IS NULL are post-filters.
    #   * the `content_type = :ct` half is indexed, but an indexed equality is cheap only on a
    #     selective value: measured on node 71 (5,657,386 rows) a 48-row type answered in 0.12s
    #     while text/markdown (~6M rows) timed out at 120s. text/x-wordnet is ~10^5 rows —
    #     mid-range, so each page re-walks the whole wordnet set rather than seeking.
    # So this is O(wordnet_rows) per page, O(wordnet_rows * pages) overall. That is the accepted
    # shape: this is a one-shot bootstrap run by hand on a publisher node rather than a per-cycle
    # path, and the drain is what makes it idempotent and crash-resumable (see below) — a keyset
    # would restart from the head of the corpus on every re-run and lose that property.
    #
    # Drain by `ic IS NULL` (content_type is indexed; the filter shrinks as rows are enriched).
    # Every processed row gets either `ic` or `ic_status`, so it drops out of the filter and the
    # loop terminates. No cursor, no string-id keyset. put_many stamps a fresh _rev on each write,
    # so enriched rows propagate via the mesh updates feed to every peer with no box-to-box copy.
    # Idempotent and resumable: a re-run only touches rows still missing `ic`. `--force` re-derives
    # everything by first clearing the guard.
    if force:
        # `--force --limit N` is data loss: the REMOVE below is unbounded while the refill loop
        # stops after N rows, so every row past N stays permanently stripped, and the pre-existing
        # values were the only copy in the corpus. The combination is rejected rather than
        # half-applied. The wipe also runs after the preflight above, so it cannot fire on a box
        # whose WordNet corpus cannot refill it.
        if limit:
            print("ABORT: --force with --limit would REMOVE ic/hypernyms from every wordnet row "
                  "but refill only %d of them, leaving the rest permanently stripped. "
                  "Run --force alone, or drop --force." % limit, flush=True)
            return 2
        # Readers see a flat WordNet until the drain finishes. Between this REMOVE and the last
        # page, the ontology driver coerces the missing fields to ic=0.0/hypernyms=[], and it
        # caches the index as a process-lifetime singleton with no invalidation, so any reasoning
        # process that loads during the window keeps the flat WordNet until it restarts. Run this
        # with readers stopped, or restart them afterwards.
        arts.c.command("UPDATE Artifact REMOVE ic, ic_status, hypernyms, instance_hypernyms, "
                       "lemma_counts WHERE content_type = :ct", {"ct": CT})

    no_ic = 0
    while True:
        # The drain guard is (ic IS NULL AND ic_status IS NULL). A row that cannot be measured gets
        # `ic_status` in place of an `ic`, so it drops out of the filter and the loop terminates
        # without a number behind it that no consumer could trust.
        rows = arts.c.query(
            f"SELECT FROM Artifact WHERE content_type = :ct AND ic IS NULL AND ic_status IS NULL "
            f"LIMIT {page}", {"ct": CT})
        if not rows:
            break
        batch = []
        for raw in rows:
            doc = {k: v for k, v in raw.items() if not k.startswith("@")}   # drop @rid/@type/@cat
            name = doc["id"][3:] if doc["id"].startswith("wn-") else doc["id"]
            try:
                s = wn.synset(name)
            except Exception:
                # The synset is absent from this nltk build: `ic` is left absent and `ic_status`
                # records why. Writing `ic = 0.0` here would be a sentinel meaning "absent" wearing
                # the shape of a real measurement — 0.0 is exactly the IC of the corpus root, so no
                # consumer could separate them, and `jc_tree` between two such synsets is 0+0-0 =
                # 0.0, making `forgetting._sim = exp(-0)` report similarity 1.0 for two unrelated
                # unknown synsets.
                #
                # The structure fields are written empty because nltk holds nothing to read for
                # this synset. An empty `hypernyms` leaves `sparse_vec` empty and `dense_vec` the
                # zero vector, and terminates every descendant's `tree_path` here, which inflates
                # JC distances for nodes that were enriched.
                doc["ic_status"] = "absent_in_nltk"
                doc["hypernyms"] = []
                doc["instance_hypernyms"] = []
                doc["lemma_counts"] = {}
                missing += 1
                batch.append(doc)
                continue
            doc["hypernyms"] = [h.name() for h in s.hypernyms()]
            doc["instance_hypernyms"] = [h.name() for h in s.instance_hypernyms()]
            doc["lemma_counts"] = {l.name(): int(l.count()) for l in s.lemmas()}
            v, why = measure_ic_detail(s, ic)
            if v is None:
                # Resolved, but IC could not be measured. `why` names which failure it was
                # (zero-frequency / no POS table / out of range); a single "ic_unavailable" bucket
                # would hide a 50,278 vs 21,777 split, and this row belongs to neither the
                # `enriched` nor the `absent` count.
                doc["ic_status"] = why or "ic_unavailable"
                no_ic += 1
            else:
                doc["ic"] = v
                enriched += 1
            batch.append(doc)
        # The return value is the only failure signal, so it is checked. `put_many` returns what it
        # handled rather than what it was handed (it retries per-doc and subtracts failures). Rows
        # whose write failed still match the drain filter, so the next iteration would re-select
        # the same page — a loop whose counters climb on every pass, printing a rising "processed"
        # count over the same 2000 rows while writing nothing.
        wrote = arts.put_many(batch)                # author stamps _rev -> mesh updates propagate
        if isinstance(wrote, int) and wrote < len(batch):
            print("ABORT: store handled %d of %d rows in this page. Those rows still match the "
                  "drain filter, so continuing would re-select the same page forever. Fix the "
                  "store and re-run — progress so far is durable." % (wrote, len(batch)), flush=True)
            return 1
        done = enriched + missing + no_ic
        print(f"  ...{done} processed ({enriched} real, {no_ic} no-ic, {missing} absent)"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        if limit and done >= limit:
            break

    print(f"DONE enriched={enriched} ic_unavailable={no_ic} absent_in_nltk={missing} "
          f"in {time.time()-t0:.0f}s", flush=True)
    # A run that measured nothing is not a success, whatever the drain did.
    if enriched == 0 and (missing or no_ic):
        print("WARNING: zero real IC values were measured. The corpus is now marked drained but "
              "carries no information content — check the nltk build before trusting any "
              "IC-derived metric (see wn_store.ic_coverage()).", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
