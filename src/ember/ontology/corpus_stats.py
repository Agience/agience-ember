"""Corpus token & document-frequency measurement — the grounding primitive retrieval sits on.

These are the corpus's own measures of how much information a term carries (document frequency /
IDF), read straight off the lexical index (`ember.corpus.fts`). They are pure measurement (no
answering, no condensation), which is why they stay ember-side even though the BM25 retrieval tekton
that consumes them lives in sage: the reach path (`ember.ontology.match.fired_field`) and the recall
path (sage's `content_search`) must agree about which words carry a question, so they both read this
one module's measure from the same store — one path, no stop-list, nothing hand-authored.

`ember.ontology.match` takes `_salient` from here by injection (`ember/ontology/__init__.py` wires
it), and sage's `content_search` imports the whole cluster (`terms`/`_df`/`_salient`/…).

Provenance
----------
Vendored from `agience-mantle`, `src/mantle/ontology/corpus_stats.py` (Apache-2.0, Copyright Ikailo
Inc.), at commit `9768c5f`; mantle removed it in `a7cfd0c` alongside the FTS index it reads. It
follows that index to ember. Copied unmodified apart from this note and the `fts` import path.
"""
from __future__ import annotations

import math
import re
from typing import List, Sequence

from ember.corpus import fts as _fts   # the index — one implementation, vendored beside this one


def terms(query: str) -> List[str]:
    """The query's tokens, verbatim — alphanumerics, lowercased. No filtering: the index's IDF is
    what separates informative words from filler, not a hand-authored list."""
    return [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 1]


def _q(t: str) -> str:
    return '"%s"' % t.replace('"', "")


#
# df is exact, read off `fts5vocab` — one row per term carrying `doc`, straight out of the
# index. No scan, no cap, nothing to tune.


def _df(conn, t: str) -> int | None:
    """The document frequency of a term — exact, from the index's own vocabulary table.

    Returns None when the store cannot report it (an older index with no vocab table). That is
    "unmeasurable", not "very common": a caller must say so rather than substitute a number."""
    return _fts.document_frequency(conn, t)


def _salient(conn, ts: Sequence[str]) -> List[str]:
    """The query's informative terms — those carrying at least the query's own MEAN information.

    A fraction is the wrong shape here: how many terms carry information is not a property of how
    many words were typed. Information is, and the corpus already measures it: `IDF = log(N/df)`. A
    term is salient when it carries at least the mean information of this query's own terms.
    Nothing is chosen: the bar is the query's own mean, so it scales with the query and with the
    corpus.

    ## Two terms cannot be separated (§103)

    Below three distinct terms the filter does not run at all. A threshold splits a population into
    a kept group and a dropped one, and calling one of exactly two terms "scaffolding" is a claim
    about a split that two points cannot support — with two terms ANY interior bar keeps exactly
    one, whichever is rarer, whether the bar is a mean, a median, a largest gap or a maximum
    between-class variance. That is arithmetic, not a reading of the corpus, and it made a two-word
    proper name unsearchable AS a name: `prism protocol` narrowed on `prism` alone and answered
    with the solid geometry.

    So the rule is the honest one: a separation needs a group on each side of it to be a
    separation.

    ## The bar was blamed for a number that was wrong (§103 -> §105)

    A rare term used to drag the mean above EVERY other term, collapsing a query to one word:

        universal artifact model     univers 11.292   artifact 6.711   model 6.129
                                     mean 8.044    ->   kept 1 of 3

    The mean is not robust to outliers, so it was replaced with the median — which fixed those
    queries, kept the long ones identical, and cost 9 modifier answers of 50 (z = -3.01), because
    `>= median` keeps ceil(n/2) terms ALWAYS and a four-word question frame then keeps a frame
    word.

    The outlier was not real. `univers` is carried by 20,326 documents, not 27: `document_frequency`
    re-stemmed an already-stemmed term and Porter's plural rule fired twice (§105). With the
    frequency corrected the same query reads

        universal artifact model     artifact 6.711   model 6.129   univers 4.671
                                     mean 5.837    ->   kept 2 of 3

    and the mean is right here, on every question and gloss shape measured, and on the modifiers
    the median lost. The estimator was never the defect. Measured across four constant-free bars —
    mean, median, largest gap, maximum between-class variance — with the frequencies corrected,
    only the mean is right on all of them; `gap` and `otsu` keep 11 terms of 12 on a gloss, which
    §102 prices at 22x the candidates and 15x the latency.

    When every term carries the same information (all-common, all-rare, or one word) the mean equals
    each of them and everything is kept — which is the honest reading: the corpus reports that
    nothing here distinguishes anything. The `>= mean` comparison, not `>`, is what makes that so."""
    ts = list(ts)
    if len(ts) <= 1:
        return ts
    total = _corpus_rows(conn)
    if not total:
        return ts          # N unmeasurable → IDF unmeasurable → nothing distinguishes anything
    dfs = {t: _df(conn, t) for t in ts}
    if any(d is None for d in dfs.values()):
        return ts          # the index cannot report df → say so, do not guess
    idf = {t: math.log(max(1.0, total / max(1.0, float(dfs[t])))) for t in ts}
    if len(idf) <= 2:
        return ts          # a split needs a group on each side; two points cannot supply one
    n = len(idf)
    total_idf = sum(idf.values())
    keep = [t for t in ts if idf[t] * n >= total_idf]
    return keep or ts


def _corpus_rows(conn) -> float:
    """How many rows the corpus holds — read off the store's maintained counter, never `count(*)`
    (which dereferences every record). An empty or unreadable counter reads 0.0, and `_salient`
    reports that nothing distinguishes anything."""
    try:
        row = conn.execute("SELECT n FROM counter WHERE name = 'vertex'").fetchone()
        if row and int(row[0]) > 0:
            return float(row[0])
    except Exception:
        pass
    return 0.0


def _match_expr(ts: Sequence[str]) -> str:
    """A safe FTS5 MATCH: every term quoted (so punctuation/operators can't reach the parser) and
    OR-ed, so BM25 ranks by how much of the query each document carries."""
    return " OR ".join(_q(t) for t in ts)


__all__ = ["terms", "_q", "_df", "_corpus_rows", "_salient", "_match_expr"]
