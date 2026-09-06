"""`tokenize_spans`' regex must split exactly where FTS5 splits — measured against FTS5 itself.

## Why this is load-bearing

`corpus_fts` writes its index with FTS5's own `unicode61` tokenizer, and then locates spans in the
source text with a Python regex. If the two disagree about where a word starts, the index holds one
term while the span extractor looks for another: BM25 still ranks correctly and the snippets point
at the wrong characters. That failure surfaces as bad grounding, never as an error — the same shape
as the `snippet()`-returns-NULL defect this module already carries a warning about.

`fts.py` documents the regex as "deliberately not byte-exact with FTS5's tokenizer" and calls the
worst case safe. This measures that claim.

## What is compared

`_TOKEN_RE` against a live FTS5 `unicode61` table — the tokenizer the index is actually built with.
Boundaries only: the table is created without `porter`, because stemming is a separate stage here
(`_Stemmer` proxies FTS5's stemmer precisely so it cannot drift) and mixing the two would compare
tokenization against stemming and report differences that are not tokenization's.

Measured over the probe set below: 0 disagreements out of 13, spanning apostrophes, underscores,
hyphens, digits and Windows paths.

Accented text is held in a second set and asserted differently. `unicode61` removes diacritics and
this regex does not, so `café` is indexed as `cafe` while the extractor reads `café`. That is a
difference in the key and never in the split — folding happens inside a token and cannot move a
boundary — so exact list equality is the wrong assertion for those inputs and would report a
boundary defect that is not there. What the key difference would break is a lookup against the
index's own vocabulary, and it does not, because every token that becomes an index key goes through
`_Stemmer`, which is FTS5 itself. Both halves of that are asserted below.

## The sibling arm deliberately differs, and that is fine

`mantle`'s SSE tokenizer is `\\w+(?:'\\w*)*` and disagrees with `unicode61` on 6 of these 13 probes.
Both differences are understood and neither is drift:

  * **Apostrophes — deliberate, and better.** SSE keeps `dog's` whole so `strip_possessive` can take
    it to `dog`. `unicode61` splits it into `dog` and a junk `s` term. `tokenizer.py` states this
    intent ("Mirrors Lucene's StandardTokenizer").
  * **Underscores — latent, with zero measured impact.** Python's `\\w` includes `_`, so SSE keeps
    `snake_case_name` as one term where FTS5 makes three. Measured over the live store: **0 tokens
    containing `_` out of 78,341** emitted across 60,000 real artifact offers. SSE indexes the
    offer (title / description / tags), which is prose. Changing the regex would force a full
    reindex — the tokenizer is part of the index format — to fix a case the corpus does not contain.

The two arms index different stores and a query reaches one or the other, so they are not required
to agree. What IS required is that each matches the index it writes, and that is what this file
pins for the FTS5 arm.
"""
from __future__ import annotations

import sqlite3

import pytest

from ember.corpus.fts import _TOKEN_RE


#: Chosen to hit every character class where a tokenizer can plausibly differ.
_PROBES = [
    "don't stop",                 # apostrophe
    "it's the dog's bowl",        # possessives
    "O'Brien said",               # leading-capital apostrophe
    "foo_bar baz",                # underscore
    "snake_case_name",            # multiple underscores
    "x86_64 arch",                # underscore between digit runs
    "e-mail address",             # hyphen
    "co-operate",                 # hyphen mid-word
    "run-time error",             # hyphen
    "3.14 pi",                    # decimal point
    "U.S.A. today",               # dotted acronym
    "naive cafe",                 # plain ascii control for the accented row below
    r"C:\Users\example",          # windows path: colon and backslashes
]

#: Held separately, because `unicode61` folds diacritics and this regex does not. Exact list
#: equality is the wrong assertion for them and would report a boundary defect that is not there;
#: `test_a_diacritic_changes_the_key_and_not_the_split` states what actually holds.
_ACCENTED = [
    "café résumé",                # acute accents
    "naïve",                      # diaeresis
    "Ångström über",              # ring above, umlaut
]


@pytest.fixture(scope="module")
def fts5_boundaries():
    """FTS5's own `unicode61` tokenizer, as ground truth. No stemmer — boundaries only."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE s USING fts5(w, content='', tokenize='unicode61')")
    conn.execute("CREATE VIRTUAL TABLE sv USING fts5vocab(s,'instance')")
    counter = {"n": 0}

    def split(text: str):
        counter["n"] += 1
        rid = counter["n"]
        conn.execute("INSERT INTO s(rowid, w) VALUES (?, ?)", (rid, text))
        return [r[0] for r in conn.execute(
            "SELECT term FROM sv WHERE doc = ? ORDER BY offset", (rid,))]

    yield split
    conn.close()


def _ours(text: str):
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def test_the_ground_truth_is_actually_fts5(fts5_boundaries):
    """Control. A fixture that silently returned nothing would make every comparison vacuous."""
    got = fts5_boundaries("the quick brown fox")
    assert got == ["the", "quick", "brown", "fox"], (
        "the unicode61 table did not tokenize a trivial sentence (%r) — the ground truth is not "
        "ground truth and nothing below means anything." % (got,))


@pytest.mark.parametrize("text", _PROBES)
def test_our_regex_splits_exactly_where_fts5_splits(text, fts5_boundaries):
    """Exact list equality, including order. Not a set: position is what a span needs."""
    ours, theirs = _ours(text), fts5_boundaries(text)
    assert ours == theirs, (
        "\nthe span regex and FTS5 disagree about where words start.\n\n"
        "  input : %r\n  ours  : %r\n  fts5  : %r\n\n"
        "The index is written by FTS5 and the spans are located by this regex, so a disagreement "
        "here does not raise — the index holds one term while the extractor looks for another, "
        "BM25 still ranks correctly, and the snippet points at the wrong characters. Grounding "
        "degrades silently." % (text, ours, theirs))


@pytest.mark.parametrize("text", _ACCENTED)
def test_a_diacritic_changes_the_key_and_not_the_split(text, fts5_boundaries):
    """Folding happens inside a token, so it must leave the boundaries alone.

    Asserted as: the same number of tokens, in the same order, equal after the same folding FTS5
    applies. A weaker check on count alone would pass on a regex that split in the wrong place and
    happened to produce the same number of pieces.
    """
    import unicodedata

    def fold(t):
        return "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))

    ours, theirs = _ours(text), fts5_boundaries(text)
    assert len(ours) == len(theirs), (
        "diacritic handling moved a boundary.\n  input : %r\n  ours  : %r\n  fts5  : %r"
        % (text, ours, theirs))
    assert [fold(t) for t in ours] == theirs, (
        "the two disagree by more than the diacritics.\n  input : %r\n  ours folded : %r\n"
        "  fts5        : %r" % (text, [fold(t) for t in ours], theirs))


def test_the_folded_key_is_taken_from_fts5_rather_than_reimplemented():
    """The consequence of the difference above, closed.

    The index is written in the folded alphabet, so an accented token used directly as an index key
    would miss — `fts5vocab` holds `cafe` and a lookup for `café` returns no row, which reads as a
    document frequency that is "unmeasurable" rather than "absent" and silently drops that word's
    information weight.

    It does not happen, because `document_frequency` stems through `_Stemmer`, whose scratch table
    is declared `tokenize='porter unicode61'` — so FTS5 does the folding on the way in, the same
    discipline `_Stemmer` exists to enforce for stemming. Pinned here rather than assumed: it is the
    only thing standing between the two alphabets.
    """
    from ember.corpus.fts import _STEMMER

    assert _STEMMER.stem_many(["café", "naïve"]) == {"café": "cafe", "naïve": "naiv"}, \
        "the stemmer stopped folding diacritics, so an accented word no longer reaches its own row"


def test_the_comparison_would_notice_a_real_difference(fts5_boundaries):
    """Negative control: prove the assertion above can fail.

    `\\w+` — mantle's SSE class — keeps apostrophes and underscores. It must disagree with FTS5 on
    exactly the inputs where those appear, or this file is comparing something else.
    """
    import re
    sse_like = re.compile(r"\w+(?:'\w*)*", re.UNICODE)

    def theirs(t):
        return [m.group(0).lower() for m in sse_like.finditer(t)]

    disagreements = [p for p in _PROBES if theirs(p) != fts5_boundaries(p)]
    assert disagreements, (
        "a regex that keeps apostrophes and underscores agreed with unicode61 on every probe, so "
        "the probe set cannot distinguish two tokenizers and this file proves nothing.")
    # And it must be the apostrophe/underscore probes specifically, not some accident.
    assert any("'" in p for p in disagreements), "no apostrophe probe separated the two"
    assert any("_" in p for p in disagreements), "no underscore probe separated the two"


class TestTheStemmerIsUsableFromAnyThread:
    """`_STEMMER` is a module-level singleton, so its connection cannot live on the instance.

    A `sqlite3.Connection` may only be used from the thread that created it. The stemmer holds an
    in-memory FTS5 scratch table, and while that connection sat on `self`, the first thread to
    miss the cache created it and every other thread raised `ProgrammingError`. Under a server
    that runs requests on a threadpool that is almost every request after the first.

    It was invisible for two reasons, and both are why this test exists rather than a comment:
    the CACHE means only a miss touches the connection, so a warm word works from any thread and
    the failure is word-dependent; and `document_frequency` catches `sqlite3.Error` and returns
    None, which `corpus_stats._salient` reads as "unmeasurable" and answers by keeping every term.
    So it never failed — it silently stopped weighting, and the recall path stopped filtering its
    stems on every query after the first.
    """

    def test_a_second_thread_can_stem_a_word_the_first_never_saw(self):
        import threading

        from ember.corpus import fts

        stemmer = fts._Stemmer()
        assert stemmer.stem("running") == "run", "the creating thread must work at all"

        out, err = {}, []

        def other():
            try:
                # A word the first thread never stemmed, so this MUST reach the connection
                # rather than being served from the shared cache — which is the whole point.
                out["v"] = stemmer.stem("jumping")
            except BaseException as exc:      # noqa: BLE001 — the failure under test
                err.append(exc)

        t = threading.Thread(target=other)
        t.start()
        t.join()

        assert not err, "stemming from a second thread raised %r" % (err[0],)
        assert out["v"] == "jump"

    def test_the_cache_is_shared_across_threads(self):
        """Per-thread connections must not mean per-thread work for a word already stemmed."""
        import threading

        from ember.corpus import fts

        stemmer = fts._Stemmer()
        stemmer.stem("walking")
        seen = {}

        def other():
            # No connection is created in this thread at all: the word is already cached, and
            # `_ensure` is only reached on a miss.
            seen["cached"] = stemmer.stem("walking")
            seen["conn"] = getattr(stemmer._local, "conn", None)

        t = threading.Thread(target=other)
        t.start()
        t.join()

        assert seen["cached"] == "walk"
        assert seen["conn"] is None, (
            "a cache hit must not open a scratch table; if it does, every thread pays for a "
            "word the process already knows"
        )
