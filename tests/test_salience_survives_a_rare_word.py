r"""A rare word on the query must not silence the rest of it (§103, §105).

`_salient` keeps the terms carrying at least the query's own MEAN information. Measured on the live
store it appeared to collapse a query to its rarest word:

    universal artifact model     univers 11.292   artifact 6.711   model 6.129
                                 mean 8.044    ->   kept 1 of 3

`univers` also stems `university`, so what survived was a different question, and the canon document
of that exact name came back behind `undergrad`, `cosmic time` and `prof`.

The mean was blamed for that, and replaced with the median — robust, constant-free, identical on
every long query, and it cost 9 modifier answers of 50 (z = -3.01), because `>= median` keeps
ceil(n/2) terms ALWAYS and a four-word question frame then keeps a frame word.

**The outlier was not real.** `univers` is carried by 20,326 documents, not 27; `document_frequency`
re-stemmed an already-stemmed term (§105, `test_document_frequency_does_not_restem`). With the
frequency corrected the mean bar keeps 2 of 3 here on its own, and is right on every shape the
median was right on plus the modifiers it was not.

So these tests use the CORRECTED frequencies, and pin the two things that survived: what the bar
does on each query shape, and that two terms are never separated at all.
"""
from __future__ import annotations

import sqlite3

import pytest

from ember.ontology import corpus_stats as CS


def _corpus(monkeypatch, rows: int, dfs: dict) -> sqlite3.Connection:
    """A conn reporting `rows` documents and the given frequencies, and nothing else.

    `_df` is stubbed rather than an FTS index built: the quantity under test is the arithmetic on
    top of df, and stubbing df is what keeps that arithmetic the only thing that can fail. That
    the frequencies themselves are read correctly is `test_document_frequency_does_not_restem`.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE counter (name TEXT PRIMARY KEY, n INTEGER)")
    conn.execute("INSERT INTO counter VALUES ('vertex', ?)", (rows,))
    monkeypatch.setattr(CS, "_df", lambda c, t: dfs.get(t))
    return conn


# Measured on the live store, 2,165,558 rows, AFTER the §105 correction.
N = 2_165_558
UNIVERSAL = {"univers": 20_326, "artifact": 2_638, "model": 4_721}
GLACIER = {"glacier": 418, "what": 3_929, "is": 17_971, "a": 1_283_401}
MODIFIER = {"what": 3_929, "doe": 1_405, "turgid": 39, "mean": 9_612}
PALM = {"syria": 464, "feather": 649, "palm": 984, "edibl": 1_199, "tall": 1_378,
        "sweet": 1_606, "bear": 2_529, "fruit": 3_168, "tropic": 3_343, "nativ": 3_994,
        "tree": 7_491, "to": 66_316}


class TestARareWordDoesNotSilenceTheRest:

    def test_a_three_word_name_keeps_more_than_one_word(self, monkeypatch):
        """The §103 symptom, with the frequency §105 corrected. No estimator change needed."""
        keep = CS._salient(_corpus(monkeypatch, N, UNIVERSAL), list(UNIVERSAL))
        assert len(keep) > 1, (
            "the query collapsed to one word — this is what made `universal artifact model` "
            "answer with `undergrad`; kept %r" % (keep,))
        assert set(keep) == {"artifact", "model"}, keep


class TestTwoTermsAreNeverSeparated:

    def test_a_two_word_name_keeps_both_words(self, monkeypatch):
        """`prism protocol` is a name. Narrowed on `prism` alone it answers with solid geometry.

        With two terms ANY interior bar keeps exactly one — mean, median, largest gap or maximum
        between-class variance alike — so this is not a threshold that can be tuned, it is a split
        two points cannot support.
        """
        conn = _corpus(monkeypatch, N, {"prism": 379, "protocol": 763})
        assert sorted(CS._salient(conn, ["prism", "protocol"])) == ["prism", "protocol"]

    @pytest.mark.parametrize("dfs", [
        {"a": 1, "b": 1_000_000},          # maximally far apart
        {"a": 500, "b": 501},              # all but identical
    ])
    def test_no_pair_is_ever_split_however_far_apart(self, monkeypatch, dfs):
        assert len(CS._salient(_corpus(monkeypatch, N, dfs), list(dfs))) == 2


class TestEachQueryShapeKeepsWhatItShould:
    """§77 measured this filter as load-bearing on short questions and §102 priced it at 15x
    latency on long ones, so the kept set per shape is the contract, pinned here."""

    def test_a_question_drops_its_scaffolding(self, monkeypatch):
        keep = set(CS._salient(_corpus(monkeypatch, N, GLACIER), list(GLACIER)))
        assert keep == {"glacier", "what"}, keep

    def test_a_modifier_question_keeps_only_the_word_asked_about(self, monkeypatch):
        """"what does turgid mean" must not keep `doe` — the stem of `does`, and also a deer.

        This is the shape the median bar lost 9 answers of 50 on: it keeps ceil(n/2) = 2 terms
        here however uninformative the second one is.
        """
        keep = set(CS._salient(_corpus(monkeypatch, N, MODIFIER), list(MODIFIER)))
        assert keep == {"turgid"}, keep

    def test_a_gloss_keeps_half_its_terms(self, monkeypatch):
        keep = CS._salient(_corpus(monkeypatch, N, PALM), list(PALM))
        assert len(keep) == 6, "the 15x latency lever changed size: %r" % (keep,)
        assert "to" not in keep

    def test_all_equally_informative_terms_are_all_kept(self, monkeypatch):
        """The honest reading when nothing distinguishes anything — `>=`, not `>`."""
        conn = _corpus(monkeypatch, N, {"a": 100, "b": 100, "c": 100})
        assert len(CS._salient(conn, ["a", "b", "c"])) == 3


class TestTheUnmeasurableCorpusStillKeepsEverything:

    def test_an_unreadable_df_keeps_every_term(self, monkeypatch):
        assert CS._salient(_corpus(monkeypatch, N, {}), ["x", "y", "z"]) == ["x", "y", "z"]

    def test_an_uncountable_corpus_keeps_every_term(self, monkeypatch):
        conn = _corpus(monkeypatch, 0, {"x": 1, "y": 2, "z": 3})
        assert CS._salient(conn, ["x", "y", "z"]) == ["x", "y", "z"]
