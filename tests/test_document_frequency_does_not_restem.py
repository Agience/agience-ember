r"""A term that is already a stem must not be stemmed again (§105).

FTS5 is declared `tokenize='porter unicode61'`, and the vocab table therefore holds STEMS.
`document_frequency` stemmed whatever it was handed before looking it up — correct for a raw word,
wrong for a stem, because Porter is not idempotent. Step 1a strips a trailing `s` and does not ask
whether it has run before:

    univers  ->  univ            vocab[univers] = 20,326      vocab[univ] = 27

Every caller passes stems: `_salient` receives the query's analyzed terms. So the frequency read for
such a term was the frequency of a word nobody asked about, 750x too small on the live store. IDF is
`log(N/df)`, so a df that small produces an enormous information score, and `_salient` keeps terms by
comparing those scores — one such term silenced every other word of the query.

Porter fixed points (`water`, `prism`, `doe`) were never affected, which is why this survived: it is
wrong only for the terms it is wrong for, and they look like ordinary words.
"""
from __future__ import annotations

import sqlite3

import pytest

from ember.corpus import fts as F


@pytest.fixture()
def indexed() -> sqlite3.Connection:
    """A real FTS5 index with the production schema, so the stemmer under test is the real one."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE %s USING fts5(title, description, tags, content, "
            "content='', contentless_delete=1, tokenize='porter unicode61')" % F.FTS_TABLE)
    except sqlite3.OperationalError:                      # pragma: no cover
        pytest.skip("this sqlite build has no fts5")
    conn.execute("CREATE VIRTUAL TABLE %s USING fts5vocab(%s, row)"
                 % (F.VOCAB_TABLE, F.FTS_TABLE))
    rows = [("university of the air", "", "", "a university offering courses"),
            ("universe", "", "", "everything that exists anywhere"),
            ("universal joint", "", "", "a universal coupling"),
            ("water", "", "", "binary compound of oxygen and hydrogen")]
    conn.executemany("INSERT INTO %s(title, description, tags, content) VALUES (?,?,?,?)"
                     % F.FTS_TABLE, rows)
    conn.commit()
    return conn


class TestAStemIsLookedUpAsGiven:

    def test_the_stem_finds_every_document_the_raw_word_finds(self, indexed):
        """`univers` is what the index wrote for all three; it must not read as one of them."""
        assert F.document_frequency(indexed, "university") == 3
        assert F.document_frequency(indexed, "univers") == 3, (
            "the stem was re-stemmed to `univ` and read a frequency for a word nobody asked "
            "about — this is the §105 regression")

    def test_a_porter_fixed_point_is_unaffected(self, indexed):
        """The terms that always worked must keep working — the repair must not move them."""
        assert F.document_frequency(indexed, "water") == 1

    def test_an_absent_term_is_zero_not_none(self, indexed):
        """Zero is a measurement. `None` is reserved for an index that cannot answer at all."""
        assert F.document_frequency(indexed, "quasar") == 0

    def test_stemming_is_not_idempotent_which_is_why_this_matters(self):
        """The premise, asserted directly, so the test above cannot pass for a different reason."""
        once = F._STEMMER.stem_many(["university"])["university"]
        twice = F._STEMMER.stem_many([once])[once]
        assert once == "univers"
        assert twice != once, (
            "Porter became idempotent; if that isreal the §105 guard is dead weight and this "
            "test should be re-derived rather than deleted")


class TestTheVocabIsTheAuthority:

    def test_an_index_with_no_vocab_table_reports_unmeasurable(self):
        conn = sqlite3.connect(":memory:")
        assert F.document_frequency(conn, "anything") is None


class TestASurfaceFormMustNotShadowItsStem:
    r"""The other direction of §105, measured on the live 71/home store 2026-08-27.

    §105's repair was "look the term up AS GIVEN first, and only stem if that misses". That is
    right whenever the surface form is ABSENT — `ontology` misses, stems to `ontolog`, reads 409.
    It is wrong when the surface form is PRESENT with a spurious small count, because then the
    as-given lookup returns the artifact and the real mass under the stem is never consulted:

        this   as-given 2    stem `thi`  25,706     12,853x understated
        are    as-given 3    stem `ar`    6,597      2,199x understated
        collection 1         stem `collect` 5,233    5,233x understated
        operator  11         stem `oper`  8,488        772x understated

    896 of the 904 terms where BOTH forms exist are under-reported — 9.2% of all word-occurrences
    in real workspace prose — and the list is this domain's own vocabulary, not just function
    words. The consequence is the one §105 already wrote down: IDF is `log(N/df)`, `_salient` keeps
    terms by comparing IDFs against their mean, so one hugely-overstated term silences every other
    word of the query.

    WHY THIS CANNOT BE BUILT THROUGH THE REAL TOKENIZER, which is itself the finding. Porter stems
    every token it is given (`this`->`thi`, `key`->`kei`, `collection`->`collect`), so a literal
    `this` can never be written to a `porter unicode61` vocab. The live entries are contamination
    from writes made under a different tokenizer, and they are tiny precisely because there are few
    of them. So the shadowing case is set up directly on the vocab table rather than through an
    INSERT that cannot produce it.

    THE RULE. The index stored the mass under exactly ONE form; the larger count identifies which.
    That is correct in both directions at once — it keeps §105's `univers` 20,326 over `univ` 27,
    and it stops `this` 2 shadowing `thi` 25,706 — so neither repair has to know which kind of
    caller it is serving.
    """

    @staticmethod
    def _vocab(pairs):
        """A stand-in vocab table holding exactly `pairs`, with the production table names."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE %s (term TEXT PRIMARY KEY, doc INTEGER)" % F.VOCAB_TABLE)
        conn.executemany("INSERT INTO %s(term, doc) VALUES (?,?)" % F.VOCAB_TABLE, pairs)
        conn.commit()
        return conn

    def test_a_contaminated_surface_entry_does_not_shadow_the_stem(self):
        conn = self._vocab([("this", 2), ("thi", 25706)])
        assert F._STEMMER.stem_many(["this"])["this"] == "thi", "premise: `this` stems to `thi`"
        assert F.document_frequency(conn, "this") == 25706, (
            "the as-given lookup returned a contaminated 2 and never consulted `thi` — the term "
            "then carries an enormous IDF and silences every other word of the query")

    def test_the_domain_vocabulary_case(self):
        conn = self._vocab([("collection", 1), ("collect", 5233)])
        assert F.document_frequency(conn, "collection") == 5233

    def test_105_still_holds_in_its_own_direction(self):
        """The repair must not undo the one it is extending: `univers` outranks a re-stemmed
        `univ`, and it does so under the same rule rather than a second special case."""
        conn = self._vocab([("univers", 20326), ("univ", 27)])
        assert F.document_frequency(conn, "univers") == 20326

    def test_a_surface_form_that_is_genuinely_the_larger_is_kept(self):
        conn = self._vocab([("water", 900), ("wat", 1)])
        assert F.document_frequency(conn, "water") == 900

    def test_an_absent_surface_still_falls_through_to_its_stem(self):
        """`ontology` is absent from the vocab and must still read `ontolog` — the behaviour that
        was already correct, asserted so the repair cannot quietly drop it."""
        conn = self._vocab([("ontolog", 409)])
        assert F.document_frequency(conn, "ontology") == 409

    def test_absent_everywhere_is_still_zero(self):
        conn = self._vocab([("ontolog", 409)])
        assert F.document_frequency(conn, "quasar") == 0
