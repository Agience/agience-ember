"""The two Porter stemmers must agree.

## The two

    mantle/search/mantle/sse/tokenizer.py::porter_stem   hand-written Porter (1980), ~200 lines
    ember/corpus/fts.py::_Stemmer                        proxies SQLite FTS5's own `porter`

Both exist for good reasons and neither can be deleted. mantle's is client-side by necessity: SSE
stems a term before HMAC'ing it into a blind token, and there is no SQLite in that path. Ember's is
a proxy by necessity: reimplementing Porter beside the FTS index is exactly how the index comes to
hold one stem while the span extractor looks for another — a failure that shows up as bad snippets
and never as an error, which is why `fts.py` refuses to do it.

## Why they must agree anyway

They are two stemmers over one English, and where both touch the same corpus a divergence does not
raise — it silently changes which documents match. A term stemmed one way at index time and the
other way at query time simply misses.

## What each stemmer is asserted against

mantle's is tested against Porter's 1980 paper (`test_sse_tokenizer.py`, steps 1a-5b). Ember's
header states the agreement in prose, naming `quickly -> quickli`, `happy -> happi` and
`universities -> univers`. Neither establishes that the two agree with each other: `porter_stem`
has no consumer outside mantle, so no import exists that could construct the comparison, and "both
are Porter" is not a specification — Porter has well-known implementation variants (step-5a
`e`-deletion, `y`-consonant handling), and two faithful implementations can still disagree.

ember is the right home for this test: it is the only package that imports both.

## What this asserts, and how

A real word list, stemmed by both, compared exactly. Disagreements are reported in full rather
than as a count, because the interesting output of this test is not "they differ" but "here is the
class of word where they differ".

This is a conformance test, not a correctness test. It cannot say which stemmer is right — if it
fails, the question is which index is already written in which stem, and that is answered by
`geom.ic-basis`-style provenance, not by rerunning Porter.
"""
from __future__ import annotations

import pytest

# Both are importable here: ember -> mantle is a declared edge, and `ember.corpus.fts` is ember's own.
from mantle.search.mantle.sse.tokenizer import porter_stem
from ember.corpus.fts import _Stemmer


#: A word list chosen to exercise the steps Porter implementations diverge on, not to be large.
#: Each group names the step it probes, so a failure localises immediately.
_WORDS = [
    # step 1a — plurals
    "caresses", "ponies", "ties", "caress", "cats", "universities", "queries",
    # step 1b — -eed / -ed / -ing, and the post-1b cleanup that variants disagree on
    "feed", "agreed", "plastered", "bled", "motoring", "sing", "conflated", "troubled",
    "sized", "hopping", "tanned", "falling", "hissing", "fizzed", "failing", "filing",
    # step 1c — y -> i, the classic variant point
    "happy", "sky", "cry", "by", "say", "enjoy",
    # steps 2, 3, 4 — the long suffix tables
    "relational", "conditional", "rational", "valenci", "hesitanci", "digitizer",
    "conformabli", "radicalli", "differentli", "vileli", "analogousli", "vietnamization",
    "predication", "operator", "feudalism", "decisiveness", "hopefulness", "callousness",
    "formaliti", "sensitiviti", "sensibiliti", "triplicate", "formative", "formalize",
    "electriciti", "electrical", "hopeful", "goodness", "revival", "allowance",
    "inference", "airliner", "gyroscopic", "adjustable", "defensible", "irritant",
    "replacement", "adjustment", "dependent", "adoption", "homologou", "communism",
    "activate", "angulariti", "homologous", "effective", "bowdlerize",
    # step 5 — final e, and double-l
    "probate", "rate", "cease", "controll", "roll",
    # retrieval-shaped vocabulary, where a divergence would actually cost recall
    "quickly", "searching", "searches", "searched", "indexing", "indexes", "encryption",
    "encrypted", "documents", "documented", "retrieval", "retrieving", "ranking", "ranked",
    "similarity", "similarities", "propagation", "propagating", "attenuated", "spectral",
]


@pytest.fixture(scope="module")
def stems():
    """Both stemmers over the same list. FTS5's is batched — one round-trip, then cached."""
    fts = _Stemmer().stem_many(_WORDS)
    return {w: (porter_stem(w), fts.get(w)) for w in _WORDS}


def test_the_word_list_actually_reaches_both_stemmers(stems):
    """Control. A batch that silently returned nothing would make every comparison below vacuous."""
    missing = [w for w, (_, f) in stems.items() if f is None]
    assert not missing, (
        f"FTS5 returned no stem for {len(missing)} words ({missing[:8]}) — the scratch-table "
        "round-trip did not run, so the comparison proves nothing."
    )
    # And the stemmers must actually be stemming, not echoing.
    changed = [w for w, (m, f) in stems.items() if m != w or f != w]
    assert len(changed) > len(_WORDS) // 3, (
        "almost nothing was stemmed by either implementation — one of them is a pass-through and "
        "this test would agree with itself about doing nothing."
    )


def test_the_hand_written_porter_agrees_with_fts5s(stems):
    """The conformance assertion. Exact string equality, no normalisation."""
    disagree = {w: (m, f) for w, (m, f) in stems.items() if m != f}
    assert not disagree, (
        f"\nthe two Porter stemmers disagree on {len(disagree)} of {len(_WORDS)} words:\n\n"
        + "\n".join(f"    {w:>16} : mantle={m!r}  fts5={f!r}" for w, (m, f) in sorted(disagree.items()))
        + "\n\nWhere both stemmers touch one corpus this does not raise — it silently changes "
          "which documents match, because a term stemmed one way at index time and the other at "
          "query time simply misses.\n\n"
          "This is CONFORMANCE, not correctness: it cannot say which is right. The question to "
          "answer first is which index is already written in which stem."
    )


def test_the_comparison_would_notice_a_divergence(stems):
    """Negative control: prove the assertion above is capable of failing.

    Without this, a fixture that produced two references to the same dict would make the
    conformance test unconditionally green.
    """
    m, f = stems["happy"]
    assert m == f, "precondition: the two agree on 'happy'"
    assert porter_stem("happy") != "happy", (
        "'happy' was not stemmed at all, so this control cannot demonstrate the comparison works"
    )
    # A deliberately wrong pair must be detected by the same expression the real test uses.
    tampered = {"happy": (m, f + "X")}
    assert {w: v for w, v in tampered.items() if v[0] != v[1]}, \
        "the disagreement expression does not detect a differing pair"
