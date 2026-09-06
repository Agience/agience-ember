"""How much a word narrows this corpus, in bits.

The one definition, shared by `activation.seeds_from_text` and `match.fired_field`. Bits because
that is the unit the rest of the instrument reports information in: the expectation of this
quantity is the Shannon entropy `entroptics.entropy` computes, so a surprisal in bits composes with
the entropy reads.

## What it is

    I(w) = -log2(df(w) / N)

the self-information of "a document contains w", read from EXACT document frequency off
`fts5vocab` — no scan, no cap, no estimate. A word in every document carries 0 bits; a word in one
document out of a million carries ~20.

## Unmeasurable is None, never a substituted number

`None` means the corpus could not measure this word: no index, no counter row, a store fault, or a
word the vocabulary has never seen. Every caller is required to handle it by leaving its weight
alone rather than by substituting a default, because a fabricated information value is
indistinguishable from a measured one at the call site and there is no way to tell afterwards
which words were guessed.

A `df` of 0 is a measurement rather than an absence of one: it means the word is not in the
corpus. `sage.content_search` depends on exactly that distinction to name an absent token rather
than retrieve a filler self-hit. This function returns `None` for it anyway, because "absent" has
no finite surprisal (`-log2(0)` diverges) and a caller wanting the absence should ask `_df`
directly, which is what `content_search` does.

## Compounds

A compound ("albert einstein") is not in the term vocabulary, so it has no df of its own. Its
information is that of its most informative word — the one that does the narrowing. Taking the
mean would let a common half dilute a rare half, and "the einstein" would read as less informative
than "einstein".
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

__all__ = ["word_information", "BITS"]

#: The unit every value from this module is in. Stated as a constant so a caller storing or
#: comparing these numbers can assert what it holds rather than assume it.
BITS = "bits"


def word_information(store) -> Optional[Callable[[str], Optional[float]]]:
    """`word -> surprisal in bits`, or `None` if this corpus cannot be measured at all.

    Returns a CALLABLE rather than a mapping because the caller does not know its own word set in
    advance — a need is tokenized, morphed and compound-matched on the way in — and because the
    result is cached per call site: a turn asks for a handful of words and the same word repeats
    across turns.

    A `None` return (as opposed to a callable that returns `None` per word) means there is no index
    or no counter row, so NOTHING can be weighted by information here. Callers keep their
    unweighted priors in that case; see `activation.seeds_from_text`.
    """
    try:
        conn = store.artifacts.db.read()
        from ember.ontology import corpus_stats as _cs
        total = float(_cs._corpus_rows(conn))
        if total <= 0.0:
            return None
    except Exception:
        return None                      # no index to measure against — the caller keeps its prior

    # Imported here rather than at module scope: `ember.optics` is the single sanctioned entroptics
    # door, and importing it eagerly would pull entroptics into every process that merely imports
    # the ontology. `ember/tests/test_ember_holds_the_instrument.py` asserts that laziness.
    from ember.optics import self_information_bits

    cache: Dict[str, Any] = {}

    def info(word: str) -> Optional[float]:
        if word in cache:
            return cache[word]
        best = None
        for part in str(word).split():
            try:
                df = _cs._df(conn, part)
            except Exception:
                df = None
            b = self_information_bits(df, total) if df else None
            if b is not None and (best is None or b > best):
                best = b
        cache[word] = best
        return best

    return info
