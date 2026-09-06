"""`sense_prior` and `fired_field` can actually reach WordNet. Neither could.

THE DEFECT. `ember/ontology/activation.py` and `match.py` import `crystal.ontology.driver as wn`
inside SOME function bodies — deliberately function-local, because `match` reaches `activation` the
same way and one eager import in either direction closes the cycle. But `sense_prior` and
`fired_field` never had one, so `wn` was simply not in their namespace.

Both uses sit inside `try: … except Exception:`, and neither handler is a bare `pass` — they
substitute a WORSE ANSWER:

  * `sense_prior` fell to `1.0 / (1.0 + k)`, the positional fallback, for EVERY word;
  * `fired_field` fell to the unmorphed token, so "dogs" never became "dog".

So there was no crash and no log — just a permanently degraded answer under a docstring describing
the good one. Measured before the fix: `sense_prior(['dog.n.01','dog.n.03','frump.n.01'], 'dog')`
returned exactly `[1.0, 0.5, 0.3333]`, identical to the fallback. With WordNet reachable it returns
the SemCor counts — a different ranking, not a scaled one.

That matters beyond tidiness: this function's own docstring says the prior is what makes a bare
"dog" resolve to the animal rather than to `frump`. It had never once applied.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from crystal.ontology import driver
from ember.ontology import activation, match


class _Lemma:
    def __init__(self, count: int, name: str = "dog"):
        self._count, self._name = count, name

    def count(self) -> int:
        return self._count

    def name(self) -> str:
        return self._name


class _Synset:
    def __init__(self, count: int):
        self._count = count

    def lemmas(self):
        return [_Lemma(self._count)]


def test_wn_is_reachable_from_both_functions():
    """The bug as a one-line assertion, in both places it lived.

    A name that is not in scope, used inside a handler that substitutes a fallback, produces no
    error and no log — only a quietly worse answer. That is why this is asserted directly rather
    than inferred from behaviour."""
    for fn in (activation.sense_prior, match.fired_field):
        src = __import__("inspect").getsource(fn)
        assert " as wn" in src, (
            "%s has no local `import … as wn`; `wn` is not in its namespace and every use will "
            "fall to the degraded branch, silently" % fn.__name__)


def test_the_prior_is_the_semcor_count_not_the_position():
    counts = {"dog.n.01": 40, "dog.n.03": 2, "frump.n.01": 1}
    with patch.object(driver, "synset", side_effect=lambda n: _Synset(counts[n])):
        got = activation.sense_prior(list(counts), "dog")

    positional = [1.0 / (1.0 + k) for k in range(len(counts))]
    assert got != pytest.approx(positional), (
        "the prior is still the positional fallback %r — WordNet is not being reached" % positional)
    # The commonest sense must dominate by the COUNT ratio, not by its position in the list.
    assert got[0] > got[1] > got[2]
    assert got[0] / got[1] > 10, (
        "40 vs 2 SemCor counts should separate these sharply; got %r — this is the difference "
        "between ranking by usage and ranking by list order" % got)


def test_the_fallback_still_applies_when_wordnet_genuinely_cannot_answer():
    """The inverted guard. The fallback is CORRECT when WordNet has no answer — the defect was that
    it applied always. If this failed, the fix would have removed a legitimate path."""
    with patch.object(driver, "synset", side_effect=KeyError("no such synset")):
        got = activation.sense_prior(["a.n.01", "b.n.01"], "a")
    assert got == pytest.approx([1.0, 0.5])


def test_a_missing_wordnet_does_not_raise_out_of_sense_prior():
    """Fail-soft is right and is kept: a ranking helper must not take down the caller."""
    with patch.object(driver, "synset", side_effect=RuntimeError("store not bound")):
        activation.sense_prior(["a.n.01"], "a")          # must not raise
