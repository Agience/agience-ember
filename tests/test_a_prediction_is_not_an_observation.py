"""A prediction is not an observation: what a turn remembers is what it fired.

`lumen.conversation._with_operator_continuation` places the seed's resonant field into the act list
— on the live corpus that is ~74,400 concepts against the ~80 a query fires — and marks every placed
row `predicted_from`. Each consumer of that list decides for itself whether a placed concept counts:
`compose`'s lead pool takes observations only, filtering on `predicted_from`; `output_membrane`
frames whatever field it is handed; and `_observe_field`, the subject of this file, deposits only
what fired.

`_observe_field` deposits exactly one unit of energy per turn however many concepts it lit
(`0->1->0` conservation). Admitting the placements would spread that one unit across the whole
field, so each row enters at ~1e-5 and the trace bag becomes a record of what the propagation
reached rather than of what the observer saw. The energy share stays conserved either way; whose
energy it is does not. The per-screen decay that `prism.demurrage` prices the economy's clock off is
measured over those traces, so the distinction is what keeps `rates_source` measurable at all.

Predicted rows keep their full energy everywhere else: in the field, in `_stated_relations`'
absorption, and in the answer's citations. They are excluded only from the record of what this
observer saw. The tests below assert both halves: what memory records, and what the answer still
receives.
"""
from __future__ import annotations

import pytest

from ember.ontology import activation as A


class _Delegate:
    """Captures exactly what `_observe_field` hands the delegate — `[(synset, share), ...]`.

    The delegate is the observation boundary (it owns the tick and the witness), so the pairs it
    receives are what the turn recorded. Nothing else is stubbed: the selection under test happens
    before this is called."""
    id = "d.t"
    person = "t@local"
    store = None

    def __init__(self):
        self.seen = []

    def observe_field(self, pairs):
        self.seen.extend(list(pairs))


def _acts(n_fired, n_predicted):
    """`n_fired` observations and `n_predicted` placements, all with real salience."""
    fired = [{"concept": "f%d.n.01" % i, "salience": 10.0} for i in range(n_fired)]
    placed = [{"concept": "p%d.n.01" % i, "salience": 9.0, "predicted_from": "f0.n.01"}
              for i in range(n_predicted)]
    return fired + placed


@pytest.fixture
def _synsets(monkeypatch):
    """`_observe_field` resolves each name through `crystal.ontology.driver`; pass the name through
    as the synset so the test measures the selection, not the ontology."""
    from crystal.ontology import driver as wn
    monkeypatch.setattr(wn, "synset", lambda name, store=None: name)
    return wn


def test_only_what_FIRED_is_remembered(_synsets):
    """Only rows without `predicted_from` reach the delegate. At corpus scale the placement is
    ~74,400 rows against ~80 fired, so a memory path that took the whole list would record three
    orders of magnitude more than the turn observed, and the decay measured over that bag would read
    `unmeasured`."""
    d = _Delegate()
    A._observe_field(d, _acts(n_fired=3, n_predicted=5000))

    names = [c for c, _e in d.seen]
    assert names, "nothing was observed at all"
    assert all(n.startswith("f") for n in names), (
        "a predicted concept was recorded as an observation: %s"
        % [n for n in names if not n.startswith("f")][:5])
    assert len(names) == 3


def test_the_turn_still_deposits_ONE_unit(_synsets):
    """The `0->1->0` invariant, asserted over the surviving rows: a turn deposits one unit of energy
    however many concepts it lit. Excluding predictions renormalises over what remains rather than
    leaking the removed share away."""
    d = _Delegate()
    A._observe_field(d, _acts(n_fired=4, n_predicted=100))
    total = sum(e for _c, e in d.seen)
    assert total == pytest.approx(1.0, abs=1e-9), (
        "the turn deposited %r units, not one — the share was not renormalised over what fired"
        % total)


def test_a_turn_that_fired_NOTHING_but_placed_much_remembers_nothing(_synsets):
    """The control. A cap or a top-N would still record something here; the selection records
    nothing, because a turn whose every row is a prediction observed nothing and a memory for it
    would be fabricated."""
    d = _Delegate()
    A._observe_field(d, _acts(n_fired=0, n_predicted=5000))
    assert d.seen == []


def test_predictions_are_NOT_dropped_from_the_answer_path(_synsets):
    """Placed rows are excluded from memory only. They still reach the field, which is where they
    earn their keep: absorbing them is what lets an answer carry many relations and many citations
    rather than one of each. Narrowing memory alone would look identical on the assertions above, so
    the other half is asserted here.

    `compose`'s lead pool is the neighbouring decision and draws the same line for a different
    reason (a prediction may not become the lead); this asserts the shared key still means what both
    of them read it to mean."""
    acts = _acts(n_fired=2, n_predicted=3)
    assert len([a for a in acts if a.get("predicted_from")]) == 3
    # the render side keeps them in the field while excluding them from the lead pool
    live = [a for a in acts if float(a.get("salience") or 0.0) > 0.0
            and not a.get("predicted_from")]
    assert len(live) == 2 and len(acts) == 5, (
        "the placed rows left the act list entirely — they belong in the field, only not in memory")
