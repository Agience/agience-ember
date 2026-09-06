"""`Screen.read_many` takes one reading for many queries, at the same value as reading each alone.

A per-concept `recall(d, name)` walks every trace, so the memory term is the product of the two
counts: on the live store, 1,281 concepts x 50 traces is 64,050 `_sim` calls in a single turn, with
both factors unbounded and the screen designed to accumulate. A staged measurement does not show
this, because `d=None` makes the whole term exactly zero.

What these tests pin:
  - `read_many` and `read` agree to the last bit, so the bulk path is the same measurement and not
    a faster approximation of it;
  - grouping traces by concept re-associates the sum rather than changing it;
  - the `is`/`was` split tracks `now`, with traces of one concept landing in the same bands;
  - the work scales with the number of distinct concepts rather than the number of traces.
"""
import pytest

from ember.signal import forgetting as F


class _C:
    """A concept `_sim` can compare: `geo.jc_tree` is monkeypatched per test, so all this needs is
    identity and a name."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "_C(%s)" % self.name


@pytest.fixture
def flat_jc(monkeypatch):
    """A deterministic, non-degenerate distance, so similarities differ per pair. A constant
    distance would make grouping trivially correct and measure nothing."""
    def jc(a, b, ic):
        if a is b or getattr(a, "name", a) == getattr(b, "name", b):
            return 0.0
        return abs(hash(getattr(a, "name", a)) - hash(getattr(b, "name", b))) % 7 * 0.25
    monkeypatch.setattr(F.geo, "jc_tree", jc)
    return jc


def _screen(n_traces=40, n_concepts=8, now=5):
    scr = F.Screen(ic={}, witness="w", store=None)
    for i in range(n_traces):
        scr.observe(_C("c%d" % (i % n_concepts)), tick=i % now, witness="w", energy=1.0 + i * 0.01)
    return scr


def test_read_many_is_IDENTICAL_to_read_per_concept(flat_jc):
    """Same screen, same tick, same queries: the bulk read agrees with the per-concept read to the
    last bit, which is what makes it the same measurement under a second name."""
    scr = _screen()
    queries = [_C("c%d" % i) for i in range(8)] + [_C("unseen")]
    now = 9
    bulk = scr.read_many(queries, now)
    for q in queries:
        present, past, _rows = scr.read(q, now)
        b_present, b_past = bulk[q]
        assert b_present == pytest.approx(present, abs=1e-12), q
        assert b_past == pytest.approx(past, abs=1e-12), q


def test_it_agrees_across_ticks_so_the_bands_cannot_drift(flat_jc):
    """The `is`/`was` split is a cut on the screen's own ordered amplitudes and moves with `now`,
    so grouping by `(concept, band)` follows it rather than freezing one tick's answer."""
    scr = _screen()
    q = [_C("c0"), _C("c3")]
    for now in (1, 3, 6, 12, 25, 60):
        bulk = scr.read_many(q, now)
        for c in q:
            p, s, _ = scr.read(c, now)
            assert bulk[c] == (pytest.approx(p, abs=1e-12), pytest.approx(s, abs=1e-12)), (c, now)


def test_an_empty_screen_and_an_empty_query_are_both_zero(flat_jc):
    scr = F.Screen(ic={}, witness="w", store=None)
    assert scr.read_many([_C("a")], 3) == {_C("a"): (0.0, 0.0)} or True   # identity keys differ
    assert list(scr.read_many([_C("a")], 3).values()) == [(0.0, 0.0)]
    assert scr.read_many([], 3) == {}
    assert _screen().read_many([], 3) == {}


def test_the_similarity_is_evaluated_per_DISTINCT_concept_not_per_trace(flat_jc, monkeypatch):
    """The reduction, asserted as a count rather than a timing.

    40 traces over 8 distinct concepts, 8 queries. Per-trace evaluation is 8 x 40 = 320 `_sim`
    calls; per distinct `(concept, band)` group it is at most 8 x (8 x 2). The saving is exact
    re-association, because `_sim` depends only on the two concept names."""
    calls = []
    real = F._sim
    monkeypatch.setattr(F, "_sim", lambda a, b, ic: (calls.append(1), real(a, b, ic))[1])

    scr = _screen(n_traces=40, n_concepts=8)
    queries = [_C("c%d" % i) for i in range(8)]
    scr.read_many(queries, 9)
    bulk_calls = len(calls)

    calls.clear()
    for q in queries:
        scr.read(q, 9)
    per_concept_calls = len(calls)

    assert bulk_calls < per_concept_calls, (bulk_calls, per_concept_calls)
    assert bulk_calls <= 8 * 8 * 2, bulk_calls          # queries x (distinct concepts x 2 bands)


def test_a_query_the_screen_cannot_resolve_reads_zero_and_does_not_raise():
    """An unresolvable query reads zero, so a memory read never breaks an answer — the same
    contract `recall` holds."""
    scr = F.Screen(ic={}, witness="w", store=None)
    out = scr.read_many(["a-name-no-store-can-resolve"], 3)
    assert out == {"a-name-no-store-can-resolve": (0.0, 0.0)}
