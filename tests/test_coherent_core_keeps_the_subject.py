r"""A prose artifact's positions must agree with what it says it is about (§110).

`ranking._position` gives a prose candidate one position per key term it was indexed on. Measured on
the live shard that is a median of 26 positions whose mean pairwise distance is 88% of the corpus
diameter — the document is smeared across meaning-space rather than located in it.

`coherent_core` keeps the title's positions and those body positions within one correlation length
of them. `xi` is derived (`mu = ln(d_max/d_edge)`, `xi = 1/mu`), not chosen.

Two failures are pinned here because both actually happened:

  * **Inert by construction.** The first version guarded `if not ic: return wanted`, and
    `geometry.load_ic` is a NO-OP SENTINEL that always returns `None` — IC lives on the synset. The
    function could never do anything, in any process, and every bench returned byte-identical
    numbers. `TestItActuallyPrunes` fails if that returns.

  * **Densest neighbourhood selects the cardinal numbers.** Before anchoring, the core was the
    tightest cluster. Numbers are siblings under one parent, so they are maximally tight, they
    appear in every document's lemma bag, and they are about nothing. Spread improved by 61 points
    of diameter while selecting pure boilerplate. `TestTightnessIsNotAboutness` pins that.
"""
from __future__ import annotations

import pytest

from ember.ontology import match as M


class _Node:
    """A stand-in synset. `coherent_core` only needs identity and a name."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


@pytest.fixture()
def geometry(monkeypatch):
    """A tiny explicit metric, so the grouping rule is the only thing under test.

    Two clusters a long way apart, plus `six`/`two` — siblings at distance 0.01, the shape that
    made the unanchored version select the numbers.
    """
    at = {"host": 0.0, "server": 0.1, "authority": 0.15,
          "glacier": 1.4, "moraine": 1.45,
          "two": 3.0, "six": 3.01, "one": 3.02}

    def _jc(a, b, ic):
        return abs(at[a.name()] - at[b.name()])

    monkeypatch.setattr(M, "_synset_or_none", lambda n: _Node(n) if n in at else None)
    monkeypatch.setattr(M, "xi", lambda: 0.2)
    monkeypatch.setitem(
        __import__("sys").modules,
        "crystal.ontology.geometry",
        type("G", (), {"jc_tree": staticmethod(_jc), "load_ic": staticmethod(lambda *a: None)}),
    )
    M._CORE_CACHE.clear()
    yield
    M._CORE_CACHE.clear()


class TestItActuallyPrunes:

    def test_a_body_position_far_from_the_title_is_dropped(self, geometry):
        out = M.coherent_core(["host", "server", "glacier", "two"], anchor=["host"])
        assert set(out) == {"host", "server"}, (
            "either the far positions survived or the whole function is inert again — "
            "`load_ic()` returning None is NORMAL and must not short-circuit it; got %r" % (out,))

    def test_the_title_position_is_kept_even_with_nothing_near_it(self, geometry):
        out = M.coherent_core(["host", "glacier", "two", "six"], anchor=["host"])
        assert out == ["host"]

    def test_several_anchors_each_gather_their_own(self, geometry):
        """A document about two things is about two things — `colimit means overlap`."""
        out = M.coherent_core(["host", "server", "glacier", "moraine", "two"],
                              anchor=["host", "glacier"])
        assert set(out) == {"host", "server", "glacier", "moraine"}


class TestTightnessIsNotAboutness:

    def test_a_tight_cluster_of_siblings_does_not_win_on_its_own(self, geometry):
        """`two`/`six`/`one` sit 0.01 apart — far tighter than the subject cluster.

        Selecting by tightness returns them; anchoring on the title returns the subject. On the
        live corpus the unanchored version answered `1`, `2` and `6` for `canon:prism-protocol#1`.
        """
        out = M.coherent_core(["host", "server", "two", "six", "one"], anchor=["host"])
        assert "two" not in out and "six" not in out and "one" not in out, out
        assert set(out) == {"host", "server"}


class TestItDoesNotTouchWhatItCannotJudge:

    def test_no_anchor_returns_everything(self, geometry):
        """There is no coherence without something to be coherent WITH."""
        names = ["host", "glacier", "two"]
        assert M.coherent_core(names, anchor=None) == names
        assert M.coherent_core(names, anchor=[]) == names

    def test_a_synset_candidate_is_never_perturbed(self, geometry):
        """A synset has ONE position, itself. Fewer than three is returned unchanged."""
        assert M.coherent_core(["host"], anchor=["host"]) == ["host"]
        assert M.coherent_core(["host", "glacier"], anchor=["host"]) == ["host", "glacier"]

    def test_an_unmeasurable_geometry_keeps_everything(self, geometry, monkeypatch):
        monkeypatch.setattr(M, "xi", lambda: None)
        M._CORE_CACHE.clear()
        names = ["host", "server", "glacier", "two"]
        assert M.coherent_core(names, anchor=["host"]) == names

    def test_a_name_that_resolves_to_nothing_is_not_a_position(self, geometry):
        out = M.coherent_core(["host", "server", "not-a-synset"], anchor=["host"])
        assert "not-a-synset" not in out
