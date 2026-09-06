"""The provenance ladder and the citation anchor live in `prism.mass`; `grounding` imports them.

A rung string is a value already written into every stored row, so a second definition of it is a
second thing that can drift — and drift here means a leaf weighing artifacts differently from the
server.

What each test pins:

  * `test_the_rung_strings_are_not_written_in_ember` reads `grounding`'s source and looks for the
    rung literals and `"cite.genesis"`. Value equality cannot tell "imported" from "duplicated",
    because two copies compare equal, so the check is on where the value comes from.
  * `test_the_meaning_is_kept_exactly` compares byte for byte against the law spelled out below. A
    tidier spelling would re-rung the whole corpus.
  * `test_beam_is_where_the_law_lives` imports `CITE_GENESIS` from `prism.mass`, so it fails with
    `ImportError` if the anchor is anywhere else.
"""
from __future__ import annotations

import inspect
import re

from prism.mass import CITE_GENESIS as BEAM_CITE, Provenance
from prism import grounding


# The law, spelled out here so this test is an independent oracle rather than a mirror of the
# implementation: these are the byte values GENESIS §3.3 / prism.mass fixed and every stored row
# already carries. A suite that reads only the implementation agrees with whatever it says.
_LAW = {
    "P_HUMAN": "human_validated",
    "P_OBSERVED": "observed",
    "P_SPAN_CITED": "span_cited",
    "P_HYPOTHESIS": "hypothesis",
    "P_ASSERTION": "assertion",
}
_ANCHOR = "cite.genesis"


def test_beam_is_where_the_law_lives():
    """The rungs and the system-authorship anchor both live in `prism.mass`."""
    assert Provenance.HUMAN_VALIDATED.value == _LAW["P_HUMAN"]
    assert BEAM_CITE == _ANCHOR
    # `grounding` re-exports the same object, so a caller sees one value.
    assert grounding.CITE_GENESIS == BEAM_CITE


def test_the_meaning_is_kept_exactly():
    """Byte for byte. A rung is a value written into rows, so respelling it re-rungs the corpus."""
    for name, value in _LAW.items():
        assert getattr(grounding, name) == value, name
        # a plain `str`, not the Enum member: `%s`/`format()` on a str-Enum yields
        # "Provenance.HUMAN_VALIDATED" in 3.11+, and these values reach both.
        assert type(getattr(grounding, name)) is str, name


def test_the_rung_strings_are_not_written_in_ember():
    """The law reaches `grounding` by import, so its source carries none of the rung literals.

    Value equality cannot tell "imported" from "duplicated", since two copies compare equal. This
    reads the source instead, and asserts the import is present.
    """
    src = inspect.getsource(grounding)
    # strip the docstrings/comments' prose mentions: only actual string literals matter, so look for
    # each value in quotes.
    for value in list(_LAW.values()) + [_ANCHOR]:
        for quoted in ('"%s"' % value, "'%s'" % value):
            assert quoted not in src, (
                "ember/grounding.py restates the law as a literal %s — it must come from "
                "prism.mass" % quoted)
    assert re.search(r"^from prism\.mass import ", src, re.M), (
        "ember/grounding.py no longer imports the law from prism.mass")


def test_the_runner_keeps_only_what_is_a_runners():
    """What the runner's own module carries: the conversation triple type, the transducer op-id and
    the clock, alongside the re-exported law.

    The guard is on `__all__`, which is what a caller sees.
    """
    assert set(grounding.__all__) == {
        "P_HUMAN", "P_OBSERVED", "P_SPAN_CITED", "P_HYPOTHESIS", "P_ASSERTION",
        "CITE_GENESIS", "TRIPLE_TYPE", "TRANSDUCER_OP", "_now",
    }
    # `TRANSDUCER_OP` names a value stored in the database, so the constant and the rows carry it
    # together. Changing the constant alone leaves `crystal.ontology.geometry._keyed_ready()`
    # looking for an op id no store holds, and the keyed path falls back to the unkeyed one.
    assert grounding.TRANSDUCER_OP == "op.transducer."
    assert grounding.TRIPLE_TYPE == "application/vnd.agience.triple+json"
