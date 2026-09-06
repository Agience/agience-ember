"""The canonical pick carries no provenance ladder.

`consolidate_nearvdup` groups by `content_ref`, so every member of a group is byte-identical. There
is no validity question between identical copies for a rank to answer, so `genesis._pick_canonical`
returns the lowest id: arbitrary, deterministic, and stable across nodes.

A rank of the shape

    rung_rank = {P_HUMAN: 5, P_OBSERVED: 4, P_SPAN_CITED: 3, P_HYPOTHESIS: 2, P_ASSERTION: 0}
    return max(group, key=lambda a: (rung_rank.get(a.get("provenance"), 1), ...))

orders a difference that does not exist, and its `.get` default puts an absent provenance at 1,
above an explicit ASSERTION at 0. These tests pin both halves: the runtime pick, permuting the
labels rather than reading the source, and a static check that no such dict literal sits in
`genesis.py` waiting to be wired back in.
"""
from __future__ import annotations

import ast
import pathlib

from ember import genesis

GENESIS_SRC = pathlib.Path(genesis.__file__)


def _member(aid, provenance=None, lemmas=None):
    a = {"id": aid, "content_ref": "sha:same-bytes"}
    if provenance is not None:
        a["provenance"] = provenance
    if lemmas is not None:
        a["lemmas"] = lemmas
    return a


def test_the_pick_ignores_PROVENANCE_entirely():
    """The same group, with the provenance labels moved around, gives the same representative.

    Asserted by permuting the labels rather than by reading the source, so it catches a rank keyed
    on `provenance` however that rank is spelled.
    """
    ids = ["art.c", "art.a", "art.b"]
    plain = genesis._pick_canonical([_member(i) for i in ids])
    for labels in ((genesis.P_HUMAN, genesis.P_ASSERTION, genesis.P_OBSERVED),
                   (genesis.P_ASSERTION, genesis.P_HUMAN, genesis.P_HYPOTHESIS),
                   (genesis.P_SPAN_CITED, genesis.P_SPAN_CITED, genesis.P_HUMAN)):
        got = genesis._pick_canonical([_member(i, p) for i, p in zip(ids, labels)])
        assert got["id"] == plain["id"], (
            "provenance %s changed the canonical to %s — a ladder is back" % (labels, got["id"]))


def test_ABSENT_provenance_does_not_outrank_a_STATED_one():
    """An absent provenance carries no standing over a stated one. `rung_rank.get(provenance, 1)`
    scores an artifact with no provenance at 1, above an explicit ASSERTION at 0, so absence
    outranks a stated claim — the shape of any `.get(..., default)` rank whose default sits above
    the bottom.
    """
    stated = _member("art.a", genesis.P_ASSERTION)
    absent = _member("art.b")
    assert genesis._pick_canonical([stated, absent])["id"] == "art.a"   # lowest id, not the absence
    assert genesis._pick_canonical([absent, stated])["id"] == "art.a"   # and order-independent


def test_the_pick_is_DETERMINISTIC_and_order_independent():
    """Two nodes consolidating the same class choose the same representative, so the mesh agrees on
    which id is head. The lowest id promises this; a metadata-dependent rank cannot, because `max`
    over ties returns whichever maximum came first in the input.
    """
    members = [_member("art.b"), _member("art.a"), _member("art.c")]
    first = genesis._pick_canonical(members)["id"]
    assert first == "art.a"
    for perm in ([members[2], members[0], members[1]], list(reversed(members))):
        assert genesis._pick_canonical(perm)["id"] == first


def test_other_METADATA_cannot_move_the_pick_either():
    """Byte-identical members differing only in how much metadata they carry pick the same
    representative. A rank on `len(lemmas)` or on id length is the same ordering-of-nothing under
    another name.
    """
    a = _member("art.a", lemmas=[])
    b = _member("art.b", lemmas=["x", "y", "z", "w"])
    assert genesis._pick_canonical([a, b])["id"] == "art.a"


def test_NO_rung_rank_LADDER_REMAINS_IN_THE_SOURCE():
    """The static half: no dict literal maps provenance constants to integers anywhere in
    `genesis.py`. The runtime tests above pass on a ladder that is built and then unused, so this
    catches one sitting there waiting to be wired back in.

    `SEED_RUNGS` is outside the pattern by construction — it is a vocabulary registry of provenance
    names and their meanings, which is a different thing from an order over them.
    """
    tree = ast.parse(GENESIS_SRC.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or not node.keys:
            continue
        keys = [k for k in node.keys if isinstance(k, ast.Name) and k.id.startswith("P_")]
        vals = [v for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, int)]
        if len(keys) >= 2 and len(vals) >= 2:
            offenders.append(node.lineno)
    assert not offenders, (
        "a provenance->integer ladder is back in genesis.py at line(s) %s. Byte-identical members "
        "have no validity ordering; the canonical is the lowest id, arbitrarily and openly."
        % offenders)
