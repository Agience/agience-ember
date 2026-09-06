"""Merkle anti-entropy properties — the guarantees the mesh's convergence rests on. When one of
them breaks, reconciliation stops finding differences without saying so, and nodes diverge
invisibly."""
# The import names `mantle.mesh` explicitly rather than using a relative import. This module lives
# in `tests/`, so `from . import merkle` would mean `tests.merkle` — a silent retarget onto a
# different module.
from __future__ import annotations

from mantle.mesh import merkle


def _rows(n, start=0, rev=1):
    return [("art-%d" % i, rev) for i in range(start, start + n)]


def test_identical_stores_have_equal_roots_and_no_diff():
    a = merkle.build(_rows(5000))
    b = merkle.build(_rows(5000))
    assert merkle.root(a) == merkle.root(b)
    assert merkle.diff(a, b) == []          # converged peers exchange nothing further


def test_order_independent():
    """Different nodes scan in different orders and still agree. XOR makes this hold without anyone
    sorting millions of rows."""
    rows = _rows(2000)
    assert merkle.build(rows) == merkle.build(list(reversed(rows)))


def test_one_missing_row_is_localized_to_one_leaf():
    full = merkle.build(_rows(5000))
    missing = merkle.build(_rows(4999))
    d = merkle.diff(full, missing)
    assert len(d) == 1                      # cost is proportional to the DIFFERENCE, not the corpus
    assert d[0] == merkle.leaf_of("art-4999")


def test_mutation_is_detected_not_just_insertion():
    """A row whose `_rev` changed (compaction, is->was decay) differs in the tree. Catching only
    insertions would leave in-place mutations to never propagate — the gap the updates feed
    covers."""
    before = merkle.build([("a", 1), ("b", 1)])
    after = merkle.build([("a", 2), ("b", 1)])
    assert before != after
    assert merkle.diff(before, after) == [merkle.leaf_of("a")]


def test_absent_rev_is_hashed_not_skipped():
    """A row with no `_rev` hashes as if it carried 0, so rows written before `_rev` existed stay
    comparable. Skipping them would make that part of the corpus invisible to the change feed."""
    assert merkle.row_hash("a", None) == merkle.row_hash("a", 0)
    assert merkle.build([("a", None)]) != [0] * merkle.DEFAULT_LEAVES


def test_leaf_distribution_is_even_despite_skewed_ids():
    """Real ids are wildly non-uniform (wn-*, wiki titles, op.*), and hashing spreads them anyway.
    A skewed distribution puts most of the corpus in a few leaves, and localization stops paying."""
    leaves = 256
    ids = (["wn-%d" % i for i in range(3000)] + ["op.source.thing-%d" % i for i in range(300)]
           + ["Wikipedia Article Title %d" % i for i in range(3000)])
    counts = [0] * leaves
    for i in ids:
        counts[merkle.leaf_of(i, leaves)] += 1
    avg = len(ids) / leaves
    assert min(counts) > avg * 0.4 and max(counts) < avg * 2.0
    assert all(c > 0 for c in counts)


def test_divergence_then_repair_converges():
    """The end-to-end property: apply only what diff() reports and the roots must match. This is
    recovery and steady state being the same operation."""
    theirs = {("art-%d" % i): 1 for i in range(3000)}
    mine = {k: v for k, v in theirs.items() if k != "art-42"}
    mine["art-7"] = 9                                   # one missing row + one stale row
    build = lambda d: merkle.build(list(d.items()))
    d = merkle.diff(build(mine), build(theirs))
    assert set(d) == {merkle.leaf_of("art-42"), merkle.leaf_of("art-7")}
    for aid, rev in theirs.items():                     # repair ONLY the differing leaves
        if merkle.leaf_of(aid) in set(d):
            mine[aid] = rev
    assert merkle.root(build(mine)) == merkle.root(build(theirs))


def test_summary_round_trips():
    a = merkle.build(_rows(500))
    assert merkle.load(merkle.summary(a)) == a
    assert merkle.load({"leaves": 4, "digests": [1, 2]}) is None       # malformed -> None, not junk


def test_store_leaf_matches_merkle_leaf():
    """The store stamps `_leaf` using its own copy of `leaf_of`, since it does not import the mesh
    package. If the two drift — different hash, different modulus — `_leaf` points at leaves the
    tree does not have, and reconciliation compares nonsense while looking healthy. So the two
    copies are pinned to each other over a spread of real id shapes."""
    from mantle.db.constants import DEFAULT_LEAVES, leaf_of as store_leaf
    assert DEFAULT_LEAVES == merkle.DEFAULT_LEAVES
    for aid in ("wn-dog-n-01", "Wikipedia Article", "op.source.wikipedia-en", "art-1", "x" * 200):
        assert store_leaf(aid) == merkle.leaf_of(aid), aid


def test_natural_leaves_is_derived_sqrt_and_store_matches():
    """The leaf count is derived rather than typed: L = sqrt(N) rounded to 2^k, the point where
    publishing the L-leaf tree and pulling one differing leaf cost the same. It grows with the
    corpus, and the store and mesh copies agree byte for byte — a disagreement puts `_leaf` and the
    tree on different resolutions."""
    from mantle.db.constants import natural_leaves as store_nl
    import math
    for n in (1, 10, 1000, 6_000_000, 600_000_000):
        L = merkle.natural_leaves(n)
        assert L == store_nl(n), n                       # store == mesh
        assert (L & (L - 1)) == 0, ("power of two", n, L)
        assert L == 1 << max(0, round(math.log2(max(1, n)) / 2.0))
    assert merkle.natural_leaves(600_000_000) > merkle.natural_leaves(6_000_000)   # grows with corpus


def test_fold_is_exact_and_order_free():
    rows = _rows(500)
    fine = merkle.build(rows, 1 << 10)
    # folding a 2^10 tree down to 2^7 == building at 2^7 directly (leaf_of nests; XOR is order-free)
    assert merkle.fold(fine, 1 << 7) == merkle.build(rows, 1 << 7)
    assert merkle.fold(fine, 3) is None                  # not a clean power-of-two divisor


def test_different_resolutions_converge_and_localize():
    """A node talks to a peer without re-sharding. Two nodes over the same rows at different derived
    resolutions show an empty diff (converged), and one changed row still localizes to one leaf at
    the common, coarser resolution."""
    rows = _rows(500)
    for ka, kb in ((11, 8), (8, 11), (4, 9)):
        assert merkle.diff(merkle.build(rows, 1 << ka), merkle.build(rows, 1 << kb)) == []
    A, B = merkle.build(rows, 1 << 10), merkle.build(rows[:-1], 1 << 8)   # B missing one row, coarser
    assert len(merkle.diff(A, B)) == 1


# The cursor rule — a consume cursor advances only past a segment that applied — belongs to the S3
# segment-log plane and is pinned by the mesh tests on that plane. Everything in this file is about
# merkle itself, and every test in it runs: a permanently skipped test is indistinguishable from a
# check somebody disabled, and it trains the eye to ignore the column that carries the real ones.
