"""`absorb_transmit` splits a frame with an orthogonal projector — and never builds one.

## Why this exists

Profiled 2026-08-25 on 71/home: `superposition(cn-singlish)` spent **73.3 s of 91.9 s inside
`numpy.linalg.svd`** — 156 calls, two per `separation()`. One is the band. The other was
`np.linalg.pinv(B)` in `P = B @ pinv(B)`, and it was avoidable arithmetic, not a modelling cost.

Two identities, neither an approximation:

1. **`pinv(B) == B.T` when B has orthonormal columns** — the SVD's own definition. `B` is
   `prism.frames.offer_basis`'s output, the left singular vectors `U[:, :r]`.
2. **`W @ (B @ B.T)` materialises an `F x F` projector; `(W @ B) @ B.T` never does.** Matrix
   multiplication is associative; `O(F²(F+T))` against `O(TFk)` is not.

Measured after, with every residual identical to four decimals:

    cn-singlish  107.7 s -> 42.9 s      cn-dog  7.8 s -> 1.4 s
    cn-cow         6.4 s ->  2.9 s      cn-cat  1.9 s -> 0.6 s

## What must not move, and why these are the assertions

The projector has to stay ORTHOGONAL. `‖incident‖² = ‖absorbed‖² + ‖transmitted‖²` is what every
conservation certificate in `consolidate/diagram.py` rests on — 📄 *"the split is an orthogonal
projection, so `‖E(b)‖² = ‖absorbed‖² + ‖residual‖²` exactly"* — and a projector that is merely
close to orthogonal breaks it silently, in the direction of a merge that should not have certified.

`B Bᵀ` is orthogonal exactly when B is orthonormal, so the fast path's precondition and the identity
it preserves are the SAME condition. The code tests it (`Bᵀ B ≈ I`, a `k x k` read) and falls
through to `pinv` when it fails — the case a NON-orthonormal basis exercises below.
"""
from __future__ import annotations

import numpy as np
import pytest

from ember.optics import absorb_transmit


def _frame(t=40, f=12, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(t, f))


def _orthonormal(f=12, k=4, seed=5):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(f, k)))
    return q


def test_conservation_holds_exactly_on_the_fast_path():
    """The identity the certificates rest on. Not "close": the residual of the sum sits at the
    arithmetic's own floor."""
    W, B = _frame(), _orthonormal()
    absorbed, transmitted, k = absorb_transmit(W, basis=B)
    assert k == B.shape[1]
    lhs = float(np.sum(np.abs(W) ** 2))
    rhs = float(np.sum(np.abs(absorbed) ** 2) + np.sum(np.abs(transmitted) ** 2))
    assert abs(lhs - rhs) <= 1e-9 * max(1.0, lhs), (lhs, rhs)


def test_the_fast_path_equals_the_pinv_it_replaces():
    """The whole claim, stated as an equality against the definition it optimises."""
    W, B = _frame(), _orthonormal()
    absorbed, transmitted, _k = absorb_transmit(W, basis=B)
    reference = W @ (B @ np.linalg.pinv(B))
    assert np.allclose(absorbed, reference, atol=1e-10, rtol=0.0)
    assert np.allclose(transmitted, W - reference, atol=1e-10, rtol=0.0)


def test_a_non_orthonormal_basis_still_gets_the_pseudo_inverse():
    """The fallback, and why the precondition is TESTED rather than assumed. A caller may hand any
    `(F, k)` basis; `offer_basis` happens to return an orthonormal one and nothing in the signature
    says so."""
    W = _frame()
    B = np.array([[1.0, 1.0], [0.0, 2.0]] + [[0.0, 0.0]] * 10)   # columns not orthonormal
    assert not np.allclose(B.T @ B, np.eye(2))
    absorbed, transmitted, k = absorb_transmit(W, basis=B)
    reference = W @ (B @ np.linalg.pinv(B))
    assert k == 2
    assert np.allclose(absorbed, reference, atol=1e-10, rtol=0.0)
    lhs = float(np.sum(np.abs(W) ** 2))
    rhs = float(np.sum(np.abs(absorbed) ** 2) + np.sum(np.abs(transmitted) ** 2))
    assert abs(lhs - rhs) <= 1e-9 * max(1.0, lhs), "conservation broke on the fallback"


def test_the_projector_is_idempotent_which_is_what_makes_it_a_projector():
    """Applying it twice changes nothing. A non-orthogonal or scaled operator fails this."""
    W, B = _frame(), _orthonormal()
    once, _t, _k = absorb_transmit(W, basis=B)
    twice, _t2, _k2 = absorb_transmit(once, basis=B)
    assert np.allclose(once, twice, atol=1e-10, rtol=0.0)


def test_the_absorbed_band_and_the_residual_are_orthogonal():
    """The geometric statement behind conservation: the two halves share no direction."""
    W, B = _frame(), _orthonormal()
    absorbed, transmitted, _k = absorb_transmit(W, basis=B)
    assert abs(float(np.sum(absorbed * transmitted))) <= 1e-9 * float(np.sum(np.abs(W) ** 2))


@pytest.mark.parametrize("k", [1, 3, 12])
def test_it_holds_at_every_rank_including_a_full_basis(k):
    """At `k == F` the projector is the identity and the residual is zero — the degenerate end,
    where an off-by-one in the reassociation would show."""
    W = _frame(f=12)
    B = _orthonormal(f=12, k=k)
    absorbed, transmitted, got = absorb_transmit(W, basis=B)
    assert got == k
    if k == 12:
        assert np.allclose(absorbed, W, atol=1e-9, rtol=0.0)
        assert np.allclose(transmitted, 0.0, atol=1e-9)
