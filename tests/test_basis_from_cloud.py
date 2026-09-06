"""`screen._basis_from_cloud` — the corpus basis derivation. The count and the directions come from
a single correlation read through `ember.optics.principal_directions`, so the resolved rank and the
basis it returns always agree. Verified on synthetic clouds because the real corpus is a
consolidation-phase input this dev environment does not carry."""
import numpy as np
import pytest

from ember.signal.projection import _basis_from_cloud


def _cloud(n, D, k, *, dead=0, seed=0):
    """A tall (n, D) cloud with a planted k-dim structure, optionally with `dead` all-zero columns."""
    rng = np.random.RandomState(seed)
    live = D - dead
    Bt = np.linalg.qr(rng.randn(live, k))[0]              # k orthonormal directions in the live block
    amp = rng.randn(n, k) * np.array([6.0, 5.0, 4.0, 3.0, 2.0])[:k]
    X = amp @ Bt.T + 0.25 * rng.randn(n, live)
    M = np.zeros((n, D))
    M[:, :live] = X                                       # trailing `dead` columns stay structurally 0
    return M


def test_basis_shape_and_count_come_from_the_instrument():
    M = _cloud(400, 16, 3)
    B, k, rd, src = _basis_from_cloud(M)
    assert src == "principal_directions"                 # the instrument answered, not the fallback
    assert B.shape == (k, 16)
    assert k == 3                                         # the planted rank resolves
    # count and directions are the same read: k is the number of basis directions and the resolved count
    from ember.optics import principal_directions
    assert principal_directions(M).shape[1] == k


def test_dead_columns_are_zero_rows_in_the_basis():
    # A hashed coordinate leaves columns structurally empty; the basis must scatter back to full D
    # with exactly zero weight on the dead channels (they carried no data to have a direction in).
    D, dead = 20, 6
    M = _cloud(400, D, 3, dead=dead)
    B, k, rd, src = _basis_from_cloud(M)
    assert B.shape[1] == D
    assert np.allclose(B[:, D - dead:], 0.0)             # dead columns carry no direction
    assert not np.allclose(B[:, :D - dead], 0.0)         # live columns do


def test_basis_rows_are_orthonormal():
    M = _cloud(500, 24, 4)
    B, k, rd, src = _basis_from_cloud(M)
    live = np.nonzero(M.any(axis=0))[0]
    G = B[:, live] @ B[:, live].T                        # (k, k) Gram over the live block
    assert np.allclose(G, np.eye(k), atol=1e-6)


def test_noise_only_falls_back_and_still_returns_a_basis():
    # Nothing resolves above the floor: the job still produces a basis, through the SVD fallback.
    rng = np.random.RandomState(1)
    M = rng.randn(40, 8) * 1.0
    B, k, rd, src = _basis_from_cloud(M)
    assert src == "svd-fallback"
    assert B.shape == (k, 8) and k >= 1
