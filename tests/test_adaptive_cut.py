"""Tests for the model-free adaptive relevance cut (`adaptive_cut`).

Inputs are raw BM25 scores (negative; most-negative is best, sorted best-first) — the same thing
the serve path hands `_relevance_cut`. These run against the real entroptics engine through
`ember/optics.py`, the one seam onto it. The contract: with a real (T, F) frame the instrument
answers (`k_signal`); with only a score column there is no frame to read, and the cut defers.
"""
import ember  # noqa: F401  — a host declaration. `cut(frame=...)` resolves the injected
#              instrument, and importing `ember` is what registers `ember.optics` as the
#              process default (see `_register_instrument` in `ember/__init__.py`).
#              `prism.adaptive_cut` does not register it, so this file asks for it here
#              rather than depending on an earlier test having imported `ember`.

import prism.adaptive_cut as ac
from prism.adaptive_cut import cut


def test_WITHOUT_A_FRAME_IT_DEFERS_AND_THAT_IS_THE_CONTRACT():
    """A score column alone carries no frame, so the cut returns None and the caller keeps its
    baseline.

    A list of scores is one dimension. The instrument resolves modes across features, so it reads a
    (T, F) frame with F >= 2; handed a single column there is no reading to take. Reshaping the
    column into `entroptics.read` produces a number, but it answers "how many resolvable spots does
    a line have" — and it enters past the streaming front door `ember/optics.py` provides, applying
    the entropy fold guard that flattens a sparse carrier.

    None is the result here, and the serve path falls back on its baseline cut."""
    assert cut([-9.0, -1.0, -0.9, -0.8, -0.7]) is None
    assert cut([-5.0] * 8) is None


def test_WITH_A_REAL_FRAME_THE_INSTRUMENT_ANSWERS():
    """Given the (T, F) evidence behind the scores, the instrument reads `k_signal`: one planted mode
    reads as one, and noise resolves nothing, which keeps the whole pool."""
    import numpy as np
    rng = np.random.default_rng(0)
    planted = np.hstack([rng.normal(size=(200, 1)) @ np.ones((1, 8)),
                         0.1 * rng.normal(size=(200, 8))])
    scores = [-float(x) for x in np.linspace(9.0, 1.0, 200)]
    assert cut(scores, frame=planted) == 1
    noise = rng.normal(size=(200, 16))
    assert cut(scores, frame=noise) == 200          # nothing resolves: the whole pool is kept


def test_a_pool_too_small_to_carry_structure_defers():
    assert cut([-5.0, -1.0]) is None                # below _MIN_POOL
    assert cut([-5.0]) == 1


def test_mode_gate_is_unchanged():
    import os
    from prism import adaptive_cut as ac
    old = os.environ.get("EMBER_ADAPTIVE_MODE")
    try:
        # An unrecognised value falls back to the DEFAULT, which is the property; naming "off" here
        # pinned the default of the day and broke when it moved, without any behaviour changing.
        for v, want in (("off", "off"), ("on", "on"), ("shadow", "shadow"),
                        ("nonsense", ac._DEFAULT_MODE)):
            os.environ["EMBER_ADAPTIVE_MODE"] = v
            assert ac.mode() == want
    finally:
        if old is None:
            os.environ.pop("EMBER_ADAPTIVE_MODE", None)
        else:
            os.environ["EMBER_ADAPTIVE_MODE"] = old
