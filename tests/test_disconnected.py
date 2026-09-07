"""Ember answers with the network unplugged, and reports plainly when it holds nothing.

These tests construct no channel. `refill=None` stands for the real thing rather than mocking it:
there is genuinely no transport in the process, so a path that needed the network could not pass.
"""
from __future__ import annotations

import numpy as np
import pytest

from _fakes import _StubAnswerer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ember import LocalCache, answer_query, build
from mantle.search.anchors.anchorset import AnchorSet

DIM = 16
MODEL = "test-embed"
PRINCIPAL = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
# Artifact id, not a name (collections are artifacts with the right edges).
COLLECTION = "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"


def _anchorset() -> AnchorSet:
    """Well-separated anchors: one axis per topic."""
    a = AnchorSet(model_id=MODEL, dim=DIM)
    for i in range(4):
        v = np.full(DIM, 0.05, dtype=np.float32)
        v[i] = 1.0
        a.add_text(f"anchor-{i}", v)
    return a


def _vec(axis: int, jitter: float = 0.0) -> np.ndarray:
    v = np.full(DIM, 0.05, dtype=np.float32)
    v[axis] = 1.0
    if jitter:
        v[(axis + 5) % DIM] += jitter
    return v


@pytest.fixture()
def cache() -> LocalCache:
    c = LocalCache(_anchorset(), PRINCIPAL, COLLECTION, nprobe=3)
    priv = Ed25519PrivateKey.generate()
    # Author local artifacts on axis 0 — "your own content is a region this node originates".
    c.put(
        [
            ("art-1", b"Ontologies name the things a system can talk about.", _vec(0)),
            ("art-2", b"An anchor set is the shared coordinate system.", _vec(0, 0.02)),
            ("art-3", b"A shard is a set of anchor-region artifacts.", _vec(0, 0.04)),
        ],
        version=1, authority="ember-local", priv=priv,
    )
    return c


def test_answers_with_no_network_at_all(cache: LocalCache) -> None:
    """The core claim: a hit is served from local shards, with no transport in the process."""
    res = answer_query(cache, _StubAnswerer(), "what is an ontology?", _vec(0), refill=None)
    assert res.routing.hit                    # the nearest cell is held locally
    assert res.served_offline
    assert res.refilled == []
    assert res.answer.grounded
    assert "Ontologies" in res.answer.text
    assert res.answer.cited                   # attributed to artifacts rather than to weights


def test_hit_means_the_nearest_cell_specifically(cache: LocalCache) -> None:
    """A hit means the nearest routed cell is held. The best match indexes into that one cell, so
    holding some other routed cell would serve a confident answer from the wrong neighbourhood."""
    r_local = cache.route(_vec(0))
    assert r_local.hit and r_local.regions[0] in r_local.held

    r_away = cache.route(_vec(2))             # a topic we hold nothing for
    assert not r_away.hit
    assert r_away.missing                     # and we know exactly what we'd need




# The composers live in `lumen/composers.py`, so the tests for what an answerer says with
# structureless or absent evidence live in `lumen/tests/test_composers.py`. The read-path tests here
# use an injected stub, because they are about routing and refill rather than prose.


def test_the_runner_refuses_to_build_an_answerer() -> None:
    """`ember.runtime.engine.build()` raises `NoAnswerer` for every name: the runner selects no
    answerer. A default composer here would let a node answer from the runner with nothing saying
    so, which is the one arrangement a caller cannot detect from the outside."""
    from ember.runtime.engine import NoAnswerer, build as _build
    for name in ("extractive", "entroptics", "distill", "gpt-9"):
        with pytest.raises(NoAnswerer):
            _build(name)


