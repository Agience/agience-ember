"""An artifact is indexed by what its writer said about it, not by the store's record of the write.

`ember.corpus.ingest._offer_text` turns a stored context into the offer text the retrieval vector
encodes, and `ember.ontology.activation.seeds_from_artifact` seeds from the same text. Mantle stores
its own facets (`minted`, `addressing`, `provenance`, the content address) beside the writer's keys,
and the writer's keys sit at the top level, under `caller`, or some in each. Each shape is pinned
here: the offer names the writer's keys and nothing the store owns — no principal id, no timestamp,
no content hash.
"""
from __future__ import annotations

import json

import pytest

from ember.corpus.ingest import _offer_text

MINT = {"minted_by": "mantle.mint_context",
        "minted": {"at": "2026-09-24T00:00:00+00:00", "principal_id": "u-owner-7f3a"},
        "placement": {"collection_id": "ws-1"},
        "addressing": {"sha256": "ab" * 32, "bytes": 3},
        "provenance": "unknown", "content_key": "u/x.content",
        "content_cas_ref": "cas/" + "cd" * 32}


def _offer(ctx) -> dict:
    return json.loads(_offer_text(json.dumps(ctx)))


@pytest.mark.parametrize("shape,ctx", [
    ("the top level", {**MINT, "title": "Quarterly", "tags": ["a"]}),
    ("under caller alone", {**MINT, "caller": {"title": "Quarterly", "tags": ["a"]}}),
    ("some in each", {**MINT, "title": "Quarterly", "caller": {"tags": ["a"]}}),
])
def test_the_offer_is_the_writers_keys_and_nothing_the_store_owns(shape, ctx):
    got = _offer(ctx)
    assert got == {"title": "Quarterly", "tags": ["a"]}, shape
    text = _offer_text(json.dumps(ctx))
    assert "u-owner-7f3a" not in text and "2026-09-24" not in text, shape


def test_the_top_level_wins_where_it_and_caller_differ():
    ctx = {**MINT, "title": "new", "caller": {"title": "old", "source": "/x.md"}}
    assert _offer(ctx) == {"title": "new", "source": "/x.md"}


def test_a_context_the_store_never_touched_is_indexed_as_written():
    raw = '{"title": "flat", "k": 1}'
    assert _offer_text(raw) == raw
    compact = '{"title":"flat","k":1}'
    assert _offer_text(compact) == compact
    assert _offer_text("an opaque note") == "an opaque note"


def test_a_context_of_nothing_but_the_stores_facets_has_no_offer():
    assert _offer_text(json.dumps(MINT)) == ""


def test_the_offer_keeps_the_writers_characters():
    """Re-serialised with mantle's separators and the characters as written, not escaped."""
    assert _offer_text(json.dumps({**MINT, "title": "Café"})) == '{"title": "Café"}'
    assert _offer_text({"title": "Café"}) == '{"title": "Café"}'


# ── activation seeds from the same offer ─────────────────────────────────────────────────────

def _seeded_text(monkeypatch, art) -> list:
    from ember.ontology import activation
    seen: list = []
    monkeypatch.setattr(activation, "seeds_from_text",
                        lambda text, **kw: seen.append(text) or {"seeded": 1.0})
    activation.seeds_from_artifact(art)
    return seen


def test_an_artifact_seeds_from_its_writers_keys_not_the_mint(monkeypatch):
    seen = _seeded_text(monkeypatch, {"context": json.dumps({**MINT, "title": "wolves howl"})})
    assert len(seen) == 1 and "wolves howl" in seen[0]
    assert "u-owner-7f3a" not in seen[0] and "2026-09-24" not in seen[0], seen[0]


def test_a_context_with_no_offer_seeds_nothing_and_the_content_does(monkeypatch):
    seen = _seeded_text(monkeypatch, {"context": json.dumps(MINT), "content": "wolves"})
    assert seen == ["wolves"]
