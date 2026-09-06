"""The lexical index a fresh store builds for itself — one index, and the trap around it.

An index is derived data, so the writer derives it: a store built by the ordinary ingest path comes
with its index, rather than depending on a separate migration step to create one.

The index is `ember.corpus.fts` — one implementation, chosen on a measured baseline, vendored out of
mantle when mantle shed the plaintext index (see that module's header, which states plainly what it
stores in the clear). These tests call it at the surface the ingest path calls.
"""
import sqlite3
import tempfile
import os

import pytest

from ember.corpus import fts as lx


def _lattice():
    """The lattice `db` handle — what every entry point below takes."""
    from mantle.db import open_lattice
    return open_lattice(os.path.join(tempfile.mkdtemp(), "c.db"), origin="test")


def _doc(aid, **kw):
    d = {"id": aid, "content_type": "text/x-wordnet", "state": "committed",
         "created_by": "u", "created_time": "2026-01-01T00:00:00+00:00"}
    d.update(kw)
    return d


def test_a_fresh_store_gets_an_index_from_the_writer():
    L = _lattice()
    assert lx.coverage_for(L.db)["indexed"] == 0
    n = lx.index_for(L.db, [_doc("wn-a", title="dog", gloss="a domesticated canine"),
                          _doc("wn-b", title="cat", gloss="a domesticated feline")])
    assert n == 2
    cov = lx.coverage_for(L.db)
    assert cov["indexed"] == 2 and cov["built"] is True


def test_the_posting_resolves_back_to_ITS_document():
    """The whole point of the map. A contentless table cannot say which document a posting is."""
    L = _lattice()
    lx.index_for(L.db, [_doc("wn-dog", gloss="a domesticated canine"),
                      _doc("wn-cat", gloss="a domesticated feline")])
    rows = L.db.read().execute(
        "SELECT m.vertex_id FROM %s JOIN %s m ON m.fts_rowid = %s.rowid "
        "WHERE %s MATCH 'canine'" % (lx.FTS_TABLE, lx.MAP_TABLE, lx.FTS_TABLE, lx.FTS_TABLE)).fetchall()
    assert [r[0] for r in rows] == ["wn-dog"]


def test_READING_A_COLUMN_OFF_THE_CONTENTLESS_TABLE_IS_NULL():
    """The trap, pinned. The table declares an `id` column, so joining `fts_vertex.id = vertex.id`
    reads as the obvious approach. `content=''` means FTS5 stores postings and no text, so every
    column reads back NULL and that join matches nothing, with an empty result and no error.

    A `count(*) WHERE v.id <> f.id` check over that join reads as agreement, because `x <> NULL` is
    NULL rather than true and the count can only ever be 0. Both halves are asserted below, which
    is why the map table carries the vertex id instead."""
    L = _lattice()
    lx.index_for(L.db, [_doc("wn-dog", gloss="a domesticated canine")])
    got = [r[0] for r in L.db.read().execute("SELECT title FROM %s" % lx.FTS_TABLE).fetchall()]
    assert got == [None]                          # the text is not readable back — hence the map
    vacuous = L.db.read().execute(
        "SELECT count(*) FROM %s JOIN vertex v ON v.rowid = %s.rowid "
        "WHERE v.id <> %s.title" % (lx.FTS_TABLE, lx.FTS_TABLE, lx.FTS_TABLE)).fetchone()[0]
    assert vacuous == 0                            # and this 0 means nothing at all


def test_reindexing_replaces_rather_than_accumulates():
    """A re-described artifact keeps one posting, so it cannot out-vote itself on term counts."""
    L = _lattice()
    lx.index_for(L.db, [_doc("wn-a", gloss="a domesticated canine")])
    lx.index_for(L.db, [_doc("wn-a", gloss="a domesticated canine")])
    assert lx.coverage_for(L.db)["indexed"] == 1
    assert L.db.read().execute(
        "SELECT count(*) FROM %s WHERE vertex_id='wn-a'" % lx.MAP_TABLE).fetchone()[0] == 1
    # and re-indexing leaves the term frequency where it was
    assert L.db.read().execute(
        "SELECT count(*) FROM %s WHERE %s MATCH 'canine'" % (lx.FTS_TABLE, lx.FTS_TABLE)).fetchone()[0] == 1


def test_an_archived_snapshot_leaves_the_searchable_index():
    """Head-only: a content-decides snapshot does not surface alongside its own head."""
    L = _lattice()
    lx.index_for(L.db, [_doc("wn-a", gloss="a domesticated canine")])
    assert lx.coverage_for(L.db)["indexed"] == 1
    lx.index_for(L.db, [_doc("wn-a", state="archived", gloss="a domesticated canine")])
    assert lx.coverage_for(L.db)["indexed"] == 0


def test_rebuild_covers_the_whole_store():
    L = _lattice()
    for i in range(5):
        L.artifacts.put_artifact(_doc("wn-%d" % i, gloss="gloss number %d" % i))
    assert lx.rebuild_for(L.db) == 5
    cov = lx.coverage_for(L.db)
    assert cov["indexed"] == cov["vertices"] == 5 and cov["missing"] == 0


def test_a_store_that_cannot_be_indexed_RAISES_rather_than_reporting_zero():
    """A store with no SQL handle cannot be indexed, and that raises rather than answering 0.

    A 0 would be indistinguishable from "indexed nothing because there was nothing to index". "I
    could not run this" and "this returned nothing" are different facts and carry different
    values."""
    class _NoSql:
        pass
    import pytest as _pt
    with _pt.raises(Exception):
        lx.index_for(_NoSql(), [_doc("x")])
    assert lx.is_built_for(_NoSql()) is False


def test_the_projection_is_the_UNIVERSAL_shape_not_one_corpus_spelling():
    """An artifact advertises an offer, so the offer is its description and its keyed lemmas are
    its tags. `gloss`, `context` and `lemmas` are this corpus's spelling of the universal four
    fields, which is why the projection lives with the index rather than in whichever leaf writes
    rows."""
    d = lx.project_artifact({"id": "wn-a", "title": "dog", "gloss": "a canine",
                             "lemmas": ["dog", "domestic dog"], "content": "a canine"})
    assert d.vertex_id == "wn-a" and d.title == "dog"
    assert d.description == "a canine"                  # undescribed row: the gloss stands in
    assert "domestic dog" in d.tags
    # a described row prefers its own offer — that is what a need is matched against
    d2 = lx.project_artifact({"id": "wn-a", "gloss": "a canine",
                              "description": "the word dog: a canine"})
    assert d2.description == "the word dog: a canine"

    # ── the offer is TOP-LEVEL; a structured `context` is the compatibility path ─────────────
    d3 = lx.project_artifact({"id": "wn-a", "gloss": "a canine",
                              "context": {"description": "the word dog: a canine"}})
    assert d3.description == "the word dog: a canine"

    # A bare string in `context` is provenance rather than an offer. The only two ingests that
    # write such a string write where the row came from rather than what it is about:
    #
    #     sage/canon.py       "canon knowledge: best-practices §intro"
    #     stage0_sources.py   "the concept 0: a ConceptNet 5.7 English term node"
    #
    # Promoted whole, that makes the stated offer of all 6,480 canon documents their provenance,
    # and every one of them positions on the same two nodes — `canon.n.01` and `cognition.n.01`.
    # A field that says the same thing about every member of a corpus cannot tell them apart.
    # Provenance belongs in `citation` / `source_path` / `via`.
    d4 = lx.project_artifact({"id": "wn-a", "gloss": "a canine",
                              "context": "canon knowledge: best-practices §intro"})
    assert d4.description == "a canine", (
        "a bare `context` string was promoted to the offer again; got %r" % (d4.description,))


def test_a_dark_row_is_still_findable_or_it_can_never_be_described():
    """An artifact lands dark (no `context`) and the describer keys it later. If the projection
    required a description, a dark row would be unindexed, unfindable, and therefore permanently
    dark — the illuminate step could never reach it."""
    d = lx.project_artifact({"id": "wn-a", "title": "dog", "gloss": "a canine"})
    assert d.description or d.title
