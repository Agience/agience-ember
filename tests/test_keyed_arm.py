"""The keyed arm, and the serve-path properties that rest on it (D2, D4, D5, D7, D8, D1).

Each test states the wrong reading alongside the right one. A store that answers everything with a
plausible blank satisfies an assertion that only checks for success, so the fixtures are shaped to
make the two readings differ: a contaminated index, an unbuilt index, an unmeasured stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import _paths

sys.path.insert(0, str(_paths.repo("agience-mantle") / "src"))

from mantle.db import ListIndexUnbuilt, open_lattice  # noqa: E402

WN = "text/x-wordnet"
WIKI = "text/x-wikipedia"


def _store(tmp_path, *, synsets=3, distractors=60):
    """A store shaped like the live corpus: a few synsets carrying 'dog', swamped by wiki rows
    carrying the same lemma. The ratio is the point — an uncontaminated fixture is one an
    undiscriminated lookup also satisfies."""
    L = open_lattice(str(tmp_path / "lattice.db"), origin="test-node")
    L.artifacts.ensure_schema()
    docs = []
    for i in range(synsets):
        docs.append({"id": "wn-dog.n.%02d" % (i + 1), "content_type": WN, "state": "committed",
                     "word": "dog", "pos": "n", "lemmas": ["dog", "canine"],
                     "content": "dog — a domesticated carnivore %d" % i})
    for i in range(distractors):
        docs.append({"id": "wiki-%05d" % i, "content_type": WIKI, "state": "committed",
                     "lemmas": ["dog", "misc"], "content": "Dog (disambiguation) %d" % i})
    L.artifacts.put_many(docs, batch=200)
    return L


# ── D2: the keyed arm exists, and it discriminates by type ───────────────────
def test_lattice_store_has_the_keyed_arm(tmp_path):
    """The lattice store carries the keyed arm — `lookup_by_lemma` and `lookup_by_list_field` —
    which is the surface `define` calls."""
    L = _store(tmp_path)
    assert hasattr(L.artifacts, "lookup_by_lemma")
    assert hasattr(L.artifacts, "lookup_by_list_field")


def test_type_discrimination_happens_before_the_limit(tmp_path):
    """The type discrimination happens in the seek, so the limit applies to matching rows.

    Filtered afterwards, a limit of 200 against a contaminated index returns 200 rows and no
    synsets, because the limit was already spent. This asserts the other order: a small limit
    against a heavily contaminated index still comes back all-synsets."""
    L = _store(tmp_path, synsets=3, distractors=60)

    untyped = L.artifacts.lookup_by_lemma("dog", limit=10)
    assert sum(1 for d in untyped if d["id"].startswith("wn-")) < 10, \
        "fixture is not contaminated, so it cannot test the discriminator"

    typed = L.artifacts.lookup_by_lemma("dog", limit=10, content_type=WN)
    assert typed, "the typed lookup found nothing in a store that holds 3 synsets"
    assert all(d["id"].startswith("wn-") for d in typed), \
        "a distractor survived a typed lookup: %r" % [d["id"] for d in typed]
    assert len(typed) == 3


def test_unbuilt_index_refuses_instead_of_answering_empty(tmp_path):
    """An unbuilt index raises `ListIndexUnbuilt`, so it is distinguishable from a word nobody
    holds. Both read as `[]` otherwise, and they are opposite facts: one is a measurement over the
    corpus, the other is the absence of any index to measure with. Rebuilding restores the read."""
    db = tmp_path / "lattice.db"
    L = open_lattice(str(db), origin="test-node")
    L.artifacts.ensure_schema()
    L.artifacts.put_many([{"id": "wn-x.n.01", "content_type": WN, "lemmas": ["dog"]}], batch=10)

    # A store migrated before `listkey` existed: rows present, index unbuilt.
    with L.db.write() as cur:
        cur.execute("DELETE FROM listkey")
        cur.execute("DELETE FROM counter WHERE name = 'listkey:built'")

    with pytest.raises(ListIndexUnbuilt):
        L.artifacts.lookup_by_lemma("dog", limit=5, content_type=WN)

    assert L.artifacts.rebuild_list_index()["built"] is True
    assert len(L.artifacts.lookup_by_lemma("dog", limit=5, content_type=WN)) == 1


def test_unindexed_field_refuses(tmp_path):
    """An unindexed field has no postings to read, so the lookup raises rather than reporting that
    there were no matches."""
    L = _store(tmp_path, synsets=1, distractors=1)
    with pytest.raises(ValueError):
        L.artifacts.lookup_by_list_field("tags", "anything", limit=5)


def test_index_follows_updates_and_deletes(tmp_path):
    """The derived index tracks its table across updates and deletes. An index that drifts still
    answers, from postings that no longer describe any row."""
    L = _store(tmp_path, synsets=1, distractors=0)
    assert L.artifacts.lookup_by_lemma("dog", limit=5, content_type=WN)

    L.artifacts.put_artifact({"id": "wn-dog.n.01", "content_type": WN, "state": "committed",
                              "word": "cat", "lemmas": ["cat"], "content": "cat — feline"})
    assert L.artifacts.lookup_by_lemma("dog", limit=5, content_type=WN) == [], \
        "a stale lemma posting survived an update"
    assert len(L.artifacts.lookup_by_lemma("cat", limit=5, content_type=WN)) == 1

    L.artifacts.delete_artifact("wn-dog.n.01")
    assert L.artifacts.lookup_by_lemma("cat", limit=5, content_type=WN) == []


def test_ct_change_repoints_the_discriminator(tmp_path):
    """An update that changes `content_type` moves the row's postings with it."""
    L = _store(tmp_path, synsets=1, distractors=0)
    L.artifacts.put_artifact({"id": "wn-dog.n.01", "content_type": WIKI, "state": "committed",
                              "lemmas": ["dog"], "content": "now a wiki row"})
    assert L.artifacts.lookup_by_lemma("dog", limit=5, content_type=WN) == []
    assert len(L.artifacts.lookup_by_lemma("dog", limit=5, content_type=WIKI)) == 1


# ── the cross-backend adapter ────────────────────────────────────────────────
def test_legacy_backend_still_answers_and_admits_it_did_not_discriminate():
    """A backend whose `lookup_by_list_field` takes no `content_type` is served by over-fetching
    and filtering, and the adapter reports `typed=False` alongside the rows.

    Over-fetching recovers what a small limit would have excluded, but it is bounded by how far it
    over-fetches, so it is a different measurement from discriminating in the seek. `typed` is what
    lets a caller tell which one it got."""
    from mantle.shard import keyed

    class LegacyStore:                      # a backend with no content_type parameter
        def lookup_by_list_field(self, field, value, *, limit=20):
            rows = [{"id": "wiki-%d" % i, "content_type": WIKI, "lemmas": [value]}
                    for i in range(50)]
            rows.append({"id": "wn-dog.n.01", "content_type": WN, "word": "dog",
                         "lemmas": [value]})
            return rows[:limit]

    s = LegacyStore()
    assert keyed._supports_typed(s, "lookup_by_list_field") is False
    rows, typed = keyed.lookup_by_lemma(s, "dog", limit=5, content_type=WN)
    assert typed is False, "the degraded path must not claim it discriminated in the seek"
    assert [r["id"] for r in rows] == ["wn-dog.n.01"], \
        "over-fetch failed to recover the synset the LIMIT would have excluded"


# ── D4/D5: groundedness is read from evidence; faults are reported ───────────
def test_refusal_is_not_reported_as_grounded():
    """`serve.py` derives `grounded` from the citations. A hardcoded `True` marks every non-None
    answer as grounded, including the ones that cite nothing."""
    src = _paths.read_ember_src("serve.py")
    assert "text, grounded = r[\"answer\"], bool(cited)" in src, \
        "serve.py no longer derives groundedness from citations"
    assert "text, grounded, cited = r[\"answer\"], True, r.get(\"cited\", [])" not in src


def test_no_arm_of_serve_ASSERTS_groundedness():
    """The general form of the test above, which is why it stands separately.

    That test names one literal line of source, so it covers that line and nothing else — a second
    arm spelling it `text, grounded, cited = control, True, []` further down the file serves every
    `remember`/`status` reply as grounded, including the ones that cite nothing.

    This walks the AST instead: on every arm, `grounded` is derived from the citations or carried on
    an `Answer` that already measured it, and never assigned a literal `True`. A new arm that
    hardcodes it is caught however the line is spelled."""
    import ast, re
    src = _paths.read_ember_src("serve.py")
    tree = ast.parse(src)
    offenders = []

    def _is_literal_true(node) -> bool:
        return isinstance(node, ast.Constant) and node.value is True

    for node in ast.walk(tree):
        # `grounded = True` and `x, grounded, y = ..., True, ...`
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                names = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                values = (node.value.elts if isinstance(node.value, ast.Tuple)
                          and len(node.value.elts) == len(names) else None)
                for i, n in enumerate(names):
                    if not (isinstance(n, ast.Name) and n.id == "grounded"):
                        continue
                    val = values[i] if values else node.value
                    if _is_literal_true(val):
                        offenders.append(f"line {n.lineno}: grounded = True")
        # `grounded=True` passed as a keyword (e.g. Answer(grounded=True, cited=[]))
        if isinstance(node, ast.Call):
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if _is_literal_true(kw.get("grounded")):
                cited = kw.get("cited")
                # grounded=True with a non-empty literal citation list is backed by that list;
                # grounded=True with nothing cited is the case under test.
                empty = isinstance(cited, ast.List) and not cited.elts
                if cited is None or empty:
                    offenders.append(f"line {node.lineno}: grounded=True with no citation")

    assert not offenders, (
        "serve.py ASSERTS groundedness instead of deriving it:\n  " + "\n  ".join(offenders)
        + "\nDerive it (`grounded = bool(cited)`) or carry an Answer that measured it.")

    # The mirror check: the control arm goes through the derivation rather than around it.
    assert re.search(r"text,\s*cited\s*=\s*control\s*\n\s*grounded\s*=\s*bool\(cited\)", src), \
        "the control arm no longer derives groundedness from what it cited"


def test_serve_does_not_swallow_answering_faults():
    """The router and activation arms record their faults through `_note_fault`. Wrapped in a bare
    `except Exception: pass`, an error there arrives at the caller as a blank answer, and every
    later error on the same path arrives the same way."""
    src = _paths.read_ember_src("serve.py")
    assert "_note_fault(faults, \"router\"" in src
    assert "_note_fault(faults, \"activation\"" in src




def test_serve_imports_os():
    """`serve` calls `os.getenv` on the `/v1/invoke` path, so the module carries a module-level
    `import os`. A missing one raises `NameError` only once a request clears the 403 gate, which
    reaches the client as a dropped connection."""
    import ember.surface.serve as s
    assert s.os is not None
