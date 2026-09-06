"""The chorus mirror of the corpus index must stay behaviourally identical to this one.

## Why a copy exists at all, and why it must not be merged

`chorus/src/corpus_fts.py` and `ember/src/ember/corpus/fts.py` are the same module, as are
`chorus/src/corpus_stats.py` and `ember/src/ember/ontology/corpus_stats.py`. That is not drift:
the layer law forbids `chorus -> ember` (`chorus/src/tests/test_chorus_does_not_import_ember.py`
asserts it in both directions), and mantle deleted the original, so neither repo can import the
other's copy. Deleting either breaks six or more modules in that repo outright.

## Why this test exists

`chorus/src/corpus_fts.py` states the invariant in prose:

    schema / tokenizer / stemmer / ranking MUST stay byte-identical, because an ember node
    WRITES the index a sage query READS.

A divergence therefore does not raise. It **silently re-ranks the corpus** — a different stemmer
produces different posting lists, a different tokenizer produces different terms, a different
column order re-assigns every bm25 weight — and the first symptom is worse answers, with nothing
in any log. Prose is not enforcement, and until this file the invariant was held by nothing.

## What is compared, and what is deliberately allowed to differ

The comparison is over the **parsed AST with the module docstring removed**, not the raw bytes.
That is the right granularity for this invariant:

  * it is blind to prose, which is where the two copies legitimately differ (ember's header
    carries the Lumen grounding contract; chorus's carries the two-copies warning), and
  * it is total over behaviour: any change to a regex, a constant, a SQL string, the FIELDS
    tuple order, or a function body moves the dump.

Exactly two textual differences are permitted, both normalised below, and both are the *identity*
of the copy rather than its behaviour:

    ember/corpus/fts.py            logging.getLogger("ember.corpus.fts")
    chorus/corpus_fts.py           logging.getLogger("chorus.corpus_fts")

    ember/ontology/corpus_stats.py from ember.corpus import fts as _fts
    chorus/corpus_stats.py         import corpus_fts as _fts

Anything else that differs is a real finding. If a change is intended, make it in BOTH copies in
the SAME commit — that is the whole discipline this file exists to force.
"""
from __future__ import annotations

import ast
import difflib
from pathlib import Path

import pytest

# ── locating the sibling ──────────────────────────────────────────────────────────────────────
# The mirror lives in a sibling repo, which is a checkout-layout fact rather than a dependency:
# nothing is imported, only read.
#
# Both paths go through `_paths`, which resolves each tree once and asserts it, because
# `test_no_test_module_skips_on_a_wrong_path.py` is right that a hand-rolled
# `parents[1] / ".."` here would make a typo indistinguishable from an absent checkout — and this
# module's whole job is to fail when two files disagree, which a silent skip would hide.
from _paths import EMBER_REPO as _EMBER, repo as _repo   # noqa: E402

_CHORUS = _repo("agience-chorus")

#: (this repo's file, the chorus mirror, [(ours, theirs, canonical), ...] permitted swaps).
#:
#: `canonical` must be VALID PYTHON — the normalised source is re-parsed, so a bare sentinel would
#: turn a divergence check into a SyntaxError and the test would fail for the wrong reason.
_MIRRORS = [
    (
        _EMBER / "src" / "ember" / "corpus" / "fts.py",
        _CHORUS / "src" / "corpus_fts.py",
        [('logging.getLogger("ember.corpus.fts")',
          'logging.getLogger("chorus.corpus_fts")',
          'logging.getLogger("<mirror>")')],
    ),
    (
        _EMBER / "src" / "ember" / "ontology" / "corpus_stats.py",
        _CHORUS / "src" / "corpus_stats.py",
        [("from ember.corpus import fts as _fts",
          "import corpus_fts as _fts",
          "import _mirror_fts as _fts")],
    ),
]


def _behaviour(path: Path, swaps, *, is_theirs: bool) -> str:
    """The module's AST with the docstring dropped and the permitted identity swaps normalised."""
    src = path.read_text(encoding="utf-8-sig")
    for ours, theirs, canonical in swaps:
        # Normalise toward a canonical form so neither side is privileged.
        src = src.replace(theirs if is_theirs else ours, canonical)
    tree = ast.parse(src, filename=str(path))
    # Drop the module docstring — the one place the two copies are meant to diverge.
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body = tree.body[1:]
    return ast.dump(tree, indent=1)


@pytest.mark.parametrize("ours,theirs,swaps", _MIRRORS, ids=lambda p: getattr(p, "name", ""))
def test_the_chorus_mirror_is_behaviourally_identical(ours: Path, theirs: Path, swaps):
    if not theirs.exists():
        pytest.skip(f"chorus is not checked out beside ember ({theirs} absent)")
    assert ours.exists(), f"{ours} is missing — this repo's copy of the mirror is gone"

    a = _behaviour(ours, swaps, is_theirs=False)
    b = _behaviour(theirs, swaps, is_theirs=True)
    if a == b:
        return

    diff = "\n".join(list(difflib.unified_diff(
        a.splitlines(), b.splitlines(), fromfile=str(ours), tofile=str(theirs), lineterm="", n=2,
    ))[:80])
    pytest.fail(
        f"\n{ours.name} and its chorus mirror have DIVERGED BEHAVIOURALLY.\n\n"
        f"  ours   : {ours}\n  theirs : {theirs}\n\n"
        "An ember node writes the index a sage query reads. A divergence here does not raise at "
        "runtime — it silently re-ranks the corpus. Make the change in BOTH copies, in the same "
        "commit; if the difference is intended and permanent, add it to `_MIRRORS`' swap list "
        "with a reason.\n\n"
        f"AST diff (module docstring excluded, first 80 lines):\n{diff}"
    )


def test_the_comparison_can_actually_fail():
    """A negative control: the comparison must notice a one-token behavioural change.

    Without this, a bug in `_behaviour` — a swallowed parse error, an over-eager normalisation —
    would make every mirror look identical and the suite would go green while the invariant rotted.
    """
    base = "import re\nX = re.compile(r'[a-z]+')\n"
    tampered = "import re\nX = re.compile(r'[a-z0-9]+')\n"

    def dump(s: str) -> str:
        return ast.dump(ast.parse(s), indent=1)

    assert dump(base) != dump(tampered), "the AST comparison is blind to a changed regex"


def test_the_docstring_really_is_excluded():
    """The two copies' headers differ on purpose; the comparison must not see that."""
    a = '"""Header A, which is long and specific to ember."""\nY = 1\n'
    b = '"""Header B — completely different prose."""\nY = 1\n'

    def body(s: str) -> str:
        t = ast.parse(s)
        if (t.body and isinstance(t.body[0], ast.Expr)
                and isinstance(t.body[0].value, ast.Constant)
                and isinstance(t.body[0].value.value, str)):
            t.body = t.body[1:]
        return ast.dump(t, indent=1)

    assert body(a) == body(b), "differing module docstrings must not count as a divergence"
