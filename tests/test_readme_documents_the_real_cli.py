"""The README must name commands that exist.

Documentation can be well-formed prose and still describe a program that is not there, so reading
does not catch this class of drift. The check is mechanical instead: every `python -m ember.<x>`
the README tells a reader to run must be an importable module, and every subcommand registered by
`ember.cli` must appear in the README.

Both directions are checked, because each misses the other's failure. A command that exists and is
undocumented is a capability nobody finds; a command that is documented and does not exist fails on
a reader's first action.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import textwrap

import pytest


def _repo() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _readme_text() -> str:
    return (_repo() / "README.md").read_text(encoding="utf-8", errors="replace")


def _registered_subcommands() -> set[str]:
    """Names passed to `sub.add_parser("<name>", ...)` in `ember/cli.py`."""
    src = (_repo() / "src" / "ember" / "cli.py").read_text(encoding="utf-8-sig", errors="replace")
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_parser"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value)
    return names


def _runnable_blocks() -> list[str]:
    """Only fenced code blocks. Prose may name a module that does not exist — the README names
    `ember.demo` while explaining this very scoping — and naming one is documentation rather than an
    instruction to run it."""
    # The language tag alternation matches any tag, so fences pair as written. A tag the pattern
    # does not cover pairs an opening fence with the wrong closing fence, and the extractor then
    # returns ordinary prose as a "block" containing no commands.
    return re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)```", _readme_text(), flags=re.S)


def _module_targets_in_readme() -> set[str]:
    """Every `python -m <dotted.module>` the README instructs the reader to run."""
    targets: set[str] = set()
    for block in _runnable_blocks():
        targets |= set(re.findall(r"python\s+-m\s+([A-Za-z_][\w.]*)", block))
    return targets


def test_every_python_m_target_in_the_readme_is_importable():
    """Every `python -m <module>` a fenced block tells the reader to run must be importable.

    The Quick Start uses the `ember` console script, so no `python -m` invocation appears in any
    fenced block and this asserts over an empty set. It is a forward guard: it acquires teeth the
    moment one is added. An empty result here means "nothing to check" rather than "the scan is
    broken", because the extractor itself is exercised against a synthetic README in the control
    below."""
    missing = []
    for target in sorted(_module_targets_in_readme()):
        try:
            found = importlib.util.find_spec(target) is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        if not found:
            missing.append(target)
    assert not missing, (
        "README tells the reader to run modules that do not exist: "
        + ", ".join(f"python -m {m}" for m in missing)
    )


def test_every_registered_subcommand_is_documented():
    """A subcommand nobody documents is a capability nobody finds."""
    text = _readme_text()
    undocumented = sorted(c for c in _registered_subcommands() if f"ember {c}" not in text)
    assert not undocumented, (
        f"registered in ember/cli.py but absent from README.md: {undocumented}\n"
        f"Document them as `ember <subcommand>` rather than describing the CLI in prose."
    )


def test_the_readme_does_not_document_a_subcommand_that_does_not_exist():
    """The mirror of the above: the README documents no `ember <cmd>` that the CLI does not
    register."""
    registered = _registered_subcommands()
    claimed = set(re.findall(r"`ember ([a-z][a-z0-9_-]*)`", _readme_text()))
    phantom = sorted(claimed - registered)
    assert not phantom, (
        f"README documents `ember <cmd>` for commands the CLI does not register: {phantom}\n"
        f"Registered: {sorted(registered)}"
    )


def test_the_required_anchors_flag_is_documented_where_it_is_required():
    """`ingest` and `reindex` both require `--anchors`, because anchors are provisioned and never
    derived locally. A reader who does not know that reads the non-zero exit as a bug, so the README
    documents the flag.

    The check asks the CLI rather than reading `cli.py`. Each subcommand runs without `--anchors`
    and must exit non-zero naming the flag. That is the property a caller depends on — no ingest
    without a provisioned AnchorSet — and it holds across any refactor that preserves behaviour.

    A static reading cannot pin it. A whole-file substring search for `required=True` matches
    `add_subparsers(dest="cmd", required=True)`, which has nothing to do with anchors; and counting
    `node.lineno` matches once for the two subparsers, because a loop such as
    `for sp in (ing, ri): sp.add_argument("--anchors", …)` is a single `ast.Call`.
    """
    import subprocess
    import sys

    for cmd in ("ingest", "reindex"):
        r = subprocess.run([sys.executable, "-m", "ember.cli", cmd],
                           capture_output=True, cwd=str(_repo()))
        err = (r.stderr or b"").decode("utf-8", "replace")
        assert r.returncode != 0, (
            f"`ember {cmd}` succeeded WITHOUT --anchors. Anchors are provisioned, never derived "
            f"locally; an optional flag lets a caller fall back to deriving them."
        )
        assert "--anchors" in err, (
            f"`ember {cmd}` failed without naming --anchors, so the refusal does not tell the "
            f"caller what is missing. stderr: {err[:300]!r}"
        )

    assert "--anchors" in _readme_text(), "README must document the required --anchors flag"


def test_the_extractors_are_not_silently_empty():
    """The control. Both assertions above compare sets, so an extractor that returned nothing would
    make them pass on nothing. This pins each extractor's output shape so a set comparison stays
    meaningful."""
    subs = _registered_subcommands()
    assert len(subs) >= 5, f"cli.py parsed to too few subcommands: {sorted(subs)}"
    assert {"ingest", "reindex", "ask", "status", "serve"} <= subs, sorted(subs)
    assert len(_readme_text()) > 2000, "README extractor returned implausibly little text"

    # the block extractor must actually find fenced blocks, and the module regex must match inside one
    blocks = _runnable_blocks()
    assert blocks, "no fenced code blocks found in README.md"
    # A count rather than a bare truth test: one "block" is what an extractor returns when its
    # language alternation misses a tag and two real fences pair across the prose between them.
    assert len(blocks) >= 2, f"expected several fenced blocks, got {len(blocks)}: {[b[:40] for b in blocks]}"
    assert any("ember status" in b for b in blocks), (
        "the Quick Start block is not among the extracted blocks — fences are pairing wrong again")
    found = re.findall(r"python\s+-m\s+([A-Za-z_][\w.]*)", "run python -m ember.cli now")
    assert found == ["ember.cli"], found

    # The exclusion half of the scoping: a module named only in prose stays out of the runnable
    # targets. The README names `ember.demo` while explaining this scoping, which is what makes it
    # available as the fixture here.
    assert "ember.demo" in _readme_text(), "the tombstone quoting the dead module should still be there"
    assert "ember.demo" not in _module_targets_in_readme(), "prose mention leaked into runnable targets"

    # The inclusion half. An extractor that matched nothing at all would satisfy every assertion
    # above, including the exclusion check. Running it over a synthetic README keeps it honest while
    # the real one has no `python -m` line.
    import re as _re
    synthetic = textwrap.dedent(
        """\
        text
        ```bash
        python -m pkg.mod --flag
        ```
        ```python
        x = 1
        ```
        """
    )
    # The real callables are exercised, pointed at the synthetic README by patching the source they
    # read. Re-inlining copies of the two regexes here would verify the regex literals instead, and
    # would stay green if either extractor's body were replaced by `return set()`.
    import unittest.mock as _mock
    with _mock.patch(__name__ + "._readme_text", return_value=synthetic):
        blocks = _runnable_blocks()
        assert len(blocks) == 2, blocks
        hits = _module_targets_in_readme()
    assert hits == {"pkg.mod"}, (
        f"`_module_targets_in_readme` cannot see a fenced `python -m` line: {hits}. "
        f"An extractor that returns nothing makes every assertion in this file vacuous."
    )


@pytest.mark.parametrize("cmd", sorted(_registered_subcommands()))
def test_each_subcommand_parses(cmd):
    """`--help` on every registered subcommand. A command that cannot parse its own arguments is
    the same defect one layer down from a command that does not exist."""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "ember.cli", cmd, "--help"],
                       capture_output=True, cwd=str(_repo()))
    assert r.returncode == 0, f"`ember {cmd} --help` exited {r.returncode}: {r.stderr[:400]!r}"
