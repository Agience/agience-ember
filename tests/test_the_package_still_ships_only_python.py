"""`pyproject.toml` says ember has no data to ship. This is what makes that stay true.

The claim is in `pyproject.toml`, where `[tool.setuptools.package-data]` is deliberately absent:

    No `[tool.setuptools.package-data]`. `src/ember/` is Python and nothing else — measured, zero
    non-`.py` files — so a data declaration here would be a no-op that reads like a shipped payload.

⛔ THAT IS A MEASUREMENT, AND A MEASUREMENT IS ONLY TRUE ON ITS DAY. The moment someone adds a
schema, a prompt template or a default config under `src/ember/`, the absent declaration stops being
correct and becomes the bug — setuptools ships no undeclared data file, and nothing raises. The code
that reads it fails at the moment a user first exercises that path, on an installed node only.

⚠ THIS IS NOT HYPOTHETICAL IN THIS WORKSPACE. `agience-chorus` shipped 39 `type.json` files as 0
(2026-09-15) and 13 licensing policy files as 0 (2026-09-16), both exactly this way: data added
beside code, no package-data entry, a green suite, and a node that booted and answered `/healthz`
while serving nothing. Ember is the repo where the answer is currently *zero*, which makes it the
cheapest place to keep honest — there is no list to maintain, only a number that must stay at zero
or grow a declaration alongside it.

⚠ IT FAILS TOWARD A DECISION, NOT TOWARD A RULE. A new data file is not wrong. What is wrong is a
new data file with no declaration, so the failure says: declare it, or say here why it stays behind.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PACKAGE_DIR = "src/ember"

#: Files under the package that are not Python and deliberately do not ship, with the reason.
#: Empty today. An entry here is an argument, not an exemption granted by silence.
_NOT_SHIPPED: dict[str, str] = {}


def _tracked_under_package() -> list[str]:
    """What git carries under `src/ember`.

    Tracked-ness rather than a walk of the disk: `__pycache__`, `.ruff_cache` and build output are
    not the repository's, and a filter that named each of them would rot as tools are added. What
    setuptools ships is drawn from what the repository carries, so that is the question to ask.
    """
    proc = subprocess.run(
        ["git", "ls-files", _PACKAGE_DIR],
        cwd=_REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        pytest.skip("git could not list the package tree; this gate has no oracle here")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_the_package_tree_was_actually_read() -> None:
    """A gate over an empty listing agrees with everything asked of it."""
    tracked = _tracked_under_package()
    assert tracked, "git listed nothing under %s" % _PACKAGE_DIR
    assert any(p.endswith(".py") for p in tracked), (
        "no Python found under %s — this is no longer the package directory" % _PACKAGE_DIR)


def test_a_data_file_added_beside_the_code_must_be_declared() -> None:
    """Zero non-Python files, or a `package-data` declaration that carries them."""
    undeclared = [
        p for p in _tracked_under_package()
        if not p.endswith(".py") and p not in _NOT_SHIPPED
    ]
    if not undeclared:
        return

    declared = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = declared.get("tool", {}).get("setuptools", {}).get("package-data")
    assert package_data, (
        "%d non-Python file(s) now live under %s and `[tool.setuptools.package-data]` is still "
        "absent, so the wheel carries none of them:\n  %s\n\n"
        "If the runtime reads them, declare them. If it does not, add them to _NOT_SHIPPED with "
        "the reason. pyproject's claim that this tree is 'Python and nothing else' is now false "
        "either way and should be corrected in the same edit."
        % (len(undeclared), _PACKAGE_DIR, "\n  ".join(sorted(undeclared)[:20]))
    )


def test_the_pyproject_claim_matches_the_tree() -> None:
    """The comment and the code must not drift apart in the other direction either.

    If a `package-data` section appears while the tree is still all Python, the declaration is the
    no-op the comment warns about — it reads like a shipped payload and carries nothing.
    """
    non_python = [p for p in _tracked_under_package() if not p.endswith(".py")]
    declared = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = declared.get("tool", {}).get("setuptools", {}).get("package-data")

    if package_data and not non_python:
        pytest.fail(
            "pyproject declares package-data %r while %s holds no non-Python file. That is the "
            "no-op the comment above it warns about: it reads like a shipped payload and ships "
            "nothing." % (package_data, _PACKAGE_DIR))


def test_discovery_is_still_anchored_to_the_package_name() -> None:
    """`include` is why a sibling's wheel installed a top-level `tests` module and this one will not."""
    find = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    include = find["tool"]["setuptools"]["packages"]["find"].get("include", [])
    assert "ember*" in include, (
        "packages.find.include no longer anchors discovery to this package: %r. Without it "
        "setuptools writes every directory under src/ into top_level.txt and can ship it." % include)
