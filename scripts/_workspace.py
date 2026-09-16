"""Where the sibling repositories are, for the standalone programs in `scripts/`.

`tests/_paths.py` answers this question for the test suite. This is the same question asked from the
other side of the repository, and it exists as a second module rather than an import because the
two have different callers and different failure modes: a test may not import from `scripts/`, and a
script must not acquire a dependency on `tests/`.

⛔ WHAT A COUNTED `../..` COSTS HERE. Four programs in this directory each resolved siblings by
hand — `os.path.join(_HERE, "..", "..", "agience-mantle", "src")` and a three-candidate list for
chorus. That spelling is correct in a developer workspace and wrong everywhere else, and the failure
is not an error: `sys.path.insert` of a directory that does not exist is a NO-OP, so the import that
follows either finds an unrelated installed copy or raises much later with a message about the wrong
thing. Measured 2026-09-16 on a clean runner, `tests/test_lattice_scripts.py` reported five failures
that were all one wrong path.

THREE LAYOUTS ARE REAL, and each is a place these programs actually run:

    a developer workspace   <root>/agience-ember, <root>/agience-mantle, …
    CI                      <repo>/.siblings/agience-mantle, …
    node 45's deploy        /home/builder/genesis/chorus-src — source shipped FLATTENED, with no
                            repository directory and no `src/` level

The first two are found by looking for a marker. The third cannot be: it has a different shape, so
it is named explicitly by the one caller that needs it rather than folded into a rule that would
then be wrong about the other two.
"""
from __future__ import annotations

import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: A sibling this repository genuinely depends on, so its absence means "this path is wrong" rather
#: than "that repo moved". `tests/_paths.py` uses the same marker for the same reason.
_MARKER = "agience-mantle"

_CANDIDATE_ROOTS = (
    os.path.normpath(os.path.join(_REPO, "..")),        # developer workspace
    os.path.normpath(os.path.join(_REPO, ".siblings")),  # CI — checkout cannot write above the workspace
)

#: The first candidate that actually holds the marker. Falls back to the developer layout so the
#: error messages below name a plausible path rather than an empty string.
WORKSPACE = next((c for c in _CANDIDATE_ROOTS if os.path.isdir(os.path.join(c, _MARKER))),
                 _CANDIDATE_ROOTS[0])


def repo(name: str) -> Optional[str]:
    """The sibling repository `name`, in whichever layout is on disk — or None."""
    for root in _CANDIDATE_ROOTS:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return None


def src(name: str) -> Optional[str]:
    """`<repo>/src` for a sibling — the path that goes on `sys.path` — or None.

    None rather than a raise: every caller here has its own refusal message naming what it was
    trying to do, and those messages are better than a generic one from this module.
    """
    p = repo(name)
    if p is None:
        return None
    s = os.path.join(p, "src")
    return s if os.path.isdir(s) else None


def add_to_syspath(name: str) -> Optional[str]:
    """Put a sibling's `src/` on `sys.path` and return it, or return None if it is not checked out.

    ⚠ THE RETURN VALUE IS THE POINT. `sys.path.insert` of a non-existent directory succeeds and
    changes nothing, so a caller that ignores the result cannot tell an inserted path from a
    silently skipped one — which is exactly how the hand-written versions of this failed.
    """
    import sys
    s = src(name)
    if s is not None and s not in sys.path:
        sys.path.insert(0, s)
    return s


def require_src(name: str) -> str:
    """`src()` or a refusal naming every place that was looked in."""
    s = src(name)
    if s is None:
        raise SystemExit(
            "FATAL: %s/src not found. Looked in: %s  — this program needs the %s checkout beside "
            "agience-ember (a developer workspace), or under agience-ember/.siblings/ (CI)."
            % (name, "; ".join(os.path.join(r, name) for r in _CANDIDATE_ROOTS), name))
    return s
