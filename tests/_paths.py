"""The one place that knows where the source tree is.

Every test that reads its subject as text, or reaches a sibling repo, resolves the path through this
module. The `parents[]` depths are computed once here and asserted, so moving this file fails loudly
on import in every consumer instead of resolving to the wrong tree. That assertion is what makes a
wrong root visible: `sys.path.insert` of a non-existent directory is a no-op, so a test built on one
goes on passing while measuring nothing.
"""
from __future__ import annotations

from pathlib import Path

# tests/_paths.py → tests/ → <ember repo> → <workspace root>
EMBER_REPO = Path(__file__).resolve().parents[1]
EMBER_SRC = EMBER_REPO / "src" / "ember"

#: Where the sibling repositories are. TWO layouts are real and both must work:
#:
#:   a developer workspace   <root>/agience-ember, <root>/agience-mantle, …
#:   CI                      <repo>/.siblings/agience-mantle, …
#:
#: `actions/checkout` cannot write outside the workspace, so CI cannot lay siblings out as actual
#: siblings; it puts them under `.siblings/` instead. A single hard-coded `parents[2]` is right in
#: one of those and wrong in the other — and wrong LOUDLY, at import, taking collection down for
#: every module that reads this file.
#:
#: So the root is the first candidate that actually holds the marker, rather than a counted depth.
_MARKER = "agience-mantle"          # a sibling ember depends on directly
_CANDIDATES = (Path(__file__).resolve().parents[2], EMBER_REPO / ".siblings")
WORKSPACE = next((c for c in _CANDIDATES if (c / _MARKER).is_dir()), _CANDIDATES[0])

# ── the assertions that make a relocation loud ────────────────────────────────────────────────
# Each names the marker that proves this module resolved where it thinks it did. A path helper that
# resolves wrongly in silence is worse than no helper at all.
assert (EMBER_SRC / "genesis.py").is_file(), (
    f"tests/_paths.py: EMBER_SRC resolved to {EMBER_SRC}, which holds no genesis.py — this file "
    f"has been moved and the parents[] depths above are now wrong. Fix them here, once.")
# The workspace marker is `agience-mantle`, a sibling ember depends on directly. A marker must be a
# repo whose absence means "this path is wrong" rather than one whose absence could also mean "that
# repo moved" — an archived marker turns this guard into an import failure for every reader.
assert (WORKSPACE / _MARKER).is_dir(), (
    f"tests/_paths.py: no sibling checkout found. Looked for {_MARKER!r} in "
    f"{[str(c) for c in _CANDIDATES]}. In a developer workspace the repos are siblings; in CI they "
    f"are under `.siblings/`, named as they are on the remote. If neither holds it, either this "
    f"file moved (fix the depths above, once) or the checkout step did not run.")


def ember_src(name: str) -> Path:
    """Path to a module inside `src/ember` — for the tests that READ their subject as text.

    `name` may be a bare filename (`"serve.py"`) or a package-relative path (`"surface/serve.py"`).
    A bare name is resolved by searching the subpackages, so a caller does not have to track which
    package a module currently lives in and a move between packages leaves every reader working.

    A bare name matching in more than one package raises `FileNotFoundError` naming both packages:
    a duplicate basename has no single subject, and picking one would silently substitute a module.
    """
    p = EMBER_SRC / name
    if p.is_file():
        return p
    if "/" not in name and "\\" not in name:
        hits = sorted(EMBER_SRC.glob(f"*/{name}"))
        if len(hits) > 1:
            raise FileNotFoundError(
                f"{name!r} is ambiguous — it exists in {[h.parent.name for h in hits]}. Pass the "
                f"package-relative path (e.g. 'node/{name}') so the subject is unambiguous.")
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"{p} does not exist, and no subpackage of {EMBER_SRC} holds {name!r}. If the module was "
        f"MOVED to a chorus persona (the ember→chorus migration), the assertion that read it "
        f"belongs with the module, in that persona's suite — do not delete the check, relocate it.")


def read_ember_src(name: str) -> str:
    return ember_src(name).read_text(encoding="utf-8")


def repo(name: str) -> Path:
    """A sibling repo in the workspace, e.g. `repo("agience-mantle")`."""
    p = WORKSPACE / name
    if not p.is_dir():
        raise FileNotFoundError(f"{p} does not exist — no sibling repo named {name!r}")
    return p


#: Repos whose ON-DISK SPELLING differs between the two layouts, longest-lived first.
#:
#: ⛔ `agience-prism` IS ONE REPOSITORY WITH THREE LANGUAGE SUBTREES (`c/`, `js/`, `py/`), and only
#: the Python one is published — as its own distribution named `agience-prism-py`. So a developer
#: workspace holds `agience-prism/py/src` while `actions/checkout` of the published repo lands
#: `agience-prism-py/src`. The same source, two spellings, and a single hard-coded one is wrong in
#: exactly one of the two places this suite runs.
#:
#: Measured 2026-09-16: `test_one_instrument.py` declared `agience-prism/py` and failed on the
#: runner with "declared in _REPOS but has no `src/` on disk" — correctly, because under that name
#: it does not. The guard was right and the path was the thing that needed to move.
_SPELLINGS = {
    "agience-prism/py": ("agience-prism/py", "agience-prism-py"),
}


def repo_src(label: str) -> Path | None:
    """`<repo>/src` for a declared repo label, in whichever layout is on disk — or None.

    None rather than a raise, because the callers are checks that must be able to SAY a repo is
    missing. `repo()` raises for the callers that cannot proceed without one; this is for the ones
    whose whole job is to report the absence, and they must not be taken down by it before they can.

    ⚠ EVERY CANDIDATE ROOT, NOT JUST `WORKSPACE`. `WORKSPACE` is resolved once, at import, to
    whichever candidate holds the marker — so a lookup through it can only ever see ONE layout.
    Measured 2026-09-16 by placing a repo under `.siblings/` on a developer machine: `_workspace.src`
    found it and this function returned None, because `WORKSPACE` had already been fixed to the
    parent directory. CI never hit that (there, `.siblings/` IS the marked root), which is the
    problem — the disagreement was real and invisible from both sides, passing locally for one
    reason and in CI for another. Searching the candidates directly makes the two resolvers agree
    and makes this branch checkable on either layout.
    """
    for spelling in _SPELLINGS.get(label, (label,)):
        for root in _CANDIDATES:
            src = root / spelling / "src"
            if src.is_dir():
                return src
    return None


#: Where node 71's ember operator tooling lives. It was `agience-ember/node/`, then
#: `_fleet/peers/71/ember/`; it now sits in `agience-cloud/node/`, the repo that already carries the
#: node operational surface. `serve.env` travels with it: `run-serve.sh` sources the env file as a
#: SIBLING, so splitting the two leaves the launcher exiting before it starts anything.
NODE_TOOLING = WORKSPACE / "agience-cloud" / "node"


def node_tooling(name: str) -> Path:
    """One of node 71's operator scripts — `node_tooling("node-repair.py")`.

    Raises `FileNotFoundError` when `agience-cloud` is not checked out, and callers turn that into a
    skip. That skip is honest in a way the old one was not: `agience-cloud` is a private sibling, so
    its absence is a real absent dependency that an environment change fixes. The path this replaced
    pointed inside this repository at a directory that no longer exists, which no environment change
    fixes — the distinction `test_no_test_module_skips_on_a_wrong_path.py` exists to draw, and the
    test that caught this move.
    """
    if not NODE_TOOLING.is_dir():
        raise FileNotFoundError(
            f"{NODE_TOOLING} does not exist — node 71's operator tooling lives in the "
            f"`agience-cloud` repository, a private sibling checkout. Tests that read those scripts "
            f"skip without it.")
    p = NODE_TOOLING / name
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} does not exist. `agience-cloud` is checked out, so this is a wrong name rather "
            f"than an absent dependency: {sorted(q.name for q in NODE_TOOLING.iterdir())}")
    return p
