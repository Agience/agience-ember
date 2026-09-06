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
WORKSPACE = Path(__file__).resolve().parents[2]
EMBER_SRC = EMBER_REPO / "src" / "ember"

# ── the assertions that make a relocation loud ────────────────────────────────────────────────
# Each names the marker that proves this module resolved where it thinks it did. A path helper that
# resolves wrongly in silence is worse than no helper at all.
assert (EMBER_SRC / "genesis.py").is_file(), (
    f"tests/_paths.py: EMBER_SRC resolved to {EMBER_SRC}, which holds no genesis.py — this file "
    f"has been moved and the parents[] depths above are now wrong. Fix them here, once.")
# The workspace marker is `agience-mantle`, a sibling ember depends on directly. A marker must be a
# repo whose absence means "this path is wrong" rather than one whose absence could also mean "that
# repo moved" — an archived marker turns this guard into an import failure for every reader.
assert (WORKSPACE / "agience-mantle").is_dir(), (
    f"tests/_paths.py: WORKSPACE resolved to {WORKSPACE}, which holds no agience-mantle — this file "
    f"has been moved and the parents[] depths above are now wrong. Fix them here, once.")


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


#: Where node 71's ember operator tooling lives. It was `agience-ember/node/` until it moved to the
#: fleet tree, which is where per-box operations belong: those files name one node's id, principal,
#: bucket and endpoint, and none of that is product.
FLEET_EMBER = WORKSPACE / "_fleet" / "peers" / "71" / "ember"


def fleet_ember(name: str) -> Path:
    """One of node 71's operator scripts — `fleet_ember("node-repair.py")`.

    Raises `FileNotFoundError` when `_fleet` is not checked out, and callers turn that into a skip.
    That skip is honest in a way the old one was not: `_fleet` is a private sibling, so its absence
    is a real absent dependency that an environment change fixes. The path this replaced pointed
    inside this repository at a directory that no longer exists, which no environment change fixes
    — the distinction `test_no_test_module_skips_on_a_wrong_path.py` exists to draw, and the test
    that caught this move.
    """
    if not FLEET_EMBER.is_dir():
        raise FileNotFoundError(
            f"{FLEET_EMBER} does not exist — node 71's operator tooling lives in the `_fleet` "
            f"repository, a private sibling checkout. Tests that read those scripts skip without it.")
    p = FLEET_EMBER / name
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} does not exist. `_fleet` is checked out, so this is a wrong name rather than an "
            f"absent dependency: {sorted(q.name for q in FLEET_EMBER.iterdir())}")
    return p
