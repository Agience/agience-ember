"""Ember holds the instrument and takes its wire from `prism`. This guard holds both halves.

  1. **Nothing imports `beam`.** `agience-beam` is archived. Its wire went to `agience-prism/py`
     (the sixteen modules listed in `WIRE` below) and its two stdlib derivations, `resolution` and
     `adaptive_cut`, went to prism's dependency-free base. No submodule of `beam` is a live address,
     so the whole package is banned with no allow-list.

  2. **Ember holds the instrument.** The instrument is one module — `optics`, the one sanctioned seam
     onto entroptics — living at `ember/optics.py` under ember's AGPL and read from 14 sites in
     non-test `src/`. `tests/test_one_instrument.py` carries the companion rule that nothing else in
     the workspace names entroptics.

Both halves are read out of the source rather than exercised at runtime. A relapsed
`from beam.optics import …` does raise `ModuleNotFoundError` today, but only on the branch that
executes, and ember favours lazy imports so that measurement code is not loaded until a measurement
is taken. A `beam` import inside a rarely-taken branch runs green for weeks; reading it is
immediate.

`test_the_guard_fires_on_seeded_imports` runs the same scanner over a synthetic file and requires
every violation form back, naming file and line, with decoys that stay out. A guard whose failure
mode has never been demonstrated proves nothing ([[verification-that-cannot-fail]]).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import _paths

# The archived package. No scanned tree names it, in any spelling.
ARCHIVED = "beam"

# The sixteen wire modules, named explicitly rather than derived from what `prism/` happens to
# contain: prism holds modules that were never beam's (`trust`, `runner`, `mass`, …), and a list
# computed from the destination would widen silently every time prism grows.
WIRE = frozenset({
    "reach", "plane", "streams", "carriers", "frames", "propagation", "mcp_bridge", "schema",
    "demurrage", "minting", "settlement", "pump", "minhash", "error_threshold", "extraction",
    "conservation",
})

# The instrument: one module. The reader count below is asserted non-empty, so "ember stopped reading
# the instrument entirely" fails here instead of reading as a clean pass.
INSTRUMENT = "optics"

_REPO = _paths.EMBER_REPO
# Every tree ember ships or runs: the library, the operator tooling, and this suite itself. A scan
# restricted to `src/` leaves the scripts unread, and those are where wire imports survived longest.
#
# `node/` was a fourth entry until node 71's operator tooling moved to `_fleet/peers/71/ember/`.
# It is not listed as a workspace path here on purpose: this scan is about what THIS repository
# ships, and a scan that silently covers zero files in a directory nobody checked out reads as a
# pass. The fleet tree is checked where it lives.
_TREES = ("src", "scripts", "tests")


def _imports_of(tree: Path, package: str):
    """Every `<package>.*` import under `tree`, as `(path, lineno, submodule, statement)`.

    Both spellings are read, and the second is the reason this helper exists. `from ember import
    optics` names the submodule just as surely as `from ember.optics import …` does — it simply
    names it among the imported names rather than in the module path. A dotted-only scanner reads it
    as an unresolved bare `import ember` and misses it.

    A bare `import ember` with no submodule yields `""`, which belongs to no submodule set, so it
    counts as a reach for the package rather than a reach at the instrument.
    """
    out = []
    for py in sorted(tree.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        src = py.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src, filename=str(py))):
            if isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    if parts[0] == package:
                        sub = parts[1] if len(parts) > 1 else ""
                        out.append((py, node.lineno, sub, "import " + a.name))
            elif isinstance(node, ast.ImportFrom) and not node.level:
                parts = (node.module or "").split(".")
                if parts[0] != package:
                    continue
                if len(parts) > 1:
                    names = ", ".join(a.name for a in node.names)
                    out.append((py, node.lineno, parts[1], f"from {node.module} import {names}"))
                else:
                    for a in node.names:      # `from <package> import X` — X is the submodule
                        out.append((py, node.lineno, a.name, f"from {package} import {a.name}"))
    return out


def _all_trees():
    for name in _TREES:
        t = _REPO / name
        assert t.is_dir(), f"{t} does not exist — this guard is scanning empty space"
        yield t


def test_nothing_in_ember_imports_the_archived_beam():
    """`agience-beam` is archived: no import of it anywhere ember ships or runs.

    The wire is `prism.<name>`, the two derivations are `prism.resolution` / `prism.adaptive_cut`,
    and the instrument is `ember.optics` in this repo. No submodule of `beam` is a live address, so
    the whole package is banned with no allow-list.
    """
    bad = []
    for tree in _all_trees():
        bad.extend(_imports_of(tree, ARCHIVED))
    assert not bad, (
        "`agience-beam` is ARCHIVED — every one of these names a package that no longer exists:\n" +
        "\n".join(f"  {p.relative_to(_REPO).as_posix()}:{ln}  {txt}" for p, ln, _sub, txt in bad) +
        "\n\nThe wire is `prism.<name>`; the two derivations are `prism.resolution` / "
        "`prism.adaptive_cut`; the instrument is `ember.optics`, in this repo.")


def test_ember_holds_the_instrument():
    """The other half: ember reading `ember.optics` is correct, and a sweep that removes those reads
    is what this catches.

    Measured by this file's own scanner over non-test `src/`, counting both import spellings:
    `optics` is read at 14 sites. The floor sits 2 below that, because a real refactor may
    legitimately move the count and this is a floor rather than a pin. A drop is a module moved that
    should have stayed; zero is the instrument having left ember.

    `optics.py` itself is excluded. It is the instrument rather than a reader of it, and counting a
    module's own file would make the assertion satisfiable by the instrument merely existing.
    """
    instrument_file = _paths.EMBER_SRC / "optics.py"
    assert instrument_file.is_file(), (
        f"{instrument_file} does not exist — the instrument is not in ember. It folded here from the "
        f"archived `agience-beam/src/beam/optics.py` on 2026-08-05.")

    hits = [(p, ln, sub) for (p, ln, sub, _t) in _imports_of(_REPO / "src", "ember")
            if p != instrument_file]
    readers = [(p, ln) for (p, ln, sub) in hits if sub == INSTRUMENT]
    assert len(readers) >= 12, (
        "ember reads `ember.optics` only %d times in src/ — the instrument appears to have been moved "
        "out of ember. Ember HOLDS the instrument." % len(readers))


def test_the_guard_fires_on_seeded_imports(tmp_path):
    """The control, for both halves: seed every violation form and require file and line back.

    Both spellings appear, because both occur in the repo and the bare `from X import Y` form is the
    one a dotted-only scanner misses.
    """
    seeded = tmp_path / "seeded_violation.py"
    seeded.write_text(
        "from prism.optics import nothing\n"           # line 1 — decoy: neither beam nor ember
        "from beam.optics import absorb_transmit\n"    # line 2 — violation, not a live address
        "from beam.reach import GROUND, Reactor\n"     # line 3 — violation, the wire
        "from beam import demurrage\n"                 # line 4 — violation, bare spelling
        "import beam.resolution\n"                     # line 5 — violation, dotted `import`
        "def f():\n"
        "    from ember.optics import read_ordered\n"  # line 7 — the instrument, legal, nested
        "    from ember import optics\n",              # line 8 — the instrument, bare spelling
        encoding="utf-8")

    beam_hits = [(ln, sub) for (_p, ln, sub, _t) in _imports_of(tmp_path, ARCHIVED)]
    assert beam_hits == [(2, "optics"), (3, "reach"), (4, "demurrage"), (5, "resolution")], (
        "the archived-package ban did not name the seeded violations — it would pass over a real one "
        "too. Got %r" % (beam_hits,))

    # …and the instrument counter sees both spellings while leaving `prism.optics` and every `beam`
    # line out. "Fires" and "fires at everything" have to fail differently.
    ember_hits = [(ln, sub) for (_p, ln, sub, _t) in _imports_of(tmp_path, "ember")]
    assert ember_hits == [(7, "optics"), (8, "optics")], (
        "the instrument counter missed a spelling or swept up a decoy: %r" % (ember_hits,))


def test_holding_ember_registers_the_instrument_as_the_process_default():
    """Importing ember registers `ember.optics` as the process default instrument.

    `prism.instrument` resolves an instrument in three steps: an explicit `read=` / `dynamics=` /
    `conservation=` keyword, then the process default a host registered, and otherwise no
    instrument — an absence, reported with the measurement named. `ember/__init__.py` registers
    `ember.optics` as that default, so holding ember is what makes a process a host.

    The default is process-global, so a suite in which any file imports a host leaves every other
    file passing whether or not it declared one. This file therefore declares its own: the
    `import ember` below is the declaration rather than scaffolding, which is why it carries
    `# noqa: F401`. Without it the file passes in the suite and fails when run alone.

    The laziness of the factory is checked in a subprocess. `set_default(factory=…)` importing
    `ember.optics` eagerly would drag numpy and entroptics into every process that only wanted the
    cache, which is the cost the lazy import exists to avoid. In this process `ember.optics` is
    already loaded by the time the suite reaches here, so an in-process check would measure test
    order; a fresh interpreter is where the question has an answer.
    """
    import subprocess
    import sys

    import ember  # noqa: F401  — the host declaration. Registers `ember.optics` as the process
    #              default; without it this file passes in the suite and fails alone.
    from prism import instrument

    assert instrument.get_default() is not None, (
        "ember is imported and `prism.instrument` has no default — `ember/__init__.py` no longer "
        "registers the instrument, so every wire measurement in this process refuses. This is the "
        "registration that came across from `beam/__init__.py`.")

    read = instrument.resolve(None, "read_ordered", at="test_holding_ember_registers")
    assert read.__module__ == "ember.optics", (
        "the process default resolved to %r, not the instrument" % (read.__module__,))

    # ── the factory is lazy, asked where the answer means something ──────────────────────────────
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "import ember\n"
         "assert 'ember.optics' not in sys.modules, 'EAGER: `import ember` loaded the instrument'\n"
         # entroptics rather than numpy. `import ember` does load numpy, via `mantle.shard.cache`
         # and ember's own signal modules, for reasons unrelated to the instrument — so a numpy
         # assertion would fire on that too. entroptics is the exact proxy: the instrument is the only
         # thing in ember that pulls it, so its absence here says the factory has not run.
         "assert 'entroptics' not in sys.modules, 'EAGER: `import ember` pulled entroptics'\n"
         "from prism import instrument\n"
         "assert instrument.get_default() is not None, 'no default registered in a fresh process'\n"
         "instrument.resolve(None, 'read_ordered', at='probe')\n"
         "assert 'ember.optics' in sys.modules, 'resolving the default did not load the instrument'\n"
         "print('LAZY OK')\n"],
        capture_output=True, text=True)
    assert "LAZY OK" in r.stdout, (
        "the process-default factory is not lazy, or does not resolve, in a fresh interpreter:\n"
        + (r.stderr or r.stdout)[-2000:])


def test_the_wire_ember_imports_from_prism_is_importable():
    """Every wire module ember names resolves.

    A stale `prism.<name>` fails at run time, in whichever branch happens to reach it — which, for
    the lazy imports ember favours, can be a long way from the edit.
    """
    import importlib
    named = set()
    for tree in _all_trees():
        for _p, _ln, sub, _t in _imports_of(tree, "prism"):
            if sub in WIRE:
                named.add(sub)
    assert named, "no prism wire module is imported anywhere — the repoint did not land"
    for mod in sorted(named):
        try:
            importlib.import_module(f"prism.{mod}")
        except ImportError as exc:                      # pragma: no cover — the failure is the point
            pytest.fail(f"ember names `prism.{mod}` and it does not import: {exc}")


def test_the_screen_the_instrument_hands_back_is_entroptics_backed():
    """The instrument must RETURN an entroptics object, not merely avoid importing one.

    MOVED HERE FROM `agience-crystal` ON 2026-08-25 [John: Entroptics must not be named in
    crystal; move the tests that need it to a repo that can support it]. Crystal's
    `test_embodiment_injection.py` carried this line as the control for every parametrised test in
    that file:

        assert ember_optics.membrane_screen().__module__.startswith("entroptics")

    Without it, crystal's whole injection proof could have run against two stubs and reported
    success. Crystal is the wrong owner of the assertion anyway — `membrane_screen` is ember's
    function, and this is a fact about what ember hands over.

    IT IS NOT DUPLICATED BY THE TWO GUARDS BESIDE IT, which is why it had to move rather than
    simply be deleted. `test_one_instrument.py` reads SOURCE and says only ember names entroptics;
    `test_holding_ember_registers_the_instrument_as_the_process_default` says `import ember` does not
    pull it EAGERLY. Both are import-time facts. This one is a runtime fact about the returned
    object, and a stub screen wired in behind `membrane_screen` would satisfy both of the others.
    """
    from ember import optics

    screen = optics.membrane_screen()
    # IT RETURNS A CLASS, NOT AN INSTANCE, so read `__module__` off the object itself. Writing
    # `type(screen).__module__` reads `builtins` — the metaclass — and the assertion fails on a
    # correct instrument. Measured 2026-08-25 while moving this here, by making exactly that mistake.
    assert isinstance(screen, type), (
        "membrane_screen() returned an instance (%r); this assertion reads `__module__` off the "
        "class it is documented to return." % type(screen).__name__)
    root = screen.__module__.split(".")[0]
    assert root == "entroptics", (
        "membrane_screen() handed back %r from %r — the instrument is meant to return an entroptics "
        "class. If the instrument legitimately moved, follow it here; do not delete the assertion, "
        "because it is the only runtime check that the instrument behind the seam is the real one."
        % (screen.__name__, screen.__module__))
