"""A skip for a missing dependency is honest. A skip for a wrong path is a silent pass.

The two findings read identically in a pytest summary line and are entirely different:

    boto3 is not installed          the test cannot run here. Honest, and re-runnable elsewhere
                                    without changing a line of code.
    `_SCRIPTS` points at nothing    the test is aimed at empty space. It runs nowhere, and no
                                    environment change makes it resolve. That is a bug reporting
                                    itself as a skip.

A module-level path constant is the kind of thing a directory move invalidates: `_SCRIPTS` in
`test_lattice_scripts.py` was `tests/../../scripts`, written while the module lived elsewhere in
the tree, and it resolved to a directory that has never existed. Every `_load()` reached
`pytest.skip("… not present")`, permanently skipping 21 of that module's 24 tests with the suite
green.

So the distinction is structural rather than written down, because comments do not fail. This guard
reads every test module's AST, finds the module-level constants whose value is derived from the test
file's own location (`__file__`, the `_paths` helper, or another such constant), evaluates them the
way Python would, and requires each to name something that exists on disk.

  * A dependency skip has no such constant, so it is untouched: the legitimate case stays legal.
  * A wrong path is such a constant, and it goes red here, naming the file and line.

An expression the evaluator cannot model is reported too. Passing over an unrecognised idiom would
put the same silent-pass shape one level up — the guard green over exactly the constants it could
not check — so an unevaluable location-derived expression goes red and asks the author to route the
path through `tests/_paths.py`, the module that resolves every tree once and asserts it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import _paths

_TESTS = Path(__file__).resolve().parent
assert (_TESTS / "_paths.py").is_file(), (
    "this guard resolved its own directory to %s, which holds no _paths.py" % _TESTS)


class _Unevaluable(Exception):
    """The expression is location-derived but written in an idiom this guard does not model."""


# ── a small evaluator over the path idioms this suite uses ────────────────────────────────────
# This walks the AST rather than calling `eval()`. Importing a test module to read its globals runs
# its import-time side effects, and one of those side effects is the very `sys.path.insert` whose
# correctness is in question, so the guard reads the path as written.
_OSPATH = {"join", "dirname", "abspath", "normpath", "realpath", "expanduser"}
#: `_paths` attributes that are resolved trees. Anything else on `_paths` is a function or a doc.
#:
#: `FLEET_EMBER` is the one entry that may legitimately NOT exist: node 71's operator tooling lives
#: in `_fleet`, a private sibling checkout, so its absence is an absent dependency rather than a
#: wrong path — the distinction this whole module draws. It is listed here so the guard can
#: evaluate it, and `_is_absent_dependency` below is what stops a missing `_fleet` reading as a
#: defect while keeping a wrong FILE NAME under a present `_fleet` reading as one.
_PATHS_CONSTANTS = {"EMBER_REPO", "WORKSPACE", "EMBER_SRC", "FLEET_EMBER"}
_PATHS_CALLS = {"repo", "ember_src", "fleet_ember"}

#: Roots whose absence means "that repository is not checked out", not "this path is wrong".
_ABSENT_DEPENDENCY_ROOTS = (_paths.FLEET_EMBER,)


def _is_absent_dependency(value) -> bool:
    """True when *value* sits under a root that a checkout would supply.

    Scoped to roots that are THEMSELVES missing. With `_fleet` checked out, a path under it that
    does not exist is a wrong name and is reported, so this exemption cannot hide a typo.
    """
    candidate = Path(value)
    for root in _ABSENT_DEPENDENCY_ROOTS:
        if root.exists():
            continue
        if candidate == root or root in candidate.parents:
            return True
    return False


def _dotted(node):
    """`os.path.join` → "os.path.join"; a non-dotted expression → None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _evaluate(node, module_file, bound):
    """Value of *node* as a `Path`/`str`, or raise `_Unevaluable`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return str(module_file)
        if node.id in bound:
            return bound[node.id]
        raise _Unevaluable(ast.unparse(node))

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):        # Path / "sub"
        left = _evaluate(node.left, module_file, bound)
        right = _evaluate(node.right, module_file, bound)
        return Path(left) / str(right)

    if isinstance(node, ast.Subscript):                                      # .parents[N]
        val = _evaluate(node.value, module_file, bound)
        idx = node.slice
        if isinstance(idx, ast.Constant) and isinstance(idx.value, int) and isinstance(val, list):
            if idx.value >= len(val):
                # `parents[9]` on a short path raises IndexError at import, so report it as the
                # wrong depth it is rather than as an unmodelled idiom.
                raise _Unevaluable("parents[%d] is deeper than the path has parts" % idx.value)
            return val[idx.value]
        raise _Unevaluable(ast.unparse(node))

    if isinstance(node, ast.Attribute):
        dotted = _dotted(node)
        if dotted in ("_paths.%s" % n for n in _PATHS_CONSTANTS):
            return getattr(_paths, dotted.split(".", 1)[1])
        base = _evaluate(node.value, module_file, bound)
        if node.attr == "parent":
            return Path(base).parent
        if node.attr == "parents":
            return list(Path(base).parents)
        raise _Unevaluable(ast.unparse(node))

    if isinstance(node, ast.Call):
        dotted = _dotted(node.func)
        args = None
        if dotted in ("Path", "pathlib.Path", "str"):
            args = [_evaluate(a, module_file, bound) for a in node.args]
            return Path(args[0]) if dotted != "str" else str(args[0])
        if dotted and dotted.startswith("os.path.") and dotted.split(".")[-1] in _OSPATH:
            fn = dotted.split(".")[-1]
            args = [str(_evaluate(a, module_file, bound)) for a in node.args]
            import os.path as _p
            return getattr(_p, fn)(*args)
        if dotted in ("_paths.%s" % n for n in _PATHS_CALLS):
            name = dotted.split(".", 1)[1]
            args = [_evaluate(a, module_file, bound) for a in node.args]
            try:
                return getattr(_paths, name)(*args)
            except FileNotFoundError as e:      # _paths already asserts; surface it as our failure
                raise _Unevaluable(str(e)) from e
        # `.resolve()` / `.absolute()` on something we know
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "absolute"):
            return Path(_evaluate(node.func.value, module_file, bound)).resolve()
        raise _Unevaluable(ast.unparse(node))

    raise _Unevaluable(ast.unparse(node))


def _is_location_derived(node, bound):
    """Does this expression get its value from where the test file sits?

    That is the whole trigger. A constant like `_TIMEOUT = 30` or a literal URL is not a path and
    is none of this guard's business; `os.path.join(dirname(__file__), "..", "scripts")` is.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and (sub.id == "__file__" or sub.id in bound):
            return True
        if isinstance(sub, ast.Attribute) and _dotted(sub) and \
                _dotted(sub).startswith("_paths."):
            return True
        if isinstance(sub, ast.Call) and _dotted(sub.func) and \
                _dotted(sub.func).startswith("_paths."):
            return True
    return False


def location_derived_paths(module_file: Path):
    """``[(name, lineno, value_or_None, error_or_None), ...]`` for one module."""
    tree = ast.parse(module_file.read_text(encoding="utf-8", errors="replace"),
                     filename=str(module_file))
    bound, found = {}, []
    for stmt in tree.body:                       # module level only — that is where these live
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
            targets = [stmt.target.id]
        else:
            continue
        if not targets or stmt.value is None:
            continue
        if not _is_location_derived(stmt.value, bound):
            continue
        try:
            value = _evaluate(stmt.value, module_file, bound)
        except _Unevaluable as e:
            found.append((targets[0], stmt.lineno, None, str(e)))
            continue
        for t in targets:
            bound[t] = value
        found.append((targets[0], stmt.lineno, value, None))
    return found


def _test_modules():
    return sorted(p for p in _TESTS.rglob("*.py")
                  if p.name.startswith("test_") and p.name != Path(__file__).name)


# ══ the negative controls — first, because a guard that cannot fail proves nothing ═══════════

def test_the_guard_fires_on_a_wrong_path(tmp_path):
    """The defect, seeded: one `..` too many, pointing outside the repo at a directory that does
    not exist. The guard evaluates the constant and reports where it lands."""
    m = tmp_path / "test_seeded_wrong_path.py"
    m.write_text("import os\n"
                 "_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),"
                 " '..', '..', 'scripts')\n", encoding="utf-8")
    got = location_derived_paths(m)
    assert len(got) == 1 and got[0][0] == "_SCRIPTS"
    assert got[0][3] is None, "the guard failed to EVALUATE the constant: %s" % (got[0][3],)
    assert not Path(got[0][2]).exists(), (
        "the seeded wrong path %r unexpectedly exists, so this control proves nothing" % got[0][2])


def test_the_guard_accepts_a_right_path_and_ignores_a_non_path(tmp_path):
    """The other half: a right path evaluates and a non-path constant is left alone. A guard that
    fires on everything is as useless as one that fires on nothing, and would get suppressed."""
    m = tmp_path / "test_seeded_ok.py"
    m.write_text("import os\n"
                 "TIMEOUT = 30\n"
                 "URL = 'https://example.invalid/x'\n"
                 "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
                 "_SELF = os.path.join(_HERE, 'test_seeded_ok.py')\n", encoding="utf-8")
    got = {n: (v, e) for n, _l, v, e in location_derived_paths(m)}
    assert set(got) == {"_HERE", "_SELF"}, (
        "the guard picked up constants that are not location-derived: %r" % sorted(got))
    assert all(e is None for _v, e in got.values())
    assert all(Path(v).exists() for v, _e in got.values())


def test_the_guard_refuses_an_idiom_it_cannot_evaluate(tmp_path):
    """An unrecognised location-derived idiom is reported. Passing over it would leave the guard
    green across exactly the constants it could not check — the same silent-pass shape one level
    up."""
    m = tmp_path / "test_seeded_exotic.py"
    m.write_text("import os, functools\n"
                 "_X = functools.reduce(os.path.join, [__file__, '..', 'nowhere'])\n",
                 encoding="utf-8")
    got = location_derived_paths(m)
    assert len(got) == 1 and got[0][2] is None and got[0][3], (
        "an unevaluable location-derived expression was silently passed over: %r" % (got,))


def test_the_guard_catches_a_parents_index_that_is_too_deep(tmp_path):
    """`parents[N]` with N too large is the other shape a relocation leaves behind. Unlike a wrong
    `..` it raises at import rather than resolving to a plausible-looking sibling, so the guard
    names the depth."""
    m = tmp_path / "test_seeded_deep.py"
    m.write_text("from pathlib import Path\n"
                 "_R = Path(__file__).resolve().parents[40] / 'agience-mantle'\n", encoding="utf-8")
    got = location_derived_paths(m)
    assert len(got) == 1 and got[0][2] is None and "deeper" in (got[0][3] or "")


# ══ the guard itself ═════════════════════════════════════════════════════════════════════════

def test_the_guard_has_something_to_measure():
    """If the sweep below ever covers zero constants it passes for the wrong reason."""
    total = sum(len(location_derived_paths(m)) for m in _test_modules())
    assert total >= 4, (
        "the guard found only %d location-derived path constant(s) across %d test modules — it is "
        "no longer measuring anything, most likely because the evaluator's trigger broke"
        % (total, len(_test_modules())))


@pytest.mark.parametrize("module", _test_modules(), ids=lambda p: p.name)
def test_every_location_derived_path_in_a_test_module_resolves(module):
    """Every module-level path constant in a test module names something on disk.

    A path that resolves to nothing does not make a test fail: it makes it skip, or makes a
    `sys.path.insert` a no-op so the test measures the wrong tree and still reports a pass. Both are
    silent, and both survive indefinitely because the summary line looks fine.
    """
    bad = []
    for name, lineno, value, err in location_derived_paths(module):
        if err is not None:
            bad.append(
                "%s:%d  %s — this guard cannot evaluate %s. Route the path through "
                "`tests/_paths.py` (EMBER_REPO / WORKSPACE / repo(...) / ember_src(...)), which "
                "resolves it once and asserts it." % (module.name, lineno, name, err))
        elif not Path(value).exists() and not _is_absent_dependency(value):
            bad.append(
                "%s:%d  %s resolves to %s, which DOES NOT EXIST. That is a wrong path, not an "
                "absent dependency: no environment change makes it resolve, so any test gated on "
                "it skips forever and the module reports a silent pass. Use `tests/_paths.py`."
                % (module.name, lineno, name, value))
    assert not bad, "\n".join(bad)
