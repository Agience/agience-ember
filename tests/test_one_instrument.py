"""The instrument is reached through one instrument: `ember/optics.py` is the only module in the
scanned repos that imports `entroptics`.

The instrument exists for a measured reason recorded in its own header. `entroptics.read()` and
`Screen(W)` bypass the streaming front door and apply the entropy fold guard that the library
documents as destroying a sparse carrier — 256 feature channels folded to F_eff = 1 and reported as
`K_signal = 1`, which at the call site is indistinguishable from "there is one real mode". Ontology
coordinates are sparse, so a direct import changes what a read means.

This file is the enforcement, because a convention that is only written down drifts: nothing stops a
caller from reaching past the wrapper and calling `entroptics.read()` directly. The scan is measured
against the tree rather than trusted, and each of its own guarantees carries a control, so that a
scanner catching everything and a scanner catching nothing both go red.
"""
import ast
import pathlib

# The instrument itself is the only place entroptics is named. Everything else asks the wrapper.
#
# The exemption matches on BASENAME, so a second `optics.py` in any scanned repo would inherit it.
# `test_the_exemption_names_exactly_one_file` measures the exemption against the tree and pins it to
# the one real instrument, so a second door goes red here rather than passing on its filename.
_ALLOWED = {"optics.py"}

# The repos under enforcement. `_archive/` is outside the scan: it is a record rather than code that
# runs. `test_every_declared_repo_is_actually_scanned` holds this list to the tree, so a stale entry
# cannot shrink the scan to whatever happens to be on disk.
_REPOS = ("agience-mantle", "agience-ember", "agience-crystal",
          "agience-chorus", "agience-origin", "agience-prism/py")


def _roots():
    here = pathlib.Path(__file__).resolve()
    genesis = here.parents[2]          # tests/ -> agience-ember/ -> the workspace
    for r in _REPOS:
        src = genesis / r / "src"
        if src.is_dir():
            yield r, src


def _scan(roots):
    """Return (offenders, unparsed) over `roots` -- an iterable of (label, dir).

    Split out from `_offenders()` so the scanner can be pointed at a throwaway tree and shown to have
    teeth (`test_the_scanner_actually_catches_a_seeded_violation`). A scanner only ever run over a
    clean tree is indistinguishable from one that returns [] unconditionally.
    """
    bad, unparsed = [], []
    for repo, src in roots:
        for path in src.rglob("*.py"):
            if path.name in _ALLOWED or "__pycache__" in path.parts:
                continue
            if path.name.startswith("test_"):
                continue                     # a test may name it to assert about it
            # `utf-8-sig` rather than `utf-8` -- see `test_the_scan_can_parse_every_file_it_claims_
            # to_scan`. 48 files in mantle/origin carry a UTF-8 BOM; under plain `utf-8` the BOM
            # decodes to a leading U+FEFF and `ast.parse` raises SyntaxError. `utf-8-sig` strips a
            # BOM when there is one and is a no-op when there is not.
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
            except Exception as exc:
                # A file that cannot be read is unenforced rather than clean, so it is collected and
                # asserted on rather than dropped.
                unparsed.append("%s/%s  %s: %s"
                                % (repo, path.relative_to(src), type(exc).__name__, exc))
                continue
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                for m in mods:
                    if m == "entroptics" or m.startswith("entroptics."):
                        bad.append("%s/%s:%d  imports %s"
                                   % (repo, path.relative_to(src), node.lineno, m))
    return bad, unparsed


def _offenders():
    return _scan(_roots())[0]


def test_every_declared_repo_is_actually_scanned():
    """`_roots()` yields only repos whose `src/` is on disk, so a declared repo that has moved would
    shrink enforcement to whatever remains while the scan below still reported green. A repo path is
    the kind of thing a regroup changes, so declared-but-missing is an error here rather than a
    quiet omission from the scan.
    """
    genesis = pathlib.Path(__file__).resolve().parents[2]
    missing = [r for r in _REPOS if not (genesis / r / "src").is_dir()]
    assert not missing, (
        "declared in _REPOS but has no `src/` on disk: " + ", ".join(missing)
        + "\n\nThese would be skipped silently, so the instrument would go unenforced there. Fix the "
          "path or delete the entry -- never leave one to be quietly passed over."
    )


def test_the_exemption_names_exactly_one_file():
    """The instrument is one module, and `_ALLOWED` matches on basename.

    A second `optics.py` in any scanned repo would be exempted by its name and could import
    entroptics freely -- a second door, invisible to every other test here. So the exemption is
    measured against the tree rather than trusted: the scan finds exactly one file it exempts, and
    it is ember's.

    A rule keyed on a filename survives a move of the file it names, which is why the address is
    pinned rather than assumed.
    """
    found = sorted(
        "%s/%s" % (repo, path.relative_to(src).as_posix())
        for repo, src in _roots()
        for path in src.rglob("*.py")
        if path.name in _ALLOWED and "__pycache__" not in path.parts
    )
    assert found == ["agience-ember/ember/optics.py"], (
        "expected exactly one exempted file -- ember's instrument -- and found %r.\n"
        "Every name in this list may import entroptics with nothing checking it. If a second door "
        "is genuinely wanted, that is a decision for John, not a filename collision." % (found,))


def test_the_scan_can_parse_every_file_it_claims_to_scan():
    """A file the scanner cannot parse is unenforced, so an unreadable file goes red here rather
    than reading as clean.

    48 of the 432 files the scan walks -- 28 in mantle, 20 in origin, 11% of its declared surface --
    carry a UTF-8 BOM, which raises SyntaxError under plain `utf-8`. They recover under `utf-8-sig`.
    `test_every_declared_repo_is_actually_scanned` covers a different question: it asserts each
    repo's `src/` exists, not that its files are readable, so the scan could walk a whole repo and
    enforce nothing there with that check still green.

    [[verification-that-cannot-fail]] -- state the failure mode first.
    """
    unparsed = _scan(_roots())[1]
    assert not unparsed, (
        "the instrument scan could not parse %d file(s), so entroptics imports in them are UNENFORCED:\n  "
        % len(unparsed) + "\n  ".join(unparsed)
        + "\n\nDo not re-add a bare `except: continue` -- that is what hid 48 files. Either make the file "
          "parseable or widen the decode, and keep this assertion."
    )


def test_the_scanner_actually_catches_a_seeded_violation(tmp_path):
    """The negative control. `test_only_the_instrument_seam_imports_entroptics` passes when the tree is
    clean, and would pass identically if `_scan` returned [] unconditionally. So every violation
    form is seeded into a throwaway tree along with the exemptions, and both are asserted: a scanner
    catching everything and one catching nothing each go red.

    Falsifiable as written: reverting the decode to plain `utf-8` fails this on `bom.py`, and making
    `_scan` return [] unconditionally fails it on `plain.py`.
    """
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "plain.py").write_text("import entroptics\n", encoding="utf-8")
    (src / "pkg" / "submodule.py").write_text("from entroptics.read import x\n", encoding="utf-8")
    (src / "pkg" / "bom.py").write_text("import entroptics\n", encoding="utf-8-sig")   # the 48-file case
    (src / "pkg" / "innocent.py").write_text("import numpy\nfrom ember import optics\n", encoding="utf-8")
    (src / "pkg" / "test_named.py").write_text("import entroptics\n", encoding="utf-8")   # exempt: a test
    (src / "optics.py").write_text("import entroptics\n", encoding="utf-8")              # exempt: the door

    bad, unparsed = _scan([("fake", src)])
    hits = " ".join(bad)

    assert not unparsed, "the seeded tree should parse cleanly, got: %s" % unparsed
    assert "plain.py" in hits, "a bare `import entroptics` was MISSED -- the scanner has no teeth"
    assert "submodule.py" in hits, "`from entroptics.read import ...` was MISSED (submodule form)"
    assert "bom.py" in hits, (
        "a BOM-prefixed violation was MISSED -- this is the exact 48-file regression; the decode "
        "reverted to plain `utf-8`")
    assert "innocent.py" not in hits, "flagged a file that does not import entroptics"
    assert "test_named.py" not in hits, "the `test_*` exemption stopped working"
    assert "optics.py" not in hits, "the instrument itself was flagged -- `_ALLOWED` stopped working"
    assert len(bad) == 3, "expected exactly the 3 seeded violations, got %d: %s" % (len(bad), bad)


def test_only_the_instrument_seam_imports_entroptics():
    bad = _offenders()
    assert not bad, (
        "these reach past the instrument (`ember.optics`) into entroptics directly:\n  "
        + "\n  ".join(bad)
        + "\n\nThe wrapper exists because `read()`/`Screen()` are the wrong door: they bypass the "
          "streaming front door and apply the entropy fold guard that destroys a sparse carrier "
          "(measured: 256 channels -> F_eff=1, reported as K_signal=1). Add the read you need to "
          "`ember/optics.py` and call it from there."
    )
