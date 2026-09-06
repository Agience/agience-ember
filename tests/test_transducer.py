"""The keyed ontology path. A tiny WordNet in a throwaway store, `seed_lattice.build`, then the chat
path measured keyed end to end — stored IC, keyed lookup/morphy/hypernyms, full-DAG propagation, and
a query that loads no full corpus.

Runs in an isolated subprocess: the store binding and the driver's module singletons are process
state shared with the live-store tests, so the scenario gets its own process and temp shard.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

# The scenario, run in a fresh process (see _main). Kept as source so the test is self-contained.
_SCENARIO = r'''
import os, sys, tempfile, shutil
TMP = tempfile.mkdtemp(prefix="transducer_test_")
os.environ.update(EMBER_SQLITE_DIR=TMP, EMBER_NODE_ID="transducertest", EMBER_PRINCIPAL="author@example.com",
                  EMBER_SQLITE_CREATE="1", PYTHONIOENCODING="utf-8")
from cryptography.fernet import Fernet
K = os.path.join(TMP, "keys"); os.makedirs(K, exist_ok=True)
open(os.path.join(K, "content.key"), "wb").write(Fernet.generate_key())
os.environ["EMBER_STORE_KEYS_DIR"] = K

ok = True
def check(name, cond):
    global ok; ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + name)

try:
    from mantle.shard.sqlite_store import open_sqlite_store
    from crystal.ontology import driver as wn
    from crystal.ontology import transducer
    from crystal.ontology import seed_lattice
    from ember.ontology import activation as A
    from ember.ontology import match as M
    store = open_sqlite_store(); arts = store.artifacts
    def syn(name, lemmas, counts, pos="n"):
        return {"id": "wn-"+name, "content_type": "text/x-wordnet", "state": "committed", "pos": pos,
                "lemmas": [l.lower() for l in lemmas], "lemma_counts": counts,
                "sense_ranks": {l: 0 for l in lemmas}, "word": lemmas[0].lower(),
                "content": name+" g", "gloss": name+" g", "provenance": "observed",
                "created_by": "author@example.com", "created_time": "2026-07-28T00:00:00Z"}
    arts.put_many([
        syn("dog.n.01", ["dog", "domestic_dog"], {"dog": 42, "domestic_dog": 0}),
        syn("canine.n.02", ["canine"], {"canine": 3}),
        syn("animal.n.01", ["animal"], {"animal": 10}),
        syn("cat.n.01", ["cat"], {"cat": 18}),
        syn("run.v.01", ["run"], {"run": 5}, pos="v")])
    store.graph.add_edges([("wn-dog.n.01","wn-canine.n.02","hypernym",{}),
                           ("wn-canine.n.02","wn-animal.n.01","hypernym",{}),
                           ("wn-cat.n.01","wn-animal.n.01","hypernym",{})], batch=100)

    check("pre-build keyed_ready False", wn._keyed_ready() is False)
    wn._INDEX = None
    check("pre-build legacy synsets('dog')", [s.name() for s in wn.synsets("dog")] == ["dog.n.01"])

    wn._KEYED_READY = None
    seed_lattice.build(store, log=lambda *a: None)

    check("keyed_ready True", wn._keyed_ready() is True)
    check("IC stored (>0)", wn.synset("dog.n.01").ic() > 0.0)
    check("keyed synsets('dog')", [s.name() for s in wn.synsets("dog")] == ["dog.n.01"])
    check("keyed morphy plural", [s.name() for s in wn.synsets("dogs")] == ["dog.n.01"])
    check("keyed morphy verb rule", wn.morphy("runs", "v") == "run")
    check("keyed morphy refuses non-lemma", wn.morphy("xyzzys", "n") is None)
    check("keyed hypernyms", [h.name() for h in wn.synset("dog.n.01").hypernyms()] == ["canine.n.02"])

    # The entry/render/lossless/frame conversion classes live in `lumen/transducer.py`, which ember
    # does not import, so the `language.en` round-trip is covered by lumen's own suite. What
    # follows is ember's keyed path.

    fired = A.spread_seeds(A._seed_field(store, "dog"))
    words = {A._word(n) for n in fired}
    check("propagation full-DAG keyed", "canine" in words and "animal" in words)

    wn._INDEX = None
    _ = wn.synsets("cat"); _ = wn.synset("cat.n.01").hypernyms()
    check("no full load on chat path", wn._INDEX is None)

    # `_GEOM_CACHE` holds xi and the propagation floor as one cached geometry read, so the two agree by
    # construction. Cleared by name rather than through getattr-with-default, so a rename fails
    # here instead of silently skipping the check.
    M._GEOM_CACHE.clear()
    xi1 = M.xi()                        # cold: the whole-corpus derivation is paid here
    _n = [0]
    _real_derive = M._derive_geometry
    M._derive_geometry = lambda *a, **k: (_n.__setitem__(0, _n[0] + 1), _real_derive(*a, **k))[1]
    xi2 = M.xi()                        # warm: served from the cache
    M._derive_geometry = _real_derive
    check("xi() keyed, derives a positive scale", isinstance(xi1, float) and xi1 > 0.0)
    check("xi() is stable across calls", xi2 == xi1)
    check("warm xi() does NOT re-run the whole-corpus derivation", _n[0] == 0)
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print("RESULT:", "ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
'''


def _main() -> None:
    exec(compile(_SCENARIO, "<transducer-scenario>", "exec"), {})


def test_transducer_keyed_path():
    """The full keyed pipeline, in an isolated process against a throwaway shard."""
    try:
        import cryptography  # noqa: F401
    except Exception:                                    # pragma: no cover - environment gate
        pytest.skip("cryptography unavailable")
    # This file is loaded by path rather than by module name: the subprocess exists for process
    # isolation, not for how the scenario is imported. cwd is the ember repo root, which is what
    # `_main` expects.
    _here = os.path.abspath(__file__)
    proc = subprocess.run([sys.executable, "-c",
                           "import importlib.util as u,sys;"
                           f"spec=u.spec_from_file_location('_t', r'{_here}');"
                           "m=u.module_from_spec(spec);sys.modules['_t']=m;"
                           "spec.loader.exec_module(m);m._main()"],
                          capture_output=True, text=True, env=dict(os.environ),
                          cwd=os.path.dirname(os.path.dirname(_here)))
    out = proc.stdout + "\n" + proc.stderr
    assert "RESULT: ALL PASS" in out, out
    assert proc.returncode == 0, out
