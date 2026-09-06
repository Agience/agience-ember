"""Benchmark: the deterministic dev edge, on the Agience codebase.

Every question here has one correct answer that is a fact of this private repo — a file
path, a line, a set of call sites. `lumen.code` answers from its AST index, so the answer
is exact by construction. A pretrained model has no access to these facts, which is what
the benchmark measures: exact against plausible.

Run:  python agience-ember/tests/e2e/bench_dev.py
"""
from __future__ import annotations

import sys
import time


# (question, must_contain_all) — must_contain checked against Ember's answer text.
CASES = [
    ("where is open_store defined",        ["local_store.py"]),
    ("where is ArcadeConnection defined",  ["arcade.py"]),
    ("where is register_dev_operators defined", ["dev_ops.py"]),   # private symbol, unguessable
    ("where is find_references defined",   ["code.py"]),           # private symbol, unguessable
    ("where is index_code defined",        ["code.py"]),           # private symbol, unguessable
    ("what calls add_edges",               ["ingest.py"]),
    ("who calls lookup_by_lemma",          ["code.py"]),
    ("where is route defined",             ["router.py"]),
    # non-code domains, still exact/ground-truth:
    ("define recursion",                   ["recursion"]),
    ("what is a dog a kind of",            ["animal"]),
    ("what is 7 * 8",                      ["56"]),
]


def main() -> int:
    from mantle.shard.local_store import open_store
    from ember.corpus.ingest import open_local
    # The router lives in lumen: it dispatches and it composes prose, which are persona acts.
    # This script is operator tooling rather than the ember library, so it may import a persona
    # directly; the "ember never imports chorus" rule guards `src/ember`, and the CI gate enforces it
    # by omitting chorus from ember's PYTHONPATH. Without lumen on the path this script says so.
    try:
        from lumen import router
    except ImportError as _e:                        # pragma: no cover - operator tooling
        raise SystemExit(
            "the router lives in lumen. Put agience-chorus/src on "
            "PYTHONPATH to run this diagnostic, or reach `op.route` over the plane.") from _e
    store = open_store()
    ember = open_local()

    passed, total_ms = 0, 0.0
    print(f"{'domain':16} {'ms':>6}  question -> verdict")
    for q, needles in CASES:
        t = time.perf_counter()
        ans, dom = router.route(store.artifacts, ember, q, graph_store=store.graph, k=4)
        ms = (time.perf_counter() - t) * 1000
        total_ms += ms
        text = ans.text
        ok = ans.grounded and all(n.lower() in text.lower() for n in needles)
        passed += ok
        print(f"{dom:16} {ms:6.1f}  {'PASS' if ok else 'FAIL'}  {q!r}"
              + ("" if ok else f"  expected {needles}, got: {text.splitlines()[0][:50]!r}"))

    n = len(CASES)
    print(f"\nExactness: {passed}/{n} ({100*passed//n}%)   avg {total_ms/n:.1f} ms/query")
    print("Note: register_dev_operators / find_references / index_code were written THIS session —"
          " un-guessable by any pretrained model. That's the un-fakeable deterministic edge.")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
