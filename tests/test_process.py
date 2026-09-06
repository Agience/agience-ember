"""Unit tests for the deterministic dev-copilot process — pure functions with no live infra
(ArcadeDB and Garage faked in memory), so they run anywhere and pin the pipeline's behaviour.

Covers: content (encrypt/CAS/resolve), code_index (AST extract), doc_index (keyphrase), corpus
(dedup shingles/Jaccard), evolution (fitness/select/retire/sweep), dev_ops (gated rename) and
fetch (GET-only, SSRF-guarded).
"""
import tempfile
from pathlib import Path

from mantle.shard import content as C                    # the shard's content-addressed store
from ember.runtime.runner import code_index, corpus, doc_index, evolution   # the distributed bundles


# ── content: encrypted, content-addressed, resolvable ─────────────────────────
class _MemContent:
    def __init__(self): self.d = {}
    def put(self, k, data, content_type="application/octet-stream"): self.d[k] = data
    def get(self, k): return self.d[k]
    def exists(self, k): return k in self.d
    def delete(self, k): self.d.pop(k, None)


class _Bundle:
    def __init__(self, content, keys_dir): self.content = content; self.keys_dir = keys_dir


def test_content_ref_is_deterministic_plaintext_hash():
    assert C.content_ref(b"hello") == C.content_ref(b"hello")
    assert C.content_ref(b"hello") != C.content_ref(b"world")
    assert C.content_ref(b"hello").startswith("cas/")


def test_content_roundtrip_encrypted_and_dedup():
    kd = Path(tempfile.mkdtemp()); cs = _MemContent()
    data = b"def f():\n    return 1\n" * 500          # >20k, no truncation
    ref, size = C.put_content(cs, kd, data)
    assert size == len(data)
    assert cs.d[ref] != data                         # stored ciphertext, not plaintext
    assert C.get_content(cs, kd, ref) == data        # decrypts back exactly
    ref2, _ = C.put_content(cs, kd, data)
    assert ref2 == ref and len(cs.d) == 1            # identical content dedupes


def test_resolve_text_prefers_garage_then_inline():
    kd = Path(tempfile.mkdtemp()); cs = _MemContent()
    ref, _ = C.put_content(cs, kd, b"full garage content")
    b = _Bundle(cs, kd)
    assert C.resolve_text(b, {"content_ref": ref}) == "full garage content"
    assert C.resolve_text(b, {"content": "inline only"}) == "inline only"   # no ref -> inline


# ── code_index: AST symbol/call extraction ────────────────────────────────────
_SRC = '''\
import os
class Widget(Base):
    def frob(self):
        return helper(self.x)
def helper(x):
    return os.path.join(x, "y")
CONST = 3
'''


def test_extract_symbols_calls_kinds():
    syms, imports = code_index.extract("pkg/mod.py", _SRC)
    by = {s.name: s for s in syms}
    assert by["Widget"].kind == "class" and by["Widget"].bases == ["Base"]
    assert by["frob"].kind == "method" and "helper" in by["frob"].calls
    assert by["helper"].kind == "function" and "join" in by["helper"].calls
    assert by["CONST"].kind == "constant"
    assert "os" in imports


def test_extract_syntax_error_is_empty_not_crash():
    assert code_index.extract("x.py", "def (:\n") == ([], [])


def test_module_of_strips_src():
    assert code_index.module_of("a/agience-mantle/src/mantle/shard/local_store.py") == "mantle.shard.local_store"


# ── doc_index: deterministic keyphrase + domain/identifier terms ──────────────
def test_doc_lemmas_capture_title_and_domain_terms():
    text = "# ArcadeDB Migration\n\nMove the store to ArcadeDB. Uses opencypher and lookup_by_lemma."
    lemmas = doc_index.extract_terms("ARCADEDB-SCOPE.md", text)
    assert "arcadedb" in lemmas and "migration" in lemmas
    assert "opencypher" in lemmas                     # domain term
    assert "lookup_by_lemma" in lemmas                # snake_case identifier


# ── corpus: shingled near-duplicate detection ─────────────────────────────────
def test_shingle_jaccard_detects_near_dupes():
    a = "the quick brown fox jumps over the lazy dog again and again today"
    b = "the quick brown fox jumps over the lazy dog again and again tonight"
    c = "completely unrelated content about databases and vector search here"
    assert C_jac(a, b) > 0.7          # near-identical
    assert C_jac(a, c) < 0.1          # unrelated


def C_jac(a, b, k=5):
    return corpus._jaccard(corpus._shingles(a, k), corpus._shingles(b, k))


# ── evolution: fitness / select / retire ──────────────────────────────────────
def test_fitness_rewards_verified_penalises_refuted():
    proven = {"verified": 20, "refuted": 0, "invocations": 20}
    unproven = {"invocations": 0}
    bad = {"verified": 1, "refuted": 20, "invocations": 21}
    assert evolution.fitness(proven) > evolution.fitness(unproven) > evolution.fitness(bad)
    assert 0.0 <= evolution.fitness(bad) < 0.5 < evolution.fitness(proven) <= 1.0


def test_select_picks_fittest():
    class S:
        def __init__(self, m): self.m = m
        def get_artifact(self, i): return self.m.get(i)
    store = S({"a": {"id": "a", "verified": 1, "refuted": 9, "invocations": 10},
              "b": {"id": "b", "verified": 9, "refuted": 1, "invocations": 10}})
    assert evolution.select(store, ["a", "b"]) == "b"


# ── evolution: record + preserve + retire (the persistence/selection substrate) ──
class _MemStore:
    """In-memory ArtifactStore stand-in: get/put/list by id and content_type."""
    def __init__(self): self.m = {}
    def get_artifact(self, i): return self.m.get(i)
    def put_artifact(self, a): self.m[a["id"]] = dict(a); return a
    def list_artifacts(self, content_type=None, **_):
        return [dict(a) for a in self.m.values()
                if content_type is None or a.get("content_type") == content_type]


def test_record_invocation_counts_gate_outcomes():
    s = _MemStore(); s.put_artifact({"id": "op.x"})
    evolution.record_invocation(s, "op.x", verified=True)
    evolution.record_invocation(s, "op.x", verified=False)
    evolution.record_invocation(s, "op.x")
    op = s.get_artifact("op.x")
    assert op["invocations"] == 3 and op["verified"] == 1 and op["refuted"] == 1
    evolution.record_invocation(s, "op.missing")     # unknown id -> no-op, no crash


def test_record_use_credits_producing_operator():
    s = _MemStore()
    s.put_artifact({"id": "op.describe.python"})
    s.put_artifact({"id": "sym-widget", "via": "op.describe.python"})
    evolution.record_use(s, "sym-widget")
    assert s.get_artifact("sym-widget")["uses"] == 1
    assert s.get_artifact("op.describe.python")["output_uses"] == 1   # demand flows to producer


def test_preserve_fitness_survives_reregistration():
    s = _MemStore()
    s.put_artifact({"id": "op.y", "invocations": 7, "verified": 5, "refuted": 1,
                    "output_uses": 3, "uses": 2})
    fresh = {"id": "op.y", "content": "re-registered", "context": "offer"}
    merged = evolution.preserve_fitness(s, fresh)
    assert merged["invocations"] == 7 and merged["verified"] == 5 and merged["output_uses"] == 3
    # a brand-new operator carries no phantom counters
    assert "invocations" not in evolution.preserve_fitness(s, {"id": "op.new"})


def test_retire_only_the_proven_unfit():
    """Retirement is `fitness(op) + z(far)·fitness_resolution(op) < prior_fitness()`: the
    operator's fitness sits below the agnostic prior by more than `z` resolutions of its own
    evidence.

    Every quantity is read from the module — no number below is typed into this test — so it fails
    if:
      · `retire` and `is_unfit` stop agreeing, or `is_unfit` stops being that inequality;
      · the evidence guard stops being a consequence of the arithmetic (0, 1 and 2 refutations with
        no successes are spared, and the third is the first that can condemn);
      · the rule stops moving in both directions against a fixed fitness floor — the two pinned
        records below sit on opposite sides of any such floor;
      · retirement becomes decidable from an invocation count at all."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - evolution.DEFAULT_RETIREMENT_FAR)
    prior = evolution.prior_fitness()

    def rec(i, ver, ref):
        return {"id": i, "verified": ver, "refuted": ref, "invocations": ver + ref}

    def condemns(op):
        """The rule restated from its own parts — an independent read of the same inequality."""
        return evolution.fitness(op) + z * evolution.fitness_resolution(op) < prior

    s = _MemStore()
    s.put_artifact(rec("op.bad", 1, 30))          # reality has spoken, unambiguously
    assert evolution.retire(s, "op.bad") is True
    assert s.get_artifact("op.bad")["state"] == "archived"

    # 1. The evidence guard, derived. Nothing is known about a record with no successes and fewer
    #    than three refutations: its own resolution is wider than the gap it opens on the prior.
    for n in (0, 1, 2):
        op = rec("op.q%d" % n, 0, n)
        assert not condemns(op), "%d refutation(s) should not clear this record's own resolution" % n
        s.put_artifact(op)
        assert evolution.retire(s, op["id"]) is False
    third = rec("op.q3", 0, 3)
    assert condemns(third), "the third consecutive refutation must be the first that can condemn"

    # 2. The rule moves in both directions against a fixed fitness floor: these two records sit on
    #    opposite sides of one, which is the pairing no floor can reproduce.
    ambiguous = rec("op.amb", 8, 12)              # 20 invocations, fitness 0.327 — below the prior
    assert evolution.fitness(ambiguous) < prior, "the fixture no longer sits below the prior"
    assert not condemns(ambiguous), (
        "an 8/12 record was condemned: its gap on the prior is smaller than its own evidence can "
        "resolve, so this is a finding the evidence does not support")
    sparse = rec("op.sparse", 0, 3)               # 3 invocations, condemned on evidence not count
    assert condemns(sparse), "a clear 0/3 failure was spared — min_invocations is back"

    # 3. No invocation count decides. `sparse` has 3 invocations and is condemned while `ambiguous`
    #    has 20 and is spared, so no threshold on invocations can reproduce this pair.
    assert sparse["invocations"] < ambiguous["invocations"]
    for op in (ambiguous, sparse):
        s.put_artifact(op)
        assert evolution.retire(s, op["id"]) is condemns(op), \
            "retire disagrees with the rule it is supposed to apply, for %s" % op["id"]


def test_rank_orders_operators_fittest_first():
    s = _MemStore()
    OP = "application/vnd.agience.operator+json"
    s.put_artifact({"id": "op.lo", "content_type": OP, "verified": 1, "refuted": 9, "invocations": 10})
    s.put_artifact({"id": "op.hi", "content_type": OP, "verified": 9, "refuted": 1, "invocations": 10})
    s.put_artifact({"id": "not-an-op", "content_type": "text/plain"})
    ranked = [o["id"] for o in evolution.rank_operators(s)]
    assert ranked == ["op.hi", "op.lo"]                  # non-operators excluded, fittest first


# ── dev_ops: gated multi-file rename (keep iff green, else revert all) ────────
def test_rename_symbol_keeps_when_gate_green_reverts_when_red():
    from ember.runtime.runner import dev_ops as D
    ws = Path(tempfile.mkdtemp())
    (ws / "m.py").write_text("def old_name():\n    return 1\n", encoding="utf-8")
    (ws / "use.py").write_text("from m import old_name\nx = old_name()\nold_names = 9\n", encoding="utf-8")
    # `dev_ops` resolves the workspace lazily behind `_workspace()`, caching into the module-global
    # `_WORKSPACE`; setting that global is the documented short-circuit (`if _WORKSPACE is not None:
    # return _WORKSPACE`). This test patched a PUBLIC `WORKSPACE` that the module has not had since
    # the lazy resolver landed — it kept passing only because the SHIPPED BUNDLE was stale and still
    # carried the old name. Rebuilding the bundles surfaced it.
    saved_ws, saved_run = D._WORKSPACE, D._run_tests
    D._WORKSPACE = ws
    try:
        D._run_tests = lambda a: {"passed": True, "summary": "2 passed", "output": ""}
        r = D._rename_symbol({"old": "old_name", "new": "new_name"})
        assert r["accepted"] and r["files"] == 2 and r["sites"] == 3   # def + import + call, not old_names
        assert "new_name" in (ws / "use.py").read_text() and "old_name(" not in (ws / "use.py").read_text()
        assert "old_names = 9" in (ws / "use.py").read_text()          # whole-word: substring untouched

        # gate red -> every touched file reverts, tree left exactly as it was
        D._run_tests = lambda a: {"passed": False, "summary": "1 failed", "output": "boom"}
        before = (ws / "m.py").read_text()
        r2 = D._rename_symbol({"old": "new_name", "new": "renamed"})
        assert r2["accepted"] is False and r2["reverted"] is True
        assert (ws / "m.py").read_text() == before                     # fully restored
    finally:
        D._WORKSPACE, D._run_tests = saved_ws, saved_run


# ── fetch: GET-only, SSRF-guarded, structurally read-only ─────────────────────
def test_fetch_refuses_non_http_and_internal_hosts():
    from ember.runtime.runner import fetch
    assert fetch._get({"url": "file:///etc/passwd"})["ok"] is False        # scheme blocked
    assert fetch._get({"url": "ftp://example.com/x"})["ok"] is False        # scheme blocked
    assert fetch._get({"url": "http://127.0.0.1/x"})["ok"] is False         # loopback blocked
    assert fetch._get({"url": "http://192.168.0.1/x"})["ok"] is False       # RFC-1918 blocked
    assert fetch._get({"url": "http://localhost:9000/x"})["ok"] is False    # loopback name blocked
    assert fetch._blocked_host("10.0.0.1") and fetch._blocked_host("169.254.1.1")
    assert not fetch._blocked_host("example.com")                          # public host allowed through


def test_fetch_has_no_write_verb_path():
    """Structurally GET-only: there is exactly one handler, and no operator name maps to a
    POST/PUT/DELETE path."""
    from ember.runtime.runner import fetch
    assert set(fetch._HANDLERS) == {"op.fetch.get"}
    import inspect
    src = inspect.getsource(fetch._get)
    assert 'method="GET"' in src                                     # verb pinned to GET
    for verb in ('method="POST"', 'method="PUT"', 'method="DELETE"', 'method="PATCH"'):
        assert verb not in src                                       # no write verb constructed
    assert "data=" not in src                                        # no request body ever sent


def test_sweep_retire_sheds_only_proven_unfit():
    s = _MemStore()
    OP = "application/vnd.agience.operator+json"
    s.put_artifact({"id": "op.bad", "content_type": OP, "verified": 1, "refuted": 40, "invocations": 41})
    s.put_artifact({"id": "op.good", "content_type": OP, "verified": 40, "refuted": 1, "invocations": 41})
    s.put_artifact({"id": "op.young", "content_type": OP, "verified": 0, "refuted": 2, "invocations": 2})
    retired = evolution.sweep_retire(s)
    assert retired == ["op.bad"]
    assert s.get_artifact("op.bad")["state"] == "archived"
    assert s.get_artifact("op.good").get("state") != "archived"
    assert s.get_artifact("op.young").get("state") != "archived"           # too little evidence to condemn








if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print(f"ALL {len(fns)} PROCESS TESTS PASS"); sys.exit(0)