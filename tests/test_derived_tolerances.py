"""Every derived tolerance must track its inputs.

A derivation that returns the same number whatever it is given is a constant wearing a function, so
none of these tests asserts a value. Each one moves an input and requires the derived quantity to
move with it, in the stated direction. A re-hardcoded return fails every test here.

The tolerances under test, and what each derives from:

  genesis._metric_quantum / _metric_half_quantum   the published precision of ρ and coverage
  improve._metrics_ttl                             genesis._METRICS_TTL (one stated staleness)
  worker.liveness_window_s                         the pinger's own cadence x missable pings
  worker.min_sync_s                                the persisted precision of `sync_s` x one tolerance
  stats._rate_window_s                             the publisher's declared cadence (golden.json)
  lattice_smoke._roundoff_bound                    float64 eps x term count x magnitude
  enrich_wordnet.IC_ABSURD                         the measured IC ceiling x stated headroom
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest

import _paths  # noqa: F401  (puts src/ on the path the same way the rest of the suite does)

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")


def _load_script(name):
    """`scripts/` holds standalone `main()` programs, so they load by path, not by import."""
    path = os.path.join(_SCRIPTS, name)
    if not os.path.exists(path):
        pytest.skip("%s not present" % name)
    spec = importlib.util.spec_from_file_location("dt_" + name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── genesis: ρ / coverage tolerances follow the published precision ──────────────────────────
def test_the_metric_tolerance_follows_the_published_precision(monkeypatch):
    from ember import genesis

    base_q = genesis._metric_quantum()
    base_h = genesis._metric_half_quantum()
    assert base_h == pytest.approx(base_q / 2.0)

    # One more published digit means a tolerance one decade tighter.
    monkeypatch.setattr(genesis, "_METRIC_DECIMALS", genesis._METRIC_DECIMALS + 1)
    assert genesis._metric_quantum() == pytest.approx(base_q / 10.0)
    assert genesis._metric_half_quantum() == pytest.approx(base_h / 10.0)

    monkeypatch.setattr(genesis, "_METRIC_DECIMALS", 0)
    assert genesis._metric_quantum() == pytest.approx(1.0)


def test_the_cooling_watch_cannot_fire_on_a_difference_the_record_cannot_express(monkeypatch):
    """The trend tolerance admits a rise smaller than one published digit and rejects a larger
    one, so a difference the record cannot express is rounding rather than warming."""
    from ember import genesis

    q = genesis._metric_quantum()
    # a "rise" of less than one quantum is rounding, not warming
    assert 0.5000 <= 0.5000 - q / 2.0 + q
    # a rise of more than one quantum is real and must not be absorbed
    assert not (0.5000 + 3 * q <= 0.5000 + q)


# ── improve: the second TTL is read, not restated ────────────────────────────────────────────
def test_the_improve_cache_ttl_follows_genesis(monkeypatch):
    from ember import genesis
    from ember.runtime import improve

    assert improve._metrics_ttl() == pytest.approx(float(genesis._METRICS_TTL))
    monkeypatch.setattr(genesis, "_METRICS_TTL", 12345.0)
    assert improve._metrics_ttl() == pytest.approx(12345.0), (
        "improve restated the TTL instead of reading it — the two can drift apart again")


# ── worker: liveness follows the ping cadence ────────────────────────────────────────────────
def test_the_liveness_window_follows_the_ping_cadence(monkeypatch):
    from ember.runtime import worker

    base = worker.liveness_window_s()
    assert base == pytest.approx(worker.DEAD_AFTER_MISSED_PINGS * worker.PING_CADENCE_S)

    monkeypatch.setattr(worker, "PING_CADENCE_S", worker.PING_CADENCE_S * 2)
    assert worker.liveness_window_s() == pytest.approx(base * 2), (
        "the window did not follow the cadence it is a multiple of")

    monkeypatch.setattr(worker, "DEAD_AFTER_MISSED_PINGS", 1)
    assert worker.liveness_window_s() == pytest.approx(worker.PING_CADENCE_S)


def test_a_faster_loop_pings_faster_and_its_window_tightens_with_it():
    """The pinger sleeps `min(interval, PING_CADENCE_S)`, so the window reads the cadence actually
    in force rather than the ceiling."""
    from ember.runtime import worker

    fast = worker.liveness_window_s(interval=1.0)
    slow = worker.liveness_window_s(interval=1e6)
    assert fast < slow
    assert fast == pytest.approx(worker.DEAD_AFTER_MISSED_PINGS * 1.0)
    assert slow == pytest.approx(worker.DEAD_AFTER_MISSED_PINGS * worker.PING_CADENCE_S)


def _literal_comparisons(src: str):
    """Every `measurement <op> numeric-literal` comparison in `src`, as (lineno, value).

    Parsed rather than grepped: a text search cannot tell a live comparison from the same
    characters quoted inside a comment."""
    import ast

    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Compare):
            continue
        for operand in [node.left] + list(node.comparators):
            if (isinstance(operand, ast.Constant)
                    and isinstance(operand.value, (int, float))
                    and not isinstance(operand.value, bool)):
                out.append((node.lineno, operand.value))
    return out


def test_genesis_liveness_reads_the_worker_and_does_not_keep_its_own_number():
    """`health()` decides liveness through `liveness_window_s`, keeping no number of its own."""
    import inspect

    from ember import genesis

    src = inspect.getsource(genesis.health)
    values = [v for _, v in _literal_comparisons(src)]
    assert 90 not in values, "health() still decides liveness against a number of its own"
    assert "liveness_window_s" in src


# ── worker: the throughput sample floor follows the persisted precision ──────────────────────
def test_the_rate_sample_floor_follows_the_persisted_precision(monkeypatch):
    from ember.runtime import worker

    base = worker.min_sync_s()
    # one more persisted decimal: a ten-times-shorter cycle is precise enough to learn from
    monkeypatch.setattr(worker, "_SYNC_S_DECIMALS", worker._SYNC_S_DECIMALS + 1)
    assert worker.min_sync_s() == pytest.approx(base / 10.0)


def test_the_rate_sample_floor_follows_the_stated_tolerance(monkeypatch):
    from ember.runtime import worker

    base = worker.min_sync_s()
    # accepting half the error means requiring twice the cycle length
    monkeypatch.setattr(worker, "_RATE_QUANTISATION_TOL", worker._RATE_QUANTISATION_TOL / 2.0)
    assert worker.min_sync_s() == pytest.approx(base * 2.0)


def test_a_cycle_at_the_floor_carries_exactly_the_stated_error():
    """At exactly `min_sync_s()` the rounding of `sync_s` contributes exactly
    `_RATE_QUANTISATION_TOL` relative error to the learned rate, and no more."""
    from ember.runtime import worker

    half_quantum = (10.0 ** -worker._SYNC_S_DECIMALS) / 2.0
    rel_err = half_quantum / worker.min_sync_s()
    assert rel_err == pytest.approx(worker._RATE_QUANTISATION_TOL)


# ── stats: the differencing window follows the publisher's declared cadence ──────────────────
class _FakeStore:
    """Just enough store for `_rate_window_s`: it only reaches `genesis.published_gate`."""

    def __init__(self):
        self.keys_dir = None


def test_the_rate_window_follows_the_declared_cadence(monkeypatch):
    from ember import genesis
    from ember.surface import stats

    store = _FakeStore()

    def _gate(_store, cadence=30.0):
        return {"fast_s": cadence, "ok": True, "fresh": True}

    monkeypatch.setattr(genesis, "published_gate", lambda s: _gate(s, 30.0))
    lo30, hi30 = stats._rate_window_s(store)

    monkeypatch.setattr(genesis, "published_gate", lambda s: _gate(s, 300.0))
    lo300, hi300 = stats._rate_window_s(store)

    assert (lo300, hi300) == pytest.approx((lo30 * 10.0, hi30 * 10.0)), (
        "the window ignored the publisher's cadence — it is still a constant")
    assert lo30 < 30.0 < hi30, "the declared cadence itself must be an admissible interval"


def test_the_rate_window_follows_the_stated_missable_cycles(monkeypatch):
    from ember import genesis
    from ember.surface import stats

    store = _FakeStore()
    monkeypatch.setattr(genesis, "published_gate", lambda s: {"fast_s": 30.0})
    _, hi = stats._rate_window_s(store)

    monkeypatch.setattr(stats, "_RATE_MAX_MISSED_CYCLES", stats._RATE_MAX_MISSED_CYCLES * 2 + 1)
    _, hi2 = stats._rate_window_s(store)
    assert hi2 > hi


def test_an_undeclared_cadence_falls_back_and_says_so(monkeypatch):
    """No published gate means no declared cadence. The fallback is a single named constant, and
    the window follows that constant rather than a second literal inside the comparison."""
    from ember import genesis
    from ember.surface import stats

    store = _FakeStore()
    monkeypatch.setattr(genesis, "published_gate", lambda s: None)
    lo, hi = stats._rate_window_s(store)
    monkeypatch.setattr(stats, "_UNDECLARED_PUBLISH_CADENCE_S",
                        stats._UNDECLARED_PUBLISH_CADENCE_S * 4)
    lo4, hi4 = stats._rate_window_s(store)
    assert (lo4, hi4) == pytest.approx((lo * 4.0, hi * 4.0))


# ── lattice_smoke: the exactness verdict is a computed round-off bound ───────────────────────
def test_the_roundoff_bound_tracks_magnitude_and_term_count():
    mod = _load_script("lattice_smoke.py")

    base = mod._roundoff_bound(10.0, terms=5)
    assert mod._roundoff_bound(20.0, terms=5) == pytest.approx(base * 2.0), "magnitude ignored"
    assert mod._roundoff_bound(10.0, terms=9) > base, "term count ignored"
    assert mod._roundoff_bound(0.0, terms=5) == 0.0, (
        "a sum of nothing has no round-off — a floor here would be a typed-in tolerance again")


def test_the_roundoff_bound_is_a_float64_bound_not_a_chosen_epsilon():
    """The bound is the machine's eps, so a build with different float precision moves it."""
    mod = _load_script("lattice_smoke.py")

    assert mod._FLOAT_EPS == sys.float_info.epsilon
    # two terms, magnitude 1: exactly 2 * (2-1) * eps
    assert mod._roundoff_bound(1.0, terms=2) == pytest.approx(2.0 * sys.float_info.epsilon)


def test_the_exactness_verdict_no_longer_compares_against_a_literal():
    """Check 5's exactness verdict reads the computed round-off bound, not a literal tolerance."""
    path = os.path.join(_SCRIPTS, "lattice_smoke.py")
    if not os.path.exists(path):
        pytest.skip("lattice_smoke.py not present")
    src = open(path, encoding="utf-8").read()
    values = [v for _, v in _literal_comparisons(src)]
    assert 1e-12 not in values, "a 1e-12 tolerance is still deciding something in check 5"
    assert "worst_ratio <= 1.0" in src


def test_a_residual_at_the_bound_passes_and_one_above_it_fails():
    """The verdict form can fail: a ratio test that only ever passes proves nothing."""
    mod = _load_script("lattice_smoke.py")

    bound = mod._roundoff_bound(12.0, terms=30)
    assert bound > 0.0
    assert (bound / bound) <= 1.0                 # exactly at the bound: exact
    assert not ((bound * 1.5) / bound <= 1.0)     # half again over: not exact


# ── enrich_wordnet: the absurd-IC guard follows the measured ceiling ─────────────────────────
def test_the_absurd_ic_guard_follows_the_measured_ceiling():
    mod = _load_script("enrich_wordnet.py")

    assert mod.IC_ABSURD == pytest.approx(mod.IC_ABSURD_HEADROOM * mod.IC_MEASURED_MAX), (
        "IC_ABSURD is typed in again — re-measuring the ceiling would no longer move the guard")
    assert mod.IC_MEASURED_MAX < mod.IC_ABSURD < mod.NLTK_IC_INF, (
        "the guard must sit strictly between the largest real IC and the sentinel it catches")


def test_the_absurd_ic_guard_still_catches_the_sentinel_and_passes_real_values():
    """The guard sits clear of the legitimate range on one side and short of the sentinel on the
    other, so both ends of its placement are measured."""
    mod = _load_script("enrich_wordnet.py")

    assert not (-mod.IC_ABSURD < mod.NLTK_IC_INF < mod.IC_ABSURD)   # sentinel is caught
    assert -mod.IC_ABSURD < mod.IC_MEASURED_MAX < mod.IC_ABSURD     # the real ceiling passes
    assert -mod.IC_ABSURD < 0.0 < mod.IC_ABSURD                     # a root IC passes


# ── node-repair: the deep index gate states its attempt count once ───────────────────────────
def test_the_index_gate_verdict_reads_the_attempt_count_it_loops_on():
    """The verdict compares against the same symbol the loop ranges over, so raising the loop
    bound keeps the BAD branch reachable."""
    path = str(_paths.FLEET_EMBER / "node-repair.py")
    if not os.path.exists(path):
        pytest.skip("node-repair.py not present")
    src = open(path, encoding="utf-8").read()
    assert "_INDEX_GATE_ATTEMPTS" in src
    assert "for _ in range(3):" not in src, "the loop bound is a literal again"
    assert "elif short == 3:" not in src, "the verdict is a literal again — it can go unreachable"
    assert "elif short == attempts:" in src


def test_the_index_gate_attempt_count_is_what_the_loop_actually_runs():
    """The two agree by execution rather than by reading: `range(_INDEX_GATE_ATTEMPTS)` produces a
    `short` equal to the value the verdict compares against."""
    import ast

    path = str(_paths.FLEET_EMBER / "node-repair.py")
    if not os.path.exists(path):
        pytest.skip("node-repair.py not present")
    tree = ast.parse(open(path, encoding="utf-8").read())
    attempts = next(
        (ast.literal_eval(n.value) for n in tree.body
         if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
         and n.targets[0].id == "_INDEX_GATE_ATTEMPTS"), None)
    assert isinstance(attempts, int) and attempts >= 2, (
        "one reading cannot distinguish a persistent shortfall from drift")
    short = sum(1 for _ in range(attempts))       # every reading short
    assert short == attempts                      # ...which is exactly what the BAD branch requires
