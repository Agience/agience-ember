"""The self-improving loop — the system measurably improves itself over time.

Each cycle is deterministic, gated, and runs no LLM:
  1. Illuminate dark matter — describe undescribed artifacts (make them keyed).
  2. Measure — take a metrics snapshot (keyed coverage, dark matter, dedup pressure, operator
     fitness).
  3. Record — append the snapshot to a trend log, so improvement is visible (dark matter toward 0,
     coverage toward 100%, operators accruing fitness).

The loop runs in the background of `serve`; `/status` renders the trend. The goal is a system whose
corpus gets more complete, whose knowledge stays current, and whose operators are selected by
verified fitness — all on its own, deterministically.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_STRUCTURAL = {"text/x-python-symbol", "application/vnd.agience.operator+json",
               "application/vnd.agience.source+json"}
OPERATOR_CT = "application/vnd.agience.operator+json"


_METRICS_CACHE: Dict = {"ts": 0.0, "val": None, "store": None}


def _metrics_ttl() -> float:
    """The cache TTL, read from `genesis` rather than restated here.

    There is one stated tolerance for how stale a corpus scan may be (`genesis._METRICS_TTL`,
    stated with its reasoning at its definition); this cache obeys that same tolerance because it
    is caching the same scan under load."""
    from ember import genesis
    return float(genesis._METRICS_TTL)


def metrics(bundle, *, cache_ok: bool = True) -> Dict:
    """A deterministic snapshot of system health — the numbers self-improvement moves.

    Includes rho (ρ), the entropy gauge = bytes(generators) / bytes(corpus): 1.0 until
    consolidation draws `consolidates` edges, then falling as members become reconstructible. The
    corpus is cooling when rho falls while coverage holds (§7).

    Cached (like genesis.all_metrics), because this is a full scan of the corpus and the status
    page calls it on every load: without the cache, /status cold-scans 240k+ artifacts under write
    load and hangs.

    The full pass stays a full pass. Rho (generator_bytes/total_bytes) and keyed_coverage are
    ratios over the whole corpus — every artifact contributes bytes, so bounding the scan would
    change the number itself, not just its cost, making it a different metric rather than a faster
    one. Nothing here is index-servable either (`lemmas IS NULL` can never be index-served, LSM
    nullStrategy is SKIP; state='committed' matches nearly every row and prunes nothing). The cache
    is therefore the whole mitigation, and its cadence matters: improve_cycle (every worker tick)
    and browse.status_page (every /status load, and that page self-refreshes every 180s) plus
    serve's own 300s improve.start loop all call in. A TTL of 90s means a tick faster than that
    reuses the previous scan, and the trend log records a snapshot up to 90s old — acceptable
    because these are monotone drift numbers, not events.

    The by_content_type map this produces is also the cheap source for per-type counts elsewhere
    (browse.status_page, stats.write_stats): one scan, many readers, instead of a live count(*) on
    bulk values, which an index does not make cheap (measured on node 71: text/markdown timed out
    at 120s while a 48-row type took 0.12s)."""
    import time as _t
    sid = id(bundle.artifacts)
    if (cache_ok and _METRICS_CACHE["val"] is not None and _METRICS_CACHE["store"] == sid
            and (_t.time() - _METRICS_CACHE["ts"]) < _metrics_ttl()):
        return _METRICS_CACHE["val"]
    from ember import genesis
    from ember.runtime.runner import evolution
    store = bundle.artifacts
    consolidated = genesis._consolidated_members(bundle)
    total = 0
    by_type: Dict[str, int] = {}
    dark = keyed = content_docs = 0
    total_bytes = gen_bytes = 0
    fitnesses: List[float] = []
    for a in store.list_artifacts(state="committed"):
        total += 1
        ct = a.get("content_type", "?")
        by_type[ct] = by_type.get(ct, 0) + 1
        b = genesis._artifact_bytes(a)
        total_bytes += b
        if a["id"] not in consolidated:
            gen_bytes += b
        if ct == OPERATOR_CT:
            fitnesses.append(evolution.fitness(a))
        if ct not in _STRUCTURAL and ct != "text/x-wordnet":
            content_docs += 1
            if a.get("lemmas"):
                keyed += 1
            else:
                dark += 1
    out = {
        "total_artifacts": total,
        "symbols": by_type.get("text/x-python-symbol", 0),
        "wordnet": by_type.get("text/x-wordnet", 0),
        "operators": by_type.get(OPERATOR_CT, 0),
        "content_docs": content_docs,
        "dark_matter": dark,
        # None, not 1.0, on an empty corpus. Reporting 100% keyed coverage for a store that
        # contains nothing would be perfect health achieved by holding no data, and it goes
        # straight into the trend log and onto the status page. `rho` two lines down also returns
        # None for the same empty-corpus case, so the two stay consistent with each other.
        "keyed_coverage": round(keyed / content_docs, 4) if content_docs else None,
        "avg_operator_fitness": round(sum(fitnesses) / len(fitnesses), 4) if fitnesses else 0.0,
        "bytes": total_bytes,
        "generator_bytes": gen_bytes,
        "rho": round(gen_bytes / total_bytes, 4) if total_bytes else None,
        "by_content_type": by_type,
    }
    import time as _t
    _METRICS_CACHE.update(ts=_t.time(), val=out, store=id(bundle.artifacts))
    return out


def counts_fast(bundle) -> Dict:
    """Live artifact counts from the maintained `counter` table, with no corpus scan.

    `metrics()` streams every committed artifact to sum bytes for rho; on 6.25M rows that takes
    minutes, and the status page calls it on every load, so a cold cache hangs the page. The counts
    the page needs (total, per-type, per-collection) are already maintained incrementally in the
    `counter` table (`vertex`, `ct:<type>`, `col:<name>`), read here directly. Falls back to `{}`
    on a backend with no counter table (a test double), where the caller uses its own path."""
    out: Dict = {"total_artifacts": 0, "by_content_type": {}, "collections": {}}
    try:
        cur = bundle.artifacts.db.read()          # LatticeConn cursor; raises on non-lattice backends
    except Exception:
        return {}
    try:
        for name, n in cur.execute("SELECT name, n FROM counter"):
            n = int(n)
            if name == "vertex":
                out["total_artifacts"] = n
            elif name.startswith("ct:"):
                out["by_content_type"][name[3:]] = n
            elif name.startswith("col:") and ":" not in name[4:]:   # `col:foundation`, not `col:foundation:committed`
                out["collections"][name[4:]] = n
    except Exception:
        return {}
    return out


def metrics_for_status(bundle) -> Dict:
    """Scan-free metrics for the status page: live counts from the counter table plus the last
    recorded ratio snapshot (rho, keyed_coverage, dark_matter, bytes) from `metrics.jsonl`. Never
    triggers a corpus scan, so `/status` cannot hang. Ratios read `None`/`—` when no snapshot
    exists yet, which the page renders as "computing…" — the honest unmeasured state rather than a
    fabricated number."""
    c = counts_fast(bundle)
    if not c:                                     # non-lattice backend: fall back to a full scan
        return metrics(bundle)
    snap: Dict = {}
    try:
        t = trend(bundle, last=1)
        snap = t[-1] if t else {}
    except Exception:
        snap = {}
    bt = c.get("by_content_type", {})
    return {
        "total_artifacts": c.get("total_artifacts", 0),
        "wordnet": bt.get("text/x-wordnet", 0),
        "content_docs": bt.get("text/markdown", 0),
        "operators": bt.get("application/vnd.agience.operator+json", 0),
        "by_content_type": bt,
        "collections": c.get("collections", {}),
        "rho": snap.get("rho"),
        "keyed_coverage": snap.get("keyed_coverage"),
        "dark_matter": snap.get("dark_matter", 0),
        "bytes": snap.get("bytes", 0),
        "generator_bytes": snap.get("generator_bytes", 0),
        "avg_operator_fitness": snap.get("avg_operator_fitness", "—"),
        "snapshot_ts": snap.get("ts"),
    }


def _log_path(bundle) -> Path:
    base = bundle.keys_dir.parent if bundle.keys_dir else Path(".")
    return Path(base) / "metrics.jsonl"


def improve_cycle(bundle, *, illuminate_limit: int = 300, now: Optional[float] = None) -> Dict:
    """One self-improvement step: illuminate dark matter, apply selection (retire the proven-
    unfit), measure, record. Returns the snapshot."""
    from ember.runtime.runner import describe, evolution     # the single distribution path (ember/runner.py)
    illuminated = describe.describe_dark(bundle, limit=illuminate_limit)
    retired = evolution.sweep_retire(bundle.artifacts)     # SELECT: shed persistently-unfit observers
    snap = metrics(bundle)
    snap["illuminated_this_cycle"] = illuminated
    snap["retired_this_cycle"] = len(retired)
    snap["ts"] = now if now is not None else time.time()
    try:
        with open(_log_path(bundle), "a", encoding="utf-8") as f:
            f.write(json.dumps(snap) + "\n")
    except Exception:
        pass
    return snap


def trend(bundle, *, last: int = 50) -> List[Dict]:
    from ember.surface.stats import tail_jsonl
    return tail_jsonl(_log_path(bundle), last)


def run(bundle, *, interval: float = 300.0, stop=None) -> None:
    """Background self-improvement loop (host under a thread). Deterministic; gated; no LLM."""
    while not (stop and stop()):
        try:
            improve_cycle(bundle)
        except Exception:
            pass
        time.sleep(interval)


def start(bundle, *, interval: float = 300.0) -> None:
    threading.Thread(target=run, kwargs={"bundle": bundle, "interval": interval}, daemon=True).start()
