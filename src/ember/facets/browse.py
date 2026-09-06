"""A lightweight ontology browser, served by Ember itself, same-origin so the page and its API share
a CSP origin.

Pages through the corpus, searches by lemma, and shows each artifact with its lemmas, its
content, and its neighbors (the edges) as clickable links, so you can walk the graph:
a word -> its hypernyms; a symbol -> what it calls and who calls it; a doc -> its terms.

JSON API (consumed by the page):
  GET /api/artifacts?type=&skip=&limit=&lemma=   -> paged list
  GET /api/artifact/{id}                         -> one artifact + typed neighbors
Rendered page: GET /browse
"""
from __future__ import annotations

from typing import Dict, List, Optional


def _label(a: dict) -> str:
    """A human-readable name for an artifact: its title/word/symbol, else its first lemma, else the
    first line of any inline content, else the id. Prefers a real name over the opaque id so the
    list and header read as topics rather than as `wiki-en-974`."""
    lemmas = a.get("lemmas") or []
    return (a.get("title") or a.get("qualname") or a.get("word") or a.get("sym")
            or (lemmas[0] if lemmas else None)
            or (a.get("content", "").split("\n", 1)[0].split("/")[-1] if a.get("content") else None)
            or a.get("id", ""))


def _is_private(bundle, a: dict) -> bool:
    """True for artifacts this unauthenticated browse API keeps out of its responses: everything
    that is not public.

    The rule is the access model. Public is the un-keyed top — a collection gated by no grant —
    and anything a grant gates is private, unreachable by an anonymous caller who is neither creator
    nor delegate. So this is exactly `not access.is_public(bundle, a)`, the same light-cone the rest
    of ember uses, and it reads no flag of its own.

    What that protects: `genesis.remember` writes owner memories as `visibility: private`,
    `no_share: True`, HUMAN_VALIDATED and encrypted at rest, and `op.share` is the consent gate,
    the only path by which owner-provided information leaves the private scope. `api_artifact`
    calls `resolve_text`, which decrypts the content blob, so this predicate is what stands between
    that call and an anonymous caller. It also closes discovery: `remember` indexes a memory's own
    words as lemmas, so `GET /api/artifacts?lemma=…` would otherwise hand out the ids that
    `GET /api/artifact/{id}` then resolves.

    This surface has no authenticated-caller concept, so the rule is absolute rather than
    owner-scoped: private artifacts are not served here at all. `api_artifact` returns the same
    "not found" it returns for a genuinely absent id, since a distinct 403 would confirm which
    memories exist."""
    if not a:
        return True
    try:
        from mantle.db import access
        return not access.is_public(bundle, a)
    except Exception:
        # When the grant subsystem cannot be consulted there is no public reading to go on, so this
        # unauthenticated surface fails closed and treats the artifact as private.
        return True


# The three blocks the response groups a row into. Each is derived from the artifact rather than
# composed: a field absent from the row is absent from the response, so "no citation" reads as no
# citation rather than as an empty string that looks like one.
_PROVENANCE_FIELDS = ("provenance", "cited_from", "via", "operator", "created_by", "created_time",
                      "origin_root", "root_id", "_origin", "_seq")
_COLLECTION_FIELDS = ("collection_id", "collections", "owner", "visibility", "no_share")
# Presentation and plumbing live in their own blocks; everything else on the row is data the
# artifact actually carries, and `_data_of` passes all of it through.
_NOT_DATA = set(_PROVENANCE_FIELDS) | set(_COLLECTION_FIELDS) | {
    "id", "content_type", "content", "content_ref", "size", "context", "state",
    "_leaf", "_rev", "path", "line", "kind"}


def _provenance_of(a: dict) -> dict:
    """Where this came from, and who says so. The citation chain is the point of the system."""
    return {k: a[k] for k in _PROVENANCE_FIELDS if a.get(k) not in (None, "", [], {})}


def _collections_of(a: dict) -> dict:
    """What this belongs to, and who owns it — what curation acts on."""
    out = {k: a[k] for k in _COLLECTION_FIELDS if a.get(k) not in (None, "", [], {})}
    memberships = list(out.get("collections") or [])
    if out.get("collection_id") and out["collection_id"] not in memberships:
        memberships.insert(0, out["collection_id"])
    out["memberships"] = memberships
    return out


def _data_of(a: dict) -> dict:
    """Every remaining field the artifact actually carries — title, gloss, examples, pos, lemmas,
    sense_ranks, ili, and whatever a future type adds without this file being edited."""
    return {k: v for k, v in a.items()
            if k not in _NOT_DATA and v not in (None, "", [], {})}


def api_artifacts(bundle, *, content_type: Optional[str], skip: int, limit: int,
                  lemma: Optional[str]) -> dict:
    store = bundle.artifacts
    if lemma:
        # Untyped by default, which suits an ontology browser: "everything carrying this lemma" —
        # synsets, symbols and wiki rows alike — is the question being asked. `content_type` is
        # passed through when the page's own type selector sets it.
        from mantle.shard import keyed as _keyed
        rows, _typed = _keyed.lookup_by_lemma(
            store, lemma, limit=skip + limit,
            content_type=content_type)          # honour the page's type selector when set
        rows = rows[skip:skip + limit]
    else:
        # Offset paging is O(offset): walking deep into a multi-million-row corpus costs more with
        # every skipped row, and SQLite's OFFSET walks the skipped rows all the same. `skip` is
        # this JSON API's contract
        # (the browser page pages by ?skip=), so an id keyset would change the wire format and the
        # page's ←/→ controls. It is bounded in practice instead: the pager is human-driven at 40
        # rows a click, so real offsets stay shallow. To walk the corpus, use a keyset — see
        # describe.describe_dark.
        rows = list(store.list_artifacts(content_type=content_type, skip=skip, limit=limit))
    # Filtered after fetching: the id itself is the capability here, so a private row stays out of a
    # listing even as a bare id, which is what keeps the lemma lookup from being a discovery oracle.
    rows = [a for a in rows if not _is_private(bundle, a)]
    return {"skip": skip, "limit": limit, "count": len(rows),
            "items": [{"id": a["id"], "content_type": a.get("content_type"),
                       "label": _label(a), "lemmas": (a.get("lemmas") or [])[:8]} for a in rows]}


def api_artifact(bundle, artifact_id: str) -> dict:
    from mantle.shard import content as C
    store, store_graph = bundle.artifacts, bundle.graph
    a = store.get_artifact(artifact_id)
    if not a or _is_private(bundle, a):
        # Same response for "absent" and "private" — see _is_private.
        return {"error": "not found"}
    neigh: List[Dict] = []
    for direction in ("out", "in"):
        for nid in store_graph.neighbors(artifact_id, direction=direction)[:40]:
            na = store.get_artifact(nid)
            if na:
                neigh.append({"id": nid, "label": _label(na), "dir": direction,
                              "content_type": na.get("content_type")})
    # code call-graph (lives in the `calls` list-field, not LINK edges): show it too
    if a.get("content_type") == "text/x-python-symbol":
        # Dark: code-intelligence lives in lumen — composing "`X` is called in N place(s)" is a
        # persona act — and ember imports no chorus module. With `code` unset, the neighbour panel
        # omits call and definition links rather than inventing them.
        code = None
        for called in (a.get("calls") or [])[:30]:                 # out: what this symbol calls
            for d in (code.find_definition(store, called)[:1] if code else []):
                neigh.append({"id": d["id"], "label": _label(d) + "()", "dir": "out",
                              "content_type": d.get("content_type")})
        for r in (code.find_references(store, a.get("sym", ""))[:30] if code else []):  # in: who calls this
            neigh.append({"id": r["id"], "label": _label(r), "dir": "in",
                          "content_type": r.get("content_type")})
    # full content from the content store via content_ref (never truncated); preview fallback otherwise
    text = C.resolve_text(bundle, a)
    # Fitness telemetry: the operator's accrued track record plus its computed fitness, so selection
    # is observable in the browser. Present only on operator artifacts, which are the rows carrying
    # the fields.
    from ember.runtime.runner import evolution
    fit = {f: int(a[f]) for f in evolution.FITNESS_FIELDS if f in a}
    if fit:
        fit["fitness"] = evolution.fitness(a)
    desc = _describe_artifact(bundle, a, text)
    return {"id": a["id"], "content_type": a.get("content_type"), "label": _label(a),
            "lemmas": a.get("lemmas") or [], "context": a.get("context", ""),
            "path": a.get("path"), "line": a.get("line"), "kind": a.get("kind"),
            "content_ref": a.get("content_ref"), "size": a.get("size"),
            "fitness": fit or None,
            "offer": desc["offer"], "fields": desc["fields"],
            "content": text[:8000], "neighbors": neigh,
            # The resource, not just a rendering of it. A wn row holds `title, gloss, examples, pos,
            # sense_ranks, ili, collection_id, collections, cited_from, via, operator, provenance,
            # created_by`, and these three blocks carry all of it alongside `label`, `offer` and the
            # raw content. Two things depend on that: showing where an artifact came from, which is
            # the point of a grounded system, and curating it, which requires seeing the collection
            # it is currently in.
            "provenance": _provenance_of(a),
            "collections": _collections_of(a),
            "data": _data_of(a)}


def _describe_artifact(bundle, a: Dict, text: str) -> Dict:
    """The describer's view of an artifact: an `offer_template` line plus the `context_schema`
    fields, projected from the data, so a viewer can show a type's own presentation instead of a raw
    dump.

    Dark here. The describer (`offer.py`) lives in sage, which owns `op.describe.*`, and ember
    imports no chorus module — so this returns the same shape a type with no definition returns: no
    offer, no fields, and the viewer falls back to the plain content. There is no import to guard,
    which keeps a stray importable `offer` from shadowing the real one."""
    return {"offer": None, "fields": []}


# No sage-calling body is kept here, even unreached: a function containing `from sage import offer`
# would be an ember→chorus import in ember's source, and the guard that greps for one would fire.


def dashboard_page(bundle) -> str:
    """The mesh dashboard. Reads passive snapshots (mesh-stats/*.json on OVH, one file per host) and
    delegates to the shared pure renderer, the same code that powers the public status.agience.ai
    VPS, so the two stay identical. No live scan and no live HTTP to peers."""
    from ember.surface import stats as _stats
    from ember.facets.dashboard_render import render_dashboard
    mesh = _stats.read_mesh_stats(bundle)
    local = _stats.read_stats(bundle.keys_dir)
    my_node = (local or {}).get("node")
    if not mesh and local:                          # OVH unreachable -> show this box on its own
        mesh = [local]
    return render_dashboard(mesh, my_node)


def status_page(bundle) -> str:
    """Renders the universe's live status — mass, compression (ρ), curriculum, health, checks and
    balances — and self-refreshes so it stays current on an open tab."""
    from ember.runtime import improve
    m = improve.metrics_for_status(bundle)      # scan-free: counts from counters, ρ from last snapshot
    tr = improve.trend(bundle, last=40)

    def spark(key):
        vals = [t.get(key) for t in tr if t.get(key) is not None]
        if not vals:
            return ""
        lo, hi = min(vals), max(vals)
        blocks = "▁▂▃▄▅▆▇█"
        return "".join(blocks[int((v - lo) / (hi - lo) * 7)] if hi > lo else blocks[0] for v in vals)

    # Scan-free health. `genesis.health` runs a full corpus scan — curriculum, consistency and
    # operator-fitness — which hangs the page on a 6.25M store, so this page reads passive sources
    # only: the worker's own stats snapshot for liveness. Consistency and curriculum are the
    # worker's to measure and record, and when it has not, the page says "not measured here" below.
    #
    # `h` stays empty in that case, and `h_err` carries a failure separately: every consumer below
    # treats an empty `h` as "nothing to report" rather than "all checks passed", so the checks &
    # balances row states "not measured" for checks that never ran.
    h, h_err = {}, None
    try:
        from ember.surface import stats as _stats
        st = _stats.read_stats(bundle.keys_dir) or {}
        if isinstance(st.get("worker"), dict):
            h["worker"] = st["worker"]
    except Exception:
        pass
    cm = m.get("collections", {})
    by_type = m.get("by_content_type", {}) if isinstance(m.get("by_content_type"), dict) else {}
    # Read off the metrics scan already in hand (`m`, cached 90s). A full corpus stream costs ~500
    # keyset pages of ~743ms, about 370s at 5M rows, on every /status load, and the page
    # self-refreshes every 180s. A live `count(*) WHERE content_type = 'application/x-concept'` is
    # no cheaper: an indexed equality is only cheap on a selective value (measured on node 71,
    # 5,657,386 rows: a 48-row type 0.12s, text/markdown ~6M rows timed out at 120s), and the
    # concept count is unbounded.
    # `by_content_type` counts committed artifacts, as the wordnet and operator rows beside it in
    # this same table do, so the table is consistent within itself.
    concepts = int(by_type.get("application/x-concept", 0) or 0)

    def sec(title):
        return f'<tr><td colspan=3 class=sec>{title}</td></tr>'

    def row(k, v, s="", bad=False):
        cls = ' class=bad' if bad else ''
        return f'<tr{cls}><td>{k}</td><td class=v>{v}</td><td class=s>{s}</td></tr>'

    rho = m.get("rho")
    rows = [sec("mass")]
    rows += [row("total artifacts", f"{m['total_artifacts']:,}"),
             row("WordNet synsets", f"{m['wordnet']:,}"),
             row("content docs", f"{m['content_docs']:,}"),
             row("concepts (consolidated)", f"{concepts:,}"),
             row("operators", m["operators"]),
             row("bytes (corpus)", _hb(m.get("bytes")))]
    # mesh — the write-scaling shards read as one universe (this box plus every peer shard)
    try:
        from mantle.mesh import federation as _mf
        if _mf.peers():
            ms = _mf.mesh_status(m["total_artifacts"], local_rho=rho)
            rows += [sec(f"mesh — {ms['shards']} write-scaling shards as one universe")]
            rows += [row("total artifacts (all shards)", f"{ms['total_artifacts']:,}",
                         f"this shard {ms['local_artifacts']:,} + peers {ms['peer_artifacts']:,}")]
            for p in ms.get("peers", []):
                if p.get("reachable"):
                    rows.append(row("&nbsp;&nbsp;peer " + p["peer"].replace("http://", ""),
                                    f"{int(p.get('artifacts') or 0):,}",
                                    f"ρ={p.get('rho')} · worker {'●' if p.get('worker_alive') else '○'}"))
                else:
                    rows.append(row("&nbsp;&nbsp;peer " + p["peer"].replace("http://", ""),
                                    "unreachable", "tunnel/serve down", bad=True))
    except Exception:
        pass
    rows += [sec("compression — is the universe cooling?")]
    rows += [row("ρ  generators / corpus", f"{rho:.4f}" if rho is not None else "computing…",
                 spark("rho") + "  (want ↓)"),
             row("generator bytes", _hb(m.get("generator_bytes"))),
             # `keyed_coverage` is None when the corpus holds no content docs: improve.metrics
             # returns None rather than 1.0, so holding nothing does not read as perfect coverage.
             # Guarded like `rho` above, because multiplying None by 100 raises TypeError and takes
             # the whole /status page to a 500 on a fresh or structural-only node.
             row("keyed coverage",
                 f"{m['keyed_coverage']*100:.1f}%" if m.get("keyed_coverage") is not None
                 else "computing…",
                 spark("keyed_coverage") + "  (want ↑)"),
             row("dark matter", f"{m['dark_matter']:,}", spark("dark_matter") + "  (want ↓ 0)"),
             row("avg operator fitness", m["avg_operator_fitness"], spark("avg_operator_fitness"))]

    # curriculum
    cur = h.get("curriculum") or []
    if cur:
        rows.append(sec("curriculum (developmental stages · target = promotion threshold)"))
        for s in cur:
            # `have` may be None ("not measured"), so the comparison is guarded: an unguarded
            # `None >= int` is a TypeError that takes /status to a 500.
            _have = s.get("have")
            _target = s.get("target")
            done = s.get("promoted") or bool(
                _target and _have is not None and _have >= _target)
            pct = int((s.get("progress") or 0) * 100)
            # The formatting carries the same None guard as the comparison above: `f"{None:,}"`
            # raises `TypeError: unsupported format string passed to NoneType.__format__`, so one
            # stage with an unmeasured `have` would take the whole /status page to a 500.
            #
            # None renders as "not measured" rather than as 0, since a stage nobody has measured and
            # a stage measured at zero are different facts, and "0 / 500,000" would conflate them.
            def _n(v) -> str:
                return "not measured" if v is None else f"{v:,}"
            if done:
                val = f"{_n(_have)} ✓"
            else:
                val = f"{_n(_have)} / {_n(_target)}"
            note = "complete" if done else f"{pct}% of {_n(_target)}"
            rows.append(row(s["stage"], val, note))

    # health + checks & balances
    w = h.get("worker") or {}
    cons = h.get("consistency") or {}
    prov = h.get("provenance") or {}
    rows.append(sec("health & monitoring"))
    alive = "● live" if w.get("alive") else "○ down"
    rows.append(row("worker(s)", alive,
                    f"tick {w.get('last_tick')} · {w.get('age_s')}s ago · {w.get('recent_errors',0)} errs",
                    bad=not w.get("alive")))
    anoms = cons.get("anomalies") or []
    if "consistency" not in h:
        # Not measured on this scan-free page, and reported as such: the page states only checks
        # that ran. The worker records consistency, or `node-repair` does on demand.
        rows.append(row("checks &amp; balances", "— not measured here",
                        "run the worker or node-repair for a live consistency audit"))
    elif h_err is not None:
        # The check raised, so there is no result to report: the row states that the check did not
        # run, distinct from a pass.
        rows.append(row("checks &amp; balances", "⚠ DID NOT RUN",
                        "health check raised — this is not a pass: " + h_err, bad=True))
    else:
        rows.append(row("checks &amp; balances",
                        "✓ all pass" if not anoms else f"⚠ {len(anoms)} ANOMALY",
                        (", ".join(a["check"] for a in anoms) if anoms else
                         (", ".join(x["check"] for x in (cons.get("watch") or [])) or "ρ∈[0,1], mass↑, fitness∈[0,1]")),
                        bad=bool(anoms)))
    # Three states, read explicitly. `genesis.status` returns `invariant_holds: None` when the audit
    # was skipped — the normal 30s fast path, since the scan runs only on the 900s slow refresh — and
    # publishes a `measured` flag beside it, so unmeasured reads as unmeasured. `None` is checked
    # explicitly rather than by truthiness, since `None` is falsy and a truthiness test would read it
    # the same as a violated invariant.
    _inv = prov.get("invariant_holds")
    if _inv is None:
        rows.append(row("provenance invariant", "· not measured",
                        "scan runs on the slow refresh; no violation is being claimed", bad=False))
    else:
        rows.append(row("provenance invariant",
                        "✓ every artifact cited" if _inv else "⚠ VIOLATED",
                        "" if _inv else f"missing {prov.get('missing_cited_from')}",
                        bad=not _inv))
    healthy = h.get("healthy")

    # per-collection counts (from the counter table; ρ/coverage per collection is a scan the worker
    # records, not something this page computes). `cm` is `{collection: count}`.
    col_rows = [(cid, n) for cid, n in sorted(cm.items()) if n]
    if col_rows:
        rows.append(sec("per collection"))
        for cid, n in col_rows:
            rows.append(row(f"&nbsp;&nbsp;{cid}", f"n={n:,}", ""))

    body = "".join(rows)

    # ── workers & tasks — the operator work-pool, as a collapsible accordion ──
    def esc(x):
        return (str(x) if x is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    try:
        from ember.runtime import pool
        ps = pool.pool_status(bundle)
    except Exception:
        ps = None
    pool_html = ""
    if ps:
        q = ps["queue"]
        qsum = (f"pending {q['pending']} · running {q['claimed']} · done {q['done']}"
                + (f" · <span style=color:#cf222e>failed {q['failed']}</span>" if q['failed'] else ""))
        bm = ps.get("by_machine") or {}
        peer_html = ""
        if bm:
            peer_html = ('<tr><td colspan=3 class=sec>peering — workers by machine</td></tr>'
                         + "".join(f"<tr><td>{esc(m)}</td><td class=v>{n}</td>"
                                   f"<td class=s>{'●'*min(n,20)}</td></tr>" for m, n in sorted(bm.items())))
        act = "".join(
            f"<tr><td>{esc(a['worker'])}</td><td class=s>{esc(a['op'])} · {esc(a['task'])}</td>"
            f"<td class=v>{a['age_s']}s</td></tr>" for a in ps["active"]) or \
            "<tr><td colspan=3 style=color:#888>idle — no active tasks</td></tr>"
        rec = "".join(
            f"<tr><td>{'✓' if r['status']=='done' else '✗'}</td>"
            f"<td class=s>{esc(r['op'])} · {esc(r['task'])}</td><td class=v>{r['age_s']}s ago</td></tr>"
            for r in ps["recent"])
        machines = ps.get("machines", 0)
        pool_html = (
            f'<details open style="margin-top:18px"><summary style="cursor:pointer;font-weight:600">'
            f'operator work-pool — {ps["workers_active"]} worker(s) across {machines} machine(s) · {qsum}</summary>'
            f'<table style="margin-top:8px">{peer_html}'
            f'<tr><td colspan=3 class=sec>working now</td></tr>{act}'
            f'<tr><td colspan=3 class=sec>recently completed</td></tr>{rec}</table></details>')

    banner = ("#1a7f37" if healthy else "#cf222e")
    reasons = h.get("degraded_reasons") or []
    state = "healthy" if healthy else ("degraded" if h else "starting…")
    why = (" — " + "; ".join(reasons)) if (not healthy and reasons) else ""
    return f"""<!doctype html><meta charset=utf-8><title>Ember — universe status</title>
<meta http-equiv="refresh" content="180">
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font:14px/1.6 system-ui,sans-serif;margin:0;padding:28px;max-width:860px}}
h1{{margin:0 0 2px}} .sub{{color:#666;margin-bottom:16px}}
.pill{{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-size:12px;font-weight:600;background:{banner}}}
table{{border-collapse:collapse;width:100%}} td{{padding:7px 10px;border-bottom:1px solid #eee}}
.v{{font-weight:600;text-align:right;font-variant-numeric:tabular-nums}}
.s{{font-family:ui-monospace,monospace;color:#1a7f37;white-space:nowrap;font-size:12px}}
.sec{{padding-top:16px;color:#888;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
tr.bad td{{color:#cf222e;background:rgba(207,34,46,.06)}} tr.bad .s{{color:#cf222e}}
@media(prefers-color-scheme:dark){{body{{background:#141414;color:#eaeaea}}td{{border-color:#2b2b2b}}.sub,.sec{{color:#9a9a9a}}tr.bad td{{color:#ff6b6b;background:rgba(255,107,107,.08)}}tr.bad .s{{color:#ff6b6b}}}}</style>
<h1>Ember — universe status <span class=pill>{state}</span></h1>
<div class=sub>{('<b style=color:#cf222e>'+why[3:]+'</b> · ') if why else ''}{len(tr)} cycles · refresh in <b id=cd>3:00</b> · <a href="/chat">chat</a> · <a href="/browse">browse</a> · <a href="/library">library</a></div>
<table>{body}</table>
{pool_html}
<script>
let _t=180;const _cd=document.getElementById('cd');
setInterval(()=>{{_t=Math.max(0,_t-1);const m=Math.floor(_t/60),s=String(_t%60).padStart(2,'0');
 _cd.textContent=m+':'+s;if(_t===0)_cd.textContent='refreshing…';}},1000);
</script>"""


def _gm(bundle, key):
    """One value from the cached universe metrics (no extra scan)."""
    from ember import genesis
    try:
        return genesis.all_metrics(bundle).get(key)
    except Exception:
        return None


def _hb(nbytes):
    """Human bytes."""
    if not nbytes:
        return "—"
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


_LIB_CACHE = {}


def library_view(bundle, *, root: str = "agience-pharos", refresh: bool = False,
                 resolution: Optional[float] = None) -> dict:
    """The organized library: emergent categories against status over a doc root. Cached per
    (root, resolution); `?refresh` recomputes. Deterministic — the operators produce it.

    The view carries the zoom (§9.6). `resolution` comes from the caller and travels down to
    `library_plan`, so the grouping level is the caller's need rather than a constant inside
    `docs_ops`.

    With no resolution this returns the choices rather than a chosen one: `offered` lists the levels
    at which this corpus's structure actually changes, and `categories` is empty. A caller, or a UI
    zoom control, picks one and asks again. An empty category list alongside a populated `offered`
    means the zoom has not been chosen yet, which the shape states rather than leaving it to look
    like an empty corpus.
    """
    import time
    from ember.runtime.runner import docs_ops           # the single distribution path
    key = (root, resolution)
    if refresh or key not in _LIB_CACHE:
        plan = docs_ops.library_plan(bundle.artifacts, bundle, root=root, now=time.time(),
                                     resolution=resolution)
        if plan.get("resolution") is None:
            _LIB_CACHE[key] = {"total": plan.get("docs", 0), "by_status": {}, "categories": [],
                               "offered": plan.get("resolutions", []), "why": plan.get("why"),
                               "raw": plan}
            return _LIB_CACHE[key]
        # group per_doc under its emergent category
        cats = {}
        for d in plan["per_doc"]:
            cats.setdefault(d["category"], []).append(d)
        _LIB_CACHE[key] = {"total": plan["total_docs"], "by_status": plan["by_status"],
                           "categories": sorted(cats.items(), key=lambda kv: -len(kv[1])),
                           "offered": [], "raw": plan}
    return _LIB_CACHE[key]


def library_page(bundle, *, root: str = "agience-pharos", refresh: bool = False,
                 resolution: Optional[float] = None) -> str:
    lib = library_view(bundle, root=root, refresh=refresh, resolution=resolution)
    if lib.get("offered") is not None and not lib["categories"] and lib.get("why"):
        # The library does not pick the zoom. Offer the levels the corpus actually has.
        levels = "".join('<li><a href="?resolution=%s">%.4f</a></li>' % (v, v)
                         for v in lib["offered"][:40])
        return ("<!doctype html><meta charset=utf-8><title>Ember — %s library</title>"
                "<h1>%s</h1><p>%s</p><ul>%s</ul>" % (root, root, lib["why"], levels))
    badge = {"current": "#1a7f37", "stale": "#9a6700", "superseded": "#cf222e"}
    parts = [f"""<!doctype html><meta charset=utf-8><title>Ember — {root} library</title>
<style>body{{font:14px/1.55 system-ui,sans-serif;margin:0;padding:24px;max-width:1000px}}
h1{{margin:0 0 2px}} .sub{{color:#666;margin-bottom:18px}}
.cat{{margin:18px 0;border:1px solid #e5e5e5;border-radius:8px}} .cat h2{{margin:0;padding:10px 14px;
background:#f6f8fa;border-bottom:1px solid #eee;font-size:15px}} .cat .n{{color:#888;font-weight:400}}
.doc{{padding:5px 14px;border-bottom:1px solid #f4f4f4;display:flex;gap:10px;align-items:center}}
.b{{color:#fff;border-radius:9px;padding:0 7px;font-size:11px}} .p{{color:#333;font-family:ui-monospace,monospace;font-size:12px}}
a{{color:#0645ad;text-decoration:none}}</style>
<h1>{root} — organized library</h1>
<div class=sub>{lib['total']} docs · status {lib['by_status']} · categories emerge from the data (deterministic) ·
<a href="/library?refresh=1">refresh</a> · <a href="/browse">ontology browser</a></div>"""]
    for label, docs in lib["categories"]:
        docs = sorted(docs, key=lambda d: (d["status"] != "current", d["staleness"]))
        parts.append(f'<div class=cat><h2>{esc_(label)} <span class=n>({len(docs)})</span></h2>')
        for d in docs[:60]:
            col = badge.get(d["status"], "#666")
            short = d["path"].split(root + "/")[-1]
            parts.append(f'<div class=doc><span class=b style="background:{col}">{d["status"]}</span>'
                         f'<span class=p>{esc_(short)}</span></div>')
        parts.append("</div>")
    return "".join(parts)


def esc_(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


PAGE = """<!doctype html><meta charset=utf-8><title>Ember — ontology</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e6e3dd;--card:#fff;--accent:#2b5fa8;--in:#7a3ea1;--soft:#f3f1ec}
 @media(prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eaeaea;--mut:#9a9a9a;--line:#2b2b2b;--card:#1c1c1c;--accent:#6ea8fe;--in:#c08cf0;--soft:#1f1f1f}}
 *{box-sizing:border-box}
 body{font:15px/1.6 system-ui,-apple-system,sans-serif;margin:0;display:flex;height:100vh;background:var(--bg);color:var(--fg)}
 #left{width:340px;border-right:1px solid var(--line);display:flex;flex-direction:column;background:var(--card)}
 #ctrl{padding:12px;border-bottom:1px solid var(--line);display:flex;gap:6px;flex-wrap:wrap;align-items:center}
 #ctrl input,#ctrl select{padding:7px 9px;border:1px solid var(--line);border-radius:8px;font:inherit;background:var(--bg);color:var(--fg)}
 #ctrl input{flex:1;min-width:120px}
 #ctrl button{padding:7px 11px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);cursor:pointer}
 #ctrl button:hover{border-color:var(--accent)}
 #list{overflow:auto;flex:1}
 .it{padding:10px 14px;border-bottom:1px solid var(--line);cursor:pointer}
 .it:hover{background:var(--soft)}.it.sel{background:var(--soft);box-shadow:inset 3px 0 0 var(--accent)}
 .it .l{font-weight:600}.it .m{color:var(--mut);font-size:12.5px;margin-top:1px;word-break:break-word}
 #main{flex:1;overflow:auto;padding:34px 40px}
 #main .wrap{max-width:760px}
 .empty{color:var(--mut);font-style:italic;margin-top:40px}
 h1{margin:0 0 4px;font-size:26px;line-height:1.25;text-wrap:balance}
 .ct{color:var(--mut);font-size:12.5px;letter-spacing:.02em}
 .chips{margin:12px 0 2px}
 .chip{display:inline-block;background:var(--soft);border:1px solid var(--line);border-radius:20px;padding:2px 11px;margin:0 5px 5px 0;font-size:12.5px;color:var(--mut)}
 .offer{font-size:17px;line-height:1.55;margin:18px 0;color:var(--fg)}
 dl{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;margin:18px 0;align-items:baseline}
 dt{color:var(--mut);font-size:13px;text-transform:lowercase;letter-spacing:.03em}
 dd{margin:0;word-break:break-word}
 .sect{margin:22px 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut)}
 .edges{display:flex;flex-wrap:wrap;gap:6px}
 .n{cursor:pointer;color:var(--accent);border:1px solid var(--line);border-radius:7px;padding:3px 9px;font-size:13px;background:var(--card);text-decoration:none}
 .n:hover{border-color:var(--accent)}.n.in{color:var(--in)}
 pre{background:var(--soft);padding:14px 16px;border-radius:10px;overflow:auto;white-space:pre-wrap;
  border:1px solid var(--line);font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;margin:8px 0 30px}
 a.top{color:var(--mut);text-decoration:none;font-size:13px}a.top:hover{color:var(--fg)}
</style>
<div id=left>
 <div id=ctrl>
  <input id=lemma placeholder="search a lemma…" size=18>
  <select id=type><option value="">all types</option></select>
  <button onclick=load(0)>go</button>
  <button onclick=page(-1)>&larr;</button><button onclick=page(1)>&rarr;</button>
 </div>
 <div id=list></div>
</div>
<div id=main><div class=wrap><a class=top href="/">← chat</a><p class=empty>Pick an artifact, or search a lemma. Click any record or neighbor to walk the graph.</p></div></div>
<script>
let skip=0;const LIM=40;
const TYPES=["text/x-wordnet","text/x-python-symbol","text/markdown","text/plain","text/x-python",
 "application/x-concept","application/x-entity","application/x-citation",
 "application/vnd.agience.operator+json","application/vnd.agience.source+json",
 "application/vnd.agience.collection+json","application/vnd.agience.vtype+json",
 "application/vnd.agience.etype+json","application/vnd.agience.rung+json"];
const sel=document.getElementById('type');TYPES.forEach(t=>sel.add(new Option(t,t)));
const listEl=document.getElementById('list');const mainEl=document.getElementById('main');
function esc(s){return (s==null?'':String(s)).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function load(s){skip=Math.max(0,s);const l=document.getElementById('lemma').value.trim();
 const t=sel.value;const q=new URLSearchParams({skip,limit:LIM});if(t)q.set('type',t);if(l)q.set('lemma',l);
 try{const r=await fetch('/api/artifacts?'+q);const d=await r.json();
  listEl.innerHTML=(d.items||[]).map(i=>`<div class=it data-id="${esc(i.id)}"><div class=l>${esc(i.label)}</div>`
   +`<div class=m>${esc((i.content_type||'').replace('application/','').replace('text/',''))}${(i.lemmas&&i.lemmas.length)?(' · '+esc(i.lemmas.join(', '))):''}</div></div>`).join('')
   ||'<div class=it>none</div>';
 }catch(e){listEl.innerHTML='<div class=it>error: '+esc(e.message)+'</div>';}}
function page(d){load(skip+d*LIM);}
function link(n){return `<a class="n ${n.dir}" data-id="${esc(n.id)}">${esc(n.label)}</a>`;}
async function open_(id){
 document.querySelectorAll('.it.sel').forEach(e=>e.classList.remove('sel'));
 const li=document.querySelector('.it[data-id="'+CSS.escape(id)+'"]');if(li)li.classList.add('sel');
 mainEl.innerHTML='<div class=wrap><p class=empty>loading '+esc(id)+'…</p></div>';
 try{
  const r=await fetch('/api/artifact/'+encodeURIComponent(id));const a=await r.json();
  if(a.error){mainEl.innerHTML='<div class=wrap><p class=empty>'+esc(a.error)+': '+esc(id)+'</p></div>';return;}
  const nb=a.neighbors||[];const out=nb.filter(n=>n.dir=='out'),inc=nb.filter(n=>n.dir=='in');
  const meta=[a.content_type,(a.path?a.path+':'+a.line:''),a.kind].filter(Boolean).map(esc).join(' · ');
  let h=`<div class=wrap><a class=top href="/">← chat</a>`
   +`<h1>${esc(a.label)}</h1><div class=ct>${meta}</div>`;
  if(a.lemmas&&a.lemmas.length) h+=`<div class=chips>${a.lemmas.map(x=>`<span class=chip>${esc(x)}</span>`).join('')}</div>`;
  if(a.offer) h+=`<div class=offer>${esc(a.offer)}</div>`;               /* the type's own describer */
  if(a.fields&&a.fields.length) h+=`<dl>${a.fields.map(f=>`<dt>${esc(f.k)}</dt><dd>${esc(f.v)}</dd>`).join('')}</dl>`;
  if(a.fitness) h+=`<dl><dt>fitness</dt><dd>${esc(JSON.stringify(a.fitness))}</dd></dl>`;
  if(out.length) h+=`<div class=sect>→ out</div><div class=edges>${out.map(link).join('')}</div>`;
  if(inc.length) h+=`<div class=sect>← in</div><div class=edges>${inc.map(link).join('')}</div>`;
  if(a.content&&a.content!==a.offer) h+=`<div class=sect>content</div><pre>${esc(a.content)}</pre>`;
  mainEl.innerHTML=h+'</div>';mainEl.scrollTop=0;
 }catch(e){mainEl.innerHTML='<div class=wrap><p class=empty>error loading '+esc(id)+': '+esc(e.message)+'</p></div>';}}
document.addEventListener('click',e=>{const t=e.target.closest('[data-id]');if(t){e.preventDefault();open_(t.getAttribute('data-id'));}});
document.getElementById('lemma').addEventListener('keydown',e=>{if(e.key==='Enter')load(0);});
load(0);
</script>"""


# The conversational UI — a real chat page served same-origin so it can reach /v1 + /v1/invoke.
CHAT_PAGE = """<!doctype html><meta charset=utf-8><title>Ember — chat</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e6e3dd;--u:#1a7f37;--e:#2b5fa8;--card:#fff}
 @media(prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eaeaea;--mut:#9a9a9a;--line:#2b2b2b;--u:#3fb95d;--e:#6ea8fe;--card:#1c1c1c}}
 *{box-sizing:border-box}body{font:15px/1.55 system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg);
  display:flex;flex-direction:column;height:100vh}
 header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
 header b{font-size:16px}.dot{width:8px;height:8px;border-radius:50%;background:#c33}.dot.ok{background:var(--u)}
 header a{color:var(--mut);text-decoration:none;font-size:13px}header a:hover{color:var(--fg)}
 #log{flex:1;overflow:auto;padding:20px;max-width:900px;width:100%;margin:0 auto}
 .msg{margin:0 0 16px;display:flex;gap:10px}.msg .who{font-weight:600;min-width:56px;color:var(--mut);font-size:13px;padding-top:2px}
 .msg.u .who{color:var(--u)}.msg.e .who{color:var(--e)}
 .bub{white-space:pre-wrap;word-break:break-word;background:var(--card);border:1px solid var(--line);
  border-radius:10px;padding:10px 13px;flex:1}
 .bub.u{background:transparent;border-color:transparent;padding-left:0}
 .via{color:var(--mut);font-size:12px;margin-top:6px}
 #chips{display:flex;gap:7px;flex-wrap:wrap;padding:0 18px 6px;max-width:900px;width:100%;margin:0 auto}
 .chip{border:1px solid var(--line);background:var(--card);border-radius:14px;padding:4px 11px;font-size:13px;cursor:pointer;color:var(--fg)}
 .chip:hover{border-color:var(--e)}
 #bar{border-top:1px solid var(--line);padding:12px 18px;display:flex;gap:8px;max-width:900px;width:100%;margin:0 auto}
 #q{flex:1;padding:11px 13px;border:1px solid var(--line);border-radius:9px;font:inherit;background:var(--card);color:var(--fg)}
 #send{padding:11px 18px;border:0;border-radius:9px;background:var(--e);color:#fff;font:inherit;font-weight:600;cursor:pointer}
 #send:disabled{opacity:.5;cursor:default}
</style>
<header>
 <span class=dot id=dot></span><b>Ember</b>
 <a href="/browse">browse ontology ↗</a><a href="/status">status ↗</a>
</header>
<div id=log></div>
<div id=chips>
 <span class=chip>who was Ada Lovelace</span><span class=chip>the French Revolution</span>
 <span class=chip>photosynthesis</span><span class=chip>quantum entanglement</span>
</div>
<div id=bar>
 <input id=q placeholder="message Ember…" autocomplete=off autofocus>
 <button id=send>send</button>
</div>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send'),
 dot=document.getElementById('dot');
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function add(who,cls,text){const m=document.createElement('div');m.className='msg '+cls;
 m.innerHTML=`<div class=who>${who}</div><div class=bub ${cls}>${esc(text)}</div>`;log.appendChild(m);
 log.scrollTop=log.scrollHeight;return m.querySelector('.bub');}
/* NO status polling on the chat page — the COUNT/universe scan froze chat under write load. The
   dashboard (/browse) is the ONLY place stats live, and it reads passive snapshots (never a live scan). */
async function ask(text){
 add('you','u',text);const bub=add('ember','e','…');send.disabled=true;q.disabled=true;
 try{
  const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},
   body:JSON.stringify({model:'ember',stream:true,messages:[{role:'user',content:text}]})});
  const rd=r.body.getReader();const dec=new TextDecoder();let buf='',out='';
  bub.textContent='';
  while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
   let i;while((i=buf.indexOf('\\n\\n'))>=0){const frame=buf.slice(0,i);buf=buf.slice(i+2);
    for(const line of frame.split('\\n')){if(!line.startsWith('data:'))continue;const p=line.slice(5).trim();
     if(p==='[DONE]')continue;try{const j=JSON.parse(p);const dl=j.choices&&j.choices[0]&&j.choices[0].delta;
      if(dl&&dl.content){out+=dl.content;bub.textContent=out;log.scrollTop=log.scrollHeight;}}catch(_){}}}}
  if(!out)bub.textContent='(no response)';
 }catch(e){bub.textContent='error: '+e.message+' — is the server up?';}
 send.disabled=false;q.disabled=false;q.focus();
}
function submit(){const t=q.value.trim();if(!t)return;q.value='';ask(t);}
send.addEventListener('click',submit);
q.addEventListener('keydown',e=>{if(e.key==='Enter')submit();});
document.querySelectorAll('.chip').forEach(c=>c.addEventListener('click',()=>{q.value=c.textContent;submit();}));
dot.classList.add('ok');  /* no status COUNT on the chat page — it froze chat under write load */
</script>"""
