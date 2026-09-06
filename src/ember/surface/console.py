"""console — a local control surface for a node: what this ember is, what prisms have
registered, and running one of their operators.

This is a control plane over surfaces that already exist, server-rendered from stdlib. It is
distinct from `crystal.workbench` (`aria/crystal_workbench.py`), which is a React `web` facet with
four persona tektons and five organons and delegates rendering to `agience-facet`, a package outside
this workspace.

It lives in ember because ember already serves `/status`, `/browse`, `/library`, `/api/artifacts`
and `/api/invoke` from a stdlib `http.server`, so a console is a page over those rather than new
machinery. Ember imports no chorus module, so nothing here reaches a persona; hosts and operators
are read from the store, which is the ground both sides share.

Security. `serve.py` declines the whole `op.dev.*` namespace over HTTP, because `op.dev.run_tests`
shells out and blocking the file-writing operators alone would leave the process-spawning one open.
This console likewise executes nothing on this box: it shells no `prism init` and spawns no
processes. It reads the store and ships a sealed signal to a host that already registered. Anything
that mutates is gated behind `EMBER_INVOKE_TOKEN`, the same gate `/hosts/register` uses, and an
unset token leaves that gate closed.
"""
from __future__ import annotations

import html
import os
from typing import Any, Dict, List

# Environment keys worth showing, as an allow-list: a deny-list would leak the next secret someone
# adds. `EMBER_INVOKE_TOKEN` is reported as set/unset rather than echoed.
_SHOWN_ENV = (
    "EMBER_NODE_ID", "EMBER_PRINCIPAL", "EMBER_ROLE", "EMBER_PURPOSE",
    "EMBER_SQLITE_DIR", "EMBER_SQLITE_DB", "EMBER_STORE_BACKEND",
    "EMBER_OVH_ENDPOINT", "EMBER_OVH_REGION", "EMBER_OVH_BUCKET",
)
_SECRET_ENV = ("EMBER_INVOKE_TOKEN", "EMBER_STORE_KEYS_DIR")


def _artifacts(store):
    return getattr(store, "artifacts", None) or store


def node_config() -> Dict[str, Any]:
    """This node's identity and where its ground is, read from the environment the process actually
    runs with rather than from a config file that may disagree with it."""
    cfg = {k: os.getenv(k, "") for k in _SHOWN_ENV}
    # Presence rather than value. A console that prints a token leaks one into a screenshot, a log,
    # or a support thread.
    cfg["EMBER_INVOKE_TOKEN"] = "set" if os.getenv("EMBER_INVOKE_TOKEN") else "UNSET (writes closed)"
    cfg["EMBER_STORE_KEYS_DIR"] = "set" if os.getenv("EMBER_STORE_KEYS_DIR") else "unset"
    return cfg


def hosts(store, *, probe: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
    """Every prism host that has registered, with what it claimed and — when `probe` — what it says
    it is serving right now.

    A registration is a claim, and the two can disagree: the artifact records what a host announced
    at boot, while `GET /health` reports what it has mounted since. Showing them side by side is how
    a reader tells a stale registration from a live one, and that divergence is usually the cause of
    a "my tekton isn't there" question.
    """
    from ember.runtime import capability
    from ember.signal import signal

    arts = _artifacts(store)
    out: List[Dict[str, Any]] = []
    try:
        rows = list(arts.list_artifacts(content_type=capability.HOST_CONTENT_TYPE, limit=limit))
    except Exception as exc:                                    # noqa: BLE001 — surfaced, not hidden
        return [{"error": "cannot list hosts: %s: %s" % (type(exc).__name__, exc)}]

    for row in rows:
        endpoint = row.get("endpoint") or ""
        entry = {
            "id": row.get("id"),
            "host": row.get("host"),
            "endpoint": endpoint,
            "offers": row.get("offers") or [],
            "probed": bool(row.get("probed")),
            "operators": [],
        }
        try:
            entry["operators"] = [
                {"id": o.get("id"),
                 "operator_name": o.get("operator_name"),
                 "invokable_locally": bool(o.get("invokable_locally")),
                 "dispatch": o.get("dispatch")}
                for o in arts.list_artifacts(
                    content_type=capability.OPERATOR_CONTENT_TYPE, limit=200)
                if o.get("host") == row.get("id")
            ]
        except Exception as exc:                                # noqa: BLE001
            entry["operators_error"] = "%s: %s" % (type(exc).__name__, exc)

        if probe and endpoint:
            entry["live"] = signal.health(endpoint)
        out.append(entry)
    return out


def tektons(store, *, limit: int = 400) -> List[Dict[str, Any]]:
    """The operators this node can run — the local ones, alongside the remote pointers `hosts()`
    lists.

    A tekton is an information morphism; this is the roster of the ones present. It is read from the
    store's operator artifacts, so it tracks what is there. A hand-maintained list drifts:
    `genesis.invoke`'s error branch carries an `"invokable"` roster that already omits ids that do
    dispatch.

    `local` here means "has an executable body on this node" (`kind` + `spec`). Remote pointers are
    left to `hosts()`, which shows them with their endpoint and live health; listing them twice
    under two different truths is how a reader ends up trusting the wrong one.
    """
    from ember.runtime import capability

    arts = _artifacts(store)
    out: List[Dict[str, Any]] = []
    try:
        rows = arts.list_artifacts(content_type=capability.OPERATOR_CONTENT_TYPE, limit=limit)
    except Exception as exc:                                    # noqa: BLE001 — surfaced, not hidden
        return [{"error": "cannot list operators: %s: %s" % (type(exc).__name__, exc)}]

    for row in rows:
        if row.get("remote") or row.get("dispatch"):
            continue                                            # hosts() shows it, with its endpoint
        out.append({
            "id": row.get("id"),
            "kind": row.get("kind") or "",
            "offer": row.get("offer") if isinstance(row.get("offer"), str) else "",
            "invokable": bool(row.get("kind")),
        })
    out.sort(key=lambda o: str(o["id"]))
    return out


def run(store, delegate, operator: str, *, token: str = "", ship: bool = True) -> Dict[str, Any]:
    """Resolve an operator address, seal a signal toward it, and (for a remote host) ship it.

    The two steps stay separate: sealing is a distinct act from dialling. See `signal.ship`.
    """
    from ember.signal import signal

    sent = signal.send(delegate, operator, store=store)
    if not ship or (sent.get("target") or {}).get("kind") != "remote":
        return sent
    sent["delivery"] = signal.ship(sent, token=token or os.getenv("EMBER_HOST_TOKEN", ""))
    return sent


def state(store, *, probe: bool = True) -> Dict[str, Any]:
    """Everything the console page renders, as data, so the same answer is available to a script.
    A UI only a human can read is a UI you cannot test."""
    st: Dict[str, Any] = {"config": node_config(), "writes_enabled": bool(os.getenv("EMBER_INVOKE_TOKEN"))}
    if store is None:
        st["store"] = {"available": False,
                       "reason": "no store bound to this process — the node is serving without ground"}
        st["hosts"] = []
        return st
    arts = _artifacts(store)
    try:
        st["store"] = {"available": True, "artifacts": arts.count()}
    except Exception as exc:                                    # noqa: BLE001
        st["store"] = {"available": True, "count_error": "%s: %s" % (type(exc).__name__, exc)}
    st["hosts"] = hosts(store, probe=probe)
    st["facets"] = facets()
    st["facet_root"] = str(facet_root() or "")
    return st


# ── the page ──────────────────────────────────────────────────────────────────────────────────────
_CSS = """
:root{color-scheme:light dark}
body{font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:2rem;
 max-width:64rem;margin-inline:auto}
h1{font-size:1.25rem;margin:0 0 .25rem} h2{font-size:1rem;margin:2rem 0 .5rem}
.sub{opacity:.7;margin:0 0 1.5rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0}
th,td{text-align:left;padding:.35rem .6rem;border-bottom:1px solid rgba(128,128,128,.25);
 vertical-align:top;font-variant-numeric:tabular-nums}
th{font-weight:600;opacity:.75;font-size:.8rem;text-transform:uppercase;letter-spacing:.03em}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em}
.warn{padding:.6rem .8rem;border-left:3px solid #d97706;background:rgba(217,119,6,.08);margin:.75rem 0}
.err{border-left-color:#dc2626;background:rgba(220,38,38,.08)}
button{font:inherit;padding:.3rem .7rem;border-radius:.35rem;border:1px solid rgba(128,128,128,.4);
 background:transparent;cursor:pointer}
pre{background:rgba(128,128,128,.1);padding:.75rem;border-radius:.4rem;overflow-x:auto;max-width:100%}
"""

_JS = """
async function j(u,o){const r=await fetch(u,o);return r.json()}
async function run(op){
  const out=document.getElementById('out');
  out.textContent='running '+op+' …';
  const r=await j('/api/console/run',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({operator:op})});
  out.textContent=JSON.stringify(r,null,2);
}
"""


def page(store, *, probe: bool = True) -> str:
    st = state(store, probe=probe)
    e = html.escape

    rows = "".join(
        "<tr><td><code>%s</code></td><td>%s</td></tr>" % (e(str(k)), e(str(v)))
        for k, v in st["config"].items()
    )

    if not st["writes_enabled"]:
        gate = ('<div class="warn"><b>Writes are closed.</b> <code>EMBER_INVOKE_TOKEN</code> is unset, '
                'so <code>/hosts/register</code> refuses every registration and this console cannot run '
                'a remote operator. That is the endpoint failing closed, not a bug — set the token to '
                'enable it.</div>')
    else:
        gate = ""

    host_html = ""
    if not st["hosts"]:
        host_html = ('<div class="warn">No prism host has registered with this node. Run '
                     '<code>prism init</code>, start the host with <code>api_uri</code> pointing here, '
                     'and it will POST <code>/hosts/register</code> on boot.</div>')
    for h in st["hosts"]:
        if h.get("error"):
            host_html += '<div class="warn err">%s</div>' % e(h["error"])
            continue
        live = h.get("live") or {}
        if live and not live.get("reachable", True):
            liveline = ('<div class="warn err">Registered but UNREACHABLE at <code>%s</code>: %s</div>'
                        % (e(h.get("endpoint") or ""), e(str(live.get("reason")))))
        elif live.get("reachable"):
            served = (live.get("health") or {}).get("operators") or []
            liveline = "<p>live: <code>%s</code> serving %d operator(s)</p>" % (
                e(str((live.get("health") or {}).get("host", "?"))), len(served))
        else:
            liveline = ""
        ops = "".join(
            "<tr><td><code>%s</code></td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                e(str(o.get("id"))), e(str(o.get("operator_name"))), e(str(o.get("dispatch"))),
                ('<button onclick="run(%s)">run</button>' % html.escape(repr(str(o.get("id"))), quote=True))
                if st["writes_enabled"] else "&mdash;")
            for o in h.get("operators") or []
        ) or '<tr><td colspan="4">no operators announced</td></tr>'
        host_html += (
            "<h2>%s</h2>%s<table><tr><th>operator</th><th>name</th><th>dispatch</th><th></th></tr>%s</table>"
            % (e(str(h.get("id"))), liveline, ops))

    if not st.get("facet_root"):
        facet_html = ('<div class="warn">No facet root configured. Set <code>EMBER_FACET_ROOT</code> '
                      'to the directory an installed bundle unpacks its facets into. '
                      '<b>Unset is not the same as empty</b> — nothing is being served, and nothing '
                      'is missing.</div>')
    elif not st["facets"]:
        facet_html = ('<div class="warn">Facet root <code>%s</code> exists but holds no facets.</div>'
                      % e(st["facet_root"]))
    else:
        facet_html = "<table><tr><th>facet</th><th>state</th><th></th></tr>%s</table>" % "".join(
            "<tr><td><code>%s</code></td><td>%s</td><td>%s</td></tr>" % (
                e(f["name"]),
                "serveable" if f["serveable"] else "<b>no index.html</b> — did it build?",
                ('<a href="/facet/%s/">open</a>' % e(f["name"])) if f["serveable"] else "&mdash;")
            for f in st["facets"])

    store_line = ("store: %s artifacts" % st["store"].get("artifacts")) if st["store"].get("available") \
        else ('<span class="warn">%s</span>' % e(str(st["store"].get("reason"))))

    return (
        "<!doctype html><meta charset=utf-8><title>ember console</title>"
        "<style>%s</style><h1>ember console</h1>"
        "<p class=sub>node <code>%s</code> &middot; %s</p>%s"
        "<h2>configuration</h2><table><tr><th>key</th><th>value</th></tr>%s</table>"
        "<h2>facets</h2>%s"
        "%s<h2>result</h2><pre id=out>—</pre><script>%s</script>"
        % (_CSS, e(st["config"].get("EMBER_NODE_ID") or "?"), store_line, gate, rows,
           facet_html, host_html, _JS)
    )


# ── facets ────────────────────────────────────────────────────────────────────────────────────────
# A crystal is a collection of tektons and facets, and ember is the runner that invokes a crystal,
# so ember serving a crystal's facet is the runner doing its job. Ember finds one without importing
# chorus: the root is configured (`EMBER_FACET_ROOT`) and the files are read from disk. An installed
# bundle populates that root.
#
# This serves whatever static files a facet ships, and renders nothing itself. It is distinct from
# `crystal.workbench`, which delegates rendering to `agience-facet`, outside this workspace.

_FACET_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".ico": "image/x-icon", ".woff2": "font/woff2", ".map": "application/json",
    ".txt": "text/plain; charset=utf-8",
}


def facet_root():
    """The directory holding installed facets, or None when the root is unset or unresolvable. The
    path comes from the environment rather than a guess, and an unset root — no facets served — is
    reported differently from an empty one."""
    from pathlib import Path
    raw = os.getenv("EMBER_FACET_ROOT", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    try:
        return p.resolve(strict=True)
    except OSError:
        return None


def facets() -> List[Dict[str, Any]]:
    """Installed facets: one entry per immediate subdirectory of the root, with `serveable` set from
    whether it holds an `index.html`. A directory without one is listed as `serveable: False` rather
    than hidden — a facet that did not build is the case you most need to see."""
    root = facet_root()
    if root is None:
        return []
    out = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        out.append({"name": child.name, "serveable": (child / "index.html").is_file(),
                    "path": str(child)})
    return out


def facet_file(name: str, rel: str):
    """Resolve `(<facet>, <relative path>)` to a real file under the root, or None.

    Path traversal is the risk here, and containment is proved after resolution rather than by
    string-matching `..`. An absolute path, a symlink, a URL-encoded segment or a Windows drive
    letter all resolve outside the root while containing no literal `..`, so a check like
    `".." not in rel` passes them through. `Path.resolve()` collapses all of them, and
    `is_relative_to` is then a fact about the resulting path rather than a guess about the input
    string. Returns None for anything outside, and the caller sends 404.
    """
    root = facet_root()
    nm = str(name or "").strip()
    if root is None or not nm:
        return None
    # The facet name is a single, plain path component. Checking it here, rather than relying on
    # containment after resolution, is what lets the facet directory itself be a symlink or a Windows
    # junction — how a facet gets installed without copying its build output. Requiring the resolved
    # base to sit under the root would leave a junction pointing at the real crystal's `dist/`
    # resolving outside it, with nothing served.
    if nm in (".", "..") or "/" in nm or "\\" in nm or ":" in nm:
        return None
    try:
        base = (root / nm).resolve()
        if not base.is_dir():
            return None
        # Containment is proved against the resolved base, so traversal out of the facet stays
        # closed even though the base may live elsewhere on disk.
        target = (base / (rel.lstrip("/") or "index.html")).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return None
        return target
    except (OSError, ValueError):
        return None


def facet_content_type(path) -> str:
    """Content type by suffix, from a closed map. An unknown suffix serves as
    `application/octet-stream`: a browser that downloads an unrecognised file is a nuisance, and a
    browser that executes one on a guessed `text/html` is a vulnerability."""
    return _FACET_TYPES.get(path.suffix.lower(), "application/octet-stream")


__all__ = ["state", "page", "hosts", "node_config", "run",
           "facets", "facet_root", "facet_file", "facet_content_type"]
