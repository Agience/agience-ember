"""ember CLI — ingest, index, and query the local leaf; serve it as an OpenAI-compatible endpoint.

Subcommands: `ingest` fills the local ontology (standalone, deterministic); `reindex` rebuilds the
derived index purely from the durable store; `ask` answers a query from it, cited; `status` reports
leaf state; `serve` exposes an OpenAI-compatible `/v1` endpoint, e.g. for VS Code Continue.

Ember does not import chorus, so it carries no governance-chore command such as docs cleanup — that
surface lives in `seraph/agent/` and is reached over the plane, not imported.

Ember never constructs a model reasoner: there is no `--consolidate`, `--lumen-url`, or `--model`
flag, and passing one fails argparse's "unrecognized arguments" rather than being silently ignored.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _load_token(args) -> str:
    if getattr(args, "token_file", ""):
        with open(args.token_file, encoding="utf-8") as f:
            return f.read().strip()
    return (os.getenv("EMBER_TOKEN") or os.getenv("AGIENCE_TOKEN") or "").strip()


def _load_anchors(path):
    """Load the canonical AnchorSet from `path`. Anchors are provisioned, never derived locally: an
    anchor id is content-addressed over its embedding, so a locally-derived set would mint region
    ids no peer computes."""
    from mantle.search.anchors.anchorset import AnchorSet
    return AnchorSet.load(path)


def cmd_ingest(args) -> int:
    """Observe+describe (via describe-operators) into the durable local mantle-shard
    (SQLite lattice + local CAS), then build the derived index. Standalone, deterministic."""
    from ember.corpus.ingest import ingest_to_store
    import time
    t = time.time()
    ember, n, store = ingest_to_store(paths=list(args.paths) or None, wordnet=args.wordnet,
                                      wiki_dump=args.wiki or None,
                                      anchors=_load_anchors(args.anchors))
    print(f"ingested {n} artifacts to store in {time.time()-t:.1f}s  ->  index {ember.status()['cache']}")
    print(f"store artifacts (committed): {store.artifacts.count(state='committed')}")
    print(f"index cache: {ember.settings.cache_dir}")
    return 0


def cmd_reindex(args) -> int:
    """Rebuild the derived index purely from the durable store — proves the store is truth."""
    from ember.corpus.ingest import rebuild_index_from_store
    import time
    t = time.time()
    ember, n, _store = rebuild_index_from_store(anchors=_load_anchors(args.anchors))
    print(f"reindexed {n} artifacts from store in {time.time()-t:.1f}s -> {ember.status()['cache']}")
    return 0


def cmd_ask(args) -> int:
    """Ask the local leaf — answers only from its ontology, cited, and states plainly when it has
    no answer rather than guessing one."""
    from ember.corpus.ingest import open_local
    ember = open_local()
    if not ember.ready:
        print("ember: nothing ingested yet — run `ember ingest <paths>` first", file=sys.stderr)
        return 2
    r = ember.ask(" ".join(args.query), k=args.k)
    print(f"[hit={r.routing.hit} grounded={r.answer.grounded} offline={r.served_offline}]")
    print(r.answer.text or "(no answer)")
    if r.answer.cited:
        print("\ncited:")
        for c in r.answer.cited[:args.k]:
            print(f"  - {c}")
    return 0


def cmd_status(args) -> int:
    from ember.corpus.ingest import open_local
    print(open_local().status())
    return 0


def cmd_serve(args) -> int:
    """Serve the local leaf as an OpenAI-compatible /v1 endpoint (point VS Code Continue here)."""
    from ember.surface.serve import serve_openai, MODEL_ID
    httpd = serve_openai(port=args.port, host=args.host, k=args.k, watch=args.watch or None)
    print(f"ember serving OpenAI /v1 at http://{args.host}:{args.port}/v1  (model={MODEL_ID})")
    if args.watch:
        print(f"  watching {args.watch} — knowledge stays live as files change")
    print(f"  browse the ontology at http://{args.host}:{args.port}/browse")
    print("point Continue's apiBase here; Ctrl-C to stop.")
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        httpd.shutdown()
        return 0


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="ember", description="Ember — the local Agience leaf/agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="fill the local ontology (standalone, deterministic)")
    ing.add_argument("paths", nargs="*", help="files/dirs to ingest (code, docs)")
    ing.add_argument("--wordnet", action="store_true", help="also ingest WordNet (lexical ground truth)")
    ing.add_argument("--wiki", default="", help="path to a Wikipedia .bz2 dump to ingest")
    ing.add_argument("--anchors", metavar="PATH", required=True,
                     help="path to the canonical AnchorSet artifact (JSON). Anchors are provisioned, "
                          "never derived here: a locally-clustered set routes into cells no peer shares.")
    ing.set_defaults(func=cmd_ingest)

    ri = sub.add_parser("reindex", help="rebuild the derived index from the durable store")
    ri.add_argument("--anchors", metavar="PATH", required=True,
                    help="path to the canonical AnchorSet artifact (JSON). See `ingest --anchors`.")
    ri.set_defaults(func=cmd_reindex)

    a = sub.add_parser("ask", help="ask the local leaf (deterministic, cited, honest refusal)")
    a.add_argument("query", nargs="+", help="the question")
    a.add_argument("-k", type=int, default=4, help="max evidence spans")
    a.set_defaults(func=cmd_ask)

    st = sub.add_parser("status", help="show the local leaf's state")
    st.set_defaults(func=cmd_status)

    sv = sub.add_parser("serve", help="serve an OpenAI-compatible /v1 endpoint for VS Code Continue")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8091)
    sv.add_argument("-k", type=int, default=6, help="max evidence spans per answer")
    sv.add_argument("--watch", default="", help="workspace dir to watch for live reindex on change")
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
