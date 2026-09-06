#!/usr/bin/env python
"""Talk to Ember. A tiny interactive REPL over the local OpenAI-compatible /v1 endpoint.

    python scripts/ember-chat.py            # connects to http://127.0.0.1:8091
    python scripts/ember-chat.py --url http://127.0.0.1:8091

Type a question, get Ember's answer: cited, from its own knowledge, or an honest absence when
nothing grounds it.
Ctrl-C or 'exit' to quit. Stdlib only — no deps, no key.
"""
import argparse
import json
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8091")
args = ap.parse_args()
ENDPOINT = args.url.rstrip("/") + "/v1/chat/completions"


def ask(text: str) -> str:
    body = json.dumps({"model": "ember", "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


def main() -> int:
    print(f"ember-chat → {ENDPOINT}   (Ctrl-C or 'exit' to quit)\n")
    while True:
        try:
            q = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return 0
        if q.lower() in {"exit", "quit", ":q"}:
            return 0
        if not q:
            continue
        try:
            print("\nember ›", ask(q), "\n")
        except Exception as e:
            print(f"  [error: {e}]  — is `ember serve` running on {args.url}?\n", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
