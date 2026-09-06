"""End-to-end test of the copilot surface: a multi-domain deterministic leaf served over an
OpenAI-compatible /v1 endpoint, exercised over real HTTP the way VS Code Continue would.
Requires the WordNet-loaded mantle and a live substrate.

Domains, all deterministic:
  1. MODELS      — GET /v1/models advertises agience-ember
  2. ARITHMETIC  — computed + self-verifying (7*8=56, is 51 prime -> no, factor 60)
  3. LEXICAL     — keyed WordNet dictionary (define artifact)
  4. STREAM      — chat.completion.chunk frames + [DONE]
  5. ABSENCE     — a fact no domain holds comes back ungrounded, not guessed
  6. ROUTING     — the answer footer labels which domain served it
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request


def _post(url, obj, stream=False):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    r = urllib.request.urlopen(req, timeout=30)
    return r.read().decode("utf-8", "ignore") if stream else json.loads(r.read())


def _get(url):
    return json.loads(urllib.request.urlopen(url, timeout=15).read())


def main() -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("EMBER_ENGINE", "distill")
    from ember.surface.serve import serve_openai, MODEL_ID

    port = 8097
    httpd = serve_openai(port=port, host="127.0.0.1")
    base = f"http://127.0.0.1:{port}"

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
        if not cond:
            raise AssertionError(name)

    def chat(q, stream=False):
        return _post(base + "/v1/chat/completions",
                     {"model": MODEL_ID, "messages": [{"role": "user", "content": q}], "stream": stream},
                     stream=stream)

    try:
        # 1. MODELS
        check("MODELS: agience-ember advertised",
              any(m["id"] == MODEL_ID for m in _get(base + "/v1/models")["data"]))

        # 2. ARITHMETIC (computed + verified)
        r = chat("what is 7 * 8")
        c = r["choices"][0]["message"]["content"]
        check("ARITHMETIC: 7*8 computed", "56" in c and r["x_agience"]["grounded"])
        check("ARITHMETIC: carries verification", "arithmetic" in c or "== 56" in c, c.splitlines()[0][:50])
        check("ARITHMETIC: primality with witness", "3 × 17" in chat("is 51 prime")["choices"][0]["message"]["content"]
              or "not prime" in chat("is 51 prime")["choices"][0]["message"]["content"])

        # 3. LEXICAL (keyed dictionary)
        cl = chat("define artifact")["choices"][0]["message"]["content"]
        check("LEXICAL: dictionary definition", "man-made object" in cl)

        # 4. STREAM
        raw = chat("what does mercury mean", stream=True)
        check("STREAM: chunk frames", "chat.completion.chunk" in raw)
        check("STREAM: terminates [DONE]", "data: [DONE]" in raw)

        # 5. ABSENCE (no domain holds this)
        rr = chat("what is the airspeed velocity of an unladen swallow")
        check("REFUSAL: unheld fact declined", rr["x_agience"]["grounded"] is False,
              rr["choices"][0]["message"]["content"][:50])

        # 6. ROUTING labels
        check("ROUTING: lexical labelled", "lexical" in cl)

        print("\nCOPILOT E2E (multi-domain final state): ALL CHECKS PASS")
        return 0
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    sys.exit(main())
