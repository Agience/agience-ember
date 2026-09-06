"""Production e2e: the deployed endpoint, hit over the public URL the way a client (VS Code
Continue) would. It uses no SSH and no localhost, so a pass means the deterministic Ember
copilot is live at https://lumen.agience.ai/v1.

  EMBER_LIVE_URL overrides the base (default https://lumen.agience.ai/v1).

Domains, all deterministic; every answer is grounded and verified, or comes back ungrounded:
  MODELS · ARITHMETIC · PRIMALITY · DICTIONARY · ONTOLOGY(is-a, part-of) · LEARN · STREAM · REFUSAL
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.getenv("EMBER_LIVE_URL", "https://lumen.agience.ai/v1")


def _chat(q, stream=False):
    body = json.dumps({"model": "agience-ember",
                       "messages": [{"role": "user", "content": q}], "stream": stream}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    r = urllib.request.urlopen(req, timeout=30)
    return r.read().decode("utf-8", "ignore") if stream else json.loads(r.read())


def main() -> int:
    passed = [0]

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
        if not cond:
            raise AssertionError(name)
        passed[0] += 1

    def content(q):
        return _chat(q)["choices"][0]["message"]["content"]

    def grounded(resp):
        return resp.get("x_agience", {}).get("grounded")

    print(f"Testing LIVE endpoint: {BASE}")

    models = json.loads(urllib.request.urlopen(BASE + "/models", timeout=20).read())
    check("MODELS: agience-ember live", any(m["id"] == "agience-ember" for m in models["data"]))

    r = _chat("what is 7 * 8")
    check("ARITHMETIC: 7*8=56 grounded", "56" in r["choices"][0]["message"]["content"] and grounded(r))
    check("PRIMALITY: 51 not prime", "not prime" in content("is 51 prime"))
    check("ARITHMETIC: factor 60", "2^2" in content("factor 60") or "2 × 2" in content("factor 60") or "×" in content("factor 60"))

    check("DICTIONARY: define artifact", "man-made object" in content("define artifact"))
    check("ONTOLOGY is-a: dog", any(w in content("what is a dog a kind of") for w in ("animal", "canid")))
    check("ONTOLOGY part-of: wheel", "vehicle" in content("what is a wheel part of") or "mechanism" in content("what is a wheel part of"))

    learn = content("infer the operator: 10,3 to 7 and 5,2 to 3")
    check("LEARN: infers subtract", "subtract" in learn, learn[:50])

    raw = _chat("what does mercury mean", stream=True)
    check("STREAM: chunks + [DONE]", "chat.completion.chunk" in raw and "data: [DONE]" in raw)

    r2 = _chat("what is the airspeed velocity of an unladen swallow")
    check("REFUSAL: unheld fact declined", grounded(r2) is False)

    print(f"\nLIVE LUMEN E2E: ALL {passed[0]} CHECKS PASS — deterministic copilot is live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
