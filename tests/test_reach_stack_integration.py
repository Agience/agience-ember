"""The ember reach stack, assembled — ember's own reactor adapters, self-driving, end to end.

The unit tests cover the pieces: `ember.runtime.reach` builds a `Reactor` with
`EmberLightcone`/`EmberKeyring` (`test_reach_wiring`); `prism.carriers.StoreCarrier` persists a
round-trip through a shared lattice (`test_store_carrier_reach`); `prism.pump.PumpLoop` drives it
self-driving (`test_pump_loop`). This ties the ember side together: two reactors built by
`ember.runtime.reach.reactor` — the grant-backed light-cone contract and the fleet content-key ring,
rather than prism's test doubles — share one `StoreCarrier`, and a `PumpLoop` drives the round-trip
with nobody hand-calling `pump`. It is the local shape of what a host wires against a live store.

The invariants:

  assembled    — a reach placed on a requester built by `ember.runtime.reach.reactor` returns the
                 server's exact result once the loop runs, so ember's adapters interoperate with the
                 carrier.
  self-driving — a running `PumpLoop` resolves it with no manual pump, which is what makes the
                 wiring a cadence rather than a sequence of hand-made calls.
  key-gated    — a reactor keyed off a different fleet root (`EmberKeyring` root_secret) picks up
                 nothing off the same store: the isolation is cryptographic, not a filter.
"""
from __future__ import annotations

import time

from prism.carriers import StoreCarrier
from prism.pump import PumpLoop
from ember.runtime import reach as R
from mantle.db import open_lattice

CAP, GROUND = "op.retrieve", "ground"
ROOT = b"fleet-root-secret-0001"


def _retrieve(need):
    return [{"id": "d.hamlet", "score": 1.0, "q": (need or {}).get("query", "")}]


def _store(tmp_path):
    L = open_lattice(str(tmp_path / "ground.db"), origin="node-71")
    L.artifacts.ensure_schema()
    return L


def _lc(store, principal):
    """Injected light-cone fn (the `EmberLightcone` reach seam): both principals reach the
    capability's group, so `EmberKeyring` derives the same group key for it. The Reactor joins
    ground on its own."""
    return {CAP}


def _stack(store, *, root=ROOT):
    carrier = StoreCarrier(store)
    server = R.reactor(store, "server", root_secret=root, fallback=carrier, reach=_lc, ground=GROUND)
    server.serve(CAP, _retrieve)
    requester = R.reactor(store, "requester", root_secret=root, fallback=carrier, reach=_lc, ground=GROUND)
    return carrier, server, requester


def test_the_assembled_ember_stack_answers_a_reach(tmp_path):
    carrier, server, requester = _stack(_store(tmp_path))
    handle = requester.reach({"query": "Hamlet"}, to=CAP)
    assert requester.evidence(handle) is None                 # nothing yet: the cadence has not run
    PumpLoop(carrier, [server, requester]).tick()             # server → requester in one pass
    assert requester.evidence(handle) == _retrieve({"query": "Hamlet"})


def test_the_loop_self_drives_the_ember_stack(tmp_path):
    carrier, server, requester = _stack(_store(tmp_path))
    with PumpLoop(carrier, [server, requester], interval=0.005):
        handle = requester.reach({"query": "Newton"}, to=CAP)
        got = None
        for _ in range(400):                                  # bounded wait on the background thread (<=2s)
            got = requester.evidence(handle)
            if got is not None:
                break
            time.sleep(0.005)
    assert got == _retrieve({"query": "Newton"})


def test_a_different_fleet_root_gets_nothing(tmp_path):
    store = _store(tmp_path)
    carrier, server, requester = _stack(store)                # ROOT
    handle = requester.reach({"query": "Hamlet"}, to=CAP)
    PumpLoop(carrier, [server, requester]).tick()
    assert requester.evidence(handle)[0]["id"] == "d.hamlet"  # holder on the fleet root gets the answer

    snoop = R.reactor(store, "requester", root_secret=b"a-different-root", fallback=carrier,
                      reach=_lc, ground=GROUND)               # same principal, different key
    snoop.pump(carrier)
    assert snoop.evidence(handle) is None                     # cryptographic isolation via EmberKeyring
