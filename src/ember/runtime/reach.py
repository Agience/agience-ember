"""Ember-side reach wiring — the runner reaches a persona capability over the ground plane, no chorus.

Ember is "simply a runner" ([[ember-is-a-runner]]): it reaches personas (lumen's `op.respond`, sage's
`op.retrieve`, …) over the shared ground plane — the mantle lattice — and never imports chorus, iris,
lumen, or sage. The signal-native reach core lives in `prism.reach` (comms, propagation, and the wire
are one thing): the `Reactor` that serves capabilities and issues reaches, keyed by an abstract
`Keyring` and gated by an abstract `Lightcone` (see `prism/reach.py`, `prism/plane.py`). The ground
those reaches operate over is the mantle lattice, but the reach code itself belongs to `prism`. Those
two contracts are the only seam; this module supplies ember's own backings for them, mirroring the
chorus-side `iris/comms/wiring.py` but built entirely on `prism.reach` + `ember.access` + `mantle.*`.
Ember keeps the instrument — `ember.optics`, plus `prism.resolution` and `prism.adaptive_cut` for the cut
geometry the instrument reaches for — and reaches it from `ember.signal` and `ember.ontology`.

The separation between ember and chorus is architectural, not a licence boundary: `agience-ember/LICENSE`
is AGPL-3.0 (the GNU Affero General Public License v3.0). Ember is a runner and chorus is the persona
layer, so a runner that imported the personas it runs could only run the personas it shipped with.

  - `EmberLightcone(store)`  — `reaches(principal)` = the principal's read light-cone (CRUDEASIO grants +
                               containment) via `ember.access.reachable_collections`, plus its own ember
                               address, plus the session ground connections a `Reactor` `join`s. Comms
                               reach is read-access — one mechanism, no second sharing path. The reach fn
                               is injectable so the adapter is testable without a live store.
  - `EmberKeyring(root_secret)` — per-group AES-256 keys from the same fleet content-key derivation used at
                               rest (`mantle…content_cache.collection_key = HKDF(root, origin_root)`), so a
                               comms group key equals that group's collection key: a member derives it
                               because it reaches the group, a non-member never can (isolation is
                               cryptographic). `principal_keys` gates by the light-cone.

`reactor(store, principal, *, root_secret, fabric)` assembles a `prism.reach.Reactor` wired with those
two adapters — the one object an ember call site (serve / genesis / router) holds to reach lumen or sage.
`reach(store, principal, need, *, to, root_secret, fabric)` is the fire-and-collect convenience over a live
fabric: place a need on capability `to`, return the evidence picked up off the ground.

Import discipline. This module imports `prism.reach` (module level), `mantle.db.*` (lazy) and
`ember.access` (lazy). `test_reach_wiring.py` enforces the exclusion in two independent ways:
  · `test_reach_module_imports_nothing_from_chorus_iris_lumen_or_sage` — a source scan for the banned
    import statements;
  · `test_importing_ember_reach_pulls_in_no_chorus_module` — after exercising every path, no persona or
    platform module is in `sys.modules`, and `prism.reach` is.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from prism.reach import GROUND, Reactor

# `EmberKeyring` and `EmberLightcone` are re-exported here under the names ember's own call sites use,
# so those call sites are unaffected by where the implementation lives; the one implementation is
# `mantle/db/plane.py`. Neither depends on ember (their only imports are
# `mantle.db.access` and `…content_cache`). Keeping the real grant-backed plane in mantle rather
# than in ember means a chorus caller that needs it imports mantle directly, not the runner, to reach
# the store's grant model. A plain alias to the one implementation keeps key derivation identical
# across hosts, since both sides always resolve to the same class.
from mantle.db.plane import LatticeKeyring as EmberKeyring
from mantle.db.plane import LatticeLightcone as EmberLightcone

__all__ = ["EmberLightcone", "EmberKeyring", "reactor", "reach", "GROUND"]


def reactor(store: Any, principal: str, *, root_secret: bytes, fabric: Any = None,
            reach: Optional[Callable[[Any, str], Iterable[str]]] = None, ground: str = GROUND,
            fallback: Any = None, hlc: Any = None) -> Reactor:
    """Build the `prism.reach.Reactor` an ember call site holds — serve capabilities and issue reaches
    over the ground plane, wired with ember's own `EmberLightcone`/`EmberKeyring` (no chorus).

    `store` is ember's lattice store (holds `.artifacts`/`.graph` for the light-cone); `principal` is the
    persona/runner identity; `root_secret` the fleet content-key root; `fabric` a live streaming fabric
    (`LoopbackFabric` in tests, WebRTC/QUIC/RF in prod) — omit it and the reactor rides a `fallback`
    carrier (store-and-forward, driven by `Reactor.pump`). `reach` injects a light-cone fn for tests."""
    lightcone = EmberLightcone(store, reach=reach)
    keyring = EmberKeyring(root_secret)
    return Reactor(principal, keyring=keyring, lightcone=lightcone, fabric=fabric, ground=ground,
                   fallback=fallback, hlc=hlc)


def reach(store: Any, principal: str, need: Any, *, to: str, root_secret: bytes, fabric: Any = None,
          reach_fn: Optional[Callable[[Any, str], Iterable[str]]] = None, ground: str = GROUND,
          fallback: Any = None) -> Any:
    """Places a need on capability `to` and returns the evidence picked up off the ground — the fire-
    and-collect convenience for ember call sites (serve / genesis / router) over a live `fabric`, where
    the round-trip is synchronous propagation (place need → provider fires → discharges evidence onto
    the ground → this reactor's ground connection picks it up). Returns `None` while nothing has
    resolved (silence stays silence). For the store-and-forward path, hold a `reactor(...)` and drive
    `pump`."""
    rc = reactor(store, principal, root_secret=root_secret, fabric=fabric, reach=reach_fn, ground=ground,
                 fallback=fallback)
    handle = rc.reach(need, to=to)
    return rc.evidence(handle)
