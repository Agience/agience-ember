"""Ember-side reach wiring (`ember/runtime/reach.py`), tested to `pharos/genesis/TEST-ARCHITECTURE.md`:
named invariants measured over a generated world-space against an independent oracle, plus focused
isolation and import-discipline checks. The reach core (`prism.reach` — comms, propagation and the wire
are one thing, so prism has no `comms` subpackage) is covered by `iris/tests/test_reach.py`. This suite
covers the ember adapters (`EmberLightcone`/`EmberKeyring`), which back the reach onto the real grant
light-cone from `mantle.db.access.reachable_collections` and the fleet's at-rest `collection_key`
derivation, without importing chorus, iris, lumen or sage. Every assertion is one of:

  (1) delivery   — a need placed via `ember.runtime.reach.reactor(...).reach(need, to=cap)` reaches
                   exactly the providers whose light-cone reaches `cap` (the grant-backed entitlement).
  (2) provenance — the evidence returns through the shared ground, referencing the need (`root` and
                   `in_reply_to` equal the reach handle); the requester correlates by provenance, with
                   no carried return address.
  (3) isolation  — a requester on a different ground from the provider opens nothing; a principal whose
                   light-cone does not reach a capability may serve it and still absorb nothing, because
                   needs are key-gated. Both are cryptographic, driven by the real access backing.
  (4) backing    — the ember keyring's group key is `mantle`'s `collection_key`, and the ember
                   lightcone's default path is `reachable_collections` (measured equal, not stubbed).
  (5) discipline — `ember.runtime.reach` pulls in no iris, chorus, lumen or sage module (source scan
                   and sys.modules).

The oracle is computed directly from the membership relation (who is granted which capability) rather
than from the plane's `reaches`, so the plane and the test cannot share a bug.
"""
from __future__ import annotations

import base64
import inspect
import random
import sys

import ember.runtime.reach as er
from prism.plane import HLC
from prism.streams import LoopbackFabric

_GRANT_CT = "application/vnd.agience.grant+json"
SEEDS = range(60)   # the generated world-space (each seed = a reproducible world + reach script)


# ── a tiny real store: grants drive `reachable_collections` (no injected reach fn) ────────────────────
class _FakeArtifacts:
    """The one face the light-cone reads: grant docs by content type and state. This exercises the
    real grant path rather than an injected membership stub."""

    def __init__(self, grants):
        self._grants = list(grants)

    def list_artifacts(self, content_type=None, state=None, include_archived=False):
        if content_type != _GRANT_CT:
            return []
        rows = [g for g in self._grants if state is None or g.get("state") == state]
        # Head-only by default, as `mantle.db.vertex.list_artifacts` is — an explicit
        # `state` has already decided the set, so `include_archived` governs only the unfiltered
        # read, and a grant with no `state` is committed-by-absence and stays. The light-cone's
        # grant scan (`lattice_api._grant_docs_by`) asks with `include_archived=True` because it
        # filters lifecycle itself; a fake that hid archived grants from it would answer the
        # narrower question and let a revoked grant read as absent rather than as revoked.
        if state is None and not include_archived:
            rows = [g for g in rows if g.get("state") != "archived"]
        return rows


class _FakeStore:
    """The lattice store shape the light-cone reads: `.artifacts` and `.graph`. A `graph` of None
    means flat, with no containment."""

    def __init__(self, grants):
        self.artifacts = _FakeArtifacts(grants)
        self.graph = None


def _grant(grantee, resource):
    """An active, unexpired, readable user grant: `grantee` reaches `resource`, a capability collection.

    `effect` is as load-bearing as `can_read`. The light-cone asks `grant.mask_of(...).allows("read")`
    — bits AND effect — and `grant_is_allow` matches "allow" positively, so a doc carrying the bit and
    no effect is an unrecognized effect and confers nothing. `Grant.to_dict` always writes the field,
    so a doc without it is not a grant the store could ever hold; omitting it here would leave the
    light-cone reaching nothing and every delivery assertion below vacuously satisfied by silence."""
    return {"content_type": _GRANT_CT, "state": "active", "effect": "allow",
            "grantee_id": grantee, "grantee_type": "user",
            "can_read": True, "resource_id": resource}


def _store_granting(offers):
    """A real store whose grants encode `offers` = {provider -> {caps}}: offering a capability = being
    granted read on its collection, so the provider's light-cone reaches it."""
    grants = [_grant(p, c) for p, cs in offers.items() for c in cs]
    return _FakeStore(grants)


# ── the world harness: providers granted capabilities, requesters placing needs, one shared ground ────
class World:
    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.root = b"ember-reach-root-%d" % seed
        self.caps = ["op.cap%d" % i for i in range(self.r.randint(1, 4))]
        self.providers = ["prov%d" % i for i in range(self.r.randint(1, 4))]
        self.requesters = ["req%d" % i for i in range(self.r.randint(1, 3))]
        # the membership relation (ground truth for the oracle): who is granted which capability.
        self.offers = {p: set(self.r.sample(self.caps, self.r.randint(0, len(self.caps))))
                       for p in self.providers}
        self.store = _store_granting(self.offers)               # real grant-backed access
        self.needs = []
        for i in range(self.r.randint(1, 8)):
            self.needs.append({"req": self.r.choice(self.requesters), "cap": self.r.choice(self.caps),
                               "need": {"q": "n%d" % i, "i": i}})
        self._n = 0

    def clk(self) -> int:                                       # one shared, strictly-increasing clock
        self._n += 1
        return self._n

    @staticmethod
    def handler(cap):
        """A deterministic capability: evidence is a pure function of (cap, need). Every provider of a
        capability computes the same evidence, so the answer is well-defined whoever resolves it."""
        return lambda need, _c=cap: {"cap": _c, "hits": [_c, need]}

    def providers_of(self, cap) -> set:
        """The independent oracle: a need for `cap` is resolved by the providers granted it, read
        straight from the membership relation rather than through the plane's `reaches`."""
        return {p for p, cs in self.offers.items() if cap in cs}


def _all_handled(reactor) -> set:
    seen = set()
    for prov in reactor._providers:                            # merge every capability this persona serves
        seen |= set(prov.handled)
    return seen


def _run_fabric(w: World):
    """Live path: one Reactor per persona, built via `ember.runtime.reach.reactor(...)` on a single
    loopback fabric and grounded on the default ground plane. Providers serve, requesters reach, evidence
    discharges onto the ground and the requester picks it up — a synchronous propagation cascade."""
    fabric = LoopbackFabric()
    reactors = {}
    for p in w.providers:
        rc = er.reactor(w.store, p, root_secret=w.root, fabric=fabric, hlc=HLC(p, clock=w.clk))
        for c in w.offers[p]:
            rc.serve(c, w.handler(c))
        reactors[p] = rc
    for q in w.requesters:
        reactors[q] = er.reactor(w.store, q, root_secret=w.root, fabric=fabric, hlc=HLC(q, clock=w.clk))
    results = []
    for nd in w.needs:
        handle = reactors[nd["req"]].reach(nd["need"], to=nd["cap"])
        ev = reactors[nd["req"]].evidence(handle)
        handled = {p for p in w.providers if handle in _all_handled(reactors[p])}
        results.append((nd, handle, ev, handled))
    return reactors, results


# ── Invariant 1 and isolation: a need reaches exactly the granted providers (independent oracle) ──────
def test_need_reaches_exactly_the_capability_providers():
    for seed in SEEDS:
        w = World(seed)
        _reactors, results = _run_fabric(w)
        for nd, _handle, _ev, handled in results:
            oracle = w.providers_of(nd["cap"])
            assert handled == oracle, "seed %d cap %s: reached %s != oracle %s" % (
                seed, nd["cap"], sorted(handled), sorted(oracle))


# ── Invariant 2: evidence returns through the ground, referencing the need ────────────────────────────
def test_evidence_returns_through_the_ground_referencing_the_need():
    for seed in SEEDS:
        w = World(seed)
        reactors, results = _run_fabric(w)
        for nd, handle, ev, _handled in results:
            if w.providers_of(nd["cap"]):
                assert ev == w.handler(nd["cap"])(nd["need"])          # the answer, picked up off the ground
                prov = reactors[nd["req"]].provenance(handle)
                assert prov and all(r["root"] == handle for r in prov)  # every band references the need
                assert prov[0]["in_reply_to"] == handle                 # single hop → answers the need
            else:
                assert ev is None                                       # no provider → silence stays silence
                assert reactors[nd["req"]].provenance(handle) == []


def test_correlation_by_provenance_never_crosses_between_needs():
    for seed in SEEDS:
        w = World(seed)
        reactors, results = _run_fabric(w)
        for nd, handle, _ev, _handled in results:
            if not w.providers_of(nd["cap"]):
                continue
            assert reactors[nd["req"]].evidence(handle) == w.handler(nd["cap"])(nd["need"])
            for b in reactors[nd["req"]].bands(handle):                 # only bands rooted at ITS handle
                assert b == w.handler(nd["cap"])(nd["need"])


# ── a requester reaches sage's `op.retrieve` over the ground, with no import ──────────────────────
def test_requester_reaches_op_retrieve_over_the_ground():
    """A stub provider serves `op.retrieve` and returns fake hits; a requester built via
    `ember.runtime.reach.reactor(...)` places a need, the provider discharges evidence onto the ground,
    and the requester picks it up by following provenance — using the requester's own grant-backed
    light-cone, with no in-process import and no carried return address."""
    store = _store_granting({"sage": {"op.retrieve"}})           # sage is granted op.retrieve (real cone)
    root = b"genesis-root"
    fabric = LoopbackFabric()

    def sage_retrieve(need):                                     # stub — stands in for sage's tekton
        q = need["query"]
        return {"cap": "op.retrieve",
                "hits": [{"doc": "d1", "score": 0.91, "text": "…%s…" % q},
                         {"doc": "d2", "score": 0.72, "text": "…more on %s…" % q}]}

    sage = er.reactor(store, "sage", root_secret=root, fabric=fabric)
    sage_prov = sage.serve("op.retrieve", sage_retrieve)
    lumen = er.reactor(store, "lumen", root_secret=root, fabric=fabric)

    # the call site — place a need on `op.retrieve`; evidence returns via the ground.
    handle = lumen.reach({"query": "who wrote Hamlet?"}, to="op.retrieve")
    evidence = lumen.evidence(handle)

    assert evidence["cap"] == "op.retrieve"
    assert [h["doc"] for h in evidence["hits"]] == ["d1", "d2"]         # sage-style evidence, verbatim
    assert sage_prov.handled[handle] == evidence                       # sealed + crossed the plane, not shared
    prov = lumen.provenance(handle)
    assert prov[0]["in_reply_to"] == handle and prov[0]["root"] == handle and prov[0]["origin"] == "sage"

    # and the fire-and-collect convenience returns the same evidence in one call.
    ev2 = er.reach(store, "lumen", {"query": "who wrote Hamlet?"}, to="op.retrieve",
                   root_secret=root, fabric=fabric)
    assert ev2 == evidence


# ── Isolation (3): a requester not connected to the provider's ground opens nothing ───────────────────
def test_a_requester_not_connected_to_the_ground_gets_nothing():
    store = _store_granting({"sage": {"op.retrieve"}})
    root = b"root"
    fabric = LoopbackFabric()
    sage = er.reactor(store, "sage", root_secret=root, fabric=fabric, ground="mesh")   # provider on ground "mesh"
    sage.serve("op.retrieve", lambda need: {"hits": [need]})
    lumen = er.reactor(store, "lumen", root_secret=root, fabric=fabric, ground="mesh") # same ground → connected
    snoop = er.reactor(store, "snoop", root_secret=root, fabric=fabric, ground="other")# a different ground

    handle = lumen.reach({"q": "x"}, to="op.retrieve")
    assert lumen.evidence(handle) == {"hits": [{"q": "x"}]}            # shares the ground → gets it
    assert snoop._inbox.bands(handle) == []                           # not on sage's ground → circuit is open


# ── Isolation (3): a principal whose light-cone lacks the cap absorbs nothing (needs are key-gated) ───
def test_non_provider_on_the_capability_wire_absorbs_nothing():
    store = _store_granting({"sage": {"op.retrieve"}})               # mallory is granted nothing: no key
    root = b"root"
    fabric = LoopbackFabric()
    sage = er.reactor(store, "sage", root_secret=root, fabric=fabric)
    sage.serve("op.retrieve", lambda need: {"hits": [need]})
    mallory = er.reactor(store, "mallory", root_secret=root, fabric=fabric)
    mal_prov = mallory.serve("op.retrieve", lambda need: {"stolen": need})  # subscribes, but lacks the cap key
    lumen = er.reactor(store, "lumen", root_secret=root, fabric=fabric)

    handle = lumen.reach({"q": "x"}, to="op.retrieve")
    assert lumen.evidence(handle) == {"hits": [{"q": "x"}]}           # sage answered
    assert mal_prov.handled == {}                                     # isolation: opened no sealed need


# ── Backing (4): the ember keyring is the fleet collection-key derivation ──────────────────────────────
def test_ember_keyring_group_key_is_the_collection_key_derivation():
    from mantle.db.content_cache import collection_key
    root = b"a-fleet-root-secret"
    kr = er.EmberKeyring(root)
    for group in ("op.retrieve", "op.respond", "ground", "mesh"):
        # Raw bytes: the plane seals with AES-256-GCM, which takes the key material directly.
        assert kr.group_key(group) == collection_key(root, group)
        assert len(kr.group_key(group)) == 32, "an AES-256 key is 32 raw bytes"
    # a non-empty group is required by the derivation (empty would collide all groups on one key).
    import pytest
    with pytest.raises(ValueError):
        collection_key(root, "")


# ── Backing (4): the ember lightcone's default path is `reachable_collections` (+ self + join) ───────
def test_ember_lightcone_default_path_is_ember_access_reachable_collections():
    from mantle.db.access import reachable_collections
    store = _store_granting({"sage": {"op.retrieve", "op.index"}, "lumen": {"op.respond"}})
    lc = er.EmberLightcone(store)                                      # default path (no injected reach fn)
    for p in ("sage", "lumen", "nobody"):
        expected = set(reachable_collections(store, p)) | {p}          # the real grant light-cone + own address
        assert lc.reaches(p) == expected
    assert lc.reaches("sage") == {"op.retrieve", "op.index", "sage"}   # exactly the granted caps + self
    assert lc.reaches("nobody") == {"nobody"}                          # no grant → only its own address
    # a session ground join is an overlay ON TOP of the real light-cone, not a store write.
    lc.join("sage", "ground")
    assert lc.reaches("sage") == {"op.retrieve", "op.index", "sage", "ground"}
    assert set(reachable_collections(store, "sage")) == {"op.retrieve", "op.index"}  # store unchanged


# ── Discipline (5): ember.runtime.reach imports no chorus, iris, lumen or sage ─────────────────────────
def test_reach_module_imports_nothing_from_chorus_iris_lumen_or_sage():
    src = inspect.getsource(er)
    for banned in ("import iris", "from iris", "import chorus", "from chorus",
                   "import lumen", "from lumen", "import sage", "from sage"):
        assert banned not in src, "ember/reach.py must stay runner-side — found %r" % banned


def test_importing_ember_reach_pulls_in_no_chorus_module():
    # after a full exercise of ember.runtime.reach (all paths, including the lazy access and mantle
    # derivations), no persona or platform module is present — the AGPL boundary, measured at runtime.
    store = _store_granting({"sage": {"op.retrieve"}})
    er.reach(store, "lumen", {"q": 1}, to="op.retrieve", root_secret=b"root",
             fabric=LoopbackFabric())
    for name in list(sys.modules):
        top = name.split(".")[0]
        assert top not in ("iris", "chorus", "lumen", "sage"), "ember.reach dragged in %r" % name
    # …and it rides the reach core from prism (comms, propagation and the wire are one thing, and the
    # Apache foundation), not iris.
    assert "prism.reach" in sys.modules, "ember.reach must ride the prism.reach core"
