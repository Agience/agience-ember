"""Ember breeding — the §12.6 truth-preserving guardrails, pinned.

Pure: no cgroup and no store I/O. Heredity, variation and selection are each exercised against
in-memory crystals and `LiveEmber` records, so what is measured is the decision rule rather than
the environment it reads.
"""
import pytest

from ember.runtime import breeding
from crystal import crystal_model
from prism import demurrage


def _crystal(name="c.x", org="op.a", author="p1"):
    c = {"name": name, "facets": [{"name": "f", "direction": "both"}],
         "tektons": [{"name": "t", "domain": "d"}],
         "organons": [{"name": org, "requires": []}], "created_by": author}
    return crystal_model.crystal_artifact(c)


# ── Heredity: content addressing detects corruption ──────────────────────────────────────────────
def test_inherit_verifies_every_crystal():
    kids = breeding.inherit([_crystal("c.a"), _crystal("c.b")], {"collections": ["seed"]})
    assert {b["name"] for b in kids["crystals"]} == {"c.a", "c.b"}
    assert kids["lattice_seed"] == {"collections": ["seed"]}


def test_inherit_refuses_corrupted_crystal():
    import json
    art = _crystal("c.tampered")
    body = json.loads(art["content"]); body["organons"].append({"name": "op.injected"})
    art["content"] = json.dumps(body, sort_keys=True)          # content and sha now disagree
    with pytest.raises(breeding.BreedRefused, match="corruption cannot inherit"):
        breeding.inherit([art])


# ── Variation: a mutation yields a new signed crystal ─────────────────────────────────────────────
def test_variation_produces_a_new_signed_attributed_crystal():
    base = {"name": "c.base", "facets": [{"name": "f", "direction": "both"}],
            "tektons": [{"name": "t", "domain": "d"}],
            "organons": [{"name": "op.a", "requires": []}], "created_by": "p1"}
    variant_src = dict(base); variant_src["organons"] = [{"name": "op.a"}, {"name": "op.b"}]
    v = breeding.vary_structure(base, variant_src, author="p2")
    assert v["created_by"] == "p2" and v["varies_from"] == "c.base"
    assert v["sha256"] == crystal_model.crystal_sha(v)
    assert crystal_model.crystal_sha(v) != crystal_model.crystal_sha(base)   # a different crystal


def test_variation_refuses_identical_and_unattributed():
    base = {"name": "c.base", "facets": [{"name": "f", "direction": "both"}],
            "tektons": [{"name": "t", "domain": "d"}],
            "organons": [{"name": "op.a", "requires": []}], "created_by": "p1"}
    with pytest.raises(breeding.BreedRefused, match="byte-identical"):
        breeding.vary_structure(base, dict(base), author="p2")          # an identical structure
    with pytest.raises(breeding.BreedRefused, match="attributed"):
        breeding.vary_structure(base, {**base, "organons": [{"name": "op.z"}]}, author="")


# ── Selection: the demurrage↔cgroup seam ────────────────────────────────────────────────────────
def _ember(eid, heat, cost=1.0, mass=1.0, verified=1, now_frame=0):
    # demurrage.energy reads heat/frame from the artifact's context dict (demurrage._read_ctx)
    art = {"context": {"energy_heat": heat, "energy_frame": now_frame}}
    return breeding.LiveEmber(eid, art, maintenance_cost=cost, mass=mass, verified_since_breed=verified)


def test_starved_ember_dies_not_breeds():
    # energy below floor → die (divergence starves; guardrail 2)
    pop = [_ember("starved", heat=0.01, cost=1.0)]
    v = breeding.decide(pop, now=0, headroom={"disk_free_frac": 1.0})[0]
    assert v.action == "die" and "starved" in v.reason


def test_surplus_ember_breeds_when_all_gates_pass():
    pop = [_ember("rich", heat=10.0, cost=1.0, mass=2.0, verified=3)]
    v = breeding.decide(pop, now=0, headroom={"disk_free_frac": 1.0})[0]
    assert v.action == "breed"


def test_breed_gated_on_mass_verify_and_headroom():
    now = 0
    # no earned mass → sustain (guardrail 3)
    assert breeding.decide([_ember("nomass", 10.0, mass=0.0)], now,
                           headroom={"disk_free_frac": 1.0})[0].action == "sustain"
    # verify-throttle: no verified work since last breed → sustain (guardrail 4)
    assert breeding.decide([_ember("nowork", 10.0, verified=0)], now,
                           headroom={"disk_free_frac": 1.0})[0].action == "sustain"
    # no resource headroom → sustain (the cgroup side of the seam)
    assert breeding.decide([_ember("rich", 10.0, mass=2.0, verified=3)], now,
                           headroom={"disk_free_frac": 0.0})[0].action == "sustain"


# ── The gate: acting is opt-in through EMBER_BREEDING ────────────────────────────────────────────
def test_breeding_disabled_by_default_is_a_dry_run(monkeypatch):
    monkeypatch.delenv("EMBER_BREEDING", raising=False)
    assert breeding.breeding_enabled() is False
    spawned, retired = [], []
    verdicts = breeding.run_breeding(
        [_ember("rich", 10.0, mass=2.0, verified=3), _ember("starved", 0.01)], now=0,
        parent_crystals_of=lambda i: [_crystal()], seed_of=lambda i: {},
        spawn=lambda cid, inh: spawned.append(cid), retire=lambda i: retired.append(i),
        headroom={"disk_free_frac": 1.0})
    assert [v.action for v in verdicts] == ["breed", "die"]     # decisions still reported
    assert spawned == [] and retired == []                      # and the gate held the actions


def test_breeding_enabled_spawns_verified_child_and_retires_dead(monkeypatch):
    monkeypatch.setenv("EMBER_BREEDING", "1")
    assert breeding.breeding_enabled() is True
    spawned, retired = [], []
    breeding.run_breeding(
        [_ember("rich", 10.0, mass=2.0, verified=3), _ember("starved", 0.01)], now=0,
        parent_crystals_of=lambda i: [_crystal("c.h1"), _crystal("c.h2")],
        seed_of=lambda i: {"collections": ["s"]},
        spawn=lambda cid, inh: spawned.append((cid, [c["name"] for c in inh["crystals"]])),
        retire=lambda i: retired.append(i), headroom={"disk_free_frac": 1.0})
    assert spawned == [("rich:child", ["c.h1", "c.h2"])]        # verified heredity handed to spawn
    assert retired == ["starved"]
