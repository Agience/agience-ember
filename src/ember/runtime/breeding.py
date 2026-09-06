"""Ember breeding — heredity, variation, selection (OPERATOR-ARCHITECTURE §12.6).

When an ember breeds it passes on itself, and it may mutate — but this is evolution without the
dangerous ingredient (random corruption), because the substrate forbids it. This module is the
truth-preserving core, built as pure functions so every guardrail is testable in isolation:

  heredity   — content-addressed crystals (exact; a copy is byte-identical or it is a different
               crystal) + a seed-lattice slice (inherited mass). `inherit()` verifies every crystal
               sha; corruption cannot pass.
  variation  — structural: a mutation is a new, signed crystal, not an in-place edit of the parent
               (`vary_structure` returns a new crystal object rather than mutating the base's).
               experiential: the child's shard grows from its own prism at runtime — nothing to
               copy at breed time.
  selection  — the energy ledger decides: sustain / breed / die. `decide()` reads the demurrage
               heat (energy) and the cgroup/disk headroom (resource) — this is the demurrage↔cgroup
               seam, wired. Fitness is measured, not set by a config knob.

Guardrails, all enforced here and cross-checked by the four canon properties (§12.6):
  1. no silent corruption   — `inherit` re-verifies each crystal sha (content-addressing).
  2. no runaway             — `decide` starves a diverging ember (energy below floor) rather than
                              letting it breed; the meter's convergence is the survival test.
  3. influence needs mass   — `decide` gates breeding on accrued mass/agreement (`e.mass > 0.0`),
                              not on self-assertion.
  4. cannot mutate faster than verify — breed requires a verify-throttle token (real verified work
                              since the last breed), so variation never outruns the truth.

Breeding is off by default, gated by `EMBER_BREEDING=1`. The single-ember training run never
breeds; nothing here runs in the kickoff path. The actual child-spawn (a new OS process / node) is
an injected callback — infrastructure-specific, out of scope for this module, which owns only the
decision and the truth-preserving inheritance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from crystal import crystal_model
from prism import demurrage

# ── the three measured thresholds — derived from the ledger + resource, not arbitrary caps.
# Each is a ratio against a live measurement, so the bound scales with the ember rather than
# sitting at a fixed number. ([[no-arbitrary-caps]]: the seam is the ratio; the default below is a
# starting point, adjustable by measurement.)
MAINTENANCE_FLOOR = float(os.getenv("EMBER_BREED_FLOOR", "1.0"))     # energy < floor·cost → die
BREED_SURPLUS = float(os.getenv("EMBER_BREED_SURPLUS", "2.0"))      # energy ≥ surplus·cost → may breed
RESOURCE_HEADROOM = float(os.getenv("EMBER_BREED_HEADROOM", "0.25"))  # need ≥25% free to spawn a child


def breeding_enabled() -> bool:
    """Breeding is a canon property, gated off by default. The single-ember training run and every
    existing deployment are unaffected unless `EMBER_BREEDING` is set."""
    return (os.getenv("EMBER_BREEDING") or "").strip().lower() in ("1", "true", "on", "yes")


class BreedRefused(RuntimeError):
    """An unmet breed/heredity guardrail. Raised rather than logged and continued, so the unmet
    guardrail is the finding."""


# ── heredity ────────────────────────────────────────────────────────────────────────────────────
def inherit(parent_crystals: List[Dict[str, Any]], seed_lattice: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
    """The child's inherited structure. Content-addressed crystals are copied by verifying each: a
    crystal artifact whose content no longer hashes to its claimed sha does not inherit (corruption
    cannot inherit; §12.6 guardrail 1). Returns {crystals, lattice_seed}: exact heredity + the
    inherited-mass slice. The child grows its own shard from here; state is never copied back."""
    verified: List[Dict[str, Any]] = []
    for art in parent_crystals:
        try:
            body = crystal_model.verify(art)                 # re-hash; raises on mismatch
        except Exception as exc:
            raise BreedRefused("heredity refused: a parent crystal failed sha verification "
                               "(corruption cannot inherit) — %s" % exc)
        verified.append(body)
    return {"crystals": verified, "lattice_seed": dict(seed_lattice or {})}


# ── variation ───────────────────────────────────────────────────────────────────────────────────
def vary_structure(base_crystal: Dict[str, Any], authored_variant: Dict[str, Any],
                   *, author: str) -> Dict[str, Any]:
    """Structural mutation. A variant is a new crystal — content-addressed, re-signed, provenance
    to the base — never an in-place edit of the parent's crystal (§12.6 guardrail: mutation is by
    authorship, never by corruption). Raises if the 'variant' hashes identical to the base (that
    is not a mutation) or carries no author (unattributed mutation cannot ground — the Higgs rule)."""
    if not author:
        raise BreedRefused("structural variation refused: a mutation must be attributed (Higgs rule)")
    # Compare structure (facets/tektons/organons/seed), ignoring provenance tags — a variant that
    # differs only by a varies_from/created_by label is not a mutation, it is a relabel.
    def _structure(c: Dict[str, Any]) -> str:
        core = {k: c.get(k) for k in ("facets", "tektons", "organons", "lattice_seed")}
        return crystal_model.crystal_sha(core)
    if _structure(authored_variant) == _structure(base_crystal):
        raise BreedRefused("structural variation refused: variant is byte-identical to base in "
                           "structure — a mutation is a DIFFERENT crystal or it is not a mutation")
    variant = dict(authored_variant)
    variant["created_by"] = author
    variant.setdefault("varies_from", base_crystal.get("name"))
    problems = crystal_model.validate(variant)
    if problems:
        raise BreedRefused("structural variation refused: invalid variant crystal: %s"
                           % "; ".join(problems))
    variant["sha256"] = crystal_model.crystal_sha(variant)
    return variant                                            # a new signed crystal; the old one stands


# ── selection — the demurrage↔cgroup seam, wired ──────────────────────────────────────────────────
@dataclass
class LiveEmber:
    """A member of the population, as the selector sees it: its energy-bearing artifact (read via
    demurrage), its maintenance cost, and its accrued mass (agreement). Pure data; no I/O."""
    ember_id: str
    energy_artifact: Dict[str, Any]         # the doc demurrage.energy() reads (heat/frame/rest_mass)
    maintenance_cost: float                 # energy/frame this ember must earn to hold its mass
    mass: float = 0.0                       # accrued agreement (existence in degrees, §13.5)
    verified_since_breed: int = 0           # real verified work since last breed (the throttle)


@dataclass
class Verdict:
    ember_id: str
    action: str                             # "sustain" | "breed" | "die"
    energy: float
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)


def _resource_headroom(data_path: str) -> Dict[str, float]:
    """The cgroup/disk side of the seam. Fraction free of memory and disk, measured, so the bound
    scales with the actual box (containers-read-the-cgroup). Import is local: resource reads
    /sys/fs/cgroup and /proc, which are absent in unit tests, so callers inject via `headroom=`."""
    from prism import envelope as resource
    disk_free = resource.disk_free_bytes(data_path)
    disk_total = resource.disk_total_bytes(data_path)
    # Free disk and the memory ceiling are not comparable as a fraction: on any box whose data
    # volume exceeds its RAM, that ratio runs well past 1.0, so the two are compared in the same
    # units instead — disk free over disk total.
    # A ceiling that cannot be measured reports no fraction rather than standing in for one: the
    # selector that decides whether embers reproduce reads a real fraction or none at all, so
    # `decide` breeds only when a fraction is actually reported.
    if disk_free is None or not disk_total:
        return {"disk_free_bytes": disk_free}
    return {"disk_free_frac": float(disk_free) / float(disk_total),
            "disk_free_bytes": float(disk_free)}


def decide(population: List[LiveEmber], now: int, *,
           headroom: Optional[Dict[str, float]] = None,
           min_verified_to_breed: int = 1) -> List[Verdict]:
    """The selector: sustain / breed / die, from the energy ledger and resource headroom — the
    demurrage↔cgroup seam. Pure and deterministic given (population, now, headroom).

      die     — energy < MAINTENANCE_FLOOR·cost: the ember is starved; it cannot hold its mass.
                (§12.6 guardrail 2: a diverging ember earns no deposits, cools below floor, and is
                selected out — divergence starves, it does not breed.)
      breed   — energy ≥ BREED_SURPLUS·cost and mass > 0 (earned agreement, guardrail 3) and
                verified_since_breed ≥ min_verified_to_breed (guardrail 4: cannot mutate faster than
                verify) and resource headroom ≥ RESOURCE_HEADROOM.
      sustain — otherwise: alive, holding, not yet breeding.

    Fitness is measured end to end: demurrage.energy() is the read, the cgroup is the ceiling; no
    knob decides who lives."""
    # Absence is not surplus. A caller that supplies no headroom information gets none — not the
    # appearance of perfect headroom — so `have_headroom` below stays False until a real fraction
    # is reported.
    hr = headroom if headroom is not None else {}
    _frac = hr.get("disk_free_frac")
    have_headroom = (_frac is not None) and (float(_frac) >= RESOURCE_HEADROOM)
    out: List[Verdict] = []
    for e in population:
        energy = demurrage.energy(e.energy_artifact, now)
        cost = max(1e-9, e.maintenance_cost)
        if energy < MAINTENANCE_FLOOR * cost:
            out.append(Verdict(e.ember_id, "die", energy,
                               "starved: energy %.4g < floor %.4g" % (energy, MAINTENANCE_FLOOR * cost)))
            continue
        can_breed = (energy >= BREED_SURPLUS * cost and e.mass > 0.0
                     and e.verified_since_breed >= min_verified_to_breed and have_headroom)
        if can_breed:
            out.append(Verdict(e.ember_id, "breed", energy,
                               "surplus %.4g ≥ %.4g·cost, mass %.4g, verified %d, headroom ok"
                               % (energy, BREED_SURPLUS, e.mass, e.verified_since_breed),
                               {"disk_free_frac": hr.get("disk_free_frac")}))
        else:
            why = []
            if energy < BREED_SURPLUS * cost: why.append("energy<surplus")
            if e.mass <= 0.0: why.append("no earned mass")
            if e.verified_since_breed < min_verified_to_breed: why.append("verify-throttle")
            if not have_headroom: why.append("no resource headroom")
            out.append(Verdict(e.ember_id, "sustain", energy, "alive; not breeding (%s)"
                               % ", ".join(why)))
    return out


# ── the gated orchestration hook — the actual spawn is injected, never owned here ────────────────
def run_breeding(population: List[LiveEmber], now: int, *,
                 parent_crystals_of: Callable[[str], List[Dict[str, Any]]],
                 seed_of: Callable[[str], Dict[str, Any]],
                 spawn: Callable[[str, Dict[str, Any]], Any],
                 retire: Callable[[str], Any],
                 headroom: Optional[Dict[str, float]] = None) -> List[Verdict]:
    """Apply the verdicts, if and only if breeding is enabled. For each `breed`: build the child's
    inheritance (verified heredity + seed) and hand it to the injected `spawn(child_id, inheritance)`
    — this module never creates a process/node itself. For each `die`: `retire(ember_id)`. `sustain`
    is a no-op. Returns the verdicts either way (so a disabled call is a dry-run report)."""
    verdicts = decide(population, now, headroom=headroom)
    if not breeding_enabled():
        return verdicts                                       # dry-run: the honest off state
    for v in verdicts:
        if v.action == "breed":
            inheritance = inherit(parent_crystals_of(v.ember_id), seed_of(v.ember_id))
            spawn(v.ember_id + ":child", inheritance)          # infra owns the actual ignition
        elif v.action == "die":
            retire(v.ember_id)
    return verdicts


__all__ = [
    "breeding_enabled", "BreedRefused", "inherit", "vary_structure",
    "LiveEmber", "Verdict", "decide", "run_breeding",
    "MAINTENANCE_FLOOR", "BREED_SURPLUS", "RESOURCE_HEADROOM",
]
