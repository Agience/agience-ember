"""Compactification: the derived diagram, the colimit that carries everything, and the smoothing.

The roster is adversarial by design. Every positive is paired with a negative control that stays
silent, because a merge test that cannot fail measures nothing.

Every fixture is a fresh `open_lattice` under `tmp_path`, so nothing here reads the live store.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")
try:
    from mantle.db import open_lattice
except Exception:                                            # pragma: no cover - env guard
    open_lattice = None

pytestmark = pytest.mark.skipif(open_lattice is None, reason="lattice store not importable")

from ember.consolidate import colimit as C            # noqa: E402
from ember.consolidate import diagram as D            # noqa: E402


# ── the synthetic corpus ─────────────────────────────────────────────────────────────────────────
def _mk(store, aid, **kw):
    doc = {"id": aid, "content_type": "text/x-wordnet", "state": "active"}
    doc.update(kw)
    store.artifacts.put_many([doc], batch=1)
    return aid


def _anchor(store, lemma, concept):
    """`lemma:<form> --lex:en--> <concept>` — the shared vertex, as a real edge in the lattice."""
    store.artifacts.put_many([{"id": lemma, "content_type": "text/x-lemma", "state": "active"}],
                             batch=1)
    store.graph.add_edges([(lemma, concept, D.ANCHOR_LABEL, {})], batch=8)


@pytest.fixture()
def store(tmp_path):
    return open_lattice(str(tmp_path / "colimit.db"), origin="test-node")


@pytest.fixture()
def twins(store):
    """The Einstein shape: one concept recorded twice by two sources, joined only by shared lemmas.

    `pwn-e` and `oewn-e` share `lemma:einstein`; each points at its own universe's physicist node,
    and those two nodes are themselves joined only by `lemma:physicist`. No edge relates the two
    records, which is the shape the live corpus holds."""
    _mk(store, "pwn-e", gloss="physicist", provenance="observed", mass=2.0, lemmas=["einstein"])
    _mk(store, "oewn-e", gloss="physicist", provenance="hypothesis", mass=3.0, lemmas=["einstein"])
    _mk(store, "pwn-p", lemmas=["physicist"])
    _mk(store, "oewn-p", lemmas=["physicist"])
    _anchor(store, "lemma:einstein", "pwn-e")
    _anchor(store, "lemma:einstein", "oewn-e")
    _anchor(store, "lemma:physicist", "pwn-p")
    _anchor(store, "lemma:physicist", "oewn-p")
    store.graph.add_edges([("pwn-e", "pwn-p", "instance_of", {}),
                           ("oewn-e", "oewn-p", "instance_of", {})], batch=8)
    return store


ARM = {"label_keyed": False, "include_unanchored": False}


# ── Step 1: the diagram is derived ───────────────────────────────────────────────────────────────
def test_diagram_is_derived_from_the_concept_alone(twins):
    """Nothing is passed in but the concept id, and the derivation finds both records. A diagram of
    `{pwn-e}` alone would mean the derivation cannot reach what a string rule reaches."""
    got = D.derive_diagram(twins, "pwn-e", **ARM)
    assert set(got["members"]) == {"pwn-e", "oewn-e"}, got
    assert got["candidates"] >= 1
    assert got["unmeasured"] == 0


def test_the_residual_is_zero_not_merely_small(twins):
    """The residual is floating-point zero, orders of magnitude under the band `PathLedger` derives.
    A residual of 1e-3 against a tolerance of 1e-2 would be a threshold in disguise: a fitted
    constant wearing a certificate."""
    r = D.separation(twins, "pwn-e", "oewn-e", **ARM)
    assert r["one_object"] is True
    assert r["residual_fraction"] <= r["b_into_a"]["tolerance"], r
    # The margin is the point: a real cutoff would have the two within an order of magnitude.
    assert r["residual_fraction"] < 1e-6 * r["b_into_a"]["tolerance"] or r["residual_fraction"] == 0.0


def test_negative_control_bank_slope_vs_bank_institution_must_not_merge(store):
    """The negative control. Both senses anchor on the same `lemma:bank` vertex, so the same reach
    that finds the twins offers this pair to the measurement, and the measurement separates them.

    A `one_object` of True here would mean the read is finding lemma overlap rather than separation,
    and every polysemous word in the corpus would collapse into one blob. This is what makes the
    positive results mean anything."""
    _mk(store, "bank-slope", lemmas=["bank"])
    _mk(store, "bank-inst", lemmas=["bank"])
    _mk(store, "slope", lemmas=["slope"])
    _mk(store, "institution", lemmas=["institution"])
    _anchor(store, "lemma:bank", "bank-slope")
    _anchor(store, "lemma:bank", "bank-inst")
    _anchor(store, "lemma:slope", "slope")
    _anchor(store, "lemma:institution", "institution")
    store.graph.add_edges([("bank-slope", "slope", "hypernym", {}),
                           ("bank-inst", "institution", "hypernym", {})], batch=8)

    r = D.separation(store, "bank-slope", "bank-inst", **ARM)
    assert r["measured"] is True, r
    assert r["one_object"] is False, r
    assert r["residual_fraction"] > 0.0
    # And the reach did offer it — the separation is a measurement, not a missing candidate.
    assert "bank-inst" in set(D.candidates(store, "bank-slope"))
    assert D.derive_diagram(store, "bank-slope", **ARM)["members"] == ["bank-slope"]


def test_no_evidence_is_not_agreement(store):
    """Two artifacts nothing was observed about report `measured=False` and `one_object=False`, and
    those are distinguishable states. Their residual is trivially zero, so treating zero as agreement
    would collapse the whole unobserved tail of the corpus into one object."""
    _mk(store, "ghost-a")
    _mk(store, "ghost-b")
    r = D.separation(store, "ghost-a", "ghost-b", **ARM)
    assert r["one_object"] is False
    assert r["measured"] is False
    assert "unmeasured" in (r.get("why") or "")


def test_the_shared_anchor_is_the_reach_not_the_evidence(store):
    """The shared anchor is why a pair is offered, and it is not itself evidence about the pair.

    `candidates()` offers a pair because they share an anchor. Feeding that same anchor in as an
    evidence row would have every candidate begin the measurement already agreeing on its only
    discriminating row — the reach deciding the verdict it exists to offer. On the live store that
    matters most for satellite adjectives, which carry no relation edges, so the anchor row would be
    their whole frame: `wn-oewn-88455131-a` ("causing indignation due to hypocrisy") would read as
    one object with `wn-rich.s.01`, `.s.04` and `.s.05`, three unrelated senses of `rich`.

    Two artifacts whose only commonality is the lemma they are both called by therefore report
    `measured=False`: there is no evidence about them beyond their name, and a name is not an
    identity."""
    _mk(store, "rich-a", gloss="of great worth")
    _mk(store, "rich-b", gloss="causing indignation")
    _anchor(store, "lemma:rich", "rich-a")
    _anchor(store, "lemma:rich", "rich-b")
    assert "rich-b" in set(D.candidates(store, "rich-a"))       # the reach does offer the pair
    r = D.separation(store, "rich-a", "rich-b", **ARM)
    assert r["one_object"] is False, r
    assert r["measured"] is False, r
    assert D.derive_diagram(store, "rich-a", **ARM)["members"] == ["rich-a"]


def test_subsumption_is_not_identity(store):
    """Identity is measured in both directions. `dog`'s evidence lies inside `animal`'s band, so
    `residual(dog | animal) == 0`, and a one-directional read would merge a class with its
    instance."""
    _mk(store, "animal", lemmas=["animal"])
    _mk(store, "dog", lemmas=["dog"])
    _mk(store, "thing", lemmas=["thing"])
    _anchor(store, "lemma:animal", "animal")
    _anchor(store, "lemma:animal", "dog")          # shared anchor so the reach offers the pair
    _anchor(store, "lemma:thing", "thing")
    _mk(store, "fur", lemmas=["fur"])
    _anchor(store, "lemma:fur", "fur")
    store.graph.add_edges([("dog", "thing", "hypernym", {}),
                           ("animal", "thing", "hypernym", {}),
                           ("animal", "fur", "has_part", {})], batch=8)
    # Both argument orders, and that is not redundancy. With one ordering the result turns on which
    # side happens to be the wider band, and a mutation dropping the `a_into_b` conjunct from
    # `one_object` survives. Subsumption is asymmetric, so it is asserted from both sides.
    for a, b in (("dog", "animal"), ("animal", "dog")):
        r = D.separation(store, a, b, **ARM)
        assert r["measured"] is True, (a, b, r)
        assert r["one_object"] is False, (a, b, r)
        assert r["subsumed"] is True, (a, b, r)   # exactly one direction absorbed everything


def test_a_prefix_of_the_evidence_never_decides(twins, monkeypatch):
    """With the envelope unreadable, `_all_edges` reports a non-exhaustive read and nothing merges.
    Merging on the prefix that fit would be a check that cannot fail, because the unread tail is
    exactly where a separating fact would be."""
    monkeypatch.setattr(D, "instrument_rows", lambda _b: 0)
    r = D.separation(twins, "pwn-e", "oewn-e", **ARM)
    assert r["one_object"] is False
    assert r["measured"] is False
    assert "instrument" in (r.get("why") or "")


def test_label_keying_isolates_the_converse_edge(twins):
    """The two keyings ask two questions, and this pins the difference between their answers.

    Give OEWN the converse of the relation PWN recorded once. Label-keyed, that extra label is a
    direction PWN's band cannot absorb and the pair separates; label-free, the same fact written
    twice is the same fact. Were both arms to agree, `label_keyed` would be a dead knob and the
    report's claim about `instance_hyponym` unfounded."""
    twins.graph.add_edges([("oewn-p", "oewn-e", "instance_hyponym", {})], batch=8)
    keyed = D.separation(twins, "pwn-e", "oewn-e", label_keyed=True, include_unanchored=False)
    free = D.separation(twins, "pwn-e", "oewn-e", label_keyed=False, include_unanchored=False)
    assert keyed["one_object"] is False, keyed
    assert free["one_object"] is True, free


# ── Step 2: the colimit carries everything ───────────────────────────────────────────────────────
def test_colimit_carries_provenance_position_mass_and_a_morphism_from_each_member(twins):
    """The colimit carries every member's provenance, the summed mass with its count, the evidence
    union as (count, sum), and a morphism back to each member. Each clause below is one thing a
    pointer stub — the shape `genesis.consolidate_colimit` mints — does not carry."""
    built = C.colimit(twins, ["pwn-e", "oewn-e"], **ARM)
    doc = built["doc"]
    assert built["acceptable"] is True, built["certificate"]

    # every member's provenance, kept whole
    assert doc["member_provenance"] == {"pwn-e": "observed", "oewn-e": "hypothesis"}
    # ...and the rung is the minimum: merging an observation with a hypothesis is not an observation
    assert doc["provenance"] == "hypothesis"
    # summed mass with the count beside it -- fewer objects, more mass each
    assert doc["mass"] == pytest.approx(5.0) and doc["mass_count"] == 2
    # the evidence union as (count, sum), order-free
    assert doc["evidence"]["lemma:physicist"] == pytest.approx(2.0)
    assert doc["evidence_count"] == len(doc["evidence"])
    # a morphism from the colimit to every member -- members remain, reconstructible
    assert {m[1] for m in built["morphisms"]} == {"pwn-e", "oewn-e"}
    assert all(m[2] == C.COLIMIT_LABEL for m in built["morphisms"])
    assert doc["colimit_of"] == ["oewn-e", "pwn-e"]


def test_colimit_id_is_a_function_of_the_diagram_not_the_clock(twins):
    """Same members give the same id in any order; different members give a different id. An id
    built from a clock — as `genesis.consolidate_colimit` builds its own, with `_now_hash` — mints
    two artifacts for the same colimit taken twice."""
    a = C.colimit(twins, ["pwn-e", "oewn-e"], **ARM)["id"]
    b = C.colimit(twins, ["oewn-e", "pwn-e"], **ARM)["id"]
    assert a == b
    assert a != C._colimit_id(["pwn-e", "oewn-e", "pwn-p"])


def test_position_recovers_every_member(twins):
    """The merged position returns each member's full energy under projection. Cross-source members
    are orthogonal in the JC coordinate (cos = 0.000000 exactly), so the sum recovers every member;
    anything else is a blur that is none of them."""
    v1 = np.array([3.0, 0.0, 0.0])
    v2 = np.array([0.0, 4.0, 0.0])
    got = C.recover_from_position(v1 + v2, [v1, v2])
    assert got["members_checked"] == 2
    assert got["worst_relative_error"] == pytest.approx(0.0, abs=1e-12)


def test_mean_position_loses_half_of_each_member(twins):
    """The negative control on the position: if the mean also passed the check above, that check
    would prove nothing. The mean of two orthogonal members returns half each member's energy, a
    relative error of exactly 0.5, which is the energy a colimit carries whole."""
    v1 = np.array([3.0, 0.0, 0.0])
    v2 = np.array([0.0, 4.0, 0.0])
    got = C.recover_from_position((v1 + v2) / 2.0, [v1, v2])
    assert got["worst_relative_error"] == pytest.approx(0.5)


def test_conservation_certificate_can_fail(twins):
    """The acceptance test can fail, which is what makes it an acceptance test: a certificate that
    passes for every input certifies nothing. Certifying the members' stacked evidence against a band
    built from one member only leaves the other member's independent direction unabsorbed, and the
    ledger reports the leak as un-emitted."""
    from prism.conservation import PathLedger
    from ember.optics import absorb_transmit
    ea = D.evidence(twins, "pwn-e", **ARM)
    eb = D.evidence(twins, "oewn-e", **ARM)
    twins.graph.add_edges([("oewn-e", "oewn-p", "extra_rel", {})], batch=8)
    eb = D.evidence(twins, "oewn-e", label_keyed=True, include_unanchored=False)
    ea = D.evidence(twins, "pwn-e", label_keyed=True, include_unanchored=False)
    keys = sorted(set(ea.weights) | set(eb.weights))
    W = np.vstack([ea.matrix(keys), eb.matrix(keys)])
    narrow = D.band(ea.matrix(keys))                 # a band that spans only one member
    absorbed, residual, k = absorb_transmit(W, basis=narrow)
    led = PathLedger(W)
    led.absorb(absorbed, residual)                   # deliberately not emit()ed
    cert = led.certificate()
    assert cert["balanced"] is False, cert
    assert cert["unaccounted"] > cert["tolerance"]
    assert "never emitted" in (cert["why"] or "") or "still travelling" in (cert["why"] or "")

    # ...and the full-union band, which is the colimit, absorbs everything.
    full = C.certify_merge(twins, ["pwn-e", "oewn-e"], label_keyed=True, include_unanchored=False)
    assert full["balanced"] is True, full


def test_apply_refuses_an_unbalanced_certificate(twins):
    """An unbalanced certificate stops the write: `apply_colimit` reports `applied: False` and the
    store holds nothing. Reporting the imbalance and writing anyway would make the check
    decoration."""
    built = C.colimit(twins, ["pwn-e", "oewn-e"], **ARM)
    built["acceptable"] = False
    got = C.apply_colimit(twins, built)
    assert got["applied"] is False
    assert twins.artifacts.get_artifact(built["id"]) is None


# ── Step 3: smoothing the morphisms ──────────────────────────────────────────────────────────────
def test_edge_key_idempotency_collapses_parallel_facts_for_free(twins):
    """Deduplication comes out of edge-key idempotency, at no extra cost. `a --instance_of--> b` and
    `a' --instance_of--> b'`, with {a,a'} and {b,b'} each merged, are one fact: rewritten onto the
    colimits they become the same `(src, dst, label)` triple, hash to the same `edge_key`, and the
    store's `ON CONFLICT DO UPDATE` lands one row.

    Asserted against the store rather than the docstring, because if two rows survived a separate
    dedup pass would be owed."""
    ce = C._colimit_id(["pwn-e", "oewn-e"])
    cp = C._colimit_id(["pwn-p", "oewn-p"])
    mapping = {"pwn-e": ce, "oewn-e": ce, "pwn-p": cp, "oewn-p": cp}
    sm = C.smooth_edges(twins, mapping)
    triples = {(s, d, l) for s, d, l, _ in sm["edges"]}
    assert (ce, cp, "instance_of") in triples
    # two source facts in, one distinct fact out
    assert sm["collapsed"] >= 1, sm
    assert len([t for t in triples if t[2] == "instance_of"]) == 1

    twins.graph.add_edges(sm["edges"], batch=100)
    landed = [e for e in twins.graph.edges_of(ce, direction="out") if e["label"] == "instance_of"]
    assert len(landed) == 1, landed


def test_smoothing_drops_and_counts_self_edges(twins):
    """A converse edge between two members of one diagram would rewrite to a colimit->itself edge
    asserting nothing. Such edges are dropped and counted, so the drop is on the report."""
    twins.graph.add_edges([("oewn-e", "pwn-e", "same_as", {})], batch=8)
    ce = C._colimit_id(["pwn-e", "oewn-e"])
    sm = C.smooth_edges(twins, {"pwn-e": ce, "oewn-e": ce})
    assert sm["self_edges"] >= 1, sm
    assert all(s != d for s, d, _l, _p in sm["edges"])


def test_apply_reports_what_it_actually_wrote(twins):
    """`apply` reports what it wrote, not what it offered. `add_edges` returns the number handled,
    and the shortfall between offered and handled is on the report where a caller can see it
    ([[mesh-cursor-must-not-skip]])."""
    built = C.colimit(twins, ["pwn-e", "oewn-e"], **ARM)
    got = C.apply_colimit(twins, built)
    assert got["applied"] is True
    assert got["smoothed"]["shortfall"] == 0, got
    # the colimit and its morphisms are really in the store
    assert twins.artifacts.get_artifact(built["id"]) is not None
    members = {e["dst"] for e in twins.graph.edges_of(built["id"], label=C.COLIMIT_LABEL)}
    assert members == {"pwn-e", "oewn-e"}
