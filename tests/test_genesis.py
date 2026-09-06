"""GENESIS P0 — schema freeze, Stage-0 ingest, and ρ metrics.

The bootstrap/metrics logic is tested against a tiny in-memory fake store (deterministic,
no ArcadeDB). Store/nltk-backed paths are exercised by skip-guarded smoke tests.
"""
import os

import pytest

from ember import genesis as g

from _fakes import _FakeArtifacts, _FakeGraph, _FakeStore  # noqa: F401

# `_FakeArtifacts`, `_FakeGraph`, and `_FakeStore` live in `_fakes.py`, not in this module. A local
# redefinition of any of the three would shadow this import silently: two objects with one name, and
# the one a reader sees is not the one that runs.


# ── an in-memory stand-in for the leaf store (artifacts + labeled graph) ──────
# ── seed-table sanity (pure) ──────────────────────────────────────────────────
def test_seed_tables_wellformed():
    vids = [v[0] for v in g.SEED_VTYPES]
    eids = [e[0] for e in g.SEED_ETYPES]
    assert len(vids) == len(set(vids)) and len(eids) == len(set(eids))
    # every seed row is fully populated (no blank generators)
    assert all(all(str(x).strip() for x in row) for row in g.SEED_VTYPES)
    assert all(row[1].strip() and row[2].strip() for row in g.SEED_ETYPES)
    # the four GENESIS rungs map to real prism.mass band values
    gm = {r[0]: r[1] for r in g.SEED_RUNGS}
    assert gm == {"OBSERVED": "observed", "FETCHED": "span_cited",
                  "DERIVED": "hypothesis", "ASSERTED": "assertion"}


# ── bootstrap (fake store) ────────────────────────────────────────────────────
def test_bootstrap_mints_and_is_idempotent():
    s = _FakeStore()
    r1 = g.bootstrap(s)
    assert r1["vtypes"] == len(g.SEED_VTYPES)
    assert r1["etypes"] == len(g.SEED_ETYPES)
    assert r1["rungs"] == len(g.SEED_RUNGS)
    assert r1["collections"] == len(g.SEED_COLLECTIONS)
    assert r1["edges"] > 0
    assert g.is_bootstrapped(s)
    # +2: the cite.genesis provenance anchor AND the AUTHOR's person artifact, both minted
    # before the registries (a person IS an artifact — contract §2.1).
    total = len(g.SEED_VTYPES) + len(g.SEED_ETYPES) + len(g.SEED_RUNGS) + len(g.SEED_COLLECTIONS) + 2
    assert s.artifacts.count() == total
    # re-run: no new edges, no new artifacts (upsert + edge dedupe)
    r2 = g.bootstrap(s)
    assert r2["edges"] == 0
    assert s.artifacts.count() == total


# ── the WHO is an artifact (contract §2.1) ───────────────────────────────────
# A dangling WHO breaks grant propagation silently: nothing raises, authorization just stops flowing
# through a reference to a vertex that is not there. So these assert on whether the reference
# resolves — the property node-repair's `created_by resolves` check verifies — not on whether the
# string is present.


def test_every_created_by_resolves_to_an_existing_artifact():
    """node-repair's `created_by resolves` predicate, run in-process on a bootstrapped store."""
    s = _FakeStore()
    g.bootstrap(s)
    ids = {a["id"] for a in s.artifacts.list_artifacts()}
    dangling = sorted({a["created_by"] for a in s.artifacts.list_artifacts()
                       if a.get("created_by") and a["created_by"] not in ids})
    assert dangling == [], (
        "%d created_by value(s) reference a vertex that does not exist: %s — grant propagation "
        "is silently broken for every row citing them" % (len(dangling), dangling)
    )
    # and it is not vacuously true: rows DO carry a created_by.
    assert any(a.get("created_by") for a in s.artifacts.list_artifacts())


def test_the_author_person_artifact_is_minted_once_and_is_the_created_by_target():
    from mantle.services.principal import PERSON_CONTENT_TYPE, person_id

    # An explicit PERSON author, because that is what this test is about. `principal.py` mints a
    # person artifact for an email-shaped author and a foundation entity for anything else, so
    # leaving the author to the default — `ember-local` when `EMBER_PRINCIPAL` is unset — exercises
    # the foundation path and asserts nothing about person minting.
    author = "author@example.com"
    s = _FakeStore()
    g.bootstrap(s, author=author)
    pid = person_id(author)
    person = s.artifacts.get_artifact(pid)
    assert person is not None and person["content_type"] == PERSON_CONTENT_TYPE
    # The seed rows point AT it — the claim string itself never reaches created_by.
    assert s.artifacts.get_artifact(g.CITE_GENESIS)["created_by"] == pid
    assert not any(a.get("created_by") == "author@example.com"
                   for a in s.artifacts.list_artifacts())
    # A person is self-attesting at the root of its own chain, so it resolves alone.
    assert person["created_by"] == pid
    # Idempotent: a second bootstrap does not mint a second person.
    n = len([a for a in s.artifacts.list_artifacts()
             if a.get("content_type") == PERSON_CONTENT_TYPE])
    g.bootstrap(s)
    assert n == 1 and len([a for a in s.artifacts.list_artifacts()
                           if a.get("content_type") == PERSON_CONTENT_TYPE]) == 1


def test_the_person_id_is_the_shared_multi_issuer_derivation():
    """`uuid5(_USER_NS, f'{issuer}\\n{sub}')` — byte-identical to mantle/services/oidc.py:179.

    This is the property that makes `created_by` a legal COLUMN (mantle/db/schema.py): every
    observer derives the SAME id from the same issuer claim. A second, ember-local derivation
    would name one human with two vertex ids and every cross-observer join would under-match.
    """
    import uuid

    from mantle.services import principal as person

    assert person.person_id("author@example.com") == str(uuid.uuid5(
        person._USER_NS, "%s\n%s" % (person.LOCAL_ISSUER, "author@example.com")))
    # The issuer must be IN the hash: two IdPs can mint the same `sub`.
    assert person.person_id("s", issuer="https://a") != person.person_id("s", issuer="https://b")
    # Deterministic, not random.
    assert person.person_id("author@example.com") == person.person_id("author@example.com")


def test_a_person_vertex_carries_only_universal_fields():
    """§2.1 consequence 4 / §7.4: a shared vertex may not carry per-observer judgement.

    Trust weight, reputation and observation counts are LOCAL. Two observers holding
    different opinions about one person under one id is an unresolvable collision, so the
    opinion never goes on the shared record — it lives in private state.
    """
    from mantle.services.principal import person_artifact

    doc = person_artifact("author@example.com", public_key="ssh-ed25519 AAAA...")
    ctx = doc["context"]
    assert set(ctx) <= {"kind", "issuer", "sub", "name", "public_key"}, (
        "a person vertex grew a field: %s" % sorted(ctx)
    )
    for banned in ("trust", "trust_weight", "reputation", "observations",
                   "observation_count", "grants", "fitness", "mass"):
        assert banned not in ctx and banned not in doc


def test_processes_are_never_given_person_artifacts():
    """`ember-source` / `ember-local` author ~98% of the corpus and are not people.

    They resolve to a foundation entity — see `tests/test_foundation_entity.py` for what that is.
    A process author never wears `PERSON_CONTENT_TYPE`, and its id is never a person's id.
    `_author_ref` mints a foundation-entity artifact for a process author rather than passing the
    bare string through, so `created_by resolves` always cites a vertex that exists.
    """
    from mantle.services.principal import (PERSON_CONTENT_TYPE, FOUNDATION_CONTENT_TYPE,
                                           is_process_author, person_id, principal_artifact)

    s = _FakeStore()
    for proc in ("ember-source", "ember-local"):
        assert is_process_author(proc)
        # Resolves, and to something that is visibly NOT a person.
        assert person_id(proc) != person_id("author@example.com")
        doc = principal_artifact(proc)
        assert doc["content_type"] == FOUNDATION_CONTENT_TYPE != PERSON_CONTENT_TYPE

        ref = g._author_ref(s, proc)
        assert ref == person_id(proc) != proc, "a process author must RESOLVE, not pass through"
        minted = s.artifacts.get_artifact(ref)
        assert minted is not None, "created_by cites a vertex that must exist"
        assert minted["content_type"] == FOUNDATION_CONTENT_TYPE

    # One entity per PROCESS, not one for all — `ember-source` and `ember-local` wrote different
    # rows and collapsing them would destroy the only WHO those 200,000+ rows carry.
    assert s.artifacts.count() == 2
    assert g._author_ref(s, "ember-source") != g._author_ref(s, "ember-local")


def test_collection_dag_and_ontology_membership():
    s = _FakeStore()
    g.bootstrap(s)
    # every seed collection except the universe root is sub_collection_of universe
    kids = set(s.graph.neighbors(g.UNIVERSE, "sub_collection_of", direction="in"))
    assert "stage.0.lexicon" in kids and "ontology" in kids and "sources" in kids
    # type/rung artifacts are members of the ontology collection
    members = s.graph.neighbors(g.ONTOLOGY, "member_of", direction="in")
    assert len(members) == len(g.SEED_VTYPES) + len(g.SEED_ETYPES) + len(g.SEED_RUNGS)
    assert "vtype.Synset" in members and "etype.consolidates" in members


# ── ρ / metrics (fake store) ──────────────────────────────────────────────────
def test_rho_baseline_is_one_without_consolidation():
    s = _FakeStore()
    g.bootstrap(s)
    s.artifacts.put_artifact({"id": "wn-x", "content_type": "text/x-wordnet",
                              "state": "committed", "context": "meaning of x", "lemmas": ["x"],
                              "content": "x" * 100, "collection_id": "stage.0.lexicon"})
    m = g.collection_metrics(s, "stage.0.lexicon", consolidated=set())
    assert m["artifacts"] == 1 and m["rho"] == 1.0 and m["keyed_coverage"] == 1.0


def test_rho_falls_when_a_member_is_consolidated():
    s = _FakeStore()
    g.bootstrap(s)
    for i in range(4):
        s.artifacts.put_artifact({"id": f"wn-{i}", "content_type": "text/x-wordnet",
                                  "state": "committed", "context": "m", "content": "z" * 100,
                                  "collection_id": "stage.0.lexicon"})
    # canonical wn-0 consolidates wn-1 -> wn-1 stops counting as a generator
    m = g.collection_metrics(s, "stage.0.lexicon", consolidated={"wn-1"})
    assert m["consolidated"] == 1
    assert m["rho"] == pytest.approx(0.75, abs=1e-6)  # 3 generators of 4 equal-size


# ── consolidation (§7): EXACT identity archives; similarity only observes ─────
def test_consolidate_nearvdup_archives_only_byte_identical():
    """Members sharing a `content_ref` are the same bytes — that is the ONLY merge authority."""
    s = _FakeStore()
    g.bootstrap(s)
    base = " ".join(f"word{i}" for i in range(60))
    # d1/d2 are BYTE-IDENTICAL: same cas ref. d3 is different content.
    for i in (1, 2):
        s.artifacts.put_artifact({"id": f"d{i}", "content_type": "text/markdown",
                                  "state": "committed", "content": base,
                                  "content_ref": "cas/" + "a" * 64,
                                  "lemmas": ["a"] if i == 1 else [], "provenance": "observed",
                                  "collection_id": "stage.1.grammar"})
    s.artifacts.put_artifact({"id": "d3", "content_type": "text/markdown", "state": "committed",
                              "content": "totally different unrelated prose about cats and hats",
                              "content_ref": "cas/" + "b" * 64,
                              "lemmas": ["x"], "collection_id": "stage.1.grammar"})
    dry = g.consolidate_nearvdup(s, collection_id="stage.1.grammar", apply=False)
    assert dry["groups"] == 1 and dry["consolidated"] == 1 and dry["applied"] is False
    assert dry["merge_basis"] == "content_ref_exact"
    # nothing changed on a dry run
    assert s.artifacts.get_artifact("d2")["state"] == "committed"
    assert not s.graph.neighbors("d1", "consolidates", direction="out")

    applied = g.consolidate_nearvdup(s, collection_id="stage.1.grammar", apply=True)
    assert applied["consolidated"] == 1
    # d1 is canonical (more keyed) -> consolidates -> d2, and d2 archived (content retained)
    assert "d2" in s.graph.neighbors("d1", "consolidates", direction="out")
    d2 = s.artifacts.get_artifact("d2")
    assert d2["state"] == "archived" and d2["consolidated_by"] == "d1" and d2["content"]


def test_nearvdup_never_archives_on_similarity_only():
    """Two docs that a 0.85 similarity threshold would flag as near-duplicates, but whose bytes
    differ, survive — and are recorded as a `near_dup_candidate` observation instead.

    The LSH candidate-score distribution has no valley (§9), so any cut through it is ill-posed;
    the estimator's own resolution is ±0.0316 at J=0.85. Archiving on that estimate is the
    unrecoverable direction."""
    from prism import minhash
    s = _FakeStore()
    g.bootstrap(s)
    base = " ".join(f"word{i}" for i in range(200))
    near = base + " word200 word201"               # >0.85 Jaccard, but NOT the same bytes
    assert minhash.estimated_jaccard(minhash.signature(base), minhash.signature(near)) >= 0.85
    s.artifacts.put_artifact({"id": "n1", "content_type": "text/markdown", "state": "committed",
                              "content": base, "content_ref": "cas/" + "1" * 64,
                              "minhash": list(minhash.signature(base)),
                              "lemmas": ["a"], "provenance": "observed"})
    s.artifacts.put_artifact({"id": "n2", "content_type": "text/markdown", "state": "committed",
                              "content": near, "content_ref": "cas/" + "2" * 64,
                              "minhash": list(minhash.signature(near)),
                              "lemmas": [], "provenance": "observed"})
    r = g.consolidate_nearvdup(s, content_type="text/markdown", apply=True)
    assert r["consolidated"] == 0                  # nothing archived on a similarity estimate
    assert s.artifacts.get_artifact("n2")["state"] == "committed"
    assert not s.graph.neighbors("n1", "consolidates", direction="out")
    # ...but the signal is NOT thrown away — it is retained as an advisory observation.
    assert r["near_dup_candidate_edges"] == 1 and r["similarity_is_advisory"] is True
    assert "n2" in s.graph.neighbors("n1", "near_dup_candidate", direction="out")


def test_nearvdup_does_not_sign_from_the_content_preview():
    """`content` is a 300-char preview on legacy rows. Signing from it compares a preview-derived
    signature against full-content ones, biasing toward false merges on exactly the templated stubs
    this operator targets (templates share their opening 300 chars by construction). An artifact
    with no stored signature is not comparable, and that is reported, never faked."""
    s = _FakeStore()
    g.bootstrap(s)
    base = " ".join(f"word{i}" for i in range(200))
    for i, ref in ((1, "1"), (2, "2")):
        s.artifacts.put_artifact({"id": f"p{i}", "content_type": "text/markdown",
                                  "state": "committed", "content": base,   # content but NO minhash
                                  "content_ref": "cas/" + ref * 64,
                                  "lemmas": [], "provenance": "observed"})
    r = g.consolidate_nearvdup(s, content_type="text/markdown", apply=True)
    assert r["skipped_no_signature"] == 2          # counted as UNMEASURED, not as dissimilar
    assert r["compared"] == 0
    assert r["near_dup_candidate_edges"] == 0              # 0 here means "could not tell", and says so
    assert r["consolidated"] == 0


def test_consolidate_colimit_draws_morphisms():
    s = _FakeStore()
    g.bootstrap(s)
    for i in (1, 2, 3):
        s.artifacts.put_artifact({"id": f"m{i}", "content_type": "text/x-wordnet",
                                  "state": "committed", "content": "dog sense", "lemmas": ["dog"]})
    r = g.consolidate_colimit(s, ["m1", "m2", "m3"], concept_lemmas=["dog"], apply=True)
    assert r["members"] == 3 and r["consolidates_edges"] == 3
    canon = r["canonical"]
    assert s.artifacts.get_artifact(canon)["content_type"] == "application/x-concept"
    assert set(s.graph.neighbors(canon, "consolidates", direction="out")) == {"m1", "m2", "m3"}


# ── curriculum pump (Ember self-drives ingestion): resume + promote ───────────
def test_advance_curriculum_resumes_and_promotes(monkeypatch):
    s = _FakeStore()
    g.bootstrap(s)

    def fake_ingest(store, skip, limit):
        base = _count_content(store, "stage.1.grammar")
        for k in range(limit):
            i = base + k
            store.artifacts.put_artifact({"id": f"w{i}", "content_type": "text/markdown",
                                          "state": "committed", "content": "x", "lemmas": ["x"],
                                          "collection_id": "stage.1.grammar"})
        return {"ingested": limit}

    monkeypatch.setattr(g, "CURRICULUM", [
        {"stage": "stage.1.grammar", "ingest": fake_ingest, "target": 25, "per_tick": 10}])

    r1 = g.advance_curriculum(s)
    assert r1["stage"] == "stage.1.grammar" and r1["now"] == 10 and r1["promoted"] is False
    r2 = g.advance_curriculum(s)
    assert r2["now"] == 20 and r2["promoted"] is False
    r3 = g.advance_curriculum(s)                 # 20 -> 30 >= 25 -> promoted
    assert r3["now"] == 30 and r3["promoted"] is True
    assert g._promoted(s, "stage.1.grammar")
    # once every stage is promoted, the pump reports complete (idle)
    assert g.advance_curriculum(s) == {"curriculum": "complete"}


def _count_content(store, cid):
    return sum(1 for _ in store.artifacts.list_artifacts(collection_id=cid))


# ── hot operators: define from data, invokable immediately (no restart) ───────
def test_define_operator_is_invokable_immediately():
    s = _FakeStore()
    g.bootstrap(s)
    # define a composition operator purely as DATA
    d = g.invoke(s, "op.operator.define", {"id": "op.test.alias", "kind": "composition",
                                           "spec": {"steps": [{"op": "op.consistency"}]},
                                           "offer": "alias for consistency"})["result"]
    assert d["defined"] and d["invokable_now"]
    a = s.artifacts.get_artifact("op.test.alias")
    assert a["kind"] == "composition" and a["spec"]["steps"][0]["op"] == "op.consistency"
    assert a["cited_from"] == g.CITE_GENESIS      # defined operators are still provenanced
    # invoke the just-defined operator — no code change, no restart
    r = g.invoke(s, "op.test.alias", {})["result"]
    assert "steps" in r and r["steps"][0]["operator"] == "op.consistency"
    # a source spec stores its declarative definition (interpreted generically at invoke)
    d2 = g.invoke(s, "op.operator.define", {"id": "op.source.mydataset", "kind": "source",
                                            "spec": {"repo": "org/ds", "config": "v1",
                                                     "stage": "staging"}})["result"]
    assert d2["defined"]
    assert s.artifacts.get_artifact("op.source.mydataset")["spec"]["repo"] == "org/ds"
    # bad kind is rejected
    assert g.invoke(s, "op.operator.define", {"id": "op.x", "kind": "bogus", "spec": {}})["result"]["defined"] is False


# ── MinHash-LSH near-dup (deterministic, model-free) ─────────────────────────
def test_minhash_finds_templated_near_dups():
    from prism import minhash
    # templated stubs (near-identical) vs a distinct doc
    base = ("{name} is a small village located in the {region} administrative district of the "
            "country. According to the most recent census the village had a modest population of "
            "several hundred residents. The village is served by a local road and lies near a river. "
            "It has a primary school, a place of worship, and several small family farms nearby.")
    items = [
        ("v1", base.format(name="Aville", region="northern")),
        ("v2", base.format(name="Bville", region="southern")),
        ("v3", base.format(name="Cville", region="eastern")),
        ("distinct", "Quantum chromodynamics is the theory of the strong interaction between quarks "
                     "and gluons, the fundamental particles that make up composite hadrons such as "
                     "the proton, neutron and pion. It is a type of quantum field theory called a "
                     "non-abelian gauge theory with symmetry group SU(3)."),
    ]
    groups = minhash.near_dup_groups(items, threshold=0.5)
    # near-dup villages form a group; the distinct QCD doc is NEVER pulled in (no false positive).
    # (LSH recall is probabilistic, so we assert the villages cluster together, not exact indices.)
    assert groups, "near-dup villages should form a group"
    assert any(len(set(g) & {0, 1, 2}) >= 2 for g in groups)
    assert all(3 not in g for g in groups)
    # signatures are deterministic across calls (two workers must agree)
    assert minhash.signature(items[0][1]) == minhash.signature(items[0][1])


# ── checks & balances: consistency invariants ────────────────────────────────
def test_consistency_clean_universe_passes():
    s = _FakeStore()
    g.bootstrap(s)
    c = g.consistency(s)
    assert c["all_pass"] is True and not c["anomalies"]
    names = {ch["check"] for ch in c["checks"]}
    assert {"rho_in_[0,1]", "generators_le_corpus", "provenance_invariant",
            "fitness_in_[0,1]", "mass_monotonic"} <= names


def test_consistency_flags_broken_provenance():
    s = _FakeStore()
    g.bootstrap(s)
    # an artifact with no citation violates the §12 invariant -> hard anomaly
    s.artifacts.put_artifact({"id": "orphan", "content_type": "text/markdown",
                              "state": "committed", "content": "x", "lemmas": ["x"]})
    c = g.consistency(s)
    assert c["all_pass"] is False
    assert any(a["check"] == "provenance_invariant" for a in c["anomalies"])


# ── owner memory: private + gated + consent + stake ──────────────────────────
def test_remember_is_private_owner_scoped():
    s = _FakeStore()
    g.bootstrap(s)
    from mantle.db import access
    r = g.remember(s, "my project ships on Friday", principal="author@example.com")
    assert r["stored"] and r["private"] is True
    a = s.artifacts.get_artifact(r["id"])
    assert "owner" not in a                         # no owner FIELD — ownership is the grant
    assert a["collection_id"] == "private.author@example.com"
    assert a["provenance"] == "human_validated"     # the owner staked it — highest rung
    assert a["cited_from"] == "cite.owner.author@example.com"
    # with Garage present the index holds NO cleartext preview (content_ref only); the fake store
    # has no Garage, so content falls back inline — encryption is exercised in the live smoke path.
    assert a.get("lemmas")                          # keyed for the owner's recall either way
    # Private is a grant, not a flag: the memory carries no visibility/no_share, and `access` sees it as
    # non-public (its collection is gated by the owner's Read grant, minted by `_ensure_private`).
    assert "no_share" not in a and "visibility" not in a
    assert access.is_public(s, a) is False
    assert access.gated_collections(s) == {"private.author@example.com"}
    assert access.can_read(s, a, "author@example.com") is True
    assert access.can_read(s, a, "someone-else") is False


def test_share_requires_consent_then_stakes():
    s = _FakeStore()
    g.bootstrap(s)
    mid = g.remember(s, "a claim I might share", principal="author@example.com")["id"]
    from mantle.db import access
    # without confirm: nothing shared, nothing changes — still gated by the owner's grant
    r1 = g.share(s, mid, principal="author@example.com")
    assert r1["shared"] is False and r1["reason"] == "consent required"
    assert access.is_public(s, s.artifacts.get_artifact(mid)) is False
    # a non-owner cannot consent
    assert g.share(s, mid, principal="someone-else", confirm=True)["shared"] is False
    # #2 make public: owner consents with a stake → the same artifact is made public by a grant to the
    # public entity (no copy, no re-key). It now reads public to everyone and carries the staked claim.
    r2 = g.share(s, mid, principal="author@example.com", confirm=True, stake=2.0)
    assert r2["shared"] and r2["staked"] == 2.0 and r2["mode"] == "public" and r2["id"] == mid
    a = s.artifacts.get_artifact(mid)
    assert access.is_public(s, a) is True               # made public — readable by all + meshes out
    assert access.can_read(s, a, "anyone-at-all") is True
    assert a["staked_claim"]["by"] == "author@example.com" and a["staked_claim"]["stake"] == 2.0
    assert "subjects" in a["collections"]


def test_share_with_a_person_grants_read_without_making_public():
    """#1 — sharing with a person is a Read grant: their light-cone reaches this artifact, it stays
    private (not public), and only they gain access — not everyone."""
    s = _FakeStore()
    g.bootstrap(s)
    from mantle.db import access
    mid = g.remember(s, "for ada only", principal="author@example.com")["id"]
    assert access.can_read(s, s.artifacts.get_artifact(mid), "ada") is False   # ada cannot yet
    r = g.share(s, mid, to_principal="ada", principal="author@example.com", confirm=True)
    assert r["shared"] and r["mode"] == "grant" and r["with"] == "ada"
    a = s.artifacts.get_artifact(mid)
    assert access.can_read(s, a, "ada") is True         # ada's light-cone now reaches this artifact
    assert access.is_public(s, a) is False              # still NOT public
    assert access.can_read(s, a, "eve") is False        # and only ada, not everyone


def test_nearvdup_never_consolidates_private():
    """The private artifact shares `content_ref` with the public one on purpose.

    Merge authority is exact `content_ref` identity, so a private artifact with different bytes
    would be excluded for the wrong reason and this test would pass even with the privacy filter
    deleted — a test that cannot fail. Giving them the same ref means the only thing standing
    between them and a merge is the privacy check, which is what this test verifies. The control
    below proves the setup would otherwise consolidate."""
    s = _FakeStore()
    g.bootstrap(s)
    base = " ".join(f"tok{i}" for i in range(60))
    REF = "cas/" + "c" * 64
    s.artifacts.put_artifact({"id": "pub", "content_type": "text/markdown", "state": "committed",
                              "content": base, "content_ref": REF,
                              "lemmas": ["a"], "provenance": "observed"})
    # a private memory with BYTE-IDENTICAL content must NOT be pulled into consolidation
    from mantle.db import access
    g.remember(s, base, principal="author@example.com")
    priv = [a for a in s.artifacts.list_artifacts()
            if a.get("collection_id") == "private.author@example.com"
            and a.get("content_type") == "text/markdown" and not access.is_public(s, a)]
    assert priv, "setup failed: remember() produced no private (grant-gated) artifact to exclude"
    for a in priv:
        a["content_ref"] = REF
        a["content_type"] = "text/markdown"
        s.artifacts.put_artifact(a)
    r = g.consolidate_nearvdup(s, content_type="text/markdown", apply=True)
    assert r["consolidated"] == 0                    # private excluded from the scope
    assert s.artifacts.get_artifact("pub")["state"] == "committed"

    # CONTROL: identical setup minus the privacy marking DOES consolidate — so the assertion
    # above is load-bearing rather than incidental.
    s2 = _FakeStore()
    g.bootstrap(s2)
    for i in ("pub", "other"):
        s2.artifacts.put_artifact({"id": i, "content_type": "text/markdown", "state": "committed",
                                   "content": base, "content_ref": REF,
                                   "lemmas": ["a"] if i == "pub" else [], "provenance": "observed"})
    assert g.consolidate_nearvdup(s2, content_type="text/markdown", apply=True)["consolidated"] == 1




# ── skip-guarded integration smoke (real store + nltk) ───────────────────────
def _store_up():
    try:
        from mantle.shard.local_store import open_store
        return open_store().ready()
    except Exception:
        return False


@pytest.mark.skipif(not os.getenv("EMBER_STORE_KEYS_DIR") or not _store_up(),
                    reason="live leaf store not configured/reachable")
def test_stage0_smoke_small():
    from mantle.shard.local_store import open_store
    pytest.importorskip("nltk")
    s = open_store()
    g.bootstrap(s)
    # force=False so a store already carrying WordNet hits the idempotency guard and
    # returns without re-emitting edges (never pollutes the live corpus).
    r = g.ingest_stage0_wordnet(s, limit=50)
    if "skipped" in r:
        # already ingested at scale: the guard held, and the source triple must exist
        assert s.artifacts.get_artifact("op.source.wordnet") is not None
        assert s.artifacts.get_artifact("cite.wordnet") is not None
    else:
        assert r["synsets"] >= 1
        assert s.artifacts.get_artifact("cite.wordnet") is not None
        assert s.artifacts.get_artifact("op.source.wordnet") is not None


def test_candidate_edges_are_budgeted_and_the_drop_is_REPORTED():
    """LSH candidate pairs scale ~n^1.83, not linearly: doubling the corpus multiplies candidate
    pairs by 3.2-4.0x, so the 6.11M-blob corpus implies ~2e7 pairs — against a graph holding
    273,000 real edges. Emitting one edge each would be a 70x explosion of derived data that LSH
    can regenerate on demand.

    The budget exists because the honest number is large; this test exists because a cap that does
    not say what it dropped turns a partial run into one that reads as complete."""
    from prism import minhash
    s = _FakeStore()
    g.bootstrap(s)
    base = " ".join(f"word{i}" for i in range(200))
    # four mutually-similar docs -> several candidate pairs, but a budget of 1
    for i in range(4):
        txt = base + (" extra%d" % i)
        s.artifacts.put_artifact({"id": f"c{i}", "content_type": "text/markdown",
                                  "state": "committed", "content": txt,
                                  "content_ref": "cas/" + str(i) * 64,
                                  "minhash": list(minhash.signature(txt)),
                                  "lemmas": ["a"] if i == 0 else [],
                                  "provenance": "observed"})
    r = g.consolidate_nearvdup(s, content_type="text/markdown", apply=True,
                               max_candidate_edges=1)
    assert r["near_dup_edge_budget"] == 1
    assert r["near_dup_candidate_edges"] <= 1
    # the whole point: what LSH found is reported, not just what was written
    assert r["near_dup_candidates_found"] >= r["near_dup_candidate_edges"]
    assert (r["near_dup_candidates_dropped"]
            == r["near_dup_candidates_found"] - r["near_dup_candidate_edges"])
    # and nothing was archived on similarity regardless of budget
    assert r["consolidated"] == 0
    for i in range(4):
        assert s.artifacts.get_artifact(f"c{i}")["state"] == "committed"