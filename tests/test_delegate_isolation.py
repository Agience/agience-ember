"""Delegate isolation — each delegate holds its own cognitive state.

Grouped by invariant rather than by function, following `test_drain_invariants.py`, because one
invariant governs three separate places: attention, subjective time, and forgetting rates are each
per-delegate.

The assertions are on the mechanism rather than on the API surface: a shared screen, a shared clock
or a shared eviction budget all leave the public calls working exactly as they do now.
"""
from __future__ import annotations

import pytest

from ember.signal import forgetting
from ember.runtime.delegate import Delegate, LOCAL_PERSON

from _fakes import _install_offline_wordnet


# ── fixtures ──────────────────────────────────────────────────────────────────────────────────
def _syn(name):
    from crystal.ontology import driver as wn
    try:
        return wn.synset(name)
    except Exception:
        return None


# `_install_offline_wordnet` lives in `_fakes.py` and is imported above. Four test modules share it,
# and a shared fixture has one home outside any test module: importing a fixture out of a test
# module makes collection of every importer depend on that module (see conftest.py).


@pytest.fixture(scope="module")
def concepts():
    if not _install_offline_wordnet():
        pytest.skip("WordNet not available in this environment")
    wolf, dog, tree = _syn("wolf.n.01"), _syn("dog.n.01"), _syn("tree.n.01")
    if not (wolf and dog and tree):
        pytest.skip("WordNet index incomplete")
    return wolf, dog, tree


class _EdgeSql:
    """The raw-SQL face `crystal.ontology.driver` reads edges through. Empty is a truthful answer for
    a delegate store that holds cognition rather than an ontology — the driver's queries return no
    rows, which is what "this store has no ontology edges" should look like."""

    def __init__(self):
        import sqlite3
        self._c = sqlite3.connect(":memory:")
        self._c.execute("CREATE TABLE edge (src TEXT, dst TEXT, label TEXT, props TEXT)")

    def read(self):
        return self._c

    def write(self):
        return self._c


class _FakeStore:
    """Minimal store — a delegate holds its own cognition against whatever store it is handed, so a
    dict and an in-memory `edge` table are enough."""

    def __init__(self):
        self.docs = {}
        self.artifacts = self
        self._writes = 0
        self._origin = "delegate-fake-%d" % id(self)
        # The SQL face. `crystal.ontology.driver._conn` reads edges with raw SQL through
        # `_observe(store).db.read()` rather than through the graph API, so a store double needs a
        # `db` attribute to be readable at all. `artifacts is self` here, so one attribute serves
        # both faces.
        self.db = _EdgeSql()

    def write_mark(self):
        """The whole-of-store write mark `crystal.ontology.driver` hangs every cache on.

        A store that cannot report a mark gives `driver._gate` no reading to verify against, so no
        cache may claim currency: the generation advances on every poll and the ontology layer
        re-verifies from scratch per read, measured here at 10-17 s per test against 5.4 s with the
        mark present. See `_fakes._FakeArtifacts.write_mark` for the full measurement and why the
        mark belongs in the double.
        """
        return ((self._origin, self._writes),)

    def get_artifact(self, aid):
        return self.docs.get(aid)

    def put_artifact(self, doc):
        self.docs[doc["id"]] = dict(doc)
        self._writes += 1
        return 1


# ── INVARIANT 1: attention never crosses a delegate boundary ──────────────────────────────────
def test_foreign_traces_are_never_read(concepts):
    """A's attention stays out of B's read.

    `Screen.read()` consults `Trace.witness`, so a trace deposited by another observer contributes
    no energy. Scoring by geometric similarity alone would surface A attending to `wolf.n.01` inside
    B's query for `dog.n.01` as B's own present tense, for something B never observed. wolf and dog
    are close in JC, so this pair is the bleed path."""
    wolf, dog, _ = concepts
    scr = forgetting.Screen(witness="delegate-B")
    scr.observe(wolf, tick=1, witness="delegate-A")      # someone else's memory, in B's screen
    present, past, rows = scr.read(dog, now=1)
    assert rows == [], "a foreign trace contributed energy to this delegate's read"
    assert present == 0.0 and past == 0.0
    assert scr.foreign_witnesses() == 1, "the foreign trace was not counted"


def test_own_traces_are_read(concepts):
    """The other half of the witness check: the delegate's own memory reads back."""
    wolf, dog, _ = concepts
    scr = forgetting.Screen(witness="delegate-A")
    scr.observe(wolf, tick=1, witness="delegate-A")
    present, past, rows = scr.read(dog, now=1)
    assert rows, "the delegate's own trace was dropped"
    assert present + past > 0.0
    assert scr.foreign_witnesses() == 0


def test_two_delegates_do_not_share_a_screen(concepts):
    """Two delegates of the same person are two cognitions.

    The grain is the delegate id, not the principal. At principal grain two delegates of one human
    share `_OBS` and one screen, so B recalls what A was taught and reads A's attention as its own."""
    wolf, dog, _ = concepts
    store = _FakeStore()
    a = Delegate.get(store, person="p@example.com", id="d.a", restore=False)
    b = Delegate.get(store, person="p@example.com", id="d.b", restore=False)
    assert a.screen is not b.screen
    a.observe(wolf)
    present, past, rows = b.screen.read(dog, now=b.tick)
    assert rows == [], "delegate B read delegate A's attention"


# ── INVARIANT 2: subjective time is per-delegate ───────────────────────────────────────────────
def test_a_busy_delegate_does_not_age_out_a_quiet_one(concepts):
    """The clock that dates a delegate's memories is that delegate's own.

    Under one global `_TICK` a delegate handling 1000 turns advances the clock a quiet delegate's
    traces are read against: a trace stamped at tick 5 and read at now=1005 has cooled to nothing on
    any decay curve, so the quiet delegate's memory is gone having done nothing. The decay is the
    screen's own measured curve and the is/was boundary is `prism.resolution.signal_end` over its
    own amplitudes, so the effect is stated in ticks rather than in a typed cutoff."""
    wolf, _, _ = concepts
    store = _FakeStore()
    busy = Delegate.get(store, person="p@example.com", id="d.busy", restore=False)
    quiet = Delegate.get(store, person="p@example.com", id="d.quiet", restore=False)

    quiet.observe(wolf)                       # quiet's one and only observation
    for _ in range(1000):                     # busy does a great deal of thinking
        busy.observe(wolf)

    assert quiet.tick == 1, "another delegate's activity advanced this delegate's clock"
    present, past, rows = quiet.screen.read(wolf, now=quiet.tick)
    assert rows, "the quiet delegate's memory was aged out by a busy neighbour"
    assert present > 0.0, "the memory should still be PRESENT — one tick has passed, not 1000"


def test_tick_is_monotonic_per_delegate(concepts):
    store = _FakeStore()
    d = Delegate.get(store, person="p@example.com", id="d.mono", restore=False)
    ticks = [d.next_tick() for _ in range(50)]
    assert ticks == sorted(ticks) and len(set(ticks)) == 50


# ── INVARIANT 3: the screen is bounded, and bounding forgets the COLDEST ───────────────────────
def test_screen_is_bounded(concepts):
    """The screen holds `capacity` traces and counts what it evicted.

    The bound belongs with the per-delegate split: a bound on a shared screen would make agents
    compete for working memory. `read()` carries no amplitude cutoff — measured on the live store,
    `a <= 0.02` dropped 14 of 15 traces at lag 20, so a cutoff of that shape is a second, unmeasured
    eviction rule sitting on top of the capacity bound."""
    wolf, _, _ = concepts
    scr = forgetting.Screen(witness="w", capacity=10)
    for t in range(1, 51):
        scr.observe(wolf, tick=t, witness="w")
    assert len(scr.traces) == 10
    assert scr.evicted == 40


def test_eviction_drops_the_coldest_first(concepts):
    """Amplitude is non-increasing in elapsed ticks, so the lowest tick is the lowest amplitude.
    Eviction is 'forget what has cooled most' — the module's own model, not an arbitrary cap."""
    wolf, _, _ = concepts
    scr = forgetting.Screen(witness="w", capacity=5)
    for t in range(1, 21):
        scr.observe(wolf, tick=t, witness="w")
    kept = sorted(tr.tick for tr in scr.traces)
    assert kept == [16, 17, 18, 19, 20], "eviction did not keep the warmest traces"


# ── INVARIANT 4: attention and its clock persist together, or not at all ───────────────────────
def test_screen_and_tick_round_trip(concepts):
    """Attention and its clock are saved and restored together.

    Traces restored without their clock land in a frame whose tick has reset to 0: every trace then
    sits in the future, `amplitude()` clamps `dt` to 0, and the whole screen reads as freshly
    observed."""
    wolf, _, _ = concepts
    store = _FakeStore()
    a = Delegate.get(store, person="p@example.com", id="d.persist", restore=False)
    for _ in range(5):
        a.observe(wolf)
    assert a.save_screen() is True

    from ember.runtime import delegate as _d
    _d.reset_cache()
    b = Delegate.get(store, person="p@example.com", id="d.persist")   # restore=True
    assert b is not a
    assert b.tick == a.tick, "subjective time did not survive the restart"
    assert len(b.screen.traces) == len(a.screen.traces)
    assert b.screen.foreign_witnesses() == 0


def test_restore_drops_foreign_traces(concepts):
    """The witness check applies on the restore path too: a persisted screen carrying another
    observer's traces restores only its own."""
    wolf, _, _ = concepts
    scr = forgetting.Screen(witness="d.mine")
    doc = {"traces": [{"concept": wolf.name(), "tick": 3, "witness": "d.someone-else"},
                      {"concept": wolf.name(), "tick": 4, "witness": "d.mine"}]}
    n = scr.restore(doc)
    assert n == 1, "a foreign trace was restored into this screen"
    assert scr.foreign_witnesses() == 1


# ── INVARIANT 5: no cognitive act is silently attributed to a real human ───────────────────────
def test_unconfigured_principal_is_not_a_person(monkeypatch):
    """An unconfigured process resolves to `LOCAL_PERSON`, which is not shaped like an email, and
    `genesis._principal()` raises rather than supplying one. A default person here would record
    every unattributed act as that human's private, HUMAN_VALIDATED memory."""
    monkeypatch.delenv("EMBER_PRINCIPAL", raising=False)
    store = _FakeStore()
    from ember.runtime import delegate as _d
    _d.reset_cache()
    d = Delegate.get(store, restore=False)
    assert d.person == LOCAL_PERSON
    assert "@" not in d.person, "an anonymous process resolved to something shaped like a person"

    from ember import genesis
    with pytest.raises(RuntimeError, match="EMBER_PRINCIPAL"):
        genesis._principal()


# ── INVARIANT 6: activation is an entrypoint of its own ───────────────────────────────────────
def test_recognize_is_the_seed_then_spread_pipeline(concepts):
    """Seeding, spreading and ranking are separate stages so a non-text signal can enter at any of
    them. `recognize` over text is exactly `seeds_from_text -> spread_seeds -> rank_fired` — the
    composition is the contract — and it surfaces the query's own concepts."""
    from ember.ontology import activation
    store = _FakeStore()
    text = "the wolf and the dog"
    pipeline = activation.rank_fired(
        store, activation.spread_seeds(activation.seeds_from_text(text), spread=4))
    assert [a["concept"] for a in activation.recognize(store, text, spread=4)] == \
           [a["concept"] for a in pipeline]
    top = {a["concept"] for a in activation.recognize(store, text)[:8]}
    assert "wolf.n.01" in top and "dog.n.01" in top


def test_an_artifact_can_activate_a_delegate(concepts):
    """The peer-signal path: a mesh-applied artifact seeds activation and deposits a trace on the
    receiving delegate's own clock, so an artifact is an entry point alongside local text."""
    from ember.ontology import activation
    store = _FakeStore()
    from ember.runtime import delegate as _d
    _d.reset_cache()
    d = Delegate.get(store, person="p@example.com", id="d.peer", restore=False)

    art = {"id": "obs.xyz", "subject": "wolf", "object": "howl", "lemmas": ["wolf", "howl"]}
    seeds = activation.seeds_from_artifact(art)
    assert seeds, "an artifact produced no seeds"

    r = activation.activate(d, art, source="mesh")
    assert r["source"] == "mesh"
    assert r["activations"], "an artifact did not activate anything"
    assert d.tick == 1, "activation did not deposit a trace on the delegate's own clock"


def test_a_wn_artifact_seeds_its_own_synset_without_a_surface_round_trip(concepts):
    from ember.ontology import activation
    seeds = activation.seeds_from_artifact({"id": "wn-wolf.n.01"})
    assert "wolf.n.01" in seeds


def test_explicit_seed_map_is_accepted(concepts):
    """A signal that already knows its coordinates — no text, no artifact."""
    from ember.ontology import activation
    store = _FakeStore()
    from ember.runtime import delegate as _d
    _d.reset_cache()
    d = Delegate.get(store, person="p@example.com", id="d.seeded", restore=False)
    r = activation.activate(d, {"wolf.n.01": 3.0}, source="sensor")
    assert r["lead"] is not None
    assert r["source"] == "sensor"


# ── INVARIANT 7: activation has a memory term, and it is this delegate's own ───────────────────
def test_screen_read_is_actually_consulted(concepts):
    """Inference consults `Screen.read()`, so a deposited trace reaches the answer. The memory term
    a ranking carries is the reading delegate's own: a delegate with history scores above zero on
    it, a delegate without history scores exactly zero."""
    from ember.ontology import activation
    wolf, _, _ = concepts
    store = _FakeStore()
    from ember.runtime import delegate as _d
    _d.reset_cache()
    attentive = Delegate.get(store, person="p@example.com", id="d.attentive", restore=False)
    naive = Delegate.get(store, person="p@example.com", id="d.naive", restore=False)

    for _ in range(3):
        attentive.observe(wolf)

    p_att, past_att = activation.recall(attentive, "wolf.n.01")
    p_nai, past_nai = activation.recall(naive, "wolf.n.01")
    assert (p_att + past_att) > 0.0, "the delegate's own trace was not readable"
    assert (p_nai + past_nai) == 0.0, "a delegate with no history reported memory"

    acts = activation.rank_fired(store, {"wolf.n.01": 1.0}, d=attentive)
    assert acts[0]["memory"] > 0.0, "the memory term was not attached to the activation"
    acts_n = activation.rank_fired(store, {"wolf.n.01": 1.0}, d=naive)
    assert acts_n[0]["memory"] == 0.0


def test_memory_term_is_zero_without_a_delegate(concepts):
    """With no delegate supplied there is no memory to read, so the term is zero on every row
    rather than drawn from some ambient screen."""
    from ember.ontology import activation
    store = _FakeStore()
    acts = activation.rank_fired(store, {"wolf.n.01": 1.0, "dog.n.01": 1.0})
    assert all(a["memory"] == 0.0 for a in acts)


# ── INVARIANT 8: firing is resolution, not a salience threshold ───────────────────────────────
def test_a_signal_that_resolves_nothing_does_not_fire(concepts):
    """`fired` is the computed null rather than a chosen floor
    ([[one-resolution-not-thresholds]]): a signal fires exactly when something resolved. There is no
    threshold to clear, so a signal that resolves no concept fires nothing and leaves no lead."""
    from ember.ontology import activation
    store = _FakeStore()
    from ember.runtime import delegate as _d
    _d.reset_cache()
    d = Delegate.get(store, person="p@example.com", id="d.gate", restore=False)
    r = activation.activate(d, {}, source="peer")          # nothing to resolve
    assert r["fired"] is False and r["lead"] is None


def test_a_signal_that_resolves_fires(concepts):
    from ember.ontology import activation
    store = _FakeStore()
    from ember.runtime import delegate as _d
    _d.reset_cache()
    d = Delegate.get(store, person="p@example.com", id="d.fire", restore=False)
    r = activation.activate(d, {"wolf.n.01": 1.0}, source="peer")   # resolves -> fires, no floor
    assert r["fired"] is True and r["lead"] == "wolf.n.01"


def test_activation_observes_the_disambiguated_sense(concepts):
    """`activate` deposits the whole fired field, in its measured order, at the senses it resolved.

    Two properties are pinned together. `_observe_field` resolves `wn.synset(name)` by name, so the
    disambiguation the recognition just performed survives the deposit; taking a surface word and
    doing `wn.synsets(word)[0]` would pick sense 0 again and discard it. And the deposit is the
    field rather than the lead alone, at each concept's measured share, because nothing measured
    that a recognition contributes exactly one observation.

    The fixture makes the sense half falsifiable. The seed `wolf.n.01` fires a 14-concept is-a
    chain, and four of those concepts are not sense 0 of their own surface word:

        wolf.n.01     vs synsets("wolf")[0]   = beast.n.02
        canine.n.02   vs synsets("canine")[0] = canine.n.01   (the tooth)
        whole.n.02    vs synsets("whole")[0]  = whole.n.01
        object.n.01   vs synsets("object")[0] = aim.n.02

    so a re-lookup through `synsets(_word(name))[0]` moves at least those four. This also goes red
    if the field stops being observed in its measured order, or if the lead stops leading."""
    from ember.ontology import activation
    from crystal.ontology import driver as wn
    store = _FakeStore()
    from ember.runtime import delegate as _d
    _d.reset_cache()
    d = Delegate.get(store, person="p@example.com", id="d.sense", restore=False)
    r = activation.activate(d, {"wolf.n.01": 5.0}, source="local")
    assert d.screen.traces, "nothing was observed"
    observed = [t.concept.name() for t in d.screen.traces]
    fired = [a["concept"] for a in r["activations"]]

    # The whole field is observed, in the order it was ranked.
    assert len(observed) > 1, "only one concept was observed: the field is not being deposited"
    assert observed == fired[:len(observed)], (
        "the observed traces are not the fired field in its measured order: %r vs %r"
        % (observed, fired))
    assert observed[0] == r["lead"], "the lead is not what led the observed field"

    # The positive control: build what a sense-0 re-lookup would deposit and show it is a genuinely
    # different list. On a fixture where re-lookup happened to be the identity the assertion above
    # would pass either way ([[verification-that-cannot-fail]]).
    def _sense0(name):
        s = wn.synsets(name.split(".")[0])       # the offline lexicon `concepts` bound as default
        return s[0].name() if s else name

    relooked = [_sense0(n) for n in observed]
    divergent = [(n, m) for n, m in zip(observed, relooked) if n != m]
    assert len(divergent) >= 2, (
        "sense-0 re-lookup is the identity on every observed concept, so this test cannot detect "
        "the defect it exists to catch — re-measure the fixture: %r" % (list(zip(observed, relooked)),))
    assert observed != relooked, "the traces are the surface-word re-lookup, not the resolved senses"


# ── INVARIANT 9: authorization is the extent of the field ─────────────────────────────────────
def test_visible_to_requires_the_store_and_reads_no_flag():
    """Access is CRUDEASIO grants; the light-cone coverage lives in test_access. `visible_to` needs
    the store to resolve a grant, so called without one it fails closed. It reads no flag, so a
    `visibility` or `no_share` marker on the row changes nothing: the grant is the one mechanism."""
    from mantle.db.access import visible_to
    me = "me@example.com"
    assert visible_to({"id": "a"}, me) is False                      # no store -> cannot authorize
    assert visible_to({"id": "a", "visibility": "public"}, me) is False        # flags are not read
    assert visible_to({"id": "a", "visibility": "private", "owner": me}, me) is False


def test_unowned_private_row_fails_closed():
    """A row marked private with no owner is unattributable, so it resolves to no grant and is
    visible to nobody. Showing it to everybody is the worst available reading of a malformed row."""
    from mantle.db.access import visible_to
    assert visible_to({"id": "a", "visibility": "private"}, "me@example.com") is False
    assert visible_to({"id": "a", "no_share": True}, "me@example.com") is False
    assert visible_to(None, "me@example.com") is False


def test_filter_visible_without_store_fails_closed():
    """`filter_visible` with no store cannot resolve grants, so it yields nothing: there is one
    mechanism and no flag filter behind it. Grant-scoped filtering is covered in test_access."""
    from mantle.db.access import filter_visible
    rows = [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}]
    assert filter_visible(rows, "me@example.com") == []


def test_a_response_never_surfaces_another_persons_private_memory(concepts):
    """Privacy of surfacing, in the grant model. Two properties keep another person's private memory
    out of my answer: `compose`/`_render_concept` quote only the lead concept's public wordnet gloss,
    so no arbitrary content is aligned in; and a delegate's own memories enter through `load_obs`,
    which is scoped to that person's private collection (`private.<person>`), so another person's
    private triple never loads into my field."""
    import os, tempfile
    from ember.ontology import activation
    from ember.runtime import delegate as _d
    try:
        from mantle.db import open_lattice
    except Exception:
        pytest.skip("lattice store not importable")
    _d.reset_cache()
    L = open_lattice(os.path.join(tempfile.mkdtemp(), "d.db"), origin="t")
    # the other person's private taught triple, grounded in their own private collection
    L.artifacts.put_artifact({"id": "obs.them", "content_type": activation.TRIPLE_TYPE,
                              "operator": "howls", "subject": "wolf", "object": "howl",
                              "collection_id": "private.them@example.com"})
    d_me = Delegate.get(L, person="me@example.com", id="d.me", restore=False)
    d_me.load_obs()
    with d_me._obs_lock:
        ids = {o.get("id") for o in d_me.obs}
    assert "obs.them" not in ids, "another person's private triple entered my field"

    # and a composed response quotes only the concept's public gloss
    body, _cites = activation.compose(L, "wolf", [{"concept": "wolf.n.01"}], person="me@example.com")
    assert "diary" not in body and "howl" not in body


def test_lookup_by_lemma_scopes_when_given_a_principal():
    """`lemmas` is one namespace shared by every content type, including the private rows
    `learn()`/`remember()` write with keyed lemmas by design. A private row is one grounded in a
    grant-gated collection, so a lemma lookup carrying a principal returns that principal's rows and
    the public ones."""
    import os, tempfile
    from mantle.shard import keyed
    from mantle.db import access
    try:
        access._api()
    except Exception:
        import pytest
        pytest.skip("grant subsystem (mantle lattice_api) not on the path")
    from mantle.db import open_lattice
    me, them = "me@example.com", "them@example.com"
    L = open_lattice(os.path.join(tempfile.mkdtemp(), "k.db"), origin="t")
    access.mint_owner_read_grant(L, "private." + me, me)          # grant gates each private collection
    access.mint_owner_read_grant(L, "private." + them, them)
    L.artifacts.put_artifact({"id": "pub", "content_type": "text/markdown", "lemmas": ["dog"],
                              "collection_id": "universe"})
    L.artifacts.put_artifact({"id": "mine", "content_type": "text/markdown", "lemmas": ["dog"],
                              "collection_id": "private." + me})
    L.artifacts.put_artifact({"id": "theirs", "content_type": "text/markdown", "lemmas": ["dog"],
                              "collection_id": "private." + them})

    rows, _typed = keyed.lookup_by_lemma(L.artifacts, "dog", limit=10, principal=me)
    ids = {r["id"] for r in rows}
    assert "theirs" not in ids, "another person's private (grant-gated) row was returned"
    assert ids == {"pub", "mine"}

    # the unscoped path (no principal) does not grant-filter — internal callers only
    unscoped, _ = keyed.lookup_by_lemma(L.artifacts, "dog", limit=10)
    assert "theirs" in {r["id"] for r in unscoped}, "the unscoped path changed behaviour"


def test_delegate_carries_the_provenance_tuple(monkeypatch):
    """origin / person / host — the operator is per-observation, supplied by the caller."""
    monkeypatch.setenv("EMBER_HOST_ID", "host-7")
    store = _FakeStore()
    from ember.runtime import delegate as _d
    _d.reset_cache()
    d = Delegate.get(store, person="p@example.com", id="d.prov", origin="https://agience.ai",
                     restore=False)
    assert d.provenance() == {"origin": "https://agience.ai", "person": "p@example.com",
                              "host": "host-7"}
