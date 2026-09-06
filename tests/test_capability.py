"""Host capabilities: three-valued presence, a governed vocabulary, and an enforceable contract.

Grouped by invariant. `opsign.admit()` takes `host_offers`, and `capability.offers()` is what
produces that set, which is what makes the contract enforceable rather than declarative.
"""
from __future__ import annotations

import pytest

from ember.runtime import capability as cap
from prism.trust import opsign
from _fakes import _FakeStore


@pytest.fixture(autouse=True)
def _no_declarations(monkeypatch):
    monkeypatch.delenv("EMBER_HOST_CAPABILITIES", raising=False)


# ── Invariant 1: three-valued. None is a third value, distinct from False. ────────────────────
def test_presence_is_three_valued():
    """Presence is True, False or None, and None means the probe took no reading. Collapsing None
    into False would turn away operators on hosts that can run them, and would present an unmeasured
    host as a measured-empty one."""
    p = cap.probe_all()
    assert all(v["present"] in (True, False, None) for v in p.values())
    assert any(v["present"] is None for v in p.values()), \
        "no capability is unknown on this host — the fixture cannot exercise the third value"


def test_offers_excludes_unknown_as_well_as_absent():
    """Offering is a promise, so an unknown capability is not offered. It is not denied either, which
    is why `unknown()` reports separately."""
    p = cap.probe_all()
    offered, unknown = set(cap.offers(p)), set(cap.unknown(p))
    assert offered.isdisjoint(unknown)
    for n in offered:
        assert p[n]["present"] is True
    for n in unknown:
        assert p[n]["present"] is None


def test_no_camera_and_nobody_asked_do_not_render_the_same():
    p = dict(cap.probe_all())
    p["compute.gpu"] = {"present": False, "kind": cap.EFFECTOR, "source": "probed", "description": ""}
    p["net.egress"] = {"present": None, "kind": cap.EFFECTOR, "source": "probed", "description": ""}
    assert "compute.gpu" not in cap.offers(p) and "compute.gpu" not in cap.unknown(p)
    assert "net.egress" not in cap.offers(p) and "net.egress" in cap.unknown(p)


def test_egress_is_never_inferred():
    """Determining egress means making a request, and probes run at boot unprompted, so this probe
    takes no reading. As `boot.py` puts it: a leaf that phoned home on startup would not be a leaf."""
    assert cap.CAPABILITIES["net.egress"].probe() is None


def test_gpu_absence_is_only_claimed_where_it_is_knowable():
    """NVIDIA is probed specifically. `compute.gpu` in general reads True or unknown, because
    AMD/Intel/Apple are not detectable without vendor tooling this probe does not depend on, and an
    undetectable device is unmeasured rather than absent."""
    assert cap.CAPABILITIES["compute.gpu"].probe() in (True, None)


def test_no_torch_import_to_answer_a_gpu_question():
    """No trained weights and no model runtime: the GPU probe answers without importing torch, which
    would drag a model runtime into a boot-time probe."""
    import sys
    before = "torch" in sys.modules
    cap.CAPABILITIES["compute.gpu.nvidia"].probe()
    assert ("torch" in sys.modules) == before, "the GPU probe imported torch"


# ── Invariant 2: the vocabulary is governed ───────────────────────────────────────────────────
def test_an_unregistered_name_is_refused():
    """A name outside the vocabulary raises `UnknownCapability`, so operators stay portable: if every
    host invented its own names the scheme would be per-host scripting.

    `sensor.*` and `actuator.*` are open families accepted by prefix, because physical devices are
    unbounded and that is what makes a prism plug-and-play. The governed part is everything else,
    which is what this checks, plus the bare prefix, which names no device."""
    with pytest.raises(cap.UnknownCapability):
        cap.validate(["compute.telepathy"])
    with pytest.raises(cap.UnknownCapability):
        cap.validate(["sensor."])                 # a bare family prefix names no device


def test_validate_normalises():
    assert cap.validate(["compute.cpu", "compute.cpu", " clock.wall "]) == ["clock.wall", "compute.cpu"]
    assert cap.validate([]) == [] and cap.validate(None) == []


def test_every_capability_declares_its_kind():
    """Source and effector are asymmetric: a source pushes and needs a salience gate, an effector
    pulls and needs an effect contract. Every capability declares which it is, and describes itself."""
    for name, c in cap.CAPABILITIES.items():
        assert c.kind in (cap.SOURCE, cap.EFFECTOR), name
        assert c.description


# ── Invariant 3: a declaration is distinguishable from a measurement ──────────────────────────
def test_a_declaration_overrides_a_probe_and_says_so(monkeypatch):
    monkeypatch.setenv("EMBER_HOST_CAPABILITIES", "net.egress")
    p = cap.probe_all()
    assert p["net.egress"]["present"] is True
    assert p["net.egress"]["source"] == "declared", "an asserted capability read as measured"
    assert "net.egress" in cap.offers(p)


def test_a_declaration_can_deny(monkeypatch):
    monkeypatch.setenv("EMBER_HOST_CAPABILITIES", "storage.local=false")
    p = cap.probe_all()
    assert p["storage.local"]["present"] is False
    assert "storage.local" not in cap.offers(p)


def test_an_undeclared_capability_stays_probed(monkeypatch):
    monkeypatch.setenv("EMBER_HOST_CAPABILITIES", "net.egress")
    p = cap.probe_all()
    assert p["compute.cpu"]["source"] == "probed"


# ── Invariant 4: the host is an artifact ──────────────────────────────────────────────────────
def test_host_artifact_carries_presence_and_quantity():
    """Presence and limits are both on the artifact, because they answer different questions: can
    this host do X, versus how much of it is there."""
    a = cap.host_artifact("host-7")
    assert a["content_type"] == cap.HOST_CONTENT_TYPE and a["host"] == "host-7"
    assert set(a["capabilities"]) == set(cap.CAPABILITIES)
    assert "limits" in a and "offers" in a and "unknown" in a


def test_publish_writes_and_reports():
    s = _FakeStore()
    doc = cap.publish(s, "host-7")
    assert doc["published"] is True
    assert s.artifacts.get_artifact("host.host-7") is not None


def test_publish_failure_is_reported_not_swallowed():
    class _Broken:
        def put_artifact(self, doc):
            raise RuntimeError("store is down")
    doc = cap.publish(_Broken(), "host-7")
    assert doc["published"] is False and "store is down" in doc["publish_error"]


# ── Invariant 5: the contract is enforceable ──────────────────────────────────────────────────
def test_admit_accepts_an_operator_the_host_can_satisfy(tmp_path):
    priv, _ = opsign.authority_key(tmp_path, create=True)
    op = opsign.sign_operator({"id": "op.t.cpu", "kind": "composition", "spec": {"steps": []},
                               "requires": ["compute.cpu"]}, priv)
    ok, why = opsign.admit(op, host_offers=cap.offers())
    assert ok is True, why


def test_admit_refuses_what_the_host_does_not_offer(tmp_path):
    priv, _ = opsign.authority_key(tmp_path, create=True)
    op = opsign.sign_operator({"id": "op.t.gpu", "kind": "composition", "spec": {"steps": []},
                               "requires": ["compute.gpu.nvidia"]}, priv)
    ok, why = opsign.admit(op, host_offers=cap.offers())
    if "compute.gpu.nvidia" in cap.offers():
        pytest.skip("this host really does have NVIDIA")
    assert ok is False and "does not offer" in why


def test_an_unprobed_requirement_is_refused_conservatively(tmp_path):
    """An operator requiring an unprobed capability is not admitted: an unmeasured requirement is not
    a satisfied one, and turning away something that might work costs less than admitting something
    that cannot."""
    priv, _ = opsign.authority_key(tmp_path, create=True)
    op = opsign.sign_operator({"id": "op.t.net", "kind": "composition", "spec": {"steps": []},
                               "requires": ["net.egress"]}, priv)
    assert "net.egress" in cap.unknown()
    ok, _why = opsign.admit(op, host_offers=cap.offers())
    assert ok is False, "an operator needing an UNPROBED capability was admitted"


def test_operator_requires_are_validated_against_the_vocabulary():
    with pytest.raises(cap.UnknownCapability):
        cap.validate(["compute.cpu", "gpu"])          # 'gpu' is not the registered name


# ── Invariant 6: a remote host announcing itself ──────────────────────────────────────────────
def test_registration_writes_a_host_and_its_operators():
    """A registration writes the host artifact and one artifact per announced operator. Both prism
    legs POST `{api_uri}/hosts/register` inside a bare `except` logged "non-fatal", so this is the
    only place a host's operators reach the store at all."""
    s = _FakeStore()
    r = cap.register_remote_host(s, name="Warehouse GW-7",
                                 operators=["embeddings.embed", "sensor.read"],
                                 endpoint="http://10.0.0.7:8083")
    assert r["registered"] is True
    host = s.artifacts.get_artifact("host.warehouse-gw-7")
    assert host is not None and host["endpoint"] == "http://10.0.0.7:8083"
    assert len(r["operators"]) == 2
    for oid in r["operators"]:
        assert s.artifacts.get_artifact(oid) is not None


def test_a_remote_host_is_never_reported_as_probed():
    """A machine we are not running on cannot be probed, so its capabilities carry
    `source: declared`."""
    s = _FakeStore()
    cap.register_remote_host(s, name="gw", operators=[], capabilities=["compute.gpu.nvidia"])
    host = s.artifacts.get_artifact("host.gw")
    assert host["probed"] is False
    assert host["capabilities"]["compute.gpu.nvidia"]["source"] == "declared"


def test_unprobed_remote_capabilities_are_unknown_not_absent():
    """Three-valued discipline holds across the wire exactly as it does locally."""
    s = _FakeStore()
    cap.register_remote_host(s, name="gw", operators=[], capabilities=["compute.cpu"])
    host = s.artifacts.get_artifact("host.gw")
    assert "compute.cpu" in host["offers"]
    assert "net.egress" in host["unknown"] and "net.egress" not in host["offers"]


def test_remote_operators_are_unsigned_and_not_admissible():
    """The prism hosts send a name, not a spec, so there is nothing to content-address and nothing to
    verify. Recording such an operator leaves it unsigned, and `admit` says so."""
    s = _FakeStore()
    r = cap.register_remote_host(s, name="gw", operators=["embeddings.embed"])
    op = s.artifacts.get_artifact(r["operators"][0])
    assert op["remote"] is True
    assert "signature" not in op and "spec_hash" not in op
    ok, why = opsign.admit(op, host_offers=cap.offers())
    assert ok is False and "unsigned" in why


def test_a_remote_operator_claim_is_provenanced_as_an_assertion():
    """Nothing has verified it. It is the host's claim about itself."""
    from ember import genesis
    s = _FakeStore()
    r = cap.register_remote_host(s, name="gw", operators=["x"])
    assert s.artifacts.get_artifact(r["operators"][0])["provenance"] == genesis.P_ASSERTION


# ── Invariant 7: a registration cannot claim an id that is not its own ────────────────────────
def test_a_host_cannot_shadow_a_local_operator():
    """Announced names land under `op.host.<name>.`. Written as-is, any host that can reach the
    endpoint could claim `op.dev.run_tests` — the operator that shells out to pytest — and shadow the
    local one."""
    s = _FakeStore()
    r = cap.register_remote_host(s, name="evil",
                                 operators=["op.dev.run_tests", "../../dev.run_tests"])
    for oid in r["operators"]:
        assert oid.startswith("op.host.evil.")
    assert s.artifacts.get_artifact("op.dev.run_tests") is None


def test_slugging_strips_path_and_dot_injection():
    assert cap.remote_operator_id("a/b", "c.d") == "op.host.a-b.c-d"
    assert ".." not in cap.remote_operator_id("../..", "../..")
    assert cap.remote_operator_id("", "") == "op.host.unnamed.unnamed"


def test_an_unknown_declared_capability_is_refused():
    """`sensor.telepathy` is a valid open-family name (see `test_an_unregistered_name_is_refused`),
    so this uses `compute.telepathy`, which is outside the vocabulary in every direction."""
    s = _FakeStore()
    with pytest.raises(cap.UnknownCapability):
        cap.register_remote_host(s, name="gw", operators=[], capabilities=["compute.telepathy"])


def test_registration_invalidates_the_selection_cache():
    """A registration adds offers, and the next read of the offer table sees them.

    `register_remote_host` invalidates keyed on the artifacts store, while a reader may pass the
    bundle. Those are two cache keys over the same offers, so one `invalidate` clears both — without
    that, a freshly registered host is invisible until restart."""
    from ember.ontology import match
    from _fakes import _install_offline_wordnet
    if not _install_offline_wordnet():
        pytest.skip("WordNet not available")
    # Proven on ember's own surface, `match._offers` — the table `select` reads. The invariant is
    # ember's: a registration invalidates the offer table under both cache keys. Asserting it here
    # rather than through `sage/match.py`'s `select` keeps it free of a cross-repo dependency.
    s = _FakeStore()
    match.invalidate(s)
    before = match._offers(s)                # prime the cache under the bundle key (what a reader passes)
    cap.register_remote_host(s, name="gw", operators=["encyclopedia article"])
    after = match._offers(s)                 # bundle key again — must reflect the new registration
    assert after["n"] > before["n"], (
        f"the offer table was stale after a registration: {before['n']} → {after['n']} rows. "
        f"`_offers` normalises its cache key to `.artifacts` precisely so that one `invalidate` on "
        f"either the bundle or the bare store is sufficient; if this fails, that normalisation broke "
        f"and a freshly registered host is invisible until restart.")
