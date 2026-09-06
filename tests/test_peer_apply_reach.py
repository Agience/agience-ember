"""Peer-apply reaches the persona.

`signal.deliver(..., forward=fn)` is the onward hop: after an arriving peer signal grounds locally
(the runner's recognition, which lives in ember), the same signal propagates onward to the persona act
via `forward(artifact, result)`. A host builds that callable from
`ember.runtime.reach.reactor(...).reach(need, to=…)` over a carrier.

The hop is inactive by default. With `forward=None` on a plain node, local grounding is the whole
story: grounding is extended by the hop rather than replaced by it, so an arriving activation lands
the same way whether or not a persona is wired in.

Invariants:

  Backward-compatible: `forward=None` leaves delivery byte-identical (`forwarded == 0`), so
  plain-node behavior is a property of the seam, not of the host that skipped it.

  Grounded only: a below-salience arrival that grounded nothing stays local; only fired or
  transformed signals hop on, so the persona sees what the node itself heard.

  Fire-and-forget: a raising `forward` leaves local delivery intact, so a down persona does not
  stall the mesh.

  Real reach: wired to a `StoreCarrier`-backed `Reactor`, the onward hop places a real NEED
  artifact on the shared lattice ground.
"""
from __future__ import annotations

from ember.signal import signal
from ember.signal.signal import SIGNAL_CONTENT_TYPE


class _Delegate:
    def __init__(self, id: str) -> None:
        self.id = id
        self.store = None


def _sig(to: str, from_observer: str, sid: str) -> dict:
    return {"content_type": SIGNAL_CONTENT_TYPE, "to": to,
            "from_": {"observer": from_observer}, "id": sid}


def test_forward_none_is_backward_compatible(monkeypatch):
    monkeypatch.setattr(signal, "route",
                        lambda d, a, store=None: {"regime": "message", "delivered": True, "fired": True})
    out = signal.deliver(_Delegate("obs-1"), [_sig("obs-1", "peer-2", "s1")])
    assert out["delivered"] == 1
    assert out.get("forwarded", 0) == 0                  # no hook → nothing propagates onward


def test_forward_called_only_for_grounded(monkeypatch):
    results = {"s1": {"delivered": True, "fired": True},          # grounded → forwards
               "s2": {"delivered": True, "fired": False}}         # heard, below salience → stays local
    monkeypatch.setattr(signal, "route", lambda d, a, store=None: {"regime": "message", **results[a["id"]]})
    seen: list = []
    out = signal.deliver(_Delegate("obs-1"),
                         [_sig("obs-1", "p", "s1"), _sig("obs-1", "p", "s2")],
                         forward=lambda art, r: seen.append(art["id"]))
    assert out["forwarded"] == 1
    assert seen == ["s1"]


def test_forward_is_fire_and_forget(monkeypatch):
    monkeypatch.setattr(signal, "route", lambda d, a, store=None: {"delivered": True, "transforms": True})

    def boom(art, r):
        raise RuntimeError("persona down")

    out = signal.deliver(_Delegate("obs-1"), [_sig("obs-1", "p", "s1")], forward=boom)  # must not raise
    assert out["delivered"] == 1
    assert out["forwarded"] == 0                          # the raise is swallowed; delivery unharmed


def test_forward_places_a_real_need_on_the_ground(tmp_path, monkeypatch):
    from prism.carriers import CARRIER_LEAF_CT, StoreCarrier
    from prism.plane import HLC, Keyring, Lightcone
    from prism.reach import NEED_CT, Reactor
    from mantle.db import open_lattice

    LEARN, GROUND = "op.learn", "ground"
    L = open_lattice(str(tmp_path / "ground.db"), origin="node-71")
    L.artifacts.ensure_schema()
    carrier = StoreCarrier(L)
    reacher = Reactor("obs-1", keyring=Keyring(b"fleet-root"), lightcone=Lightcone().join("obs-1", LEARN),
                      fabric=None, fallback=carrier, hlc=HLC("obs-1"), ground=GROUND)

    monkeypatch.setattr(signal, "route", lambda d, a, store=None: {"delivered": True, "fired": True})
    signal.deliver(_Delegate("obs-1"), [_sig("obs-1", "peer-2", "s1")],
                   forward=lambda art, r: reacher.reach({"apply": art["id"]}, to=LEARN))

    leaves = list(L.artifacts.list_artifacts(content_type=CARRIER_LEAF_CT))
    kinds = [x["leaf"]["content_type"] for x in leaves]
    assert NEED_CT in kinds                              # peer-apply reached: a real NEED to op.learn landed
