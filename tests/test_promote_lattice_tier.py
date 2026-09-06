"""The promote drain over the lattice tier: local write → shared-cipher mirror → working-set evict.

Pins `promote_local_content` against `store.content_tier` (mantle `TieredContentStore.promote_one` /
`evict_local` / `evict_for_space`): promotion re-encrypts under the shared cipher so the mirror is
fleet-readable, eviction only ever removes a copy the remote has confirmed, and the working-set
floor is operator-owned (`EMBER_CACHE_MIN_FREE_GB`) — unset means nothing is evicted beyond
promote's own. The `.local`/`.remote` drain path is covered by `test_drain_invariants.py`; this
file is its lattice twin."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

mt = pytest.importorskip("mantle.db.content_tier")
from mantle.db.content_cache import FileContentCache          # noqa: E402

from mantle.shard import content_tier as CT                                  # noqa: E402


class _FakeRemote:
    def __init__(self):
        self.objects = {}

    def put(self, key, data, content_type="application/octet-stream"):
        self.objects[key] = data

    def get(self, key):
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]

    def exists(self, key):
        return key in self.objects

    def delete(self, key):
        self.objects.pop(key, None)


class _FakeArts:
    """The typed lattice walk surface the drain uses: page_by_id / page_by_origin + the cursor."""

    def __init__(self, rows):
        self.rows = rows                 # dicts with id, content_ref, _seq, doc
        self.docs = {}

    def page_by_id(self, *, after="", limit=200, content_type=None):
        out = sorted((r for r in self.rows if r["id"] > after), key=lambda r: r["id"])
        return out[:limit]

    def page_by_origin(self, *, origin=None, after_seq=0, limit=200):
        out = sorted((r for r in self.rows if r["_seq"] > after_seq), key=lambda r: r["_seq"])
        return out[:limit]

    def get_artifact(self, aid):
        return self.docs.get(aid)

    def put_artifact(self, doc, **kw):
        self.docs[doc["id"]] = doc
        return doc


def _store(tmp_path, cipher, texts):
    cache = FileContentCache(str(tmp_path / "cas"), key=b"k" * 32)
    tier = mt.TieredContentStore(cache, _FakeRemote(),
                                 decrypt=cipher.decrypt, encrypt=cipher.encrypt)
    rows = []
    for i, t in enumerate(texts):
        ref = "cas/" + hashlib.sha256(t).hexdigest()
        tier.put(ref, t, collection="c1")                      # local write — nothing goes up yet
        rows.append({"id": "a%04d" % i, "content_ref": ref, "_seq": i + 1,
                     "doc": {"collection_id": "c1"}})
    rows.append({"id": "a9999", "content_ref": None, "_seq": 99, "doc": {}})   # no-content row
    return SimpleNamespace(content=cache, content_tier=tier,
                           artifacts=_FakeArts(rows), keys_dir=None), tier


@pytest.fixture()
def cipher():
    from cryptography.fernet import Fernet
    return Fernet(Fernet.generate_key())


def test_drain_promotes_reencrypts_shared_and_evicts(tmp_path, cipher, monkeypatch):
    monkeypatch.delenv("EMBER_PROMOTE_NO_EVICT", raising=False)
    monkeypatch.delenv("EMBER_CACHE_MIN_FREE_GB", raising=False)
    texts = [b"alpha content", b"beta content", b"gamma content"]
    st, tier = _store(tmp_path, cipher, texts)
    res = CT.promote_local_content(st, max_refs=50, page=10, workers=2)
    assert res["errors"] == 0 and res["walk_error"] is None
    assert res["promoted"] == 3
    for t in texts:
        ref = "cas/" + hashlib.sha256(t).hexdigest()
        assert cipher.decrypt(tier.remote.objects[ref]) == t   # shared cipher: fleet-readable
        assert ref not in tier.cache                           # evicted only after confirmed
        assert tier.get(ref, collection="c1") == t             # ...and pulls straight back
    cur = st.artifacts.docs["content.promote.cursor"]
    # Cycle 1 is the one-time id backfill, and a short page ends the pass without latching
    # `id_backfill_done`: only an empty page retires the backfill, because empty-because-finished
    # and empty-because-short are different facts (see the `_backfill_page` docstring). Cycle 2
    # reads the empty page, latches the backfill, and runs the seq walk proper — everything skips
    # as already mirrored and the cursor lands at the tail.
    assert res["walk"] == "seq" and cur["id_backfill_done"] is False
    res2 = CT.promote_local_content(st, max_refs=50, page=10, workers=2)
    assert res2["errors"] == 0 and res2["promoted"] == 0
    cur2 = st.artifacts.docs["content.promote.cursor"]
    assert cur2["id_backfill_done"] is True and cur2["seq"] == 99


def test_no_evict_keeps_the_local_working_set(tmp_path, cipher, monkeypatch):
    monkeypatch.setenv("EMBER_PROMOTE_NO_EVICT", "1")
    monkeypatch.delenv("EMBER_CACHE_MIN_FREE_GB", raising=False)
    texts = [b"retained one", b"retained two"]
    st, tier = _store(tmp_path, cipher, texts)
    res = CT.promote_local_content(st, max_refs=50, page=10, workers=2)
    assert res["promoted"] == 2 and res["errors"] == 0
    for t in texts:
        ref = "cas/" + hashlib.sha256(t).hexdigest()
        assert tier.remote.exists(ref)                         # promoted...
        assert ref in tier.cache                               # ...and retained locally


def test_operator_floor_drives_working_set_eviction(tmp_path, cipher, monkeypatch):
    monkeypatch.setenv("EMBER_PROMOTE_NO_EVICT", "1")          # promote's own evict off...
    monkeypatch.setenv("EMBER_CACHE_MIN_FREE_GB", "99999999")  # ...but the floor is unsatisfiable
    texts = [b"floor-evicted content"]
    st, tier = _store(tmp_path, cipher, texts)
    res = CT.promote_local_content(st, max_refs=50, page=10, workers=2)
    ref = "cas/" + hashlib.sha256(texts[0]).hexdigest()
    assert tier.remote.exists(ref)
    assert res["recache_evicted"] == 1                         # evict_for_space did the bounding
    assert ref not in tier.cache
    # And the second cycle is a clean steady state: everything skips, nothing errors.
    res2 = CT.promote_local_content(st, max_refs=50, page=10, workers=2)
    assert res2["errors"] == 0 and res2["promoted"] == 0
