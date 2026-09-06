"""The read-seam contract: `resolve_text` prefers the tiered store (local ⊕ S3/CDN).

Pins `LocalStore.content_tier` and the tier-first branch in `content.resolve_text`: a cold artifact
is pulled through the mirror with a sha256 verify, a miss falls back to inline content rather than
to b"" served as text, and `ContentKeyMissing` propagates so a node-wide key fault does not read as
an empty corpus. Runs inside the ember suite (mantle on PYTHONPATH, both path styles)."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

mt = pytest.importorskip("mantle.db.content_tier")
from mantle.db.content_cache import FileContentCache          # noqa: E402

from mantle.shard.content import resolve_text                                # noqa: E402


class _FakeRemote:
    def __init__(self):
        self.objects, self.gets = {}, 0

    def put(self, key, data, content_type="application/octet-stream"):
        self.objects[key] = data

    def get(self, key):
        self.gets += 1
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]

    def exists(self, key):
        return key in self.objects

    def delete(self, key):
        self.objects.pop(key, None)


def _bundle(tmp_path, cipher):
    cache = FileContentCache(str(tmp_path / "cas"), key=b"k" * 32)
    tier = mt.TieredContentStore(cache, _FakeRemote(),
                                 decrypt=cipher.decrypt, encrypt=cipher.encrypt)
    # resolve_text only touches .content / .content_tier / .keys_dir on the bundle.
    return SimpleNamespace(content=cache, content_tier=tier, keys_dir=None), tier


@pytest.fixture()
def cipher():
    from cryptography.fernet import Fernet
    return Fernet(Fernet.generate_key())


def test_cold_artifact_pulls_through_the_mirror(tmp_path, cipher):
    bundle, tier = _bundle(tmp_path, cipher)
    text = "the CDN substrate serving a cold read"
    ref = "cas/" + hashlib.sha256(text.encode()).hexdigest()
    tier.remote.put(ref, cipher.encrypt(text.encode()))       # only the mirror holds it
    art = {"content_ref": ref, "collection_id": "c1"}
    assert resolve_text(bundle, art) == text                  # pulled, verified, served
    assert resolve_text(bundle, art) == text                  # now local
    assert tier.remote.gets == 1                              # the mirror was hit exactly once


def test_miss_everywhere_falls_back_to_inline_never_wrong(tmp_path, cipher):
    bundle, _ = _bundle(tmp_path, cipher)
    art = {"content_ref": "cas/" + "0" * 64, "collection_id": "c1", "content": "inline fallback"}
    assert resolve_text(bundle, art) == "inline fallback"


def test_content_key_missing_propagates_through_the_seam(tmp_path):
    def decrypt(_):
        raise mt.ContentKeyMissing("no content.key")
    cache = FileContentCache(str(tmp_path / "cas"), key=b"k" * 32)
    remote = _FakeRemote()
    ref = "cas/" + "a" * 64
    remote.put(ref, b"ciphertext-ish")
    tier = mt.TieredContentStore(cache, remote, decrypt=decrypt)
    bundle = SimpleNamespace(content=cache, content_tier=tier, keys_dir=None)
    with pytest.raises(mt.ContentKeyMissing):                 # config fault ≠ empty artifact
        resolve_text(bundle, {"content_ref": ref, "collection_id": "c1"})
