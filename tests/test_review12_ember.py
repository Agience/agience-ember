"""Review 12, Ember: Ember's content mirror is never Mantle's content bucket.

Ember's own store mirrors its corpus content to an object store (`mantle.shard.content_tier`,
selected by `ovh.access_key`/`ovh.secret_key` in Ember's keys folder and `EMBER_OVH_BUCKET`). In
production that bucket was `agience-genesis-content`, which is also Mantle's durable content bucket
(`CONTENT_DURABLE_BUCKET`), and its default when the setting is absent. A credential for it would
let Ember list, read and overwrite Mantle's stored content. Ember reaches Mantle by its API only, so
its mirror must be a bucket of its own: one named explicitly, and never one Mantle uses.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

MANTLE_BUCKET = "agience-genesis-content"


class _S3:
    def __init__(self, bucket):
        self.bucket = bucket


def _store(bucket):
    return SimpleNamespace(content=SimpleNamespace(remote=_S3(bucket)), keys_dir=None)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("EMBER_OVH_BUCKET", "EMBER_FORBIDDEN_BUCKETS", "CONTENT_DURABLE_BUCKET",
              "CONTENT_EDGE_BUCKET"):
        monkeypatch.delenv(k, raising=False)


def test_a_mirror_on_mantle_s_content_bucket_is_refused(monkeypatch):
    from ember import mantle_api
    monkeypatch.setenv("EMBER_OVH_BUCKET", MANTLE_BUCKET)
    monkeypatch.setenv("EMBER_FORBIDDEN_BUCKETS", MANTLE_BUCKET)
    with pytest.raises(RuntimeError, match="Mantle's"):
        mantle_api.refuse_mantle_bucket(_store(MANTLE_BUCKET))


def test_a_mirror_on_the_default_bucket_is_refused_even_unnamed():
    """The mirror's default bucket is Mantle's; a mirror Ember did not name is refused."""
    from ember import mantle_api
    with pytest.raises(RuntimeError, match="names no bucket of its own"):
        mantle_api.refuse_mantle_bucket(_store(MANTLE_BUCKET))


def test_mantle_s_own_bucket_settings_are_forbidden_too(monkeypatch):
    from ember import mantle_api
    monkeypatch.setenv("EMBER_OVH_BUCKET", "shared-one")
    monkeypatch.setenv("CONTENT_DURABLE_BUCKET", "shared-one")
    with pytest.raises(RuntimeError, match="Mantle's"):
        mantle_api.refuse_mantle_bucket(_store("shared-one"))


def test_a_bucket_of_ember_s_own_is_accepted_and_no_mirror_is_fine(monkeypatch):
    from ember import mantle_api
    monkeypatch.setenv("EMBER_OVH_BUCKET", "agience-ember-corpus")
    monkeypatch.setenv("EMBER_FORBIDDEN_BUCKETS", MANTLE_BUCKET)
    mantle_api.refuse_mantle_bucket(_store("agience-ember-corpus"))
    mantle_api.refuse_mantle_bucket(SimpleNamespace(content=SimpleNamespace(remote=None)))
    mantle_api.refuse_mantle_bucket(SimpleNamespace(content=None))


def test_every_store_open_checks_the_mirror(monkeypatch, tmp_path):
    from ember import mantle_api
    seen = []
    monkeypatch.setattr(mantle_api, "refuse_mantle_store", lambda: None)
    monkeypatch.setattr(mantle_api, "refuse_opened_store", lambda p: None)
    monkeypatch.setattr(mantle_api, "ember_store_file", lambda: str(tmp_path / "x.db"))
    monkeypatch.setattr(mantle_api, "refuse_mantle_bucket", lambda s: seen.append(s))
    import mantle.shard.local_store as ls
    sentinel = object()
    monkeypatch.setattr(ls, "open_store", lambda *a, **k: sentinel)
    assert mantle_api.open_own_store() is sentinel and seen == [sentinel]


def test_the_mesh_stats_fallback_checks_the_mirror(monkeypatch):
    from ember import mantle_api
    from ember.surface import stats
    import mantle.shard.content_tier as ct
    monkeypatch.setattr(ct, "open_ovh_store", lambda kd: _S3(MANTLE_BUCKET))
    store = SimpleNamespace(content=SimpleNamespace(remote=None), keys_dir="k")
    assert stats._mesh_remote(store) is None                 # refused, not used
    monkeypatch.setenv("EMBER_OVH_BUCKET", "agience-ember-corpus")
    monkeypatch.setattr(ct, "open_ovh_store", lambda kd: _S3("agience-ember-corpus"))
    assert stats._mesh_remote(store).bucket == "agience-ember-corpus"
    assert mantle_api  # imported for the module under test
