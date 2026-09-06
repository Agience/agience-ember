"""A missing content key is a fault, and is answered as one rather than by minting a key.

`_content_key` is the single entry point for `put_content`, `get_content` and `sync._fernet`. It
raises `ContentKeyMissing` when `content.key` is absent on a read path, and when the keys directory
itself does not exist — an unmounted keys volume is a different state from a fresh node.

Generating a key in either case partitions the node with nothing to see: the pending decrypt fails,
`resolve_text` returns "", every subsequent write is encrypted under a key no peer holds, published
mesh segments become undecryptable fleet-wide, and row counts, keyed_coverage and rho all stay
healthy throughout. The project's `content-encryption` note calls that "a silent partition". These
tests pin the behaviours that keep the key fault visible instead.
"""
import tempfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from mantle.shard import content as C


@pytest.fixture(autouse=True)
def _clear_key_cache():
    C._KEY_CACHE.clear()
    yield
    C._KEY_CACHE.clear()


def test_a_read_never_mints_a_key():
    """A read only happens when content already exists, so a missing key on the read path is
    always a fault — and a freshly minted key would fail the pending decrypt anyway."""
    d = Path(tempfile.mkdtemp())
    with pytest.raises(C.ContentKeyMissing):
        C._content_key(d)
    assert not (d / "content.key").exists(), "a read created a key file"


def test_an_unmounted_volume_is_an_error_even_on_the_write_path():
    """`keys_dir` not existing means the volume is not mounted, so the write path reports the fault
    instead of creating the directory — creating it would make an unmounted volume look like a
    fresh node."""
    d = Path(tempfile.mkdtemp()) / "not-mounted"
    with pytest.raises(C.ContentKeyMissing):
        C._content_key(d, create=True)
    assert not d.exists(), "the keys directory was created for an unmounted volume"


def test_a_provisioned_but_empty_dir_may_bootstrap_on_write():
    """The positive control: a genuinely new node, with the keys volume mounted and empty, still
    bootstraps its key on the first write."""
    d = Path(tempfile.mkdtemp())
    mf = C._content_key(d, create=True)
    assert (d / "content.key").exists()
    assert mf.decrypt(mf.encrypt(b"hello")) == b"hello"


def test_an_existing_key_is_never_replaced():
    """Overwriting a live key would orphan every blob already written under it."""
    d = Path(tempfile.mkdtemp())
    original = Fernet.generate_key()
    (d / "content.key").write_bytes(original)
    C._content_key(d, create=True)
    assert (d / "content.key").read_bytes() == original


def test_resolve_text_does_not_hide_a_missing_key():
    """A node-wide key fault must not read as "this artifact is empty".

    Everything else `resolve_text` catches is per-artifact (blob evicted, stale ref) where the
    inline fallback is the right answer. This one is a configuration fault, and swallowing it makes
    the entire corpus resolve to "" with no error anywhere.
    """
    class _Store:
        def get(self, ref):
            raise AssertionError("should not be reached — the key check fails first")

    class _Bundle:
        content = _Store()
        keys_dir = Path(tempfile.mkdtemp())

    with pytest.raises(C.ContentKeyMissing):
        C.resolve_text(_Bundle(), {"content_ref": "cas/deadbeef"})


def test_resolve_text_still_falls_back_for_ordinary_per_artifact_failures():
    """Ordinary per-artifact failures — an evicted blob, a stale ref — still fall back to inline
    content. Only the key fault propagates."""
    d = Path(tempfile.mkdtemp())
    (d / "content.key").write_bytes(Fernet.generate_key())

    class _Store:
        def get(self, ref):
            raise FileNotFoundError(ref)          # blob evicted — not a key fault

    class _Bundle:
        content = _Store()
        keys_dir = d

    assert C.resolve_text(_Bundle(), {"content_ref": "cas/x", "content": "inline"}) == "inline"
