"""Key custody for ember's system-initiated work.

Operator artifacts must seal at write time (`artifacts_holding_inline_plaintext` stays pinned at
0). The scoping said that was one change at mantle's write chokepoint, needing no chorus or ember
change at all. Measured 2026-08-26: that is false, and the reason is custody.

Sealing calls `doc_boundary.encrypt_artifact_content` -> `content_crypto.encrypt_content`, whose own
docstring says it "requires an acting principal in scope", and `require_acting_principal` raises
`NoActingPrincipal` naming the remedy: "Request paths get one from the auth dependency; background
work must declare one explicitly (see `system_acting_context`)."

The entity path works because it runs inside a request. The raw-dict path — operator registration
— is used by exactly the things that do not. And ember uses `system_acting_context` nowhere else:
zero other occurrences across the repo.

The failure mode is worse than a crash: `runtime/pool.py` calls the registrars inside
`try: … except Exception: pass`, so a seal that raises does not fail loudly — it makes operator
registration silently stop happening, converting a visible plaintext defect into an invisible
capability loss.

And wrapping the registrars unconditionally would be worse still. `register_control_operators`'s
own docstring says it "runs on every `invoke`, and therefore on every task the work pool
executes" — which includes request-driven work that already has a principal. Establishing system
custody there would replace a user's identity with the platform system principal for the rest of
that call: work done for a person, authorized as the platform. So custody is established only when
none is already in scope, which is what :func:`system_custody_if_unowned` means and the only shape
that is safe on a path shared by background and request work.
"""
from __future__ import annotations

import contextlib
import logging

log = logging.getLogger(__name__)

#: Said once, not per call. A node with no resolvable instance namespace cannot mint the system
#: principal, and registration then runs uncustodied — correct today, and the thing to look at the
#: moment a write starts refusing.
_warned = False

#: Whether `init_encryption_key` has been attempted in this process. Once, success or not — a
#: retry per call would re-read the key file on every operator registration.
_key_ready = False


def _ensure_platform_key() -> None:
    """Initialise the platform encryption key in this process, once.

    Without it the key oracle does not exist here: `content_crypto._default_master_key` calls
    `wiring._build_oracle`, which returns None with "Encryption key not initialized — call
    `init_encryption_key` at startup", and every seal then fails as
    `ContentEncryptionError: content encryption unavailable`. `init_encryption_key` is called in
    mantle's process only (`main.py:237`) and by mantle's own maintenance scripts; ember has
    never called it.

    This is not a new exposure, which is the part worth writing down. It looks like making the
    local leaf a custodian of the platform KEK — a security-architecture change — and it is not.
    Measured 2026-08-26 on 71/home: `service_common.sh:252` exports `KEYS_DIR` to every service
    on the node, ember included, and that directory holds `encryption.key`. Ember already
    possesses the key by configuration; what it lacked was the one call that lets it use what it
    was already given. Ember has no store of its own — mantle is ember's store — and
    `EMBER_SQLITE_DIR` and `MANTLE_LATTICE_PATH` are the same file, opened through mantle's own
    `open_store`: the SQLite lattice + FS content, via mantle, the one data path. The split is
    between processes, never between trust domains.

    Best effort. A node with no `encryption.key` (pre-setup, or a leaf that genuinely holds none)
    must still start; the seal then refuses at the write, loudly, which is the correct failure.
    """
    global _key_ready
    if _key_ready:
        return
    _key_ready = True                       # once per process, success or not
    try:
        from prism.trust.key_manager import init_encryption_key
        init_encryption_key()
    except Exception as exc:
        log.warning(
            "the platform encryption key is not available in this process (%s); content that must "
            "be sealed will refuse at the write rather than be stored in the clear", exc)


@contextlib.contextmanager
def system_custody_if_unowned(scope: str):
    """Run the block under the platform system principal, **unless one is already in scope**.

    The guard is the point: `current_acting_principal` is the non-raising accessor; when it
    answers, this yields immediately and changes nothing. A request that reaches here keeps the
    caller's identity, so nothing gains reach by passing through operator registration.

    Fails soft, deliberately, and only here: `system_acting_context` raises
    `SystemPrincipalUnavailable` where the instance namespace does not resolve. Propagating that
    would take a node down at startup over a registration that has always been best-effort — so it
    is logged once and the block runs uncustodied. That is not a fail-open in the sealing sense: a
    write that then needs custody still refuses, loudly, at the write. What this refuses to do is
    turn a missing namespace into a dead node.
    """
    global _warned
    try:
        from mantle.services.acting_principal import current_acting_principal
    except Exception:                       # mantle absent: nothing to establish custody with
        yield None
        return

    _ensure_platform_key()

    if current_acting_principal() is not None:
        # A request path. Its identity is the one that must stand.
        yield None
        return

    try:
        from mantle.services.acting_principal import SystemPrincipalUnavailable
        from mantle.services.system_identity import system_acting_context
    except Exception:
        yield None
        return

    try:
        with system_acting_context(scope=scope) as principal:
            yield principal
        return
    except SystemPrincipalUnavailable as exc:
        if not _warned:
            _warned = True
            log.warning(
                "no system principal is available for %s, so this work runs without key custody "
                "(%s). Anything on this path that must seal content will refuse at the write "
                "rather than store plaintext.", scope, exc)
    yield None
