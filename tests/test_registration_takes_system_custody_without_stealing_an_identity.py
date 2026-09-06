"""Operator registration runs under the platform system principal — unless a caller already has one.

Operator artifacts must seal at write time, scoped as "one change at mantle's write chokepoint, no
chorus or ember change needed". Measured 2026-08-26: false. Sealing calls
`content_crypto.encrypt_content`, which "requires an acting principal in scope";
`require_acting_principal` raises `NoActingPrincipal` and names the remedy — "background work must
declare one explicitly (see `system_acting_context`)". The entity path works because it runs
inside a request. Operator registration is used by exactly the things that do not, and ember uses
`system_acting_context` nowhere else: zero other occurrences.

And the failure mode is worse than a crash: `runtime/pool.py` calls the registrars inside
`try: … except Exception: pass`, so a seal that raises does not fail loudly — it makes operator
registration silently stop, converting a visible plaintext defect into an invisible capability
loss. That swallow now reports.

The guard is the half that matters. `register_control_operators`'s own docstring says it "runs on
every `invoke`, and therefore on every task the work pool executes" — which includes request work
that already carries a principal. A plain `with system_acting_context(...)` there would replace a
user's identity with the platform system principal for the rest of that call: work done for a
person, authorized as the platform. So custody is established only when none is in scope, and
`test_a_caller_with_an_identity_keeps_it` is that asserted.
"""
from __future__ import annotations


from ember import custody


def _declare(principal_id="user-1"):
    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal
    return set_acting_principal(ActingPrincipal(
        principal_id=principal_id, principal_type="principal", source="test"))


# ── the guard ────────────────────────────────────────────────────────────────────────────────────

def test_a_caller_with_an_identity_keeps_it():
    """The privilege-escalation shape this refuses. Registration is reachable from `invoke`, so
    a request path passes through it. Taking system custody there would authorize the rest of that
    call as the platform."""
    from mantle.services.acting_principal import current_acting_principal, reset_acting_principal

    token = _declare("user-1")
    try:
        with custody.system_custody_if_unowned("test.scope"):
            inside = current_acting_principal()
        assert inside is not None, "the caller's identity was cleared"
        assert inside.principal_id == "user-1", (
            "registration replaced the caller's identity with %r — work done for a person would be "
            "authorized as the platform for the rest of the call" % inside.principal_id)
    finally:
        reset_acting_principal(token)


def test_the_callers_identity_survives_the_block():
    """A context manager that leaked would corrupt the request that called it."""
    from mantle.services.acting_principal import current_acting_principal, reset_acting_principal

    token = _declare("user-2")
    try:
        with custody.system_custody_if_unowned("test.scope"):
            pass
        after = current_acting_principal()
        assert after is not None and after.principal_id == "user-2", after
    finally:
        reset_acting_principal(token)


def test_an_unowned_path_gets_custody_or_a_stated_reason():
    """The background case — the one ruling 2 needs. Either a system principal is established, or
    the block runs uncustodied because the instance namespace does not resolve. Both are acceptable;
    silently doing neither while claiming to is not.

    Which one happens depends on whether this box can mint the system principal, so the assertion
    is on the invariant (no exception escapes, and nothing else's identity is left behind), not on
    the branch — a test that demanded one branch would be asserting this machine's configuration.
    """
    from mantle.services.acting_principal import current_acting_principal

    assert current_acting_principal() is None, "a previous test leaked an identity"
    with custody.system_custody_if_unowned("test.scope") as principal:
        inside = current_acting_principal()
    if principal is not None:
        assert inside is not None, "a principal was yielded but never installed"
    assert current_acting_principal() is None, "custody leaked out of the block"


def test_a_missing_system_principal_does_not_take_the_node_down(monkeypatch):
    """Fails soft, and only here. `system_acting_context` raises `SystemPrincipalUnavailable`
    where the namespace does not resolve. Propagating it would kill a node at startup over a
    registration that has always been best-effort.

    This is not a fail-open in the sealing sense: a write that needs custody still refuses, at the
    write, loudly. What is refused is turning a missing namespace into a dead node."""
    from mantle.services import system_identity
    from mantle.services.acting_principal import SystemPrincipalUnavailable

    def _boom(*a, **k):
        raise SystemPrincipalUnavailable("no instance namespace")

    monkeypatch.setattr(system_identity, "system_acting_context", _boom)
    custody._warned = False
    with custody.system_custody_if_unowned("test.scope") as principal:
        assert principal is None
    assert custody._warned, "the node ran uncustodied and said nothing"


# ── the wiring ───────────────────────────────────────────────────────────────────────────────────

def test_both_registrars_establish_custody():
    """At the two registrars, not at the seven call sites — `genesis.py` (×5), `runtime/pool.py`
    and `runtime/worker.py` all reach them, and a per-call-site wrap is seven chances to miss one."""
    import inspect

    from ember import genesis

    for fn in (genesis.register_control_operators, genesis.register_consolidate_operators):
        src = inspect.getsource(fn)
        assert "system_custody_if_unowned" in src, (
            "%s does not establish key custody, so sealing its inline `content` would raise "
            "NoActingPrincipal on every background path" % fn.__name__)


def test_the_pool_no_longer_swallows_a_registration_failure_silently():
    """The silence is the defect, not the catch. The lost-upsert race between workers is real and
    a lost one is harmless — but a bare `pass` cannot tell that from a node that has stopped
    registering its operators entirely."""
    import ast
    import inspect
    import textwrap

    from ember.runtime import pool

    src = inspect.getsource(pool.pool_worker)
    assert "register_operators_failed" in src, (
        "the pool's registration failure is silent again; a seal that raised would make operator "
        "registration stop happening with nothing said")

    # Scoped to the registration `try`, not the whole function: asserting that no bare
    # `except: pass` appears anywhere in `pool_worker` would also match an unrelated one — a true
    # statement about the wrong region. The claim is about this catch, so this is the one that
    # gets read.
    tree = ast.parse(textwrap.dedent(src))
    blocks = [n for n in ast.walk(tree) if isinstance(n, ast.Try)
              and "register_control_operators" in ast.unparse(ast.Module(body=n.body, type_ignores=[]))]
    assert len(blocks) == 1, "expected exactly one registration try/except, found %d" % len(blocks)
    for handler in blocks[0].handlers:
        body = handler.body
        assert not (len(body) == 1 and isinstance(body[0], ast.Pass)), (
            "the registration failure is a bare `pass` again — a lost upsert race and a node that "
            "has stopped registering its operators are indistinguishable from silence")


def test_custody_sits_at_the_registrar_except_where_the_registrar_is_bundled():
    """Custody belongs at the registrar — with one exception, and the exception is the bundles.

    Ember's own registrars (`genesis.register_control_operators` /
    `register_consolidate_operators`) are wrapped inside themselves, so all seven of their callers
    are covered by one rule and `runtime/pool.py` needs nothing.

    The four chorus registrars cannot be: `arithmetic.register_transform_operators`,
    `operators.register_operators`, `dev_ops.register_dev_operators` and
    `fetch.register_fetch_operators` live in chorus and ship as bundles — the runtime reads the
    payload, not the source file — so wrapping them at their own definition would be inert until
    `build_bundles.py` rebuilt them. Those four write every one of the 25 rows
    `artifacts_holding_inline_plaintext` reports. Their caller is `runtime/worker.py`, which is an
    ember file and is not bundled, so that is where custody has to go.
    """
    import inspect

    from ember.runtime import pool, worker

    assert "system_custody_if_unowned" not in inspect.getsource(pool), (
        "runtime/pool.py establishes custody itself; it calls only EMBER's registrars, which wrap "
        "themselves, so this would be a second place for one rule")
    assert "system_custody_if_unowned" in inspect.getsource(worker), (
        "runtime/worker.py no longer establishes custody around the CHORUS registrars — those four "
        "cannot wrap themselves (bundles), so without this the 25 operator rows have no custody at "
        "the only place it can be given")
