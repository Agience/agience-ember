"""Ember's host seams: the runner's answer to which module fills each name a persona declares.

`prism/runner.py` states the general rule: a bundle declares the host modules it may reach for
(`host_seams`), and the seam-to-module mapping is registered by the host (`register_seam`).
`ember/runtime/runner.py` states the half that matters here: which module fills a seam is the
host's answer, not the loader's. This module is that answer, written once.

Four modules are registered here because each is a measurement the runner performs — recognition,
screened propagation, the signal frame, per-delegate cognition — rather than a tool it hosts. Ember
reads all four itself (`genesis.py`, `runtime/capability.py`, `signal/signal.py`,
`surface/serve.py`); chorus holds the tools ember authenticates, energises and runs, so a persona
reaching into the runner would be backwards for a tool, but these four are measurements, not tools,
and a persona names a measurement by name — which is what a seam is for.

A seam is a named indirection, not an import dependency: registering one binds a dotted string,
resolved by the declarer at the point of use, so a process that never touches a seam never imports
the module behind it. A persona running on a host that registered nothing gets the absence its own
call site already reports (`basis="unavailable"`, `{"reach": "unavailable"}`, `_basis = None`),
never a silently substituted module.

Seams are registered at import of `ember`, so the seam is in place before first use — the same
ordering `attach(store)` follows. Binding in `ember/__init__.py` means any path that imports ember
at all has its seams bound; binding again in `ember/runtime/runner.py` is idempotent and keeps the
bundle path's behaviour unchanged.
"""
from __future__ import annotations

from typing import Dict

from prism.runner import register_seam

#: seam name (as a declarer spells it) → the dotted ember module that fills it.
#:
#: The keys are the declarer's vocabulary, not ember's file layout. `projection` is spelled that way
#: because `sage/content_search.py` asks for "the frame every cut and condensation sits on"; that it
#: currently lives at `ember/signal/projection.py` is this table's own business. Renaming the module
#: changes only the value here.
HOST_SEAMS: Dict[str, str] = {
    # the screened propagator + the per-store operator-OFFER table and its cache. `_offers` is
    # invalidated by `runtime/capability.register_operators`, so the cache cannot leave the runner.
    "match": "ember.ontology.match",
    # recognition: grounding text into activation state over the ontology coordinate.
    "activation": "ember.ontology.activation",
    # the (T,F) signal frame a cut and a condensation are both read off.
    "projection": "ember.signal.projection",
    # the working memory of a read: what has gone past, and — measured, never configured — how far
    # back a co-occurrence still reaches. `astra/reading/organon_reader.py` declares it because a reader
    # needs somewhere to accumulate; that the decay is measured rather than a typed window length is
    # exactly why it is ember's measurement and not the declarer's own dictionary.
    "forgetting": "ember.signal.forgetting",
    # the instrument itself. `test_only_the_instrument_seam_imports_entroptics` pins ember/optics.py as the
    # one module that may reach entroptics; a declarer that needs a read therefore needs a seam, or
    # it reaches past the wrapper — and the wrapper is what keeps the entropy fold guard from
    # destroying a sparse carrier. Reads are added to optics.py rather than worked around here.
    "optics": "ember.optics",
    # per-delegate cognition. `Delegate.get` is the per-process-per-person registry that exists to
    # guarantee there is never shared cognitive state — pooled witnesses turn first-hand memory into
    # hearsay — so it is the runner's by charter, and a persona must be given one.
    "delegate": "ember.runtime.delegate",
    # ── what a FACET reads off the running engine ───────────────────────────────────────────────
    #
    # Facets are chorus's; the engine is ember's. A facet that renders what this node holds needs to
    # read the engine, and it reaches it HERE rather than by importing ember — the same rule every
    # other declarer follows, and the one `agience-chorus/src/tests/test_chorus_does_not_import_ember.py`
    # enforces from the other side.
    #
    # These four were added when `ember/facets/browse.py` moved to `aria/facets/`. It had been
    # importing `ember.genesis`, `ember.runtime.improve`, `ember.surface.stats` and
    # `ember.runtime.pool` directly, which is what a facet living inside the engine lets you do.
    "genesis": "ember.genesis",
    "improve": "ember.runtime.improve",
    "stats": "ember.surface.stats",
    "pool": "ember.runtime.pool",
}


def register_host_seams() -> None:
    """Bind every seam ember fills. Idempotent (last-writer-wins per name, per `register_seam`),
    so calling it from both `ember/__init__.py` and `ember/runtime/runner.py` is free."""
    for name, target in HOST_SEAMS.items():
        register_seam(name, target)


__all__ = ["HOST_SEAMS", "register_host_seams"]
