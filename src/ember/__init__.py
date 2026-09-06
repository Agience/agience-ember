"""Ember — an observer unit.

The leaf. The local touch — present where observation happens. Operationally Ember is the
local cache: it holds its own shards, answers from them when it can, and reaches the mesh
only on a miss. It is not a client of the platform; it is the platform present locally.

Two rules keep it what it is:

1. **Ember is the local cache.** Everything it does follows from being a cache with a socket.
2. **Ember connects to the platform rather than containing it.** Ember imports no `chorus` and
   no `lumen`: personas reach into ember, and ember reaches a persona over the ground plane —
   the mantle lattice and the wire — rather than by linking. That keeps the dependency graph
   acyclic and ember standalone-deployable. It is architecture, not licence; ember itself is
   AGPL-3.0-only, because it holds the entroptics instrument (`ember/optics.py`).
"""

# ── the BLAS thread pin — set before numpy is imported ───────────────────────────────────────────
# OpenBLAS sizes its worker pool when the library loads under `import numpy`, so the variable has to
# be set first: measured through `threadpoolctl`, unset gives 8 threads, set-before gives 1, and
# set-after gives 8. It therefore sits above the imports below, which reach `mantle.shard.cache`,
# and at package scope, because Python initialises a parent package before its submodules. That one
# placement covers ember's own BLAS callers (`signal/projection.svd`, `signal/forgetting.norm`) and
# the instrument: `import ember.optics` runs this file first, and `optics.pinv` is a LAPACK caller.
# The pin is per-package and is not inherited from an importer, so it lives with the callers.
#
# Why 1: two threads in `numpy.linalg.eigh` fault this box's OpenBLAS (exit 139) and can hang
# instead — 3 of 3 runs; pinned, 0 of 3. The measurement table is at `prism/pump.py::PumpLoop.tick`.
#
# `setdefault`, so an operator's exported value wins, including one that reinstates the fault. It
# also cannot help a process that imported numpy first. Both limits are pinned by
# `tests/test_blas_thread_pin.py`.
import os as _os

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
del _os


# ── the instrument registers itself as the host's instrument ───────────────────────────────────────
#
# Holding ember makes a process a host: this line registers `ember.optics` as the process-default
# instrument.
#
# Why the default exists. Two wire modules take a measurement — `prism.frames.absorb_at_tekton` (the
# membrane split) and `prism.reach.Provider._route_next` (which tekton couples next). The wire lives
# in `prism`, which carries no entroptics, so both resolve an injected instrument through
# `prism.instrument`: an explicit keyword first, then this process default, and with neither there
# is no reading to give. A caller's own `read=` / `dynamics=` / `conservation=` wins over the
# default.
#
# The registration is global, so a test file that measures relies on some file in its process having
# imported a host. That is why the `import ember` lines in the chorus and crystal suites carry
# `# noqa: F401` and a header naming them: they are host declarations, and a file run on its own
# needs its own.
#
# A factory, not the module. Registering `importlib.import_module("ember.optics")` eagerly would
# pull numpy and entroptics into a process that only wanted the cache or the runner; the factory
# runs on the first measurement.
def _register_instrument() -> None:
    import importlib

    try:
        from prism import instrument as _instrument
    except ImportError:                       # an ember checkout without prism on the path
        return
    _instrument.set_default(factory=lambda: importlib.import_module("ember.optics"))


_register_instrument()

# ── the host seams — bound before anything else, because a seam is in place before its first use
# (the same ordering rule `runner.attach(store)` follows). This costs no imports: `register_seam`
# stores dotted strings, so a process that never reaches a seam never loads the module behind it.
#
# It sits at package scope so that any path holding ember is a host. Persona modules declare the
# measurements they need by name instead of importing them, which is what keeps `chorus → ember` at
# zero; on a host that binds nothing, a declaration has no reading to give and says so. See
# `ember/runtime/seams.py` for the table.
from ember.runtime.seams import register_host_seams as _register_host_seams

_register_host_seams()

# ── the ontology's store — named by the host ─────────────────────────────────────────────────────
# The ontology coordinate lives in `crystal.ontology`, which sits below ember and beside mantle;
# crystal and mantle reach each other in neither direction, so the driver names no lattice of its
# own. A host knows what lattice it is running on, so the host supplies it.
#
# The driver resolves a store in order: an explicit `store=`, then `bind(store)`, then this default,
# with a once-per-process warning when the default is what answers.
#
# The provider is handed over as a callable and resolved on the first ambient read, so a process
# that always passes `store=` opens nothing, and `import ember` touches no disk on account of this
# line.
#
# With no default registered, an unbound read raises `OntologyStoreRequired`. That is the outcome to
# want: an empty ontology in its place would answer "this corpus does not contain that concept" for
# every concept, which is a reading nobody took ([[absence-is-not-an-affirmative-claim]]).
from crystal.ontology import driver as _ontology_driver
from mantle.shard.local_store import open_store as _open_store

_ontology_driver.set_default_store_provider(_open_store)

from ember.runtime.boot import Ember, NotInitialised
from mantle.shard.cache import Item, LocalCache, Routing
from ember.config import Settings, load
# Ember's surface carries the engine's factory and its no-answerer signal, and no answer composer:
# composing an answer is a persona's work and lives in `lumen/composers.py`. A runner advertises what
# it runs, not what answers.
from ember.runtime.engine import NoAnswerer, build
from mantle.shard.local_collection import (
    COLLECTION_CONTENT_TYPE, local_collection_artifact, local_collection_id,
)
from ember.runtime.read_path import Result, answer_query
from ember.runtime.relay import Channel, Disconnected, MeshChannel
from mantle.shard.store import ShardStore

__version__ = "0.1.0"

__all__ = [
    "Ember", "NotInitialised",
    "LocalCache", "Item", "Routing",
    "Settings", "load",
    "NoAnswerer", "build",
    "COLLECTION_CONTENT_TYPE", "local_collection_id", "local_collection_artifact",
    "answer_query", "Result",
    "Channel", "Disconnected", "MeshChannel",
    "ShardStore",
    "__version__",
]
