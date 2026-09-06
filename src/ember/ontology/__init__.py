"""The ontology, ember's half — `activation` and `match`, the two modules that need the signal.

The coordinate itself is `crystal.ontology`: `driver`, `geometry`, `lookup`, `freshness`,
`transducer`, `coupling`, `seed_lattice`. crystal sits below both L3 consumers, so ember and chorus
each read the coordinate without reaching the other, and an ontology reader is not obliged to become
a mantle reader. The driver takes its store by injection — `store=`, `bind()`, or the provider
ember's package `__init__` registers — so it names no lattice of its own.

What lives here is what reaches the instrument as well as the store. `activation` and `match` need
both prism's measurement modules (`law`, `resolution`, `propagation`, `frames`, `vector`) and the
lattice, and ember is the layer that holds both. mantle reaches only origin and prism, so
measurement over the store belongs above it.

`corpus_stats` lives here too — a reading taken off the lexical index (`ember.corpus.fts`) rather
than a coordinate, vendored out of `mantle.ontology` when mantle shed the code that was not database
code. Its only readers are `activation` and the wiring below, both inside this package, so it costs
an importer of `ember` nothing.

Its former neighbour `embed` was vendored to `ember/embed.py` instead of here, for exactly the
reason the wirings below exist: `ember/runtime/boot.py` imports the embedder at module scope and
`ember/__init__.py` imports boot, so filing it in this package would run this `__init__` — and its
import of `match`, a seam target — on every `import ember`.

The two wirings at the bottom of this file are the reason this `__init__` carries code.
"""

# `match` takes its corpus-statistics provider by injection rather than importing `corpus_stats`.
# Unwired, salience weighting goes uniform, and three `test_match` assertions pin that difference —
# so the wiring is load-bearing and sits here, in view.
from ember.ontology import corpus_stats as _corpus_stats   # noqa: E402
from ember.ontology import match as _match                 # noqa: E402

_match.set_corpus_stats(_corpus_stats)

# The same shape, for the lattice writer. `crystal.ontology.seed_lattice` writes the substrate
# through a store it is handed and names none. The `xi`/`gap` it stamps on the transducer come from
# a derivation that needs the store and the instrument together, so ember supplies it here, beside the
# corpus-stats wiring it mirrors.
#
# Unwired, the summary is None — the same value a derivation that could not be taken reports. Wired,
# the readiness flag carries the scales the substrate was built at.
from crystal.ontology import seed_lattice as _seed_lattice   # noqa: E402

_seed_lattice.set_geometry_provider(_match._derive_geometry)
