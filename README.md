# Agience Ember

[![PyPI](https://img.shields.io/pypi/v/agience-ember)](https://pypi.org/project/agience-ember/)
[![Python](https://img.shields.io/pypi/pyversions/agience-ember)](https://pypi.org/project/agience-ember/)
[![License](https://img.shields.io/pypi/l/agience-ember)](LICENSE)
[![CI](https://github.com/Agience/agience-ember/actions/workflows/ci.yml/badge.svg)](https://github.com/Agience/agience-ember/actions/workflows/ci.yml)
[![Sponsor](https://img.shields.io/badge/Sponsor-Agience-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Agience)

**An observer unit.**

Ember is the unit that observes: the *leaf* — the local touch, present where observation
happens. Operationally it is the local cache, and that operational bound is the scope bound. It
runs on your machine (desktop app, browser extension), holds its own shards, answers from them
when it can, and reaches the mesh only on a miss. It is the platform **present locally**.

It is **two-way**: it connects chorus + tools to your local services, and your local services
to chorus / lumen / mantle.

## The boundary

Being a cache is the boundary. Ember is a cache with a socket — an **assembly** of
`prism.host.Host` + the mesh + the relay channel (one observer unit in the canonical component
set — pharos `working/genesis/COMPONENTS.md`). Personas live in **chorus**, reached over the wire.
The full control plane lives in the `authority` mode.

## Why a cache can be trusted here

Every shard is content-addressed and authority-signed, so a cached shard is **exactly as
trustworthy as the origin's**, verified independently of who served it. Ember holds the truth,
locally.

And replication is **blind**: Ember can hold ciphertext it cannot read, decrypting only what
the holder has grants for. Caching decisions decouple from authorization decisions — you can
pre-warm a neighborhood of the manifold and still read only what you're allowed to.

The mesh supplies every cache primitive already:

| cache concern | mechanism |
|---|---|
| invalidation | monotonic `version` per region; gossip surfaces "newer exists" |
| working set / eviction | `route_query_regions()` — what answers a query is *computable* |
| refill source | the gossip directory ranks providers freshest → most-coherent → densest |
| integrity | Merkle `content_root` + Ed25519 authority signature |

## The reasoning / information split

Ember carries its **own** engine, because it must work with the network unplugged. That engine
is one side of a deliberate split:

> **Reasoning is lightweight and weightless. Knowledge lives in artifacts and is retrieved.**

Knowledge is never compressed into weights and the engine holds no corpus. It reasons over what
the cache hands it — which is the same thing the cache exists to hold.

**The split is a type signature:**

```python
answer(query: str, evidence: Sequence[Evidence]) -> Answer
```

`evidence` is the engine's *only* knowledge input. An engine with no weights and no corpus
**structurally cannot** answer from memory — there is none. A wrong answer can only come from
wrong evidence, which is inspectable, attributable and fixable. You cannot audit a weight.

Composing prose from evidence is a persona act, so ember constructs no answerer of its own:
`ember.runtime.engine.build()` raises `NoAnswerer`, and the composers live in
`lumen/composers.py`. An answerer arrives by injection — `Ember(..., engine=<answerer>)`. Left
unwired, `ask()` says it has no answer rather than producing one from the runner. `EMBER_ENGINE`
is still read into settings so the variable keeps its meaning, and it selects nothing.

This is also why there is no GGUF, no model runtime and no `[llm]` extra. The declared dependency
list is `numpy`, `cryptography`, `agience-prism[trust,vector,wire]`, `agience-crystal` and
`agience-mantle` — a leaf that had to ship a 350MB model would not be a leaf.

## Ember holds the instrument

Ember wraps the `entroptics` instrument, and every read in this repository goes through one module.

- **`ember/optics.py` is the one seam.** Exactly one module here imports `entroptics`, enforced by
  an AST test (`tests/test_one_instrument.py`), with `prism` / `mantle` / `crystal` at zero.
- The dependency is the `[optics]` extra, `entroptics>=0.2,<0.3`. It is an extra rather than a
  hard dependency because ember's cache, runner, relay, identity, corpus and config all work
  with no instrument present: a node that **carries** frames installs the base, a node that
  **measures** them installs `[optics]`, and `prism.instrument` reports the absent measurement at
  the point of measurement rather than crashing at import.
- Both the base and `[optics]` resolve from PyPI — `entroptics` 0.2.1 is published and satisfies
  the pin below. The extra exists for the reason above, not because of availability.
- `optics.py` pins `ENTROPTICS_REQUIRED = "0.2"` and matches major.minor exactly, patch releases
  floating, so a version skew fails at import rather than in a reading.

Importing `ember` registers the instrument as the process-default instrument (`ember/__init__.py`).
The registration is a factory, so `import ember` does not drag numpy and entroptics into a process
that only wanted the cache or the runner — the module loads on the first measurement. The
registration is global: an `import ember` line in another suite is a host declaration, which is why
those lines carry `# noqa: F401` and a header saying so.

## The architecture rule (read this before adding a dependency)

Ember is AGPL-3.0-only. The rule below is architecture, not license:

> **Ember is the substrate/node the platform reaches into. It reaches the rest of the platform over
> the ground plane, never by linking.**

- Ember imports no `chorus` and no `lumen`. Personas reach into ember (persona→ember); ember
  reaches a persona over the ground plane — the mantle lattice and the wire,
  provenance-correlated. That keeps the DAG acyclic and ember standalone-deployable.
- Depend on the foundational Apache libraries — `crystal`, `mantle`, `prism`, `numpy`,
  `cryptography` — which is exactly what `pyproject.toml` declares. `entroptics` is
  reached through `ember/optics.py` and nowhere else; that single seam is what keeps the
  permissive half publishable.

## First light — what a leaf cannot do for itself

Ember **runs** disconnected. It is **initialised** from outside, once, and two things must arrive:

1. **The canonical AnchorSet** — without it there is no routing. A leaf authors none of its own:
   anchor ids are content-addressed over `(label, model_id, embedding)`, so a home-made set
   computes region ids nobody else uses. It would route, answer, and share with nobody.
2. **The authority's public key** — without it a cached shard cannot be *verified*, and an
   unverified shard is a rumour rather than a cache. Ember declines to load instead of trusting it.

So a fresh leaf is **uninitialised** — a distinct condition with its own remedy. An *empty* leaf
answers "I don't hold that" (true, useful); an *uninitialised* one cannot route, so answering at
all would be a fabrication about its own state. `seed(anchors, authority_pub)` is an explicit
call: **boot is purely local** (tested by making `socket.connect` raise), so it is fast, offline,
and cannot hang on a dead cloud.

## Status

Scaffold. The read path is real and tested disconnected; the cache survives a restart; the
**outward relay refill is live** (miss → pull from the cloud, verify, answer) over an injected
transport, so the refill path is tested with no socket. The inward `serve()` direction is a seam.

```bash
pip install agience-ember           # the leaf
pip install 'agience-ember[optics]' # ...and the instrument that measures

ember status                       # what this leaf holds, and whether it is seeded
ember ask "..."                    # answer from local shards; cites what it grounded on
ember ingest --anchors <path> ...  # observe + describe into the durable local store, then index
ember reindex --anchors <path>     # rebuild the derived index purely from the store
ember serve                        # the inward direction (a seam)
```

Those five are what `ember.cli` registers (console script `ember = "ember.cli:main"`).

Requires Python 3.11 or newer. Five dependencies, all from PyPI:
[`agience-prism`](https://pypi.org/project/agience-prism/),
[`agience-crystal`](https://pypi.org/project/agience-crystal/),
[`agience-mantle`](https://pypi.org/project/agience-mantle/), numpy and cryptography — plus
[`entroptics`](https://pypi.org/project/entroptics/) behind `[optics]`, because a leaf that only
holds shards does not need an instrument.

To work on Ember itself, see [CONTRIBUTING.md](CONTRIBUTING.md) — the suite needs an
`AGIENCE_BUNDLE_ROOT`, and that file says why.

`--anchors` is a **path to the canonical AnchorSet artifact**, and it is required on both `ingest`
and `reindex`: anchors are provisioned, never derived locally, because an anchor id is
content-addressed over its embedding and a home-made set routes into cells no peer shares. A fresh
leaf therefore cannot ingest until it has been seeded — the same "uninitialised, not broken"
condition described under *First light*.

### How this file is checked

`tests/test_readme_documents_the_real_cli.py` reads this file mechanically: every registered
subcommand must appear here, every `ember <cmd>` named here must be registered, and every
`python -m <module>` inside a fenced block must be importable. The module scan is scoped to fenced
blocks, so prose may name a module that does not exist — `ember.demo`, for instance — without that
reading as an instruction to run it.

## Layout

| module | role |
|---|---|
| `runtime/boot.py` | the assembly — `Ember.boot()` / `.seed()` / `.connect()` / `.ask()` / `.remember()` / `.revise()` / `.status()` |
| `config.py` | env-driven settings (cache dir, engine choice, cloud URL) |
| `runtime/engine.py` | the reasoning half; the answerer is injected, and `build()` raises `NoAnswerer` |
| `runtime/read_path.py` | query → route → hit? answer local : miss → refill → answer |
| `runtime/relay.py` | the two-way channel: `Channel` / `Disconnected` / `MeshChannel`, transport injected |
| `runtime/seams.py` | the host seam table — bound at package scope so any process holding ember is a host |
| `optics.py` | the instrument: the one module that imports `entroptics` |
| `genesis.py` | the operator corpus and the provenance rungs |
| `ontology/` | `activation` (spreading activation over the taxonomy) and `match` (the propagation kernel and its measured constants) |
| `signal/` | `projection`, `forgetting`, `pooling`, `signal`, `state` |
| `corpus/` | `ingest` and the stage-0 sources |
| `consolidate/` | `colimit` and `diagram` |
| `surface/` | `serve` (the inward HTTP direction), `console`, `stats` |
| `cli.py` | the `ember` console script |

## The local embedder, and why it's swappable

A leaf embeds queries locally or disconnected operation is a fiction. But region ids are
`{principal}/{collection}/{anchor_id}`, and `anchor_id` is a UUID5 of
`sha256(label, model_id, embedding)` — so **a leaf that embedded with its own model would mint
its own anchor ids and compute region names nobody else uses.** Its shards would be
unshareable, silently, because routing would still appear to work.

So Ember caches the **canonical** AnchorSet and projects into it with a **cross-walk**: same dim →
orthogonal Procrustes, cross dim → rectangular least-squares. That is what makes the ontology
embedding-dimension agnostic in practice — a 384-dim leaf routes against a 1024-dim canonical
space and lands on **the same region ids**.

It aligns **offline**: an anchor carries both its `label` and its canonical `embedding`, so
Ember embeds the cached labels with its own small model, pairs them against the canonical
vectors already present, and fits. The cloud is called only to learn about *new* anchors.

**"Small while disconnected, upgrade during sync"** is then just: refresh the AnchorSet artifact
and refit. The cross-walk reports an `error_bound`, so what the small embedder costs is a number.

**The embedder: there isn't one, deliberately** (no-models rule). Semantic retrieval is the
**computed** ontology coordinate in `crystal.ontology.geometry`: an exact Jiang–Conrath vector
derived from our own WordNet information content, not a learned space. `HashEmbedder`
(`ember.embed`) is the deterministic plumbing exerciser — it is not semantic, and it
does not ship as a retrieval path. The no-models rule covers distilled lookup tables too: static
at inference, but every entry is a **trained weight**.

## Everything is an artifact

Mantle is explicit:
*"Container-as-artifact: a workspace IS a collection IS an artifact"* (`Collection = Artifact`,
discriminated by `content_type`, membership via edges). So:

- **A region id is a path of artifact identities** — `{principal}/{collection_id}/{anchor_id}`,
  where the collection is an artifact and the anchor is an artifact. That is *why* two nodes
  that never met agree on what a cell is called.
- **Ember's local collection is an artifact it authors**, not a config string. Its id is
  `uuid5(principal)`, so one person's laptop and desktop derive the **same** id and can sync
  directly.
- **The cross-walk is an artifact.** It is derived data with provenance and a measured
  `error_bound`. The fit is deterministic given (embedder, AnchorSet), so *every leaf on the same
  embedder computes the identical matrix*. Fit once, publish, adopt. Its id includes an
  **anchorset fingerprint**, because a walk fitted against older anchors still projects and still
  routes — into yesterday's cells, silently. Making the fingerprint part of the identity turns
  that into an ordinary cache miss.

## Where the code lives

| | why |
|---|---|
| **`mantle.search.anchors`** | the shared coordinate system. Mantle's search stack *and* Ember both use it — `ember/cli.py` and `ember/corpus/ingest.py` import `AnchorSet` from it, and `runtime/boot.py` reads `crosswalk_artifact` from it. |
| **`mantle.shard`** | the cache, the store and the local collection — `cache.LocalCache`, `store.ShardStore`, `local_store.open_store`, `local_collection`. Ember re-exports them from `ember/__init__.py`. |
| **`mantle.mesh`** | gossip, sync and federation. Ember reads it from `runtime/boot.py`, `surface/stats.py` and `facets/browse.py`. |
| **`ember.embed`** | `Embedder`, `HashEmbedder` and the `Aligner` that fits the cross-walk. It lives here rather than in mantle because an embedder is not database code, and it reads `mantle.search.anchors` for the canonical space. |
| **`crystal.ontology`** | the ontology itself — `geometry` (the Jiang–Conrath coordinate) and `driver` (the ambient read). Ember names the driver's store from `ember/__init__.py`, because a host knows what lattice it is running on and the coordinate does not. |
| **`prism.mass`** | belief. The server and the leaf must weigh artifacts **identically**, or belief forks at the edge. |
| **`prism.instrument`** | the injection point the wire resolves a measurement through: explicit keyword, then the process default ember registers, then no reading. |

What is shared lives where the table says: `mantle` for the coordinate system and the shard layer,
`crystal` for the ontology, `prism` for the wire and for belief.

## License

**Dual-licensed: AGPL-3.0-only *or* commercial.** See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE);
commercial and white-label terms in [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md). Contributing:
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CLA.md`](CLA.md).
