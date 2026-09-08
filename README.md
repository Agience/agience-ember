# Agience Ember

[![PyPI](https://img.shields.io/pypi/v/agience-ember)](https://pypi.org/project/agience-ember/)
[![Python](https://img.shields.io/pypi/pyversions/agience-ember)](https://pypi.org/project/agience-ember/)
[![License](https://img.shields.io/pypi/l/agience-ember)](LICENSE)
[![CI](https://github.com/Agience/agience-ember/actions/workflows/ci.yml/badge.svg)](https://github.com/Agience/agience-ember/actions/workflows/ci.yml)
[![Sponsor](https://img.shields.io/badge/Sponsor-Agience-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Agience)

**An observer unit — the local leaf.**

Ember runs on your machine, holds its own shards, answers from them, and reaches the mesh only on a
miss. It is two-way: it serves your local services to the network, and the network to your local
services.

## Install

```bash
pip install agience-ember             # the leaf
pip install 'agience-ember[optics]'   # ...and the instrument that measures
```

Requires Python 3.11 or newer.

```bash
ember status                       # what this leaf holds, and whether it is seeded
ember ask "..."                    # answer from local shards, citing what it grounded on
ember ingest --anchors <path> ...  # observe and describe into the durable store, then index
ember reindex --anchors <path>     # rebuild the derived index from the store alone
ember serve                        # an OpenAI-compatible /v1 endpoint on 127.0.0.1:8091
```

`--anchors` is a path to the canonical AnchorSet artifact and is required on both `ingest` and
`reindex`. An anchor id is content-addressed over `(label, model_id, embedding)`, so anchors are
provisioned rather than derived locally: a home-made set would route into cells no peer shares.

## First light

A leaf runs disconnected and is initialised from outside, once. Two things have to arrive:

1. **The canonical AnchorSet** — without it there is no routing.
2. **The authority's public key** — without it a cached shard cannot be verified, and an unverified
   shard is a rumour rather than a cache.

`seed(anchors, authority_pub)` is an explicit call, and boot is purely local — tested by making
`socket.connect` raise — so startup is fast, offline, and cannot hang on a dead network.

Until it is seeded a leaf is **uninitialised**, which is its own condition with its own remedy. An
*empty* leaf answers "I don't hold that"; an uninitialised one cannot route, so `status` says which
it is.

## Cache with a socket

Every shard is content-addressed and authority-signed, so a cached shard is exactly as trustworthy
as the origin's, verified independently of who served it. Replication is blind: Ember can hold
ciphertext it cannot read, decrypting only what the holder has grants for. Caching decisions and
authorization decisions are therefore separable — a neighbourhood can be pre-warmed while reads stay
bounded by grants.

| cache concern | mechanism |
|---|---|
| invalidation | monotonic `version` per region; gossip surfaces "newer exists" |
| working set / eviction | `route_query_regions()` — what answers a query is computable |
| refill source | the gossip directory ranks providers freshest → most-coherent → densest |
| integrity | Merkle `content_root` plus an Ed25519 authority signature |

A region id is a path of artifact identities — `{principal}/{collection_id}/{anchor_id}` — which is
why two nodes that never met agree on what a cell is called. Ember's local collection is an artifact
it authors, with id `uuid5(principal)`, so one person's laptop and desktop derive the same id and
can sync directly.

## Reasoning over evidence

> Reasoning is lightweight. Knowledge lives in artifacts and is retrieved.

The engine's signature is the whole contract:

```python
answer(query: str, evidence: Sequence[Evidence]) -> Answer
```

`evidence` is its only knowledge input, so a wrong answer can only come from wrong evidence — which
is inspectable, attributable and fixable.

Composing prose is a persona act, so `ember.runtime.engine.build()` raises `NoAnswerer` and an
answerer arrives by injection: `Ember(..., engine=<answerer>)`. Left unwired, `ask()` reports that it
has no answer. `EMBER_ENGINE` is still read into settings, and selects nothing.

## The local projection

A leaf embeds queries locally or disconnected operation is a fiction — but region ids are derived
from the canonical anchors, so a leaf that embedded with its own model would mint its own ids and
name cells nobody else uses. Its shards would be unshareable, silently, because routing would still
appear to work.

So Ember caches the **canonical** AnchorSet and projects into it with a **cross-walk**: same
dimension → orthogonal Procrustes, cross dimension → rectangular least squares. A 384-dim leaf
routes against a 1024-dim canonical space and lands on the same region ids.

The fit is offline. An anchor carries both its `label` and its canonical `embedding`, so Ember
embeds the cached labels with its own model, pairs them against the canonical vectors already
present, and solves. The network is called only to learn about *new* anchors. "Small while
disconnected, upgrade during sync" is then: refresh the AnchorSet and refit.

The cross-walk is itself an artifact — derived data with provenance and a measured `error_bound`,
deterministic given (embedder, AnchorSet), so every leaf on the same embedder computes the identical
matrix. Fit once, publish, adopt. Its id includes an **anchorset fingerprint**, because a walk fitted
against older anchors still projects and still routes — into yesterday's cells. The fingerprint turns
that into an ordinary cache miss.

Semantic retrieval is the **computed** ontology coordinate: an exact Jiang–Conrath vector derived
from our own WordNet information content. `HashEmbedder` (`ember.embed`) is the deterministic
plumbing exerciser and is not a retrieval path.

## The instrument

`ember/optics.py` is the one module here that imports `entroptics`, enforced by an AST test
(`tests/test_one_instrument.py`). It pins `ENTROPTICS_REQUIRED = "0.2"` and matches major.minor
exactly with patch floating, so a version skew fails at import rather than inside a reading.

The dependency is the `[optics]` extra: a node that **carries** frames installs the base, a node that
**measures** them installs `[optics]`, and `prism.instrument` reports an absent measurement at the
point of measurement.

Importing `ember` registers the instrument as the process-default through a factory, so `import
ember` loads numpy and entroptics only at the first measurement. The registration is process-global,
which is why an `import ember` line elsewhere is a host declaration and carries `# noqa: F401` with a
header saying so.

## Configuration

`EMBER_CACHE_DIR` · `EMBER_PRINCIPAL` · `EMBER_COLLECTION_ID` · `EMBER_ENGINE` · `EMBER_EMBED_MODEL`
· `EMBER_CLOUD_URI` · `EMBER_NPROBE`. The cache defaults to `%LOCALAPPDATA%\agience-ember` on
Windows and `$XDG_CACHE_HOME/agience-ember` elsewhere.

## Layout

| module | role |
|---|---|
| [`runtime/boot.py`](src/ember/runtime/boot.py) | the assembly — `Ember.boot()` · `.seed()` · `.connect()` · `.ask()` · `.remember()` · `.revise()` · `.status()` |
| [`runtime/read_path.py`](src/ember/runtime/read_path.py) | query → route → hit? answer local : miss → refill → answer |
| [`runtime/relay.py`](src/ember/runtime/relay.py) | the two-way channel: `Channel` · `Disconnected` · `MeshChannel`, transport injected |
| [`runtime/engine.py`](src/ember/runtime/engine.py) | the reasoning half; the answerer is injected |
| [`runtime/seams.py`](src/ember/runtime/seams.py) | the host seam table, bound at package scope so any process holding ember is a host |
| [`config.py`](src/ember/config.py) | env-driven settings |
| [`optics.py`](src/ember/optics.py) | the instrument — the one module importing `entroptics` |
| [`genesis.py`](src/ember/genesis.py) | the operator corpus and the provenance rungs |
| [`ontology/`](src/ember/ontology/) | `activation` (spreading activation over the taxonomy), `match` (the propagation kernel and its measured constants), `information`, `corpus_stats` |
| [`signal/`](src/ember/signal/) | `projection` · `forgetting` · `pooling` · `signal` · `state` |
| [`corpus/`](src/ember/corpus/) | `ingest`, the stage-0 sources and full-text search |
| [`consolidate/`](src/ember/consolidate/) | `colimit` and `diagram` |
| [`surface/`](src/ember/surface/) | `serve` (the inward HTTP direction), `console`, `stats` |
| [`embed.py`](src/ember/embed.py) | `Embedder`, `HashEmbedder` and the `Aligner` that fits the cross-walk |
| [`cli.py`](src/ember/cli.py) | the `ember` console script |

## Status

Scaffold. The read path is real and tested disconnected, the cache survives a restart, and the
outward relay refill — miss → pull → verify → answer — is live over an injected transport, so it is
tested with no socket. The inward `serve()` direction is a seam.

To work on Ember itself, see [`CONTRIBUTING.md`](CONTRIBUTING.md), which lists what the suite needs.

Security issues: email **connect@agience.ai** rather than opening a public issue.

Dual-licensed — see [`LICENSE`](LICENSE), [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md),
[`NOTICE`](NOTICE) and [`CLA.md`](CLA.md).
