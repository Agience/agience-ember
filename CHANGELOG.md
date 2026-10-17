# Changelog

All notable changes to `agience-ember` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.1] - 2026-10-17

### Changed
- WordNet ingest (`op.source.oewn`, `op.source.omw`, the stage-0 PWN spine) no longer mints a `type.text/x-wordnet` content-type artifact. A synset's describer is declared in Sage's `text/x-wordnet` type.json, beside the offer Mantle indexes.
- `scripts/lattice_offers.py` takes each type's describer from its type.json declaration when one exists, and from a `type.<ct>` store row only for a type no type.json declares. It stops only when neither exists.
- `tests/test_peer_apply_delivery.py` wires the peer-signal sink as Ember's boot does before delivering; it tested a direct call Mantle no longer makes, and failed.

## [0.2.0] - 2026-10-09

### Breaking
- Stored artifact state is now `live` or `archived`, replacing `committed`. Everything ember writes, including ingest, sources, operators, hosts and the `scripts/backfill_leaf.py` cursor, is written `live`. Stores, queries and scripts that filter on `state="committed"` will not match it.
- `ingest_stage0_wordnet` and `ingest_stage0_oewn` report the existing count under `live` instead of `committed` when they skip an already-ingested source. The `ember` CLI `ingest` command prints `store artifacts (live)`.
- Observations from `corpus/sources` and `genesis.ingest_dataset` have their state resolved on write. `archived` is kept, and any other value or none is stored as `live`. This covers the `committed` that `prism.source.Observation.to_doc()` writes below prism 0.1.3.
- `surface/serve` refuses the control-intent `remember` request unless the caller is authenticated, either with an Origin access token or with `EMBER_INVOKE_TOKEN`. It lands in that caller's own scope. `advance` and `ingest` require the `EMBER_INVOKE_TOKEN` credential, the same one `/v1/invoke` requires.
- `surface/serve` refuses requests that carry a browser `Origin` header unless the origin is listed in the new `EMBER_ALLOWED_ORIGINS` setting (comma-separated exact origins). Requests without an `Origin` header are unaffected.
- The instrument package (0.2.8) is a new required dependency, pinned exactly because measured values change between 0.2.x releases. The minimum versions rise to `agience-prism>=0.1.4`, including the `host` extra, and `agience-mantle>=0.1.3`.

### Changed
- `ember.optics` binds the installed instrument package and checks its version (`INSTRUMENT_REQUIRED`, `INSTRUMENT_SOURCE`); it no longer requires a source checkout in the workspace.
- `surface/serve` verifies Origin bearer tokens with prism's `verify_access_token`. It rejects Origin's refresh, email-verification and password-reset tokens, requires user tokens to be addressed to the authority issuer, and accepts no delegation audiences. Verification still fails closed to the anonymous commons.
- `consolidate/colimit` writes an explicit `state` on every colimit. It is `archived` only when every member is archived, and `live` otherwise.
- `corpus/ingest._offer_text` and `ontology/activation.seeds_from_artifact` read an artifact's description through mantle's `client_view`. The caller's keys are used wherever the context holds them, and the store's own mint facets are dropped. A context holding only store facets yields an empty offer, and non-ASCII characters are kept as written.
- `genesis._count_in` counts members of a collection in every state, archived included. `_count_collection` no longer takes `committed_only` and always counts every state.

### Security
- Bearer-token identity in `surface/serve` now requires a token that passes prism's access-token claim rules, not just a valid signature.
- Browser-originated cross-origin requests to `surface/serve` are refused unless the origin is explicitly allowed.
