# Changelog

All notable changes to `agience-ember` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.5] - 2026-11-08

### Removed
- `tests/e2e/test_live_lumen_e2e.py`, the live test against the retired `lumen.agience.ai` site.

### Fixed (release candidate)
- `httpx` is a declared dependency. `ember/mantle_api.py`, which `ember/__init__.py` imports, calls Mantle's API with it, and Mantle declares httpx only for its service install, so a clean install of Ember could not import `ember`.

### Changed (review 31)
- The `agience-prism` floor is 0.1.9, the `host` extra's included.

### Changed (review 22)
- Ember's own credential, a delegation narrowed to Ember, is exchanged at Origin for one addressed to Mantle before Ember's own calls there; a token addressed to Ember is never sent.

### Changed (review 21)
- `EMBER_INVOKE_TOKEN` runs the node's own operators and chooses no principal: `X-Ember-Principal` and `X-Ember-Delegate` name nobody, and `/v1/invoke` gives `op.remember` and `op.share` the verified caller's principal.
- A person's write into Mantle is made on the delegation Origin addresses to Mantle, from Ember's own exchange of the one it was sent (`MantleAPI.for_caller`, `mantle_api.exchange_for_mantle`); a token addressed to Ember is never sent to Mantle.

### Changed (review 12)
- Ember's content mirror is never Mantle's content bucket (`mantle_api.refuse_mantle_bucket`, on every store open and in the mesh-stats fallback): the mirror must be a bucket Ember names in `EMBER_OVH_BUCKET` (the mirror's default is Mantle's) and none in `EMBER_FORBIDDEN_BUCKETS` or Mantle's own bucket settings. Ember reaches Mantle's data by its API only.

- Ember's refresh names its client (`client_id`, `EMBER_ORIGIN_CLIENT_ID`, default `platform`), which Origin now checks against the family.

### Changed (review 11)
- Ember shares nothing (John, 2026-11-02): `share` and `op.share` are refused in every mode, and Ember's Mantle client makes no grant. Sharing happens only in the person's own session. Ember keeps `remember` (a Create to the drop key) and the read check (a verdict).
- Ember's own credential is the delegation Origin issues to Ember's account: Ember no longer exchanges it to narrow itself. The refresh answer is capped while it streams.
- The store refusal sees through Windows long-path (`\\?\`) and `localhost` administrative-share spellings, refuses a hard-linked store file or one that is, by identity, a file in Mantle's folders, covers `EMBER_CONTENT_DIR`, and checks the opened store again.
- A delegation issued before its person's published sign-in bound is refused (`prism.trust.not_before`).

### Changed (review 10)
- Ember opens no person's content. A memory goes into the person's own memory collection by its derived id (Mantle makes it at their sign-in), as a Create Mantle seals to the collection's drop key; Ember never creates or reads it. A share is one Read grant on one artifact to one named person, expiring within 7 days (`SHARE_DAYS`); Ember never shares with the commons. The read check is Mantle's verdict (`GET /grants/my-access`), never a read.
- Ember's index holds none of Mantle's rows (`refuse_mantle_rows`), so no answer rests on Mantle text a caller may not read; the index rebuild no longer reads Mantle.
- Ember's own credential is a delegation narrowed to Ember (token exchange after each refresh), never a full person token; a refused refresh family switches it off once, with an error, and is not asked again.
- Answers from Mantle and Origin are capped while they stream, and a declared length over the cap is refused before anything is read.
- A withheld answer reads exactly like an empty one (`WITHHELD_TEXT` is `EMPTY_TEXT`).
- The store refusal also covers Ember's keys folders (`KEYS_DIR`, `EMBER_STORE_KEYS_DIR`) against Mantle's folders (`MANTLE_KEYS_DIR` too), and every store open goes through it (`open_own_store`).

### Changed (review 8)
- Ember's HTTP surface takes only a delegation Origin narrows to it (`principal_type` delegation, `aud` and `act.sub` `ember`, no `platform:admin`); a caller's full platform token is refused.
- Ember's own refresh token rotates one process at a time (a file lock beside `mantle.refresh`), a lost answer is retried once with the same token, a refresh token with no keys folder to keep its successor is refused, and once the seed has been spent it is never presented again, even if the kept file is lost.
- Rows read from Mantle are indexed under Mantle's namespace (`mantle:<id>`) and their own author, never re-signed as Ember's; an answer citing one is shown only to a caller who may Read it in Mantle now (`mantle_api.readable_by`).
- `op.share` over `/v1/invoke` keeps `to_principal`, and shares to the commons only when asked (`to_public=true`).
- Ember's store is refused anywhere inside Mantle's folders, compared as real paths (the store file as well as its folder), and with Mantle configured Ember must be told Mantle's folders (`EMBER_FORBIDDEN_DIRS` or `MANTLE_DATA_DIR`). `MantleAPI.from_env(caller_token="")` never falls back to Ember's own credential; artifact ids are quoted into one path segment, redirects are not followed, and answers are capped at 4 MiB and 200 items a page.

### Changed
- Ember keeps the newest refresh token Origin returns in its own keys folder (`$KEYS_DIR/mantle.refresh`, 0600, written atomically) and presents only that one: Origin rotates refresh tokens on use and revokes a reused one's whole family. `EMBER_MANTLE_REFRESH_TOKEN` seeds the first exchange only.
- The BLAS negative control runs its child with every thread cap removed.
- Ember reaches Mantle's data through Mantle's HTTP API, never Mantle's store (`ember.mantle_api`). With `EMBER_MANTLE_URI` set, a memory (`remember`) is created in the caller's own memory collection in Mantle and a share (`share`) is a grant made there, both on the caller's own Origin token; the index rebuild also reads what Ember's own credential may see (`GET /artifacts/visible`). Ember's own credential is the refresh token of its Origin account (`EMBER_MANTLE_REFRESH_TOKEN`, exchanged at `EMBER_ORIGIN_URI`), or a fixed `EMBER_MANTLE_TOKEN` for development.
- Ember keeps its own store and key, and refuses to start on a store folder that holds Mantle's lattice (`MANTLE_LATTICE_PATH`).

## [0.2.4] - 2026-10-31

### Changed
- The lexical index holds only what the store holds in the clear, the platform's public corpus: a row whose fields are sealed, whose body is under its owner's envelope, or whose fields are under the node's key never enters it, and leaves it when it becomes so (`fts.is_sealed`). The index is node-wide and checks no grant; a person's content is searched through Mantle's per-owner index.

## [0.2.3] - 2026-10-30

### Changed
- The lexical index (`ember.corpus.fts`) holds keyed tokens, never words: each word is stemmed by FTS5's `porter` and replaced by a keyed token of the stem (`mantle.db.node_seal.keyed_token`), and a query and a document-frequency lookup are tokenized the same way, so BM25 and the frequencies are unchanged. The tables are `fts_kt`, `fts_kt_map` and `fts_kt_vocab`, counted by `fts_kt:total`; `rebuild_from_vertices` drops a plaintext index an earlier version wrote. Requires `agience-mantle>=0.1.9`.

## [0.2.2] - 2026-10-29

### Added
- `.github/workflows/publish.yml` now also runs on pushes to `release/**` branches. It builds and checks the package as it does for version tags, then keeps the result as the `dist` artifact so the platform release can deploy exactly those files. Nothing is published to a package index or GitHub Release from these branches.

## [0.2.1] - 2026-10-17

### Changed
- WordNet ingest (`op.source.oewn`, `op.source.omw`, the stage-0 PWN spine) no longer mints a `type.text/x-wordnet` content-type artifact. A synset's describer is declared in Sage's `text/x-wordnet` type.json, beside the offer Mantle indexes.
- `scripts/lattice_offers.py` takes each type's describer from its type.json declaration when one exists, and from a `type.<ct>` store row only for a type no type.json declares. It stops only when neither exists.
- `tests/test_peer_apply_delivery.py` wires the peer-signal sink as Ember's boot does before delivering; it tested a direct call Mantle no longer makes, and failed.
- A basis generation is linked to the one before it by the new generation's `parent_id`, as Mantle now records it; no row carries `superseded_by`.

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
