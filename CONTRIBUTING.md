# Contributing to Agience Ember

Ember is the **leaf**: the local touch, present where observation happens. Operationally a cache,
and **that operational bound is the scope bound** - it holds its own shards, answers from them when
it can, and reaches the mesh only on a miss.

## Tests

```bash
pip install -e '.[dev]'
export AGIENCE_BUNDLE_ROOT=/path/to/agience-observe/bundles
python -m pytest -q
```

Every test lives under `tests/`, which `pyproject.toml` sets as the only `testpaths` root.

**`AGIENCE_BUNDLE_ROOT` is not optional.** `prism.runner` resolves an operator group to a
sha-verified payload and there is no in-package copy to fall back to, so without it five test files
fail *at collection* with `UnknownBundleGroupError` — before anything they mean to test has run.
Point it at `bundles/` in an `agience-observe` checkout.

The suite is disk-bound and runs about three minutes serially. `pyproject.toml` explains why there
is no `-n auto` default: two reps per config could not distinguish any worker count from any other.

## Being a cache is the boundary

Ember is a cache with a socket - `prism.host.Host` plus the mesh plus the relay channel. **Personas
live in chorus, reached over the wire.** Anything that starts to look like a persona implementation
here is in the wrong repo; move it rather than widening Ember.

A cached shard is exactly as trustworthy as the origin's: content-addressed, authority-signed, and
verified independently of who served it. **That property is what lets a cache be trusted.** A change
that accepts a shard on the strength of where it came from will not merge.

## It registers the seams other repos rely on

`ember/runtime/seams.py` registers `match`, `projection`, `optics`, `activation`, `delegate` and
`forgetting` with `prism.runner`. Mantle's `MANTLE_ONTOLOGY_HOST=ember` names this module so
Mantle's ranking arm can find them - and **Mantle's static import graph names no ember module**, so
nothing in a grep of Mantle will warn you.

**Renaming or unregistering a seam silently turns off a ranking arm in another service, with no
error on either side.** Grep the seam name across the workspace first, and report what you found.

## The architecture rule is not a licence rule

Ember must not import `chorus` or `lumen`. Both are AGPL, same as Ember, so no licence boundary is
crossed - the rule exists because Ember is the substrate personas reach *into*.

## Contributing

**Sign the CLA** - Ember is AGPL-3.0-only **or** commercially licensed
([`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md)), so the project must hold the right to relicense
every line it ships. The bot checks on PR open and links [`CLA.md`](CLA.md).

Fork, branch from `main`, sign off every commit (`git commit -s`), open a PR. Commit format:
`fix:` / `feat(scope):` / `docs:` / `test:` / `chore:`.

**Security vulnerabilities: do not open a public issue** - email **connect@agience.ai**.

## License

**Dual-licensed: AGPL-3.0-only or commercial.** See [`LICENSE`](LICENSE),
[`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) and [`NOTICE`](NOTICE).
