"""The surfaces a running ember presents — serve, stats, console.

`serve` is the HTTP surface a live node exposes, `stats` the decoupled snapshot the mesh reads, and
`console` the local control console.

Named `surface` rather than `node` because the repo already has a top-level `node/` holding the
operator tooling that runs a node — the loops and the repair suite. One name per layer.
"""
