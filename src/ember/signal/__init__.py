"""The signal and its measurement primitives. Measurement only, with no domain vocabulary.

  · `signal`     — one signed primitive whose grounding selects message vs event.
  · `projection` — the ordered `(T, F)` frame every cut sits on, and the reads taken of it.
  · `pooling`    — the accumulating Screen, which states the coordinate it holds.
  · `forgetting` — decay: tense is amplitude on a measured curve, not deletion.
  · `state`      — `op.measure`, the node's own envelope placed as a signal.
"""
