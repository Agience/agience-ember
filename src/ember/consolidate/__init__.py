"""Compactification — the colimit machinery: derive the diagram, take the colimit, smooth the
morphisms.

Two modules, in the order the work happens:

  · `diagram`  — derives the diagram of a concept: the artifacts the evidence cannot separate from
                 it. The concept id is the only input, and the verdict is a measurement.
  · `colimit`  — takes the colimit of a derived diagram: one object carrying every member's
                 provenance, position and mass, with a morphism from each member. Conservation is
                 the acceptance test. Morphism smoothing is `colimit.smooth_edges`, part of the
                 same write.

`ember.genesis.consolidate_colimit_derived` is the operator surface, reached through
`op.consolidate.colimit` with a `concept_id`; it delegates here.
"""
