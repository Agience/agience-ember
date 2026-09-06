"""Corpus acquisition: domain, grouped apart from the runner.

Stage-0 lexical sources (OEWN, CILI, ConceptNet, OMW), the ingest-side source runtime, and the
local repo ingester. Acquiring a particular corpus is a knowledge concern rather than a runner
concern, so these modules sit together and nothing in `ember/runtime/` depends on them.

`nltk` and `wn` are imported inside the functions that use them. They are corpus-build tools — the
`bootstrap` extra — used once to materialise WordNet and the license-vetted OMW lexicons into the
lattice. A node that serves from the rows these ingesters mint reads them from the corpus and needs
neither package.
"""
