"""The local embedder, and the alignment that makes it swappable.

The problem
-----------
Region ids are ``{principal}/{collection}/{anchor_id}``, and ``anchor_id`` is a UUID5 of
``sha256(label, model_id, embedding)``. So a different embedder mints *different anchor ids*
and therefore *different region ids* — a leaf that embedded with its own model would compute
cell names nobody else uses, and its shards would be unshareable with the mesh, silently: the
routing would still "work"; it would just be routing into a private universe.

The fix — project, don't re-anchor
-----------------------------------
Ember does **not** build its own AnchorSet. It caches the **canonical** one (anchors are
artifacts, so they cache like anything else) and projects its local query vectors into that
space with a cross-walk (``mesh.anchors.crosswalk``): same dim → orthogonal
Procrustes; **cross dim → rectangular least-squares**. That is what makes the ontology
embedding-dimension agnostic in practice — a 384-dim leaf can route against a 1024-dim
canonical space and land on **the same region ids**.

And it can get aligned with the network unplugged: an ``Anchor`` carries both its ``label``
and its canonical ``embedding``, so Ember embeds the cached anchors' *labels* with its own
small model, pairs them against the canonical vectors already present, and fits the cross-walk
locally. No cloud call to become alignable — only to learn about *new* anchors.

Upgrading
---------
"Something small while disconnected, upgrade during sync" is then just: refresh the cached
AnchorSet artifact and refit.

Whether that upgrade is measurable depends on which number is read. ``error_bound`` is
deprecated and always returns ``None``: its in-sample residual falls as the fit becomes more
underdetermined, so a smaller, worse embedder can report a better score than a good one. Read
:attr:`Aligner.residual` (``.held_out``) against
:func:`~mantle.search.anchors.crosswalk.null_residual`, a derived reference rather than a chosen
threshold, and read :attr:`Aligner.carries_information` for the boolean verdict — ``None`` means
unknown and must not be read as fine.

Provenance
----------
Vendored from `agience-mantle`, `src/mantle/ontology/embed.py` (Apache-2.0, Copyright Ikailo Inc.),
at commit `9768c5f`; mantle removed it in `a7cfd0c`. Mantle is the storage layer and an embedder is
not database code, so it moved to its only consumer. Copied unmodified apart from this note and the
module path. It still reads mantle's anchor set (`mantle.search.anchors`) — the canonical coordinate
system is mantle's and stays there; what moved is the leaf-side embedder and the fit.

It sits at package top level rather than in `ember/ontology/`, beside `optics.py` and `config.py`,
because `ember/runtime/boot.py` imports it at module scope and `ember/__init__.py` imports boot.
`ember/ontology/__init__.py` carries eager wiring (it imports `ember.ontology.match`, a seam
target), so filing this module inside that package would make `import ember` resolve a seam eagerly
and turn `test_runner_seam.py::test_registering_a_seam_imports_NOTHING` red. This module imports
nothing of ember's, which is what lets boot reach it during package initialisation.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Protocol, Sequence

import numpy as np

from mantle.search.anchors.anchorset import AnchorSet
# Deferred, so a missing `mantle.search.anchors.crosswalk` cannot take down all of `ember` — every
# reader, the completion path and the chat, none of which fit a crosswalk. The two sibling imports
# in this file are deferred the same way. Deferring does not repair the crosswalk: `fit` needs the
# module and raises if it is called without it.


class Embedder(Protocol):
    """Turn text into vectors. Deliberately tiny surface — the cross-walk is what lets the
    implementation be swapped without changing a single region id."""

    model_id: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    """A deterministic, dependency-free embedder. Not semantic — for tests and demos.

    It exists so the whole read path (routing, alignment, hit/miss, and reads that carry no
    information) can be exercised with numpy alone, and so the scaffold has no model download
    in its critical path. It is stable across processes and machines, which is all the
    plumbing needs.

    Do not ship this as a real embedder: hashed tokens have no meaning, so "nearest" is
    arbitrary. There is no swappable "real" embedder: semantic retrieval is the computed
    ontology coordinate (see the note below), not a learned vector space.
    """

    model_id = "hash-embed-v1"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in (t or "").lower().split():
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                out[i, idx] += sign
        from prism import vector as _vec
        return _vec.unit(out, axis=1)


#
# There is no trained or distilled embedder here: the semantic arm computes its ontology
# coordinate rather than learning one (`ember/geometry.py`'s exact Jiang-Conrath coordinate,
# LATTICE-IMPLEMENTATION §2.2, from this project's own WordNet IC), so no model — small,
# Apache-2.0, or otherwise — belongs in this file. `HashEmbedder` above is only the plumbing
# exerciser: it carries no semantic role.


class Aligner:
    """Projects this leaf's local vectors into the canonical anchor space.

    Holds the cross-walk. If the local embedder already *is* the canonical model, the
    cross-walk is skipped entirely (identity) rather than fitted — a leaf running the real
    model should pay nothing for the abstraction.
    """

    def __init__(self, embedder: Embedder, anchors: AnchorSet) -> None:
        self.embedder = embedder
        self.anchors = anchors
        self._crosswalk = None
        #
        # Comparing dim as well is not merely a guard, it is the correct behaviour: a differing
        # dim simply means not-native, so the normal path fits a rectangular least-squares
        # crosswalk — which this module already supports and advertises ("cross dim ->
        # rectangular least-squares"). The mismatch stops being silent and starts being handled.
        self._native = (embedder.model_id == anchors.model_id
                        and int(getattr(embedder, "dim", -1)) == int(getattr(anchors, "dim", -2)))
        self._adopted = False   # did we reuse a shared walk instead of refitting?

    @property
    def native(self) -> bool:
        return self._native

    @property
    def residual(self):
        """What the projection cost, as a :class:`FitResidual` — never a bound on what it will cost.

        ``None`` when native (nothing is being projected).

        Read ``.held_out``, and read it against :func:`null_residual`. ``.in_sample`` falls toward
        zero as the fit becomes more underdetermined, which is the opposite of what a quality
        number does.
        """
        return None if self._native else getattr(self._crosswalk, "residual", None)

    @property
    def carries_information(self) -> Optional[bool]:
        """Whether this cross-walk says anything at all, or sits at the shared-nothing null.

        On one measured path (6 anchors, 16-dim leaf → 64-dim canonical): in-sample 0.107,
        held-out 1.026, null 1.0. The projection sits at the null — it carries no information —
        while `error_bound`'s in-sample number on the same path reads as ~89% fidelity. That gap
        is why this property exists rather than a fidelity score.

        ``None`` means unknown (native, unfitted, or too few pairs to hold any out), which a
        caller must not read as fine. The comparison is against a derived null, not a chosen
        threshold.
        """
        res = self.residual
        if res is None or res.held_out is None:
            return None
        from mantle.search.anchors.crosswalk import null_residual
        null, se = null_residual(res.dim_out, max(1, res.n_held_out))
        return res.held_out < null - 2.0 * se

    @property
    def error_bound(self) -> Optional[float]:
        """Deprecated — always ``None``. Use :attr:`residual` and :attr:`carries_information`.

        This returned the cross-walk's in-sample residual while its docstring called it *"how much
        the small embedder is costing"*, so a descriptive number was read downstream as a fidelity
        guarantee. It returns ``None`` (unknown) rather than a plausible number, because a wrong
        number here is worse than an absent one.
        """
        return None

    @property
    def adopted(self) -> bool:
        """True when we reused a shared cross-walk artifact rather than refitting."""
        return self._adopted

    def fit(self, cached: Optional[dict] = None) -> "Aligner":
        """Align this leaf. Offline by construction.

        ``cached`` is an optional cross-walk **artifact** (see `crosswalk_artifact`). The fit is
        deterministic given (embedder, AnchorSet), so every leaf on the same embedder computes
        the identical matrix — refitting per device is pure waste. If a valid one is supplied we
        adopt it; otherwise we fit from the anchors themselves, which is always possible with no
        network because an Anchor carries both its label and its canonical embedding.

        A cached walk that does not verify is treated as a **cache miss, not an error**: we
        refit silently. That is deliberate — a stale or foreign walk still projects and still
        routes, just into the wrong cells, so the only safe response is to ignore it.
        """
        if self._native:
            return self
        if cached is not None:
            from mantle.search.anchors.crosswalk_artifact import from_artifact, verify_crosswalk_artifact
            ok, _why = verify_crosswalk_artifact(cached, self.anchors, self.embedder.model_id)
            if ok:
                self._crosswalk = from_artifact(cached)
                self._adopted = True
                return self
        anchors = self.anchors.anchors
        if not anchors:
            raise ValueError("cannot fit a cross-walk against an empty AnchorSet")
        labels = [a.label for a in anchors]
        source = self.embedder.encode(labels)                       # our space
        target = np.vstack([a.embedding for a in anchors])          # canonical space
        from mantle.search.anchors.crosswalk import fit_crosswalk
        self._crosswalk = fit_crosswalk(
            source, target,
            source_model_id=self.embedder.model_id,
            target_model_id=self.anchors.model_id,
            method="auto",   # same dim -> procrustes; cross dim -> rectangular linear
        )
        return self

    def as_artifact(self, fitted_by: Optional[str] = None) -> dict:
        """This leaf's cross-walk, as a shareable artifact — so the next leaf need not refit."""
        if self._native:
            raise ValueError("a native embedder has no cross-walk to publish")
        if self._crosswalk is None:
            raise RuntimeError("Aligner.fit() must run before as_artifact()")
        from mantle.search.anchors.crosswalk_artifact import to_artifact
        return to_artifact(self._crosswalk, self.anchors, fitted_by=fitted_by)

    def encode_query(self, text: str) -> np.ndarray:
        """Text -> a vector in the canonical space, ready to route."""
        v = self.embedder.encode([text])[0]
        if self._native:
            return v
        if self._crosswalk is None:
            raise RuntimeError("Aligner.fit() must run before encode_query() (no cross-walk)")
        return self._crosswalk.apply(v)


__all__ = ["Embedder", "HashEmbedder", "Aligner"]
