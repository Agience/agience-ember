"""The local engine — lightweight reasoning over artifacts.

Reasoning and information are kept separate. Knowledge is not compressed into weights; it
lives in **artifacts** and is retrieved. Reasoning is a small, weightless operation over what
was retrieved, so a wrong answer can only come from wrong evidence — inspectable,
attributable, and fixable in a way a model weight is not.

Composing prose from evidence is a persona act, so this module builds no answerer of its own.
`ExtractiveEngine` / `EntropticsEngine` / `DistillEngine` live in **`lumen/composers.py`**, and
so do `Evidence` and `Answer`: a typed gate at the conduit (`Engine.answer(query, evidence) ->
Answer`) would be a step-I/O pipeline contract, which the propagation rule forbids — the
content type is born at the tekton, not declared ahead of it. Evidence carried off the plane is
an `EVIDENCE_CT` artifact, not a dataclass.

Disconnected operation is not a degraded mode here: the engine never needed the network; the
*cache* did, and only on a miss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, NoReturn, Optional, Protocol, Sequence

import numpy as np


# `Evidence` and the `Engine` protocol are not defined in this module. The runner declares
# neither the question's evidence shape nor the answer's type — a typed gate at the conduit
# would be a step-I/O pipeline contract, which the propagation rule forbids, since the content
# type is born at the tekton. The runner hands over rows (`id/text/score/region/vector` — what
# an `EVIDENCE_CT` artifact carries) and returns whatever the injected answerer produces.
#
# `Evidence` and `Answer` live with the things that consume them, in `lumen/composers.py`. What
# is left below is the injection point: an answerer arrives only by injection.


class NoAnswerer(RuntimeError):
    """No answerer is wired, so there is nothing to compose an answer with."""


def build(name: str, **kw) -> NoReturn:
    """The runner constructs no answerer. Composing prose from evidence is a persona act, and
    the composers live in **`lumen/composers.py`**.

    It raises rather than returning a default: silently degrading to a baseline composer would
    hide a node answering from the runner instead of from a persona.

    Wire an answerer by injection (`Ember(..., engine=<answerer>)`) or leave it absent and let
    `ask()` report that it has no answer. `EMBER_ENGINE` is retained in settings only so an
    existing environment variable keeps its meaning; it selects nothing here."""
    raise NoAnswerer(
        f"ember no longer builds answerers (asked for {name!r}). The composers live in "
        f"lumen/composers.py; inject one via Ember(engine=...), or leave it unwired and ask() will "
        f"refuse. Answers come only from an injected composer.")


# `Evidence`, `Answer` and `Engine` REMOVED from `__all__` 2026-08-25. None of the three exists
# in this module — or anywhere in `ember/src` — since the answerer machinery was retired, and the
# export list was not trimmed with it. `from ember.runtime.engine import *` therefore failed with
# `AttributeError: module ... has no attribute 'Evidence'`: a star-import that cannot succeed, in a
# module whose whole remaining job is to raise a named error.
#
# Nothing was relying on them — every importer in the tree takes `NoAnswerer` and/or `build`
# (`runtime/boot.py`, `ember/__init__.py`, and two tests), checked before trimming.
#
# `build` is annotated `NoReturn` rather than `Engine`, which is what it actually does: it always
# raises `NoAnswerer`. The old annotation named a type that had gone AND described a return that
# never happens.
__all__ = ["NoAnswerer", "build"]
