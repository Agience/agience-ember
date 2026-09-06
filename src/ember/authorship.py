"""Who authors a row when the caller names nobody.

Every ingest and mint entry point takes an `author=` and needs a default. That default used to be a
maintainer's own address, written into the source at sixty-odd call sites — which is wrong twice
over in a published package: it puts a personal address in every install, and it attributes every
artifact anyone ingests to a person who had nothing to do with it. Provenance that names the wrong
author is worse than provenance that admits it does not know.

So the default is read from the environment, and the fallback is deliberately *unresolvable*.
`EMBER_PRINCIPAL` is the same variable `ember/config.py` reads and a node's own environment already
sets. Absent it, rows are authored by `ember-local`, which resolves to no principal — and the node's
integrity checks fail on exactly that, so an unattributed ingest is loud rather than silent. A
plausible-looking default would pass those checks while recording a fiction.

Read once, at import: these are default parameter values, which Python evaluates when the function
is defined. A caller that needs a different author passes one.
"""

from __future__ import annotations

import os

#: The unresolvable fallback. Named rather than inlined so the integrity checks and this module
#: cannot disagree about which string means "nobody said".
UNATTRIBUTED = "ember-local"

DEFAULT_AUTHOR = os.environ.get("EMBER_PRINCIPAL") or UNATTRIBUTED


def default_author() -> str:
    """`DEFAULT_AUTHOR`, re-read from the environment.

    For callers that set `EMBER_PRINCIPAL` after import — a test, or a process that resolves its
    principal during boot. The module constant is what the `author=` defaults bind to, and it is
    fixed at import; this is the live answer.
    """
    return os.environ.get("EMBER_PRINCIPAL") or UNATTRIBUTED
