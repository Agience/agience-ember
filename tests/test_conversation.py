"""Placeholder for the conversation-behavior suite, which lives in lumen.

The conversation acts (`respond`/`think`/`learn`/`act`) belong to `lumen/conversation.py`; ember
holds only the recognition primitives. The behavior tests therefore run from
`agience-chorus/src/lumen/tests/test_conversation.py` (with `--rootdir=agience-chorus/src`). They
need a built WordNet substrate and skip cleanly without one, so full conversation-behavior
verification is gated on that substrate.

This file keeps the ember suite from duplicating those tests or referencing conversation acts that
ember does not hold.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="conversation tekton + its behavior tests moved to lumen (P7); "
                         "see agience-chorus/src/lumen/tests/test_conversation.py")
def test_conversation_moved_to_lumen():
    pass
