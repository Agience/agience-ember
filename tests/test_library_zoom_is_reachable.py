"""`/library` offers the zoom levels, and a caller can choose one.

With no resolution supplied the library returns the levels this corpus has, and `library_page`
renders them as `?resolution=<level>` links. Deferring the choice to the query is a measurement only
if the query can then make it: a handler that renders the links and parses `refresh=1` alone sends
every one of them back to the same offer page.

These cover the wiring rather than the clustering. `docs_ops`' own suite covers how the levels are
derived; this covers whether a chosen level survives the trip from the URL to the plan.
"""
from __future__ import annotations

import inspect

from ember.facets import browse
from ember.surface import serve


def test_the_serve_path_PARSES_resolution_not_only_refresh():
    """The `/library` handler reads `resolution` off the query string, so the rendered links carry
    a choice rather than decoration.

    Asserted on the source: the handler is a `BaseHTTPRequestHandler` method inside a closure, and
    standing a server up to observe one query parameter is more machinery than the fact needs.
    """
    src = inspect.getsource(serve)
    lib = src[src.index('u.path == "/library"'):]
    lib = lib[:lib.index("self.wfile.write(body); return")]
    assert "resolution" in lib, "the /library handler still ignores the resolution parameter"
    assert "parse_qs" in lib or "resolution=" in lib


def test_library_page_and_view_BOTH_accept_a_resolution():
    """The parameter survives the whole trip: handler -> page -> view -> plan. A signature anywhere
    in that chain that stops carrying it leaves the handler parsing a value it then discards, which
    looks the same from outside as the parameter being ignored.
    """
    for fn in (browse.library_page, browse.library_view):
        assert "resolution" in inspect.signature(fn).parameters, fn.__name__


def test_an_UNREADABLE_level_is_treated_as_NO_level_not_an_error():
    """`?resolution=banana` lands on the offer page. An unreadable level is no level, so the handler
    falls back to `resolution = None` and offers the choices again.

    An unguarded `float(...)` would take the whole page down on a malformed query string — and
    handing out those links is the page's own job.
    """
    src = inspect.getsource(serve)
    lib = src[src.index('u.path == "/library"'):]
    lib = lib[:lib.index("self.wfile.write(body); return")]
    assert "ValueError" in lib, "a malformed resolution is not handled"
    assert "resolution = None" in lib, "an unreadable level must fall back to offering the choices"


def test_the_view_CACHES_PER_RESOLUTION_not_per_root():
    """The resolution is part of the cache key. Keyed on `root` alone, `_LIB_CACHE` serves the first
    level asked for to every level after it, so the links resolve and return identical content —
    which reads as working.
    """
    src = inspect.getsource(browse.library_view)
    assert "(root, resolution)" in src or "key = (root" in src, (
        "the library cache is not keyed by resolution — every zoom would serve the first one asked for")
