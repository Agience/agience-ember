"""Ember's share of the pattern runner: it registers the host seams.

The loader lives in `prism.runner`. Which module fills a seam is the host's answer, and ember is a
host — so `prism.runner._HOST_SEAMS` is an injected registry and ember's `runtime/seams.py` is what
injects into it.

A seam nobody injects is a silent downgrade rather than an error: the bundle falls back to its own
honest default (`operators.select_for` answers `basis="generic"`), which looks like working software.
Nothing else in either suite notices a dropped registration, so it is pinned here.
"""
from __future__ import annotations

import importlib

import prism.runner as prism_runner


def test_importing_embers_runner_registers_embers_match_seam():
    """Ember binds `match` to its own ontology module, on the loader, at import."""
    import ember.runtime.runner  # noqa: F401  — importing is what registers
    assert prism_runner.registered_seams().get("match") == "ember.ontology.match"


def test_the_seam_is_registered_on_the_loader_not_on_embers_shim():
    """The registration lands where `load()` reads it.

    Ember's module is a re-export shim with a PEP 562 `__getattr__`: reads forward to prism, writes
    stay local. A registration that set state on the shim would look identical from ember's side and
    be invisible to the loader, so the assertion is on prism's map rather than ember's view of it.
    """
    import ember.runtime.runner as ember_runner
    assert prism_runner._HOST_SEAMS.get("match") == "ember.ontology.match"
    # And ember's shim reads the same value THROUGH prism rather than shadowing it.
    assert ember_runner.registered_seams() == prism_runner.registered_seams()


def test_registration_survives_a_reimport_and_does_not_duplicate():
    """Re-importing re-registers idempotently — last-writer-wins per name, one entry."""
    import ember.runtime.runner as ember_runner
    before = dict(prism_runner.registered_seams())
    importlib.reload(ember_runner)
    assert prism_runner.registered_seams() == before


def test_an_unregistered_seam_is_absent_rather_than_guessed():
    """A seam no host registered has no target; the loader states the absence rather than inventing
    one."""
    assert "no_such_seam" not in prism_runner.registered_seams()


# ── the whole table, bound at import of the `ember` package ──────────────────────────────────────

def test_importing_EMBER_binds_every_seam_it_fills():
    """Holding ember at all is enough to bind every seam ember fills.

    Chorus modules name `ember.{ontology.match, ontology.activation, signal.projection,
    runtime.delegate}` as declarers rather than importing them, and a declarer is served by whatever
    the host bound — so the binding happens for any process holding ember, not only one that has
    reached the loader.

    An unfilled seam leaves a persona running: `sage/content_search` answers `{"reach":
    "unavailable"}` and `reach_provider` answers `_basis = None`. That is correct on a host that
    fills nothing and wrong on ember, and nothing else in either suite reads the difference — so the
    table is measured here.

    This goes red if a name leaves `HOST_SEAMS`, if `ember/__init__.py` stops calling
    `register_host_seams()`, or if a target is renamed without the table following.
    """
    import ember  # noqa: F401  — importing the package is what registers
    from ember.runtime.seams import HOST_SEAMS

    assert set(HOST_SEAMS) == {"match", "activation", "projection", "delegate",
                           "forgetting", "optics"}, HOST_SEAMS
    bound = prism_runner.registered_seams()
    for name, target in HOST_SEAMS.items():
        assert bound.get(name) == target, (name, bound.get(name), target)


def test_every_seam_target_ACTUALLY_IMPORTS_and_carries_what_the_declarers_read():
    """`register_seam` stores text, so a dotted string binds cleanly whether or not it names a real
    module. A typo, or a module renamed by a later move, surfaces only when a persona reaches for it
    — in another repo, with no mention of this table in the traceback.

    So each target is imported, and each is checked for the attributes the declarers read. The
    attribute list is the measured read surface at the chorus call sites: it is what makes a rename
    of `tekton_basis_for` land here rather than there.

    This goes red on a target that does not import, and on one that imports but has lost a member a
    declarer reads.
    """
    import importlib

    import ember  # noqa: F401
    from ember.runtime.seams import HOST_SEAMS

    #: seam -> what chorus reads off it, measured at the call sites.
    READ = {
        "match": ("propagate", "_offers", "signal_offers", "tekton_basis_for",
                  "fired_field", "expand_associative", "offer_synsets"),
        "activation": ("spread_seeds", "seeds_from_text", "recognize", "compose",
                       "vertex_field", "output_membrane"),
        "projection": ("frame", "coherent", "read_cloud", "read_basis", "read_unit_contexts"),
        "delegate": ("Delegate",),
        # the reader's working memory. `astra/reading/organon_reader.py` reads one name off it —
        # the class — because the decay that decides how far a co-occurrence reaches is the
        # Screen's own measurement rather than a parameter the declarer passes in.
        "forgetting": ("Screen",),
        # the instrument, reached by name because `ember/optics.py` is the only module that imports
        # entroptics. `astra/reading/` (ingest) and `lumen/reading/` (reasoning) both read the
        # ordered-stream operator and the shuffled control off it.
        "optics": ("sequence_operator", "surrogate_significance", "embed"),
    }
    assert set(READ) == set(HOST_SEAMS), "the read-surface list and the seam table disagree"

    for name, target in HOST_SEAMS.items():
        mod = importlib.import_module(target)              # raises if the string is wrong
        missing = [a for a in READ[name] if not hasattr(mod, a)]
        assert not missing, (
            "seam %r is bound to %r, which imports but is missing %s — a declarer reading those "
            "would degrade silently on a host that thinks it filled the seam"
            % (name, target, missing))


def test_registering_a_seam_imports_NOTHING():
    """Registration is dotted strings, resolved by the declarer at the point of use: a process that
    binds `projection` and never reaches it has not paid for numpy. A fresh interpreter is the only
    honest way to ask, so the check runs in a subprocess.

    This goes red if `register_host_seams()` ever resolves its targets eagerly.
    """
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "from prism.runner import register_seam, registered_seams\n"
         "from ember.runtime.seams import HOST_SEAMS, register_host_seams\n"
         "register_host_seams()\n"
         "assert registered_seams()['projection'] == 'ember.signal.projection'\n"
         "print('LOADED', [t for t in HOST_SEAMS.values() if t in sys.modules])\n"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "LOADED []" in r.stdout, (
        "registering the seams imported their targets: %s" % r.stdout)
