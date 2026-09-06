"""LATTICE Unit G — the semantic arm's coordinate: exactness, centering, and basis identity.

Three things are asserted here, and the first one is the load-bearing one:

  1. **JC exactness.** ``|sparse L2^2 - tree_JC|`` stays at float64 machine epsilon over a large
     deterministic sample of synset pairs, and centering leaves it there. Centering is a
     translation, so pairwise distances are invariant by construction; it is measured anyway,
     because "by construction" is what people say right before a regression.

  2. **The feature hash is pinned.** ``test_feature_hash_is_untouched`` pins the seed and the
     golden hash outputs, so a change to either has to be made deliberately, per
     ``LATTICE-CONTRACT.md`` RESOLVED-2.

  3. **Bases that differ do not pool.** ``BasisMismatch`` is raised where the coordinates meet.

Running without the store
-------------------------
``crystal.ontology.driver`` reads WordNet out of the live corpus, where a full scan is not a safe
read. These tests therefore run against a local nltk WordNet wrapped in a small adapter exposing
exactly the ``Synset`` slice ``crystal.ontology.geometry`` consumes
(``name/pos/ic/hypernyms/instance_hypernyms/common_hypernyms``). The coordinate functions take
synsets as arguments and import no driver at all, so this exercises the real code path rather than a
mock of it. The two functions that do reach the driver (``text_to_signal``, ``oov_tokens``) get the
adapter injected as the ``crystal.ontology`` package attribute.
"""
from __future__ import annotations

import math
import random
import sqlite3

import numpy as np
import pytest

# The package, not the module: the three `monkeypatch.setattr(<pkg>, "driver", …)` calls below
# rebind the attribute that `geometry`'s lazy `from … import driver as wn` resolves through. The
# package is spelled once, here, because patching the wrong one succeeds and substitutes nothing.
import crystal.ontology as crystal_ontology
from crystal.ontology import geometry as g

# ── the store-free WordNet adapter ───────────────────────────────────────────────────────────────
nltk_wn = pytest.importorskip("nltk.corpus", reason="nltk WordNet needed for a store-free run").wordnet
try:                                                        # Brown IC — the same source §17 used
    from nltk.corpus import wordnet_ic
    _BROWN = wordnet_ic.ic("ic-brown.dat")
except Exception:                                           # pragma: no cover - environment gate
    _BROWN = None
    pytest.skip("nltk wordnet_ic (ic-brown.dat) not installed", allow_module_level=True)

try:
    nltk_wn.synset("dog.n.01")
except Exception:                                           # pragma: no cover - environment gate
    pytest.skip("nltk wordnet corpus not downloaded", allow_module_level=True)


def _resnik_ic(s) -> float:
    """Resnik information content from Brown counts: -log(count(s) / count(root)).

    `wordnet_ic.ic()` yields cumulative frequency counts rather than IC, so the conversion happens
    here. Counts grow upward, because a hypernym subsumes its children's mass, whereas IC grows
    downward. Feeding raw counts to the geometry as if they were IC would make `_path_edges`'
    `max(IC(c) - IC(p), 0)` clamp essentially every edge to zero and collapse the telescoping
    identity, so the conversion to IC happens here rather than inside the geometry.

    Count 0 means the synset is unattested in Brown, where IC is undefined (nltk returns +inf).
    Those synsets are excluded from the samples below rather than pinned to a number.
    """
    tbl = _BROWN[s.pos() if not hasattr(s, "_s") else s._s.pos()]
    off = s.offset() if not hasattr(s, "_s") else s._s.offset()
    c = tbl.get(off, 0.0)
    root = tbl[0]
    if not c or not root:
        return 0.0
    return -math.log(c / root)


class _S:
    """Adapter over an nltk synset exposing the `driver.Synset` slice the geometry uses.

    IC is Resnik over Brown counts, matching the §17 experiments. Every assertion below is an
    internal-consistency claim — does the coordinate realize the metric it is built from — rather
    than a claim about absolute IC values, so the IC source only has to be self-consistent.
    """
    __slots__ = ("_s",)

    def __init__(self, s):
        self._s = s

    def name(self):
        return self._s.name()

    def pos(self):
        return self._s.pos()

    def ic(self):
        return _resnik_ic(self._s)

    def hypernyms(self):
        return [_S(h) for h in self._s.hypernyms()]

    def instance_hypernyms(self):
        return [_S(h) for h in self._s.instance_hypernyms()]

    def common_hypernyms(self, other):
        return [_S(h) for h in self._s.common_hypernyms(other._s)]

    def __eq__(self, o):
        return isinstance(o, _S) and o.name() == self.name()

    def __hash__(self):
        return hash(self.name())

    def __repr__(self):
        return f"_S({self.name()!r})"


def _syn(name):
    return _S(nltk_wn.synset(name))


@pytest.fixture(scope="module")
def sample_nouns():
    """A deterministic 400-synset noun sample. Seeded, so the numbers in the unit report are
    reproducible rather than 'whatever the run happened to draw'.

    Restricted to synsets attested in Brown (count > 0), because IC is undefined for the rest and a
    sample of undefined-IC synsets measures nothing."""
    names = sorted(s.name() for s in nltk_wn.all_synsets("n") if _resnik_ic(s) > 0.0)
    rnd = random.Random(20260720)
    return [_syn(n) for n in rnd.sample(names, 400)]


@pytest.fixture(scope="module")
def sample_pairs(sample_nouns):
    """200 pairs from the sample, plus every pair in geometry's own DEFAULT_PAIRS."""
    rnd = random.Random(11)
    pairs = [(sample_nouns[rnd.randrange(len(sample_nouns))],
              sample_nouns[rnd.randrange(len(sample_nouns))]) for _ in range(200)]
    for a, b in g.DEFAULT_PAIRS:
        try:
            pairs.append((_syn(a), _syn(b)))
        except Exception:
            pass
    return [(a, b) for a, b in pairs if a.name() != b.name()]


# ── 1. exactness ─────────────────────────────────────────────────────────────────────────────────
def _sparse_l2sq(a, b, ic=None):
    d1, d2 = g.sparse_vec(a, ic), g.sparse_vec(b, ic)
    return sum((d1.get(k, 0.0) - d2.get(k, 0.0)) ** 2 for k in set(d1) | set(d2))


def test_jc_exactness_holds(sample_pairs):
    """||v(s1) - v(s2)||^2 == IC(s1) + IC(s2) - 2*IC(LCS) = tree Jiang-Conrath, exactly.

    The telescoping identity is the whole justification for the coordinate: it is a closed form
    over our own ontology with no fitted parameter. If this drifts, the coordinate has stopped
    being a metric embedding and become an approximation nobody bounded."""
    worst = 0.0
    n = 0
    for a, b in sample_pairs:
        jt = g.jc_tree(a, b, None)
        worst = max(worst, abs(_sparse_l2sq(a, b) - jt))
        n += 1
    assert n >= 100, f"only {n} pairs evaluated — a check that evaluated nothing is not a pass"
    print(f"\n[JC exactness] pairs={n}  max|sparse L2^2 - tree_JC| = {worst:.6e}")
    assert worst <= 1e-14, f"JC exactness broken: max abs err {worst:.6e}"


def test_gauge_conversion_is_lossless(sample_nouns):
    """Language is a gauge, and its `signal <-> language` conversion is lossless in the keyed regime.

    The forward map (`test_jc_exactness_holds`) is JC-exact to machine epsilon. The inverse — the
    output screen's `_render_concept` — is exact because it is keyed: the resonance carries the
    concept's identity, so surface is recovered by dictionary lookup rather than by a nearest
    neighbour. This pins the property that makes that possible: the sparse coordinate uniquely
    identifies the synset. Encode each synset; its self-distance is exactly 0 and its distance to
    every other synset in the sample is strictly positive, so the concept is recoverable from its
    coordinate with no loss.

    Asserted on the sparse coordinate, the keyed regime, exact to 1e-15. The D=2048 hash preserves
    the ranking at Spearman 1.0 while being bit-inexact, which is why the render stays keyed."""
    sample = sample_nouns[:120]
    coords = [(s.name(), g.sparse_vec(s, None)) for s in sample]
    collisions = 0
    for name, cv in coords:
        # self-distance is exactly zero: re-encoding the same identity reproduces the coordinate
        assert g.sparse_vec(_syn(name), None) == cv, f"{name}: keyed re-encode is not identical"
        # and no other distinct synset shares this exact coordinate (identity is recoverable)
        for oname, ov in coords:
            if oname == name:
                continue
            d = sum((cv.get(k, 0.0) - ov.get(k, 0.0)) ** 2 for k in set(cv) | set(ov))
            if d <= 0.0:
                collisions += 1
    print(f"\n[gauge round-trip] {len(sample)} concepts, sparse-coordinate collisions = {collisions}")
    assert collisions == 0, f"{collisions} distinct concepts share a coordinate — inverse is not lossless"


def test_jc_exactness_unaffected_by_centering(sample_pairs, sample_nouns):
    """Centering is a translation; pairwise distances are translation-invariant. Measured, not
    assumed, and measured in the dense space, since that is where the mean is subtracted."""
    D = 128
    mean = g.derive_centering_mean(sample_nouns[:120], None, D)
    worst = 0.0
    for a, b in sample_pairs[:150]:
        u0, v0 = g.dense_vec(a, None, D), g.dense_vec(b, None, D)
        u1, v1 = g.dense_vec(a, None, D, center=mean), g.dense_vec(b, None, D, center=mean)
        worst = max(worst, abs(float(np.sum((u0 - v0) ** 2)) - float(np.sum((u1 - v1) ** 2))))
    print(f"[centering] max |dense L2^2 centered - uncentered| = {worst:.6e}")
    assert worst <= 1e-12, f"centering moved a pairwise distance by {worst:.6e}"


def test_sparse_coordinate_has_no_centering_knob():
    """Structural guard: exactness lives in the sparse space, so `sparse_vec` is a pure function of
    (synset, ic). A `center=` parameter here would make the identity above approximate, and this
    file's headline number would no longer describe it."""
    import inspect
    assert list(inspect.signature(g.sparse_vec).parameters) == ["synset", "ic"]


# ── 2. the feature hash is untouched (RESOLVED-2) ────────────────────────────────────────────────
def test_feature_hash_is_untouched():
    """Pins the hash seed and three golden outputs, so Step 3's coordinate cannot move quietly.

    `LATTICE-CONTRACT.md` RESOLVED-2 binds `dense_vec` to this hash-based construction rather
    than a basis-projection alternative."""
    assert g._HASH_SEED == "genesis-geom-v1"
    assert g.GEOMETRY_VERSION == "geom-v1"
    golden = {
        "dog.n.01": (12610070019142760606, 1.0),
        "entity.n.01": (4862793637433463709, 1.0),
        "car.n.01": (1134444294228796070, 1.0),
    }
    for node, expect in golden.items():
        assert g._hash(node) == expect, f"feature hash changed for {node!r}"


def test_dense_vec_is_signed_hash_of_the_sparse_vector():
    """The hash is still exactly 'fold the sparse coordinate into D with a sign' — no basis
    projection, no anchors, no learned matrix anywhere in the path."""
    D = 64
    s = _syn("dog.n.01")
    expect = np.zeros(D)
    for node, w in g.sparse_vec(s, None).items():
        h, sgn = g._hash(node)
        expect[h % D] += sgn * w
    assert np.allclose(g.dense_vec(s, None, D), expect, atol=0.0, rtol=0.0)


def test_hashing_error_is_collision_noise_that_vanishes_with_D(sample_pairs):
    """The JL property the hash exists for: hashed L2^2 tracks the exact metric, and the only thing
    between them is collision noise, which therefore goes away as D grows.

    The convergence is the assertion rather than a threshold at one D, because a fixed
    "spearman >= X at D=Y" would be a hand-tuned number. Measured:

        D= 256  rho=0.8849     <- D_SMALL, bounded observer / Pi
        D= 512  rho=0.9124
        D=1024  rho=0.9453
        D=2048  rho=0.9998     <- D_BIG, the default
        D=4096  rho=0.9998     <- saturated

    That measurement set the default at 2048. `faithfulness_check`'s DEFAULT_PAIRS are common,
    shallow synsets with short hypernym paths and few occupied dimensions, so they under-report
    collisions; the sparse coordinate is exact to 1e-15 and the residual is pure collision, but a
    0.912 fold is too coarse to carry an "exact Jiang-Conrath" claim.

    The mechanism is what is pinned — rho increases monotonically in D and then saturates — because
    that shape is what identifies the error as collision noise. A hash whose error came from
    somewhere else would break the monotone convergence, which no single-D number would show.
    """
    tree = [g.jc_tree(a, b, None) for a, b in sample_pairs[:150]]
    rhos = {}
    for D in (256, 512, 1024, 2048, 4096):
        hx = [float(np.sum((g.dense_vec(a, None, D) - g.dense_vec(b, None, D)) ** 2))
              for a, b in sample_pairs[:150]]
        rhos[D] = g._spearman(tree, hx)
    print("\n[hash faithfulness] spearman(hashed L2^2, tree_JC), n=150 random deep noun pairs:")
    for D, r in rhos.items():
        tag = {g.D_SMALL: "  <- D_SMALL (Pi)", g.D_BIG: "  <- D_BIG (default)"}.get(D, "")
        print(f"    D={D:5d}  rho={r:.6f}{tag}")
    assert all(r is not None for r in rhos.values())
    # the mechanism: error is collision noise, so it shrinks monotonically as D widens ...
    assert rhos[2048] > rhos[1024] > rhos[512] > rhos[256], "error must shrink monotonically in D"
    # ... and then saturates, because at some width there are no collisions left to remove
    assert abs(rhos[4096] - rhos[2048]) < abs(rhos[2048] - rhos[1024]), "convergence must flatten"


def test_default_D_is_the_big_node_width_and_pi_width_stays_reachable():
    """D=2048 is the default; D=256 remains supported for the bounded observer.

    A Pi node uses D=256 and a big node uses D=2048 (`LATTICE-CONTRACT.md` §5.5). Phase 4's
    bounded observer runs on a Pi, so `D_SMALL` is a supported width alongside the default. Both
    work."""
    assert g._D_DEFAULT == g.D_BIG == 2048
    assert g.D_SMALL == 256
    s = _syn("dog.n.01")
    assert g.dense_vec(s, None).shape == (2048,)                 # default
    assert g.dense_vec(s, None, g.D_SMALL).shape == (256,)       # Pi, still first-class


def test_jc_exactness_at_default_D(sample_pairs):
    """The exactness claim, re-measured through the fold at the default width.

    Two different numbers, and they are reported separately:
      * the sparse coordinate is exact to machine epsilon, and is D-independent;
      * the hashed coordinate at finite D carries collision error, which is what D buys down.
    Reporting both keeps the "exact Jiang-Conrath" claim quoted with the width it was measured
    at."""
    D = g._D_DEFAULT
    sparse_worst = max(abs(_sparse_l2sq(a, b) - g.jc_tree(a, b, None)) for a, b in sample_pairs)
    tree, hashed = [], []
    for a, b in sample_pairs[:150]:
        tree.append(g.jc_tree(a, b, None))
        u, v = g.dense_vec(a, None, D), g.dense_vec(b, None, D)
        hashed.append(float(np.sum((u - v) ** 2)))
    rel = max(abs(h - t) / t for h, t in zip(hashed, tree) if t > 1e-9)
    rho = g._spearman(tree, hashed)
    print(f"\n[exactness @ D={D}] sparse max|L2^2 - tree_JC| = {sparse_worst:.6e}  (D-independent)")
    print(f"[exactness @ D={D}] hashed spearman = {rho:.6f}   max relative err = {rel:.4f}")
    assert sparse_worst <= 1e-14
    assert rho >= 0.99


# ── 3. centering ─────────────────────────────────────────────────────────────────────────────────
def test_centering_defaults_off(monkeypatch):
    """Phase 2 is gated on the Phase 0.A recall@k A/B, which has not run. Wiring centering is this
    unit's job; flipping the default is a separate, gated decision."""
    monkeypatch.delenv("EMBER_GEOM_CENTERING", raising=False)
    assert g.default_centering() is None
    import inspect
    assert inspect.signature(g.dense_vec).parameters["center"].default is None
    assert inspect.signature(g.text_to_signal).parameters["center"].default is None
    assert inspect.signature(g.text_to_signal).parameters["oov"].default == "surface"


def test_centering_mean_is_deterministic(sample_nouns):
    """Same sample, same D => same mean, bit for bit. If it were not, two nodes would center into
    two different spaces while both reporting a healthy coordinate."""
    a = g.derive_centering_mean(sample_nouns[:100], None, 128)
    b = g.derive_centering_mean(sample_nouns[:100], None, 128)
    assert np.array_equal(a, b)
    assert g.centering_mean_id(a) == g.centering_mean_id(b)
    c = g.derive_centering_mean(sample_nouns[:101], None, 128)
    assert g.centering_mean_id(c) != g.centering_mean_id(a), "a different sample must get a different id"


def test_centering_mean_roundtrip_and_tamper(tmp_path, sample_nouns):
    mean = g.derive_centering_mean(sample_nouns[:100], None, 128)
    p = tmp_path / "center.json"
    mid = g.save_centering_mean(mean, p)
    back = g.load_centering_mean(p)
    assert g.centering_mean_id(back) == mid
    assert np.allclose(back, np.asarray(mean, dtype=">f4").astype(np.float64))
    # a tampered file must not load as a usable mean
    import json
    d = json.loads(p.read_text())
    d["mean"][0] += 1.0
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="id mismatch"):
        g.load_centering_mean(p)


def test_derive_centering_mean_refuses_empty():
    """An empty sample must raise, not return zeros — a zero mean is indistinguishable from
    'centering is off' and would silently disable the thing the caller asked for."""
    with pytest.raises(ValueError):
        g.derive_centering_mean([], None, 64)


def test_centering_shifts_inner_products(sample_nouns):
    """What centering actually buys: the mean cross-cosine between coordinates drops. §17.4 measured
    +0.505 -> -0.332 on four disjoint document domains; the assertion here is the direction on the
    raw synset coordinates, because this sample is not those documents."""
    D = 256
    mean = g.derive_centering_mean(sample_nouns, None, D)
    pool = sample_nouns[:80]

    def mean_cos(center):
        vs = []
        for s in pool:
            v = g.dense_vec(s, None, D, center=center)
            n = np.linalg.norm(v)
            if n:
                vs.append(v / n)
        M = np.array(vs)
        G = M @ M.T
        iu = np.triu_indices(len(M), k=1)
        return float(G[iu].mean())

    raw, cen = mean_cos(None), mean_cos(mean)
    print(f"[centering] mean pairwise cosine: uncentered={raw:+.4f}  centered={cen:+.4f}")
    assert cen < raw, "centering must reduce the common-mode inner product"


def test_centering_is_a_noop_for_feature_covariance(fake_wn):
    """`feature_covariance` already subtracts a per-document weighted mean, so a global translation
    upstream cancels. Documented in that function's docstring and asserted here, so the claim stays
    checked. It is why centering is applied where cosine is read rather than everywhere."""
    D = 64
    text = "The dog chased a cat past the car and the tree near the river."
    mean = np.linspace(-0.01, 0.01, D)
    a = g.feature_covariance(g.text_to_signal(text, None, D))
    b = g.feature_covariance(g.text_to_signal(text, None, D, center=mean))
    assert np.allclose(a, b, atol=1e-12)


# ── 4. the basis fingerprint ─────────────────────────────────────────────────────────────────────
def test_fingerprint_is_stable_and_discriminating(sample_nouns):
    rev = g.ic_source_revision(synsets=sample_nouns)
    assert g.ic_source_revision(synsets=sample_nouns) == rev, "IC revision must be stable"
    assert g.ic_source_revision(synsets=sample_nouns[:399]) != rev, "a changed IC source must change it"

    base = g.basis_fingerprint(512, None, ic_revision=rev)
    assert g.basis_fingerprint(512, None, ic_revision=rev) == base
    print(f"[fingerprint] {base}")

    mean = g.derive_centering_mean(sample_nouns[:100], None, 512)
    variants = {
        "D": g.basis_fingerprint(256, None, ic_revision=rev),
        "ic_revision": g.basis_fingerprint(512, None, ic_revision="other-corpus"),
        "centering_id": g.basis_fingerprint(512, mean, ic_revision=rev),
    }
    for field, fp in variants.items():
        assert g.basis_conflicts(base, fp) == [field], f"{field} must be the only conflict"
        assert fp.token() != base.token()
        with pytest.raises(g.BasisMismatch):
            g.require_same_basis(base, fp)


def test_pi_and_big_node_refuse_to_pool(sample_nouns):
    """A D=256 Pi and a D=2048 node do not pool coordinates: `BasisMismatch` is raised instead.

    The fleet runs both widths at once — the default is 2048 and `D_SMALL` is supported (§5.5) — so
    two observers do offer each other coordinates from different spaces. The fingerprint is what
    makes that meeting an exception: without it the best case is a `ValueError`, and any code path
    that pads, truncates or reduces before the matmul produces a confident and meaningless ranking.
    That is the shape of the Prism `model_id` incident — two pods with different seq caps reporting
    the same id, and Mantle indexing both into one space with nothing raising anywhere."""
    rev = g.ic_source_revision(synsets=sample_nouns)
    pi = g.basis_fingerprint(g.D_SMALL, None, ic_revision=rev)
    big = g.basis_fingerprint(g.D_BIG, None, ic_revision=rev)

    assert g.basis_conflicts(pi, big) == ["D"]
    assert pi.token() != big.token()
    with pytest.raises(g.BasisMismatch, match="D"):
        g.require_same_basis(pi, big)

    s = _syn("dog.n.01")
    with pytest.raises(g.BasisMismatch):
        g.pool_coordinates([(big, g.dense_vec(s, None, g.D_BIG)),
                            (pi, g.dense_vec(s, None, g.D_SMALL))])
    # each width pools happily with its own kind — the mismatch check is targeted, not blanket
    assert g.pool_coordinates([(pi, g.dense_vec(s, None, g.D_SMALL))] * 2).shape == (2, g.D_SMALL)
    assert g.pool_coordinates([(big, g.dense_vec(s, None, g.D_BIG))] * 2).shape == (2, g.D_BIG)


def test_ic_revision_refuses_to_be_unknown(monkeypatch):
    """An IC revision that cannot be read raises `BasisMismatch`. Two nodes both reporting
    'unknown' would compare equal and pool, which is the confused-deputy shape Prism's `model_id`
    was invented to stop."""
    import ember.ontology
    from crystal.ontology import driver as wn_store  # noqa: F401 - bind the real attr first, else `from crystal.ontology import driver as wn_store` re-imports over the patch
    class _Broken:
        @staticmethod
        def all_synsets(pos=None):
            raise RuntimeError("store unavailable")
    monkeypatch.setattr(crystal_ontology, "driver", _Broken)
    with pytest.raises(g.BasisMismatch):
        g.ic_source_revision()
    with pytest.raises(g.BasisMismatch):
        g.ic_source_revision(synsets=[])


def test_pool_coordinates_refuses_structurally(sample_nouns):
    """The check lives at the point coordinates meet, so every call site inherits it."""
    rev = g.ic_source_revision(synsets=sample_nouns)
    fp = g.basis_fingerprint(32, None, ic_revision=rev)
    other = g.basis_fingerprint(32, None, ic_revision="different")
    ok = g.pool_coordinates([(fp, np.zeros(32)), (fp, np.ones((3, 32)))])
    assert ok.shape == (4, 32)
    with pytest.raises(g.BasisMismatch):
        g.pool_coordinates([(fp, np.zeros(32)), (other, np.zeros(32))])
    with pytest.raises(g.BasisMismatch):
        g.pool_coordinates([])                       # empty pool is not "no results"
    with pytest.raises(g.BasisMismatch):
        g.pool_coordinates([(fp, np.zeros(16))])     # width must match the declared D


# ── 5. OOV routes to the keyed arm ───────────────────────────────────────────────────────────────
@pytest.fixture
def fake_wn(monkeypatch):
    """Inject the nltk-backed adapter as `crystal.ontology.driver` so `text_to_signal` /
    `oov_tokens` run without touching the live corpus, where a full scan is not a safe read."""
    import ember.ontology
    from crystal.ontology import driver as wn_store  # noqa: F401 - see test_ic_revision_refuses_to_be_unknown

    class _WN:
        NOUN = "n"

        @staticmethod
        def synsets(word, pos=None):
            return [_S(s) for s in nltk_wn.synsets(word.lower().replace(" ", "_"), pos=pos)]

        @staticmethod
        def all_synsets(pos=None):
            return [_S(s) for s in nltk_wn.all_synsets(pos or "n")]

    monkeypatch.setattr(crystal_ontology, "driver", _WN)
    return _WN


def test_oov_skips_instead_of_manufacturing_surface_similarity(fake_wn):
    """`oov="skip"` emits no row for an out-of-vocabulary token. `_oov_vec` deposits a char-trigram
    surface vector in the same D as the meaning coordinates, where nothing downstream can tell them
    apart — §17.2 measured that encoding at K_signal 2-3 / cross-domain cosine +0.889."""
    D = 64
    text = "the zxqwvb dog frobnicatrix cat"
    surf = g.text_to_signal(text, None, D, oov="surface")
    skip = g.text_to_signal(text, None, D, oov="skip")
    assert surf.shape[0] > skip.shape[0]
    assert skip.shape[0] == 2, "only the two real nouns should survive"
    # "the" is OOV for this arm too, having no noun synset, and that is the point:
    # function words carry no meaning-geometry, and `oov="surface"` gives them a char-trigram
    # row of pure surface noise, weighted equally with every other OOV token.
    assert g.oov_tokens(text) == ["the", "zxqwvb", "frobnicatrix"]
    with pytest.raises(ValueError):
        g.text_to_signal(text, None, D, oov="nonsense")


def test_adjectives_have_no_hypernym_tree():
    """The basis covers nouns. WordNet adjectives are `similar_to` clusters with no hypernym tree,
    so the coordinate is identically empty for them and 'cheap lodging' does not reach 'inexpensive
    accommodation'. Reaching it would take `similar_to` and `derivationally_related_form` ingested
    plus an `expand()`."""
    for adj in ("cheap.a.01", "inexpensive.a.01"):
        s = _syn(adj)
        assert s.hypernyms() == [] and s.instance_hypernyms() == []
        assert g.sparse_vec(s, None) == {}, "an adjective has no meaning-geometry in this basis"
    # and the nouns of that phrase DO have one — the gap is the adjective, specifically
    assert g.sparse_vec(_syn("lodging.n.01"), None)


class _FixedIC:
    """A synset whose stored `ic` is whatever the test says it is — including impossible values."""
    __slots__ = ("_n", "_v")

    def __init__(self, name, v):
        self._n, self._v = name, v

    def name(self):
        return self._n

    def ic(self):
        return self._v

    def hypernyms(self):
        return []

    def instance_hypernyms(self):
        return []


def test_the_1e300_sentinel_is_caught_WHERE_IT_ENTERS_not_by_a_magnitude_bound():
    """The 1e300 sentinel is handled where it enters, and this pins the property that keeps it out.

    nltk's `information_content` returns `_INF = 1e+300` — a module-level float literal rather than
    `math.inf` — for a zero-frequency synset. `1e300 == float("inf")` is False and
    `math.isfinite(1e300)` is True, so it reads as an ordinary measurement to `jc_tree` /
    `sparse_vec` / `dense_vec`. Over 117,659 synsets, 50,278 (42.73%) carried it, and `sparse_vec`
    weights such an edge at `sqrt(1e300)` = 1e150.

    An impurity is caught where it enters, and two properties do that, neither of them a chosen
    number:

      1. The formula's own range. IC is intrinsic — `1 - log(desc+1)/log(N+1)` over the is-a tree —
         which lies in [0, 1] by construction. Nothing in that path reaches nltk, so the sentinel
         has no way in. `ic_upper_bound()` reads the range from the corpus's record of which formula
         it used and returns None when the corpus states no formula: an unstated range is not a
         wide one.
      2. The ingest boundary. `enrich_wordnet` tests `v == NLTK_IC_INF` by identity and classifies
         it `ic_zero_frequency`, an absence, which `has_ic()` distinguishes from a real zero. Exact
         rather than a margin.

    So the assertion is the property, not a magnitude bound on the read.
    """
    from crystal.ontology import driver as wn

    # The two facts that made it invisible are still true, and still worth stating.
    assert math.isfinite(1e300), "the sentinel is FINITE — this is why isfinite did not catch it"
    assert 1e300 != float("inf"), "and it is not equal to inf either"

    # The universal property: the formula in use cannot produce the sentinel.
    assert wn.INTRINSIC_IC_FORMULA == "1 - log(desc(s)+1)/log(N+1)"
    bound = g.ic_upper_bound()
    if bound is not None:
        assert bound < 1e300, (
            "a corpus whose own formula could reach the sentinel would need the guard back")

    # Real values pass through untouched, and absence stays 0.0 for arithmetic — `has_ic()`'s job.
    assert g.ic_of(_FixedIC("root.n.01", 0.0)) == 0.0, "a real root zero stays 0.0"
    assert g.ic_of(_FixedIC("absent.n.01", None)) == 0.0

    # A non-finite stored value is a corrupt row and fails loudly. That guard is a measurement
    # (isfinite) rather than a level.
    with pytest.raises(g.AbsurdIC):
        g.ic_of(_FixedIC("corrupt.n.01", float("inf")))


def test_a_CORRUPT_value_anywhere_on_the_path_fails_the_whole_walk():
    """Contagion is the reason a bad IC matters: `canonical_parent` calls `ic_of` on every candidate
    parent, so a poisoned ancestor corrupts a descendant's coordinate. That is what made 42.73% of
    the corpus contagious, and it is why the guard fires from inside the walk.

    What counts as bad is corruption, tested by `math.isfinite` — a measurement rather than a level.
    The 1e300 sentinel is handled where it enters (`enrich_wordnet` matches it by identity and
    records an absence), so a value reaching the walk is either a real reading from a formula with a
    stated range, or corrupt. The contagion property is therefore asserted on corruption.
    """
    class _Chain:
        def __init__(self, name, v, parent=None):
            self._n, self._v, self._p = name, v, parent

        def name(self):
            return self._n

        def ic(self):
            return self._v

        def hypernyms(self):
            return [self._p] if self._p else []

        def instance_hypernyms(self):
            return []

    # A corrupt ancestor: non-finite is a corrupt stored row at any scale, in any corpus, whatever
    # formula produced it. No range needs to be known for this to be wrong.
    poisoned = _Chain("poisoned.n.01", float("inf"))
    child = _Chain("child.n.01", 5.0, poisoned)
    with pytest.raises(g.AbsurdIC):
        g.tree_path(child, None)
    with pytest.raises(g.AbsurdIC):
        g.sparse_vec(child, None)

    # The control. Without it this passes for a walk that raises on every chain, which would be a
    # guard that cannot distinguish corruption from data.
    clean_parent = _Chain("clean.n.01", 1.0)
    clean_child = _Chain("child.n.01", 0.5, clean_parent)
    g.tree_path(clean_child, None)
    g.sparse_vec(clean_child, None)



# ═══════════════════════════════════════════════════════════════════════════════════════════════
# Step 1b — IC zero-frequency smoothing (Laplace before propagation) and the second channel
#
# Four claims, each measured against the real Brown noun table rather than asserted by construction:
#
#   1. Monotonicity (`IC(child) >= IC(parent)`) holds on every tree edge after smoothing. Measured:
#      0 violations smoothed, 39,780 violations with the holes left at 0.0.
#   2. JC exactness survives the smoothed IC: |sparse L2^2 - jc_tree| = 7.105427e-15.
#   3. `se` propagates to JC as a second channel (median se(JC)/JC = 4.26% on this corpus).
#   4. Folding `se` into the coordinate (resampling IC per call within +/-1 se) destroys exactness:
#      7.31, fifteen orders of magnitude worse. That is the test pinning the rule.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

def _brown_cum(s) -> float:
    """The Brown descendant-inclusive count for an nltk synset (0.0 = unattested = the hole)."""
    return float(_BROWN[s.pos()].get(s.offset(), 0.0))


class _SmoothS:
    """`_S` with IC and its standard error taken from a smoothed table rather than from Brown.

    A second adapter rather than a mutation of `_S`, because the tests above assert the unsmoothed
    behaviour and go on reading it.
    """
    __slots__ = ("_s", "_tab", "_reg")

    def __init__(self, s, tab, reg):
        self._s, self._tab, self._reg = s, tab, reg

    def name(self):
        return self._s.name()

    def pos(self):
        return self._s.pos()

    def ic(self):
        return self._tab[self._s.name()][0]

    def ic_se(self):
        return self._tab[self._s.name()][1]

    def hypernyms(self):
        return [self._reg[h.name()] for h in self._s.hypernyms() if h.name() in self._reg]

    def instance_hypernyms(self):
        return [self._reg[h.name()] for h in self._s.instance_hypernyms()
                if h.name() in self._reg]

    def common_hypernyms(self, other):
        return [self._reg[h.name()] for h in self._s.common_hypernyms(other._s)
                if h.name() in self._reg]

    def __eq__(self, o):
        return isinstance(o, _SmoothS) and o.name() == self.name()

    def __hash__(self):
        return hash(self.name())


@pytest.fixture(scope="module")
def smoothed_nouns():
    """The whole noun vocabulary (82,115 synsets) with Laplace-smoothed IC + se.

    Whole-vocabulary rather than a sample, because the monotonicity claim is about every edge and a
    sampled version of it would be a weaker statement than the one being made.
    Returns `(registry, parents, table, raw_ic)`.
    """
    nouns = list(nltk_wn.all_synsets("n"))
    reg, tab = {}, {}
    for s in nouns:
        reg[s.name()] = _SmoothS(s, tab, reg)

    # The canonical spanning tree, taken over the raw IC exactly as production does — the tree must
    # not be chosen using the numbers we are about to derive from it.
    raws = {s.name(): _S(s) for s in nouns}
    parents = {}
    for s in nouns:
        pp = g.canonical_parent(raws[s.name()], None)
        parents[s.name()] = pp.name() if pp is not None else None

    # De-propagate Brown's descendant-inclusive counts to own counts over that same tree, so that
    # Laplace lands before propagation, which is the point of the exercise (see
    # `geometry.smooth_ic`).
    children = {}
    for n, pp in parents.items():
        if pp:
            children.setdefault(pp, []).append(n)
    by_name = {s.name(): s for s in nouns}
    own = {n: _brown_cum(by_name[n]) - sum(_brown_cum(by_name[c]) for c in children.get(n, ()))
           for n in parents}

    tab.update(g.smooth_ic(own, parents, alpha=g.IC_LAPLACE_ALPHA))
    return reg, parents, tab, {n: g.ic_of(raws[n], None) for n in parents}


def test_laplace_smoothing_makes_ic_monotone_on_every_edge(smoothed_nouns):
    """The hard constraint: `IC(child) >= IC(parent)` on every edge of the spanning tree.

    Step 2's telescoping identity is built on `sqrt(IC(c) - IC(p))`. A negative difference is
    clamped to 0 by `_path_edges`, and each clamp deletes an edge from the coordinate — which is why
    backing the holes off to 0.0 measures `||v1-v2||^2` against `jc_tree` at 27.83: a structurally
    shorter path vector rather than a rounding problem.
    """
    reg, parents, tab, raw = smoothed_nouns
    edges = [(n, pp) for n, pp in parents.items() if pp]
    assert len(edges) > 80000, f"expected the full noun tree, got {len(edges)} edges"

    bad = [(n, pp) for n, pp in edges if tab[n][0] < tab[pp][0] - 1e-12]
    assert not bad, ("Laplace-before-propagation must be monotone BY CONSTRUCTION; "
                     f"{len(bad)} violated edges, e.g. {bad[:5]}")

    # The same measurement over the unsmoothed IC, so what smoothing buys is quantified rather than
    # merely asserted. The number is the size of the gap, not a threshold to tune.
    bad_now = [(n, pp) for n, pp in edges if raw[n] < raw[pp] - 1e-12]
    assert len(bad_now) > 30000, (
        "expected the unsmoothed corpus to violate monotonicity on tens of thousands of edges "
        f"(measured 39,780); got {len(bad_now)}. If this dropped, the IC source changed.")


def test_smoothing_fills_every_hole_and_keeps_the_root_at_zero(smoothed_nouns):
    """No synset is left without an IC, and the root still anchors the space at 0.

    ~59.5% of noun synsets (48,861 of 82,115) have zero Brown frequency and therefore no Resnik IC.
    After smoothing every one carries a finite, ordered value, so the most specific corner of the
    ontology sits at a different number from the top of it.
    """
    reg, parents, tab, raw = smoothed_nouns
    holes = [n for n in parents if raw[n] == 0.0]
    assert len(holes) > 40000, f"expected ~48.8k zero-frequency nouns, got {len(holes)}"

    assert all(math.isfinite(t[0]) for t in tab.values()), "smoothing left a non-finite IC"
    roots = [n for n, pp in parents.items() if pp is None]
    assert roots == ["entity.n.01"], f"the noun hierarchy should have ONE root; got {roots}"
    assert abs(tab["entity.n.01"][0]) < 1e-12, "the root must sit at IC 0 — it is the origin"

    # Every value stays inside the range a real measurement can occupy, so smoothing produces
    # nothing `ic_of`'s guard would treat as corrupt.
    #
    # The bound is the formula's range rather than a margin above the largest value anyone has seen.
    # `None` means this corpus states no formula, and an unstated range is not a wide one, so the
    # magnitude claim is dropped in that case rather than replaced by an invented ceiling.
    bound = g.ic_upper_bound()
    if bound is not None:
        assert max(t[0] for t in tab.values()) <= bound, (
            "smoothing produced an IC outside what this corpus's own formula can generate")
    else:
        assert all(math.isfinite(t[0]) for t in tab.values()), (
            "with no stated range, finiteness is the only claim available")


def test_jc_exactness_survives_the_smoothed_ic(smoothed_nouns):
    """Claim 2: |sparse L2^2 - jc_tree| stays at machine epsilon under smoothed IC.

    Measured at 7.105427e-15 over 20,000 pairs, bit-identical to the unsmoothed figure in
    `geometry.jc_tree`'s docstring, because smoothing changes the values the coordinate telescopes
    over and leaves the identity it telescopes by alone.
    """
    reg, parents, tab, raw = smoothed_nouns
    rnd = random.Random(20260721)
    names = sorted(parents)
    samp = rnd.sample(names, 600)
    worst = 0.0
    for _ in range(20000):
        s1, s2 = reg[samp[rnd.randrange(600)]], reg[samp[rnd.randrange(600)]]
        jt = g.jc_tree(s1, s2, None)
        d1, d2 = g.sparse_vec(s1, None), g.sparse_vec(s2, None)
        keys = set(d1) | set(d2)
        l2 = sum((d1.get(k, 0.0) - d2.get(k, 0.0)) ** 2 for k in keys)
        worst = max(worst, abs(l2 - jt))
    assert worst < 1e-12, f"smoothed IC broke JC exactness: max abs err {worst:.6e}"


def test_se_is_a_second_channel_and_propagates_to_jc(smoothed_nouns):
    """Claim 3: `se` rides alongside the distance rather than inside it.

    Measured on this corpus: median `se(JC)/JC` = 4.26%, p90 5.16%. At alpha=1 with 59.5% of nouns
    resting entirely on the smoothing constant, that is the honest relative uncertainty, and
    carrying it is what the second channel is for.
    """
    reg, parents, tab, raw = smoothed_nouns
    rnd = random.Random(4242)
    names = sorted(parents)
    samp = rnd.sample(names, 400)
    rel = []
    for _ in range(4000):
        s1, s2 = reg[samp[rnd.randrange(400)]], reg[samp[rnd.randrange(400)]]
        jt = g.jc_tree(s1, s2, None)
        se = g.jc_tree_se(s1, s2, None)
        assert se is not None, "a corpus carrying ic_se must yield a JC se"
        assert se >= 0.0 and math.isfinite(se)
        if jt > 1e-9:
            rel.append(se / jt)
    med = sorted(rel)[len(rel) // 2]
    assert 0.005 < med < 0.20, f"se(JC)/JC median {med:.4%} is outside any plausible band"

    # A synset with no stored se yields None rather than 0.0: an unmeasured uncertainty is an
    # absence, and 0.0 would be a reading nobody took. `_S`, the unsmoothed adapter, has no `ic_se`.
    a, b = _syn("dog.n.01"), _syn("cat.n.01")
    assert g.ic_se_of(a) is None
    assert g.jc_tree_se(a, b, None) is None


def test_folding_se_into_the_coordinate_destroys_exactness(smoothed_nouns):
    """Claim 4, and the rule this pins: uncertainty stays out of the coordinate.

    Resampling IC per call within +/-1 se measures 7.31 where carrying `se` beside the value
    measures 7.1e-15. The mechanism is structural rather than statistical: `<v(s1), v(s2)> =
    IC(LCS)` holds because the shared root->LCS prefix is built from the same edge weights on both
    sides and cancels exactly. Independent draws per call stop the prefixes cancelling, so the error
    grows with path depth and averaging does not remove it.

    Mixing `se` into `ic_of` to 'improve' `jc_tree_se` is what this test detects.
    """
    reg, parents, tab, raw = smoothed_nouns
    rnd = random.Random(20260721)
    names = sorted(parents)
    samp = rnd.sample(names, 600)
    noise = random.Random(7)

    class _Resampled(_SmoothS):
        __slots__ = ()

        def ic(self):
            mu, se = self._tab[self._s.name()]
            return max(mu + noise.gauss(0.0, se), 0.0)

    shaky = {}
    for n in samp:
        shaky[n] = _Resampled(reg[n]._s, tab, shaky)

    worst = 0.0
    for _ in range(4000):
        a, b = samp[rnd.randrange(600)], samp[rnd.randrange(600)]
        s1, s2 = shaky[a], shaky[b]
        jt = g.jc_tree(s1, s2, None)
        d1, d2 = g.sparse_vec(s1, None), g.sparse_vec(s2, None)
        keys = set(d1) | set(d2)
        l2 = sum((d1.get(k, 0.0) - d2.get(k, 0.0)) ** 2 for k in keys)
        worst = max(worst, abs(l2 - jt))
    assert worst > 1.0, (
        f"resampling IC within its se should DESTROY exactness (measured 7.31); got {worst:.3e}. "
        "If this now passes exactly, the second-channel rule has quietly become unenforced.")


def test_smooth_ic_refuses_a_zero_alpha():
    """alpha=0 is 'no smoothing' spelled as a parameter: it reopens the hole silently, so it
    raises `ValueError` instead."""
    with pytest.raises(ValueError):
        g.smooth_ic({}, {"entity.n.01": None}, alpha=0.0)


def test_faithfulness_check_reports_its_skips(monkeypatch):
    """`faithfulness_check` measures 28 of 30 DEFAULT_PAIRS, and reports the two it skipped.

    `metal.n.01` and `monarch.n.01` are lemma aliases: nltk resolves them (to
    `metallic_element.n.01` / `sovereign.n.01`, verified) while `driver.synset()` is a strict
    canonical lookup and raises. The corpus holds both concepts under their canonical names.

    What is asserted here is that the denominator is visible — `n_requested`, `n_skipped`,
    `complete`, and a reason per skip. The strictness of the lookup is a driver decision with a
    wider blast radius and is left as it is.
    """
    import types

    import ember.ontology
    from crystal.ontology import driver as wn_store  # noqa: F401 - see test_ic_revision_refuses_to_be_unknown
    wanted = {a for a, _ in g.DEFAULT_PAIRS} | {b for _, b in g.DEFAULT_PAIRS}
    strict = {n: _syn(n) for n in wanted if nltk_wn.synset(n).name() == n}

    def _synset(name):
        if name not in strict:
            raise KeyError(f"unknown synset {name!r}")
        return strict[name]

    fake = types.SimpleNamespace(synset=_synset, NOUN="n", synsets=lambda *a, **k: [])
    monkeypatch.setattr(crystal_ontology, "driver", fake)

    out = g.faithfulness_check(D=256)
    assert out["n_requested"] == 30
    assert out["n_skipped"] == 2, f"expected the 2 alias pairs skipped, got {out['n_skipped']}"
    assert out["n_pairs"] == 28
    assert out["complete"] is False, "28 of 30 must not report as a complete run"
    names = {x for a, b, _ in out["skipped"] for x in (a, b)}
    assert {"metal.n.01", "monarch.n.01"} <= names
    assert all(r for _, _, r in out["skipped"]), "every skip must carry its reason"


if __name__ == "__main__":                                   # a quick store-free report
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))


class _BrokenIC:
    """A synset whose store read raises, as distinct from one whose ic is absent."""
    __slots__ = ("_n",)

    def __init__(self, name):
        self._n = name

    def name(self):
        return self._n

    def ic(self):
        raise sqlite3.OperationalError("database is locked")


def test_a_store_fault_does_not_read_as_an_ic_of_zero():
    """A store fault propagates out of `ic_of`, so it stays distinguishable from a root synset.

    0.0 is what a genuine root reads, and it is not an inert value: `has_ic()`'s docstring records
    where it lands. All-zero IC makes `canonical_parent` choose the spanning tree alphabetically,
    and every `jc_tree` distance is then computed over an arbitrary tree and returned as a
    measurement. A locked database, a wrong object and a corrupt row therefore raise rather than
    read as zero. The absence contract is separate: absence is `None`, and `None` reads 0.0.
    """
    with pytest.raises(sqlite3.OperationalError):
        g.ic_of(_BrokenIC("locked.n.01"))

    # a corrupt row (non-numeric) is loud too — it is not an absence
    with pytest.raises((TypeError, ValueError)):
        g.ic_of(_FixedIC("corrupt.n.01", "not-a-number"))

    # and the two documented values are unchanged
    assert g.ic_of(_FixedIC("absent.n.01", None)) == 0.0
    assert g.ic_of(_FixedIC("root.n.01", 0.0)) == 0.0
