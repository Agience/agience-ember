"""docs_ops clustering — §9.6: emit the filtration, the cut is a query parameter.

Let the information decide. A topical continuum has no valley to cut at (measured for the
analogous MinHash case, §9.3), so a single hardwired clustering threshold would impose one. The
structure — the agglomerative filtration over lemma-Jaccard — is emitted whole instead, and a
threshold chooses which resolution to read from it.
"""
from __future__ import annotations

from ember.runtime.runner import docs_ops as D    # the distributed bundle — the single path under test


def test_filtration_is_descending_and_covers_every_nonzero_link():
    docs = [{"id": "a", "lemmas": ["x", "y"]},
            {"id": "b", "lemmas": ["x", "y"]},          # identical -> jaccard 1.0
            {"id": "c", "lemmas": ["x", "z"]},          # shares x with a,b
            {"id": "d", "lemmas": ["q"]}]               # shares nothing
    links = D.lemma_filtration(docs)
    assert links == sorted(links, key=lambda x: -x[0]), "must be descending by similarity"
    assert links[0][0] == 1.0                            # a,b are identical -> strongest link
    # d shares no lemma -> it appears in no link (a zero link is not a link)
    assert all("d" not in (a, b) for _, a, b in links)


def test_one_filtration_reads_at_many_resolutions():
    """The data supplies clusters at every scale, and a threshold chooses which scale to read.
    One filtration: cut high for fine and many, cut low for coarse and few. The caller owns the
    threshold, so the topics are the ones it asked for."""
    docs = [
        {"id": "a", "lemmas": ["dog", "cat", "pet"]},
        {"id": "b", "lemmas": ["dog", "cat", "leash"]},         # close to a
        {"id": "c", "lemmas": ["car", "engine", "wheel"]},
        {"id": "d", "lemmas": ["car", "engine", "road"]},       # close to c
        {"id": "e", "lemmas": ["dog", "car"]},                  # weak bridge across the two
    ]
    links = D.lemma_filtration(docs)
    ids = [x["id"] for x in docs]

    high = D.cut_filtration(links, ids, threshold=0.5)          # only strongest links survive
    tight = sorted(sorted(g) for g in high if len(g) >= 2)
    assert ["a", "b"] in tight and ["c", "d"] in tight          # the two tight pairs, separate

    low = D.cut_filtration(links, ids, threshold=0.15)          # weak bridge merges them
    assert len([g for g in low if len(g) >= 2]) <= len(tight)


def test_topics_is_a_cut_of_the_filtration_and_the_cut_is_overridable():
    docs = [{"id": "a", "lemmas": ["x", "y", "z"]},
            {"id": "b", "lemmas": ["x", "y", "w"]},             # jaccard(a,b) = 2/4 = 0.5
            {"id": "c", "lemmas": ["p", "q", "r"]}]
    assert D.topics(docs, threshold=0.9) == []                  # too strict -> nothing merges
    grouped = D.topics(docs, threshold=0.4)                     # a,b merge; c alone
    ids = sorted(sorted(x["id"] for x in g) for g in grouped)
    assert ids == [["a", "b"]]


def test_drop_common_frac_removes_ubiquitous_vocabulary():
    """A lemma shared by nearly every doc carries no topical signal and links the whole set into
    one group. Dropping it is a declared option the caller passes, so the filtration stays a
    function of the arguments it was given."""
    docs = [{"id": "a", "lemmas": ["the", "dog"]},
            {"id": "b", "lemmas": ["the", "cat"]},
            {"id": "c", "lemmas": ["the", "car"]}]
    # kept: 'the' links all three; dropped at >0.5: only distinctive terms remain
    kept = D.lemma_filtration(docs)
    dropped = D.lemma_filtration(docs, drop_common_frac=0.5)
    assert kept and not dropped                                 # 'the' was the only shared term
