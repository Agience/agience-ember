"""Boot — the assembly that makes the parts a program.

A cache, a store, an aligner and an answerer are libraries that know nothing about each other.
This module wires them into a leaf, and that wiring settles two questions the pieces leave open.

First light: a leaf is initialised from outside
----------------------------------------------
Ember *runs* disconnected. It is *initialised* from outside, because two things can only come
from elsewhere:

1. **The canonical AnchorSet.** Without it there is no routing. A leaf adopts one rather than
   authoring one: anchor ids are content-addressed over `(label, model_id, embedding)`, so a
   home-made anchor set computes region ids nobody else uses — it would route, it would answer,
   and it would share with nobody.
2. **The authority's public key.** Cached shards are verified against it, and an unverified shard
   is a rumour rather than a cache, so none is loaded without it.

A fresh leaf is therefore uninitialised rather than broken — a different condition with a
different remedy — and `ready` reports which one it is.

What boot touches
-----------------
Disk, and nothing else. Boot makes no network call, so it is fast, offline, and cannot hang on a
dead cloud. Acquiring first light is a separate, explicit act (`seed`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from mantle.shard.cache import LocalCache
from ember.config import Settings, load as load_settings
from ember.embed import Aligner, Embedder, HashEmbedder
try:                                    # absent during mantle's restructure — see ember/embed.py
    from mantle.search.anchors.crosswalk_artifact import CROSSWALK_CONTENT_TYPE
except ModuleNotFoundError:             # the reading path never reads it; anything that does fails
    CROSSWALK_CONTENT_TYPE = None       # loudly on a None rather than silently on a made-up value
from mantle.search.anchors.anchorset import AnchorSet
from prism.mass import Provenance
from ember.runtime.read_path import Result, answer_query
from mantle.shard.store import ShardStore


class NotInitialised(RuntimeError):
    """The leaf has never had first light: no anchors, or no authority key.

    Distinct from "empty". An empty cache answers "I don't hold that", which is true and useful.
    An uninitialised leaf cannot route at all, so it has no such answer to give.
    """


@dataclass
class Ember:
    """A booted leaf."""

    settings: Settings
    store: ShardStore
    embedder: Embedder
    # `engine` is injected and optional: absent, `ask()` reports that nothing is wired to answer.
    # It sits below `embedder` because a defaulted dataclass field cannot precede a non-defaulted
    # one; every construction site passes it by keyword.
    engine: Optional[Any] = None         # duck-typed: anything with .answer(query, rows)
    anchors: Optional[AnchorSet] = None
    aligner: Optional[Aligner] = None
    cache: Optional[LocalCache] = None
    authority_pub: object = None
    loaded: int = 0
    rejected: int = 0
    channel: object = None       # a relay.Channel; Disconnected until connect() is called

    # ------------------------------------------------------------------ state
    @property
    def ready(self) -> bool:
        """Can this leaf route and verify? Both, or neither is useful."""
        return self.cache is not None and self.authority_pub is not None

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "anchors": len(self.anchors) if self.anchors else 0,
            "has_authority_key": self.authority_pub is not None,
            # `aligned` means a cross-walk exists. Whether it says anything is the separate
            # question `alignment_carries_information` answers, and status reports both.
            "aligned": bool(self.aligner and (self.aligner.native or self.aligner.adopted
                                              or self.aligner.residual is not None)),
            "crosswalk_adopted": bool(self.aligner and self.aligner.adopted),
            # The HELD-OUT residual, read against a derived null of 1.0. The in-sample residual on
            # the same fit reads far lower and would publish a fidelity the projection does not
            # have. `None` means the residual was not measured.
            "alignment_residual_held_out": (
                self.aligner.residual.held_out
                if self.aligner and self.aligner.residual is not None else None),
            "alignment_carries_information": (
                self.aligner.carries_information if self.aligner else None),
            "shards_loaded": self.loaded,
            "shards_rejected": self.rejected,
            "offline": self.settings.offline,
            "cache_dir": str(self.settings.cache_dir),
            **({"cache": self.cache.summary()} if self.cache else {}),
        }

    # ------------------------------------------------------------------ boot
    @classmethod
    def boot(cls, settings: Optional[Settings] = None, *,
             embedder: Optional[Embedder] = None,
             authority_pub=None,
             engine: Optional[Any] = None) -> "Ember":
        """Assemble from disk. No network, ever — see the module docstring."""
        s = settings or load_settings()
        store = ShardStore(s.cache_dir)

        # The runner decides that new rows mean cognition; the store only reports that they
        # arrived. `mantle.mesh.sync` declares a sink and ember fills it, so the store depends on
        # nothing in the runner.
        _wire_peer_signal_delivery()
        _wire_demurrage_clock(store)
        _wire_collection_proximity(store)
        emb = embedder or _default_embedder(s)
        # The runner does not construct an answerer: ember runs energized crystals and prisms, and
        # composing prose from evidence is a persona act. An answerer arrives by injection from
        # whoever wired the persona, or not at all — see `ask()` below.
        eng = engine

        self = cls(settings=s, store=store, engine=eng, embedder=emb, authority_pub=authority_pub)

        self.anchors = store.load_anchors()
        if self.anchors is None:
            return self          # uninitialised: no first light yet. Honest, not broken.

        # Align. A cached cross-walk is adopted if it verifies; otherwise we refit locally,
        # which needs no network because an Anchor carries its label AND its canonical vector.
        self.aligner = Aligner(emb, self.anchors)
        cached = store.find_artifact(CROSSWALK_CONTENT_TYPE)
        self.aligner.fit(cached=cached)
        if not self.aligner.native and not self.aligner.adopted:
            # We did the work; publish it so the next leaf on this embedder need not repeat it.
            store.put_artifact(self.aligner.as_artifact())

        # Region ids can be blinded under a per-principal secret, so a shard never spells out whose
        # data it is or what concept it holds.
        #
        # `create=False`: blinding activates when a shared `principal.secret` has been provisioned
        # to every party that routes to these cells — a person's devices and any peer that serves
        # them — exactly like `content.key`. The region id is the routing key: two parties reach the
        # same cell only when they derive the same id, so a node that blinds with a secret its peers
        # lack routes to nobody, and `test_relay.py` covers that case. Absent the secret,
        # `principal_secret` returns None and ids stay as its peers compute them, so routing is
        # unchanged. Turning blinding on is a coordinated provisioning step.
        from mantle.shard import region as _region
        secret = _region.principal_secret(s.cache_dir)          # create=False: never auto-mint
        # The read-time head resolver. There is no stored head: every revision commits and stands,
        # and which one answers is measured when a query is made, against the reader's own recall
        # set. `ember/optics.py` is the one module that reaches entroptics, so the measurement
        # arrives here as an injection rather than being taken inside mantle.
        #
        # Too little frame, an unmeasurable candidate, or no candidate above the frame's own null
        # each yield no reading, and every revision stays standing — which is what the cache does
        # with no resolver at all.
        from ember.optics import revision_resolver
        self.cache = LocalCache(self.anchors, s.principal, s.collection_id, nprobe=s.nprobe,
                                secret=secret, resolve=revision_resolver())

        if authority_pub is not None:
            # Rehydrate, verifying every shard: disk is just another untrusted server.
            self.loaded, self.rejected = store.load_node(self.cache.node, authority_pub)
            # Rebuild the derived views the shards cannot carry: the version lineage (which
            # revisions share a root) and the readable/searchable view. Without this a restart
            # keeps the bytes and forgets both.
            #
            # This is lineage, not a head chain. Every revision stands and which one answers is
            # measured at read time, so a sidecar's `heads` key is read and dropped by `hydrate`
            # (the drop is counted); honouring it would restore a write-time decision on boot.
            self.cache.hydrate(store.load_views())
        # else: we hold no verifiable shards. Better an empty cache than an unverified one.
        return self

    def seed(self, anchors: AnchorSet, authority_pub) -> "Ember":
        """First light: adopt the canonical anchors + the authority key, and persist them.

        Explicit on purpose. This is the one moment a leaf depends on the outside world, so it
        is a call you make rather than a thing that happens to you on startup.
        """
        self.store.save_anchors(anchors)
        # `seed` re-boots, so every injected dependency is forwarded here — `embedder` and
        # `engine`. A dependency left out of this call is dropped while the injection still looks
        # correct at the call site, so anything added to `boot()` belongs here too.
        return type(self).boot(self.settings, embedder=self.embedder,
                               authority_pub=authority_pub, engine=self.engine)

    # ------------------------------------------------------------------ use
    def connect(self, channel) -> "Ember":
        """Attach an outward channel (relay.MeshChannel) so a cache miss can refill from the
        cloud. Boots disconnected; this is the explicit opt-in to the network — nothing phones
        home on its own. The channel's transport is injected, so no platform code is imported."""
        self.channel = channel
        return self

    def ask(self, text: str, *, refill=None, k: Optional[int] = None) -> Result:
        """Answer from local shards, refilling from the connected channel on a miss.

        ``refill`` overrides the channel for one call (e.g. tests). With neither, the answer is
        purely local — which is not a failure mode but the design: a leaf is useful offline, and
        reaches the network only when it has a channel AND misses.

        ``k=None`` cuts the read where its own relevance stops being signal
        (`prism.resolution.signal_end`, which computes its own null and can decline to cut at all —
        see `read_path.answer_query`). A caller that states ``k`` gets exactly ``k``."""
        self._require_ready()
        # With no answerer wired there is nothing to compose an answer with, and `NoAnswerer` says
        # so. `engine` is injected by whoever wired a persona; a plain boot has none. Raising is
        # chosen over an empty Result because a caller cannot otherwise tell "nothing was found"
        # from "nothing is wired to look", and the computed null belongs to the persona that was
        # asked rather than to the runner that was not.
        if self.engine is None:
            from ember.runtime.engine import NoAnswerer
            raise NoAnswerer(
                "no answerer is wired: ember runs energized crystals and prisms, it does not compose "
                "answers. Inject one via Ember(engine=...) — see lumen/composers.py — or reach a "
                "persona's op.respond over the plane.")
        r = refill
        if r is None and self.channel is not None:
            r = self.channel.as_refill(on_import=self._adopt_regions)
        return answer_query(self.cache, self.engine, text,
                            self.aligner.encode_query(text), refill=r, k=k)

    def _adopt_regions(self, regions) -> None:
        """Make freshly-refilled shards searchable: decode each imported item into the retrieval
        view. Content the leaf cannot read (blind) yields no vector and stays held-not-searchable
        — the honest outcome, not an error. Text is embedded through the SAME aligner the query
        uses, so a refilled item routes into the same cell it was authored in."""
        for region in regions:
            manifest, items = self.cache.node.get_shard(region)
            if manifest is None:
                continue
            vectors = {}
            for item_id, content in items.items():
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    continue                  # blind/binary: held, not searchable
                vectors[item_id] = self.aligner.encode_query(text)
            # The manifest is the only carrier of each item's `consensus` (its provenance mass), so
            # it is passed through: without it a refilled item lands at mass 0.0 and any later local
            # revision at any rung outranks it. See `LocalCache.adopt`.
            self.cache.adopt(region, items, vectors, manifest=manifest)

    def remember(self, items: Sequence[Tuple[str, bytes, str]], *, provenance: Provenance,
                 authority: str, priv, evidence: Optional[dict] = None,
                 version: int = 1) -> List[str]:
        """Author local artifacts — ``(id, content, text_to_embed)`` — and persist them.

        Your own content is a region this node **originates**: signed and content-addressed on
        the same terms as anything from the mesh, so it can be served to peers as an equal.

        ``provenance`` is **required, with no default** — this is the proofreading layer, and the
        caller is the only one who knows how the content was obtained. A default would hand every
        artifact a free rung, filling the store with well-formed claims whose grounding nobody
        asserted.

        Weight follows the rung, so a leaf's own notes are weighed on the same ladder as anything
        pulled from the mesh. There is no local exemption.
        """
        self._require_ready()
        if not isinstance(provenance, Provenance):
            provenance = Provenance(provenance)   # raises on nonsense rather than guessing
        vecs = [(i, c, self.aligner.encode_query(t)) for i, c, t in items]
        regions = self.cache.put(vecs, version=version, authority=authority, priv=priv,
                                 provenance=provenance, evidence=evidence)
        self.flush()
        return regions

    def revise(self, root_id: str, new_content: bytes, text: str, *, provenance,
               authority: str, priv, version: int = 2):
        """Author a new version of an existing artifact. Artifacts are immutable, so this commits a
        new version under ``root_id`` rather than editing in place, and that is all it does.
        Returns `Landed` (id, root, how many versions now stand).

        Committing a revision does not decide which version answers: head is resolved at read time,
        by the reader's own measurement. A headcount of independent origins reads agreement rather
        than validity, and `prism.resolution` has no separation to report between a pair of counts
        at n=2, where the computed null is 1.0000. A correction is exactly the claim that starts
        with one witness."""
        self._require_ready()
        landed = self.cache.revise(root_id, new_content, self.aligner.encode_query(text),
                                    provenance=provenance, authority=authority, priv=priv,
                                    version=version)
        self.flush()
        return landed

    def flush(self) -> int:
        """Persist every shard and the derived views (version lineage + readable vectors), so a
        restart restores the bytes, which versions belong to which root, and the searchable index.

        Lineage, not a head chain: there is no stored head to persist, and a snapshot that carried
        one would decide for its reader on the next boot."""
        self._require_ready()
        n = self.store.save_node(self.cache.node)
        self.store.save_views(self.cache.export_views())
        return n

    def _require_ready(self) -> None:
        if self.cache is None:
            raise NotInitialised(
                "no canonical AnchorSet — this leaf has never had first light. "
                "Call seed(anchors, authority_pub). It must NOT author its own anchors: the "
                "region ids would match nobody."
            )
        if self.authority_pub is None:
            raise NotInitialised(
                "no authority public key — shards could not be verified, and an unverified "
                "shard is a rumour, not a cache. Call seed(anchors, authority_pub)."
            )


def _default_embedder(s: Settings) -> Embedder:
    """The deterministic embedder (`ember.embed.HashEmbedder`). Ember carries no trained
    weights, so there is no alternative to select.

    `EMBER_EMBED_MODEL` selects nothing, and a value in it raises. A deployment still carrying the
    setting stops at boot rather than falling through to `HashEmbedder`: the two embedders mint
    different anchor ids, and that leaf would route into a private universe.
    """
    if s.embed_model:
        raise ValueError(
            f"EMBER_EMBED_MODEL={s.embed_model!r} selects a TRAINED embedder, which has been "
            "removed. Unset it. Semantic retrieval is the computed ontology coordinate "
            "(ember/geometry.py), not a learned vector space; do not substitute another model."
        )
    return HashEmbedder()


__all__ = ["Ember", "NotInitialised"]


def _wire_demurrage_clock(store) -> None:
    """Give `prism.demurrage` the measured slow timescale to cool by. Idempotent.

    The Screen is the market, and its rates are measured ([[universal-economics]], canon §12-13).
    Demurrage is the second law applied to deposited energy; forgetting is the second law applied
    to attention. They are one measured timescale, and this is where it enters.

    prism sits below ember, so it cannot import the measuring Screen: it declares a socket and the
    runner fills it, the same shape as `_wire_peer_signal_delivery` below. Wiring supplies the how,
    not the whether — with nothing measured yet `demurrage.slow_rate()` returns None and callers
    fall back explicitly, and can see that they did.

    The clock is the node's Screen rather than a delegate's. The economy's rate is one fact about
    one box, like the node's headroom, so it does not vary with whose conversation is busy."""
    try:
        from prism import demurrage as _dem
        from ember.signal import forgetting as _forgetting
        from ember.signal import pooling as _pooling
    except Exception:
        return

    _screen = {"s": None}

    def _measured_slow():
        s = _screen["s"]
        if s is None:
            # One forgetting Screen for the node, distinct from `pooling.node_screen()` (which is
            # the node's ACCUMULATOR, a different instrument). Built lazily so a runner that never
            # prices anything pays nothing.
            s = _forgetting.Screen(witness=_pooling.node_id(), store=store)
            _screen["s"] = s
        s._rates_current()
        return s.tau_slow if s.rates_source == "measured" else None

    _dem.set_slow_rate_source(_measured_slow)


def _wire_collection_proximity(store) -> None:
    """Install the proximity digest refresher into the store. Idempotent; safe every boot.

    mantle declares the seam and cannot fill it: `CollectionDigestRefresher` needs an instrument
    to take the spectral read with, and mantle never imports entroptics. ember is the sole
    sanctioned door to it (`ember.optics`), so the host supplies `read` and `engine_id` here and
    the store reports membership changes without knowing what reads them.

    A store with nothing installed maintains no digests, which is the state a plain
    `pip install agience-mantle` is in. Every failure below leaves it in that state rather than
    failing the boot: a digest is an accelerator for a query nobody has issued yet.
    """
    try:
        from mantle.db.lattice_api import list_collection_artifacts
        from mantle.search.ingest.digest_refresh import install_digest_refresher
        from mantle.search.mantle.wiring import build_digest_refresher

        from ember import optics
    except Exception:
        return

    def _members(collection_id: str):
        """`(members, exhaustive)` — the pairs `build_frame` digests.

        `exhaustive` is the store's own answer about whether it handed over the whole collection.
        Above `edges_of`'s limit it has not, and `digest_collection` refuses rather than digest a
        prefix: a digest of part of a collection is a fabricated measurement of all of it.
        """
        rows = list_collection_artifacts(store.artifacts.db, collection_id)
        pairs = [(r.get("id"), r) for r in rows if isinstance(r, dict) and r.get("id")]
        return pairs, len(pairs) < 1000

    try:
        refresher = build_digest_refresher(
            _members,
            read=optics.proximity_read(),
            engine_id=optics.proximity_engine_id(),
        )
    except Exception:
        return
    if refresher is not None:
        install_digest_refresher(refresher)


def _wire_peer_signal_delivery() -> None:
    """Deliver peer-replicated docs into local cognition. Idempotent; safe to call every boot.

    Gated by `EMBER_DELIVER_PEER_SIGNALS` inside sync itself — this only supplies the HOW, never
    the WHETHER, so wiring it does not switch anything on."""
    from mantle.mesh import sync as _sync
    from ember.signal import signal as _signal
    from ember.runtime.delegate import Delegate

    def _deliver(store, new_docs):
        _signal.deliver(Delegate.get(store), new_docs, store=store)

    _sync.set_peer_signal_sink(_deliver)
