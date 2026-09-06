"""ember ingest — author local repos and cached offloads into Ember's cache.

Standalone: no account, no network. `bootstrap_local` gives a leaf its first light for local-only
use — an AnchorSet built in the embedder's space and a self-generated Ed25519 authority, since the
leaf is its own authority for content it originates. Region ids are local until the leaf adopts the
mesh's canonical anchors, which suits a machine that may never connect.

Code enters at provenance `OBSERVED`: it was retrieved from disk, which is a high rung. The same
authoring path takes cached offloads at provenance `HYPOTHESIS` — an unverified LLM answer, dark
matter that has yet to earn mass — so the cache holds both on one ladder. See
[[content-context-operator-triple]].
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from ember.runtime.boot import Ember
from ember.config import load as load_settings
from ember.embed import Embedder, HashEmbedder
from mantle.search.anchors.anchorset import AnchorSet

try:                                    # provenance rung = the mass ladder (shared with the platform)
    from prism.mass import Provenance
except Exception:                       # local-only fallback: ingest runs without the core present
    class Provenance:                   # pragma: no cover
        OBSERVED = "observed"; HYPOTHESIS = "hypothesis"

CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml", ".toml",
            ".sh", ".rs", ".go", ".java", ".c", ".h", ".cpp", ".sql", ".html", ".css"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "venvs", "dist", "build",
             ".mypy_cache", ".pytest_cache", "target", ".continue", ".idea", "site-packages"}

# A spread of anchor labels — the local coordinate system for routing. Content-derived centroids
# would route better; a fixed spread is deterministic and dependency-free for a first light.
ANCHOR_LABELS = [
    "function definition and return value", "class and object model", "http request and response",
    "database query and schema", "file read write and path", "json and serialization",
    "test assertion and fixture", "error handling and exception", "authentication and token",
    "configuration and environment", "data structure and algorithm", "async and concurrency",
    "logging and telemetry", "encryption and signing", "vector embedding and search",
    "reasoning and dynamics", "artifact and provenance", "cache and storage", "mesh and node",
    "cli and argument parsing", "documentation and overview", "build and packaging",
]


def _embedder() -> Embedder:
    """The numpy HashEmbedder: deterministic and model-free. Nothing is downloaded and nothing is
    trained, so every node computes the same vector space and therefore the same anchor ids."""
    return HashEmbedder(dim=256)


def bootstrap_local(*, embedder: Optional[Embedder] = None) -> Tuple[Ember, str, object]:
    """First light for a standalone leaf: build the AnchorSet in the embedder's space and seed the
    leaf with a self-generated authority. Returns `(ember, authority_id, priv)`."""
    s = load_settings()
    s.cache_dir.mkdir(parents=True, exist_ok=True)   # first light writes anchors here; ensure it exists
    emb = embedder or _embedder()
    aset = AnchorSet(model_id=emb.model_id, dim=emb.dim)
    for label, vec in zip(ANCHOR_LABELS, emb.encode(list(ANCHOR_LABELS))):
        aset.add_text(label, vec)
    priv = Ed25519PrivateKey.generate()
    ember = Ember.boot(s, embedder=emb).seed(aset, priv.public_key())
    return ember, "ember-local", priv


def walk_chunks(paths: Sequence[str], *, chunk_lines: int = 40,
                exts: Optional[set] = None) -> Iterable[Tuple[Path, int, str]]:
    """Yield (path, start_line, chunk_text) over code files, skipping junk dirs."""
    exts = exts or CODE_EXT
    for root in paths:
        for p in Path(root).rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for i in range(0, len(lines), chunk_lines):
                chunk = "\n".join(lines[i:i + chunk_lines]).strip()
                if chunk:
                    yield p, i + 1, chunk


def _md_offer(text: str, max_chars: int = 700):
    """A doc's topic — its H1 or title — plus a short intro. Docs state their subject up top."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = next((l.lstrip("#").strip() for l in lines if l.startswith("#")), (lines[0] if lines else ""))
    intro = " ".join(lines[:8])[:max_chars]
    return title, intro


def _py_offer(text: str, max_chars: int = 700):
    """A module's purpose — the first line of its docstring — plus the symbols it defines. Bias-free:
    the code describing itself."""
    import ast
    try:
        tree = ast.parse(text)
    except Exception:
        return "", _generic_summary(text)
    doc = (ast.get_docstring(tree) or "").strip()
    purpose = doc.split("\n", 1)[0]
    syms = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            syms.append(node.name)
    detail = (doc + ("  defines: " + ", ".join(syms[:24]) if syms else "")).strip()[:max_chars]
    return purpose, detail


def _generic_summary(text: str, max_chars: int = 400) -> str:
    return "\n".join(text.splitlines()[:8]).strip()[:max_chars]


# A hand-authored kind vocabulary. Matching a need to an offer requires knowing kinds ("this is
# a description"), and the synonyms below are authored rather than learned from the corpus — the
# corpus defines these words the way a child learns "describe" from real examples rather than
# from a synonym table. The bias-free part is the file describing itself, through its own
# docstring or title. See [[offers-needs-retrieval]] /
# [[developmental-learning-not-self-reinforcement]].
KIND_SYNONYMS = {
    "overview":       "overview, description, introduction, summary, about",
    "description":    "description, overview, explanation, summary, information about",
    "guide":          "guide, tutorial, how-to, walkthrough, getting started, usage",
    "reference":      "reference, API documentation, specification, contract",
    "design":         "design, architecture, plan, proposal, rationale",
    "implementation": "implementation, source code, module",
    "example":        "example, test, sample, demonstration",
}


def classify_kind(path: Path, text: str) -> str:
    """The foundational kind of an artifact, from its content-type and name. Deterministic."""
    name = path.stem.lower(); suf = path.suffix.lower()
    if suf in {".md", ".markdown", ".rst", ".txt"}:
        if any(k in name for k in ("readme", "overview", "about", "intro", "index", "home")):
            return "overview"
        if any(k in name for k in ("guide", "tutorial", "howto", "how-to", "getting", "usage", "quickstart")):
            return "guide"
        if any(k in name for k in ("reference", "api", "spec", "protocol", "contract")):
            return "reference"
        if any(k in name for k in ("plan", "design", "architecture", "rfc", "proposal")):
            return "design"
        return "description"
    if suf == ".py" and ("test" in name):
        return "example"
    return "implementation"


def describe_file(path: Path, text: str) -> str:
    """The context of a file as an offer of a kind, framed so a query — a need of a kind — matches
    it. Built from the file's own title, docstring and structure plus the foundational kind, so it
    is deterministic. Indexing and retrieval run on this rather than on raw content
    (ontology-indexed-on-context; offers matched to needs)."""
    suf = path.suffix.lower()
    name = path.stem.replace("_", " ").replace("-", " ")
    kind = classify_kind(path, text)
    syn = KIND_SYNONYMS.get(kind, kind)
    if suf in {".md", ".markdown", ".rst", ".txt"}:
        title, intro = _md_offer(text)
        topic = title or name
        offer = (f"This document is a {kind} of {topic}. It provides a {syn} of {topic}. "
                 f"It is documentation about {topic}.\n{intro}")
    elif suf == ".py":
        purpose, detail = _py_offer(text)
        offer = f"This is the {kind} ({name}) — it provides {purpose or name}. ({syn})\n{detail}"
    else:
        offer = f"This file ({name}) is {kind} that contains:\n{_generic_summary(text)}"
    return f"{path.as_posix()}\n{offer}".strip()


def _describe_files(paths: Sequence[str], exts: Optional[set] = None) -> List[Tuple[str, bytes, str]]:
    """One artifact per file: `(id, content, embed_key)`, where the embed key is the description.
    Retrieval runs on the description (the context), and the path inside it leads to the real
    file."""
    import os
    exts = exts or CODE_EXT
    out: List[Tuple[str, bytes, str]] = []
    seen = set()
    for root in paths:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]   # prune before descending
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in exts:
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if not text.strip():
                    continue
                key = p.as_posix()
                if key in seen:
                    continue
                seen.add(key)
                # Describe at first observation, per content-type. The description is the context,
                # which is what gets embedded and indexed; the raw file is the content, retrieved
                # after a context match and handed to predict. Two fields: search context, then
                # retrieve content.
                desc = describe_file(p, text)
                content = f"{key}\n{text[:20000]}"
                out.append(("file-" + hashlib.sha256(key.encode()).hexdigest()[:20],
                            content.encode("utf-8"), desc))     # (id, content, embed_key=context)
    return out


class AnchorSetNotProvisioned(RuntimeError):
    """No canonical AnchorSet was supplied, and one cannot be invented here."""


def _require_anchorset(anchors: Optional[AnchorSet], emb: Embedder) -> AnchorSet:
    """The canonical AnchorSet, or `AnchorSetNotProvisioned`.

    The AnchorSet is provisioned rather than derived, and `seed(anchors, authority_pub)` is how it
    arrives. An anchor id is content-addressed over `(label, model_id, embedding)`, so anchors
    clustered from whatever corpus a node happens to hold would mint region ids no other node
    computes: the leaf would route, answer, and share with nobody. README.md §First light names
    that as the property a leaf holds to.

    The dimension check is the second half. An AnchorSet whose dim differs from the embedder's
    routes into different cells, so the mismatch is named here rather than carried into the index.
    """
    if anchors is None or len(anchors) == 0:
        raise AnchorSetNotProvisioned(
            "ingest needs the canonical AnchorSet and none was given. It is deliberately not "
            "derived from the corpus: locally-clustered anchors mint region ids no peer computes, "
            "so the index would look healthy and share with nobody. Pass anchors=<AnchorSet> "
            "(AnchorSet.load(...) or the set this node was seeded with)."
        )
    if anchors.dim != emb.dim:
        raise AnchorSetNotProvisioned(
            "AnchorSet dim %d != embedder dim %d — routing would land in the wrong cells. The "
            "cross-walk (ontology/embed.Aligner) exists for this; fit it rather than re-anchoring."
            % (anchors.dim, emb.dim)
        )
    return anchors


def wordnet_items():
    """WordNet as ground-truth lexical artifacts. Each synset becomes an artifact whose offer is the
    meaning and definition of its words — the source of what words mean and how they relate, which
    is what `KIND_SYNONYMS` stands in for until it is read from here. Local and deterministic.
    Needs `nltk` and the wordnet corpus (`nltk.download('wordnet')`); with neither present it yields
    nothing."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return
    for syn in wn.all_synsets():
        words = sorted({l.name().replace("_", " ") for l in syn.lemmas()})
        if not words:
            continue
        defn = syn.definition() or ""
        also = f" (also: {', '.join(words[1:6])})" if len(words) > 1 else ""
        offer = f"the definition and meaning of the word {words[0]}{also}: {defn}"
        content = (f"{words[0]} — {defn}\npart of speech: {syn.pos()}\n"
                   f"synonyms: {', '.join(words)}\nexamples: {'; '.join(syn.examples()[:3])}")
        # A structured lemma index: a dictionary is a keyed lookup rather than an
        # embedding-similarity problem, so lowercased lemmas let the lexical path key straight to
        # the synset.
        meta = {"lemmas": [w.lower() for w in words], "word": words[0].lower(), "pos": syn.pos()}
        yield ("wn-" + syn.name(), content.encode("utf-8"), offer, meta)


_WIKI_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
_WIKI_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_WIKI_FILE = re.compile(r"\[\[(?:File|Image|Category):[^\]]*\]\]", re.IGNORECASE)
_WIKI_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_WIKI_HTML = re.compile(r"<[^>]+>")


def _wiki_clean(text: str) -> str:
    """Deterministic wikitext -> plain text (lead-quality). Strips refs/comments/files/tables/
    templates/HTML, unwraps links and bold/italic. Not a full parser — good enough for a summary."""
    t = _WIKI_COMMENT.sub("", text)
    t = _WIKI_REF.sub("", t)
    t = _WIKI_TABLE.sub("", t)
    t = _WIKI_FILE.sub("", t)
    for _ in range(6):                                   # iteratively remove {{templates}} (handle nesting)
        t2 = re.sub(r"\{\{[^{}]*\}\}", "", t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)    # [[link|text]] -> text
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)             # [[link]] -> link
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)  # [url text] -> text
    t = t.replace("'''", "").replace("''", "")
    t = _WIKI_HTML.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# The shortest line `_wiki_lead` accepts as the article's lead sentence rather than as leftover
# markup, chosen rather than measured from the dump. The prefix test beside it is a specification
# — `=`, `|`, `*`, `#`, `{` are MediaWiki's own section, table, list and template markers — where
# this length test stands in for "this line is a sentence".
#
# A lead shorter than this is passed over and the next paragraph is offered in its place, so the
# cost of a short-lead article is a mis-stated offer rather than a lost artifact.
_LEAD_MIN_CHARS = 40


def _wiki_lead(clean: str, max_chars: int = 500) -> str:
    for para in clean.split("\n"):
        p = para.strip()
        if len(p) > _LEAD_MIN_CHARS and not p.startswith(("=", "|", "*", "#", "{")):
            return p[:max_chars]
    return ""


def wikipedia_items(dump_path: str, max_articles: Optional[int] = None):
    """Wikipedia as conceptual ground-truth artifacts: each article becomes an artifact whose offer
    is the article's own lead ("X is a ..."). Streams the .bz2 XML dump at flat memory.
    Deterministic and local."""
    import bz2
    from xml.etree.ElementTree import iterparse
    opener = bz2.open if str(dump_path).endswith(".bz2") else open
    title = text = None
    count = 0
    with opener(dump_path, "rt", encoding="utf-8", errors="ignore") as fh:
        for _ev, elem in iterparse(fh, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = elem.text
            elif tag == "text":
                text = elem.text
            elif tag == "page":
                if (title and text and ":" not in title
                        and not text.lstrip()[:12].upper().startswith("#REDIRECT")):
                    clean = _wiki_clean(text)
                    lead = _wiki_lead(clean)
                    if lead:
                        offer = f"the description and definition of {title}: {lead}"
                        content = f"{title}\n{clean[:20000]}"
                        yield ("wiki-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:16],
                               content.encode("utf-8"), offer)
                        count += 1
                        if max_articles and count >= max_articles:
                            elem.clear(); return
                title = text = None
                elem.clear()                             # free the parsed page (streaming, flat memory)


def _local_authority(cache_dir) -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """A stable self-authority for the leaf, generated once and persisted in the cache dir. A stable
    key means the ontology survives restarts, since boot can verify it, and the foundation is
    exportable under a fixed authority. The private key stays on the machine; the public key
    travels."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    kp = Path(cache_dir) / "authority.key"
    if kp.exists():
        priv = Ed25519PrivateKey.from_private_bytes(kp.read_bytes())
    else:
        priv = Ed25519PrivateKey.generate()
        kp.write_bytes(priv.private_bytes(serialization.Encoding.Raw,
                                          serialization.PrivateFormat.Raw, serialization.NoEncryption()))
    return priv, priv.public_key()


def open_local(*, embedder: Optional[Embedder] = None) -> Ember:
    """Reopen an already-filled leaf from disk, rehydrating the ontology. No re-ingest, no
    network."""
    s = load_settings()
    _, pub = _local_authority(s.cache_dir)
    return Ember.boot(s, embedder=embedder or _embedder(), authority_pub=pub)


def ingest_local(paths=None, *, wordnet: bool = False, wiki_dump: Optional[str] = None,
                 embedder: Optional[Embedder] = None, anchors: Optional[AnchorSet] = None,
                 batch: int = 1000) -> Tuple[Ember, int, str, object]:
    """Stand up a standalone leaf and fill its ontology from any of: file paths (code/docs), WordNet
    (lexical ground truth), a Wikipedia dump (conceptual ground truth). Each item is (id, content,
    offer); indexing runs on the offer, retrieval on the content. Content-derived anchors, persisted,
    clean rebuild. Strictly local + deterministic. Returns (ember, n_items, authority, priv)."""
    import shutil
    s = load_settings(); s.cache_dir.mkdir(parents=True, exist_ok=True)
    emb = embedder or _embedder()

    items: List[Tuple[str, bytes, str]] = []
    if paths:
        items += _describe_files([paths] if isinstance(paths, str) else list(paths))
    if wordnet:
        items += [(aid, content, offer) for aid, content, offer, _meta in wordnet_items()]
    if wiki_dump:
        items += list(wikipedia_items(wiki_dump))
    if not items:
        raise ValueError("nothing to ingest (give paths, wordnet=True, or wiki_dump=...)")

    for f in Path(s.cache_dir).iterdir():                    # clean rebuild (keep the stable authority)
        if f.name == "authority.key":
            continue
        shutil.rmtree(f) if f.is_dir() else f.unlink()

    vectors = emb.encode([offer for _, _, offer in items])   # embed the offers (the retrieval field)
    aset = _require_anchorset(anchors, emb)
    priv, pub = _local_authority(s.cache_dir)
    ember = Ember.boot(s, embedder=emb, authority_pub=pub).seed(aset, pub)
    # Author with the already-computed vectors (cache.put), avoiding a second embed pass — essential
    # at WordNet/Wikipedia scale. Native embedder => raw vector == the aligner's, so routing is exact.
    n = 0
    for i in range(0, len(items), batch):
        vecs = [(items[j][0], items[j][1], vectors[j]) for j in range(i, min(i + batch, len(items)))]
        ember.cache.put(vecs, version=1, authority="ember-local", priv=priv, provenance=Provenance.OBSERVED)
        n += len(vecs)
    ember.flush()
    return ember, n, "ember-local", priv


def ingest(ember: Ember, paths: Sequence[str], authority: str, priv, *,
           chunk_lines: int = 40, batch: int = 200) -> int:
    """Author repo code into Ember's cache. The path+line is kept as a header IN the content, so the
    location travels with the artifact (context vars). Provenance OBSERVED — real code, high rung."""
    items: List[Tuple[str, bytes, str]] = []
    n = 0
    for p, start, chunk in walk_chunks(paths, chunk_lines=chunk_lines):
        body = f"# {p.as_posix()}:{start}\n{chunk}"
        aid = "code-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]
        items.append((aid, body.encode("utf-8"), body))
        if len(items) >= batch:
            ember.remember(items, provenance=Provenance.OBSERVED, authority=authority, priv=priv)
            n += len(items); items = []
    if items:
        ember.remember(items, provenance=Provenance.OBSERVED, authority=authority, priv=priv)
        n += len(items)
    return n


def _observe_files(paths: Sequence[str], exts: Optional[set] = None):
    """Walk files and describe each at first observation by invoking a describe-operator
    (operators.describe_at_first_observation). Yields per file:
        (artifact_id, content_type, offer, content, operator_name)
    `offer` -> artifact.context (indexed); `content` -> artifact.content (retrieved);
    `operator_name` -> the operator edge (content-context-operator triple)."""
    import os
    from ember.runtime.runner import operators as ops
    exts = exts or CODE_EXT
    seen = set()
    for root in paths:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                suf = p.suffix.lower()
                if suf not in exts:
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if not text.strip():
                    continue
                key = p.as_posix()
                if key in seen:
                    continue
                seen.add(key)
                op_name, offer = ops.describe_at_first_observation(None, suffix=suf, text=text, path=key)
                content = f"{key}\n{text[:20000]}"
                aid = "file-" + hashlib.sha256(key.encode()).hexdigest()[:20]
                # a content_type from the suffix (prop-ish, but faithful mime-ish tag)
                ct = {".md": "text/markdown", ".py": "text/x-python"}.get(suf, "text/plain")
                yield aid, ct, offer, content, op_name


def ingest_to_store(paths=None, *, wordnet: bool = False, wiki_dump: Optional[str] = None,
                    embedder: Optional[Embedder] = None, anchors: Optional[AnchorSet] = None, batch: int = 1000):
    """Store-backed ingest: write artifacts (context=offer via operator-describe, plus
    operator edges) into the durable local mantle-shard (SQLite lattice + local CAS), then build the
    Ember anchor/IVF index from those offers as a derived, rebuildable cache. The store is
    truth; the index is throwaway. Returns (ember, n_items, store).

    Uses the same describe-operator mechanism as the platform (operators.invoke mirrors
    Mantle's InvokeArtifactRequest). Strictly local + deterministic."""
    import shutil
    from mantle.shard.local_store import open_store
    from ember.runtime.runner import operators as ops

    store = open_store()
    ops.register_operators(store.artifacts, store.graph)

    s = load_settings(); s.cache_dir.mkdir(parents=True, exist_ok=True)
    emb = embedder or _embedder()

    # 1. observe + describe -> durable artifacts + operator edges.
    docs: List[dict] = []
    edges: List[tuple] = []
    index_rows: List[Tuple[str, bytes, str]] = []           # (id, content_bytes, offer) for the index
    if paths:
        for aid, ct, offer, content, op_name in _observe_files([paths] if isinstance(paths, str) else list(paths)):
            docs.append({"id": aid, "content_type": ct, "state": "committed",
                         "context": offer, "content": content, "created_by": "ember-local"})
            edges.append((aid, op_name, "operator", {"at": "first-observation"}))
            index_rows.append((aid, content.encode("utf-8"), offer))
    if wordnet:
        for aid, content_bytes, offer, meta in wordnet_items():
            docs.append({"id": aid, "content_type": "text/x-wordnet", "state": "committed",
                         "context": offer, "content": content_bytes.decode("utf-8", "ignore"),
                         "created_by": "ember-local",
                         "lemmas": meta["lemmas"], "word": meta["word"], "pos": meta["pos"]})
            index_rows.append((aid, content_bytes, offer))
    if wiki_dump:
        for aid, content_bytes, offer in wikipedia_items(wiki_dump):
            docs.append({"id": aid, "content_type": "text/x-wikipedia", "state": "committed",
                         "context": offer, "content": content_bytes.decode("utf-8", "ignore"),
                         "created_by": "ember-local"})
            index_rows.append((aid, content_bytes, offer))
    if not docs:
        raise ValueError("nothing to ingest (give paths, wordnet=True, or wiki_dump=...)")

    # The cache dir is wiped only now, once there is something to rebuild the index from: the wipe
    # clears the index only (not the store), and the index is not built until step 3 below.
    for f in Path(s.cache_dir).iterdir():                    # clean rebuild of the index (not the store)
        if f.name == "authority.key":
            continue
        shutil.rmtree(f) if f.is_dir() else f.unlink()

    # 2. write durable (batched, idempotent).
    n = store.artifacts.put_many(docs, batch=500)
    if edges:
        store.graph.add_edges(edges, batch=500)

    # 3. build index from the offers (derived cache; rebuildable via rebuild_index_from_store).
    vectors = emb.encode([offer for _, _, offer in index_rows])
    aset = _require_anchorset(anchors, emb)
    priv, pub = _local_authority(s.cache_dir)
    ember = Ember.boot(s, embedder=emb, authority_pub=pub).seed(aset, pub)
    for i in range(0, len(index_rows), batch):
        vecs = [(index_rows[j][0], index_rows[j][1], vectors[j]) for j in range(i, min(i + batch, len(index_rows)))]
        ember.cache.put(vecs, version=1, authority="ember-local", priv=priv, provenance=Provenance.OBSERVED)
    ember.flush()
    return ember, n, store


def rebuild_index_from_store(*, embedder: Optional[Embedder] = None, anchors: Optional[AnchorSet] = None, batch: int = 1000):
    """The store is the source of truth: deletes the Ember index and regenerates it purely from the
    durable artifacts (embeds their contexts/offers, rebuilds the cache). Anchors are provisioned
    and passed in rather than rebuilt, so a reindex lands in the same cells as before. No
    re-observation, no re-describe — the offers already live in the store."""
    import shutil
    from mantle.shard.local_store import open_store
    from ember.runtime.runner import operators as ops
    store = open_store()
    s = load_settings()
    emb = embedder or _embedder()

    # Reads and validates the store before deleting anything: an empty store raises before the
    # index is touched. A precondition checked after the destructive step is not a precondition.
    rows: List[Tuple[str, bytes, str]] = []
    rungs: List[str] = []
    for a in store.artifacts.list_artifacts(state="committed"):
        if a.get("content_type") == ops.OPERATOR_CONTENT_TYPE:  # skip operator artifacts in the retrieval index
            continue
        offer = a.get("context") or ""
        content = (a.get("content") or "").encode("utf-8")
        rows.append((a["id"], content, offer))
        # preserves the stored rung; see the put loop below.
        rungs.append(str(a.get("provenance") or Provenance.OBSERVED))
    if not rows:
        raise ValueError("store is empty — nothing to rebuild from")

    for f in Path(s.cache_dir).iterdir():
        if f.name == "authority.key":
            continue
        shutil.rmtree(f) if f.is_dir() else f.unlink()

    vectors = emb.encode([offer for _, _, offer in rows])
    aset = _require_anchorset(anchors, emb)
    priv, pub = _local_authority(s.cache_dir)
    ember = Ember.boot(s, embedder=emb, authority_pub=pub).seed(aset, pub)
    # Each artifact keeps its own stored rung rather than being rebuilt at OBSERVED: store docs
    # carry their own provenance (`Observation.to_doc`, and an imported shard's stamp can be
    # anything), and `cache.put` ships the rung to peers as `ShardItem.consensus`. Rebuilding
    # everything at OBSERVED's 0.90-0.98 would inflate an artifact stored at `hypothesis`
    # (0.20-0.55) or `assertion` (0.02), inverting the ladder invariant `prism.mass` enforces —
    # belief follows how a thing was obtained. An unknown/unparseable rung falls back to OBSERVED
    # because that is what a locally-read file genuinely is; a doc that records a lower rung keeps
    # it. Grouped because `cache.put` takes one rung per call, and the groups are tiny in practice
    # (a handful of distinct rungs across the corpus).
    order = sorted(range(len(rows)), key=lambda j: rungs[j])
    grouped: Dict[str, List[int]] = {}
    for j in order:
        grouped.setdefault(rungs[j], []).append(j)
    for rung, idxs in grouped.items():
        try:
            prov = Provenance(rung)
        except Exception:
            prov = Provenance.OBSERVED          # unrecognised rung -> do not invent authority
        for i in range(0, len(idxs), batch):
            chunk = idxs[i:i + batch]
            vecs = [(rows[j][0], rows[j][1], vectors[j]) for j in chunk]
            ember.cache.put(vecs, version=1, authority="ember-local", priv=priv, provenance=prov)
    ember.flush()
    return ember, len(rows), store


def remember_offload(ember: Ember, question: str, answer: str, authority: str, priv) -> None:
    """Cache an offloaded (Lumen/LLM) answer as a triple at HYPOTHESIS rung — unverified dark matter
    that a future confirmation can raise. Next time the question recurs, the leaf hits this locally
    instead of re-offloading. The other half of live-learning."""
    body = f"Q: {question}\nA: {answer}"
    aid = "offload-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]
    ember.remember([(aid, body.encode("utf-8"), question)],   # keyed/embedded on the question
                   provenance=Provenance.HYPOTHESIS, authority=authority, priv=priv)


__all__ = ["bootstrap_local", "walk_chunks", "ingest", "remember_offload", "CODE_EXT"]
