"""Ember's way to Mantle: Mantle's own HTTP API, never Mantle's store (John, 2026-11-02).

Ember keeps a store of its own, in its own folder and under its own key, for what is Ember's: its
corpus, operators, index and records. What belongs to a person is Mantle's, and Ember reaches it
over Mantle's routes like any other client, so every read and write passes Mantle's access checks
and, under threshold custody, Origin's verdict:

=================================  ======================================  =====================
Ember's act                        Mantle route                            Credential
=================================  ======================================  =====================
a memory (``remember``)            ``POST /artifacts`` (``container_id``)  the caller's delegation
the read check                     ``GET /grants/my-access`` (a verdict)   the caller's delegation
=================================  ======================================  =====================

Ember opens no key of a person's (John, 2026-11-02, review 10). The memory collection is the
person's own, made by Mantle at their sign-in on their own token; Ember names it by its derived id
and never creates or reads it, so a memory is sealed to the collection's drop key, written without
Read. Ember shares nothing (John, 2026-11-02, review 11): sharing happens only in the person's own
session. The read check asks Mantle for a verdict, never for content. Ember's index holds none of Mantle's rows.

A person's act is made on that person's own Origin token, the one Ember verified for the request,
so it is theirs by Mantle's rules (their creation grant; their Share or Admin for a grant). Ember's
own reads are made on Ember's own credential: an Origin account for this Ember, whose reach is
whatever it has been granted. Its refresh token (``EMBER_MANTLE_REFRESH_TOKEN``, secret, the first
one only) is exchanged at Origin's ``/auth/token`` for an access token, kept until a minute before it
expires. Origin rotates the refresh token on every use, so the newest it returns is kept in Ember's
own keys folder (``$KEYS_DIR/mantle.refresh``, 0600) and is the only one ever presented again;
``EMBER_MANTLE_TOKEN`` (secret) is a fixed bearer, for development. Deactivating that account at
Origin ends it.

``EMBER_MANTLE_URI`` names Mantle. Unset, Ember is a standalone leaf with no Mantle, and its person
paths write into its own store as before.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, NamedTuple, Optional, Tuple, Union
from urllib.parse import quote

import httpx

TIMEOUT = 15.0
#: Ember's memory collection, per person: a natural key Mantle derives the id from per caller, so
#: each person's lands in their own collection and a repeat write finds it again.
MEMORY_IDENTITY = "ember:memory"
#: The commons, as Mantle names it: a Read grant to it publishes (`db/access.PUBLIC_PRINCIPAL`).
PUBLIC_PRINCIPAL = "public"
#: The file in Ember's own keys folder that keeps the newest refresh token Origin returned, the
#: lock that lets one process rotate at a time, and the mark that the seed has been spent.
KEPT_REFRESH = "mantle.refresh"
ROTATION_LOCK = "mantle.refresh.lock"
SEED_SPENT = "mantle.refresh.seed-spent"
#: The largest answer Ember reads from Mantle or Origin, and the most items it takes per page.
MAX_RESPONSE = 4 * 1024 * 1024
MAX_PAGE = 200
#: How Ember names a row of Mantle's: never one of Ember's own ids, and never in Ember's index.
MANTLE_PREFIX = "mantle:"
#: How long a Read grant Ember makes for a person lasts: Mantle refuses longer (review 10, R2).
SHARE_DAYS = 7


class MantleRefused(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"Mantle refused ({status}): {detail}")
        self.status, self.detail = status, detail


class MantleUnavailable(RuntimeError):
    pass


def _transport():
    """The HTTP transport (a seam for tests); None is httpx's default."""
    return None


def _client() -> httpx.Client:
    t = _transport()
    kw = {"timeout": TIMEOUT, "follow_redirects": False}
    return httpx.Client(transport=t, **kw) if t is not None else httpx.Client(**kw)


def configured() -> bool:
    return bool((os.getenv("EMBER_MANTLE_URI") or "").strip())


def _read_capped(r: httpx.Response) -> bytes:
    """The answer's bytes, refused past `MAX_RESPONSE`: a declared length over the cap before
    anything is read, and a streamed one as soon as it passes it."""
    declared = r.headers.get("content-length")
    if declared and declared.strip().isdigit() and int(declared) > MAX_RESPONSE:
        raise MantleRefused(r.status_code, f"answer too large ({declared} bytes declared)")
    buf = bytearray()
    for chunk in r.iter_bytes():
        buf += chunk
        if len(buf) > MAX_RESPONSE:
            raise MantleRefused(r.status_code, "answer too large")
    return bytes(buf)


def _request(method: str, url: str, **kw) -> Tuple[int, bytes]:
    """One request, its answer read capped while it streams."""
    with _client() as c:
        with c.stream(method, url, **kw) as r:
            return r.status_code, _read_capped(r)


def _json_of(status: int, raw: bytes) -> dict:
    try:
        out = json.loads(raw or b"{}")
    except ValueError as exc:
        raise MantleRefused(status, "answer is not JSON") from exc
    if not isinstance(out, dict):
        raise MantleRefused(status, "answer is not an object")
    return out


def _body(r: httpx.Response) -> dict:
    if len(r.content) > MAX_RESPONSE:
        raise MantleRefused(r.status_code, f"answer too large ({len(r.content)} bytes)")
    try:
        out = r.json()
    except ValueError as exc:
        raise MantleRefused(r.status_code, "answer is not JSON") from exc
    if not isinstance(out, dict):
        raise MantleRefused(r.status_code, "answer is not an object")
    return out


@contextlib.contextmanager
def _file_lock(path: Path):
    """An exclusive lock on `path` across processes: one rotation at a time."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.02)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_private(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".next")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass


def refresh_client_id() -> str:
    """The client Ember's refresh family was issued to: Origin's refresh grant checks it against
    the family (review 11, M1). `platform` unless `EMBER_ORIGIN_CLIENT_ID` names another."""
    return (os.getenv("EMBER_ORIGIN_CLIENT_ID") or "platform").strip()


class EmberCredential:
    """Ember's own credential for Mantle: an access token from Origin, minted from Ember's refresh
    token and kept until shortly before it expires, or a fixed bearer.

    Origin rotates a refresh token on every use and treats a used one presented again as stolen,
    revoking its family. So: the newest token is kept in Ember's own keys folder and is the only one
    presented; one process rotates at a time (a file lock beside it), and re-reads the kept token
    under the lock; a lost answer is retried once with the same token, which Origin's grace answers
    with the same next token; and once the seed has been spent it is never presented again, even if
    the kept file is lost. With nowhere to keep the next token a refresh token is refused."""

    def __init__(self, *, origin: str = "", refresh_token: str = "", token: str = "",
                 keys: Optional[Path] = None):
        self._origin = origin.rstrip("/")
        self._seed = refresh_token
        self._fixed = token
        self._keys = keys
        self._held: Tuple[str, float] = ("", 0.0)
        self._lock = threading.Lock()
        #: Set when Origin refused the refresh family (revoked, or a used token): Ember's own
        #: credential is then off until re-provisioned, and is not asked for again.
        self._refused: Optional[str] = None

    @classmethod
    def from_env(cls) -> Optional["EmberCredential"]:
        refresh = (os.getenv("EMBER_MANTLE_REFRESH_TOKEN") or "").strip()
        fixed = (os.getenv("EMBER_MANTLE_TOKEN") or "").strip()
        if not refresh and not fixed:
            return None
        origin = (os.getenv("EMBER_ORIGIN_URI") or os.getenv("ORIGIN_URI") or "").strip()
        if refresh and not origin:
            raise MantleUnavailable("EMBER_MANTLE_REFRESH_TOKEN needs EMBER_ORIGIN_URI (Origin "
                                    "exchanges it for an access token)")
        keys = (os.getenv("KEYS_DIR") or os.getenv("EMBER_STORE_KEYS_DIR") or "").strip()
        if refresh and not keys:
            raise MantleUnavailable("EMBER_MANTLE_REFRESH_TOKEN needs Ember's own keys folder "
                                    "(KEYS_DIR) to keep the token Origin rotates it into")
        return cls(origin=origin, refresh_token=refresh, token=fixed,
                   keys=Path(keys) if refresh else None)

    def _seed_digest(self) -> str:
        return hashlib.sha256(self._seed.encode("utf-8")).hexdigest()

    def _current_refresh(self) -> str:
        """The refresh token to present: the kept one, else the seed if it was never spent."""
        kept_path = self._keys / KEPT_REFRESH
        try:
            kept = kept_path.read_text(encoding="ascii").strip()
        except OSError:
            kept = ""
        if kept:
            return kept
        try:
            spent = (self._keys / SEED_SPENT).read_text(encoding="ascii").strip()
        except OSError:
            spent = ""
        if spent == self._seed_digest():
            raise MantleUnavailable("Ember's kept refresh token is missing and its seed was already "
                                    "spent; re-provision EMBER_MANTLE_REFRESH_TOKEN")
        return self._seed

    def _exchange(self, refresh: str) -> Tuple[int, bytes]:
        """The refresh grant, its answer read capped while it streams."""
        last: Optional[Exception] = None
        for _attempt in (1, 2):              # a lost answer: once more, with the same token
            try:
                return _request("POST", f"{self._origin}/auth/token",
                                data={"grant_type": "refresh_token", "refresh_token": refresh,
                                      "client_id": refresh_client_id()})
            except httpx.HTTPError as exc:
                last = exc
        raise MantleUnavailable(f"Origin is unreachable ({type(last).__name__})") from last

    def __call__(self) -> str:
        if not self._seed:
            return self._fixed
        with self._lock:
            if self._refused:
                raise MantleRefused(400, self._refused)
            tok, until = self._held
            if tok and time.time() < until:
                return tok
            self._keys.mkdir(parents=True, exist_ok=True)
            with _file_lock(self._keys / ROTATION_LOCK):
                refresh = self._current_refresh()
                status, raw = self._exchange(refresh)
                if status != 200:
                    # A revoked family, or a used token: stop here, say so once, and leave every
                    # person's own path untouched (they never use this credential).
                    self._refused = ("Origin refused Ember's refresh token; Ember's own "
                                     "credential is off until EMBER_MANTLE_REFRESH_TOKEN is "
                                     "provisioned again")
                    import logging
                    logging.getLogger(__name__).error(self._refused)
                    raise MantleRefused(status, self._refused)
                body = _json_of(status, raw)
                nxt = str(body.get("refresh_token") or "")
                if nxt:
                    _write_private(self._keys / KEPT_REFRESH, nxt)
                    if refresh == self._seed:
                        _write_private(self._keys / SEED_SPENT, self._seed_digest())
            # Origin issues Ember's account only a delegation narrowed to Ember (review 11, M5):
            # it is held as Origin gave it, and Mantle enforces it as an Ember delegation.
            life = float(300 if body.get("expires_in") is None else body["expires_in"])
            self._held = (str(body.get("access_token") or ""), time.time() + max(0.0, life - 30.0))
            return self._held[0]


def _addressed_to_ember(token: str) -> bool:
    """Is `token` (unverified) a JWT addressed to Ember?"""
    import base64

    try:
        seg = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))
    except (IndexError, ValueError, TypeError):
        return False
    aud = claims.get("aud") if isinstance(claims, dict) else None
    return aud == "ember" or (isinstance(aud, list) and "ember" in aud)


TOKEN_EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN = "urn:ietf:params:oauth:token-type:access_token"
_EXCHANGED: dict = {}
_EXCHANGED_LOCK = threading.Lock()


def exchange_for_mantle(delegation: str) -> Optional[str]:
    """The person's delegation addressed to Mantle, from Ember's exchange of `delegation` at
    Origin's token endpoint (RFC 8693), Ember's own account's token as `actor_token`; None when
    Origin makes none. Kept until shortly before it expires."""
    origin = (os.getenv("EMBER_ORIGIN_URI") or os.getenv("ORIGIN_URI") or "").strip().rstrip("/")
    if not origin:
        return None
    key = hashlib.sha256(delegation.encode("utf-8")).hexdigest()
    with _EXCHANGED_LOCK:
        held = _EXCHANGED.get(key)
        if held and held[1] > time.time():
            return held[0]
    try:
        own = EmberCredential.from_env()
        actor = own() if own is not None else ""
    except (MantleRefused, MantleUnavailable):
        actor = ""
    if not actor:
        return None
    try:
        status, raw = _request("POST", f"{origin}/auth/token", data={
            "grant_type": TOKEN_EXCHANGE, "subject_token": delegation,
            "subject_token_type": ACCESS_TOKEN, "audience": "mantle", "actor_token": actor})
    except httpx.HTTPError:
        return None
    if status != 200:
        return None
    try:
        body = _json_of(status, raw)
    except MantleRefused:
        return None
    tok = str(body.get("access_token") or "")
    if not tok or _addressed_to_ember(tok):
        return None
    life = float(300 if body.get("expires_in") is None else body["expires_in"])
    with _EXCHANGED_LOCK:
        _EXCHANGED[key] = (tok, time.time() + max(0.0, life - 30.0))
        for k in [k for k, (_t, until) in _EXCHANGED.items() if until <= time.time()]:
            _EXCHANGED.pop(k, None)
    return tok


class _OwnForMantle:
    """Ember's own credential as Mantle sees it (review 22, F5). Origin issues Ember's account a
    delegation narrowed to Ember (`aud` `ember`), which Ember itself takes; so Ember exchanges it,
    as it does a caller's, for one addressed to Mantle, and never sends it on. A fixed bearer that
    is not a token addressed to Ember (a development credential) is sent as it is."""

    def __init__(self, own: "EmberCredential"):
        self._own = own

    def __call__(self) -> str:
        tok = self._own()
        if not _addressed_to_ember(tok):
            return tok
        exchanged = exchange_for_mantle(tok)
        if not exchanged:
            raise MantleUnavailable("Origin made no exchange of Ember's own credential for "
                                    "Mantle, so Ember makes no call of its own there")
        return exchanged


class MantleAPI:
    """Mantle's HTTP API, on one credential: a token, or a callable that returns one."""

    def __init__(self, base: str, credential: Union[str, Callable[[], str]]):
        self._base = base.rstrip("/")
        self._credential = credential

    @classmethod
    def from_env(cls, *, caller_token: Optional[str] = None) -> Optional["MantleAPI"]:
        """Mantle as configured. `caller_token=None` asks for Ember's own credential; any other
        value is the caller's credential addressed to Mantle, and an empty one is no credential
        at all, never Ember's own. A delegation addressed to Ember is never sent (review 21, F6):
        `for_caller` exchanges it first."""
        base = (os.getenv("EMBER_MANTLE_URI") or "").strip()
        if not base:
            return None
        if caller_token is not None:
            if not caller_token.strip() or _addressed_to_ember(caller_token):
                return None
            return cls(base, caller_token)
        own = EmberCredential.from_env()
        return cls(base, _OwnForMantle(own)) if own is not None else None

    @classmethod
    def for_caller(cls, delegation: Optional[str]) -> Optional["MantleAPI"]:
        """Mantle on the caller's credential addressed to Mantle (review 21, F6): Ember exchanges
        the delegation it was sent (`aud` `ember`) at Origin, with its own account's token as the
        actor, for the same person's delegation addressed to Mantle. None when there is no Mantle,
        no delegation, or Origin makes no exchange; the delegation sent to Ember never leaves."""
        if not (os.getenv("EMBER_MANTLE_URI") or "").strip() or not (delegation or "").strip():
            return None
        exchanged = exchange_for_mantle(delegation)
        return cls.from_env(caller_token=exchanged) if exchanged else None

    def _token(self) -> str:
        c = self._credential
        return c() if callable(c) else str(c or "")

    def _call(self, method: str, path: str, *, body: Optional[dict] = None,
              params: Optional[dict] = None, ok=(200, 201)) -> Optional[dict]:
        headers = {"Authorization": f"Bearer {self._token()}"}
        try:
            status, raw = _request(method, f"{self._base}{path}", headers=headers, json=body,
                                   params=params)
        except httpx.HTTPError as exc:
            raise MantleUnavailable(f"Mantle is unreachable ({type(exc).__name__})") from exc
        if status == 404:
            return None
        if status not in ok:
            try:
                detail = str(_json_of(status, raw).get("detail") or "")[:300]
            except MantleRefused:
                detail = ""
            raise MantleRefused(status, detail)
        return _json_of(status, raw)

    def subject(self) -> str:
        """The person this credential acts for: its `sub` (Ember verified the token already)."""
        from jose import jwt as jose_jwt
        try:
            return str(jose_jwt.get_unverified_claims(self._token()).get("sub") or "")
        except Exception:  # noqa: BLE001 — not a JWT: no person
            return ""

    @staticmethod
    def _seg(artifact_id: str) -> str:
        return quote(str(artifact_id), safe="")

    def memory_collection_id(self) -> str:
        """The caller's own memory collection, by the id Mantle derives for it: Mantle makes it
        at their sign-in, on their own token. Ember never creates or reads it."""
        from mantle.services.artifact_identity import derive_artifact_id
        person = self.subject()
        if not person:
            raise MantleRefused(401, "this credential names no person")
        return derive_artifact_id(person, MEMORY_IDENTITY)

    def create_in(self, container_id: str, *, content: str, content_type: str, name: str,
                  context: Optional[dict] = None) -> dict:
        body = {"container_id": container_id, "content": content, "content_type": content_type,
                "name": name}
        if context:
            body["context"] = context
        return self._call("POST", "/artifacts", body=body)

    def get(self, artifact_id: str) -> Optional[dict]:
        return self._call("GET", f"/artifacts/{self._seg(artifact_id)}")

    def may(self, resource_id: str, action: str) -> bool:
        """Mantle's verdict on whether the caller may `action` the resource: no content read."""
        got = self._call("GET", "/grants/my-access",
                         params={"resource_id": str(resource_id), "action": action})
        return bool((got or {}).get("allowed"))

    def visible(self, *, limit: int = MAX_PAGE, offset: int = 0) -> dict:
        return self._call("GET", "/artifacts/visible",
                          params={"limit": min(int(limit), MAX_PAGE),
                                  "offset": int(offset)}) or {"items": []}


class MantleRow(NamedTuple):
    """One artifact Ember's credential may read in Mantle, as Ember indexes it: under an id of
    Mantle's namespace (`mantle:<id>`), so it can never stand for one of Ember's own, and with its
    own author (`mantle:<created_by>`), never re-signed as Ember's."""
    id: str
    content: bytes
    context: str
    content_type: str
    author: str
    provenance: Optional[str]


def visible_rows(api: Optional[MantleAPI], *, page: int = MAX_PAGE,
                 cap: int = 100_000) -> Iterator[MantleRow]:
    """Every artifact Ember's credential may read in Mantle, for the index. Mantle decides what
    that is."""
    if api is None:
        return
    offset = 0
    while offset < cap:
        got = api.visible(limit=page, offset=offset)
        items = (got.get("items") or [])[:MAX_PAGE]
        for a in items:
            if not isinstance(a, dict) or not a.get("id"):
                continue
            ctx = a.get("context")
            yield MantleRow(
                id=MANTLE_PREFIX + str(a["id"]),
                content=str(a.get("content") or "").encode("utf-8"),
                context=ctx if isinstance(ctx, str) else json.dumps(ctx or {}),
                content_type=str(a.get("content_type") or ""),
                author=MANTLE_PREFIX + str(a.get("created_by") or "unknown"),
                provenance=a.get("provenance") if isinstance(a.get("provenance"), str) else None)
        if not items or not got.get("has_more"):
            return
        offset += len(items)


def readable_by(cited, api: Optional[MantleAPI]) -> bool:
    """May the caller see an answer citing these ids? Ember's own corpus, yes; a row from Mantle
    only if the caller's own credential (`api`, on their delegation) reads it in Mantle now."""
    for cid in cited or ():
        cid = str(cid)
        if not cid.startswith(MANTLE_PREFIX):
            continue
        if api is None:
            return False
        try:
            if not api.may(cid[len(MANTLE_PREFIX):], "read"):
                return False
        except (MantleRefused, MantleUnavailable):
            return False
    return True


def refuse_mantle_rows(rows) -> None:
    """Ember's index never holds a row of Mantle's (review 10, R7): nothing in an answer can rest
    on Mantle text a caller may not read, whatever the answer cites."""
    for row in rows or ():
        if str(row[0]).startswith(MANTLE_PREFIX):
            raise ValueError(f"Ember's index never holds a row of Mantle's ({row[0]})")


def _plain(p: str) -> str:
    """`p` without the Windows spellings that name the same place another way: the long-path
    prefix (`\\\\?\\`, `\\\\?\\UNC\\`) and this machine's own administrative share
    (`\\\\localhost\\c$\\...`, `\\\\127.0.0.1\\c$\\...`)."""
    q = str(p).strip()
    if os.name != "nt":
        return q
    q = q.replace("/", "\\")
    low = q.lower()
    if low.startswith("\\\\?\\unc\\"):
        q = "\\\\" + q[8:]
        low = q.lower()
    elif low.startswith("\\\\?\\") or low.startswith("\\\\.\\"):
        q = q[4:]
        low = q.lower()
    for host in ("localhost", "127.0.0.1", os.environ.get("COMPUTERNAME", "").lower() or "\0"):
        lead = "\\\\" + host + "\\"
        if low.startswith(lead):
            rest = q[len(lead):]
            share, _, tail = rest.partition("\\")
            if len(share) == 2 and share[1] == "$" and share[0].isalpha():
                q = share[0] + ":\\" + tail
            break
    return q


def _real(p: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(_plain(p))))


def _forbidden_dirs() -> list:
    """Mantle's folders, from settings: the lattice's folder, `MANTLE_DATA_DIR`, and every folder
    in `EMBER_FORBIDDEN_DIRS` (separated by the path separator or commas)."""
    out = []
    lattice = (os.getenv("MANTLE_LATTICE_PATH") or "").strip()
    if lattice:
        out.append(os.path.dirname(lattice) or ".")
    for key in ("MANTLE_DATA_DIR", "MANTLE_KEYS_DIR", "EMBER_FORBIDDEN_DIRS"):
        for part in (os.getenv(key) or "").replace(",", os.pathsep).split(os.pathsep):
            if part.strip():
                out.append(_plain(part.strip()))
    return [_real(d) for d in out]


def ember_store_file() -> str:
    """The store file Ember's `open_sqlite_store` opens: `EMBER_SQLITE_DB` under
    `EMBER_SQLITE_DIR` (an absolute `EMBER_SQLITE_DB` stands alone)."""
    root = (os.getenv("EMBER_SQLITE_DIR") or "").strip() or str(Path.home() / "genesis-shard")
    db = (os.getenv("EMBER_SQLITE_DB") or "").strip() or "lattice.db"
    return os.path.join(root, db)


def _within(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:
        return False


def _same_file_in(fd_stat, folder: str) -> bool:
    """Is the file `fd_stat` describes one of the files directly in `folder` or its `keys`,
    compared by identity (device and inode), so a hard link is seen for what it is?"""
    for d in (folder, os.path.join(folder, "keys")):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            try:
                st = os.stat(os.path.join(d, n), follow_symlinks=False)
            except OSError:
                continue
            if (st.st_dev, st.st_ino) == (fd_stat.st_dev, fd_stat.st_ino) and st.st_ino:
                return True
    return False


def refuse_opened_store(path: str) -> None:
    """Check the store file through an opened handle: refused if it is in Mantle's folders by real
    path, if it has another name (a hard link: a store file of Ember's is never linked), or if it
    is, by identity, a file in Mantle's folders."""
    bad = _forbidden_dirs()
    real = _real(path)
    for d in bad:
        if _within(real, d):
            raise RuntimeError(f"Ember's store ({real}) lies in Mantle's folder {d}")
    try:
        fd = os.open(_plain(path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        return                                   # nothing there yet: the store makes it
    try:
        st = os.fstat(fd)
    finally:
        os.close(fd)
    if st.st_nlink > 1:
        raise RuntimeError(f"Ember's store ({real}) has another name (a hard link); Ember's own "
                           f"store is never linked")
    for d in bad:
        if _same_file_in(st, d):
            raise RuntimeError(f"Ember's store ({real}) is a file in Mantle's folder {d}")


def refuse_mantle_store() -> None:
    """Refuse a store of Ember's that lies in any of Mantle's folders, compared as real paths
    (links, junctions, `..`, case and Windows long-path and administrative-share spellings
    resolved): the store's folder and file, the content folder, and the keys folders; and the
    store file itself, by identity, if it exists."""
    bad = _forbidden_dirs()
    if not bad:
        if configured() or (os.getenv("AGIENCE_PLANE") or "").strip().lower() == "production":
            raise RuntimeError("Ember reaches Mantle by its API here, so it must know Mantle's "
                               "folders to keep out of them: set EMBER_FORBIDDEN_DIRS (or "
                               "MANTLE_DATA_DIR) to Mantle's data and keys folders")
        return
    mine = [_real((os.getenv("EMBER_SQLITE_DIR") or "").strip() or str(Path.home() / "genesis-shard")),
            _real(ember_store_file())]
    content = (os.getenv("EMBER_CONTENT_DIR") or "").strip()
    if content:
        mine.append(_real(content))
    for m in mine:
        for d in bad:
            if _within(m, d):
                raise RuntimeError(f"Ember's store ({m}) lies in Mantle's folder {d}; Ember keeps "
                                   f"its own store and reaches Mantle's data through Mantle's API")
    for key in ("EMBER_STORE_KEYS_DIR", "KEYS_DIR"):
        k = (os.getenv(key) or "").strip()
        if not k:
            continue
        for d in bad:
            if _within(_real(k), d):
                raise RuntimeError(f"Ember's keys folder ({key}={k}) lies in Mantle's folder {d}; "
                                   f"Ember holds only its own keys")
    refuse_opened_store(ember_store_file())


#: The content mirror's default bucket when `EMBER_OVH_BUCKET` is unset
#: (`mantle.shard.content_tier.open_ovh_store`): Mantle's own durable content bucket.
_MIRROR_DEFAULT_BUCKET = "agience-genesis-content"


def _forbidden_buckets() -> set:
    named = {b.strip() for b in (os.getenv("EMBER_FORBIDDEN_BUCKETS") or "").split(",")
             if b.strip()}
    for k in ("CONTENT_DURABLE_BUCKET", "CONTENT_EDGE_BUCKET"):
        if (os.getenv(k) or "").strip():
            named.add(os.getenv(k).strip())
    return named


def refuse_mantle_bucket(store) -> None:
    """Refuse a content mirror on Mantle's content bucket (review 12). Ember's own store mirrors
    its corpus to an object store; a credential for Mantle's bucket would let Ember list, read and
    overwrite Mantle's stored content, and Ember reaches Mantle by its API only. The mirror must be
    a bucket Ember names (`EMBER_OVH_BUCKET`: the default is Mantle's) and none Mantle uses
    (`EMBER_FORBIDDEN_BUCKETS`, and Mantle's own bucket settings when present). No mirror is fine."""
    content = getattr(store, "content", None)
    remote = getattr(content, "remote", None) if content is not None else None
    bucket = getattr(remote, "bucket", None)
    if remote is None or bucket is None:
        return
    mine = (os.getenv("EMBER_OVH_BUCKET") or "").strip()
    if not mine or bucket != mine:
        raise RuntimeError(f"Ember's content mirror names no bucket of its own (it would open "
                           f"{bucket!r}); set EMBER_OVH_BUCKET to Ember's own bucket")
    if bucket in _forbidden_buckets() | {_MIRROR_DEFAULT_BUCKET}:
        raise RuntimeError(f"Ember's content mirror {bucket!r} is Mantle's content bucket; Ember "
                           f"reaches Mantle's data through Mantle's API only")


def open_own_store(*args, **kwargs):
    """Ember's own store, after refusing one that lies in Mantle's folders: every store open
    goes through here."""
    refuse_mantle_store()
    from mantle.shard.local_store import open_store
    store = open_store(*args, **kwargs)
    # Checked again on what was opened: a folder swapped after the check is caught here.
    refuse_opened_store(ember_store_file())
    refuse_mantle_bucket(store)
    return store
