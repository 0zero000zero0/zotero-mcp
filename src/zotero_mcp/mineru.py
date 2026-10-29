"""MinerU cloud PDF parsing backend.

MinerU's ``vlm`` model is layout-aware: it reads double-column papers in
order and emits formulas as LaTeX (``$$ ... \\tag{3}$$``), tables as HTML,
and figure captions as separate blocks. That makes it a useful alternative
to pdf-inspector for papers whose equations matter; the cost is an upload,
a network round trip, and the MinerU page quota.

The public entry point is :func:`extract_pdf`. It uploads the file through
MinerU's batch file-URL API, polls the extraction job, downloads the result
zip, and turns ``content_list.json`` into per-page Markdown. Parsed results
are cached by file content + parsing options so a page-range read does not
pay for the same parse twice.

A note on page ranges: MinerU's batch API accepts ``page_ranges`` inside
each ``files[]`` entry, not at the top level. A top-level field is silently
ignored (that is how the first probe ended up parsing all nine pages of a
paper when only two were requested).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from .utils import USER_AGENT, ensure_private_dir, write_json_atomic

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://mineru.net/api/v4"
DEFAULT_MODEL = "vlm"
DEFAULT_LANGUAGE = "ch"
DEFAULT_TIMEOUT = 1800.0
DEFAULT_POLL_INTERVAL = 5.0

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_ZIP_BYTES = 1024 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_PAGES_PER_FILE = 200

_KEY_RE = re.compile(
    r"^\s*(?:export\s+)?mineru_api_key\s*=\s*(.*)$",
    re.IGNORECASE,
)

_API_ERROR_CODES = {
    "A0202": (
        "mineru_invalid_token",
        "MinerU rejected the API key (A0202). Check `mineru_api_key` in ~/.shell-env.sh.",
    ),
    "A0211": (
        "mineru_expired_token",
        "MinerU API key has expired (A0211). Create a new token at https://mineru.net/apiManage.",
    ),
    "-60005": (
        "mineru_file_too_large",
        "MinerU rejects files larger than 200 MB (-60005).",
    ),
    "-60006": (
        "mineru_page_limit",
        "MinerU accepts at most 200 pages per file (-60006). Use a page range.",
    ),
    "-60010": (
        "mineru_parse_failed",
        "MinerU could not parse this PDF (-60010).",
    ),
    "-60018": (
        "mineru_quota_exceeded",
        "MinerU daily page quota is exhausted (-60018). Try again tomorrow or use pdf-inspector.",
    ),
}


class MineruError(RuntimeError):
    """A MinerU request or response failed.

    ``code`` is stable enough for callers to branch on
    (``mineru_missing_key``, ``mineru_quota_exceeded``, ...).
    """

    def __init__(self, message: str, *, code: str = "mineru_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MineruPage:
    """One page of Markdown from a MinerU parse."""

    page: int
    markdown: str


@dataclass(frozen=True)
class MineruResult:
    """A completed MinerU parse."""

    pages: tuple[MineruPage, ...]
    page_count: int
    from_cache: bool = False

    @property
    def markdown(self) -> str:
        return "\n\n".join(page.markdown for page in self.pages if page.markdown.strip())


def _clean_key(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if value[:1] in ("'", '"'):
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            return None
        value = value[1:-1].strip()
    # Shell expansion or escape sequences mean this is not a literal key.
    if not value or any(char in value for char in ("$", "`", "\\")):
        return None
    return value


def _read_key_file(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        match = _KEY_RE.match(line)
        if match:
            key = _clean_key(match.group(1))
            if key:
                return key
    return None


def resolve_api_key(api_key: str | None = None) -> str:
    """Return the MinerU API key.

    Precedence: explicit argument, ``MINERU_API_KEY`` (or the lower-case
    spelling), then ``mineru_api_key`` in ``~/.shell-env.sh``. The last path
    matches the shell file used by the rest of this project's tooling.
    """

    if api_key and api_key.strip():
        return api_key.strip()
    for name in ("MINERU_API_KEY", "mineru_api_key"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    for filename in (".shell-env.sh", ".shell-env"):
        key = _read_key_file(Path.home() / filename)
        if key:
            return key
    raise MineruError(
        "MinerU API key not found. Set MINERU_API_KEY, or add "
        "`mineru_api_key=...` to ~/.shell-env.sh.",
        code="mineru_missing_key",
    )


def _api_error(code: object, message: str | None) -> MineruError:
    known = _API_ERROR_CODES.get(str(code))
    if known:
        stable, hint = known
        return MineruError(f"{hint} (server said: {message or 'no message'})", code=stable)
    return MineruError(
        f"MinerU API error {code}: {message or 'unknown error'}",
        code="mineru_api_error",
    )


def _join_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _render_block(block: dict) -> str:
    """Turn one MinerU ``content_list`` block into Markdown."""

    kind = block.get("type") or ""
    if kind == "text":
        text = _join_text(block.get("text"))
        if not text:
            return ""
        try:
            level = int(block.get("text_level") or 0)
        except (TypeError, ValueError):
            level = 0
        if 1 <= level <= 6:
            return f"{'#' * level} {text}"
        return text
    if kind == "equation":
        return _join_text(block.get("text"))
    if kind == "table":
        parts: list[str] = []
        caption = _join_text(block.get("table_caption"))
        if caption:
            parts.append(f"**{caption}**")
        body = _join_text(block.get("table_body"))
        if body:
            parts.append(body)
        footnote = _join_text(block.get("table_footnote"))
        if footnote:
            parts.append(f"*{footnote}*")
        return "\n\n".join(parts)
    if kind in ("image", "chart"):
        caption = _join_text(block.get(f"{kind}_caption"))
        label = "Figure" if kind == "image" else "Chart"
        return f"[{label}: {caption}]" if caption else f"[{label}]"
    if kind == "code":
        return _join_text(block.get("code_body"))
    if kind in ("ref_text", "aside_text"):
        return _join_text(block.get("text"))
    # Headers, footers and page numbers are layout furniture, not content.
    if kind in ("header", "footer", "page_number"):
        return ""
    return _join_text(block.get("text"))


def _page_markdown(blocks: list[dict]) -> str:
    rendered = [_render_block(block) for block in blocks]
    return "\n\n".join(part for part in rendered if part.strip()).strip()


def _parse_result_zip(content: bytes) -> MineruResult:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise MineruError(
            "MinerU returned an invalid result archive.",
            code="mineru_bad_zip",
        ) from exc

    with archive:
        infos = archive.infolist()
        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
            raise MineruError(
                "MinerU result archive is too large to unpack safely.",
                code="mineru_zip_too_large",
            )
        names = archive.namelist()

        content_name = next(
            (
                name
                for name in names
                if name.endswith("content_list.json")
                and not name.endswith("content_list_v2.json")
            ),
            None,
        )
        middle_name = next((name for name in names if name.endswith("middle.json")), None)
        full_md_name = next((name for name in names if name.endswith("full.md")), None)

        page_count = 0
        if middle_name:
            try:
                middle = json.loads(archive.read(middle_name))
                page_count = len(middle.get("pdf_info") or [])
            except (ValueError, TypeError):
                page_count = 0

        if content_name:
            try:
                blocks = json.loads(archive.read(content_name))
            except ValueError as exc:
                raise MineruError(
                    "MinerU content_list.json is not valid JSON.",
                    code="mineru_bad_response",
                ) from exc
            grouped: dict[int, list[dict]] = {}
            order: list[int] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                try:
                    page_idx = int(block.get("page_idx", 0))
                except (TypeError, ValueError):
                    page_idx = 0
                if page_idx not in grouped:
                    grouped[page_idx] = []
                    order.append(page_idx)
                grouped[page_idx].append(block)
            pages = tuple(
                MineruPage(page=page, markdown=_page_markdown(grouped[page]))
                for page in order
            )
            if page_count <= 0:
                page_count = max(order) + 1 if order else 1
            return MineruResult(pages=pages, page_count=page_count)

        if full_md_name:
            markdown = archive.read(full_md_name).decode("utf-8", errors="replace").strip()
            return MineruResult(
                pages=(MineruPage(page=0, markdown=markdown),),
                page_count=page_count or 1,
            )

        raise MineruError(
            "MinerU result archive has neither content_list.json nor full.md.",
            code="mineru_bad_response",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(path: Path, *, page_ranges: str | None, model: str, language: str,
               enable_formula: bool, enable_table: bool) -> str:
    options = json.dumps(
        {
            "page_ranges": page_ranges or "",
            "model": model,
            "language": language,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
        },
        sort_keys=True,
    ).encode()
    return f"{_sha256_file(path)}-{hashlib.sha256(options).hexdigest()[:16]}"


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "zotero-mcp" / "mineru"


def _cache_read(cache_dir: Path, key: str) -> MineruResult | None:
    path = cache_dir / f"{key}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages = tuple(
            MineruPage(page=int(item["page"]), markdown=str(item["markdown"]))
            for item in payload["pages"]
        )
        return MineruResult(
            pages=pages,
            page_count=int(payload["page_count"]),
            from_cache=True,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _cache_write(cache_dir: Path, key: str, result: MineruResult) -> None:
    try:
        ensure_private_dir(cache_dir)
        write_json_atomic(
            cache_dir / f"{key}.json",
            {
                "page_count": result.page_count,
                "pages": [
                    {"page": page.page, "markdown": page.markdown}
                    for page in result.pages
                ],
            },
        )
    except OSError as exc:  # A cache failure must not fail the parse.
        logger.warning("Could not write MinerU cache entry %s: %s", key, exc)


class MineruClient:
    """Thin synchronous client for MinerU's batch extraction API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_API_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "*/*",
            "User-Agent": USER_AGENT,
        }

    def _json(self, response: requests.Response, action: str) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MineruError(
                f"MinerU {action} returned a non-JSON response (HTTP {response.status_code}).",
                code="mineru_bad_response",
            ) from exc
        if response.status_code >= 400:
            message = payload.get("msg") or payload.get("message") or response.text[:200]
            raise MineruError(
                f"MinerU {action} failed (HTTP {response.status_code}): {message}",
                code="mineru_http_error",
            )
        code = payload.get("code")
        if code not in (None, 0):
            raise _api_error(code, payload.get("msg") or payload.get("message"))
        return payload.get("data") or {}

    def extract_file(
        self,
        file_path: str | Path,
        *,
        page_ranges: str | None = None,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> MineruResult:
        path = Path(file_path)
        if not path.is_file():
            raise MineruError(f"PDF not found: {path}", code="mineru_file_not_found")
        size = path.stat().st_size
        if size <= 0:
            raise MineruError(f"PDF is empty: {path}", code="mineru_file_not_found")
        if size > MAX_UPLOAD_BYTES:
            raise MineruError(
                f"MinerU rejects files larger than 200 MB ({size} bytes).",
                code="mineru_file_too_large",
            )

        file_entry: dict[str, object] = {
            "name": path.name,
            "data_id": hashlib.sha1(f"{path.name}:{size}".encode()).hexdigest()[:24],
        }
        if page_ranges:
            file_entry["page_ranges"] = page_ranges
        body = {
            "files": [file_entry],
            "model_version": model,
            "language": language,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
        }
        response = self._session.post(
            f"{self.base_url}/file-urls/batch",
            headers={**self._auth_headers, "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        data = self._json(response, "submit")
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls") or []
        if not batch_id or not file_urls:
            raise MineruError(
                "MinerU submit response is missing batch_id or file_urls.",
                code="mineru_bad_response",
            )

        with path.open("rb") as handle:
            upload = self._session.put(file_urls[0], data=handle, timeout=self.timeout)
        if upload.status_code >= 400:
            raise MineruError(
                f"MinerU upload failed (HTTP {upload.status_code}): {upload.text[:200]}",
                code="mineru_upload_failed",
            )

        return self._wait_for_result(str(batch_id))

    def _wait_for_result(self, batch_id: str) -> MineruResult:
        deadline = self._clock() + self.timeout
        last_state: str | None = None
        url = f"{self.base_url}/extract-results/batch/{batch_id}"
        while True:
            response = self._session.get(
                url,
                headers=self._auth_headers,
                timeout=min(60, max(1.0, deadline - self._clock())),
            )
            data = self._json(response, "poll")
            results = data.get("extract_result") or []
            if not results:
                raise MineruError(
                    "MinerU poll response has no extract_result entry.",
                    code="mineru_bad_response",
                )
            item = results[0]
            state = str(item.get("state") or "")
            if state != last_state:
                logger.debug("MinerU batch %s state: %s", batch_id, state)
                last_state = state

            if state == "done":
                zip_url = item.get("full_zip_url")
                if not zip_url:
                    raise MineruError(
                        "MinerU finished without a result URL.",
                        code="mineru_bad_response",
                    )
                return self._download_result(str(zip_url))
            if state == "failed":
                raise MineruError(
                    f"MinerU parsing failed: {item.get('err_msg') or 'unknown error'}",
                    code="mineru_parse_failed",
                )
            if self._clock() >= deadline:
                raise MineruError(
                    f"MinerU parsing did not finish within {self.timeout:.0f}s "
                    f"(last state: {state or 'unknown'}).",
                    code="mineru_timeout",
                )
            self._sleep(self.poll_interval)

    def _download_result(self, url: str) -> MineruResult:
        response = self._session.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout,
            stream=True,
        )
        if response.status_code >= 400:
            raise MineruError(
                f"MinerU result download failed (HTTP {response.status_code}).",
                code="mineru_download_failed",
            )
        content = bytearray()
        for chunk in response.iter_content(chunk_size=256 * 1024):
            if not chunk:
                continue
            content.extend(chunk)
            if len(content) > MAX_ZIP_BYTES:
                raise MineruError(
                    "MinerU result archive exceeds 1 GiB.",
                    code="mineru_zip_too_large",
                )
        return _parse_result_zip(bytes(content))


def extract_pdf(
    file_path: str | Path,
    *,
    page_ranges: str | None = None,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    enable_formula: bool = True,
    enable_table: bool = True,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = DEFAULT_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    api_key: str | None = None,
) -> MineruResult:
    """Parse a PDF with MinerU and return per-page Markdown.

    Args:
        file_path: Local PDF to upload.
        page_ranges: 1-based MinerU page ranges, e.g. ``"1-10"`` or
            ``"2,4-6"``. ``None`` parses the whole document.
        model: MinerU model version; ``vlm`` is the layout-aware one.
        language: OCR language code (``ch`` for Chinese/English papers).
        enable_formula / enable_table: Ask MinerU to emit LaTeX and HTML.
        cache_dir: Override the default ``~/.cache/zotero-mcp/mineru``.
        use_cache: Reuse a previous result for the same bytes + options.
        api_key: Override the key from the environment / ``~/.shell-env.sh``.
    """

    path = Path(file_path)
    cache_path = Path(cache_dir) if cache_dir else _default_cache_dir()
    key: str | None = None
    if use_cache:
        key = _cache_key(
            path,
            page_ranges=page_ranges,
            model=model,
            language=language,
            enable_formula=enable_formula,
            enable_table=enable_table,
        )
        cached = _cache_read(cache_path, key)
        if cached is not None:
            logger.debug("MinerU cache hit for %s (%s)", path, key)
            return cached

    client = MineruClient(
        resolve_api_key(api_key),
        base_url=api_base,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    result = client.extract_file(
        path,
        page_ranges=page_ranges,
        model=model,
        language=language,
        enable_formula=enable_formula,
        enable_table=enable_table,
    )
    if use_cache and key:
        _cache_write(cache_path, key, result)
    return result


def format_page_ranges(pages: list[int] | tuple[int, ...]) -> str | None:
    """Convert 0-indexed page numbers to MinerU's 1-based range syntax.

    Consecutive pages collapse: ``[2,3,4,7]`` → ``"3-5,8"``. ``None`` means
    "the whole document" (used when every page is requested).
    """

    if not pages:
        return ""
    ordered = sorted({int(page) for page in pages})
    one_based = [page + 1 for page in ordered]
    parts: list[str] = []
    start = previous = one_based[0]
    for page in one_based[1:]:
        if page == previous + 1:
            previous = page
            continue
        parts.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = page
    parts.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(parts)
