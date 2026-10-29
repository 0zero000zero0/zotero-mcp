"""MinerU parsing: zip rendering, client flow, page alignment and config."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from zotero_mcp import extract, mineru


def _result_zip(blocks: list[dict], *, middle_pages: int | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        prefix = "abc"
        archive.writestr(f"{prefix}_content_list.json", json.dumps(blocks))
        if middle_pages is not None:
            archive.writestr(
                f"{prefix}_middle.json",
                json.dumps({"pdf_info": [{} for _ in range(middle_pages)]}),
            )
        archive.writestr("full.md", "# Full document")
    return buffer.getvalue()


def test_parse_result_zip_preserves_latex_and_page_order():
    content = _result_zip(
        [
            {"type": "text", "text": "Introduction", "text_level": 1, "page_idx": 0},
            {
                "type": "equation",
                "text": "$$\nR(s)=\\sum_i e^{s_i}\\tag{3}\n$$",
                "page_idx": 1,
            },
            {
                "type": "table",
                "table_caption": ["Table 1: Results"],
                "table_body": "<table><tr><td>1</td></tr></table>",
                "page_idx": 1,
            },
        ],
        middle_pages=2,
    )

    result = mineru._parse_result_zip(content)

    assert result.page_count == 2
    assert [page.page for page in result.pages] == [0, 1]
    assert "# Introduction" in result.pages[0].markdown
    assert "\\sum" in result.pages[1].markdown
    assert "<table>" in result.pages[1].markdown
    assert result.pages[1].markdown.index("\\sum") < result.pages[1].markdown.index("<table>")


class _FakeResponse:
    def __init__(self, *, payload=None, status_code=200, content=b""):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if payload is not None else ""
        self._content = content

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def iter_content(self, chunk_size=256 * 1024):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start:start + chunk_size]


class _FakeSession:
    def __init__(self, *, poll_payload, zip_bytes):
        self.poll_payload = poll_payload
        self.zip_bytes = zip_bytes
        self.posts: list[tuple[str, dict]] = []
        self.puts: list[tuple[str, object]] = []
        self.gets: list[str] = []

    def post(self, url, *, headers, json, timeout):
        self.posts.append((url, json))
        return _FakeResponse(
            payload={
                "code": 0,
                "data": {"batch_id": "batch-1", "file_urls": ["https://upload.test/x"]},
            }
        )

    def put(self, url, *, data, timeout):
        self.puts.append((url, data))
        return _FakeResponse(payload={}, status_code=200)

    def get(self, url, *, headers=None, timeout=None, stream=False):
        self.gets.append(url)
        if "extract-results" in url:
            return _FakeResponse(payload=self.poll_payload)
        return _FakeResponse(content=self.zip_bytes)


def test_client_submits_page_ranges_at_file_level(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    zip_bytes = _result_zip(
        [{"type": "text", "text": "Body", "page_idx": 0}],
        middle_pages=1,
    )
    session = _FakeSession(
        poll_payload={
            "code": 0,
            "data": {
                "extract_result": [
                    {
                        "state": "done",
                        "full_zip_url": "https://result.test/x.zip",
                        "err_msg": "",
                    }
                ]
            },
        },
        zip_bytes=zip_bytes,
    )
    client = mineru.MineruClient(
        "test-key", session=session, sleep=lambda _: None, clock=lambda: 0.0,
    )

    result = client.extract_file(pdf, page_ranges="3-4")

    assert result.pages[0].markdown == "Body"
    body = session.posts[0][1]
    assert body["model_version"] == "vlm"
    assert body["language"] == "ch"
    assert body["enable_formula"] is True
    assert body["enable_table"] is True
    assert body["files"][0]["page_ranges"] == "3-4"
    assert "page_ranges" not in body


def test_extract_pdf_maps_relative_page_indices_to_source_pages(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(extract, "pdf_page_count", lambda _: 9)
    monkeypatch.setattr(
        mineru,
        "extract_pdf",
        lambda *args, **kwargs: mineru.MineruResult(
            pages=(
                mineru.MineruPage(page=0, markdown="page three"),
                mineru.MineruPage(page=1, markdown="page four"),
            ),
            page_count=2,
        ),
    )

    doc = extract.extract_pdf(pdf, pages=[2, 3], engine="mineru")

    assert doc.page_numbers == (2, 3)
    assert doc.pages == ("page three", "page four")
    assert doc.page_count == 9
    assert not doc.truncated


def test_extract_file_does_not_swallow_mineru_errors(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    def boom(*args, **kwargs):
        raise mineru.MineruError("missing key", code="mineru_missing_key")

    monkeypatch.setattr(extract, "extract_pdf", boom)

    with pytest.raises(mineru.MineruError, match="missing key"):
        extract.extract_file(pdf)


def test_resolve_api_key_reads_shell_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".shell-env.sh").write_text(
        "# secrets\nexport mineru_api_key='abc123'\n", encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("MINERU_API_KEY", raising=False)
    monkeypatch.delenv("mineru_api_key", raising=False)

    assert mineru.resolve_api_key() == "abc123"


def test_format_page_ranges_collapses_consecutive_pages():
    assert mineru.format_page_ranges([2, 3, 4, 7]) == "3-5,8"
    assert mineru.format_page_ranges([0]) == "1"
    assert mineru.format_page_ranges([]) == ""
