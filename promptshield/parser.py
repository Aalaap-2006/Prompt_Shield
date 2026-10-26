"""PDF parsing + boundary-aware chunking + data-isolation wrapping (Person 1).

Design goals
- Preserve structure: chunks never cut through a PDF text block; headings start new chunks.
- Surface hidden-text attacks: tiny fonts, white text, off-page text are flagged per chunk
  and isolated in their own chunks so they cannot hide inside a benign paragraph.
- Sanitize: NFKC normalisation, zero-width/control char removal, and neutralisation of
  anything that looks like our isolation tags (tag-spoofing).
- Isolate: every chunk is wrapped in a per-request, random-nonce tag so document text
  cannot forge a closing tag and "escape" into the instruction channel.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import statistics
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

import pymupdf as fitz  # PyMuPDF (>=1.24.3)

MIN_VISIBLE_FONT_PT = 2.0
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAGLIKE = re.compile(r"</?\s*untrusted[^>]*>", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int                      # 1-indexed
    source: str
    kind: str = "paragraph"        # "heading_section" | "paragraph" | "hidden"
    hidden_reasons: list[str] = field(default_factory=list)

    @property
    def hidden(self) -> bool:
        return bool(self.hidden_reasons)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = self.hidden
        return d


@dataclass
class _Block:
    text: str
    page: int
    size: float
    is_heading: bool
    hidden_reasons: list[str]


def new_nonce() -> str:
    return secrets.token_hex(6)


def sanitize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = _CONTROL.sub("", text)
    text = _TAGLIKE.sub("[removed-tag]", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def wrap_untrusted(chunk: Chunk, nonce: str) -> str:
    """Wrap chunk text in nonce-tagged data-isolation delimiters."""
    return (
        f'<untrusted_document nonce="{nonce}" id="{chunk.chunk_id}" '
        f'source="{chunk.source}" page="{chunk.page}">\n{chunk.text}\n</untrusted_document>'
    )


def wrap_untrusted_text(text: str, nonce: str, label: str = "tool_output") -> str:
    return f'<untrusted_document nonce="{nonce}" id="{label}">\n{sanitize_text(text)}\n</untrusted_document>'


def _span_hidden_reason(span: dict, page_rect: "fitz.Rect") -> str | None:
    if span["size"] < MIN_VISIBLE_FONT_PT:
        return "tiny_font"
    if span["color"] == 0xFFFFFF:  # assumes white page background (conservative flag)
        return "white_text"
    if not fitz.Rect(span["bbox"]).intersects(page_rect):
        return "off_page"
    return None


def _extract_blocks(doc: "fitz.Document") -> list[_Block]:
    raw: list[_Block] = []
    for pno, page in enumerate(doc, start=1):
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            lines, sizes, bold, hidden = [], [], False, set()
            for line in b.get("lines", []):
                seg = ""
                for s in line.get("spans", []):
                    t = s["text"]
                    if t.strip():
                        reason = _span_hidden_reason(s, page.rect)
                        if reason:
                            hidden.add(reason)
                        sizes.append(s["size"])
                        bold = bold or bool(s["flags"] & 16)
                    seg += t
                if seg.strip():
                    lines.append(seg.strip())
            text = sanitize_text(" ".join(lines))
            if text:
                raw.append(_Block(text, pno, statistics.mean(sizes) if sizes else 0.0,
                                  bold, sorted(hidden)))
    if not raw:
        return raw
    median = statistics.median(b.size for b in raw if b.size) or 1.0
    for b in raw:
        b.is_heading = (len(b.text) < 120 and (b.size >= 1.2 * median or b.is_heading)
                        and not b.text.endswith((".", ",", ";")))
    return raw


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        while len(sent) > max_chars:  # pathological no-punctuation text
            out.append(sent[:max_chars]); sent = sent[max_chars:]
        if len(cur) + len(sent) + 1 > max_chars and cur:
            out.append(cur); cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def parse_pdf(
    pdf: Union[str, Path, bytes],
    *,
    source: str | None = None,
    max_chars: int = 1200,
    drop_hidden: bool = False,
) -> list[Chunk]:
    """Parse a PDF into structure-aware chunks. `pdf` may be a path or raw bytes."""
    if isinstance(pdf, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(pdf), filetype="pdf")
        doc_id = hashlib.sha256(bytes(pdf)).hexdigest()[:8]
        source = source or "upload.pdf"
    else:
        doc = fitz.open(str(pdf))
        doc_id = hashlib.sha256(Path(pdf).read_bytes()).hexdigest()[:8]
        source = source or Path(pdf).name
    try:
        blocks = _extract_blocks(doc)
    finally:
        doc.close()

    chunks: list[Chunk] = []

    def emit(text: str, page: int, kind: str, reasons: list[str]) -> None:
        chunks.append(Chunk(f"{doc_id}-{len(chunks):04d}", text, page, source, kind, reasons))

    buf: list[str] = []
    buf_page = 1

    def flush() -> None:
        nonlocal buf
        if buf:
            for piece in _split_long("\n".join(buf), max_chars):
                emit(piece, buf_page, "paragraph", [])
        buf = []

    for b in blocks:
        if b.hidden_reasons:                      # isolate hidden text
            flush()
            if not drop_hidden:
                for piece in _split_long(b.text, max_chars):
                    emit(piece, b.page, "hidden", b.hidden_reasons)
            continue
        if b.is_heading:
            flush(); buf_page = b.page; buf = [b.text]; continue
        if not buf:
            buf_page = b.page
        if sum(len(x) for x in buf) + len(b.text) > max_chars:
            flush(); buf_page = b.page
        buf.append(b.text)
    flush()
    return chunks
