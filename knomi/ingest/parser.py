"""Document parser — raw file bytes → normalised plain text.

Responsibilities
----------------
- Dispatch to the correct extractor based on file extension.
- Normalise output: collapse whitespace, de-hyphenate line breaks,
  strip page headers/footers.
- Return an empty string (not raise) for unreadable or empty files.
"""

from __future__ import annotations

import re
from pathlib import Path


def parse(path: Path) -> str:
    """Extract plain text from *path*.

    Routes to the appropriate format-specific extractor.

    Args:
        path: Path to the document.

    Returns:
        Normalised plain text, or empty string if extraction fails.
    """
    suffix = path.suffix.lower()
    extractors = {
        ".pdf": _parse_pdf,
        ".md": _parse_text,
        ".txt": _parse_text,
        # ".docx": _parse_docx,
        # ".html": _parse_html,
    }
    extractor = extractors.get(suffix)
    if extractor is None:
        return ""
    return extractor(path)


def _normalise(text: str) -> str:
    """Collapse whitespace and de-hyphenate line-broken words."""
    text = re.sub(r"-\n", "", text)  # de-hyphenate line breaks
    text = re.sub(r"\s+", " ", text)  # collapse all whitespace
    return text.strip()


def _parse_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    import fitz  # pymupdf

    pages: list[str] = []
    try:
        with fitz.open(path) as doc:
            for page in doc:
                pages.append(page.get_text())
    except Exception:
        return ""
    return _normalise("\n".join(pages))


def _parse_text(path: Path) -> str:
    """Read a plain-text file (txt, md), falling back to latin-1 encoding."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return _normalise(text)
