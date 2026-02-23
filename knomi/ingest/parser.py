"""Document parser — raw file bytes → normalised plain text.

Responsibilities
----------------
- Dispatch to the correct extractor based on file extension.
- Normalise output: collapse whitespace, de-hyphenate line breaks,
  strip page headers/footers.
- Return an empty string (not raise) for unreadable or empty files.

Not yet implemented — stubs only.
"""

from __future__ import annotations

from pathlib import Path


def parse(path: Path) -> str:
    """Extract plain text from *path*.

    Routes to the appropriate format-specific extractor.

    Args:
        path: Path to the document.

    Returns:
        Normalised plain text, or empty string if extraction fails.

    TODO:
        - Add per-format extractors below.
        - Log a warning (not raise) on extraction failure.
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
    """Collapse whitespace and de-hyphenate line-broken words.

    TODO: Implement regex-based normalisation pipeline.
    """
    raise NotImplementedError


def _parse_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz).

    TODO:
        - Use ``fitz.open(path)`` and iterate pages.
        - Concatenate ``page.get_text()`` for each page.
        - Pass through ``_normalise()``.
        - Handle scanned PDFs with no embedded text layer gracefully.
    """
    raise NotImplementedError


def _parse_text(path: Path) -> str:
    """Read a plain-text file (txt, md).

    TODO:
        - Read with UTF-8, fall back to latin-1.
        - Pass through ``_normalise()``.
    """
    raise NotImplementedError
