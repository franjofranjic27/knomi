"""Filesystem scanner.

Responsibilities
----------------
- Recursively walk a root directory.
- Filter files by a configurable extension whitelist.
- Compute a SHA-256 hash for each file (used for deduplication).
- Yield ``ScannedFile`` objects consumed by the pipeline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Extensions supported by the parser layer.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".txt", ".docx", ".html"})


@dataclass(frozen=True, slots=True)
class ScannedFile:
    """A discovered file with its SHA-256 digest."""

    path: Path
    sha256: str
    size_bytes: int


def scan(root: Path, extensions: frozenset[str] = SUPPORTED_EXTENSIONS) -> Iterator[ScannedFile]:
    """Yield all matching files under *root*.

    Args:
        root: Directory to walk recursively.
        extensions: File extensions to include (with leading dot, lowercase).

    Yields:
        ScannedFile for each matching file.
    """
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in extensions:
            yield ScannedFile(path=path, sha256=_sha256(path), size_bytes=path.stat().st_size)


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, streaming in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
