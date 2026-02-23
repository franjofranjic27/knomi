"""Filesystem scanner.

Responsibilities
----------------
- Recursively walk a root directory.
- Filter files by a configurable extension whitelist.
- Compute a SHA-256 hash for each file (used for deduplication).
- Yield ``ScannedFile`` objects consumed by the pipeline.

Not yet implemented — stubs only.
"""

from __future__ import annotations

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

    TODO:
        - Skip symlink loops.
        - Respect a ``.knomiignore`` file (gitignore-style).
        - Emit progress events for large trees.
    """
    raise NotImplementedError


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file.

    TODO: Stream in chunks to support large files without loading into memory.
    """
    raise NotImplementedError
