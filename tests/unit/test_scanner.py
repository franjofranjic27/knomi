"""Unit tests for knomi.ingest.scanner."""

from pathlib import Path

from knomi.ingest.scanner import ScannedFile, _sha256, scan


def test_scan_finds_supported_extensions(tmp_docs_dir: Path) -> None:
    results = list(scan(tmp_docs_dir))
    paths = {r.path for r in results}
    assert tmp_docs_dir / "sample.txt" in paths
    assert tmp_docs_dir / "subdir" / "nested.txt" in paths
    assert all(isinstance(r, ScannedFile) for r in results)


def test_scan_ignores_unsupported_extensions(tmp_docs_dir: Path) -> None:
    (tmp_docs_dir / "ignored.xyz").write_text("should be ignored")
    results = list(scan(tmp_docs_dir))
    paths = {r.path for r in results}
    assert tmp_docs_dir / "ignored.xyz" not in paths


def test_scan_respects_custom_extensions(tmp_docs_dir: Path) -> None:
    (tmp_docs_dir / "only.md").write_text("# heading")
    results = list(scan(tmp_docs_dir, extensions=frozenset({".md"})))
    assert all(r.path.suffix == ".md" for r in results)


def test_scan_skips_symlinks(tmp_docs_dir: Path) -> None:
    target = tmp_docs_dir / "sample.txt"
    link = tmp_docs_dir / "link.txt"
    link.symlink_to(target)
    results = list(scan(tmp_docs_dir))
    paths = {r.path for r in results}
    assert link not in paths


def test_sha256_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("deterministic content")
    assert _sha256(f) == _sha256(f)


def test_sha256_differs_for_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("content a")
    b.write_text("content b")
    assert _sha256(a) != _sha256(b)


def test_scanned_file_has_correct_size(tmp_path: Path) -> None:
    f = tmp_path / "sized.txt"
    f.write_bytes(b"x" * 100)
    results = list(scan(tmp_path, extensions=frozenset({".txt"})))
    assert results[0].size_bytes == 100
