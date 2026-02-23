"""Unit tests for knomi.ingest.parser."""

from pathlib import Path

from knomi.ingest.parser import _normalise, parse


def test_normalise_collapses_whitespace() -> None:
    assert _normalise("hello   world\t\nfoo") == "hello world foo"


def test_normalise_dehyphenates() -> None:
    assert _normalise("hyphen-\nated") == "hyphenated"


def test_normalise_strips_leading_trailing() -> None:
    assert _normalise("  hello  ") == "hello"


def test_parse_text_utf8(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("Hello world.", encoding="utf-8")
    result = parse(f)
    assert result == "Hello world."


def test_parse_text_md(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Title\nSome content.", encoding="utf-8")
    result = parse(f)
    assert "Title" in result
    assert "Some content" in result


def test_parse_returns_empty_for_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "data.xyz"
    f.write_text("irrelevant")
    assert parse(f) == ""


def test_parse_text_latin1_fallback(tmp_path: Path) -> None:
    f = tmp_path / "latin.txt"
    f.write_bytes("caf\xe9".encode("latin-1"))
    result = parse(f)
    assert "caf" in result
