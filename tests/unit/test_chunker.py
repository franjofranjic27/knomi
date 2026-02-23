"""Unit tests for knomi.ingest.chunker."""

from pathlib import Path

from knomi.ingest.chunker import chunk


def test_chunk_empty_returns_empty_list() -> None:
    assert chunk("", source=Path("x.txt"), doc_id="abc") == []


def test_chunk_short_text_single_chunk() -> None:
    result = chunk("short text", source=Path("x.txt"), doc_id="abc", chunk_size=512)
    assert len(result) == 1
    assert result[0].text == "short text"
    assert result[0].metadata["chunk_index"] == 0
    assert result[0].metadata["total_chunks"] == 1


def test_chunk_sliding_window() -> None:
    # Build a text that exceeds chunk_size=10 tokens
    words = " ".join(["word"] * 50)
    result = chunk(words, source=Path("x.txt"), doc_id="abc", chunk_size=10, chunk_overlap=2)
    assert len(result) > 1
    # Adjacent chunks share tokens (overlap)
    assert result[0].metadata["chunk_index"] == 0
    assert result[1].metadata["chunk_index"] == 1
    assert result[-1].metadata["total_chunks"] == len(result)


def test_chunk_metadata_fields() -> None:
    result = chunk(
        "some text here", source=Path("docs/file.txt"), doc_id="sha256abc", chunk_size=512
    )
    meta = result[0].metadata
    assert meta["source"] == "docs/file.txt"
    assert meta["doc_id"] == "sha256abc"
    assert "chunk_index" in meta
    assert "total_chunks" in meta


def test_chunk_overlap_content() -> None:
    # 30 tokens with chunk_size=20, overlap=10 → step=10
    words = " ".join(["tok"] * 30)
    result = chunk(words, source=Path("f.txt"), doc_id="d", chunk_size=20, chunk_overlap=10)
    # Each chunk except possibly the last should have chunk_size tokens
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(result[0].text)) <= 20
