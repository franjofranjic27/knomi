"""Unit tests for gold-set loading, validation, and source matching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knomi.eval.dataset import load_eval_set, source_matches

_ROWS = [
    {"id": "a", "question": "Q one?", "relevant_sources": ["one.pdf"]},
    {"question": "Q two?", "relevant_sources": ["two.pdf", "alt.pdf"]},
]


def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _ROWS), encoding="utf-8")
    queries = load_eval_set(path)
    assert [q.id for q in queries] == ["a", "q2"]  # second id auto-assigned
    assert queries[1].relevant_sources == ("two.pdf", "alt.pdf")


def test_load_json_array(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(_ROWS), encoding="utf-8")
    assert len(load_eval_set(path)) == 2


def test_load_json_container(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"queries": _ROWS}), encoding="utf-8")
    assert len(load_eval_set(path)) == 2


def test_load_single_object(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(_ROWS[0]), encoding="utf-8")
    queries = load_eval_set(path)
    assert len(queries) == 1 and queries[0].question == "Q one?"


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "nope.jsonl")


def test_empty_set_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no queries"):
        load_eval_set(path)


def test_missing_question_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps({"relevant_sources": ["x.pdf"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_eval_set(path)


def test_empty_relevant_sources_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps({"question": "Q?", "relevant_sources": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="relevant_sources"):
        load_eval_set(path)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    rows = [
        {"id": "dup", "question": "a", "relevant_sources": ["x"]},
        {"id": "dup", "question": "b", "relevant_sources": ["y"]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_set(path)


def test_source_matches_lenient() -> None:
    full = "/Users/me/OneDrive/papers/Self-Attention_Transformer.pdf"
    assert source_matches(full, "Self-Attention_Transformer.pdf")  # basename
    assert source_matches(full, "self-attention_transformer.pdf")  # case-insensitive
    assert source_matches(full, "Self-Attention")  # substring
    assert source_matches(full, full)  # exact
    assert not source_matches(full, "ConvNets.pdf")
    assert not source_matches("", "x")
