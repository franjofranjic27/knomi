"""Gold evaluation-set loading and validation.

A gold set is a JSON or JSONL file describing questions and, for each, the
source documents that *should* be retrieved. Document-level relevance is used
(rather than exact passages) so the set is practical to author by hand.

Supported formats
-----------------
- **JSONL** — one JSON object per line.
- **JSON**  — either a top-level list of objects, or ``{"queries": [ ... ]}``.

Each query object:
    {
      "id": "q1",                       # optional; defaults to the line index
      "question": "What is self-attention?",
      "relevant_sources": ["Self-Attention_Transformer.pdf", "..."]
    }

``relevant_sources`` entries are matched leniently against a chunk's stored
``source`` path (case-insensitive: exact, basename, or substring) so absolute
paths in the index need not be reproduced in the gold set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """A single gold question with its relevant source documents."""

    id: str
    question: str
    relevant_sources: tuple[str, ...]


def _coerce_query(obj: Any, index: int) -> EvalQuery:
    if not isinstance(obj, dict):
        raise ValueError(f"query #{index}: expected a JSON object, got {type(obj).__name__}")
    question = obj.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"query #{index}: missing non-empty 'question'")
    raw_sources = obj.get("relevant_sources", [])
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"query #{index}: 'relevant_sources' must be a non-empty list")
    sources = tuple(str(s) for s in raw_sources)
    query_id = str(obj.get("id", f"q{index + 1}"))
    return EvalQuery(id=query_id, question=question.strip(), relevant_sources=sources)


def _parse_document(text: str) -> list[Any]:
    """Parse the file body as JSON or JSONL, returning the raw query objects.

    Tries to read the whole file as a single JSON document first (array,
    ``{"queries": [...]}`` container, or a lone query object). If that fails
    because the file holds several JSON values (multi-line JSONL), each
    non-empty line is parsed on its own.
    """
    stripped = text.strip()
    if not stripped:
        return []
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        # Multiple JSON values on separate lines -> JSONL.
        return [json.loads(line) for line in stripped.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # A container with an explicit "queries" array, else a single query.
        queries = data.get("queries")
        return queries if isinstance(queries, list) else [data]
    raise ValueError("top-level JSON must be an array, an object, or JSONL lines")


def load_eval_set(path: Path) -> list[EvalQuery]:
    """Load and validate a gold set from *path* (JSON or JSONL).

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError:        on malformed content or an empty set.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Eval set not found: {path}")
    raw = _parse_document(path.read_text(encoding="utf-8"))
    queries = [_coerce_query(obj, i) for i, obj in enumerate(raw)]
    if not queries:
        raise ValueError(f"{path}: no queries found")
    ids = [q.id for q in queries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate query ids")
    return queries


def source_matches(source: str, relevant: str) -> bool:
    """Return True if a retrieved *source* path satisfies a *relevant* entry.

    Lenient, case-insensitive matching: exact path, matching basename, or
    substring — so gold sets can reference just a file name.
    """
    src = source.lower()
    rel = relevant.lower()
    if not src or not rel:
        return False
    return rel == src or Path(src).name == Path(rel).name or rel in src
