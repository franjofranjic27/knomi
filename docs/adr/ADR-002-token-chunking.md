# ADR-002: Token-based chunking with tiktoken

**Status:** Accepted
**Date:** 2025-01

## Context

Documents must be split into windows before embedding. The window size must fit within the embedding model's context limit. The choice of splitting unit directly affects embedding quality and token budget accuracy.

## Options considered

| Strategy | Unit | Accuracy | Speed | Notes |
|----------|------|----------|-------|-------|
| **Token-based (tiktoken)** | tokens | exact | moderate | Matches model context precisely |
| Character-based | chars | approximate | fast | Over/under-estimates depending on language |
| Sentence-based (NLTK/spaCy) | sentences | semantic | slow | Requires NLP pipeline, uneven chunk sizes |
| Recursive character split | chars/separators | approximate | fast | Used by LangChain; still char-based |

## Decision

**Token-based splitting with tiktoken** using the `cl100k_base` encoding (used by all OpenAI `text-embedding-3-*` models and compatible with most modern embedders).

Key reasons:
1. `chunk_size = 512` means **exactly 512 tokens** — no silent truncation by the embedding model.
2. tiktoken is already a dependency of the OpenAI SDK; no additional install cost.
3. `cl100k_base` is a reasonable approximation for non-OpenAI models too — close enough that using a different tokeniser per model is not worth the complexity.

## Sliding-window algorithm

```
tokens = encode(text)
step   = chunk_size - chunk_overlap      # advance per iteration

windows: for i in range(0, len(tokens), step):
    window = tokens[i : i + chunk_size]
```

Adjacent windows share `chunk_overlap` tokens, giving the embedding model continuity across boundaries (e.g. a sentence that straddles two chunks appears in both).

## Consequences

- `chunk_size` and `chunk_overlap` config values are always in **tokens**, not characters.
- Switching the embedding model to one with a different tokeniser (e.g. a local model using wordpiece) means the chunk sizes will be slightly inaccurate but still within acceptable range.
- If exact tokeniser matching is required, pass the model's encoding name as `encoding_name` in `chunk()`.
