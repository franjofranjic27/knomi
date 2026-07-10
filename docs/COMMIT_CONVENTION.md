# Commit Convention

knomi uses **Conventional Commits** (`conventionalcommits.org`).

## Format

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

- **Subject line**: ≤ 72 characters, imperative mood, no trailing period.
- **Body**: Explain *why*, not *what*. Wrap at 72 characters.
- **Footer**: `Closes #<issue>`, `BREAKING CHANGE: <description>`.

## Types

| Type | When to use |
|------|-------------|
| `feat` | New user-facing feature |
| `fix` | Bug fix |
| `refactor` | Code change that is neither a feature nor a fix |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Tooling, dependency updates, config changes |
| `docs` | Documentation only |
| `ci` | CI/CD workflow changes |
| `build` | Build system changes (pyproject.toml, compose.yml) |

## Scopes (optional but encouraged)

`ingest`, `parser`, `chunker`, `embedder`, `store`, `cli`, `config`, `ci`, `docs`

## Examples

```
feat(chunker): add token-based overlap window

Replaces character-count splitting with tiktoken-based windows so
chunk_size is always expressed in model tokens rather than bytes.

Closes #12
```

```
fix(parser): handle PDFs with no extractable text

Returns empty string instead of raising ValueError for scanned PDFs
without an embedded text layer.
```

```
chore(deps): bump qdrant-client to 1.9.0
```

## Breaking changes

Add `BREAKING CHANGE:` in the footer or append `!` to the type:

```
feat(store)!: rename upsert() to index()

BREAKING CHANGE: VectorStore.upsert() is now VectorStore.index().
Update all store implementations and callers.
```

## Branch naming

```
feat/<short-description>
fix/<short-description>
chore/<short-description>
```

Examples: `feat/token-chunker`, `fix/pdf-parser-empty`, `chore/update-qdrant`.
