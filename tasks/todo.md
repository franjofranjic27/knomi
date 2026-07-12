# knomi — TODO / Backlog

Stand: 2026-07-12 · Branch `feat/configurable-profiles-multi-backend` · PR [#3](https://github.com/franjofranjic27/knomi/pull/3)

## Nächster Schritt — Embedding-Qualität (offen, Entscheidung ausstehend)
Der bge-Benchmark ergab: **bge-base < MiniLM *wie integriert*** — vermutlich weil knomi das für `bge-*-v1.5` nötige Query-Präfix nicht setzt. Optionen:

- [ ] **(a) Query-Instruction-Support** in `LocalEmbedder` (`knomi/ingest/embedder.py`): pro-Modell-Präfix für asymmetrische Embedder (bge/e5), z. B. `"Represent this sentence for searching relevant passages: "` nur für die Query-Seite (`embed_query`), nicht für Passagen. Danach fairer Re-Benchmark bge vs. MiniLM.
- [ ] **(b) API-Embedding-Benchmark**: `text-embedding-3-large` (3072 Dim) gegen MiniLM messen (schnell, da API; braucht `OPENAI_API_KEY`). Auf dieser Hardware praktikabler als lokales bge.
- [ ] **(c)** Embedding vorerst lassen, anderen Hebel ziehen (siehe Roadmap).

## RAG-Qualitäts-Roadmap (nach Hebel priorisiert)
- [x] **Eval-Harness** (`knomi eval`, Recall@k/MRR/nDCG/Hit) — erledigt
- [x] **Reranking** (Cross-Encoder, opt-in) — erledigt; **gemessen: lohnt auf diesem Korpus nicht** (ms-marco ↓, bge-reranker ≈/leicht ↓). Default `enabled: false` beibehalten.
- [ ] **Hybrid Retrieval** (BM25/sparse + dense, RRF-Fusion) — groß bei fachlich/keyword-lastigen Texten
- [ ] **Metadaten & Zitate**: Seitenzahl/Abschnitt/Titel im Payload speichern → Filter + belastbare Quellenangaben (Parser `knomi/ingest/parser.py` liefert aktuell nur Text; PyMuPDF kann Seiten/Struktur)
- [ ] **Query-Transformation** (HyDE, Multi-Query) — robuster bei vagen Fragen
- [ ] **Contextual Retrieval** (Chunk-Kontext voranstellen, Anthropic-Ansatz)
- [ ] **Per-Datei-Chunking-Routing**: `build_chunker_for(file, config)` statt global — z. B. Paper→sentence, Folien/MD→structure (Verdrahtung in `knomi/ingest/pipeline.py`)
- [ ] **Gold-Set ausbauen** auf 50+ Fragen für belastbarere Statistik (aktuell 31 in `eval.example.jsonl`)
- [ ] **Generation-Eval** (Faithfulness/Answer-Relevance, LLM-as-Judge / RAGAS) — bislang nur Retrieval-Eval

## Security-Härtung (aus Security-Review, nicht blockierend)
- [ ] Secrets als pydantic `SecretStr` statt `str` (`store.api_key`, `store.dsn`, `embedding.api_key`, `reranking.api_key`) — maskiert in Logs/Tracebacks
- [ ] CI-Actions auf Commit-SHA pinnen (`.github/workflows/release-knomi.yaml`), v. a. `dawidd6/action-homebrew-bump-formula`
- [ ] Top-Level `permissions: contents: read` im Release-Workflow (Least Privilege für npm/brew-Jobs)
- [ ] Dockerfile: non-root `USER`, `uv`/Base-Image pinnen (aktuell `uv:latest`)
- [ ] npm-`postinstall`: PyPI-Version an die Wrapper-Version pinnen

## Release / Distribution (manuell durch Nutzer)
- [ ] Repo-Secret `NPM_TOKEN` (npm Automation-Token) anlegen; npm-Namensverfügbarkeit von `knomi` prüfen
- [ ] Tap-Repo `franjofranjic27/homebrew-knomi` anlegen, `Formula/knomi.rb` hineinkopieren, `brew update-python-resources` + `sha256` einmalig setzen
- [ ] Repo-Secret `HOMEBREW_TAP_TOKEN` (fine-grained PAT, `contents:write` auf Tap)
- [ ] Erstes echtes Release `v0.1.0` taggen (Voraussetzung für npm/brew-Artefakte)

## Aufräumen / Sonstiges
- [ ] Lokale Experiment-Collections in `./.knomi/chroma` aufräumen: `uni-sg-bgebase` (Teil-Ingest, 8007 Vektoren, hing an Goodfellow), `ml-lecture-bge`, ggf. `uni-sg-bge`
- [ ] Bekannt: voller lokaler bge-Ingest des Korpus ist auf dieser Hardware zu langsam/hängt an sehr großen PDFs → für starke Embeddings API-Modell oder GPU nutzen
- [ ] Integration-Tests (`tests/integration/`) brauchen Docker (Qdrant/Postgres) — im CI abgedeckt, lokal nicht ausgeführt

## Review / Merge
- [ ] PR #3 reviewen und nach `main` mergen (7 Commits: Profile/Multi-Backend, Store-Target-Fix, Special-Token-Fix, Eval-Harness, Reranking, Gold-Set, Eval-Recall-Fix)
