# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for financial advisors. Ingest client interaction notes from a CSV, embed them into a local vector database, and query them conversationally.

All inference runs locally via [Ollama](https://ollama.com); no cloud API keys required.

## How It Works

1. Reads 1,000 client records from `MOCK_DATA.csv`
2. Embeds each record using `mxbai-embed-large` via Ollama
3. Stores vectors in a persistent ChromaDB database
4. Accepts natural-language questions and retrieves the top-20 semantically similar records
5. Feeds retrieved context into `llama3.1:8b` to synthesize a grounded response

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) installed and running locally
- [uv](https://github.com/astral-sh/uv) for dependency management

## Setup

### 1. Install Ollama and pull models

```bash
# Install Ollama: https://ollama.com/download
ollama pull mxbai-embed-large
ollama pull llama3.1:8b
```

### 2. Create a virtual environment and install dependencies

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

### 3. Start Ollama

```bash
ollama serve
```

## Running

```bash
uv run python main.py
```

On first run, the script embeds all 1,000 client records and writes the vector index to `./chroma_db_mxbai/`. Subsequent runs reuse the persisted database (~15 seconds faster).

**Example output:**

```
Configuring models...
✓ Models configured successfully
Loading documents from MOCK_DATA.csv...
✓ Loaded 1000 documents
Setting up vector database...
Building vector index (this may take a moment)...
✓ Vector database setup complete

==================================================
RAG Pipeline Ready!
==================================================

Q: Which client discussed portfolio allocation, risk tolerance, FX hedging, and needed compliance involved?
A: Netty Toffts (Kayveo) and Ilyssa Tipple (Talane) both discussed portfolio allocation,
   risk tolerance, and FX hedging, and required compliance involvement.

Q: Which Realcube clients mentioned sector concentration?
A: Lewie Sevier, Trude Tremmel, and Ranice Poker from Realcube all mentioned sector concentration.

Q: Which contacts were marked left voicemail will retry and had no real conversation note?
A: The following contacts were marked "Left voicemail; will retry" with no further notes:
   Muriel Masham, Latia Tink, Coop Marton, Zorine Mercer...

Q: Which client has the latest follow-up due date, and what is the follow-up?
A: The system cannot reliably answer date-ordering questions — semantic search ranks
   by meaning, not by date value.
```

## Evaluation

The pipeline was evaluated against a manually curated gold set of 25 questions (`gold_set.csv`), covering four question types:

| Question Type | Example | Avg Recall@20 |
|---|---|---|
| Single-client lookup | "Which client discussed FX hedging and compliance?" | 100% |
| Edge cases (no content) | "Which contacts left voicemail with no follow-up?" | 95% |
| Multi-client filter | "Which clients watching inflation got an ESG overview?" | 60% |
| Date / temporal ordering | "Which client has the earliest follow-up date?" | 0% |

**Overall average Recall@20: 65.8%**

To run the evaluation yourself:

```bash
# Retrieval scoring only (fast, no LLM calls)
uv run python eval.py --retrieval-only

# Full eval including LLM answer generation
uv run python eval.py

# Compare experiment configurations
uv run python run_experiments.py --suite chunk --retrieval-only
```

Results are saved to `eval_results/` as timestamped CSVs.

## Configuration

All settings are at the top of [`main.py`](main.py):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `mxbai-embed-large` | Ollama embedding model |
| `LLM_MODEL` | `llama3.1:8b` | Ollama LLM for answer generation |
| `DATA_DIR` | `MOCK_DATA.csv` | Path to client data CSV |
| `DB_PATH` | `./chroma_db_mxbai/` | ChromaDB persistence directory |
| `CHUNK_SIZE` | `256` | Token chunk size for text splitting |
| `CHUNK_OVERLAP` | `32` | Token overlap between chunks |

To change the demo questions, edit the `questions` list in `main.py`:

```python
questions = [
    "Which clients are concerned about tax efficiency?",
    "Who has a follow-up scheduled for next quarter?",
]
```

## Data Format

`MOCK_DATA.csv` expects these columns:

| Column | Description |
|---|---|
| `id` | Unique record ID |
| `first_name` | Client first name |
| `last_name` | Client last name |
| `company` | Client company |
| `email` | Client email |
| `phone` | Client phone |
| `contact_notes` | Free-text interaction notes (primary field for RAG) |

## Project Structure

```
RAG-Pipeline/
├── main.py              # Simple pipeline entry point and demo
├── pipeline.py          # Parameterized pipeline for experiments
├── eval.py              # Evaluation harness (Recall@k, Precision@k)
├── run_experiments.py   # Comparative experiment runner
├── gold_set.csv         # 25 manually curated evaluation questions
├── requirements.txt     # Python dependencies
├── MOCK_DATA.csv        # Synthetic dataset (1,000 client records)
├── chroma_db/           # ChromaDB index (default embeddings)
└── chroma_db_mxbai/     # ChromaDB index (mxbai-embed-large embeddings)
```

## Tech Stack

| Component | Library |
|---|---|
| RAG orchestration | [LlamaIndex](https://www.llamaindex.ai/) |
| Vector database | [ChromaDB](https://www.trychroma.com/) |
| Embeddings & LLM | [Ollama](https://ollama.com) |
| Embedding model | `mxbai-embed-large` |
| LLM | `llama3.1:8b` |
| Dependency management | [uv](https://github.com/astral-sh/uv) |

## Known Limitations

- **Date/temporal queries** — semantic search cannot rank or compare date values; these require metadata filtering
- **High-volume questions** — when 50+ clients match a query, k=20 caps recall at ~30%
- **Synthetic data** — notes were AI-generated; a real CRM would produce messier, more varied text
