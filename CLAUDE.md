# CLAUDE.md

## Project Overview

UDA (Unstructured Document Analysis) — a RAG system for financial QA over SEC 10-K annual reports. The pipeline converts PDFs to structured chunks, embeds them into a vector store, and retrieves relevant context for question answering.

## Pipeline Stages

1. **PDF Parsing** (`code/dockling.ipynb`) — Docling converts 10-K PDFs to structured documents with layout detection and table extraction (GPU-accelerated, TableFormer ACCURATE mode). Outputs chunked JSON files to `data/src/docling_output/<timestamp>/`.
2. **Ingestion** (`code/ingest.ipynb`) — Loads chunked JSON, embeds with BGE-M3 (dense 1024d + sparse IDF), upserts into Qdrant collection `annual_reports`.
3. **Retrieval** (`code/01_retrieval(2).ipynb`) — Hybrid retrieval (dense + sparse via RRF fusion) from Qdrant, evaluated against FinQA and FeTaQA benchmarks.

## Tech Stack

- **Embedding**: `BAAI/bge-m3` (FlagEmbedding, fp16, CUDA)
- **Vector DB**: Qdrant (local/embedded mode, stored at `data/qdrant_db/`)
- **PDF Processing**: Docling with HybridChunker (max 300 tokens, 20 overlap)
- **Framework**: LangChain (partial), pandas, PyTorch (CUDA 12.8)
- **LLM**: LM Studio (local API via requests)

## Data Layout

```
data/
  src/fin_docs/           # Source 10-K PDFs (TICKER_YEAR.pdf)
  src/docling_output/     # Chunked JSON output (timestamped dirs)
  qa/fin_qa.csv           # FinQA benchmark questions
  qa/feta_qa.csv          # FeTaQA benchmark questions
  qdrant_db/              # Qdrant persistent storage
```

## Key Conventions

- Chunk IDs follow the pattern `{TICKER}_{YEAR}_text_{index}`
- PDF names follow `TICKER_YEAR` convention (e.g., `AAL_2010`)
- Qdrant collection uses dual vectors: `dense` (cosine, 1024d) and `sparse` (IDF)
- Payload fields indexed: `pdf_name`, `ticker`, `year`, `type`, `section`, `page`
- Deterministic point IDs via UUID5 from chunk_id

## Dev Environment

- Python venv at `.venv/`
- PyCharm project
- Windows 11, CUDA GPU
- Install: `pip install -r requirements.txt` + separate PyTorch CUDA + chunking_evaluation from GitHub

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/brandonstarxel/chunking_evaluation.git
```