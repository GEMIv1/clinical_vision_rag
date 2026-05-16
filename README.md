# 🏥 Vision RAG — Clinical Case Search

> **🧩 This project is a reusable template for any OCR + RAG application.**  
> Swap out the embedding model, adjust the chunking strategy, pick your preferred vision LLM, ingest your own data — and you have a ready retrieval system for **any** domain.  
> As a demonstration, this repo uses **clinical patient case reports** from [PMC-Patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients) to build a clinical decision-support search engine.

A **Retrieval-Augmented Generation** system that combines **Vision-Language OCR** with **hybrid search**. In its current configuration, physicians can query a database of past patient case reports from medical literature to find similar cases, treatments, and outcomes — using either natural language or by uploading medical document images.

**To adapt it to your own domain**, you only need to:
1. **Choose your embedding model** — replace `EMBED_MODEL` in `.env` (e.g., `all-MiniLM-L6-v2` for general text, domain-specific models for specialized fields)
2. **Choose your vision/OCR models** — replace `MODEL_NAME` and `BACKUP_MODEL_NAME` (any model on the HuggingFace Inference API)
3. **Tune chunking parameters** — adjust `max_words` and `overlap_words` in `app/rag/ingest.py` to match your document structure
4. **Prepare your CSV** — format your data with columns: `patient_uid`, `title`, `age`, `gender`, `patient` (or modify the ingest script for your schema)
5. **Ingest and search** — `POST /ingest/` then start querying

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Run with Docker (Recommended)](#run-with-docker-recommended)
  - [Run Locally (Without Docker)](#run-locally-without-docker)
- [Usage](#usage)
  - [1. Ingest Data](#1-ingest-data)
  - [2. Search via UI](#2-search-via-ui)
  - [3. Search via API](#3-search-via-api)
  - [4. OCR — Extract Text from Images](#4-ocr--extract-text-from-images)
- [API Reference](#api-reference)
- [RAG Pipeline Details](#rag-pipeline-details)

---

## Overview

**Vision RAG** combines **Vision-Language Models** for OCR with a **hybrid retrieval pipeline** to help physicians search through thousands of past patient case reports. A doctor can type a clinical question — or upload a photo of a medical document — and the system will:

1. Extract text from the image (if provided) using cloud-hosted VLMs
2. Retrieve the most relevant patient cases using hybrid search (semantic + keyword)
3. Re-rank results with a cross-encoder for precision
4. Generate a clinical summary grounded in the retrieved evidence

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (:8501)                   │
│   • Clinical question input    • Image upload (OCR)              │
│   • Filters (age, gender)      • Results display                 │
└─────────────────────────┬────────────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼────────────────────────────────────────┐
│                     FastAPI Backend (:8000)                       │
│                                                                  │
│  ┌──────────┐  ┌──────────────────────────────────────────────┐  │
│  │ OCR      │  │              RAG Pipeline                    │  │
│  │ Router   │  │                                              │  │
│  │          │  │  Query ──► Semantic Search (ChromaDB)  ──┐   │  │
│  │ Image    │  │        ──► BM25 Keyword Search ──────────┤   │  │
│  │  ▼       │  │                                          ▼   │  │
│  │ Preproc. │  │                              RRF Fusion      │  │
│  │  ▼       │  │                                  ▼           │  │
│  │ VLM API  │  │                          Cross-Encoder       │  │
│  │ (HF)     │  │                            Re-ranking        │  │
│  │          │  │                                  ▼           │  │
│  └──────────┘  │                          Deduplication       │  │
│                │                                  ▼           │  │
│                │                          Groq LLM Generation │  │
│                └──────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────┐  ┌────────────────────┐                      │
│  │ ChromaDB       │  │ BioLORD Embeddings │                      │
│  │ (Persistent)   │  │ (sentence-transf.) │                      │
│  └────────────────┘  └────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **Vision OCR** | Extract text from medical documents/scans using Qwen2.5-VL-72B or Pixtral-Large (automatic fallback) |
| **Image Preprocessing** | OpenCV-based deskewing, denoising, and contrast enhancement before OCR |
| **Hybrid Retrieval** | Combines dense semantic search (ChromaDB) with sparse keyword search (BM25) |
| **Reciprocal Rank Fusion** | Merges ranked lists from both retrievers into a single, high-quality ranking |
| **Cross-Encoder Re-ranking** | Fine-grained relevance scoring with `ms-marco-MiniLM-L-6-v2` |
| **Patient Deduplication** | Returns only the top-ranked chunk per unique patient to avoid redundancy |
| **Metadata Filtering** | Filter results by age range and gender at query time |
| **Grounded Generation** | LLM answers are strictly grounded in retrieved evidence (Groq / Llama 3.3 70B) |


---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | FastAPI + Uvicorn |
| **OCR Models** | Qwen2.5-VL-72B-Instruct, Pixtral-Large-Instruct (via HuggingFace Inference API) |
| **Embeddings** | `FremyCompany/BioLORD-2023-C` (biomedical sentence-transformers) |
| **Vector Store** | ChromaDB (persistent, cosine similarity) |
| **Keyword Search** | BM25Okapi (rank-bm25) |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Llama 3.3 70B Versatile (via Groq API) |
| **Image Processing** | OpenCV + NumPy |
| **Data Source** | PMC-Patients (PubMed Central patient case reports) |
| **Containerization** | Docker + Docker Compose |

---

## Project Structure

```
vision_rag/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic settings (reads .env)
│   ├── main.py                   # FastAPI app, lifespan, health check
│   ├── preprocessing/
│   │   ├── deskew.py             # Image deskew correction
│   │   ├── pipeline.py           # Preprocessing orchestrator
│   │   └── transforms.py         # Denoise, contrast, threshold
│   ├── rag/
│   │   ├── data/
│   │   │   ├── PMC-Patients.csv  # Raw dataset (~545 MB)
│   │   │   ├── preprocessed.csv  # Cleaned dataset (~31 MB)
│   │   │   ├── preprocessing.py  # Data cleaning script
│   │   ├── embedder.py           # SentenceTransformer embed/embed_batch
│   │   ├── generator.py          # Groq LLM answer generation
│   │   ├── ingest.py             # CSV → chunked → ChromaDB
│   │   ├── retriever.py          # Hybrid retrieval + RRF + reranking
│   │   └── store.py              # ChromaDB vector store wrapper

│   ├── routers/
│   │   ├── ingest.py             # POST /ingest/
│   │   ├── ocr.py                # POST /ocr/
│   │   └── search.py             # POST /search/, /search/retrieve
│   ├── schemas/
│   │   ├── ocrRequest.py         # OCR request model
│   │   ├── ocrResult.py          # OCR response model
│   │   ├── outputFormat.py       # plain/json enum
│   │   ├── searchQuery.py        # Search request model
│   │   └── searchQueryResult.py  # Search response model
│   └── vision/
│       ├── client.py             # HuggingFace VLM API client
│       ├── parser.py             # Parse VLM response
│       └── prompts.py            # OCR prompt builder
├── streamlit_app.py              # Streamlit UI
├── Dockerfile                    # Container image definition
├── docker-compose.yml            # Multi-service orchestration
├── .dockerignore                 # Files excluded from Docker build
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for environment variables
├── .env                          # Actual secrets (git-ignored)
└── .gitignore
```

---

## Getting Started

### Prerequisites

- **Docker** and **Docker Compose** (v2) — [Install Docker](https://docs.docker.com/get-docker/)
- A **Hugging Face** account with an API token (for vision models)
- A **Groq** account with an API key (for LLM generation)

### Environment Variables

Copy the example file and fill in your credentials and models to use:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `MODEL_NAME` | Primary vision model for OCR (Hugging Face model ID) |
| `BACKUP_MODEL_NAME` | Fallback vision model if primary fails |
| `EMBED_MODEL` | Sentence-transformer model for document embeddings |
| `HF_TOKEN` | Hugging Face API token ([get one here](https://huggingface.co/settings/tokens)) |
| `CHROMA_PATH` | Path where ChromaDB stores its persistent data |
| `GROQ_API_KEY` | Groq API key for LLM inference ([get one here](https://console.groq.com/keys)) |
| `GROQ_MODEL` | Groq model name for answer generation |

---

### Run with Docker (Recommended)

#### 1. Build and start both services

```bash
docker compose up --build
```

This starts two containers:

| Service | Container Name | URL | Description |
|---|---|---|---|
| `api` | `vision-rag-api` | http://localhost:8000 | FastAPI backend |
| `streamlit` | `vision-rag-ui` | http://localhost:8501 | Streamlit frontend |

#### 2. Verify the services are running

```bash
# Check API health
curl http://localhost:8000/health

# Or open the interactive docs
# http://localhost:8000/docs
```

#### 3. Ingest data (first time only)

The vector database starts empty. You need to ingest the patient case data:

```bash
curl -X POST "http://localhost:8000/ingest/" \
  -H "Content-Type: application/json"
```

> **Note:** This will embed all patient records from `app/rag/data/preprocessed.csv` into ChromaDB. It may take several minutes depending on your hardware. Progress is logged in the API container.

#### 4. Open the UI

Navigate to **http://localhost:8501** in your browser and start searching.

#### Common Docker Commands

```bash
# Start in detached mode (background)
docker compose up --build -d

# View logs
docker compose logs -f

# View logs for a specific service
docker compose logs -f api
docker compose logs -f streamlit

# Stop all services
docker compose down

# Stop and remove volumes (resets ChromaDB data)
docker compose down -v

# Rebuild after code changes
docker compose up --build
```

---

### Run Locally (Without Docker)

#### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Start the FastAPI backend

```bash
uvicorn app.main:app --reload --port 8000
```

#### 4. Start the Streamlit frontend (in a separate terminal)

```bash
streamlit run streamlit_app.py
```

#### 5. Ingest data

```bash
curl -X POST "http://localhost:8000/ingest/"
```

---

## Usage

### 1. Ingest Data

Before searching, the patient case data must be loaded into ChromaDB:

```bash
# Uses the default path: app/rag/data/preprocessed.csv
curl -X POST "http://localhost:8000/ingest/"

# Or specify a custom CSV path
curl -X POST "http://localhost:8000/ingest/?csv_path=app/rag/data/preprocessed.csv"
```

The CSV must contain columns: `patient_uid`, `title`, `age`, `gender`, `patient` (narrative text).

> **📋 Data Preprocessing Note:** The raw PMC-Patients dataset stores patient ages in inconsistent scales — some in **years**, others in **months** or **days**. A preprocessing step (`app/rag/data/preprocessing.py`) normalizes all age values to a **single unified scale** so that age-based filtering works correctly across the entire dataset.

#### Chunking Strategy

Each patient narrative is split using a **fixed-size sliding window** approach before embedding and storage:

| Parameter | Value | Purpose |
|---|---|---|
| `max_words` | 256 | Maximum words per chunk |
| `overlap_words` | 50 | Words shared between consecutive chunks |

**How it works:** The text is split by whitespace into words. A window of 256 words slides forward by 206 words (256 − 50) at each step, producing overlapping chunks until the entire document is covered. Short documents that fit within 256 words are kept as a single chunk.

**Why this approach:**

- **Fixed-size word-level chunking** gives consistent chunk lengths, which helps the embedding model produce comparable vectors — chunks that are too short or too long degrade retrieval quality.
- **256 words** was chosen to stay well within the context window of `BioLORD-2023-C` (512 tokens ≈ ~380 words) while being large enough to capture a meaningful clinical paragraph (a symptom description, a treatment plan, etc.).
- **50-word overlap** ensures that information at chunk boundaries isn't lost. A clinical finding mentioned at the end of one chunk will also appear at the start of the next, so retrieval can still surface it regardless of which chunk is matched.
- **Word-level** (vs. character-level or sentence-level) splitting is a natural fit for medical narratives, which are written in flowing prose rather than structured sentences.

Each chunk is then embedded using BioLORD-2023-C and stored in ChromaDB with metadata (`patient_uid`, `age`, `gender`, `chunk_index`, `total_chunks`).

### 2. Search via UI

1. Open **http://localhost:8501**
2. Type a clinical question (e.g., *"Have we seen a case of a child with respiratory failure after COVID-19?"*)
3. Optionally attach a medical document image — text will be extracted and appended to your query
4. Adjust filters (age range, gender, number of results)
5. Click **🔍 Search (RAG)** for a full answer, or **📄 Retrieve Only** for raw documents

### 3. Search via API

#### RAG Search (retrieve + generate)

```bash
curl -X POST "http://localhost:8000/search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Patient with bilateral lung infiltrates and COVID-19",
    "top_k": 5,
    "min_age": 2,
    "max_age": 18,
    "gender": "M"
  }'
```

**Response:**
```json
{
  "answer": "Based on the retrieved cases...",
  "sources": [
    {
      "id": "PMC123_chunk_0",
      "metadata": {
        "patient_uid": "PMC123",
        "title": "...",
        "age": "8",
        "gender": "M"
      },
      "rrf_score": 0.064516
    }
  ]
}
```

#### Retrieve Only (no LLM generation)

```bash
curl -X POST "http://localhost:8000/search/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "renal failure in diabetic patients", "top_k": 3}'
```

### 4. OCR — Extract Text from Images

```bash
curl -X POST "http://localhost:8000/ocr/" \
  -F "file=@/path/to/medical_document.png" \
  -F "output_format=plain"
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | *required* | Image file (PNG, JPG, JPEG, WEBP, TIFF) |
| `prompt` | string | `null` | Custom extraction prompt |
| `output_format` | string | `plain` | `plain` or `json` |
| `language_hint` | string | `null` | Hint for document language |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info (model names) |
| `GET` | `/health` | Health check (model connectivity) |
| `POST` | `/ocr/` | Extract text from an uploaded image |
| `POST` | `/search/` | RAG search — retrieve + LLM-generated answer |
| `POST` | `/search/retrieve` | Retrieve matching documents without generation |
| `POST` | `/ingest/` | Ingest patient data from CSV into ChromaDB |

Interactive API documentation is available at **http://localhost:8000/docs** (Swagger UI).

---

## RAG Pipeline Details

The retrieval pipeline follows a multi-stage approach for high-quality results:

### Stage 1 & 2 — Dual Retrieval

Two independent retrieval methods run in parallel:

- **Semantic Search** — The query is embedded with BioLORD-2023-C and searched against ChromaDB using cosine similarity. Captures meaning and clinical context.
- **BM25 Keyword Search** — The query is tokenized and scored against all documents using BM25Okapi. Captures exact medical terms, drug names, and acronyms.

Both retrievers fetch `top_k × 3` candidates to provide a wide pool for fusion.

### Stage 3 — Reciprocal Rank Fusion (RRF)

The two ranked lists are merged using RRF with `k=30`:

```
RRF_score(doc) = Σ  1 / (k + rank_i)
```

Documents found by both retrievers receive a score boost, combining the strengths of semantic understanding and exact term matching.

### Stage 4 — Cross-Encoder Re-ranking

The fused list is re-scored using `cross-encoder/ms-marco-MiniLM-L-6-v2`, which jointly encodes the (query, document) pair for fine-grained relevance assessment. Scores are sigmoid-normalized to [0, 1].

### Stage 5 — Patient Deduplication

Multiple chunks from the same patient may appear in results. The pipeline deduplicates by `patient_uid`, keeping only the highest-ranked chunk per patient.

### Stage 6 — Grounded Generation

The final documents are passed to Llama 3.3 70B (via Groq) with a clinical system prompt that instructs the model to:
- Evaluate clinical similarity
- Summarize treatments and outcomes
- Cite patient IDs
- Flag weak matches (RRF score < 0.5)
- Never use external knowledge

---
