# Fasih-Docs: Evaluated RAG for EDA Documentation

Fasih-Docs is a localized Retrieval-Augmented Generation (RAG) system designed to provide high-accuracy answers for technical EDA (Electronic Design Automation) and engineering documentation. 

**Why I built this:** Technical documentation in the semiconductor industry is dense and privacy-sensitive. This project explores building a system that runs entirely locally, ensuring IP security while maintaining high retrieval precision.

## 🚀 Performance Metrics (Benchmarked)
I built a custom evaluation suite to stress-test the pipeline across various query categories.
* **Response Rate:** 100% (on technical scope)
* **Source Citation:** 100% (Every answer is grounded in documentation)
* **Safety:** Successfully detects "Out of Scope" queries to prevent hallucinations.
* **Privacy:** 100% Local (Mistral-7B via Ollama).

## 🛠️ Tech Stack & Engineering Decisions
* **LLM:** Mistral-7B (via Ollama) - Chosen for its balance of speed and technical reasoning.
* **Vector Store:** ChromaDB - Integrated with **MMR (Maximum Marginal Relevance)** to ensure diversity in retrieved documentation chunks.
* **Embeddings:** `all-MiniLM-L6-v2` - Optimized for CPU-based inference.
* **Validation:** Custom Python evaluation script that measures latency and grounding.

## 📁 Key Features
* **Bilingual Support:** Handles English and Arabic queries (relevant for regional support).
* **Hallucination Guardrails:** Explicitly configured to admit ignorance rather than invent technical data.
* **Traceability:** Every response includes a preview of the source text and page number for engineer verification.

## ⚙️ Setup
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run `python src/ingest.py` to index docs.
4. Launch UI: `python src/app.py`
