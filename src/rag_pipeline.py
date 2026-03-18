"""
rag_pipeline.py — The core RAG logic.

This is the brain of the system. It:
  1. Loads the vector store from disk
  2. Takes a user question
  3. Finds the most relevant chunks (retrieval)
  4. Builds a prompt combining question + chunks (augmentation)
  5. Sends to Mistral and returns the answer (generation)
  6. Logs every query to a file

You will learn:
  - How retrieval works (similarity search)
  - How to build a good RAG prompt
  - How to cite sources from retrieved chunks
  - How to call a local LLM via Ollama
  - Logging patterns for production systems
"""

import os
import logging
from pathlib import Path
from datetime import datetime

# Updated modern imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM  # Changed from Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import config


# ── Logging setup ──────────────────────────────────────────────
# This writes every query + answer to a file.
# In production systems, logging is mandatory.
# At Siemens, they will love that you thought about this.

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
query_logger = logging.getLogger("queries")


# ── Prompt Template ────────────────────────────────────────────
# This is the exact instruction sent to the LLM.
# {context} = the retrieved chunks
# {question} = what the user asked
#
# WHY this prompt works:
# - "Only use the context below" → prevents hallucination
# - "If not in context, say so" → honest responses
# - "Cite the source" → engineers need to verify information
# - Arabic instruction → makes it bilingual

RAG_PROMPT_TEMPLATE = """You are Fasih-Docs, an expert AI assistant for engineering and EDA (Electronic Design Automation) documentation.

Your job is to answer technical questions accurately based ONLY on the documentation provided below.

Rules:
- Answer only from the context provided. Do not use outside knowledge.
- If the answer is not in the context, say: "This information is not available in the loaded documentation."
- Always cite which document and page number your answer comes from.
- Be precise and technical. Engineers need accurate answers.
- IMPORTANT: Always respond in the same language as the question.
  If the question is in Arabic, your entire answer must be in Arabic.
  If in English, answer in English.

---
DOCUMENTATION CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


class RAGPipeline:
    """
    The complete RAG pipeline as a class.
    
    Using a class means:
    - The embedding model loads ONCE when the app starts (not on every query)
    - The vector store loads ONCE
    - Every query reuses the loaded components = fast responses
    """

    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.chain = None
        self.is_ready = False
        self._load()

    def _load(self):
        """Load all components. Called once at startup."""

        # ── 1. Load embeddings ─────────────────────────────────
        # Same model used in ingest.py — must match or retrieval breaks
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ── 2. Load vector store ───────────────────────────────
        if not Path(config.CHROMA_DIR).exists():
            print(f"ERROR: Vector store not found at '{config.CHROMA_DIR}'")
            print("Run: python ingest.py first")
            return

        print("Loading vector store...")
        self.vector_store = Chroma(
            persist_directory=config.CHROMA_DIR,
            embedding_function=self.embeddings,
        )

        # ── 3. Create retriever ────────────────────────────────
        # The retriever takes a question and returns TOP_K most relevant chunks.
        # search_type="mmr" = Maximum Marginal Relevance
        # MMR avoids returning 4 nearly identical chunks.
        # It balances relevance AND diversity.
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.TOP_K_RESULTS,
                "fetch_k": 10,   # fetch 10, then pick the 4 most diverse
            },
        )

        # ── 4. Load LLM (Mistral via Ollama) ──────────────────
        print("Connecting to Ollama (Mistral)...")
        self.llm = OllamaLLM(  # Using the new class from langchain-ollama
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,
        )

        # ── 5. Build the chain ─────────────────────────────────
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        self.is_ready = True
        print("RAG Pipeline ready.")

    def _format_docs(self, docs) -> str:
        """
        Format retrieved documents into a string for the prompt.
        Includes source filename and page number for citations.
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(
                f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def query(self, question: str) -> dict:
        """
        Main method. Takes a question, returns answer + sources.
        
        Returns:
            dict with keys: "answer", "sources", "error"
        """
        if not self.is_ready:
            return {
                "answer": "System not ready. Please run python ingest.py first.",
                "sources": [],
                "error": True,
            }

        if not question.strip():
            return {"answer": "Please enter a question.", "sources": [], "error": True}

        try:
            # Get answer from chain
            answer = self.chain.invoke(question)

            # Get the source documents separately for display
            source_docs = self.retriever.invoke(question)
            sources = self._extract_sources(source_docs)

            # Log the query
            query_logger.info(f"Q: {question}")
            query_logger.info(f"A: {answer[:200]}...")  # Log first 200 chars
            query_logger.info(f"Sources: {sources}")
            query_logger.info("-" * 60)

            return {"answer": answer, "sources": sources, "error": False}

        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                return {
                    "answer": "Cannot connect to Ollama. Make sure Ollama is running: open a terminal and run 'ollama serve'",
                    "sources": [],
                    "error": True,
                }
            return {"answer": f"Error: {error_msg}", "sources": [], "error": True}

    def _extract_sources(self, docs) -> list:
        """Extract clean source information from retrieved documents."""
        sources = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            key = f"{source}:p{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": source,
                    "page": page,
                    "preview": doc.page_content[:150] + "...",
                })
        return sources
