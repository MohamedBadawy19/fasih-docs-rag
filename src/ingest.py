"""
ingest.py — Load PDFs, chunk them, embed, store in ChromaDB.

Run this ONCE (or again whenever you add new PDFs).
What it does step by step:
  1. Reads every PDF from /docs folder
  2. Splits text into overlapping chunks
  3. Converts each chunk into a vector (embedding)
  4. Stores all vectors in ChromaDB on disk
"""

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config

console = Console()


def load_pdfs(docs_dir: str) -> list:
    """
    Walk through the docs/ folder and load every PDF.
    Returns a list of LangChain Document objects.
    Each Document has .page_content (the text) and .metadata (filename, page number).
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        console.print(f"[red]ERROR: '{docs_dir}' folder not found.[/red]")
        console.print("[yellow]Create a 'docs' folder and put your PDFs inside it.[/yellow]")
        sys.exit(1)

    pdf_files = list(docs_path.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[red]ERROR: No PDF files found in '{docs_dir}'.[/red]")
        console.print("[yellow]Download any technical PDF and place it in the docs/ folder.[/yellow]")
        console.print("[yellow]Suggestion: search 'EDA design rules PDF' or 'VLSI tutorial PDF'[/yellow]")
        sys.exit(1)

    all_documents = []

    for pdf_path in track(pdf_files, description="Loading PDFs..."):
        console.print(f"  Reading: [cyan]{pdf_path.name}[/cyan]")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        # Add the filename to metadata so we can cite sources later
        for doc in documents:
            doc.metadata["source_file"] = pdf_path.name

        all_documents.extend(documents)
        console.print(f"  Loaded [green]{len(documents)} pages[/green] from {pdf_path.name}")

    console.print(f"\n[bold green]Total pages loaded: {len(all_documents)}[/bold green]")
    return all_documents


def split_documents(documents: list) -> list:
    """
    Split documents into smaller chunks.

    WHY we chunk:
    - LLMs have a context window limit (can't read 500 pages at once)
    - Smaller chunks = more precise retrieval
    - Overlap ensures sentences at chunk boundaries aren't lost

    RecursiveCharacterTextSplitter tries to split at:
    paragraphs → sentences → words → characters (in that order)
    so it never cuts in the middle of a sentence if it can avoid it.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "،", " ", ""],  # "،" is Arabic comma
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    console.print(f"[bold green]Total chunks created: {len(chunks)}[/bold green]")
    console.print(f"[dim]Average chunk size: ~{config.CHUNK_SIZE} characters with {config.CHUNK_OVERLAP} overlap[/dim]")
    return chunks


def build_vector_store(chunks: list) -> Chroma:
    """
    Convert chunks to vectors and store in ChromaDB.

    HOW embeddings work:
    - Each chunk of text gets converted into a list of ~384 numbers
    - Similar text = similar numbers = close together in vector space
    - When you ask a question, it gets converted the same way
    - ChromaDB finds the chunks whose vectors are closest to your question

    multilingual-e5-small supports Arabic + English.
    It downloads once (~90MB) and runs offline forever after.
    """
    console.print("\n[yellow]Loading embedding model (downloads once ~90MB)...[/yellow]")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  
        encode_kwargs={"normalize_embeddings": True},
    )

    console.print("[yellow]Building vector store (this takes 1-3 minutes)...[/yellow]")

    # Delete old DB if it exists so we start fresh
    import shutil
    if Path(config.CHROMA_DIR).exists():
        shutil.rmtree(config.CHROMA_DIR)
        console.print("[dim]Cleared old vector store.[/dim]")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
    )

    console.print(f"[bold green]Vector store saved to '{config.CHROMA_DIR}'[/bold green]")
    console.print(f"[bold green]Total vectors stored: {vector_store._collection.count()}[/bold green]")
    return vector_store


def main():
    console.print(f"\n[bold cyan]{'='*50}[/bold cyan]")
    console.print(f"[bold cyan]  Fasih-Docs Ingestion Pipeline[/bold cyan]")
    console.print(f"[bold cyan]{'='*50}[/bold cyan]\n")

    # Step 1: Load
    documents = load_pdfs(config.DOCS_DIR)

    # Step 2: Chunk
    chunks = split_documents(documents)

    # Step 3: Embed + Store
    build_vector_store(chunks)

    console.print("\n[bold green]Done! Now run: python app.py[/bold green]")


if __name__ == "__main__":
    main()
