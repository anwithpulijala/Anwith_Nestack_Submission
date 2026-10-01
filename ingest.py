#!/usr/bin/env python3
"""
PDF Ingestion Script using LangChain & LangGraph.
Nestack Technical Assessment: PDF Vectorisation Pipeline

Usage:
    python ingest.py --file data/Anwith_Major_Project_Report.pdf
    python ingest.py --file path/to/document.pdf
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, TypedDict, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

import chromadb
from langgraph.graph import StateGraph, START, END

import config
from embeddings import get_embedding_model


# ---------------------------------------------------------------------------
# LangGraph State Definition
# ---------------------------------------------------------------------------
class IngestionState(TypedDict, total=False):
    file_path: str
    chunk_size: int
    chunk_overlap: int
    persist_directory: str
    documents: List[Document]
    chunks: List[Document]
    embedding_model: Embeddings
    total_pages: int
    total_chunks: int


# ---------------------------------------------------------------------------
# Graph Nodes: load -> split -> embed -> store
# ---------------------------------------------------------------------------
def node_load(state: IngestionState) -> IngestionState:
    """Node 1: Load PDF using LangChain PyPDFLoader and preserve page metadata."""
    path = Path(state["file_path"])
    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist at: {path.resolve()}")

    loader = PyPDFLoader(file_path=str(path.resolve()))
    documents = loader.load()

    if not documents:
        raise ValueError(f"No text extracted from PDF: {path.name}")

    # Ensure each loaded page has explicit page index metadata
    for i, doc in enumerate(documents):
        if "page" not in doc.metadata:
            doc.metadata["page"] = i

    print(f"  [load]  Loaded {len(documents)} page(s) from '{path.name}' via PyPDFLoader.")
    print(f"  [load]  First page character count: {len(documents[0].page_content)}")
    if len(documents) > 1:
        print(f"  [load]  Last page character count: {len(documents[-1].page_content)}")

    return {
        **state,
        "documents": documents,
        "total_pages": len(documents)
    }


def node_split(state: IngestionState) -> IngestionState:
    """Node 2: Split loaded document using RecursiveCharacterTextSplitter with page preservation."""
    chunk_size = state.get("chunk_size", config.CHUNK_SIZE)
    chunk_overlap = state.get("chunk_overlap", config.CHUNK_OVERLAP)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False
    )

    raw_chunks = splitter.split_documents(state["documents"])

    # Ensure 1-indexed page_number metadata is explicitly attached to each chunk
    processed_chunks: List[Document] = []
    chunks_per_page = {}
    for chunk in raw_chunks:
        raw_page = chunk.metadata.get("page", 0)
        try:
            page_number = int(raw_page) + 1
        except (ValueError, TypeError):
            page_number = 1

        chunk.metadata["page_number"] = page_number
        processed_chunks.append(chunk)
        chunks_per_page[page_number] = chunks_per_page.get(page_number, 0) + 1

    print(f"  [split] Created {len(processed_chunks)} chunk(s) from {len(state['documents'])} page(s) (size={chunk_size}, overlap={chunk_overlap}).")
    print(f"  [split] Chunks per page summary: {len(chunks_per_page)} unique pages, average {len(processed_chunks)/max(len(state['documents']), 1):.2f} chunks/page.")
    print("  [split] Sample chunks:")
    for sample_idx in range(min(3, len(processed_chunks))):
        sc = processed_chunks[sample_idx]
        preview = sc.page_content.strip().replace("\n", " ")[:85]
        print(f"    - Chunk #{sample_idx + 1} [Page {sc.metadata['page_number']}, {len(sc.page_content)} chars]: \"{preview}...\"")

    return {
        **state,
        "chunks": processed_chunks,
        "total_chunks": len(processed_chunks)
    }


def node_embed(state: IngestionState) -> IngestionState:
    """Node 3: Initialize shared LangChain embedding model."""
    embed_model = get_embedding_model()
    print(f"  [embed] Initialized '{config.EMBEDDING_MODEL_NAME}' embedding integration.")
    return {
        **state,
        "embedding_model": embed_model
    }


def node_store(state: IngestionState) -> IngestionState:
    """Node 4: Persist embeddings and metadata into a clean Chroma vector store without stale duplicates."""
    persist_dir = state.get("persist_directory", config.VECTORSTORE_DIR)
    chunks = state["chunks"]
    embed_model = state["embedding_model"]

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path.resolve()))

    # Delete existing collection to prevent accumulation of stale/duplicate vectors across runs
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
        print(f"  [store] Cleared previous collection '{config.COLLECTION_NAME}' to prevent stale vectors.")
    except Exception:
        pass

    vectorstore = Chroma(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=embed_model
    )

    # Assign deterministic unique chunk IDs to guarantee exactly 1 vector per chunk
    chunk_ids = [
        f"chunk_{i + 1}_p{chunk.metadata.get('page_number', 1)}"
        for i, chunk in enumerate(chunks)
    ]

    vectorstore.add_documents(documents=chunks, ids=chunk_ids)

    # Verify collection count in vector store
    collection_count = vectorstore._collection.count()
    if collection_count != len(chunks):
        raise ValueError(
            f"Collection count mismatch: expected {len(chunks)} vectors, but found {collection_count} in ChromaDB."
        )

    print(f"  [store] Successfully persisted {collection_count} vectors to disk at '{persist_dir}'.")
    return state



# ---------------------------------------------------------------------------
# Build LangGraph Workflow
# ---------------------------------------------------------------------------
def build_ingestion_graph():
    builder = StateGraph(IngestionState)
    builder.add_node("load", node_load)
    builder.add_node("split", node_split)
    builder.add_node("embed", node_embed)
    builder.add_node("store", node_store)

    builder.add_edge(START, "load")
    builder.add_edge("load", "split")
    builder.add_edge("split", "embed")
    builder.add_edge("embed", "store")
    builder.add_edge("store", END)

    return builder.compile()


ingestion_workflow = build_ingestion_graph()


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Execution
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest a PDF document into persistent Chroma vector store using LangChain & LangGraph."
    )
    parser.add_argument(
        "--file", "-f",
        dest="file_path",
        required=False,
        default=None,
        type=str,
        help="Path to the PDF file to ingest (e.g. document.pdf)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.CHUNK_SIZE,
        help=f"Chunk size in characters (default: {config.CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=config.CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {config.CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=config.VECTORSTORE_DIR,
        help=f"Vector store directory (default: {config.VECTORSTORE_DIR})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    file_input = args.file_path

    # If --file not supplied on CLI, look for default in data/ or prompt
    if not file_input:
        candidates = [
            Path("data/Anwith_Major_Project_Report.pdf"),
            Path("data/sample_document.pdf")
        ]
        default_file = next((str(p) for p in candidates if p.exists()), None)
        if default_file and not sys.stdin.isatty():
            file_input = default_file
        else:
            prompt = f"Enter PDF file path" + (f" [default: {default_file}]: " if default_file else ": ")
            try:
                entered = input(prompt).strip().strip('"').strip("'")
                file_input = entered if entered else default_file
            except (EOFError, KeyboardInterrupt):
                print("\nOperation cancelled.")
                sys.exit(0)

    if not file_input:
        print("Error: No PDF file specified. Use: python ingest.py --file <path>", file=sys.stderr)
        sys.exit(1)

    target_path = Path(file_input)
    if not target_path.exists():
        print(f"Error: PDF file not found at '{target_path.resolve()}'", file=sys.stderr)
        sys.exit(1)

    print("=" * 64)
    print("  PDF VECTORISATION PIPELINE - INGESTION")
    print("=" * 64)
    print(f"Target Document   : {target_path.resolve()}")
    print(f"File Size         : {target_path.stat().st_size:,} bytes")
    print(f"Chunk Size        : {args.chunk_size}")
    print(f"Chunk Overlap     : {args.chunk_overlap}")
    print(f"Vector Store Path : {Path(args.db_dir).resolve()}")
    print(f"LangGraph Nodes   : [START] -> load -> split -> embed -> store -> [END]")
    print("-" * 64)

    start_time = time.time()
    try:
        final_state = ingestion_workflow.invoke({
            "file_path": str(target_path.resolve()),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "persist_directory": str(Path(args.db_dir).resolve())
        })
    except Exception as exc:
        print(f"\n[!] Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start_time
    print("-" * 64)
    print(f"SUCCESS: Ingestion completed in {elapsed:.2f} seconds.")
    print(f"Pages Processed : {final_state.get('total_pages', 0)}")
    print(f"Chunks Indexed  : {final_state.get('total_chunks', 0)}")
    print(f"Vector Store    : {Path(args.db_dir).resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
