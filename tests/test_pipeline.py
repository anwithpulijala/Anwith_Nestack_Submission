"""
Pipeline unit tests for ingest.py and LangGraph workflow.
Tests document splitting, page metadata preservation, embedding model, and graph structure.
"""

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embeddings import get_embedding_model
from ingest import build_ingestion_graph


def test_splitter_chunking_and_metadata():
    """Verify RecursiveCharacterTextSplitter respects chunk size and overlap."""
    sample_text = "Paragraph one regarding Sub Inspector qualifications.\n\n" * 30
    doc = Document(page_content=sample_text, metadata={"page": 15})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents([doc])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.page_content) <= config.CHUNK_SIZE + 50
        page_num = int(chunk.metadata.get("page", 0)) + 1
        assert page_num == 16


def test_embedding_model_initialization():
    """Verify HuggingFace embedding model initializes with configured parameters."""
    model = get_embedding_model()
    assert model is not None
    # Test single sentence embedding
    vector = model.embed_query("Test query for Sub Inspector recruitment")
    assert isinstance(vector, list)
    assert len(vector) == 384  # all-MiniLM-L6-v2 dimension


def test_langgraph_nodes_structure():
    """Verify LangGraph compiles and contains the required named nodes."""
    graph = build_ingestion_graph()
    assert graph is not None

    graph_dict = graph.get_graph().to_json()
    assert "nodes" in graph_dict
    node_ids = [n["id"] for n in graph_dict["nodes"]]

    for expected_node in ["load", "split", "embed", "store"]:
        assert expected_node in node_ids, f"Missing required node: {expected_node}"
