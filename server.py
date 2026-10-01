#!/usr/bin/env python3
"""
FastAPI Retrieval Server.
Nestack Technical Assessment: PDF Vectorisation Pipeline

Exposes strictly a single HTTP endpoint:
    POST /query   -> similarity_search_with_score() direct vector retrieval
"""

from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import uvicorn

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

import config
from embeddings import get_embedding_model


app = FastAPI(
    title="Nestack PDF Vectorisation Pipeline - Retrieval API",
    description="Exposes direct similarity search retrieval against persistent Chroma vector store without LLM chains.",
    version="1.0.0"
)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Semantic search query string")
    top_k: int = Field(default=3, ge=1, le=50, description="Number of top chunks to retrieve")

    @field_validator("query")
    def validate_non_empty_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace.")
        return cleaned


class ChunkResult(BaseModel):
    chunk_text: str = Field(..., description="Raw text of the retrieved chunk")
    page_number: int = Field(..., description="1-indexed source PDF page number")
    score: float = Field(..., description="Raw similarity/distance score from vector store")


# ---------------------------------------------------------------------------
# Lazy-loaded Vector Store Instance
# ---------------------------------------------------------------------------
_vectorstore = None


def get_vectorstore_instance():
    global _vectorstore
    persist_path = Path(config.VECTORSTORE_DIR)

    if not persist_path.exists() or not any(persist_path.iterdir()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Vector store directory '{config.VECTORSTORE_DIR}' is empty or does not exist. "
                "Please run ingestion first using: python ingest.py --file <path_to_pdf>"
            )
        )

    if _vectorstore is None:
        embeddings = get_embedding_model()
        _vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=str(persist_path.resolve()),
            embedding_function=embeddings
        )

    return _vectorstore


# ---------------------------------------------------------------------------
# Single HTTP Retrieval Endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/query",
    response_model=List[ChunkResult],
    summary="Direct vector store similarity search",
    description="Queries the vector store using similarity_search_with_score() without any LLM chain."
)
def query_endpoint(request: QueryRequest) -> List[ChunkResult]:
    """
    Directly retrieves top-k raw matching chunks with their page numbers and vector store scores.
    Uses direct similarity search without LLM chains.
    """
    vs = get_vectorstore_instance()

    try:
        results_with_scores = vs.similarity_search_with_score(
            query=request.query,
            k=request.top_k
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store similarity search error: {str(exc)}"
        )

    response_chunks: List[ChunkResult] = []
    for doc, score in results_with_scores:
        page_num = doc.metadata.get("page_number")
        if page_num is None:
            raw_page = doc.metadata.get("page", 0)
            try:
                page_num = int(raw_page) + 1
            except (ValueError, TypeError):
                page_num = 1

        response_chunks.append(
            ChunkResult(
                chunk_text=doc.page_content,
                page_number=int(page_num),
                score=float(round(score, 6))
            )
        )

    return response_chunks


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

