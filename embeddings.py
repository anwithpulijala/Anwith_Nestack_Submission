"""
Embedding helper module.
Ensures identical embedding configuration across ingestion and retrieval.
"""

from langchain_core.embeddings import Embeddings
from config import EMBEDDING_MODEL_NAME

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model() -> Embeddings:
    """
    Initializes and returns the shared LangChain HuggingFace embedding integration.
    Uses sentence-transformers/all-MiniLM-L6-v2 running offline on CPU.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

