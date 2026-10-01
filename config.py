"""
Central configuration for the PDF Vectorisation Pipeline.
Defines chunking hyperparameters, embedding model, and vector store persistence paths.
"""

from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Text Splitting Hyperparameters
# Chunk Size = 800 characters (~120-180 words): optimal semantic density for technical reports & specifications.
# Chunk Overlap = 150 characters (~18% overlap): prevents boundary cutoff without semantic dilution.
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150

# Embedding Integration
# Local Hugging Face sentence-transformers model (384-dimensional dense vectors).
# Runs locally on CPU/GPU without external API keys or network latency.
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Persistent Vector Store Directory
VECTORSTORE_DIR: str = str(BASE_DIR / "vectorstore")
COLLECTION_NAME: str = "pdf_documents"
