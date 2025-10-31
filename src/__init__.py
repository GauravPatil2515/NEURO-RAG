"""
NeuroRAG - Mental Health AI Assistant
Core package initialization
"""

from .rag_pipeline import RAGPipeline
from .utils import pdf_to_text

__version__ = "1.0.0"
__all__ = ["RAGPipeline", "pdf_to_text"]
