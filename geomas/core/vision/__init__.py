"""
Vision module for processing geological maps and visual data.

This module provides functionality to:
- Generate textual descriptions of geological maps using VLM models
- Process visual documents and integrate them into the RAG system
- Store visual data descriptions in ChromaDB for semantic search
"""

from geomas.core.vision.visual_data_processor import VisualDataProcessor
from geomas.core.vision.vlm_processor import VLMProcessor

__all__ = ["VisualDataProcessor", "VLMProcessor"]

