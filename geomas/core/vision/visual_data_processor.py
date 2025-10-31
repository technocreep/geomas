"""
Visual Data Processor for geological maps and visual documents.

This module provides the main interface for processing visual geological data,
generating descriptions, and preparing them for integration into the RAG system.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.documents import Document

from geomas.core.logging.logger import get_logger
from geomas.core.vision.vlm_processor import VLMProcessor

logger = get_logger("VISUAL_DATA_PROCESSOR")


class VisualDataProcessor:
    """
    Main processor for visual geological data (maps, schemes, plans).
    
    This class handles:
    - Processing directories with visual documents
    - Generating textual descriptions via VLM
    - Creating Document objects for integration into RAG system
    """

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        model_url: Optional[str] = None,
        detailed_descriptions: bool = False,
        custom_prompt: Optional[str] = None
    ):
        """
        Initialize Visual Data Processor.
        
        Args:
            model_url: URL or path to Vision LLM model
            detailed_descriptions: If True, generates detailed descriptions
            custom_prompt: Custom prompt template for VLM descriptions
        """
        self.vlm_processor = VLMProcessor(model_url=model_url)
        self.detailed_descriptions = detailed_descriptions
        self.custom_prompt = custom_prompt
        logger.info("VisualDataProcessor initialized")

    def describe_image(self, image_path: str) -> str:
        """
        Generate textual description for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Textual description of the image
        """
        return self.vlm_processor.describe_image(
            image_path=image_path,
            prompt_template=self.custom_prompt,
            detailed=self.detailed_descriptions
        )

    def process_visual_documents(
        self,
        document_dir: str,
        source_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Process all visual documents in a directory and generate descriptions.
        
        This method scans the directory for supported image formats, generates
        textual descriptions for each image, and returns Document objects
        ready for indexing in ChromaDB.
        
        Args:
            document_dir: Path to directory containing images
            source_name: Name of the source document/collection (for metadata)
            metadata: Additional metadata to attach to documents
            
        Returns:
            List of Document objects with descriptions and metadata
        """
        document_path = Path(document_dir)
        if not document_path.exists():
            raise ValueError(f"Directory does not exist: {document_dir}")
        
        if not document_path.is_dir():
            raise ValueError(f"Path is not a directory: {document_dir}")

        documents = []
        image_files = self._get_image_files(document_path)
        
        if not image_files:
            logger.warning(f"No supported image files found in {document_dir}")
            return documents

        logger.info(f"Processing {len(image_files)} image(s) from {document_dir}")
        
        for img_path in image_files:
            try:
                logger.info(f"Processing image: {img_path.name}")
                
                # Generate description
                description = self.describe_image(str(img_path))
                
                # Prepare metadata
                doc_metadata = {
                    "source": source_name or document_path.name,
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "type": "visual",
                    "document_type": "geological_map"
                }
                
                # Add custom metadata if provided
                if metadata:
                    doc_metadata.update(metadata)
                
                # Create Document
                doc = Document(
                    page_content=description,
                    metadata=doc_metadata
                )
                documents.append(doc)
                
                logger.info(f"Successfully processed: {img_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                # Continue with next image instead of failing completely
                continue

        logger.info(f"Processed {len(documents)} out of {len(image_files)} images")
        return documents

    def _get_image_files(self, directory: Path) -> List[Path]:
        """
        Get all supported image files from directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            List of image file paths
        """
        image_files = []
        
        for ext in self.SUPPORTED_FORMATS:
            # Search for files with extension (case insensitive)
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        return image_files

    def process_single_image(
        self,
        image_path: str,
        source_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Document:
        """
        Process a single image and return Document object.
        
        Args:
            image_path: Path to the image file
            source_name: Name of the source document/collection
            metadata: Additional metadata
            
        Returns:
            Document object with description and metadata
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Processing single image: {image_path}")
        
        description = self.describe_image(str(img_path))
        
        doc_metadata = {
            "source": source_name or img_path.stem,
            "image_path": str(img_path),
            "image_name": img_path.name,
            "type": "visual",
            "document_type": "geological_map"
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        return Document(
            page_content=description,
            metadata=doc_metadata
        )

