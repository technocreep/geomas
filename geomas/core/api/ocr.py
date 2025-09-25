"""OCR API integration for geomas.

This module provides a high-level interface to the OCR functionality,
following geomas architectural patterns and conventions.
"""

from pathlib import Path
from typing import List, Optional

from geomas.core.logger import get_logger
from geomas.core.ocr_modules.api import process_path as _process_path
from geomas.core.ocr_modules.api import process_paths as _process_paths
from geomas.core.repository.ocr_repository import get_ocr_adapters, get_default_ocr_config

logger = get_logger()


class OcrApi:
    """High-level OCR API following geomas patterns."""
    
    def __init__(self, adapter_name: str = "marker"):
        """Initialize OCR API with specified adapter.
        
        Args:
            adapter_name: Name of OCR adapter to use
        """
        self.adapter_name = adapter_name
        self.adapter = self._create_adapter(adapter_name)
        
    def _create_adapter(self, adapter_name: str):
        """Create OCR adapter instance by name."""
        adapters = get_ocr_adapters()
        adapter_class = adapters.get(adapter_name)
        if not adapter_class:
            available = list(adapters.keys())
            logger.error(f"Unknown OCR adapter: {adapter_name}. Available: {available}")
            raise ValueError(f"Unknown OCR adapter: {adapter_name}")
            
        return adapter_class()
    
    def process_document(
        self,
        path: Path | str,
        *,
        input_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        work_dir: Optional[Path | str] = None,
        **kwargs
    ) -> List[Path]:
        """Process a single document through OCR pipeline.
        
        Args:
            path: Path to document to process
            input_dir: Input directory override
            output_dir: Output directory override  
            work_dir: Working directory override
            **kwargs: Additional processing options
            
        Returns:
            List of generated markdown file paths
        """
        logger.info(f"Processing document {path} with {self.adapter_name} adapter")
        
        config = {**get_default_ocr_config(), **kwargs}
        
        try:
            result = _process_path(
                path,
                self.adapter,
                input_dir=input_dir,
                output_dir=output_dir,
                work_dir=work_dir,
                **config
            )
            logger.info(f"Successfully processed {path}, generated {len(result)} files")
            return result
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            raise
            
    def process_documents(
        self,
        paths: List[Path | str],
        *,
        input_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        work_dir: Optional[Path | str] = None,
        **kwargs
    ) -> List[Path]:
        """Process multiple documents through OCR pipeline.
        
        Args:
            paths: List of paths to documents to process
            input_dir: Input directory override
            output_dir: Output directory override
            work_dir: Working directory override
            **kwargs: Additional processing options
            
        Returns:
            List of generated markdown file paths
        """
        logger.info(f"Processing {len(paths)} documents with {self.adapter_name} adapter")
        
        config = {**get_default_ocr_config(), **kwargs}
        
        try:
            result = _process_paths(
                paths,
                self.adapter,
                input_dir=input_dir,
                output_dir=output_dir,
                work_dir=work_dir,
                **config
            )
            logger.info(f"Successfully processed {len(paths)} documents, generated {len(result)} files")
            return result
        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            raise


# Convenience functions for direct usage
def process_document_ocr(
    path: Path | str,
    adapter_name: str = "marker",
    **kwargs
) -> List[Path]:
    """Process a single document with OCR.
    
    Args:
        path: Document path
        adapter_name: OCR adapter to use
        **kwargs: Additional options
        
    Returns:
        List of markdown file paths
    """
    api = OcrApi(adapter_name)
    return api.process_document(path, **kwargs)


def process_documents_ocr(
    paths: List[Path | str], 
    adapter_name: str = "marker",
    **kwargs
) -> List[Path]:
    """Process multiple documents with OCR.
    
    Args:
        paths: List of document paths
        adapter_name: OCR adapter to use 
        **kwargs: Additional options
        
    Returns:
        List of markdown file paths
    """
    api = OcrApi(adapter_name)
    return api.process_documents(paths, **kwargs)
