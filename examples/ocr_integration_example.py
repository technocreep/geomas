#!/usr/bin/env python3
"""
Example usage of integrated OCR module in geomas.

This example demonstrates how to use OCR functionality
within the geomas architecture.
"""

from pathlib import Path
from geomas.core.api.ocr import OcrApi, process_document_ocr
from geomas.core.repository.ocr_repository import get_ocr_adapter_names, load_ocr_config
from geomas.core.logger import get_logger

logger = get_logger()

def example_basic_usage():
    """Basic example of using OCR API."""
    logger.info("=== Basic OCR usage example ===")
    
    # Show available adapters
    adapters = get_ocr_adapter_names()
    logger.info(f"Available OCR adapters: {adapters}")
    
    # Load configuration
    config = load_ocr_config()
    logger.info(f"Loaded OCR configuration: {config.get('default_adapter', 'marker')}")
    
    # Example document processing (if file exists)
    test_file = Path("input/raw/test.pdf")
    if test_file.exists():
        logger.info(f"Processing file: {test_file}")
        try:
            result = process_document_ocr(test_file, adapter_name="marker")
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Processing error: {e}")
    else:
        logger.info(f"Test file not found: {test_file}")

def example_api_usage():
    """Example usage through OcrApi class."""
    logger.info("=== OcrApi class usage example ===")
    
    try:
        # Create API with specified adapter
        ocr_api = OcrApi(adapter_name="marker")
        logger.info(f"Created OCR API with adapter: {ocr_api.adapter_name}")
        
        # Example parameter configuration
        test_files = [
            Path("input/raw/document1.pdf"),
            Path("input/raw/document2.pdf")
        ]
        
        existing_files = [f for f in test_files if f.exists()]
        
        if existing_files:
            logger.info(f"Processing {len(existing_files)} files")
            result = ocr_api.process_documents(
                existing_files,
                output_dir="output/markdown",
                batch_size=4,
                language="ru"
            )
            logger.info(f"Created {len(result)} markdown files")
        else:
            logger.info("Test files not found")
            
    except Exception as e:
        logger.error(f"Error in API example: {e}")

def example_cli_simulation():
    """CLI commands usage simulation.""" 
    logger.info("=== CLI commands simulation ===")
    
    logger.info("To use OCR via CLI:")
    logger.info("  geomas ocr ./input/document.pdf --adapter marker")
    logger.info("  geomas ocr ./input/ --adapter mineru --batch-size 16")
    logger.info("  geomas ocr-adapters  # show available adapters")
    
def main():
    """Main function with examples."""
    logger.info("OCR integration demonstration in geomas")
    
    try:
        example_basic_usage()
        example_api_usage()
        example_cli_simulation()
        
        logger.info("=== OCR integration completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        
if __name__ == "__main__":
    main()
