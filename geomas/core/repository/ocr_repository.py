"""OCR repository for configurations and adapters.

This module follows the geomas repository pattern for managing
OCR-related configurations, constants, and adapter mappings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Type
from geomas.core.ocr_modules.models.base import BaseOCR
from geomas.core.logger import get_logger

logger = get_logger()

# OCR adapter registry following geomas pattern
OCR_ADAPTERS: Dict[str, Type[BaseOCR]] = {}

def _lazy_load_adapters():
    """Lazy load OCR adapters to avoid heavy imports at module level."""
    global OCR_ADAPTERS
    if not OCR_ADAPTERS:
        try:
            from geomas.core.ocr_modules.models.adapters.marker import Marker
            OCR_ADAPTERS["marker"] = Marker
        except ImportError:
            pass
            
        try:
            from geomas.core.ocr_modules.models.adapters.mineru import MinerU  
            OCR_ADAPTERS["mineru"] = MinerU
        except ImportError:
            pass
            
        try:
            from geomas.core.ocr_modules.models.adapters.olmocr import OlmOCR
            OCR_ADAPTERS["olmocr"] = OlmOCR
        except ImportError:
            pass
            
        try:
            from geomas.core.ocr_modules.models.adapters.qwen_vl import QwenVL
            OCR_ADAPTERS["qwen_vl"] = QwenVL  
        except ImportError:
            pass
    
    return OCR_ADAPTERS

def get_ocr_adapters() -> Dict[str, Type[BaseOCR]]:
    """Get available OCR adapters with lazy loading."""
    return _lazy_load_adapters()

# Default OCR configuration following geomas config pattern
OCR_DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size": 8,
    "language": "auto",
    "allow_network": False,
    "devices": None,
    "request_id": None
}

# Supported file extensions for OCR processing
OCR_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx", 
    ".png",
    ".jpg", 
    ".jpeg",
    ".tiff",
    ".bmp",
    ".txt",
    ".csv", 
    ".tsv",
    ".html",
    ".htm", 
    ".doc",
    ".docx",
    ".xls",
    ".xlsx"
}

# Default directory structure for OCR processing  
OCR_DEFAULT_DIRS = {
    "input": "input/raw",
    "output": "output/markdown", 
    "work": "work/ocr"
}

# OCR processing constants
OCR_CONSTANTS = {
    "DEFAULT_ADAPTER": "marker",
    "MAX_BATCH_SIZE": 32,
    "MIN_BATCH_SIZE": 1,
    "DEFAULT_TIMEOUT": 300,  # 5 minutes
    "RETRY_ATTEMPTS": 3
}

def get_default_ocr_config() -> Dict[str, Any]:
    """Get default OCR configuration."""
    return OCR_DEFAULT_CONFIG.copy()

def is_supported_file(file_path: str) -> bool:
    """Check if file extension is supported for OCR processing."""
    from pathlib import Path
    return Path(file_path).suffix.lower() in OCR_SUPPORTED_EXTENSIONS

def get_ocr_adapter_names() -> list[str]:
    """Get list of available OCR adapter names.""" 
    adapters = get_ocr_adapters()
    return list(adapters.keys())

def load_ocr_config(config_name: str = "ocr-config") -> Dict[str, Any]:
    """Load OCR configuration from YAML file.
    
    Args:
        config_name: Configuration file name without extension
        
    Returns:
        Configuration dictionary
    """
    from geomas.core.utils import CONFIG_PATH
    
    config_path = Path(CONFIG_PATH) / f"{config_name}.yaml"
    
    if not config_path.exists():
        logger.warning(f"OCR config file not found: {config_path}")
        return get_default_ocr_config()
        
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded OCR configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load OCR config from {config_path}: {e}")
        return get_default_ocr_config()
