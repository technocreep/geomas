"""
VLM Processor for geological map descriptions.

This module handles Vision-Language Model inference for generating
textual descriptions of geological maps and visual data.
"""

import base64
import os
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

from langchain_core.messages import HumanMessage

from geomas.core.logging.logger import get_logger
from geomas.core.repository.constant_repository import VISION_LLM_URL

if TYPE_CHECKING:
    from geomas.core.inference.interface import LlmConnector

logger = get_logger("VLM_PROCESSOR")


class VLMProcessor:
    """
    Vision-Language Model processor for generating textual descriptions of images.
    
    Supports geological maps, schemes, and other visual geological data.
    """

    def __init__(
        self,
        model_url: Optional[str] = None,
        inference_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize VLM processor.
        
        Args:
            model_url: URL or path to the Vision LLM model. Defaults to VISION_LLM_URL.
            inference_params: Inference parameters (temperature, top_p, etc.)
        """
        self.model_url = model_url or VISION_LLM_URL
        self.inference_params = inference_params or {
            "temperature": 0.015,
            "top_p": 0.95
        }
        self._llm_model: Optional["LlmConnector"] = None
        logger.info(f"Initializing VLM processor with model: {self.model_url}")

    def _init_model(self):
        """Lazy initialization of the Vision LLM model."""
        if self._llm_model is None:
            try:
                # Check if model_url is a file path that contains the actual URL
                model_url = self.model_url
                if model_url.startswith('./') or os.path.exists(model_url):
                    # Try to read URL from file
                    if os.path.exists(model_url):
                        with open(model_url, 'r', encoding='utf-8') as f:
                            model_url = f.read().strip()
                            logger.info(f"Read Vision LLM URL from file: {model_url}")
                
                # Lazy import to avoid importing unsloth/triton at module load time
                from geomas.core.inference.interface import LlmConnector
                self._llm_model = LlmConnector(model_url)
                logger.info("Vision LLM model initialized successfully")
            except ImportError as e:
                if "triton" in str(e) or "libtriton" in str(e):
                    error_msg = (
                        f"Failed to initialize Vision LLM model due to Triton DLL error.\n"
                        f"This is a known issue with unsloth/triton on Windows.\n"
                        f"\n"
                        f"Solutions:\n"
                        f"1. Use an external Vision LLM API (e.g., OpenAI GPT-4V, Anthropic Claude)\n"
                        f"2. Configure VISION_LLM_URL to point to an API endpoint instead of a local model\n"
                        f"3. Set up Vision LLM as a separate service\n"
                        f"\n"
                        f"Current VISION_LLM_URL: {self.model_url}\n"
                        f"Error: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                else:
                    logger.error(f"Failed to initialize Vision LLM model: {e}")
                    raise
            except Exception as e:
                error_msg = (
                    f"Failed to initialize Vision LLM model: {e}\n"
                    f"Please check VISION_LLM_URL configuration: {self.model_url}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_string

    def _get_image_format(self, image_path: str) -> str:
        """Detect image format from file extension."""
        ext = Path(image_path).suffix.lower()
        format_map = {
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
            ".gif": "gif",
            ".bmp": "bmp"
        }
        return format_map.get(ext, "jpeg")

    def describe_image(
        self,
        image_path: str,
        prompt_template: Optional[str] = None,
        detailed: bool = False
    ) -> str:
        """
        Generate textual description of a geological map or visual data.
        
        Args:
            image_path: Path to the image file (geological map, scheme, etc.)
            prompt_template: Custom prompt template. If None, uses default geological prompt.
            detailed: If True, generates more detailed description.
            
        Returns:
            Textual description of the image.
        """
        self._init_model()
        
        # Check if image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Use default geological prompt if not provided
        if prompt_template is None:
            if detailed:
                sys_prompt = (
                    "Это геологическая карта или схема из геологического документа. "
                    "Опиши детально все видимые элементы: типы пород, структуры, разломы, "
                    "зоны минерализации, геологические объекты, легенду карты, масштаб, "
                    "направления и любые другие геологические особенности. "
                    "Будь максимально подробным и точным. "
                    "Используй только информацию, видимую на изображении, ничего не выдумывай."
                )
            else:
                sys_prompt = (
                    "Это геологическая карта или схема из геологического документа. "
                    "Опиши кратко основные элементы: типы пород, структуры, разломы, "
                    "геологические объекты и другие особенности. "
                    "Будь лаконичным, но информативным. "
                    "Используй только информацию, видимую на изображении."
                )
        else:
            sys_prompt = prompt_template
        
        try:
            # Encode image to base64
            image_base64 = self._image_to_base64(image_path)
            image_format = self._get_image_format(image_path)
            
            # Create message with image
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": sys_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{image_base64}"
                            }
                        }
                    ]
                )
            ]
            
            # Invoke model
            logger.info(f"Generating description for image: {image_path}")
            result = self._llm_model.invoke(messages, inference_config=self.inference_params)
            
            description = result.get("response", "") if isinstance(result, dict) else str(result)
            logger.info(f"Description generated successfully for: {Path(image_path).name}")
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating description for {image_path}: {e}")
            raise

