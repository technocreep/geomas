from typing import Dict, List
from geomas.core.vision.encoder.encoder_config import create_encode
from geomas.core.repository.constant_repository import VISION_EMBEDDING_MODOEL
from geomas.core.repository.constant_repository import TEXT_EMBEDDING_MODOEL
import ast


class MultimodalEmbeddingProcessor:
    def __init__(self, clip_model:str = VISION_EMBEDDING_MODOEL, text_model:str =TEXT_EMBEDDING_MODOEL):
        self.clip_model = create_encode('clip', clip_model)
        self.text_model = create_encode('automodel', text_model)
    
    def text_embed(self, text_list):
        return self.text_model.encode(text_list)

    def process_geological_map(self, metadata: dict) -> Dict:
        """Обработка геологической карты"""
        # Извлечение базовых признаков

        # Генерация текстового описания через VLM
        
        # Извлечение структурной информации
        # structural_data = self._extract_structural_elements(image_path)
        image_paths = []
        for doc in metadata:
            img_path_str = doc.metadata.get("image_path", "[]")
            try:
                desc_list = ast.literal_eval(img_path_str)
                if isinstance(desc_list, list) and len(desc_list) > 0:
                    img_path = desc_list[0]
                else:
                    img_path = ""
            except (ValueError, SyntaxError):
                img_path = img_path_str.strip("[]").strip("'\"")
            image_paths.append(img_path)
        descriptions=[metadata_s.page_content for metadata_s in metadata]
        return {
            "image_embedding": self.clip_model.encode(image_paths),
            "description_embedding": self.text_embed(descriptions),
            # "structural_elements": structural_data,
            "metadata": metadata
        }
    
    def _extract_structural_elements(self, image_path: str) -> List[Dict]:
        """Выделение структурных элементов карты"""
        # Детекция контуров, разломов, зон минерализации
        # Используем традиционное CV + ML
        import cv2
        contours = cv2.findContours(...)
        faults = self._detect_faults(contours)
        mineralization_zones = self._detect_mineralization(contours)
        
        return {
            "faults": faults,
            "mineralization_zones": mineralization_zones,
            "contours": contours
        }