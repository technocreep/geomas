from typing import Dict, List
import torch
from geomas.core.vision.encoder.encoder import Encoder
import logging
import numpy

logger = logging.getLogger(__name__)


class SentencetransformerEncoder(Encoder):
    @staticmethod
    def code_name() -> str:
        return "sentence_transformer"

    def __init__(self, model_name, **kwargs):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.max_length = kwargs.get("max_length", 1024)

    def _encode_image(self, texts: List[str], batch_size: int = 32):
        logging.info(
            f"Encoding {len(texts)} texts with batch size {batch_size} and max length {self.max_length}..."
        )
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=self.max_length,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            if not isinstance(embeddings, torch.Tensor):
                raise ValueError("Expected tensor output from model.encode")

            logging.info(
                f"Encoded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}."
            )

            return embeddings.cpu().numpy()
        except Exception as e:
            logging.error(f"Encoding failed: {str(e)}", exc_info=True)
            return numpy.array([])
