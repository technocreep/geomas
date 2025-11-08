from typing import Dict, List
import torch
from geomas.core.vision.encoder.encoder import Encoder
from geomas.core.vision.encoder.pooling.pooling_config import create_pooling
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import logging

logger = logging.getLogger(__name__)
import numpy


class AutoModelEncoder(Encoder):
    @staticmethod
    def code_name() -> str:
        return "automodel"

    def __init__(self, model_name, **kwargs):
        self.max_length = kwargs.get("max_length", 1024)
        pool = kwargs.get("pooling", "mean")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )
        self.pooling = create_pooling(pool)
        self.tokenizer.add_eos_token = False

    def encode(self, texts: List[str], batch_size: int = 32) -> numpy.ndarray:
        logging.info(f"Encoding {len(texts)} texts...")
        embeddings = []
        for i in tqdm(
            range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"
        ):
            batch_texts = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            token_embeddings = model_output.last_hidden_state
            batch_embeddings = self.pooling(
                token_embeddings, encoded_input["attention_mask"]
            )
            embeddings.append(batch_embeddings.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        if embeddings is None:
            logging.error("Embeddings are None.")
        else:
            logging.info(f"Encoded {len(embeddings)} embeddings.")
        return embeddings.numpy()
