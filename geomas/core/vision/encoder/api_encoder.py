from typing import Dict, List
import torch
import requests
import time
from tqdm import tqdm
from math import ceil
import numpy
from geomas.core.vision.encoder.encoder import Encoder
import logging
logger = logging.getLogger(__name__)


class ApiEncoder(Encoder):
    @staticmethod
    def code_name() -> str:
        return "api"

    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.headers = kwargs.get("headers", {"Content-Type": "application/json"})
        self.max_retries = kwargs.get("max_retries", 3)
        self.retry_delay = kwargs.get("retry_delay", 5)
        self.max_length = kwargs.get("max_length", 1024)

    def get_embedding(self, texts: list):
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = requests.post(
                    self.model_name, headers=self.headers, json={"input": texts}
                )
                response.raise_for_status()
                embeddings = [
                    data["embedding"] for data in response.json().get("data", [])
                ]
                return numpy.array(embeddings)

            except requests.exceptions.RequestException as e:
                attempt += 1
                logger.error(
                    f"Error during API request (Attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries:
                    logger.error("Max retries reached. Returning empty tensor.")
                    return numpy.array([])
                time.sleep(self.retry_delay)
            except (KeyError, TypeError, ValueError) as e:
                logger.error(f"Error processing API response: {e}")
                return numpy.array([])

    def encode(self, texts: list[str], batch_size: int = 32):
        texts = [text[: self.max_length] for text in texts]
        num_batches = ceil(len(texts) / batch_size)
        all_embeddings = []
        for batch_idx in tqdm(range(num_batches)):
            batch = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            claims_embeddings = self.get_embedding(batch)
            all_embeddings.append(claims_embeddings)
        embeddings = numpy.vstack(all_embeddings)
        return embeddings
    