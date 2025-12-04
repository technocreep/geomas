from geomas.core.vision.encoder.pooling.pooling import Pooling
import torch

class ClsPooling(Pooling):
    @staticmethod
    def code_name() -> str:
        return "cls"
    
    def forward(self, token_embeddings: torch.Tensor, attention_mask:torch.Tensor) -> torch.Tensor:
        return token_embeddings[:, 0, :]