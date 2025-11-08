from geomas.core.vision.encoder.pooling.pooling import Pooling
import torch

class MeanPooling(Pooling):
    @staticmethod
    def code_name() -> str:
        return "mean"
    
    def forward(self, token_embeddings: torch.Tensor, attention_mask:torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask