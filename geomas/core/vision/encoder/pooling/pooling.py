import abc
import torch
class Pooling(abc.ABC, torch.nn.Module):
    def code_name() -> str:
        raise NotImplementedError("abstract")        
    
    def forward(self, token_embeddings: torch.Tensor, attention_mask:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("abstract")        

    