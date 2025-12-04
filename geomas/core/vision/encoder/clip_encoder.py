from geomas.core.vision.encoder.encoder import Encoder


class ClipEncoder(Encoder):
    @staticmethod
    def code_name() -> str:
        return "clip"

    def __init__(self, model_name, **kwargs):
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

    def encode(self, image_paths: list[str], batch_size: int = 32):
        from PIL import Image
        import torch
        import numpy as np
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            print(batch_paths)
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.clip_processor(
                images=images, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                batch_embeddings = self.clip_model.get_image_features(**inputs)
            all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
