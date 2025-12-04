from typing import List, Union, Any
import requests
import base64
import os

class APIConnector:
    VLM_MODELS = {"llava", "llava-phi3", "qwen3-vl:2b"}

    def __init__(self, model_name: str, model_params: dict = None, base_url: str = "http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.base_url = base_url.rstrip("/")
        self.is_vlm = self._is_vision_model(model_name)

    def _is_vision_model(self, model_name: str) -> bool:
        return any(vlm in model_name.lower() for vlm in self.VLM_MODELS)

    def _encode_image(self, image_path: str) -> str:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def invoke(self, query: Union[List[str], List[Any]], inference_config: dict = None, image_path:str=None) -> List[str]:
        inf_config = {**self.model_params, **(inference_config or {})}
        responses = []

        for item in query:
            content = getattr(item, 'content', item) if hasattr(item, 'content') else item

            if self.is_vlm and isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype == "text":
                            text = part.get("text", "")
                            if text:
                                text_parts.append(text)
                    elif isinstance(part, str):
                        text_parts.append(part)
                prompt_text = " ".join(text_parts).strip()
            else:
                prompt_text = str(content).strip()

            message = {"role": "user", "content": prompt_text}
            if self.is_vlm:
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                message["images"] = [base64.b64encode(img_data).decode("utf-8")]

            payload = {
                "model": self.model_name,
                "messages": [message],
                "stream": False,
                **inf_config
            }
            try:
                resp = requests.post(
                    f"{self.base_url}",
                    json=payload,
                    timeout=1200
                )
                resp.raise_for_status()
                result = resp.json()
                responses.append(result.get("message", {}).get("content", "").strip())
            except requests.RequestException as e:
                error_detail = resp.text if 'resp' in locals() else ""
                raise RuntimeError(f"Ollama API error: {e} | Response: {error_detail}") from e

        return responses