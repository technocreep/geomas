from pathlib import Path
from typing import List, Union, Dict

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel, FastModel

from geomas.core.repository.config_repository import InferenceConfigTemplate



def is_pure_llm(model_name: str) -> bool:
    """Detect if a model is a pure LLM or multimodal."""
    multimodals = ["gemma"]
    name = model_name.lower()
    return not any(m in name for m in multimodals)


class Evaluator:
    def __init__(
        self,
        model_name: Union[Path, str],
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ):
        try:
            self.model_cls = FastLanguageModel if is_pure_llm(str(model_name)) else FastModel
            self.model, self.tokenizer = self.model_cls.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                dtype=None,
            )
            self.model_cls.for_inference(self.model)

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model {model_name}: {e}")

    def evaluate(self, prompts: List[str], inf_kwargs: dict) -> List[Dict[str, str]]:
        """Run inference for a list of prompts and return structured results."""
        results = []
        try:
            infer_config = InferenceConfigTemplate(**inf_kwargs)
            infer_config_dict = infer_config.__dict__ if hasattr(infer_config, "__dict__") else dict(infer_config)

            for prompt in prompts:
                inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
                text_streamer = TextStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )

                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        streamer=text_streamer,
                        **infer_config_dict,
                    )

                response = self.tokenizer.decode(output[0], skip_special_tokens=True)

                results.append({
                    "prompt": prompt,
                    "response": response,
                })
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
        return results


if __name__ == "__main__":
    prompts = [
        "Полупромышленные (заводские) технологические пробы служат для проверки", 
        "Метод ядерного гамма-резонанса (ЯГРМ) основан на эффекте Мессбауера (резонансе рассеянии гамма-квантов) и используется для"
        ]
    inf_kwargs = {"max_new_tokens": 64}
    evaluator = Evaluator("unsloth/Qwen3-14B-Base-unsloth-bnb-4bit")
    outputs = evaluator.evaluate(prompts, inf_kwargs)
    for out in outputs:
        print("Prompt:", out["prompt"])
        print("Response:", out["response"])
        print("=" * 20)
