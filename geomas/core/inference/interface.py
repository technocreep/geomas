from typing import List

from geomas.core.inference.evaluation import Evaluator


class LlmConnector:

    def __init__(self, model_name: str, model_params: dict = None):
        self.model_name = model_name
        self.llm_model = Evaluator(model_name, **model_params)

    def invoke(self, query: List[str], inference_config: dict = None):
        result = self.llm_model.evaluate(prompts=query, inf_kwargs=inference_config)
        return result["response"]
