from typing import List, Union, Any

from geomas.core.inference.evaluation import Evaluator


class LlmConnector:

    def __init__(self, model_name: str, model_params: dict = None):
        self.model_name = model_name
        self.llm_model = Evaluator(model_name, **model_params)

    def invoke(self, query: Union[List[str], List[Any]], inference_config: dict = None):
        """
        Invoke model with query.
        
        Args:
            query: List of strings or List of messages (e.g., HumanMessage objects)
            inference_config: Inference configuration parameters
            
        Returns:
            Result from model evaluation
        """
        # Handle HumanMessage objects (for vision models)
        if query and hasattr(query[0], 'content'):
            # Extract text from HumanMessage content
            prompts = []
            for msg in query:
                if hasattr(msg, 'content'):
                    # Extract text from content (could be list of dicts or string)
                    content = msg.content
                    if isinstance(content, list):
                        # Find text content
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        prompts.append(" ".join(text_parts) if text_parts else str(content))
                    elif isinstance(content, str):
                        prompts.append(content)
                    else:
                        prompts.append(str(content))
                else:
                    prompts.append(str(msg))
        else:
            # Already list of strings
            prompts = query if isinstance(query, list) else [query]
        
        result = self.llm_model.evaluate(prompts=prompts, inf_kwargs=inference_config)
        return result["response"] if isinstance(result, dict) else result
