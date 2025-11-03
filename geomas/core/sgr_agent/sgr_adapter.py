"""
Adapter for integrating langchain LLM with SGR Deep Research framework.

This module provides adapters to bridge between geomas components
(langchain-based) and SGR Deep Research requirements.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SGRLLMAdapter:
    """
    Adapter to make langchain LLM compatible with SGR Deep Research.
    
    SGR expects specific LLM interface, this adapter translates
    langchain LLM calls to SGR-compatible format.
    """
    
    def __init__(self, langchain_llm):
        """
        Initialize adapter with langchain LLM.
        
        Args:
            langchain_llm: Langchain LLM instance
        """
        self.llm = langchain_llm
        self.logger = logger
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke LLM with prompt (SGR-compatible interface).
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            LLM response as string
        """
        try:
            response = self.llm.invoke(prompt, **kwargs)
            # Handle different response types
            if isinstance(response, str):
                return response
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {e}")
            return f"Error: {str(e)}"
    
    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of LLM responses
        """
        return [self.invoke(prompt, **kwargs) for prompt in prompts]
    
    def stream(self, prompt: str, **kwargs):
        """
        Stream LLM response (if supported).
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        if hasattr(self.llm, 'stream'):
            for chunk in self.llm.stream(prompt, **kwargs):
                yield chunk
        else:
            # Fallback to non-streaming
            yield self.invoke(prompt, **kwargs)


def create_sgr_agent(
    llm,
    schema: Dict[str, Any],
    bert_ner_model_path: Optional[str] = None,
    vector_db_service=None
):
    """
    Factory function to create SGR-based geological analysis agent.
    
    Args:
        llm: Langchain LLM instance
        schema: SGR schema dictionary
        bert_ner_model_path: Path to BERT NER model
        vector_db_service: Vector database service
        
    Returns:
        Configured GeologicalAnalysisAgent
    """
    from .geological_agent import GeologicalAnalysisAgent
    
    # Wrap langchain LLM for SGR compatibility
    llm_adapter = SGRLLMAdapter(llm)
    
    # Create LLM config for SGR
    llm_config = {
        "llm": llm_adapter,
        "model_type": "langchain_adapter"
    }
    
    # Initialize agent
    agent = GeologicalAnalysisAgent(
        llm_config=llm_config,
        bert_ner_model_path=bert_ner_model_path,
        vector_db_service=vector_db_service,
        use_sgr=True
    )
    
    return agent


def adapt_rag_context_for_sgr(
    retrieved_docs: list,
    reranked_docs: Optional[list] = None
) -> Dict[str, Any]:
    """
    Adapt RAG pipeline output for SGR agent input.
    
    Args:
        retrieved_docs: Documents from retrieval step
        reranked_docs: Optional reranked documents
        
    Returns:
        SGR-compatible context dictionary
    """
    docs_to_use = reranked_docs if reranked_docs else retrieved_docs
    
    return {
        "documents": docs_to_use,
        "document_count": len(docs_to_use),
        "contexts": [
            {
                "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "source": doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
            }
            for doc in docs_to_use
        ]
    }

