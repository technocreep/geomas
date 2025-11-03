"""
Enhanced RAG API with SGR (Schema-Guided Reasoning) Integration.

This module extends the original RagApi with geological analysis capabilities
using SGR Deep Research framework for structured reasoning.
"""

import logging
from copy import deepcopy
from typing import Optional, List, Dict, Any

from langchain_core.documents import Document
from langchain_core.language_models import LLM

from geomas.core.rag_modules.database.dataloader import load_documents_to_chroma_db
from geomas.core.rag_modules.steps.ranker import LLMReranker
from geomas.core.rag_modules.steps.retriever import (
    DocsSearcherModels,
    Retriever,
    RetrievingPipeline,
)
from geomas.core.repository.database_repository import chroma_default_settings
from geomas.core.repository.promts_repository import PROMPT_LLM_RESPONSE, PROMPT_RANK
from geomas.core.repository.geological_prompts import (
    GEOLOGICAL_EXPERT_SYSTEM_PROMPT,
    PROMPT_GEOLOGICAL_GENERAL_QUERY,
    PROMPT_GEOLOGICAL_RESOURCE_ASSESSMENT,
    PROMPT_GEOLOGICAL_RISK_ANALYSIS,
    PROMPT_GEOLOGICAL_ECONOMIC_VIABILITY,
    PROMPT_GEOLOGICAL_RERANK,
    format_entities_summary,
    format_context_paragraphs,
)

from geomas.core.sgr_agent import (
    GeologicalAnalysisAgent,
    SGRLLMAdapter,
    EntityProcessor,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class RagSGRApi:
    """
    Enhanced RAG API with SGR geological analysis capabilities.
    
    Extends traditional RAG pipeline with:
    - Geological entity extraction (BERT NER)
    - Schema-guided reasoning (SGR)
    - Structured geological analysis
    - Domain-specific prompts
    """
    
    def __init__(
        self,
        llm: LLM,
        use_sgr: bool = True,
        bert_ner_model_path: Optional[str] = None,
        use_geological_prompts: bool = True
    ):
        """
        Initialize Enhanced RAG API with SGR support.
        
        Args:
            llm: Langchain LLM instance
            use_sgr: Whether to use SGR framework for structured reasoning
            bert_ner_model_path: Path to BERT NER model for entity extraction
            use_geological_prompts: Whether to use geological-specific prompts
        """
        self.llm = llm
        self.multirag = False
        self.use_sgr = use_sgr
        self.use_geological_prompts = use_geological_prompts
        
        # Initialize SGR Agent if enabled
        self.sgr_agent = None
        if use_sgr:
            try:
                self.sgr_agent = GeologicalAnalysisAgent(
                    llm_config={"llm": SGRLLMAdapter(llm)},
                    bert_ner_model_path=bert_ner_model_path,
                    use_sgr=True
                )
                logger.info("SGR Geological Agent initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize SGR agent: {e}")
                logger.warning("Falling back to traditional RAG mode")
                self.use_sgr = False
        
        # Initialize Entity Processor
        self.entity_processor = EntityProcessor()
        
        # Initialize reranker (will be set up in _init_ranker_model)
        self.reranker = None
    
    def _load_chroma_db(self, path: str, collection: str) -> None:
        """Loads data to ChromaDB."""
        chroma_default_settings.collection_name = collection
        chroma_default_settings.docs_collection_path = path
        processing_batch_size = 32
        loading_batch_size = 32
        settings = deepcopy(chroma_default_settings)
        load_documents_to_chroma_db(
            settings=settings,
            processing_batch_size=processing_batch_size,
            loading_batch_size=loading_batch_size,
        )
    
    def _init_retriever(self, docs_searcher_models: DocsSearcherModels, top_k: int = 5) -> Retriever:
        """Documents retriever object."""
        return Retriever(top_k=top_k, docs_searcher_models=docs_searcher_models)
    
    def _init_ranker_model(self, use_geological: bool = False):
        """Initialize ranker with appropriate prompt."""
        rank_prompt = PROMPT_GEOLOGICAL_RERANK if (use_geological and self.use_geological_prompts) else PROMPT_RANK
        self.reranker = LLMReranker(self.llm, rank_prompt)
    
    def _retrieve(self, user_prompt, retrievers, collection_names, retriever_pipelines):
        """Retrieve relevant documents from vector database."""
        logger.info('Retrieving ----------- IN PROGRESS')
        if self.multirag:
            contexts = [pipeline.get_retrieved_docs(user_prompt) for pipeline in retriever_pipelines]
            logger.info('Retrieving ----------- DONE')
            max_len_context = max([len(context) for context in contexts])
            [context.extend([Document(page_content='')] * (max_len_context - len(context)))
             for context in contexts if len(context) < max_len_context]
            return contexts
        else:
            context = RetrievingPipeline() \
                .set_retrievers(retrievers) \
                .set_collection_names(collection_names) \
                .get_retrieved_docs(user_prompt)
            logger.info('Retrieving ----------- DONE')
            return context
    
    def _rerank(self, context, user_prompt, rerank: bool = True, use_geological: bool = False):
        """Rerank retrieved documents."""
        response = context
        if rerank:
            logger.info('Reranking ----------- IN PROGRESS')
            self._init_ranker_model(use_geological=use_geological)
            response = self.reranker.rerank_context(context, user_prompt)
            logger.info('Reranking ----------- DONE')
        else:
            logger.info('Reranking ----------- SKIPPED')
        
        return response
    
    def _extract_entities(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract geological entities from documents."""
        if not self.sgr_agent or not self.sgr_agent.ner_model:
            logger.info('Entity extraction ----------- SKIPPED (NER model not available)')
            return None
        
        logger.info('Entity extraction ----------- IN PROGRESS')
        entities = self.sgr_agent._extract_entities_from_context(documents)
        
        # Process entities
        processed = self.entity_processor.process_entities(entities)
        logger.info(f'Entity extraction ----------- DONE ({processed["entity_count"]} entities found)')
        
        return processed
    
    def _select_analysis_type(self, user_prompt: str, entities: Optional[Dict] = None) -> str:
        """Auto-detect analysis type from query and entities."""
        if self.sgr_agent:
            return self.sgr_agent._detect_analysis_type(user_prompt)
        return "general"
    
    def _generate_response_traditional(
        self,
        response: List[Document],
        user_prompt: str,
        entities: Optional[Dict] = None
    ) -> str:
        """Generate response using traditional RAG approach."""
        logger.info('Generation ----------- IN PROGRESS')
        
        if self.use_geological_prompts and entities:
            # Use geological prompt with entities
            entities_summary = format_entities_summary(entities)
            context_paragraphs = format_context_paragraphs(response)
            
            llm_response = self.llm.invoke(
                PROMPT_GEOLOGICAL_GENERAL_QUERY.format(
                    question=user_prompt,
                    entities_summary=entities_summary,
                    context_paragraphs=context_paragraphs
                )
            )
        else:
            # Use original prompt
            paragraphs = "\n".join([
                f"Параграф {i + 1}: {doc.page_content}"
                for i, doc in enumerate(response)
            ])
            llm_response = self.llm.invoke(
                PROMPT_LLM_RESPONSE.format(paragraphs=paragraphs, question=user_prompt)
            )
        
        logger.info('Generation ----------- DONE')
        return llm_response
    
    def _generate_response_sgr(
        self,
        response: List[Document],
        user_prompt: str,
        entities: Optional[Dict] = None,
        analysis_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using SGR framework."""
        logger.info('SGR Analysis ----------- IN PROGRESS')
        
        # Auto-detect analysis type if not specified
        if not analysis_type:
            analysis_type = self._select_analysis_type(user_prompt, entities)
        
        logger.info(f'Analysis type: {analysis_type}')
        
        # Execute SGR-guided analysis
        result = self.sgr_agent.analyze_general(
            query=user_prompt,
            context_documents=response,
            analysis_type=analysis_type
        )
        
        # Format response with entities
        if entities:
            result["entities_extracted"] = entities
            result["entity_summary"] = self.entity_processor.create_entity_summary(entities)
        
        logger.info('SGR Analysis ----------- DONE')
        return result
    
    def _merge_output(self, response, user_prompt, entities=None):
        """Merge and generate final output."""
        if self.multirag:
            # Merge the most relevant paragraphs
            logger.info('Merging ----------- IN PROGRESS')
            response = self.reranker.merge_docs(user_prompt, response)
            logger.info('Merging ----------- DONE')
        
        # Generate response based on mode
        if self.use_sgr and self.sgr_agent:
            return self._generate_response_sgr(response, user_prompt, entities)
        else:
            return self._generate_response_traditional(response, user_prompt, entities)
    
    def eval(
        self,
        user_prompt: str,
        retrievers: List[Retriever],
        collection_names: List[str],
        retriever_pipelines: Optional[List[RetrievingPipeline]] = None,
        rerank: bool = False,
        use_entities: bool = True,
        analysis_type: Optional[str] = None
    ) -> Any:
        """
        Execute RAG pipeline with optional SGR analysis.
        
        Args:
            user_prompt: User question/query
            retrievers: List of retriever objects
            collection_names: Collection names for document retrieval
            retriever_pipelines: Optional pre-configured pipelines
            rerank: Whether to rerank retrieved documents
            use_entities: Whether to extract geological entities
            analysis_type: Optional explicit analysis type (resource_assessment, risk_analysis, etc.)
            
        Returns:
            str or Dict: Response (string for traditional mode, dict for SGR mode)
        """
        if retriever_pipelines is not None:
            self.multirag = True
        
        # Step 1. Retrieve
        context = self._retrieve(user_prompt, retrievers, collection_names, retriever_pipelines)
        
        # Step 2. Reranking
        use_geological_rerank = self.use_geological_prompts and self.use_sgr
        response = self._rerank(context, user_prompt, rerank, use_geological=use_geological_rerank)
        
        # Step 3. Entity Extraction (if enabled)
        entities = None
        if use_entities and self.sgr_agent and self.sgr_agent.ner_model:
            entities = self._extract_entities(response)
        
        # Step 4. Generate output
        final_response = self._merge_output(response, user_prompt, entities)
        
        return final_response
    
    def eval_structured(
        self,
        user_prompt: str,
        retrievers: List[Retriever],
        collection_names: List[str],
        analysis_type: str,
        retriever_pipelines: Optional[List[RetrievingPipeline]] = None,
        rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Execute structured geological analysis with explicit schema type.
        
        Args:
            user_prompt: User query
            retrievers: List of retriever objects
            collection_names: Collection names
            analysis_type: Explicit analysis type (resource_assessment, risk_analysis, economic_viability)
            retriever_pipelines: Optional pipelines
            rerank: Whether to rerank
            
        Returns:
            Structured analysis results
        """
        logger.info(f"Executing structured analysis: {analysis_type}")
        
        if not self.use_sgr or not self.sgr_agent:
            raise ValueError("SGR mode must be enabled for structured analysis")
        
        # Execute with explicit analysis type
        return self.eval(
            user_prompt=user_prompt,
            retrievers=retrievers,
            collection_names=collection_names,
            retriever_pipelines=retriever_pipelines,
            rerank=rerank,
            use_entities=True,
            analysis_type=analysis_type
        )

