import logging

from langchain_core.documents import Document
from langchain_core.language_models import LLM

from geomas.core.rag_modules.steps.ranker import LLMReranker
from geomas.core.rag_modules.steps.retriever import Retriever, DocsSearcherModels, RetrievingPipeline
from geomas.core.repository.promts_repository import PROMPT_LLM_RESPONSE, PROMPT_RANK

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class RagApi:

    def __init__(self, llm: LLM):
        self.llm = llm
        self.multirag = False

    def _init_retriever(self, docs_searcher_models: DocsSearcherModels, top_k: int = 5) -> Retriever:
        """Documents retriever object."""
        return Retriever(top_k=top_k, docs_searcher_models=docs_searcher_models)

    def _init_ranker_model(self):
        self.reranker = LLMReranker(self.llm, PROMPT_RANK)

    def _retrieve(self):
        logger.info('Retrieving ----------- IN PROGRESS')
        context = RetrievingPipeline() \
            .set_retrievers(retrievers) \
            .set_collection_names(collection_names) \
            .get_retrieved_docs(user_prompt)
        logger.info('Retrieving ----------- DONE')
        return context

    def _rerank(self,context, user_prompt, rerank: bool = True):
        # Step 2. Ranking
        response = context
        if rerank:
            logger.info('Reranking ----------- IN PROGRESS')
            self._init_ranker_model()
            response = self.reranker.rerank_context(context, user_prompt)
            logger.info('Reranking ----------- DONE')
        else:
            logger.info('Reranking ----------- SKIPPED')

        return response

    def _merge_output(self, response, user_prompt):
        if self.multirag:
            # Merge the most relevant paragraphs
            logger.info('Merging ----------- IN PROGRESS')
            context = self.reranker.merge_docs(user_prompt, response)
        logger.info('Generation ----------- IN PROGRESS')
        paragraphs = "\n".join([f"Параграф {i + 1}: {doc.page_content}" for i, doc in enumerate(response)])
        llm_response = self.llm.invoke(PROMPT_LLM_RESPONSE.format(paragraphs=paragraphs, question=user_prompt))
        logger.info('Generation ----------- DONE')
        return llm_response

    def eval(self,
             user_prompt: str,
             retrievers: list[Retriever],
             collection_names: list[str],
             retriever_pipelines: list[RetrievingPipeline] = None,
             rerank: bool = False) -> str:
        """
        :param user_prompt: only prompt that was received from user
        :param collection_names: retriever gets docs from the collection with name 'collection_name'
        :param rerank: if reranking is necessary
        :return: response from LLM
        """
        if retriever_pipelines is not None:
            self.multirag = True

        # Step 1. Retrieve
        context = self._retrieve()

        # Step 2. Ranking
        response = self._rerank(context, user_prompt, rerank)

        # Step 3. Get output
        response = self._merge_output(response,user_prompt)

        return response

    def run_multiple_rag(self,
                         user_prompt: str,
                         llm: LLM,
                         retriever_pipelines: list[RetrievingPipeline]) -> str:
        """
        :param user_prompt: only prompt that was received from user
        :param retriever_pipelines: ready to use retriever pipelines (retrievers and collection_names should be specified)
        :param do_reranking: if reranking is necessary
        :return: response from LLM
        """
        reranker = LLMReranker(llm, PROMPT_RANK)

        # Retrieve
        logger.info('Retrieving ----------- IN PROGRESS')
        contexts = [pipeline.get_retrieved_docs(user_prompt) for pipeline in retriever_pipelines]
        logger.info('Retrieving ----------- DONE')

        max_len_context = max([len(context) for context in contexts])
        for ctx in contexts:
            if len(ctx) < max_len_context:
                ctx.extend([Document(page_content='')] * (max_len_context - len(ctx)))




        return llm_response
