from langchain_text_splitters import HTMLSemanticPreservingSplitter, MarkdownTextSplitter
import os
import re

from geomas.core.rag_modules.parser.rag_parser import PARSE_RESULTS_PATH
from geomas.core.repository.constant_repository import USE_S3
from geomas.core.repository.parsing_repository import ChunkingParamsConfig,ParsingPatternConfig


class TextChunker:

    def __init__(self, chunking_params: dict = None):
        self.splitter_dict = {'html': HTMLSemanticPreservingSplitter,
                              'markdown': MarkdownTextSplitter,
                              }
        self.chunking_params = chunking_params if chunking_params is not None else ChunkingParamsConfig.to_dict()
        pass

    def _custom_table_extractor(self, table_tag):
        return str(table_tag).replace("\n", "")

    def extract_img_url(self, doc_text: str, p_name: str) -> list[str]:
        """
        Extracts image URLs from a document text related to scientific papers.

        This method identifies image references within the text and constructs their full paths for access.
        It focuses on JPEG images specifically referenced using a specific markdown-like syntax.

        Args:
            doc_text: The text of the scientific document to analyze.
            p_name: The name of the project or paper, used to organize image paths.

        Returns:
            list: A list of strings, where each string is a full path to an image extracted from the document,
            constructed using the project path and the image filename.
        """
        matches = re.findall(ParsingPatternConfig.image_pattern, doc_text)
        return [entry[0] for entry in matches] if USE_S3 \
            else [os.path.join(PARSE_RESULTS_PATH, p_name, entry[0]) for entry in matches]

    def apply_chunking(self, raw_text: str, document_name: str, document_type: str):
        splitter = self.splitter_dict[document_type](**self.chunking_params,
                                                     custom_handlers={"table": self._custom_table_extractor})
        documents = splitter.split_text(raw_text)
        for doc in documents:
            doc.page_content = "passage: " + doc.page_content  # Maybe delete "passage: " addition
            doc.metadata["imgs_in_chunk"] = str(self.extract_img_url(doc.page_content, document_name))
            doc.metadata["source"] = document_name + ".pdf"
        return documents
