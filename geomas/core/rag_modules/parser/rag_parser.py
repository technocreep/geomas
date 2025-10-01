import logging
import os
import re
import shutil
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from geomas.core.data.s3_data import S3BucketService
from geomas.core.inference.interface import LlmConnector
from geomas.core.rag_modules.steps.chunking import TextChunker
from geomas.core.repository.constant_repository import USE_S3, VISION_LLM_URL
from geomas.core.repository.parsing_repository import ParsingPatternConfig

_log = logging.getLogger(__name__)


class DocumentParser:

    def __init__(self, chunking_params: dict = None):
        self.vision_llm = LlmConnector(VISION_LLM_URL)
        self.chunking_agent = TextChunker(chunking_params)

    def _llm_image_to_text(self, current_img, local_img_path, images):
        cls_prompt = None
        prompt_func = None
        query = None
        res_1 = self.vision_llm.invoke(query)
        if res_1.strip() == "False":
            parent_p = current_img.find_parent('p')
            if parent_p:
                parent_p.decompose()
                os.remove(local_img_path)
        else:
            table_extraction_prompt = None
            res_2 = self.vision_llm.invoke(table_query)
            if res_2.strip() != "No table":
                match = re.search(ParsingPatternConfig.table_pattern, res_2, re.DOTALL)
                if match:
                    html_table = match.group(0)
                    table_soup = BeautifulSoup(html_table, 'html.parser')
                    parent_p = current_img.find_parent('p')
                    if parent_p:
                        parent_p.replace_with(table_soup)
                        os.remove(local_img_path)
        return images

    def _saving_after_preprocess(self, file_name: str, save_dir: str):
        new_path = str(Path(save_dir, f"{file_name}_processed.html"))
        with open(new_path, "w", encoding='utf-8') as file:
            file.write(str(self.soup.prettify()))

        # Possible integration with S3
        # if s3_service and paper_s3_prefix:
        #     s3_service.upload_file_object(paper_s3_prefix, new_file_name, new_path)

    def _preprocess_headers_with_bf(self, html):
        self.soup = BeautifulSoup(html, "lxml")
        for header in self.soup.find_all(["h1", "h2", "h3"]):
            header_text = header.get_text(strip=True).lower()

            if any(exclude in header_text for exclude in ParsingPatternConfig.ignored_topics):
                next_node = header.next_sibling
                elements_to_remove = []
                while next_node and next_node.name not in ["h1", "h2"]:
                    elements_to_remove.append(next_node)
                    next_node = next_node.next_sibling

                header.decompose()
                for element in elements_to_remove:
                    if isinstance(element, Tag):
                        element.decompose()

    def _preprocess_img_with_bf(self, save_dir):
        image_url_mapping = {}
        for img in self.soup.find_all('img'):
            img_src = img.get("src")
            if not img_src:
                continue

            local_img_path = str(Path(save_dir) / img_src)
            try:
                images = list(map(convert_to_base64, [local_img_path]))
            except OSError as e:
                if e.errno == 2:
                    print(f"File not found: {e}")
                    continue
                else:
                    print(f"Error from OS: {e}")
                    continue
            images = self._llm_image_to_text(img, local_img_path, images)
            image_url_mapping[local_img_path] = local_img_path

            # Possible integration with S3
            # elif s3_service and paper_s3_prefix:
            #     s3_key = f"{paper_s3_prefix}/{img_src}"
            #     s3_service.upload_file_object(paper_s3_prefix, img_src, local_img_path)
            #     s3_url = f"{s3_service.endpoint.rstrip('/')}/{s3_service.bucket_name}/{s3_key}"
            #     img['src'] = s3_url
            #     image_url_mapping[local_img_path] = s3_url
            # else:
            #     image_url_mapping[local_img_path] = local_img_path
        return image_url_mapping

    def preprocessing(self,
                      file_name: str,
                      save_dir: Path,
                      raw_text: str) -> (str, dict):
        """
        Cleans up HTML content by removing irrelevant sections like acknowledgements and references,
        and processes images to either remove them or replace them with extracted tables.

        Args:
            save_dir (Path): The directory containing the document.
            file_name (Path): The name of the HTML file.
            raw_text (str): The HTML content as a string.

        Returns:
            str: The cleaned content as a string, potentially with images replaced by tables.
        """
        self._preprocess_headers_with_bf(raw_text)
        image_url_mapping = self._preprocess_img_with_bf(save_dir)
        self._saving_after_preprocess(file_name, save_dir)
        return self.soup.prettify(), image_url_mapping

    def parse(self, raw_text: str, document_name: str, document_type: str):
        chunked_text = self.chunking_agent.apply_chunking(raw_text, document_name, document_type)
        return chunked_text

    def postprocessing(self, doc_dir: str) -> None:
        if os.path.exists(doc_dir):
            try:
                shutil.rmtree(doc_dir)
                print(f"Directory '{doc_dir}' and its contents removed successfully.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"Directory '{doc_dir}' does not exist.")
