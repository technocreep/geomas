import os
from pathlib import Path

from langchain_core.document_loaders import BaseLoader

from geomas.core.data.custom_dataloaders import (
    LangChainDocumentLoader,
    PDFLoader,
    RecursiveDirectoryLoader,
    WordDocumentLoader,
    ZipLoader,
)
from geomas.core.repository.parsing_repository import DataTypeLoaderConfig as LoaderType


def get_loader(**loader_params) -> BaseLoader:
    document_path = loader_params.get('file_path', '')
    if not isinstance(document_path, (str, Path)) or document_path == '':
        print('Input file (directory) path is not assigned')
    doc_extension = str(document_path).lower().split('.')[-1]
    if os.path.isdir(document_path):
        doc_extension = LoaderType.directory

    is_pdf = doc_extension == LoaderType.pdf
    is_json = doc_extension == LoaderType.json
    is_txt = any([doc_extension == LoaderType.docx, doc_extension == LoaderType.doc,
                  doc_extension == LoaderType.rtf, doc_extension == LoaderType.odt])
    is_zip = doc_extension == LoaderType.zip
    is_directory = doc_extension == LoaderType.directory

    parsing_scheme = loader_params.pop('parsing_scheme', 'lines')
    extract_images = loader_params.pop('extract_images', False)
    extract_tables = loader_params.pop('extract_tables', False)
    parse_formulas = loader_params.pop('parse_formulas', False)
    remove_service_info = loader_params.pop('remove_service_info', False)
    loader_params = dict(
        pdf_parsing_scheme=parsing_scheme,
        pdf_extract_images=extract_images,
        pdf_extract_tables=extract_tables,
        pdf_parse_formulas=parse_formulas,
        pdf_remove_service_info=remove_service_info,
        word_doc_parsing_scheme=parsing_scheme,
        word_doc_extract_images=extract_images,
        word_doc_extract_tables=extract_tables,
        word_doc_parse_formulas=parse_formulas,
        word_doc_remove_service_info=remove_service_info,
        **loader_params,
    )

    if is_pdf:
        return PDFLoader(**loader_params)
    elif is_json:
        return LangChainDocumentLoader(**loader_params)
    elif is_txt:
        return WordDocumentLoader(**loader_params)
    elif is_zip:
        return ZipLoader(**loader_params)
    elif is_directory:
        return RecursiveDirectoryLoader(**loader_params)
