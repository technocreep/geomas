from langchain_community.document_loaders.directory import _is_visible
import zipfile
from typing import Iterator, Union, Any, Optional, Sequence
from json import load
from tqdm import tqdm
from pathlib import Path
from typing import Iterator, Union, Any, Optional
from langchain_core.document_loaders import BaseLoader, Blob
from langchain_core.documents import Document
from langchain_core.load import load as ln_load


def preprocess_documents(func):
    def wrapper(self, *args, **kwargs):
        documents = func(self, *args, **kwargs)
        for doc in documents:
            if len(doc.page_content) < 15:
                continue
            if sum(c.isdigit() for c in doc.page_content) / len(doc.page_content) > 0.2:
                continue
            yield doc

    return wrapper

class RecursiveDirectoryLoader(BaseLoader):
    """
    Load files from directory into list of documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        pdf_parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        pdf_extract_images: bool = False,
        pdf_extract_tables: bool = False,
        pdf_extract_formulas: bool = False,
        pdf_remove_service_info: bool = False,
        word_doc_parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        word_doc_extract_images: bool = False,
        word_doc_extract_tables: bool = False,
        word_doc_extract_formulas: bool = False,
        timeout_for_converting: Optional[int] = None,
        exclude_files: Sequence[Union[Path, str]] = (),
        parsing_logger: Optional[ParsingLogger] = None,
        silent_errors: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize with a directory path."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Directory not found: '{self.file_path}'")
        if not self.file_path.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.file_path}'")

        self._logger = parsing_logger or ParsingLogger(
            silent_errors=silent_errors, name=__name__
        )

        self._pdf_kwargs = {
            "parsing_scheme": pdf_parsing_scheme,
            "extract_images": pdf_extract_images,
            "extract_tables": pdf_extract_tables,
            "extract_formulas": pdf_extract_formulas,
            "remove_headers": pdf_remove_service_info,
            "parsing_logger": self._logger,
        }
        self._word_doc_kwargs = {
            "parsing_scheme": word_doc_parsing_scheme,
            "extract_images": word_doc_extract_images,
            "extract_tables": word_doc_extract_tables,
            "extract_formulas": word_doc_extract_formulas,
            "timeout_for_converting": timeout_for_converting,
            "parsing_logger": self._logger,
        }
        self._zip_kwargs = {
            "pdf_parsing_scheme": pdf_parsing_scheme,
            "pdf_extract_images": pdf_extract_images,
            "pdf_extract_tables": pdf_extract_tables,
            "pdf_extract_formulas": pdf_extract_formulas,
            "pdf_remove_service_info": pdf_remove_service_info,
            "word_doc_parsing_scheme": word_doc_parsing_scheme,
            "word_doc_extract_images": word_doc_extract_images,
            "word_doc_extract_tables": word_doc_extract_tables,
            "word_doc_extract_formulas": word_doc_extract_formulas,
            "timeout_for_converting": timeout_for_converting,
            "exclude_files": exclude_files,
            "parsing_logger": self._logger,
        }
        self._exclude_names = [Path(file).name for file in exclude_files]

    @property
    def logs(self):
        return self._logger.logs

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path"""
        paths = [
            path
            for path in self.file_path.rglob("**/[!.]*")
            if path.is_file()
            and _is_visible(path)
            and path.name not in self._exclude_names
            and correct_path_encoding(path.name) not in self._exclude_names
        ]
        for path in tqdm(paths, desc="Directory processing", ncols=80):
            doc_type = BaseParser.get_doc_type(path)
            match doc_type:
                case DocType.pdf:
                    _loader = PDFLoader(path, **self._pdf_kwargs)
                case DocType.docx | DocType.doc | DocType.odt | DocType.rtf:
                    _loader = WordDocumentLoader(path, **self._word_doc_kwargs)
                case DocType.zip:
                    _loader = ZipLoader(path, **self._zip_kwargs)
                case _:
                    self._logger.info(
                        f"Skip file processing, no suitable loader for {path}"
                    )
                    continue

            self._logger.info(f"Processing file: {path}")
            yield from _loader.lazy_load()

class WordDocumentLoader(BaseLoader):
    """
    Load Word Document into list of documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        byte_content: Optional[bytes] = None,
        parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_formulas: bool = False,
        timeout_for_converting: Optional[int] = None,
        parsing_logger: Optional[ParsingLogger] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with a file path."""
        self.file_path = str(file_path)
        doc_type = WordDocumentParser.get_doc_type(self.file_path)
        if doc_type not in [DocType.docx, DocType.doc, DocType.odt, DocType.rtf]:
            if doc_type is DocType.unsupported:
                raise ValueError("The file type is unsupported")
            else:
                raise ValueError(
                    f"The {doc_type} file type does not match the Loader! Use a suitable one."
                )
        self.byte_content = byte_content
        self._doc_type = doc_type.value
        self._logger = parsing_logger or ParsingLogger(name=__name__)
        self.parser = WordDocumentParser(
            parsing_scheme,
            extract_images,
            extract_tables,
            extract_formulas,
            timeout_for_converting,
        )

    @property
    def logs(self):
        return self._logger.logs

    @preprocess_documents
    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path"""
        if self.byte_content is None:
            blob = Blob.from_path(self.file_path, mime_type=self._doc_type)
        else:
            blob = Blob.from_data(
                self.byte_content, path=self.file_path, mime_type=self._doc_type
            )
        with self._logger.parsing_info_handler(self.file_path):
            yield from self.parser.lazy_parse(blob)


class PDFLoader(BaseLoader):
    """
    Load PDF into list of documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        byte_content: Optional[bytes] = None,
        parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_formulas: bool = False,
        remove_headers: bool = False,
        parsing_logger: Optional[ParsingLogger] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with a file path."""
        self.file_path = str(file_path)
        doc_type = PDFParser.get_doc_type(self.file_path)
        if doc_type is not DocType.pdf:
            if doc_type is DocType.unsupported:
                raise ValueError("The file type is unsupported")
            else:
                raise ValueError(
                    f"The {doc_type} file type does not match the Loader! Use a suitable one."
                )
        self.byte_content = byte_content
        self._logger = parsing_logger or ParsingLogger(name=__name__)
        self.parser = PDFParser(
            parsing_scheme,
            extract_images,
            extract_tables,
            extract_formulas,
            remove_headers,
        )

    @property
    def logs(self):
        return self._logger.logs

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path"""
        if self.byte_content is None:
            blob = Blob.from_path(self.file_path)
        else:
            blob = Blob.from_data(
                self.byte_content, path=self.file_path, mime_type=DocType.pdf.value
            )
        with self._logger.parsing_info_handler(self.file_path):
            yield from self.parser.lazy_parse(blob)


class ZipLoader(BaseLoader):
    """
    Load files from zip archive into list of documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        byte_content: Optional[bytes] = None,
        pdf_parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        pdf_extract_images: bool = False,
        pdf_extract_tables: bool = False,
        pdf_extract_formulas: bool = False,
        pdf_remove_service_info: bool = False,
        word_doc_parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        word_doc_extract_images: bool = False,
        word_doc_extract_tables: bool = False,
        word_doc_extract_formulas: bool = False,
        timeout_for_converting: Optional[int] = None,
        exclude_files: Sequence[Union[Path, str]] = (),
        parsing_logger: Optional[ParsingLogger] = None,
        silent_errors: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize with a file path."""
        self.file_path = str(file_path)
        doc_type = BaseParser.get_doc_type(self.file_path)
        if doc_type is not DocType.zip:
            if doc_type is DocType.unsupported:
                raise ValueError("The file type is unsupported")
            else:
                raise ValueError(
                    f"The {doc_type} file type does not match the Loader! Use a suitable one."
                )
        self.byte_content = byte_content
        self._logger = parsing_logger or ParsingLogger(
            silent_errors=silent_errors, name=__name__
        )
        self.pdf_parser = PDFParser(
            pdf_parsing_scheme,
            pdf_extract_images,
            pdf_extract_tables,
            pdf_extract_formulas,
            pdf_remove_service_info,
        )
        self.word_doc_parser = WordDocumentParser(
            word_doc_parsing_scheme,
            word_doc_extract_images,
            word_doc_extract_tables,
            word_doc_extract_formulas,
            timeout_for_converting,
        )
        self._exclude_names = [Path(file).name for file in exclude_files]

    @property
    def logs(self):
        return self._logger.logs

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path"""
        content = self.byte_content or self.file_path
        with zipfile.ZipFile(content) as z:
            for info in tqdm(z.infolist(), desc="Zip processing", ncols=80):
                file_name = Path(correct_path_encoding(info.filename))
                if file_name.name in self._exclude_names:
                    continue
                path = str(Path(self.file_path, file_name))
                doc_type = BaseParser.get_doc_type(file_name)
                match doc_type:
                    case DocType.pdf:
                        _parser = self.pdf_parser
                    case DocType.docx | DocType.doc | DocType.odt | DocType.rtf:
                        _parser = self.word_doc_parser
                    case _:
                        if Path(file_name).suffix:
                            self._logger.info(
                                f"Skip file processing in zip, no suitable parser for {path}"
                            )
                        continue

                self._logger.info(f"Processing file in zip: {path}")
                blob = Blob.from_data(
                    z.open(info).read(), path=path, mime_type=doc_type.value
                )
                with self._logger.parsing_info_handler(path):
                    yield from _parser.lazy_parse(blob)

class LangChainDocumentLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, 'r') as f:
            for i, doc_dict in load(f).items():
                yield ln_load(doc_dict)