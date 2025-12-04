from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from geomas.cli import describe_image
from geomas.core.rag_modules.data_adapter import AdapterResult, DataLoaderAdapter

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class InsertionResult:
    """Summary of a vector store insertion batch."""
    inserted: int = 0
    skipped: int = 0

    def __bool__(self) -> bool:
        return self.inserted > 0


@dataclass(slots=True, frozen=True)
class ProcessingResult:
    success: bool
    documents_ingested: int = 0
    documents_skipped: int = 0
    summaries_created: int = 0


class DatabaseRagPipeline:
    """Ingest artefacts into the Chroma store using the adapter pipeline."""
    def __init__(
        self,
        *,
        vector_store: Chroma,
        embedding_function: Embeddings | None = None,
        parser: object | None = None,
        data_loader: DataLoaderAdapter | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_function = vector_store.embeddings or embedding_function
        self.parser = parser
        if data_loader is None:
            self.data_loader = DataLoaderAdapter(parser=parser)
        else:
            self.data_loader = data_loader
            if parser is not None and hasattr(self.data_loader, "parser"):
                self.data_loader.parser = parser

    def process(
        self,
        folder_path: Path | str,
        *,
        loader_overrides: Mapping[str, object] | None = None,
        document_name: str | None = None,
        namespace: str,
        generate_image_descriptions: bool = False,
        include_images: bool = False,
    ) -> ProcessingResult:
        path = Path(folder_path)
        try:
            adapter_result: AdapterResult = self.data_loader.load_and_transform(
                path,
                document_name=document_name,
                loader_overrides=loader_overrides,
            )
        except Exception as exc:
            logger.exception("Failed to load artefact '%s': %s", path, exc)
            return ProcessingResult(success=False)

        documents: list[Document] = list(adapter_result.documents)
       
        image_documents = self._collect_images(
            path, document_name=document_name, namespace=namespace
        )
        description_documents = self._describe_images(
            # TODO: unlimit this if describing will speed up
            image_documents[:3] or image_documents[:2] or image_documents,
            namespace=namespace,
            enabled=generate_image_descriptions,
        )
        
        if include_images:
            documents.extend(image_documents)

        if not documents:
            logger.info("No documents produced for '%s'", path)
            return ProcessingResult(success=False)

        try:
            insertion = self._ingest_documents(
                documents,
                namespace=namespace,
                description_documents=description_documents,
            )
        except Exception as exc:
            logger.exception("Failed to store documents in ChromaDB: %s", exc)
            return ProcessingResult(success=False)

        inserted_total = insertion.inserted
        skipped_total = insertion.skipped

        if inserted_total == 0 and skipped_total == 0:
            logger.warning(
                "ChromaDB ingestion produced no changes or skips for '%s'", path
            )
            return ProcessingResult(success=False)

        if insertion.inserted > 0:
            logger.info("Stored %s new chunks for '%s'", insertion.inserted, path)
        elif documents:
            logger.info("No changes detected for '%s'; skipping reinsertion", path)

        self._cleanup_ingest_artifacts(adapter_result.cleanup_paths)

        return ProcessingResult(
            success=True,
            documents_ingested=inserted_total,
            documents_skipped=skipped_total,
            summaries_created=0,
        )

    def _ingest_documents(
        self,
        documents: Sequence[Document],
        *,
        namespace: str,
        description_documents: Sequence[Document] | None = None,
    ) -> InsertionResult:
        text_documents: list[Document] = []
        image_uris: list[str] = []
        image_metadatas: list[dict[str, object]] = []
        image_ids: list[str] = []
        skipped_images = 0

        for chunk in documents:
            metadata = dict(chunk.metadata or {})
            metadata.setdefault("scope", namespace)
            identifier = chunk.id

            if metadata.get("type") == "image":
                source_path = metadata.get("source_path") or metadata.get("source")
                if isinstance(source_path, str) and Path(source_path).exists():
                    image_uris.append(source_path)
                    image_metadatas.append(metadata)
                    image_ids.append(str(identifier) if identifier else None)
                else:
                    skipped_images += 1
                continue

            text_documents.append(
                Document(
                    page_content=str(chunk.page_content),
                    metadata=metadata,
                    id=identifier,
                )
            )
        inserted_texts = 0
        batch_size = 5461
        if text_documents:
            try:
                while len(text_documents) > batch_size:
                    documents = text_documents[:batch_size]
                    text_documents = text_documents[batch_size:]
                    self.vector_store.add_documents(documents=documents)
                    inserted_texts += batch_size
                self.vector_store.add_documents(documents=text_documents)
                inserted_texts += len(text_documents)
            except Exception as exc:
                logger.warning(
                    "Failed to store text documents for namespace %s: %s",
                    namespace,
                    exc,
                )
        if image_uris:
            self.vector_store.add_images(
                uris=image_uris,
                metadatas=image_metadatas,
                ids=image_ids if any(image_ids) else None,
            )
        inserted_descriptions = 0
        if description_documents:
            try:
                self.vector_store.add_documents(documents=list(description_documents))
                inserted_descriptions = len(description_documents)
            except Exception as exc:
                logger.warning(
                    "Failed to store image descriptions for namespace %s: %s",
                    namespace,
                    exc,
                )
        return InsertionResult(
            inserted=inserted_texts + len(image_uris) + inserted_descriptions,
            skipped=skipped_images,
        )

    def _describe_images(
        self,
        images: Sequence[Document],
        *,
        namespace: str,
        enabled: bool,
    ) -> list[Document]:
        if not enabled:
            return []

        descriptions: list[Document] = []
        for image in images:
            metadata = dict(image.metadata or {})
            source_path = metadata.get("source_path")
            if not isinstance(source_path, str):
                continue

            candidate_path = Path(source_path)
            if not candidate_path.is_file():
                logger.debug("Skipping description; missing file at %s", candidate_path)
                continue
            if candidate_path.suffix.lower() not in DataLoaderAdapter.IMAGE_SUFFIXES:
                continue

            try:
                description_text = describe_image(image_path=str(candidate_path), output=None)
                self.description = description_text
            except Exception as exc:
                logger.warning(
                    "Failed to describe image '%s': %s", candidate_path, exc
                )
                continue

            if not description_text:
                logger.debug(
                    "Received empty description for image '%s'; skipping", candidate_path
                )
                continue

            description_metadata = {
                **metadata,
                "type": "image_description",
                "description_for": metadata.get("source")
                or metadata.get("document_name")
                or candidate_path.stem + candidate_path.suffix,
                "scope": namespace,
            }
            descriptions.append(
                Document(page_content=str(description_text), metadata=description_metadata)
            )

        return descriptions

    @staticmethod
    def _collect_images(
        root: Path, *, document_name: str | None, namespace: str
    ) -> list[Document]:
        images: list[Document] = []
        for current_root, _, files in os.walk(root):
            base = Path(current_root)
            for filename in sorted(files):
                candidate = base / filename
                if candidate.suffix.lower() not in DataLoaderAdapter.IMAGE_SUFFIXES:
                    continue
                try:
                    fingerprint, last_modified, size_bytes = DataLoaderAdapter._collect_file_metadata(
                        candidate
                    )
                except Exception as exc:
                    logger.warning("Failed to read image metadata '%s': %s", candidate, exc)
                    continue

                caption = f"Image: {candidate.name}"
                metadata = {
                    "type": "image",
                    "scope": namespace,
                    "source": document_name or candidate.stem + candidate.suffix,
                    "document_name": document_name or candidate.stem + candidate.suffix,
                    "source_path": str(candidate),
                    "source_fingerprint": fingerprint,
                    "source_last_modified": last_modified,
                    "source_size_bytes": size_bytes,
                    "caption": caption,
                }
                images.append(Document(page_content=caption, metadata=metadata))
        return images

    def _cleanup_ingest_artifacts(self, cleanup_paths: Iterable[Path]) -> None:
        for candidate in cleanup_paths:
            path = Path(candidate)
            try:
                if not path.exists():
                    continue
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning("Failed to remove ingest artefact '%s': %s", path, exc)