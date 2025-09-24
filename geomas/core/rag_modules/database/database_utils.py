from typing import Callable
from pathlib import Path

from chromadb.utils.data_loaders import ImageLoader
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.models import Collection
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

# from ChemCoScientist.paper_analysis.prompts import summarisation_prompt
# from ChemCoScientist.paper_analysis.settings import allowed_providers
# from ChemCoScientist.paper_analysis.settings import settings as default_settings
# from CoScientist.paper_parser.s3_connection import s3_service
from geomas.core.repository.database_repository import DATABASE_PORT, DATABASE_HOST, RESET_DATABASE

class ExpandedSummary(BaseModel):
    """
    Expanded version of paper's summary.
    """

    paper_summary: str = Field(description="Summary of the paper.")
    paper_title: str = Field(
        description="Title of the paper. If the title is not explicitly specified, use the default value - 'NO TITLE'"
    )
    publication_year: int = Field(
        description=(
            "Year of publication of the paper. If the publication year is not explicitly specified, use the default"
            " value - 9999."
        )
    )


class CustomEmbeddingFunction(EmbeddingFunction):
    """
    Creates embeddings from text using a custom function.

        This class provides a way to generate embeddings for text data using a
        user-defined function. It takes the embedding function as a constructor argument.

        Attributes:
        - embedding_function: The function used to generate embeddings.
    """



    def __call__(self, texts: Documents, DatabaseStore:Callable) -> Embeddings:
        """
        Retrieves embeddings for a list of documents using a ChromaDBPaperStore.

        Args:
            self: The instance of the class.
            texts: The documents to retrieve embeddings for (list of strings).

        Returns:
            Embeddings: The embeddings for the input documents (list of lists of floats).

        This method transforms text into numerical vector representations (embeddings).
        These embeddings capture the semantic meaning of the documents,
        allowing for efficient comparison and retrieval of relevant information.
        """
        return DatabaseStore.get_embeddings(texts)

class ChromaDatabaseClient:
    """
    A client for interacting with a Chroma database.

        This class provides methods to manage Chroma collections, including
        creating, querying, and deleting them. It abstracts the underlying
        ChromaDB client for easier use.

        Attributes:
        - client: The ChromaDB client instance.
        - collection_name: The name of the Chroma collection to use.
        - embedding_function: The embedding function used for vectorizing data.
    """

    def __init__(self):
        """
        Initializes the ChromaDB client.

        Connects to a ChromaDB instance to enable storage and retrieval of scientific paper data for question answering.

        Args:
            self: The instance of the class.

        Initializes the following class fields:
            client: The ChromaDB client object used for interacting with the database.
                    It is initialized using the host, port, and reset settings
                    from the default settings.

        Returns:
            None
        """
        # self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.client = chromadb.HttpClient(host=DATABASE_HOST,
                                          port=DATABASE_PORT,
                                          settings=chromadb.Settings(allow_reset=RESET_DATABASE),)
        self.image_loader = ImageLoader()
        self.collection_methods = {'show':self.client.list_collections,
                              'delete':self.client.delete_collection,
                              'get':self.client.get_or_create_collection}


    def collection(self,method:str,collection_metadata:dict):
        """
        Apply action on Chroma collection.

        This method ensures a Chroma collection exists for storing and retrieving document data.
        It prioritizes retrieving an existing collection by name. If a collection with the given name doesn't exist,
        it creates a new one, configured with the specified embedding function and a default data loader.
        This enables efficient storage and search of scientific documents.

        Args:
            method (str): The name of the Chroma collection to retrieve or create.
            collection_metadata (dict): The name of the Chroma collection to retrieve or create.
        Returns:
            Collection: The Chroma collection.
        """
        self.collection_methods[method](collection_metadata)

    def _get_or_create_chroma_collection(
            self,
            collection: str,
            embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> Collection:
        """
        Gets or creates a Chroma collection.

        This method ensures a Chroma collection exists for storing and retrieving document data.
        It prioritizes retrieving an existing collection by name. If a collection with the given name doesn't exist,
        it creates a new one, configured with the specified embedding function and a default data loader.
        This enables efficient storage and search of scientific documents.

        Args:
            collection (str): The name of the Chroma collection to retrieve or create.
            embedding_function (EmbeddingFunction[Documents] | None, optional):
            An optional embedding function to use for the collection.
            If None, the default embedding function is used. Defaults to None.

        Returns:
            Collection: The Chroma collection.
        """
        return self.client.get_or_create_collection(
            name=collection,
            embedding_function=embedding_function,
            data_loader=DATA_LOADER,
        )

    @staticmethod
    def query_chromadb(
            collection: Collection,
            query_text: str,
            metadata_filter: dict = None,
            chunk_num: int = 3,
    ) -> dict:
        """
        Queries a ChromaDB collection to find relevant information based on a text query.

        Args:
            collection: The ChromaDB collection to query.
            query_text: The text query to perform.  This is the information the user is seeking.
            metadata_filter: Optional dictionary to filter results based on metadata.
            chunk_num: The number of results to return.  Determines how many of the most relevant documents will be returned.

        Returns:
            dict: A dictionary containing the query results, including:
                - 'documents': The text of the retrieved documents.
                - 'metadatas': The metadata associated with each document.
                - 'distances':  A measure of similarity between the query and each document.
        """
        return collection.query(
            query_texts=[query_text],
            n_results=chunk_num,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )