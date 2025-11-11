# geomas/core/rag_modules/steps/max_min_chunking.py

import re
from typing import List
import numpy as np
from langchain_core.documents.base import Document

from geomas.core.repository.constant_repository import ROOT_DIR, USE_S3
from geomas.core.repository.parsing_repository import ParsingPatternConfig
import os
import logging

logger = logging.getLogger(__name__)

PARSE_RESULTS_PATH = os.path.join(ROOT_DIR, os.environ.get("PARSE_RESULTS_PATH", "./parse_results"))


class MaxMinChunkingParams:
    """Parameters for Max-Min semantic chunking algorithm"""
    
    def __init__(
        self,
        c: float = 0.8,
        hard_thr: float = 0.5,
        init_const: float = 1.2,
        min_chunk_sentences: int = 1,
        max_chunk_sentences: int = 20,
    ):
        """
        Initialize Max-Min chunking parameters.
        
        Args:
            c: Coefficient for computing adaptive threshold
            hard_thr: Hard minimum similarity threshold
            init_const: Constant for initial chunk (with one sentence)
            min_chunk_sentences: Minimum number of sentences in a chunk
            max_chunk_sentences: Maximum number of sentences in a chunk
        """
        self.c = c
        self.hard_thr = hard_thr
        self.init_const = init_const
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
    
    @classmethod
    def from_dict(cls, params: dict):
        """Create an instance from a dictionary of parameters"""
        return cls(**params) if params else cls()
    
    def to_dict(self) -> dict:
        """Convert parameters to a dictionary"""
        return {
            'c': self.c,
            'hard_thr': self.hard_thr,
            'init_const': self.init_const,
            'min_chunk_sentences': self.min_chunk_sentences,
            'max_chunk_sentences': self.max_chunk_sentences,
        }


class MaxMinTextChunker:
    """
    Implementation of Max-Min semantic chunking algorithm.
    
    The algorithm splits text into semantically related chunks using
    cosine similarity between sentence embeddings.
    
    Algorithm workflow:
    1. Split text into sentences
    2. Get embeddings for each sentence
    3. Iterate through sentences and decide whether to add to current chunk or start new one
    4. Decision is based on:
       - For empty chunk: start new chunk
       - For chunk with 1 sentence: use init_const * sim >= hard_thr
       - For chunk with multiple sentences: use max_sim(s_k, C) >= threshold
         where threshold = max(c * min_sim(C) * σ(|C|), hard_thr)
    """
    
    def __init__(self, chunking_params: dict = None, model_name: str = None):
        """
        Initialize the Max-Min text chunker.
        
        Args:
            chunking_params: Dictionary with chunking parameters or None for defaults
            model_name: Name of sentence-transformers model to use
        """
        if chunking_params is None:
            self.params = MaxMinChunkingParams()
        else:
            self.params = MaxMinChunkingParams.from_dict(chunking_params)
        
        # Initialize embedding model directly
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or "paraphrase-multilingual-MiniLM-L12-v2"
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Model loaded successfully")
    
    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts using local sentence-transformers model.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of numpy arrays with embeddings
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses a more advanced sentence splitter that preserves abbreviations
        like "etc.", "Dr.", "Mr.", etc.
        
        Args:
            text: Source text
            
        Returns:
            List of sentences
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split by periods, exclamation marks, and question marks followed by capital letters
        sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', text)
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1, emb2: Embedding vectors
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def _compute_min_sim(self, chunk_embeddings: List[np.ndarray]) -> float:
        """
        Compute minimum cosine similarity between all pairs of sentences in the chunk.
        
        This corresponds to min_sim(C) in the algorithm - it measures the internal
        coherence of the current chunk.
        
        Args:
            chunk_embeddings: List of sentence embeddings in the current chunk
            
        Returns:
            Minimum similarity value
        """
        if len(chunk_embeddings) < 2:
            return 1.0
        
        min_sim = 1.0
        for i in range(len(chunk_embeddings)):
            for j in range(i + 1, len(chunk_embeddings)):
                sim = self._cosine_similarity(chunk_embeddings[i], chunk_embeddings[j])
                min_sim = min(min_sim, sim)
        
        return min_sim
    
    def _compute_max_sim(self, sentence_emb: np.ndarray, chunk_embeddings: List[np.ndarray]) -> float:
        """
        Compute maximum cosine similarity between a sentence and the chunk.
        
        This corresponds to max_sim(s_k, C) in the algorithm.
        
        Args:
            sentence_emb: Embedding of the current sentence
            chunk_embeddings: List of sentence embeddings in the chunk
            
        Returns:
            Maximum similarity value
        """
        if not chunk_embeddings:
            return 0.0
        
        max_sim = -1.0
        for chunk_emb in chunk_embeddings:
            sim = self._cosine_similarity(sentence_emb, chunk_emb)
            max_sim = max(max_sim, sim)
        
        return max_sim
    
    def _sigma_function(self, chunk_size: int) -> float:
        """
        Function σ(|C|) for adapting threshold based on chunk size.
        
        Different strategies can be used. Here we use a simple constant function.
        Alternative: return 1.0 / (1.0 + 0.1 * chunk_size) for decreasing threshold
        
        Args:
            chunk_size: Size of the current chunk (number of sentences)
            
        Returns:
            Value of σ function
        """
        # Simple implementation: constant 1.0
        # Can be made decreasing so larger chunks have stricter threshold
        return 1.0
    
    def _compute_threshold(self, min_sim: float, chunk_size: int) -> float:
        """
        Compute adaptive threshold for adding a sentence to the chunk.
        
        Corresponds to: thr(C) = max{c * min_sim(C) * σ(|C|), hard_thr}
        
        Args:
            min_sim: Minimum similarity within the current chunk
            chunk_size: Size of the current chunk
            
        Returns:
            Threshold value
        """
        adaptive_thr = self.params.c * min_sim * self._sigma_function(chunk_size)
        return max(adaptive_thr, self.params.hard_thr)
    
    def _max_min_chunking(
        self, 
        sentences: List[str], 
        embeddings: List[np.ndarray]
    ) -> List[List[int]]:
        """
        Implementation of Max-Min semantic chunking algorithm.
        
        Algorithm 1 from the paper:
        - Lines 5-7: Empty chunk - start new chunk
        - Lines 8-14: Chunk with one sentence - use init_const * sim >= hard_thr
        - Lines 15-24: Chunk with multiple sentences - use max-min approach
        
        Args:
            sentences: List of sentences
            embeddings: List of sentence embeddings
            
        Returns:
            List of chunks, where each chunk is a list of sentence indices
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_chunk_embeddings = []
        
        for k, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            # Lines 5-7: Empty chunk
            if not current_chunk:
                current_chunk = [k]
                current_chunk_embeddings = [embedding]
            
            # Lines 8-14: Chunk with one sentence
            elif len(current_chunk) == 1:
                sim = self._cosine_similarity(current_chunk_embeddings[0], embedding)
                
                if self.params.init_const * sim >= self.params.hard_thr:
                    # Add to current chunk
                    current_chunk.append(k)
                    current_chunk_embeddings.append(embedding)
                else:
                    # Start new chunk
                    chunks.append(current_chunk)
                    current_chunk = [k]
                    current_chunk_embeddings = [embedding]
            
            # Lines 15-24: Chunk with multiple sentences
            else:
                # Check maximum chunk size
                if len(current_chunk) >= self.params.max_chunk_sentences:
                    chunks.append(current_chunk)
                    current_chunk = [k]
                    current_chunk_embeddings = [embedding]
                    continue
                
                min_sim = self._compute_min_sim(current_chunk_embeddings)
                max_sim = self._compute_max_sim(embedding, current_chunk_embeddings)
                thr = self._compute_threshold(min_sim, len(current_chunk))
                
                if max_sim >= thr:
                    # Add to current chunk
                    current_chunk.append(k)
                    current_chunk_embeddings.append(embedding)
                else:
                    # Start new chunk
                    chunks.append(current_chunk)
                    current_chunk = [k]
                    current_chunk_embeddings = [embedding]
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def extract_img_url(self, doc_text: str, p_name: str) -> list[str]:
        """
        Extract image URLs from document text.
        
        This method identifies image references within the text and constructs their full paths.
        It focuses on JPEG images specifically referenced using a markdown-like syntax.
        
        Args:
            doc_text: Document text
            p_name: Project/paper name
            
        Returns:
            List of image paths
        """
        matches = re.findall(ParsingPatternConfig.image_pattern, doc_text)
        
        return [entry[0] for entry in matches] if USE_S3 \
            else [os.path.join(PARSE_RESULTS_PATH, p_name, entry[0]) for entry in matches]
    
    def apply_chunking(
        self, 
        raw_text: str, 
        document_name: str, 
        document_type: str
    ) -> List[Document]:
        """
        Apply Max-Min semantic chunking to document text.
        
        Signature is compatible with existing TextChunker.apply_chunking()
        
        Args:
            raw_text: Source document text
            document_name: Document name
            document_type: Document type (html, markdown, etc.)
            
        Returns:
            List of LangChain Document objects with text chunks
        """
        logger.info(f"Applying Max-Min chunking to document: {document_name}")
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(raw_text)
        
        if not sentences:
            logger.warning(f"Document {document_name} contains no sentences")
            return []
        
        logger.info(f"Found {len(sentences)} sentences")
        
        # Step 2: Get embeddings for all sentences
        # For optimization, can be done in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embeddings = self._get_embeddings(batch)
            all_embeddings.extend(embeddings)
        
        logger.info(f"Retrieved {len(all_embeddings)} embeddings")
        
        # Step 3: Apply Max-Min algorithm
        chunk_indices = self._max_min_chunking(sentences, all_embeddings)
        
        logger.info(f"Created {len(chunk_indices)} chunks")
        
        # Step 4: Form LangChain Document objects
        documents = []
        
        for chunk_idx, indices in enumerate(chunk_indices):
            # Combine sentences into chunk text
            chunk_text = ' '.join([sentences[i] for i in indices])
            
            # Create Document with metadata
            doc = Document(
                page_content="passage: " + chunk_text,
                metadata={
                    "source": document_name + ".pdf",
                    "chunk_index": chunk_idx,
                    "sentence_count": len(indices),
                    "sentence_indices": indices,
                    "imgs_in_chunk": str(self.extract_img_url(chunk_text, document_name)),
                }
            )
            
            documents.append(doc)
        
        return documents

