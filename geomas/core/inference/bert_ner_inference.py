"""
BERT NER Inference Module for Geological Entity Extraction.

This module provides functionality to use trained BERT NER models
for extracting geological entities from text documents.
"""

import json
import os
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Suppress warnings for faster execution
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress torch distributed warnings
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)

from geomas.core.logging.logger import get_logger

logger = get_logger()


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float


@dataclass
class EntityExtractionResult:
    """Result of entity extraction from text."""
    text: str
    entities: List[Entity]
    tokens: List[str]
    labels: List[str]


class BertNerInference:
    """BERT NER inference for geological entity extraction."""
    
    def __init__(self, model_path: Path):
        """
        Initialize BERT NER inference.
        
        Args:
            model_path: Path to the trained BERT NER model directory
        """
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        self.nlp_pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained BERT NER model and tokenizer."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        # Load label mappings
        label2id_file = self.model_path / "label2id.json"
        id2label_file = self.model_path / "id2label.json"
        
        if label2id_file.exists():
            with open(label2id_file, 'r', encoding='utf-8') as f:
                self.label2id = json.load(f)
        
        if id2label_file.exists():
            with open(id2label_file, 'r', encoding='utf-8') as f:
                self.id2label = json.load(f)
        
        logger.info(f"Loading BERT NER model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model with proper configuration
        from transformers import AutoConfig
        
        # Load config first
        config = AutoConfig.from_pretrained(self.model_path)
        
        # Override config with our label mappings
        config.num_labels = len(self.label2id)
        config.id2label = self.id2label
        config.label2id = self.label2id
        
        logger.info(f"Config num_labels: {config.num_labels}")
        logger.info(f"Label2id keys: {list(self.label2id.keys())[:5]}...")
        logger.info(f"ID2label sample: {dict(list(self.id2label.items())[:3])}")
        
        # Load model with config
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path,
            config=config,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        # Ensure model config matches our mappings
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        self.model.config.num_labels = len(self.label2id)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            logger.info("Model moved to GPU")
        else:
            logger.info("Model using CPU")
        
        # Create NER pipeline (simplified to avoid compatibility issues)
        try:
            self.nlp_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not create NER pipeline: {e}. Using manual inference.")
            self.nlp_pipeline = None
        
        logger.info(f"Model loaded with {len(self.id2label)} labels")
    
    def extract_entities(self, text: str) -> EntityExtractionResult:
        """
        Extract geological entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            EntityExtractionResult with extracted entities
        """
        # Get tokens, labels, and logits for detailed analysis
        tokens, labels, logits = self._tokenize_and_predict(text)
        
        # Extract entities from BIO labels with real confidence scores
        entities = []
        current_entity = None
        
        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # Get confidence score from probabilities
            if i < probabilities.shape[1]:
                token_probs = probabilities[0, i]  # Get probabilities for this token
                max_prob = torch.max(token_probs).item()
                confidence = float(max_prob)
            else:
                confidence = 0.5  # Fallback confidence
            
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # Remove 'B-' prefix
                current_entity = {
                    'text': token,
                    'label': entity_type,
                    'start': i,
                    'end': i,
                    'confidence': confidence
                }
            elif label.startswith('I-'):
                if current_entity and label[2:] == current_entity['label']:
                    # Continue current entity
                    current_entity['text'] += ' ' + token
                    current_entity['end'] = i
                    # Update confidence to minimum (worst case)
                    current_entity['confidence'] = min(current_entity['confidence'], confidence)
                else:
                    # Start new entity even if it's I- (fallback)
                    if current_entity:
                        entities.append(current_entity)
                    
                    entity_type = label[2:]  # Remove 'I-' prefix
                    current_entity = {
                        'text': token,
                        'label': entity_type,
                        'start': i,
                        'end': i,
                        'confidence': confidence * 0.9  # Slightly lower confidence for I- start
                    }
            else:
                # End of entity or non-entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        # Post-process entities: merge adjacent entities and fix WordPiece tokens
        entities = self._post_process_entities(entities, tokens)
        
        # Convert to Entity objects
        entity_objects = []
        for entity_dict in entities:
            entity = Entity(
                text=entity_dict['text'],
                label=entity_dict['label'],
                start=entity_dict['start'],
                end=entity_dict['end'],
                confidence=entity_dict['confidence']
            )
            entity_objects.append(entity)
        
        return EntityExtractionResult(
            text=text,
            entities=entity_objects,
            tokens=tokens,
            labels=labels
        )
    
    def _post_process_entities(self, entities: List[Dict], tokens: List[str]) -> List[Dict]:
        """
        Post-process entities to merge adjacent ones and fix WordPiece tokens.
        
        Args:
            entities: List of entity dictionaries
            tokens: List of tokens
            
        Returns:
            List of post-processed entities
        """
        if not entities:
            return entities
        
        # Sort entities by start position
        entities = sorted(entities, key=lambda x: x['start'])
        
        # Merge adjacent entities of the same type
        merged_entities = []
        current_entity = entities[0].copy()
        
        for i in range(1, len(entities)):
            next_entity = entities[i]
            
            # Check if entities are adjacent and same type
            if (next_entity['start'] == current_entity['end'] + 1 and 
                next_entity['label'] == current_entity['label']):
                # Merge entities
                current_entity['text'] += next_entity['text']
                current_entity['end'] = next_entity['end']
                current_entity['confidence'] = min(current_entity['confidence'], next_entity['confidence'])
            else:
                # Add current entity and start new one
                merged_entities.append(current_entity)
                current_entity = next_entity.copy()
        
        # Add last entity
        merged_entities.append(current_entity)
        
        # Fix WordPiece tokens and clean text
        cleaned_entities = []
        for entity in merged_entities:
            # Clean WordPiece tokens (remove ## and merge)
            cleaned_text = self._clean_wordpiece_text(entity['text'])
            
            # Filter out very short or meaningless entities
            if len(cleaned_text.strip()) >= 2 and self._is_meaningful_entity(cleaned_text, entity['label']):
                entity['text'] = cleaned_text
                cleaned_entities.append(entity)
        
        return cleaned_entities
    
    def _clean_wordpiece_text(self, text: str) -> str:
        """Clean WordPiece tokens by removing ## and merging properly."""
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Fix WordPiece tokens (## prefix)
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if word.startswith('##'):
                # Merge with previous word if exists
                if cleaned_words:
                    cleaned_words[-1] += word[2:]  # Remove ##
                else:
                    cleaned_words.append(word[2:])
            else:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    def _is_meaningful_entity(self, text: str, label: str) -> bool:
        """Check if entity is meaningful and not noise."""
        text = text.strip().lower()
        
        # Filter out very short entities
        if len(text) < 2:
            return False
        
        # Filter out common stopwords for geological entities
        geological_stopwords = {
            'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'у', 'о', 'об',
            'что', 'как', 'где', 'когда', 'почему', 'который', 'которая', 'которое',
            'это', 'этот', 'эта', 'это', 'тот', 'та', 'то', 'такие', 'такая', 'такое',
            'есть', 'быть', 'иметь', 'являться', 'находиться', 'располагаться'
        }
        
        # Skip if it's just a stopword
        if text in geological_stopwords:
            return False
        
        # Skip single punctuation
        if len(text) == 1 and text in '.,!?;:':
            return False
        
        # Skip numbers without context
        if text.isdigit() and len(text) < 4:
            return False
        
        return True
    
    def _tokenize_and_predict(self, text: str) -> Tuple[List[str], List[str], torch.Tensor]:
        """
        Tokenize text and predict labels for each token.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (tokens, labels, logits)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move inputs to the same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Debug logits shape
            logger.info(f"Logits shape: {logits.shape}")
            
            # Ensure logits has correct shape
            if len(logits.shape) == 3 and logits.shape[2] > 0:
                predictions = torch.argmax(logits, dim=2)
            elif len(logits.shape) == 2:
                predictions = torch.argmax(logits, dim=1)
                predictions = predictions.unsqueeze(0)
            else:
                # Fallback: create dummy predictions
                logger.warning(f"Unexpected logits shape: {logits.shape}")
                predictions = torch.zeros(inputs['input_ids'].shape[0], inputs['input_ids'].shape[1], dtype=torch.long, device=logits.device)
        
        # Convert to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        label_ids = predictions[0].tolist()
        
        # Use model's config for label mapping
        labels = []
        for label_id in label_ids:
            if str(label_id) in self.model.config.id2label:
                labels.append(self.model.config.id2label[str(label_id)])
            else:
                labels.append('O')
        
        # Filter out special tokens
        filtered_tokens = []
        filtered_labels = []
        
        for token, label in zip(tokens, labels):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                filtered_tokens.append(token)
                filtered_labels.append(label)
        
        return filtered_tokens, filtered_labels, logits
    
    def extract_entities_batch(self, texts: List[str]) -> List[EntityExtractionResult]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of EntityExtractionResult objects
        """
        results = []
        for text in texts:
            try:
                result = self.extract_entities(text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process text: {e}")
                results.append(EntityExtractionResult(
                    text=text,
                    entities=[],
                    tokens=[],
                    labels=[]
                ))
        
        return results
    
    def get_entities_by_type(self, result: EntityExtractionResult, entity_type: str) -> List[Entity]:
        """
        Get entities of a specific type from extraction result.
        
        Args:
            result: Entity extraction result
            entity_type: Type of entities to filter (e.g., 'GENERAL_INFO', 'ORE_COMPONENT')
            
        Returns:
            List of entities of the specified type
        """
        return [entity for entity in result.entities if entity.label == entity_type]
    
    def get_all_entity_types(self, result: EntityExtractionResult) -> List[str]:
        """
        Get all unique entity types found in the extraction result.
        
        Args:
            result: Entity extraction result
            
        Returns:
            List of unique entity types
        """
        return list(set(entity.label for entity in result.entities))
    
    def format_entities_as_text(self, result: EntityExtractionResult) -> str:
        """
        Format extracted entities as readable text.
        
        Args:
            result: Entity extraction result
            
        Returns:
            Formatted string with entities
        """
        if not result.entities:
            return "No entities found"
        
        formatted = []
        for entity in result.entities:
            formatted.append(f"{entity.text} ({entity.label}, confidence: {entity.confidence:.3f})")
        
        return "\n".join(formatted)


def load_bert_ner_model(model_path: str) -> BertNerInference:
    """
    Load a trained BERT NER model for inference.
    
    Args:
        model_path: Path to the trained model directory
        
    Returns:
        BertNerInference instance
    """
    return BertNerInference(Path(model_path))
