"""
Annotation converter module for Label Studio to BERT format conversion.

This module handles conversion of Label Studio annotations to BERT-compatible
training format for geological entity recognition.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from geomas.core.logging.logger import get_logger

logger = get_logger()


@dataclass
class AnnotationData:
    """Data class for annotation information."""
    text: str
    entities: List[Dict[str, Any]]
    document_id: str
    source_file: str


@dataclass
class BertTrainingExample:
    """Data class for BERT training example."""
    tokens: List[str]
    labels: List[str]
    document_id: str
    original_text: str


class LabelStudioConverter:
    """
    Converts Label Studio annotations to BERT training format.
    
    This converter processes geological annotations from Label Studio
    and creates training data suitable for BERT NER fine-tuning.
    """
    
    def __init__(self, label_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the converter.
        
        Args:
            label_mapping: Optional mapping from Label Studio labels to BERT labels.
                         If None, uses default mapping for geological entities.
        """
        self.label_mapping = label_mapping or self._get_default_label_mapping()
        self.bio_labels = self._create_bio_labels()
        
    def _get_default_label_mapping(self) -> Dict[str, str]:
        """
        Get default mapping from Label Studio labels to BERT labels.
        
        Returns:
            Dictionary mapping Label Studio labels to BERT entity types.
        """
        # Raw labels with invisible characters
        raw_labels = {
            "Общие сведения (лицензия, положение, инфраструктура, физико-географические условия (рельеф, водн.режим, мерзлота и т.п.))​": "GENERAL_INFO",
            "Изученность – общая информация (объемы и виды работ)​": "STUDY_INFO",
            "Рудная формация/ Геолого-промышленный тип оруденения​": "ORE_FORMATION",
            "Полезный компонент руд​": "ORE_COMPONENT",
            "Ресурсный потенциал​": "RESOURCE_POTENTIAL",
            "Металлогенические характеристики​": "METALLOGENIC_CHAR",
            "Структурно-тектонические характеристики​": "STRUCTURAL_CHAR",
            "Рудные зоны / тела (морфология, размеры и условия залегания рудных зон и тел)​": "ORE_BODIES",
            "Геохимические признаки​": "GEO_CHEMICAL",
            "Состав руд​": "ORE_COMPOSITION",
            "Минералогические признаки​": "MINERALOGICAL",
            "Метасоматические изменения ​": "METASOMATIC",
            "Условия формирования ​": "FORMATION_CONDITIONS",
            "Технологические признаки / обогащение / горное дело": "TECHNOLOGICAL",
            "Источники информации​": "SOURCES",
            "Стратиграфия и типы пород ​": "STRATIGRAPHY",
            "Геодинамические характеристики​": "GEODYNAMIC"
        }
        
        # Create cleaned mapping
        cleaned_mapping = {}
        for raw_label, bert_label in raw_labels.items():
            cleaned_label = self._clean_label(raw_label)
            cleaned_mapping[cleaned_label] = bert_label
        
        return cleaned_mapping
    
    def _create_bio_labels(self) -> List[str]:
        """
        Create BIO labels for NER training.
        
        Returns:
            List of BIO labels for all entity types.
        """
        bio_labels = ["O"]  # Outside entity
        
        for bert_label in set(self.label_mapping.values()):
            bio_labels.extend([f"B-{bert_label}", f"I-{bert_label}"])
            
        return bio_labels
    
    def _clean_label(self, label: str) -> str:
        """
        Clean label from invisible characters and normalize.
        
        Args:
            label: Raw label string.
            
        Returns:
            Cleaned label string.
        """
        # Remove zero-width characters and normalize
        import unicodedata
        cleaned = label.strip()
        # Remove zero-width space (200b) and other invisible characters
        cleaned = cleaned.replace('\u200b', '').replace('\u200d', '').replace('\u200c', '')
        # Normalize unicode
        cleaned = unicodedata.normalize('NFC', cleaned)
        return cleaned
    
    def load_annotation_file(self, file_path: Path) -> List[AnnotationData]:
        """
        Load and parse Label Studio annotation file.
        
        Args:
            file_path: Path to the Label Studio JSON file.
            
        Returns:
            List of AnnotationData objects.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
        
        annotations = []
        
        for item in data:
            if 'data' not in item or 'annotations' not in item:
                logger.warning(f"Skipping item without required fields: {item.get('id', 'unknown')}")
                continue
                
            text = item['data'].get('text', '')
            if not text:
                logger.warning(f"Skipping item with empty text: {item.get('id', 'unknown')}")
                continue
            
            entities = []
            for annotation in item['annotations']:
                if 'result' not in annotation:
                    continue
                    
                for result in annotation['result']:
                    if 'value' not in result:
                        continue
                        
                    value = result['value']
                    if 'labels' not in value or 'start' not in value or 'end' not in value:
                        continue
                    
                    # Use the first label if multiple labels exist
                    label = value['labels'][0] if value['labels'] else None
                    if not label:
                        continue
                    
                    # Clean label from invisible characters
                    label = self._clean_label(label)
                    
                    entities.append({
                        'start': value['start'],
                        'end': value['end'],
                        'text': value.get('text', ''),
                        'label': label
                    })
            
            annotations.append(AnnotationData(
                text=text,
                entities=entities,
                document_id=str(item.get('id', '')),
                source_file=str(file_path)
            ))
        
        logger.info(f"Loaded {len(annotations)} annotations from {file_path}")
        return annotations
    
    def convert_to_bert_format(self, annotation_data: AnnotationData) -> BertTrainingExample:
        """
        Convert annotation data to BERT training format.
        
        Args:
            annotation_data: AnnotationData object to convert.
            
        Returns:
            BertTrainingExample with tokens and BIO labels.
        """
        text = annotation_data.text
        
        # Simple tokenization (can be improved with proper tokenizer)
        tokens = self._simple_tokenize(text)
        
        # Initialize all labels as 'O' (outside entity)
        labels = ['O'] * len(tokens)
        
        # Create mapping from character positions to token indices
        char_to_token = self._create_char_to_token_mapping(text, tokens)
        
        # Process each entity
        for entity in annotation_data.entities:
            label = entity['label']
            start_char = entity['start']
            end_char = entity['end']
            
            # Map to BERT label
            bert_label = self.label_mapping.get(label, 'O')
            if bert_label == 'O':
                continue
            
            # Find token indices for this entity with fallback search
            start_token = char_to_token.get(start_char)
            end_token = char_to_token.get(end_char - 1)
            
            # If direct mapping fails, try to find nearby tokens
            if start_token is None:
                for offset in range(1, 20):  # Search within 20 characters
                    if start_char + offset in char_to_token:
                        start_token = char_to_token[start_char + offset]
                        break
                    if start_char - offset in char_to_token:
                        start_token = char_to_token[start_char - offset]
                        break
            
            if end_token is None:
                for offset in range(1, 20):  # Search within 20 characters
                    if end_char - 1 + offset in char_to_token:
                        end_token = char_to_token[end_char - 1 + offset]
                        break
                    if end_char - 1 - offset in char_to_token:
                        end_token = char_to_token[end_char - 1 - offset]
                        break
            
            if start_token is None or end_token is None:
                logger.warning(f"Could not map entity to tokens: {entity}")
                continue
            
            # Apply BIO labeling
            if start_token < len(labels):
                labels[start_token] = f"B-{bert_label}"
            
            for i in range(start_token + 1, min(end_token + 1, len(labels))):
                labels[i] = f"I-{bert_label}"
        
        return BertTrainingExample(
            tokens=tokens,
            labels=labels,
            document_id=annotation_data.document_id,
            original_text=text
        )
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """
        Tokenize using BERT tokenizer to match model training.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of tokens.
        """
        # Use BERT tokenizer for consistency with model
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
            tokens = tokenizer.tokenize(text)
            return tokens
        except ImportError:
            logger.warning("transformers not available, falling back to simple tokenization")
            # Fallback to simple tokenization
            tokens = re.findall(r'\S+', text)
            result = []
            for token in tokens:
                sub_tokens = re.findall(r'\w+|[^\w\s]', token)
                result.extend(sub_tokens)
            return result
    
    def _create_char_to_token_mapping(self, text: str, tokens: List[str]) -> Dict[int, int]:
        """
        Create mapping from character positions to token indices.
        
        Args:
            text: Original text.
            tokens: List of tokens.
            
        Returns:
            Dictionary mapping character positions to token indices.
        """
        char_to_token = {}
        text_pos = 0
        token_idx = 0
        
        # Create a more robust mapping by reconstructing tokens from text
        while text_pos < len(text) and token_idx < len(tokens):
            # Skip whitespace
            while text_pos < len(text) and text[text_pos].isspace():
                text_pos += 1
            
            if text_pos >= len(text):
                break
                
            current_token = tokens[token_idx]
            
            # Try to find the token at current position
            if text[text_pos:].startswith(current_token):
                # Map each character in the token
                for i in range(len(current_token)):
                    char_to_token[text_pos + i] = token_idx
                text_pos += len(current_token)
                token_idx += 1
            else:
                # Try to find token nearby (within reasonable distance)
                found = False
                for search_start in range(max(0, text_pos - 10), min(len(text), text_pos + 50)):
                    if text[search_start:].startswith(current_token):
                        for i in range(len(current_token)):
                            char_to_token[search_start + i] = token_idx
                        text_pos = search_start + len(current_token)
                        token_idx += 1
                        found = True
                        break
                
                if not found:
                    # Skip this token if we can't find it
                    token_idx += 1
        
        return char_to_token
    
    def process_annotations_directory(self, annotations_dir: Path, output_dir: Path) -> None:
        """
        Process all annotation files in a directory.
        
        Args:
            annotations_dir: Directory containing Label Studio JSON files.
            output_dir: Directory to save BERT training files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_examples = []
        
        # Process all JSON files recursively
        for json_file in annotations_dir.rglob("*.json"):
            if json_file.name == "реестр аннотаций.xlsx":
                continue  # Skip Excel files
                
            logger.info(f"Processing {json_file}")
            
            try:
                annotations = self.load_annotation_file(json_file)
                
                # Extract individual entities as separate examples
                for annotation_data in annotations:
                    # Create separate example for EACH entity
                    for entity in annotation_data.entities:
                        # Extract entity text from original document using start/end positions
                        entity_text = annotation_data.text[entity['start']:entity['end']]
                        
                        # Adjust entity positions to be relative to extracted text (starting at 0)
                        adjusted_entity = {
                            'start': 0,
                            'end': len(entity_text),
                            'text': entity_text,
                            'label': entity['label']
                        }
                        
                        # Create new annotation data with just this entity
                        entity_annotation = AnnotationData(
                            text=entity_text,           # Text extracted from original document
                            entities=[adjusted_entity], # Entity with adjusted positions
                            document_id=f"{annotation_data.document_id}_entity_{entity['start']}",
                            source_file=annotation_data.source_file
                        )
                        
                        bert_example = self.convert_to_bert_format(entity_annotation)
                        all_examples.append(bert_example)
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        # Save training data
        self._save_training_data(all_examples, output_dir)
        
        logger.info(f"Processed {len(all_examples)} training examples")
    
    def _save_training_data(self, examples: List[BertTrainingExample], output_dir: Path) -> None:
        """
        Save training data in various formats.
        
        Args:
            examples: List of BERT training examples.
            output_dir: Output directory.
        """
        # Save as JSON for easy inspection
        json_data = []
        for example in examples:
            json_data.append({
                'tokens': example.tokens,
                'labels': example.labels,
                'document_id': example.document_id,
                'original_text': example.original_text
            })
        
        json_file = output_dir / "bert_training_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Save as CoNLL format for standard NER training
        conll_file = output_dir / "bert_training_data.conll"
        with open(conll_file, 'w', encoding='utf-8') as f:
            for example in examples:
                for token, label in zip(example.tokens, example.labels):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")  # Empty line between sentences
        
        # Save label mappings
        mappings_file = output_dir / "label_mappings.json"
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump({
                'label_studio_to_bert': self.label_mapping,
                'bio_labels': self.bio_labels
            }, f, ensure_ascii=False, indent=2)
        
        # Save statistics
        stats = self._calculate_statistics(examples)
        stats_file = output_dir / "training_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved training data to {output_dir}")
    
    def _calculate_statistics(self, examples: List[BertTrainingExample]) -> Dict[str, Any]:
        """
        Calculate training data statistics.
        
        Args:
            examples: List of training examples.
            
        Returns:
            Dictionary with statistics.
        """
        label_counts = defaultdict(int)
        total_tokens = 0
        
        for example in examples:
            total_tokens += len(example.tokens)
            for label in example.labels:
                label_counts[label] += 1
        
        return {
            'total_examples': len(examples),
            'total_tokens': total_tokens,
            'average_tokens_per_example': total_tokens / len(examples) if examples else 0,
            'label_distribution': dict(label_counts),
            'unique_labels': len(label_counts)
        }


def main():
    """Main function for testing the converter."""
    # Example usage
    annotations_dir = Path("АННОТАЦИИ")
    output_dir = Path("bert_training_data")
    
    converter = LabelStudioConverter()
    converter.process_annotations_directory(annotations_dir, output_dir)


if __name__ == "__main__":
    main()
