"""
QA Pair Generator for Instruct Fine-Tuning.

This module generates question-answer pairs based on extracted geological entities
to create training data for instruction fine-tuning of language models.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from geomas.core.logging.logger import get_logger

_log = get_logger("QA_GENERATOR")


@dataclass
class QAPair:
    """Question-Answer pair with metadata."""
    
    question: str
    answer: str
    entity_type: str
    source_text: str
    context: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class QAGenerator:
    """
    Generate question-answer pairs from extracted geological entities.
    
    Uses template-based approach with entity type-specific question patterns.
    """
    
    # Question templates for each entity type
    QUESTION_TEMPLATES = {
        "GENERAL_INFO": [
            "Какая общая информация о месторождении {entity}?",
            "Расскажи об общих сведениях о {entity}.",
            "Что известно о месторождении {entity}?",
            "Опиши общую информацию о {entity}.",
        ],
        "GEODYNAMIC": [
            "Каковы геодинамические характеристики {entity}?",
            "Опиши геодинамические условия {entity}.",
            "Какие геодинамические процессы характерны для {entity}?",
        ],
        "GEO_CHEMICAL": [
            "Каковы геохимические признаки {entity}?",
            "Опиши геохимические характеристики {entity}.",
            "Какой геохимический состав имеет {entity}?",
        ],
        "METALLOGENIC_CHAR": [
            "Каковы металлогенические характеристики {entity}?",
            "Опиши металлогению {entity}.",
            "Какие металлогенические особенности имеет {entity}?",
        ],
        "METASOMATIC": [
            "Какие метасоматические изменения характерны для {entity}?",
            "Опиши метасоматические процессы в {entity}.",
            "Какие метасоматические породы присутствуют в {entity}?",
        ],
        "MINERALOGICAL": [
            "Каковы минералогические признаки {entity}?",
            "Опиши минеральный состав {entity}.",
            "Какие минералы присутствуют в {entity}?",
        ],
        "ORE_COMPONENT": [
            "Какой полезный компонент содержится в {entity}?",
            "Каковы содержания полезных компонентов в {entity}?",
            "Опиши рудный состав {entity}.",
            "Какие содержания золота и серебра в {entity}?",
        ],
        "RESOURCE_POTENTIAL": [
            "Каков ресурсный потенциал {entity}?",
            "Опиши запасы {entity}.",
            "Какие запасы полезных ископаемых в {entity}?",
        ],
        "ORE_FORMATION": [
            "К какой рудной формации относится {entity}?",
            "Опиши геолого-промышленный тип оруденения {entity}.",
            "Какой тип оруденения характерен для {entity}?",
        ],
        "ORE_BODIES": [
            "Опиши морфологию и размеры рудных тел {entity}.",
            "Каковы условия залегания рудных зон {entity}?",
            "Какова форма и размеры рудных тел в {entity}?",
        ],
        "ORE_COMPOSITION": [
            "Каков состав руд {entity}?",
            "Опиши минеральный состав руд {entity}.",
            "Какие минералы входят в состав руд {entity}?",
        ],
        "STRATIGRAPHY": [
            "Опиши стратиграфию {entity}.",
            "Какие типы пород характерны для {entity}?",
            "Какой стратиграфический разрез имеет {entity}?",
        ],
        "STRUCTURAL_TECTONIC": [
            "Каковы структурно-тектонические характеристики {entity}?",
            "Опиши тектоническое строение {entity}.",
            "Какие структурные элементы характерны для {entity}?",
        ],
        "TECHNOLOGICAL": [
            "Какие технологические характеристики имеет {entity}?",
            "Опиши технологию обогащения руд {entity}.",
            "Какая схема переработки применяется для {entity}?",
        ],
        "FORMATION_CONDITIONS": [
            "Каковы условия формирования {entity}?",
            "Опиши генезис {entity}.",
            "При каких условиях образовался {entity}?",
        ],
        "STUDY_INFO": [
            "Какова изученность {entity}?",
            "Опиши историю изучения {entity}.",
            "Какие работы проводились на {entity}?",
        ],
        "INFO_SOURCES": [
            "Какие источники информации о {entity}?",
            "Опиши библиографию по {entity}.",
        ],
    }
    
    # Generic question templates for unknown entity types
    GENERIC_TEMPLATES = [
        "Что можно сказать о {entity}?",
        "Опиши {entity}.",
        "Какая информация доступна о {entity}?",
    ]
    
    # Stopwords and meaningless patterns to filter
    STOPWORDS = {
        "и", "в", "на", "с", "по", "для", "от", "до", "из", "к", "о", "об",
        "но", "а", "же", "или", "при", "за", "под", "над", "между", "через",
        "т", "г", "м", "км", "кв", "куб", "тыс", "млн"
    }
    
    def __init__(
        self,
        add_context: bool = True,
        max_context_length: int = 700,
        min_entity_length: int = 5,
        filter_entities: bool = True
    ):
        """
        Initialize QA Generator.
        
        Args:
            add_context: Whether to add surrounding context to answers
            max_context_length: Maximum length of context in characters
            min_entity_length: Minimum length for entity to be considered valid
            filter_entities: Whether to filter out low-quality entities
        """
        self.add_context = add_context
        self.max_context_length = max_context_length
        self.min_entity_length = min_entity_length
        self.filter_entities = filter_entities
        _log.info(
            f"QA Generator initialized (add_context={add_context}, "
            f"filter={filter_entities}, min_length={min_entity_length})"
        )
    
    def _is_valid_entity(self, entity_text: str) -> bool:
        """
        Check if entity is valid and meaningful.
        
        Args:
            entity_text: Text of the entity
            
        Returns:
            True if entity is valid, False otherwise
        """
        text = entity_text.strip()
        
        # Check minimum length
        if len(text) < self.min_entity_length:
            return False
        
        # Filter out single stopwords
        if text.lower() in self.STOPWORDS:
            return False
        
        # Filter out pure punctuation or numbers
        if not any(c.isalpha() for c in text):
            return False
        
        # Filter out incomplete words (starting with lowercase or punctuation)
        if text and not (text[0].isupper() or text[0].isdigit()):
            return False
        
        # Filter out entities that are just numbers with units
        words = text.split()
        if len(words) <= 2 and any(unit in text.lower() for unit in ['г/т', 'кв.', 'куб.', 'тыс.', 'млн']):
            # Allow if it contains actual values, but not just units
            if not any(c.isdigit() for c in text):
                return False
        
        # Filter out very long entities (likely extraction errors)
        if len(text) > 300:
            return False
        
        return True
    
    def _clean_entity_text(self, entity_text: str) -> str:
        """
        Clean entity text for use in questions.
        
        Args:
            entity_text: Raw entity text
            
        Returns:
            Cleaned entity text
        """
        text = entity_text.strip()
        
        # Remove trailing punctuation
        while text and text[-1] in '.,;:!?-–—':
            text = text[:-1].strip()
        
        # Remove leading punctuation
        while text and text[0] in '.,;:!?-–—':
            text = text[1:].strip()
        
        return text
    
    def _create_question_entity_ref(self, entity_text: str, max_length: int = 50) -> str:
        """
        Create a reference to entity for question (shortened if too long).
        
        Args:
            entity_text: Full entity text
            max_length: Maximum length for entity reference
            
        Returns:
            Entity reference for question
        """
        text = self._clean_entity_text(entity_text)
        
        # If short enough, use as-is
        if len(text) <= max_length:
            return text
        
        # Extract key terms (nouns, capitalized words)
        words = text.split()
        key_words = []
        
        for word in words:
            # Keep capitalized words (proper nouns, location names)
            if word and word[0].isupper():
                key_words.append(word)
            # Stop if we have enough
            if len(' '.join(key_words)) >= max_length:
                break
        
        # If we found key words, use them
        if key_words:
            result = ' '.join(key_words)
            # Truncate if still too long
            if len(result) > max_length:
                result = result[:max_length] + '...'
            return result
        
        # Otherwise, just truncate
        return text[:max_length] + '...'
    
    def generate_qa_pairs(
        self,
        entities: List[Dict],
        source_text: str,
        num_pairs_per_entity: int = 2
    ) -> List[QAPair]:
        """
        Generate QA pairs from extracted entities.
        
        Args:
            entities: List of extracted entities with 'text', 'label', 'start', 'end'
            source_text: Original source text
            num_pairs_per_entity: Number of QA pairs to generate per entity
            
        Returns:
            List of QAPair objects
        """
        qa_pairs = []
        filtered_count = 0
        
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('label', 'UNKNOWN')
            
            # Apply filtering if enabled
            if self.filter_entities and not self._is_valid_entity(entity_text):
                filtered_count += 1
                continue
            
            # Get context around entity
            context = self._extract_context(
                source_text,
                entity.get('start', 0),
                entity.get('end', len(entity_text))
            )
            
            # Clean entity text
            cleaned_entity = self._clean_entity_text(entity_text)
            
            # Generate multiple QA pairs for this entity
            for _ in range(num_pairs_per_entity):
                question = self._generate_question(cleaned_entity, entity_type)
                answer = self._generate_answer(cleaned_entity, entity_type, context, source_text)
                
                qa_pair = QAPair(
                    question=question,
                    answer=answer,
                    entity_type=entity_type,
                    source_text=source_text[:100] + "...",  # Store preview
                    context=context if self.add_context else None
                )
                
                qa_pairs.append(qa_pair)
        
        if filtered_count > 0:
            _log.info(f"Filtered out {filtered_count} low-quality entities")
        
        _log.info(f"Generated {len(qa_pairs)} QA pairs from {len(entities) - filtered_count} valid entities")
        return qa_pairs
    
    def _generate_question(self, entity_text: str, entity_type: str) -> str:
        """
        Generate question based on entity and its type.
        
        Args:
            entity_text: Text of the entity
            entity_type: Type of the entity
            
        Returns:
            Generated question
        """
        # Get templates for this entity type
        templates = self.QUESTION_TEMPLATES.get(entity_type, self.GENERIC_TEMPLATES)
        
        # Select random template
        template = random.choice(templates)
        
        # Create entity reference (shortened if too long)
        entity_ref = self._create_question_entity_ref(entity_text, max_length=50)
        
        # Format question
        question = template.format(entity=entity_ref)
        
        return question
    
    def _generate_answer(
        self,
        entity_text: str,
        entity_type: str,
        context: str,
        source_text: str
    ) -> str:
        """
        Generate answer based on entity and context.
        
        Args:
            entity_text: Text of the entity
            entity_type: Type of the entity
            context: Context around the entity
            source_text: Full source text
            
        Returns:
            Generated answer
        """
        # If context is available, use it as the answer base
        if self.add_context and context:
            # Clean context
            answer = context.strip()
            
            # Don't prepend entity if:
            # 1. Entity is already at the start of context
            # 2. Entity is very long (>100 chars)
            # 3. Context already contains the entity
            if (not answer.startswith(entity_text) and 
                len(entity_text) <= 100 and 
                entity_text not in answer[:200]):
                # Only prepend if entity is meaningful and not just a fragment
                if len(entity_text) > 10:
                    answer = f"{entity_text}. {answer}"
        else:
            # Use context from source text if no explicit context
            answer = entity_text
        
        return answer
    
    def _extract_context(
        self,
        text: str,
        start: int,
        end: int,
        window_size: int = 300
    ) -> str:
        """
        Extract context around entity, avoiding sentence breaks.
        
        Args:
            text: Source text
            start: Entity start position
            end: Entity end position
            window_size: Number of characters before/after entity
            
        Returns:
            Context string
        """
        # Calculate initial context boundaries
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        
        # Try to extend to sentence boundaries
        # Look for sentence start (after . ! ?)
        for i in range(context_start, start):
            if i > 0 and text[i-1] in '.!?' and text[i].isupper():
                context_start = i
                break
        
        # Look for sentence end
        for i in range(end, context_end):
            if text[i] in '.!?':
                context_end = min(i + 1, len(text))
                break
        
        # Extract context
        context = text[context_start:context_end].strip()
        
        # Limit to max context length, but try to keep complete sentences
        if len(context) > self.max_context_length:
            # Find last sentence boundary within limit
            truncate_pos = self.max_context_length
            for i in range(self.max_context_length - 1, max(0, self.max_context_length - 100), -1):
                if i < len(context) and context[i] in '.!?':
                    truncate_pos = i + 1
                    break
            
            context = context[:truncate_pos].strip()
            
            # Add ellipsis if truncated mid-sentence
            if truncate_pos < len(context) and context[-1] not in '.!?':
                context += "..."
        
        return context
    
    def save_qa_pairs(self, qa_pairs: List[QAPair], output_path: Path):
        """
        Save QA pairs to JSON file.
        
        Args:
            qa_pairs: List of QA pairs
            output_path: Path to save the file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [qa_pair.to_dict() for qa_pair in qa_pairs]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        _log.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
    
    def load_qa_pairs(self, input_path: Path) -> List[QAPair]:
        """
        Load QA pairs from JSON file.
        
        Args:
            input_path: Path to the file
            
        Returns:
            List of QA pairs
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = [QAPair(**item) for item in data]
        
        _log.info(f"Loaded {len(qa_pairs)} QA pairs from {input_path}")
        return qa_pairs


def generate_qa_from_chunks(
    chunks_path: Path,
    model_path: Path,
    output_path: Path,
    num_pairs_per_entity: int = 2,
    add_context: bool = True
) -> int:
    """
    Generate QA pairs from chunks using BERT NER model.
    
    Args:
        chunks_path: Path to chunks.json file
        model_path: Path to trained BERT NER model
        output_path: Path to save QA pairs
        num_pairs_per_entity: Number of QA pairs per entity
        add_context: Whether to add context to answers
        
    Returns:
        Number of QA pairs generated
    """
    from geomas.core.inference.bert_ner_inference import load_bert_ner_model
    
    _log.info(f"Generating QA pairs from {chunks_path}")
    _log.info(f"Using model: {model_path}")
    
    # Load BERT NER model
    _log.info("Loading BERT NER model...")
    ner_model = load_bert_ner_model(str(model_path))
    
    # Load chunks
    _log.info("Loading chunks...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(chunks_data, list):
        chunks = chunks_data
    else:
        chunks = chunks_data.get('chunks', [])
    
    _log.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize QA generator
    qa_generator = QAGenerator(add_context=add_context)
    
    # Generate QA pairs for each chunk
    all_qa_pairs = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        
        if not text:
            continue
        
        _log.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Extract entities
        result = ner_model.extract_entities(text)
        
        if not result.entities:
            _log.warning(f"No entities found in chunk {i+1}")
            continue
        
        # Convert entities to dict format
        entities = [
            {
                'text': entity.text,
                'label': entity.label,
                'start': entity.start,
                'end': entity.end,
                'confidence': entity.confidence
            }
            for entity in result.entities
        ]
        
        # Generate QA pairs
        qa_pairs = qa_generator.generate_qa_pairs(
            entities,
            text,
            num_pairs_per_entity=num_pairs_per_entity
        )
        
        all_qa_pairs.extend(qa_pairs)
    
    # Save QA pairs
    qa_generator.save_qa_pairs(all_qa_pairs, output_path)
    
    _log.info(f"Total QA pairs generated: {len(all_qa_pairs)}")
    
    return len(all_qa_pairs)

