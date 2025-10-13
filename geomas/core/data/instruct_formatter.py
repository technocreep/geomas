"""
Instruct Dataset Formatter for Fine-Tuning.

This module formats QA pairs into instruction-following format
suitable for supervised fine-tuning (SFT) of language models.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from geomas.core.logging.logger import get_logger

_log = get_logger("INSTRUCT_FORMATTER")


@dataclass
class InstructExample:
    """Instruction-following example for fine-tuning."""
    
    instruction: str
    input: str
    output: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }
    
    def to_chat_format(self, system_prompt: Optional[str] = None) -> Dict:
        """
        Convert to chat format (for ChatML, Llama, etc.).
        
        Args:
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with 'messages' key containing conversation
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Combine instruction and input into user message
        user_content = self.instruction
        if self.input:
            user_content = f"{self.instruction}\n\nКонтекст: {self.input}"
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        messages.append({
            "role": "assistant",
            "content": self.output
        })
        
        return {"messages": messages}


class InstructFormatter:
    """
    Format QA pairs into instruction-following format.
    
    Supports multiple output formats:
    - Alpaca format (instruction, input, output)
    - Chat format (messages with roles)
    - Custom formats
    """
    
    # Default system prompt for geological domain
    DEFAULT_SYSTEM_PROMPT = (
        "Ты — геологический эксперт-ассистент. Твоя задача — отвечать на вопросы "
        "о месторождениях полезных ископаемых, основываясь на предоставленной информации. "
        "Давай точные, информативные ответы, используя геологическую терминологию."
    )
    
    # Instruction templates for different entity types
    INSTRUCTION_TEMPLATES = {
        "GENERAL_INFO": "Предоставь общую информацию о месторождении.",
        "ORE_COMPONENT": "Опиши содержания полезных компонентов в руде.",
        "RESOURCE_POTENTIAL": "Предоставь информацию о запасах и ресурсном потенциале.",
        "ORE_FORMATION": "Опиши геолого-промышленный тип оруденения.",
        "MINERALOGICAL": "Опиши минералогические характеристики.",
        "TECHNOLOGICAL": "Предоставь информацию о технологических характеристиках и методах обогащения.",
        "STRATIGRAPHY": "Опиши стратиграфию и типы пород.",
        "STRUCTURAL_TECTONIC": "Опиши структурно-тектонические характеристики.",
        "ORE_BODIES": "Опиши морфологию и условия залегания рудных тел.",
        "ORE_COMPOSITION": "Опиши состав руд.",
        "GEODYNAMIC": "Опиши геодинамические характеристики.",
        "GEO_CHEMICAL": "Опиши геохимические признаки.",
        "METALLOGENIC_CHAR": "Опиши металлогенические характеристики.",
        "METASOMATIC": "Опиши метасоматические изменения.",
        "FORMATION_CONDITIONS": "Опиши условия формирования.",
        "STUDY_INFO": "Предоставь информацию об изученности месторождения.",
        "INFO_SOURCES": "Предоставь информацию об источниках данных.",
    }
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        use_context: bool = True,
        format_type: str = "alpaca"
    ):
        """
        Initialize Instruct Formatter.
        
        Args:
            system_prompt: Custom system prompt (uses default if None)
            use_context: Whether to include context in input field
            format_type: Output format type ('alpaca' or 'chat')
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.use_context = use_context
        self.format_type = format_type
        
        _log.info(f"Instruct Formatter initialized (format={format_type}, use_context={use_context})")
    
    def format_qa_pair(self, qa_pair: Dict) -> InstructExample:
        """
        Format a single QA pair into instruction format.
        
        Args:
            qa_pair: Dictionary with 'question', 'answer', 'entity_type', 'context'
            
        Returns:
            InstructExample object
        """
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        entity_type = qa_pair.get('entity_type', 'UNKNOWN')
        context = qa_pair.get('context', '')
        
        # Get instruction template based on entity type
        instruction = self.INSTRUCTION_TEMPLATES.get(
            entity_type,
            "Ответь на следующий вопрос о геологическом объекте."
        )
        
        # Combine instruction with question
        full_instruction = f"{instruction}\n\nВопрос: {question}"
        
        # Use context as input if available and enabled
        input_text = ""
        if self.use_context and context:
            input_text = context
        
        return InstructExample(
            instruction=full_instruction,
            input=input_text,
            output=answer
        )
    
    def format_dataset(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Format entire dataset of QA pairs.
        
        Args:
            qa_pairs: List of QA pair dictionaries
            
        Returns:
            List of formatted examples
        """
        formatted_examples = []
        
        for qa_pair in qa_pairs:
            try:
                instruct_example = self.format_qa_pair(qa_pair)
                
                if self.format_type == "chat":
                    formatted_examples.append(
                        instruct_example.to_chat_format(self.system_prompt)
                    )
                else:  # alpaca format
                    formatted_examples.append(instruct_example.to_dict())
                    
            except Exception as e:
                _log.warning(f"Failed to format QA pair: {e}")
                continue
        
        _log.info(f"Formatted {len(formatted_examples)} examples")
        return formatted_examples
    
    def save_dataset(
        self,
        qa_pairs: List[Dict],
        output_path: Path,
        split_ratio: float = 0.9
    ):
        """
        Format and save dataset, optionally splitting into train/val.
        
        Args:
            qa_pairs: List of QA pair dictionaries
            output_path: Path to save the formatted dataset
            split_ratio: Train/validation split ratio (0-1)
        """
        # Format all examples
        formatted_examples = self.format_dataset(qa_pairs)
        
        if split_ratio < 1.0:
            # Split into train and validation
            split_idx = int(len(formatted_examples) * split_ratio)
            train_examples = formatted_examples[:split_idx]
            val_examples = formatted_examples[split_idx:]
            
            # Save train set
            train_path = output_path.parent / f"{output_path.stem}_train{output_path.suffix}"
            self._save_json(train_examples, train_path)
            _log.info(f"Saved {len(train_examples)} training examples to {train_path}")
            
            # Save validation set
            val_path = output_path.parent / f"{output_path.stem}_val{output_path.suffix}"
            self._save_json(val_examples, val_path)
            _log.info(f"Saved {len(val_examples)} validation examples to {val_path}")
        else:
            # Save all as single file
            self._save_json(formatted_examples, output_path)
            _log.info(f"Saved {len(formatted_examples)} examples to {output_path}")
    
    def _save_json(self, data: List[Dict], path: Path):
        """Save data to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def format_qa_dataset(
    qa_pairs_path: Path,
    output_path: Path,
    format_type: str = "alpaca",
    split_ratio: float = 0.9,
    use_context: bool = True,
    system_prompt: Optional[str] = None
):
    """
    Format QA pairs dataset for instruct fine-tuning.
    
    Args:
        qa_pairs_path: Path to QA pairs JSON file
        output_path: Path to save formatted dataset
        format_type: Output format ('alpaca' or 'chat')
        split_ratio: Train/validation split ratio
        use_context: Whether to include context
        system_prompt: Custom system prompt
    """
    _log.info(f"Formatting QA dataset from {qa_pairs_path}")
    _log.info(f"Output format: {format_type}")
    _log.info(f"Split ratio: {split_ratio}")
    
    # Load QA pairs
    with open(qa_pairs_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    _log.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Initialize formatter
    formatter = InstructFormatter(
        system_prompt=system_prompt,
        use_context=use_context,
        format_type=format_type
    )
    
    # Format and save
    formatter.save_dataset(qa_pairs, output_path, split_ratio=split_ratio)
    
    _log.info("Dataset formatting completed")


