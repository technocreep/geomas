"""
BERT NER Trainer for Geological Entity Recognition.

This module provides functionality to fine-tune BERT models for Named Entity Recognition
on geological documents using the converted Label Studio annotations.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    __version__ as transformers_version
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from geomas.core.logging.logger import get_logger

logger = get_logger()


def _get_eval_strategy_param():
    """Get the correct parameter name for evaluation strategy based on transformers version."""
    # Check if we need the old or new parameter name
    version_parts = transformers_version.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    
    if major >= 4 and minor >= 19:
        return "eval_strategy"
    else:
        return "evaluation_strategy"


@dataclass
class BertNerConfig:
    """Configuration for BERT NER training."""
    model_name: str = "DeepPavlov/rubert-base-cased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "./bert_ner_output"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    dataloader_num_workers: int = 4
    # Early stopping configuration
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


class BertNerDataset(Dataset):
    """Dataset for BERT NER training."""
    
    def __init__(self, examples: List[Dict], tokenizer, label2id: Dict[str, int], max_length: int = 512):
        """
        Initialize BERT NER dataset.
        
        Args:
            examples: List of training examples with 'tokens' and 'labels'
            tokenizer: Hugging Face tokenizer
            label2id: Mapping from labels to IDs
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        tokens = example['tokens']
        labels = example['labels']
        
        # Tokenize the input
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokenizer output
        word_ids = encoding.word_ids()
        aligned_labels = []
        
        for word_id in word_ids:
            if word_id is None:
                # Special tokens (CLS, SEP, PAD)
                aligned_labels.append(-100)
            else:
                # Map to label ID
                if word_id < len(labels):
                    label = labels[word_id]
                    aligned_labels.append(self.label2id.get(label, self.label2id['O']))
                else:
                    # Padding
                    aligned_labels.append(-100)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


class BertNerTrainer:
    """BERT NER Trainer for geological entity recognition."""
    
    def __init__(self, config: BertNerConfig):
        """
        Initialize BERT NER trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        
    def load_training_data(self, data_path: Path) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Load training data from converted annotations.
        
        Args:
            data_path: Path to the BERT training data directory
            
        Returns:
            Tuple of (examples, label2id mapping)
        """
        # Load training data
        training_file = data_path / "bert_training_data.json"
        if not training_file.exists():
            raise FileNotFoundError(f"Training data not found: {training_file}")
            
        with open(training_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        # Create label mappings
        all_labels = set()
        for example in examples:
            all_labels.update(example['labels'])
        
        all_labels = sorted(list(all_labels))
        self.label2id = {label: idx for idx, label in enumerate(all_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        logger.info(f"Loaded {len(examples)} training examples")
        logger.info(f"Found {len(all_labels)} unique labels: {all_labels}")
        
        return examples, self.label2id
    
    def setup_model_and_tokenizer(self):
        """Setup BERT model and tokenizer."""
        logger.info(f"Loading model and tokenizer: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        
        logger.info(f"Model loaded with {len(self.label2id)} labels")
    
    def create_datasets(self, examples: List[Dict]) -> Tuple[BertNerDataset, BertNerDataset]:
        """
        Create training and validation datasets.
        
        Args:
            examples: Training examples
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(examples))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        train_dataset = BertNerDataset(
            train_examples, 
            self.tokenizer, 
            self.label2id, 
            self.config.max_length
        )
        
        val_dataset = BertNerDataset(
            val_examples, 
            self.tokenizer, 
            self.label2id, 
            self.config.max_length
        )
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten for metrics
        true_predictions_flat = [p for pred in true_predictions for p in pred]
        true_labels_flat = [l for label in true_labels for l in label]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels_flat, true_predictions_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels_flat, true_predictions_flat, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, data_path: Path):
        """
        Train BERT NER model.
        
        Args:
            data_path: Path to training data
        """
        logger.info("Starting BERT NER training...")
        
        # Load data
        examples, label2id = self.load_training_data(data_path)
        
        # Setup model
        self.setup_model_and_tokenizer()
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(examples)
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save label mappings
        with open(output_dir / "label2id.json", 'w', encoding='utf-8') as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)
        
        with open(output_dir / "id2label.json", 'w', encoding='utf-8') as f:
            json.dump(self.id2label, f, ensure_ascii=False, indent=2)
        
        # Training arguments
        eval_param = _get_eval_strategy_param()
        training_args_kwargs = {
            "output_dir": str(output_dir),
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "learning_rate": self.config.learning_rate,
            "logging_dir": str(output_dir / "logs"),
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",  # Use eval_loss for early stopping
            "greater_is_better": False,  # Lower loss is better
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "fp16": self.config.fp16,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "report_to": None,  # Disable wandb/tensorboard
            "save_total_limit": 1
        }
        
        # Add the correct evaluation strategy parameter
        training_args_kwargs[eval_param] = "steps"

        training_args = TrainingArguments(**training_args_kwargs)
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Create early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=self.config.early_stopping_threshold
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping_callback],
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed! Model saved to: {output_dir}")
        
        return trainer


def train_bert_ner(
    data_path: str,
    model_name: str = "DeepPavlov/rubert-base-cased",
    output_dir: str = "./bert_ner_output",
    **kwargs
):
    """
    Train BERT NER model for geological entities.
    
    Args:
        data_path: Path to BERT training data directory
        model_name: BERT model name to fine-tune
        output_dir: Directory to save the trained model
        **kwargs: Additional training parameters
    """
    config = BertNerConfig(
        model_name=model_name,
        output_dir=output_dir,
        **kwargs
    )
    
    trainer = BertNerTrainer(config)
    return trainer.train(Path(data_path))
