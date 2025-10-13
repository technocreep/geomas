"""
Supervised Fine-Tuning (SFT) Trainer for Instruction Following.

This module implements instruction fine-tuning for LLMs using Unsloth.
"""

from functools import partial
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
    FastModel
)
import mlflow

from geomas.core.config import prepare_settings
from geomas.core.data.dataset import get_dataset
from geomas.core.inference.evaluation import is_pure_llm
from geomas.core.logging import (
    get_logger,
    _init_mlflow_logging,
    posttrain_report,
    pretrain_report
)
from geomas.core.utils import PROJECT_PATH


load_dotenv(dotenv_path="/app/geomas/.env")
logger = get_logger("SFT_TRAINER")


class SFTTrainer:
    """
    Orchestrates supervised fine-tuning (SFT) for instruction following.

    Parameters:
        model_name: Hugging Face model identifier to load and adapt.
        dataset_path: Local path to the instruct dataset (JSON with instruction/input/output).
        quantization_mode: Unsloth quantization strategy applied when saving.
        tag: Optional label used to namespace MLflow experiments and runs.
        max_seq_length: Maximum sequence length for training.
    """

    def __init__(
        self,
        model_name: str,
        dataset_path: Path,
        quantization_mode: str = "fast_quantized",
        tag: str = "",
        max_seq_length: int = 2048
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.quantization_mode = quantization_mode
        self.tag = tag
        self.max_seq_length = max_seq_length

        # Correct model name for MLflow logging
        self.correct_model_name = model_name.split("/")[-1].translate(
            str.maketrans("", "", "/:.%\"'")
        )

    def load_configs(self, correct_model_name: str):
        """Load sequentially `Trainer`, `PEFT`, `Model` configs."""
        return prepare_settings(f"cpt-{correct_model_name}")

    def formatting_prompts_func(self, tokenizer, examples):
        """
        Format instruction examples into prompts for training.

        Args:
            tokenizer: Tokenizer to use for EOS token
            examples: Batch dictionary containing 'instruction', 'input', 'output'

        Returns:
            Dictionary with formatted 'text' field
        """
        texts = []
        EOS_TOKEN = tokenizer.eos_token

        for instruction, input_text, output in zip(
            examples.get("instruction", []),
            examples.get("input", []),
            examples.get("output", [])
        ):
            # Build prompt in Alpaca format
            if input_text:
                prompt = f"""Ниже приведена инструкция, описывающая задачу, в паре с входными данными, которые предоставляют дополнительный контекст. Напишите ответ, который правильно выполняет запрос.

### Инструкция:
{instruction}

### Входные данные:
{input_text}

### Ответ:
{output}{EOS_TOKEN}"""
            else:
                prompt = f"""Ниже приведена инструкция, описывающая задачу. Напишите ответ, который правильно выполняет запрос.

### Инструкция:
{instruction}

### Ответ:
{output}{EOS_TOKEN}"""

            texts.append(prompt)

        return {"text": texts}

    def train(self):
        """Execute supervised fine-tuning workflow from data prep to MLflow logging."""
        logger.info("SFT Started")
        logger.info(f"Model - {self.model_name}")
        logger.info(f"Dataset path: {self.dataset_path}")
        dataset_name = str(self.dataset_path).split("/")[-1]

        _init_mlflow_logging(
            correct_model_name=self.correct_model_name,
            tag=self.tag
        )

        trainer_config, peft_config, model_config = self.load_configs(self.correct_model_name)
        run_name = f"sft-{self.correct_model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

        with mlflow.start_run(run_name=run_name):
            # Load model
            model_cls = FastLanguageModel if is_pure_llm(self.model_name) else FastModel
            model, tokenizer = model_cls.from_pretrained(
                model_name=self.model_name,
                **model_config
            )

            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total params in model: {total_params:,}")

            # Apply PEFT (LoRA)
            model = model_cls.get_peft_model(model, **peft_config)

            trainable_params = model.get_nb_trainable_parameters()[0]
            trainable_percentage = 100 * trainable_params / total_params
            logger.info(
                f"With rank {peft_config['r']} number of trainable params: "
                f"{trainable_params:,}, {trainable_percentage:.2f}%"
            )

            # Load and format dataset
            dataset = get_dataset(str(self.dataset_path))
            logger.info(f"Loaded dataset with {len(dataset)} examples")

            # Format dataset with prompts
            dataset = dataset.map(
                partial(self.formatting_prompts_func, tokenizer),
                batched=True,
                remove_columns=dataset.column_names  # Remove original columns
            )

            logger.info("Dataset samples (first 2):")
            for i, row in enumerate(dataset[:2]["text"]):
                logger.info(f"\n--- Sample {i+1} ---")
                logger.info(row[:500] + "..." if len(row) > 500 else row)

            # Create trainer
            trainer = UnslothTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                dataset_num_proc=8,
                args=UnslothTrainingArguments(
                    **trainer_config,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    output_dir=f"outputs/sft-{self.correct_model_name}-{dataset_name}",
                    run_name=run_name,
                    report_to="mlflow",
                ),
            )

            # Train
            start_gpu_memory, max_memory = pretrain_report()
            logger.info("TRAINING ...")
            trainer_stats = trainer.train()
            train_report = posttrain_report(
                start_gpu_memory=start_gpu_memory,
                max_memory=max_memory,
                trainer_stats=trainer_stats,
            )

            # Save
            save_directory = os.path.join(
                PROJECT_PATH,
                "..",
                "models",
                f"sft-{self.correct_model_name}"
            )
            os.makedirs(save_directory, exist_ok=True)
            logger.info(f"Saving model <{self.model_name}> to: <{save_directory}>")
            model.save_pretrained(save_directory, quantization_method=self.quantization_mode)
            tokenizer.save_pretrained(save_directory)

            # Log all params to MLFLOW
            mlflow.log_params(train_report | trainer_config | peft_config | model_config)

            # Log model to MLFlow
            mlflow.transformers.log_model(
                transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                artifact_path=f"sft-{self.correct_model_name}",
            )

            logger.info("SFT Training completed successfully!")

        return save_directory


def sft_train(
    model_name: str,
    dataset_path: str,
    quantization_mode: str = "fast_quantized",
    tag: str = "",
    max_seq_length: int = 2048
) -> str:
    """
    Train a model with supervised fine-tuning on instruct dataset.

    Args:
        model_name: HuggingFace model name
        dataset_path: Path to instruct dataset JSON
        quantization_mode: Quantization method for saving
        tag: Tag for MLflow experiment
        max_seq_length: Maximum sequence length

    Returns:
        Path to saved model directory
    """
    trainer = SFTTrainer(
        model_name=model_name,
        dataset_path=Path(dataset_path),
        quantization_mode=quantization_mode,
        tag=tag,
        max_seq_length=max_seq_length
    )

    return trainer.train()


if __name__ == "__main__":
    # Example usage
    trainer = SFTTrainer(
        model_name="unsloth/mistral-7b-v0.3-bnb-4bit",
        dataset_path=Path("./instruct_dataset_train.json"),
        quantization_mode="fast_quantized",
        tag="geological_instruct",
        max_seq_length=2048
    )

    trainer.train()


