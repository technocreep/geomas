from functools import partial
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
    FastModel
)

import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

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
logger = get_logger()


class CPTrainer:
    """Orchestrates continued pretraining, logging, and model checkpointing.

    Parameters:
        model_name: Hugging Face model identifier to load and adapt.
        dataset_path: Local path to the text dataset used for continued pretraining.
        quantization_mode: Unsloth quantization strategy applied when saving.
        tag: Optional label used to namespace MLflow experiments and runs.
    """
    def __init__(self,
                 model_name: str,
                 dataset_path: Path,
                 quantization_mode: str = "fast_quantized",
                 tag: str = ""):
        self.model_name = model_name
        self.dataset-path = dataset_path
        self.quantization_mode = quantization_mode
        self.tag = tag
        
        # need this correction to log model with mlflow
        self.correct_model_name = model_name.split("/")[-1].translate(
            str.maketrans("", "", "/:.%\"'")
            )

    def _init_mlflow_logging(self,
                             correct_model_name: str,
                             tag: str = ""):
        """Ensure the MLflow experiment exists and attach system metrics.

        Args:
            correct_model_name: Sanitised model identifier used for MLflow naming.
            tag: Optional experiment prefix that groups related runs.
        """

        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.MlflowClient()
        prefix = f"{tag}-" if tag else ""
        exp_name = f"{prefix}CPT-{correct_model_name}"
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            logger.info(f"Experiment {exp_name} not found. Creating...")
            exp_id = client.create_experiment(
            name=exp_name,
            artifact_location=f"s3://mlflow/experiments/{exp_name}"
        )
        else:
            logger.info(f"Experiment {exp_name} exists")
            exp_id = exp.experiment_id

        mlflow.set_experiment(experiment_id=exp_id)
        mlflow.enable_system_metrics_logging()

    def load_configs(correct_model_name: str):
        """Loads sequentially `Trainer`, `PEFT`, `Model` configs"""
        return prepare_settings(f"cpt-{correct_model_name}")
    
    def formatting_prompts_func(self, EOS_TOKEN, examples):
        """Add the tokenizer EOS token to every example in the input batch.

        Args:
            EOS_TOKEN: Token string appended to terminate each sample.
            examples: Batch dictionary containing raw `text` entries.
        """
        return {"text": [example + EOS_TOKEN for example in examples["text"]]}

    def train(self):
        """Execute continued pretraining workflow from data prep to MLflow logging."""
        logger.info("CPT Started")
        logger.info(f"Model - {self.model_name}")
        logger.info(f"Dataset path: {self.dataset_path}")
        dataset_name = self.dataset_path.split("/")[-1]
        
        self._init_mlflow_logging(
            correct_model_name=self.correct_model_name,
            tag=self.tag
            )
        
        trainer_config, peft_config, model_config = self.load_configs(self.correct_model_name)
        run_name = f"{self.correct_model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}"

        with mlflow.start_run(run_name=run_name):
            max_seq_length = 2048

            # need this because e.g. gemma is multimodal
            model_cls = FastLanguageModel if is_pure_llm(self.model_name) else FastModel # if multimodal
            model, tokenizer = model_cls.from_pretrained(
                model_name=self.model_name,
                **model_config
            )
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total params in model: {total_params:,}")
            model = model_cls.get_peft_model(model, **peft_config)
    
            trainable_params = model.get_nb_trainable_parameters()[0]
            trainable_percentage = 100 * trainable_params / total_params
            logger.info(
                f"With rank {peft_config['r']} number of trainable params: {trainable_params:,}, {trainable_percentage:.2f}%"
            )

            dataset = get_dataset(self.dataset_path)
            EOS_TOKEN = tokenizer.eos_token
            dataset = dataset.map(partial(self.formatting_prompts_func, EOS_TOKEN), batched=True)

            logger.debug("Dataset samples:")
            for row in dataset[:2]["text"]:
                logger.debug(row)

            trainer = UnslothTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                dataset_num_proc=8,
                args=UnslothTrainingArguments(
                    **trainer_config,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    output_dir=f"outputs/{self.correct_model_name}-{dataset_name}",
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
            save_directory = PROJECT_PATH + "/../" + "models" + "/" + self.correct_model_name
            os.makedirs(save_directory, exist_ok=True)
            logger.info(f"Saving model <{self.model_name}> to: <{save_directory}>")
            model.save_pretrained(save_directory, quantization_method=self.quantization_mode)
            tokenizer.save_pretrained(save_directory, quantization_method=self.quantization_mode)

            # Log all paramas to MLFLOW
            mlflow.log_params(train_report | trainer_config | peft_config |model_config)
            # Log model to MLFlow
            mlflow.transformers.log_model(
                transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                name=self.correct_model_name,
                # task="text_generation",
            )


if __name__ == "__main__":
    trainer = CPTrainer(
        model_name="unsloth/Mistral-Nemo-Base-2407",
        dataset_path="/app/test_dataset",
        quantization_mode="fast_quantized",
        tag="test_usage"
    )

    trainer.train()
