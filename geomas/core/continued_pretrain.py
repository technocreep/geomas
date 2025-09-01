import os
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Optional, Tuple

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from transformers import TextIteratorStreamer
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
)

from geomas.core.config import prepare_settings
from geomas.core.dataset import get_dataset
from geomas.core.logger import get_logger
from geomas.core.report.pretrain import posttrain_report, pretrain_report
from geomas.core.utils import PROJECT_PATH

load_dotenv()
logger = get_logger()


class CPTTrainingError(Exception):
    pass


class ModelLoadError(CPTTrainingError):
    pass


class DatasetLoadError(CPTTrainingError):
    pass


class MLflowError(CPTTrainingError):
    pass


@dataclass
class CPTConfig:
    model_name: str
    dataset_path: Path
    infer_at_once: bool = False
    quantization_mode: str = "fast_quantized"
    tag: str = ""
    max_seq_length: int = 2048

    def __post_init__(self):
        self.correct_model_name = self.model_name.split("/")[-1].translate(
            str.maketrans("", "", "/:.%\"'")
        )
        self.prefix = f"{self.tag}-" if self.tag else ""
        self.exp_name = f"{self.prefix}CPT-{self.correct_model_name}"
        self.run_name = (
            f"{self.correct_model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}"
        )


class MLflowService:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        self.client = None

    def setup(self, exp_name: str, artifact_location: str) -> str:
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient()

            exp = self.client.get_experiment_by_name(exp_name)
            if exp is None:
                logger.info(f"Experiment {exp_name} not found. Creating...")
                exp_id = self.client.create_experiment(
                    name=exp_name, artifact_location=artifact_location
                )
            else:
                logger.info(f"Experiment {exp_name} exists")
                exp_id = exp.experiment_id
            mlflow.set_experiment(experiment_id=exp_id)
            mlflow.enable_system_metrics_logging()
            return exp_id

        except Exception as e:
            raise MLflowError(f"Failed to setup MLflow: {e}") from e

    @contextmanager
    def start_run(self, run_name: str):
        try:
            with mlflow.start_run(run_name=run_name) as run:
                yield run
        except Exception as e:
            raise MLflowError(f"Failed to start MLflow run: {e}") from e

    def log_params(self, params: Dict[str, Any]) -> None:
        try:
            mlflow.log_params(params)
        except Exception as e:
            raise MLflowError(f"Failed to log parameters: {e}") from e

    def log_model(self, model, tokenizer, name: str) -> None:
        try:
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                name=name,
            )
        except Exception as e:
            raise MLflowError(f"Failed to log model: {e}") from e


class ModelService:
    def __init__(self, config: CPTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer_config = None
        self.peft_config = None

    def load_configs(self) -> None:
        try:
            self.trainer_config, self.peft_config = prepare_settings(
                f"cpt-{self.config.correct_model_name}"
            )
        except Exception as e:
            raise CPTTrainingError(f"Failed to load configs: {e}") from e

    def load_model_and_tokenizer(self) -> None:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            self.model, self.tokenizer = model, tokenizer
        except Exception as e:
            raise ModelLoadError(f"Failed to load model and tokenizer: {e}") from e

    def apply_lora(self) -> None:
        try:
            self.model = FastLanguageModel.get_peft_model(
                self.model, **self.peft_config
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to apply LoRA: {e}") from e


class DatasetService:
    def __init__(self, config: CPTConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None

    def load_and_preprocess(self) -> None:
        try:
            dataset = get_dataset(self.config.dataset_path)
            eos_token = self.tokenizer.eos_token

            def formatting_prompts_func(examples):
                return {"text": [example + eos_token for example in examples["text"]]}

            self.dataset = dataset.map(formatting_prompts_func, batched=True)
            logger.debug("Dataset samples:")
            for row in self.dataset[:2]["text"]:
                logger.debug(row)
        except Exception as e:
            raise DatasetLoadError(f"Failed to load dataset: {e}") from e


class TrainingService:
    def __init__(
        self,
        config: CPTConfig,
        model_service: ModelService,
        dataset_service: DatasetService,
    ):
        self.config = config
        self.model_service = model_service
        self.dataset_service = dataset_service
        self.trainer = None

    def build_trainer(self) -> None:
        try:
            self.trainer = UnslothTrainer(
                model=self.model_service.model,
                tokenizer=self.model_service.tokenizer,
                train_dataset=self.dataset_service.dataset,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                dataset_num_proc=8,
                args=UnslothTrainingArguments(
                    **self.model_service.trainer_config,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    output_dir=f"outputs/{self.config.correct_model_name}",
                    run_name=self.config.run_name,
                    report_to="mlflow",
                ),
            )
        except Exception as e:
            raise CPTTrainingError(f"Failed to build trainer: {e}") from e

    def train(self):
        try:
            logger.info("TRAINING ...")
            trainer_stats = self.trainer.train()
            return trainer_stats
        except Exception as e:
            raise CPTTrainingError(f"Training failed: {e}") from e


class ModelPersistenceService:
    def __init__(self, config: CPTConfig):
        self.config = config

    def save_model(self, model, tokenizer) -> None:
        try:
            save_directory = Path.joinpath(
                PROJECT_PATH, "models", self.config.correct_model_name
            )
            os.makedirs(save_directory, exist_ok=True)
            logger.info(
                f"Saving model <{self.config.model_name}> to: <{str(save_directory)}>"
            )
            model.save_pretrained(
                save_directory, quantization_method=self.config.quantization_mode
            )
            tokenizer.save_pretrained(
                save_directory, quantization_method=self.config.quantization_mode
            )
        except Exception as e:
            raise CPTTrainingError(f"Failed to save model: {e}") from e


class InferenceService:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def demo_inference(self) -> None:
        try:
            text_streamer = TextIteratorStreamer(self.tokenizer)
            max_print_width = 100

            inputs = self.tokenizer(
                [
                    "На рудном поле преобладают разломы северо-восточного и северо-западного направлений, \
                    в меньшей степени развиты"
                ]
                * 1,
                return_tensors="pt",
            ).to("cuda")

            generation_kwargs = dict(
                inputs,
                streamer=text_streamer,
                max_new_tokens=256,
                use_cache=True,
            )
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            length = 0
            for j, new_text in enumerate(text_streamer):
                if j == 0:
                    wrapped_text = textwrap.wrap(new_text, width=max_print_width)
                    length = len(wrapped_text[-1])
                    wrapped_text = "\n".join(wrapped_text)
                    print(wrapped_text, end="")
                else:
                    length += len(new_text)
                    if length >= max_print_width:
                        length = 0
                        print()
                    print(new_text, end="")
        except Exception as e:
            logger.error(f"Inference failed: {e}")


class CPTTrainer:
    def __init__(
        self,
        model_name: str,
        dataset_path: Path,
        infer_at_once: bool = False,
        quantization_mode: str = "fast_quantized",
        tag: str = "",
        mlflow_service: Optional[MLflowService] = None,
    ) -> None:
        self.config = CPTConfig(
            model_name=model_name,
            dataset_path=dataset_path,
            infer_at_once=infer_at_once,
            quantization_mode=quantization_mode,
            tag=tag,
        )

        self.mlflow_service = mlflow_service or MLflowService()
        self.model_service = ModelService(self.config)
        self.persistence_service = ModelPersistenceService(self.config)
        self.dataset_service = None
        self.training_service = None
        self.inference_service = None

    def run(self) -> None:
        try:
            logger.info("CPT Started")
            logger.info(f"Model - {self.config.model_name}")
            logger.info(f"Dataset path: {self.config.dataset_path}")

            self.mlflow_service.setup(
                exp_name=self.config.exp_name,
                artifact_location=f"s3://mlflow/experiments/{self.config.exp_name}",
            )

            self.model_service.load_configs()
            self.model_service.load_model_and_tokenizer()
            self.model_service.apply_lora()

            self.dataset_service = DatasetService(
                self.config, self.model_service.tokenizer
            )
            self.dataset_service.load_and_preprocess()

            self.training_service = TrainingService(
                self.config, self.model_service, self.dataset_service
            )
            self.training_service.build_trainer()

            self.inference_service = InferenceService(
                self.model_service.model, self.model_service.tokenizer
            )

            with self.mlflow_service.start_run(run_name=self.config.run_name):
                start_gpu_memory, max_memory = pretrain_report()
                trainer_stats = self.training_service.train()
                train_report = posttrain_report(
                    start_gpu_memory=start_gpu_memory,
                    max_memory=max_memory,
                    trainer_stats=trainer_stats,
                )

                self.persistence_service.save_model(
                    self.model_service.model, self.model_service.tokenizer
                )
                self.mlflow_service.log_params(train_report)
                self.mlflow_service.log_model(
                    self.training_service.trainer.model,
                    self.model_service.tokenizer,
                    self.config.correct_model_name,
                )

                if self.config.infer_at_once:
                    self.inference_service.demo_inference()

        except CPTTrainingError as e:
            logger.error(f"CPT training failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during CPT training: {e}")
            raise CPTTrainingError(f"Unexpected error: {e}") from e


def cpt_train(
    model_name: str,
    dataset_path: Path | str,
    infer_at_once: bool = False,
    quantization_mode: str = "fast_quantized",
    tag: str = "",
):
    trainer = CPTTrainer(
        model_name=model_name,
        dataset_path=dataset_path,
        infer_at_once=infer_at_once,
        quantization_mode=quantization_mode,
        tag=tag,
    )
    trainer.run()


if __name__ == "__main__":
    from geomas.core.utils import ALLOWED_MODELS

    cpt_train(
        model_name=ALLOWED_MODELS["mistral-7b"],
        dataset_path="/app/test_dataset",
        tag="cock",
    )
