from datetime import datetime
from pathlib import Path
from typing import Optional

from unsloth import FastLanguageModel
import torch
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap
import os

import mlflow
from mlflow.tracking import MlflowClient

from geomas.core.logger import get_logger
from geomas.core.dataset import get_dataset
from geomas.core.utils import PROJECT_PATH
from geomas.core.report import pretrain_report, posttrain_report
from geomas.core.config import prepare_settings

from dotenv import load_dotenv


load_dotenv(dotenv_path="/app/geomas/.env")
logger = get_logger()



class CPTTrainer:
    def __init__(self,
                 model_name: str,
                 dataset_path: Path,
                 infer_at_once: bool = False,
                 quantization_mode: str = "fast_quantized") -> None:
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.infer_at_once = infer_at_once
        self.quantization_mode = quantization_mode

        self._resolved_model_name: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None

        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True

    def _validate_and_resolve_model(self) -> None:
        if self.model_name not in MODEL_MAP:
            logger.error(f"Model <{self.model_name}> is wrong. Available: {list(MODEL_MAP.keys())}")
            raise ValueError(f"Unknown model name: {self.model_name}")
        self._resolved_model_name = MODEL_MAP[self.model_name]
        logger.info("CPT Started")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Dataset path: {self.dataset_path}")

    def _load_model_and_tokenizer(self) -> None:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=self._resolved_model_name,
                                                             max_seq_length=self.max_seq_length,
                                                             dtype=self.dtype,
                                                             load_in_4bit=self.load_in_4bit)
        self.model, self.tokenizer = model, tokenizer

    def _apply_lora(self) -> None:
        self.model = FastLanguageModel.get_peft_model(self.model,
                                                      r=128,
                                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                      "gate_proj", "up_proj", "down_proj",
                                                                      "embed_tokens", "lm_head",],
                                                      lora_alpha=32,
                                                      lora_dropout=0,
                                                      bias="none",
                                                      use_gradient_checkpointing="unsloth",
                                                      random_state=3407,
                                                      use_rslora=True,
                                                      loftq_config=None)

    def _load_dataset(self) -> None:
        dataset = get_dataset(self.dataset_path)
        eos_token = self.tokenizer.eos_token

        def _formatting_prompts_func(examples):
            return {"text": [example + eos_token for example in examples["text"]]}

        self.dataset = dataset.map(_formatting_prompts_func, batched=True)
        logger.debug("Dataset samples:")
        for row in self.dataset[:2]["text"]:
            logger.debug(row)

    def _build_trainer(self) -> None:
        self.trainer = UnslothTrainer(model=self.model,
                                      tokenizer=self.tokenizer,
                                      train_dataset=self.dataset,
                                      dataset_text_field="text",
                                      max_seq_length=self.max_seq_length,
                                      dataset_num_proc=8,
                                      args=UnslothTrainingArguments(per_device_train_batch_size=2,
                                                                    gradient_accumulation_steps=8,
                                                                    warmup_ratio=0.1,
                                                                    num_train_epochs=1,
                                                                    learning_rate=5e-5,
                                                                    embedding_learning_rate=5e-6,
                                                                    fp16=not is_bfloat16_supported(),
                                                                    bf16=is_bfloat16_supported(),
                                                                    logging_steps=1,
                                                                    optim="adamw_8bit",
                                                                    weight_decay=0.00,
                                                                    lr_scheduler_type="cosine",
                                                                    seed=3407,
                                                                    output_dir="outputs",
                                                                    report_to="none",))

    def _train(self) -> any:
        logger.info("TRAINING ...")
        trainer_stats = self.trainer.train()
        return trainer_stats

    def _save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving model <{self._resolved_model_name}> to: <{save_dir}>")
        self.model.save_pretrained_gguf(save_dir, self.tokenizer, quantization_method=self.quantization_mode)

    def _infer(self) -> None:
        pass

    def run(self, save_dir: Optional[str] = None) -> None:
        self._validate_and_resolve_model()
        self._load_model_and_tokenizer()
        self._apply_lora()
        self._load_dataset()
        self._build_trainer()

        start_gpu_memory, max_memory = _log_pre_training_report()
        trainer_stats = self._train()
        _log_post_training_report(trainer_stats, start_gpu_memory, max_memory)

        if save_dir:
            self._save(save_dir)

        if self.infer_at_once:
            self._infer()

def _log_pre_training_report() -> tuple[float, float]:
    logger.info("Pre-Training report >>> ")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")
    return start_gpu_memory, max_memory

def _log_post_training_report(trainer_stats, start_gpu_memory: float, max_memory: float) -> None:
    logger.info("After-Training report >>>")
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# TODO: remove mock infer method
def mock_infer(self):
    text_streamer = TextIteratorStreamer(self.tokenizer)
    max_print_width = 100

    inputs = self.tokenizer(
        [
            "На рудном поле преобладают разломы северо-восточного и северо-западного направлений,"
            "в меньшей степени развиты"
        ]
        * 1,
        return_tensors="pt",
    ).to("cuda")

    generation_kwargs = dict(inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True)
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

# TODO: remove backwards compatability method
def cpt_train(model_name: str,
              dataset_path: Path,
              infer_at_once: bool = False,
              quantization_mode: str = "fast_quantized"):
    trainer = CPTTrainer(model_name=model_name,
                         dataset_path=dataset_path,
                         infer_at_once=infer_at_once,
                         quantization_mode=quantization_mode)
    trainer._infer = mock_infer

    save_dir = PROJECT_PATH + "/" + "models"
    trainer.run(save_dir=save_dir)
