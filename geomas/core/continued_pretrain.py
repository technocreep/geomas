from pathlib import Path
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap
import sys
import os

from geomas.core.logger import get_logger
from geomas.core.dataset import get_dataset
from geomas.core.utils import PROJECT_PATH

logger = get_logger()


MODEL_MAP = {
    "gpt-oss": "unsloth/gpt-oss-20b",
    "gemma-3n": "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "mistral-7b": "mistral-7b-v0.3-bnb-4bit",
    "gemma-7b": "gemma-7b-bnb-4bit",
}


def cpt_train(
        model_name: str,
        dataset_path: Path,
        infer_at_once: bool = False,
        quantization_mode: str = "fast_quantized"
):
    logger.info('CPT Started')
    logger.info(f'Model - {model_name}')
    logger.info(f'Dataset path: {dataset_path}')
    
    if not model:
        logger.error(f'Model <{model_name}> is wrong. Available: {MODEL_MAP.keys()}')
        return

    _model = MODEL_MAP[model_name]

    max_seq_length = 2048
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = _model, # "unsloth/mistral-7b" for 16bit loading
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",

                        "embed_tokens", "lm_head",], # Add for continual pretraining
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    dataset = get_dataset(dataset_path)

    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples):
        return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
    
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    logger.debug("Dataset samples:")
    for row in dataset[:2]["text"]:
        logger.debug(row)

    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 8,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,

            warmup_ratio = 0.1,
            num_train_epochs = 1,

            learning_rate = 5e-5,
            embedding_learning_rate = 5e-6,

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.00,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # TODO: Use this for WandB etc
        ),
    )

    logger.info("Pre-Training report >>> ")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    logger.info("TRAINING ...")
    trainer_stats = trainer.train()

    logger.info("After-Training report >>>")
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    save_directory = PROJECT_PATH + "/" + "models"
    os.makedirs(save_directory, exist_ok=True)
    logger.info(f'Saving model <{_model}> to: <{save_directory}>')
    model.save_pretrained_gguf("directory", tokenizer, quantization_method = quantization_mode)

    if infer_at_once:
        text_streamer = TextIteratorStreamer(tokenizer)
        max_print_width = 100

        inputs = tokenizer(
        [
            "На рудном поле преобладают разломы северо-восточного и северо-западного направлений, \
             в меньшей степени развиты"
        ]*1, return_tensors = "pt").to("cuda")

        generation_kwargs = dict(
            inputs,
            streamer = text_streamer,
            max_new_tokens = 256,
            use_cache = True,
        )
        thread = Thread(target = model.generate, kwargs = generation_kwargs)
        thread.start()

        length = 0
        for j, new_text in enumerate(text_streamer):
            if j == 0:
                wrapped_text = textwrap.wrap(new_text, width = max_print_width)
                length = len(wrapped_text[-1])
                wrapped_text = "\n".join(wrapped_text)
                print(wrapped_text, end = "")
            else:
                length += len(new_text)
                if length >= max_print_width:
                    length = 0
                    print()
                print(new_text, end = "")
            pass
        pass
