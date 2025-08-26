from datetime import datetime
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap
import os

import mlflow

from geomas.core.logger import get_logger
from geomas.core.dataset import get_dataset
from geomas.core.utils import PROJECT_PATH
from geomas.core.report import pretrain_report, posttrain_report
from geomas.core.config import prepare_settings


logger = get_logger()



def cpt_train(
        model_name: str,
        dataset_path: Path,
        infer_at_once: bool = False,
        quantization_mode: str = "fast_quantized"
):
    logger.info('CPT Started')
    logger.info(f'Model - {model_name}')
    logger.info(f'Dataset path: {dataset_path}')

    # need this correction to log model with mlflow
    correct_model_name = model_name.split('/')[-1].translate(str.maketrans('', '', '/:.%"\''))

    trainer_config, peft_config = prepare_settings(f"cpt-{correct_model_name}")

    mlflow.set_experiment(experiment_name=f"SPT-train-{correct_model_name}")
    run_name = f"{correct_model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}"
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # "unsloth/mistral-7b" for 16bit loading
        max_seq_length = max_seq_length,
        dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    model = FastLanguageModel.get_peft_model(model, **peft_config)

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
            **trainer_config,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            output_dir = f"outputs/{correct_model_name}",
            run_name = run_name,
            report_to = "mlflow",
        ),
    )

    # Train
    start_gpu_memory, max_memory = pretrain_report()
    logger.info("TRAINING ...")
    trainer_stats = trainer.train()
    train_report = posttrain_report(
        start_gpu_memory=start_gpu_memory,
        max_memory=max_memory,
        trainer_stats=trainer_stats
    )

    # Save
    save_directory = PROJECT_PATH + "/../" + "models" + "/" + correct_model_name
    os.makedirs(save_directory, exist_ok=True)
    logger.info(f'Saving model <{model_name}> to: <{save_directory}>')
    model.save_pretrained(save_directory, quantization_method = quantization_mode)
    tokenizer.save_pretrained(save_directory, quantization_method = quantization_mode)

    # Report to MLFLOW
    last_run_id = mlflow.last_active_run().info.run_id
    with mlflow.start_run(run_id=last_run_id):
        mlflow.log_params(train_report)
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            name=correct_model_name,
        )

    # TODO: probably make correct infer of trained model and log into mlflow
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


if __name__ == "__main__":
    from geomas.core.utils import ALLOWED_MODELS

    cpt_train(
        model_name=ALLOWED_MODELS["gpt-oss"],
        dataset_path="/app/test_dataset",

    )