from datetime import datetime
import mlflow
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from geomas.core.logger import get_logger
from dotenv import load_dotenv


logger = get_logger(logger_name="INS")
load_dotenv(dotenv_path="/app/geomas/.env")

max_seq_length = 2048
dtype = None
load_in_4bit = True 


MODELNAME = "/app/mistral-nemo-12b-base-16bit-23-10"

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = MODELNAME,
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )
# model.save_pretrained_merged("Qwen3-30B-A3B-Base-bnb-4bi", tokenizer, save_method = "merged_8bit",)


correct_name = MODELNAME.split('/')[-1]


exp_name = f"IT-{correct_name}"

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.MlflowClient()
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

run_name = f"{correct_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}"

with mlflow.start_run(run_name=run_name):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODELNAME,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    # model.save_pretrained_merged("mistral-nemo-12b-base-16bit-23-10", tokenizer, save_method = "merged_16bit",)

    # unsloth/Mistral-Nemo-Base-2407



    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "lm_head"
                        ],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = True,
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    alpaca_prompt = """Ниже приведена инструкция, описывающая задачу, и входные данные, предоставляющие дополнительный контекст. Напишите ответ, который соответствующим образом дополняет запрос.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def format_ru_instruct_as_alpaca(examples):
        chats = examples["conversations"]
        texts = []

        for conv in chats:
            system_msg = ""
            user_msg = ""
            assistant_msg = ""

            # извлекаем сообщения по ролям
            for turn in conv:
                role = turn["role"]
                content = turn["content"].strip()
                if role == "system":
                    system_msg = content
                elif role == "user":
                    user_msg = content
                elif role == "assistant":
                    assistant_msg = content

            # форматируем в Alpaca-стиле
            text = alpaca_prompt.format(user_msg, system_msg, assistant_msg) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("d0rj/ru-instruct", split = "train[:10000]")
    dataset = dataset.map(format_ru_instruct_as_alpaca, batched = True,)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True)


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 10,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 200,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = f"outputs/instruct/{correct_name}-IT",
            report_to="mlflow",
            do_eval=True,
            save_strategy="steps",
            save_steps=40,
            eval_strategy="steps",
            eval_steps=40,
            eval_on_start=True
        ),
    )


    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # mlflow.transformers.log_model(
    #     transformers_model={"model": trainer.model, "tokenizer": tokenizer},
    #     name=correct_name,
    #     # task="text_generation",
    # )

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Ты - ассистент геолога, владеющий знаниями по геологоразведке, переработке полезных ископаемых, химии и физике.", # instruction
            "Метод ядерного гамма-резонанса (ЯГРМ) основан на эффекте Мессбауера (резонансе рассеянии гамма-квантов) и используется для определения содержания олова или касситерита. Что является Источником излучения?", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    try:
        (tokenizer.batch_decode(outputs))
    except:
        print('Failed to decode')

    model.save_pretrained_merged(f"{correct_name}-INS-23-10", tokenizer, save_method = "merged_8bit",)

    # model.save_pretrained_gguf("mistral-nemo-geo-instruct", tokenizer, quantization_method = "f16")