import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

CPT_MODEL_PATH = "/app/outputs/Qwen3-14B-Base-unsloth-bnb-4bit/checkpoint-451"
BASE_MODEL_PATH = "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"

# CPT_MODEL_PATH = "/app/outputs/gemma-3n-E4B-unsloth-bnb-4bit/checkpoint-2"
# BASE_MODEL_PATH = "unsloth/mistral-7b-v0.3-bnb-4bit"

# CPT_MODEL_PATH = "/app/outputs/mistral-7b-v03-bnb-4bit/checkpoint-226"
# BASE_MODEL_PATH = "unsloth/mistral-7b-v0.3-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = CPT_MODEL_PATH,
    # model_name      = BASE_MODEL_PATH,
    max_seq_length  = 2048,
    dtype           = None,
    load_in_4bit    = True,
)


if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


prompt = "Метаморфические рудные формации, образованные в ходе регионального метаморфизма и плутонометаморфизма, в целом синхронны вмещающим их"
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

FastLanguageModel.for_inference(model)
gen_kwargs = dict(
    max_new_tokens = 64,
    do_sample      = False,
    temperature    = 0.0,
    top_p          = 1.0,
    num_beams      = 1,
    eos_token_id   = tokenizer.eos_token_id,
)

text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
with torch.inference_mode():
    _ = model.generate(**inputs, streamer=text_streamer, **gen_kwargs)
