from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel


# questions = [
    # "Важнейшими внешними факторами гипергенной миграции химических элементов являются характер рельефа, абсолютный уровень и колебания температуры,",
    # "Месторождения филизчайского (бесси) типа ассоциируют с породами базальтсодержащей терригенно-углеродисто-флишоидной формации. По своей роли в общем мировом балансе",
    # "Инъекционные тела гранитов в большинстве случаев являются апофизами глубинных гранитных батолитов. Граниты весьма разнообразны по",
    # "При отсутствии четких геологических границ оконтуривание тел полезных ископаемых производится на основе данных опробования",
    # "При выборе оптимального варианта разведки следует учитывать степень изменчивости содержаний золота и характер пространственного его распределения, текстурно-структурные особенности руд (главным образом",
# ]

questions = [
    "Полупромышленные (заводские) технологические пробы служат для проверки эффективности переработки руды в заводских условиях или в опытных цехах по схеме непрерывного технологического процесса. Полузаводские испытания осуществляются только тогда,",
    "Метод ядерного гамма-резонанса (ЯГРМ) основан на эффекте Мессбауера (резонансе рассеянии гамма-квантов) и используется для определения содержания олова или касситерита. Источником излуче-ния",
]

answers = [
    "когда намечается переработка нового типа руды, не освоенного промышленностью, или руда имеет весьма сложную техно-логию переработки. В большинстве случаев к полузаводским испытаниям не прибегают, ограничиваясь валовыми технологическими пробами.",
    " изотоп 119Sn. При неподвижном источнике происходит резонансное поглощение гамма-квантов атомами олова (природным изотопом 119Sn). Сравнение результатов резонанса при подвижном ис-точнике и поглощения при неподвижном источнике позволяет судить о содержании олова"
]

token_limits = [
    64,
    128
]


CPT_MODEL_PATH = "/app/outputs/Qwen3-14B-Base-unsloth-bnb-4bit-test_razvedka/checkpoint-530"
BASE_MODEL_PATH = "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"

# CPT_MODEL_PATH = "/app/outputs/Qwen3-14B-Base-unsloth-bnb-4bit/checkpoint-2706"
# BASE_MODEL_PATH = "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"

# CPT_MODEL_PATH = "/app/outputs/mistral-7b-v03-bnb-4bit/checkpoint-1808"
# BASE_MODEL_PATH = "unsloth/mistral-7b-v0.3-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name      = CPT_MODEL_PATH,
    model_name      = BASE_MODEL_PATH,
    max_seq_length  = 2048,
    dtype           = None,
    load_in_4bit    = True,
)

print(model)
_ = 1
FastLanguageModel.for_inference(model)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

for i, prompt in enumerate(questions):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    print("=========")
    print(f"Q: {prompt}")

    # for lim in token_limits:
        # gen_kwargs = dict(
        #     max_new_tokens = lim,
        #     # do_sample      = False,
        #     do_sample      = True,
        #     temperature    = 0.2,
        #     # temperature    = 0.4,
        #     top_p          = 1.0,
        #     num_beams      = 1, # always: ValueError: `streamer` cannot be used with beam search (yet!)
        #     eos_token_id   = tokenizer.eos_token_id,
        # )

    qwen_kwargs = dict(
        max_new_tokens = 64,
        do_sample      = False,
        temperature    = 0.0,
        top_p          = 1,
        top_k          = 50,
        min_p          = 0.0,
        num_beams      = 1,
        eos_token_id   = tokenizer.eos_token_id,
    )

    # mistral_kwargs = dict(
    #     max_new_tokens = 128,
    #     do_sample      = True,
    #     temperature    = 0.5,
    #     top_p          = 0.8,
    #     top_k          = 10,
    #     num_beams      = 1,
    #     min_p          = 0.01,
    #     eos_token_id   = tokenizer.eos_token_id,
    # )

    text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    print("=========")
    # print(f'Max tokens: {lim}')
    with torch.inference_mode():
        _ = model.generate(**inputs, streamer=text_streamer, **qwen_kwargs)
    
    print(f'Answer: {answers[i]}')
    print("=========")
