from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel


# questions = [
#     "Важнейшими внешними факторами гипергенной миграции химических элементов являются характер рельефа, абсолютный уровень и колебания температуры,",
#     "Месторождения филизчайского (бесси) типа ассоциируют с породами базальтсодержащей терригенно-углеродисто-флишоидной формации. По своей роли в общем мировом балансе",
#     "Инъекционные тела гранитов в большинстве случаев являются апофизами глубинных гранитных батолитов. Граниты весьма разнообразны по",
#     "При отсутствии четких геологических границ оконтуривание тел полезных ископаемых производится на основе данных опробования",
#     "При выборе оптимального варианта разведки следует учитывать степень изменчивости содержаний золота и характер пространственного его распределения, текстурно-структурные особенности руд (главным образом",
# ]

# answers = [
#     "",
#     "",
#     "",
#     "",
#     "",
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
    # 128
]


# CPT_MODEL_PATH = "/app/outputs/Mistral-Nemo-Base-2407-cpt_full_dataset/checkpoint-180" like

# CPT_MODEL_PATH = "/app/outputs/Mistral-Nemo-Base-2407-test_razvedka/checkpoint-120"

# BASE_MODEL_PATH = "unsloth/Mistral-Nemo-Base-2407"

# CPT_MODEL_PATH = "/app/outputs/Mistral-NeMo-Minitron-8B-Base-test_razvedka/checkpoint-20"

# BASE_MODEL_PATH = "nvidia/Mistral-NeMo-Minitron-8B-Base"

# CPT_MODEL_PATH = "/app/outputs/Qwen3-30B-A3B-Base-cpt_full_dataset/checkpoint-190"
# BASE_MODEL_PATH = "unsloth/Qwen3-30B-A3B-Base-bnb-4bit"

# CPT_MODEL_PATH = "/app/outputs/gemma-3-27b-pt-cpt_full_dataset/checkpoint-909"
# BASE_MODEL_PATH = "unsloth/gemma-3-27b-pt"

# CPT_MODEL_PATH = "/app/outputs/gemma-3-1b-pt-unsloth-bnb-4bit-cpt_full_dataset/checkpoint-414"
# BASE_MODEL_PATH = "unsloth/gemma-3-1b-pt-unsloth-bnb-4bit"

# CPT_MODEL_PATH = "/app/outputs/Qwen3-14B-Base-unsloth-bnb-4bit-test_razvedka/checkpoint-530"
# BASE_MODEL_PATH = "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"

# CPT_MODEL_PATH = "/app/outputs/Qwen3-14B-Base-unsloth-bnb-4bit-cpt_full_dataset/checkpoint-1353"
BASE_MODEL_PATH = "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"

# CPT_MODEL_PATH = "/app/outputs/mistral-7b-v03-bnb-4bit/checkpoint-1808"
# BASE_MODEL_PATH = "unsloth/mistral-7b-v0.3-bnb-4bit"
def eval():
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name      = CPT_MODEL_PATH,
        model_name      = BASE_MODEL_PATH,
        max_seq_length  = 2048,
        dtype           = None,
        load_in_4bit    = True,
    )

    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i, prompt in enumerate(questions):
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        print("=========")
        print(f"Question:")
        print(f"{prompt}")
        print(f'Ground Truth:')
        print(f'{answers[i]}')

        for lim in token_limits:

            # gemma1b_kwargs = dict(
            #     max_new_tokens = lim,
            #     do_sample      = True,
            #     temperature    = 0.3,
            #     top_p          = 0.95,
            #     top_k          = 64,
            #     min_p          = 0.01,
            #     num_beams      = 1,
            #     eos_token_id   = tokenizer.eos_token_id,
            # )
            qwen_kwargs = dict(
                max_new_tokens = lim,
                do_sample      = True,
                temperature    = 0.3,
                top_p          = 0.95,
                top_k          = 50,
                min_p          = 0.0,
                num_beams      = 1,
                eos_token_id   = tokenizer.eos_token_id,
            )
            # mistral_kwargs = dict(
            #     max_new_tokens = lim,
            #     do_sample      = True,
            #     temperature    = 0.7,
            #     top_p          = 0.95,
            #     top_k          = 50,
            #     min_p          = 0.0,
            #     num_beams      = 1,
            #     eos_token_id   = tokenizer.eos_token_id,
            # )

            text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            f1 = []
            for retry in range(3):
                print(f"Response [{retry}] - max tokens: {lim}:")
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        streamer=text_streamer,
                        **qwen_kwargs
                        # **mistral_kwargs
                        )

                response = tokenizer.decode(output[0], skip_special_tokens=True).split(prompt)[-1]

                # from rouge_score import rouge_scorer
                from bert_score import score as bert_score


                P, R, F1 = bert_score(
                    [response],
                    [answers[i]],
                    # lang='ru'
                    model_type="xlm-roberta-large"
                    )

                f1.append(F1)
                print(f"F1 score - {F1}")
                print(f"Precision score - {P}")
                print(f"Recall score - {R}")
                print("=========")
            print(f"Average F1: {sum(f1) / 3}")
