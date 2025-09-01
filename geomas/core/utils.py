from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent.parent)
CONFIG_PATH = PROJECT_PATH + "/geomas/core/config/"


ALLOWED_QUANTS = {
    "not_quantized": "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized": "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized": "Recommended. Slow conversion. Fast inference, small files.",
    "f32": "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16": "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0": "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k": "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l": "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m": "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s": "Uses Q3_K for all tensors",
    "q4_0": "Original quant method, 4-bit.",
    "q4_1": "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s": "Uses Q4_K for all tensors",
    "q4_k": "alias for q4_k_m",
    "q5_k": "alias for q5_k_m",
    "q5_0": "Higher accuracy, higher resource usage and slower inference.",
    "q5_1": "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s": "Uses Q5_K for all tensors",
    "q6_k": "Uses Q8_K for all tensors",
    "iq2_xxs": "2.06 bpw quantization",
    "iq2_xs": "2.31 bpw quantization",
    "iq3_xxs": "3.06 bpw quantization",
    "q3_k_xs": "3-bit extra small quantization",
}

ALLOWED_MODELS = {
    # "gpt-oss": "unsloth/gpt-oss-20b",
    # "gemma-3n": "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "mistral-7b": "unsloth/mistral-7b-v0.3-bnb-4bit",
    "qwen3-14b": "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit",
    # "gemma-7b": "unslosth/gemma-7b-bnb-4bit",
}
