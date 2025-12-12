import json
from tqdm import tqdm
import requests
import os
from geomas.core.config.prompt_manager import PromptManager
from geomas.core.repository.constant_repository import SUMMARY_LLM_URL
MODEL_NAME = "gpt-oss:20b"

def make_messages(entry):
    context_text = " ".join(entry.get("context", []))
    pm = PromptManager()
    description_gen = pm.render(
        "final_description_generator",
        description=entry.get("description", ""),
        context_text=context_text,
        image_caption=entry.get("image_caption", ""),
    )

    system_prompt = description_gen["system"]
    user_prompt = description_gen["user"]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_description(entry):
    messages = make_messages(entry)
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "max_tokens": 500,
    }
    resp = requests.post(SUMMARY_LLM_URL, json=payload, timeout=2400)
    resp_json = resp.json()
    res = resp_json.get("message", {}).get("content", "") or ""
    return res.strip()


def generate_final_description(input_json):
    # Read raw JSON
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    output_json=input_json.split('.')[0]+'_final.json'
    # If the output file exists, load the existing data first.
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        # Update data using the existing final_description
        existing_map = {d.get("id", i): d for i, d in enumerate(existing_data)}
        for idx, entry in enumerate(data):
            key = entry.get("id", idx)
            if key in existing_map and "final_description" in existing_map[key]:
                entry["final_description"] = existing_map[key]["final_description"]

    # Generate line by line
    for entry in tqdm(data):
        if "final_description" in entry and entry["final_description"]:
            continue  
        try:
            entry["final_description"] = generate_description(entry)
        except Exception as e:
            print(f"generate failed: {e}")
            entry["final_description"] = ""

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Final description generated and saved to {output_json}")

