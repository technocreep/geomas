import yaml

from geomas.core.utils import CONFIG_PATH


def load_config(model_name):
    with open(CONFIG_PATH + model_name + ".yaml", "r") as f:
        return yaml.safe_load(f)
    

def prepare_settings(model_name: str) -> tuple[dict, dict]:
    
    config = load_config(model_name)
    return config["UnslothTrainingArguments"], config["PEFTParams"]


if __name__ == "__main__":
    m = "cpt-mistral-7b-v03-bnb-4bit"
    t, p = prepare_settings(m)
    _ = 1