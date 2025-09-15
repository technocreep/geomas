import yaml

from geomas.core.utils import CONFIG_PATH
from geomas.core.logging.logger import get_logger


logger = get_logger()


def load_config(model_name):
    path = CONFIG_PATH + model_name + ".yaml"
    logger.info(f"Reading <{path}> config file")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_settings(model_name: str) -> tuple[dict, dict]:
    config = load_config(model_name)
    return config["UnslothTrainingArguments"], config["PEFTParams"], config["ModelConfig"]


if __name__ == "__main__":
    m = "cpt-mistral-7b-v03-bnb-4bit"
    t, p = prepare_settings(m)
    _ = 1
