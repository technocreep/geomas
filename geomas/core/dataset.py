from pathlib import Path
from datasets import load_dataset


def get_dataset(path: Path):
    data_files = list(Path(path).rglob("*.json"))
    if not data_files:
        raise FileNotFoundError(f"JSON files not found at {path}")

    dataset = load_dataset("json", data_files=[str(f) for f in data_files], split="train")
    return dataset
