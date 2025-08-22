from pathlib import Path
from datasets import load_dataset


def get_dataset(path: Path):
    return load_dataset("roneneldan/TinyStories", split = "train[:2500]")