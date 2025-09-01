from pathlib import Path

from datasets import load_dataset


def get_dataset(path: Path):
    data_files = list(Path(path).rglob("*.json"))
    if not data_files:
        raise FileNotFoundError(f"JSON files not found at {path}")

    dataset = load_dataset(
        "json", data_files=[str(f) for f in data_files], split="train"
    )
    return dataset


if __name__ == "__main__":
    ds = get_dataset(path="/app/cpt_full_dataset")
    _ = 1
