# geomas

inside environment around `geomas directory`
```bash
pip install -e .
```

To check core libraries and CUDA device:

```bash
geomas health
```

## Continued pretraining

```bash
geomas train MODEL DATASET_PATH
```

Model could be any of:

```json
    "gpt-oss": "unsloth/gpt-oss-20b",
    "gemma-3n": "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "mistral-7b": "mistral-7b-v0.3-bnb-4bit",
    "gemma-7b": "gemma-7b-bnb-4bit",
```

## Supervised fine-tuning


# Clean code recipie

```cmd
make lint    # Check code with Ruff
make fix     # Auto-fix lint errors
make format  # Format code (Ruff)
make sort    # Sort imports
make all     # Run all checks (lint + fix + format)
```