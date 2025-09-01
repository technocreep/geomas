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

### Make dataset
Recursively walks through source directory and produce destination directory with json versions of initial documents

```bash
geomas makedataset ./source ./destination
```

Check that all services are UP:

```bash
docker ps -a | grep mlflow
```
There must be 3 of them: 

* `mlflow_postgres` – postgres database for metadata on port `5432`
* `mlflow_minio` – local S3 for heavy artifacts on port `9000`
* `mlflow-mlflow-1` – MLFlow server itself on port `5000`

In `geomas` directory there must be `.env` file with content:

```yaml
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
AWS_ACCESS_KEY_ID=***
AWS_SECRET_ACCESS_KEY=***
```

```bash
export CUDA_VISIBLE_DEVICES=1
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


# Clean code recipe

```cmd
make lint    # Check code with Ruff
make fix     # Auto-fix lint errors
make format  # Format code (Ruff)
make sort    # Sort imports
make all     # Run all checks (lint + fix + format)
```