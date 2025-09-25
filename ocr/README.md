# Reader OCR Framework

Modular OCR pipeline that converts heterogeneous documents into normalized
Markdown for retrieval‑augmented generation (RAG) systems.

See the [architectural guide](AGENTS.md) for detailed design goals and coding
standards.

## Installation

Requires **Python 3.11+**. From the repository root:

```bash
pip install -e ocr
pre-commit install
```

## Quickstart

The refactored pipeline is driven entirely through the Python API. Pick an OCR
adapter (Marker, MinerU, OlmOCR, or Qwen-VL), instantiate it with defaults, and
hand it to `core.api.process_path` or `core.api.process_paths`. The helper
functions create the expected `input/raw`, `work`, and `output/markdown`
directories if they do not already exist, orchestrate conversion → preprocessing
→ OCR, and return the Markdown paths that were produced.

```python
from pathlib import Path

from core.api import process_path
from core.ocr.adapters import Marker

markdown_paths = process_path(Path("input/raw/contract.pdf"), Marker())
print(markdown_paths[0])
```

`process_path` and `process_paths` accept optional directory overrides, but all
adapter-level tuning is handled internally by the framework.

For batched processing the repository ships with
[`ocr/examples/process_folder.py`](ocr/examples/process_folder.py). Set
`INPUT_DIR`, `OUTPUT_DIR`, `WORK_DIR`, and `ADAPTER_NAME` at the top of the
module, then execute it directly to mirror the tree under
`output/markdown/<adapter_name>`.

## Minimal End-to-End Example

Given the default folders:

```text
input/
└── raw/
    └── contract.pdf
```

Run the simplified pipeline with default locations:

```bash
python - <<'PY'
from pathlib import Path

from core.api import process_path
from core.models.adapters import Marker

generated = process_path(Path("input/raw/contract.pdf"), Marker())
print("Markdown saved to:", generated[0])
PY
```

Resulting output structure:

```text
output/
└── markdown/
    └── contract.md
```

The generated Markdown mirrors the source tree, includes provenance metadata,
and is ready for downstream RAG ingestion. When a document fails during
conversion, preprocessing, or OCR, a JSON record describing the failure is
written to `output/failures` so you can inspect the error without rerunning the
entire job.
