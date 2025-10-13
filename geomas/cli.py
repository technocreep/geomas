import os
import sys
import json

# Aggressively suppress ALL warnings before importing anything else
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="pynvml")
warnings.filterwarnings("ignore", module="transformers")

# Suppress torch distributed warnings
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import typer

from geomas.core.logging.logger import get_logger
from geomas.core.rag_modules.convertation.pdf_to_json import process_folder
from geomas.core.utils import ALLOWED_MODELS, PROJECT_PATH

app = typer.Typer(help="GEOMAS: CLI tool for LLM Training")
logger = get_logger()


@app.command()
def train(
    model: str = typer.Argument(
        help=f"Model to train. Available: {ALLOWED_MODELS.keys()}"
    ),
    dataset_path: str = typer.Argument(help="Path to dataset"),
    tag: str = typer.Argument(help="Any prefix to experiment name", default=""),
    # quantization_mode: str = typer.Argument(
    #     help=f"Allowed methods: {ALLOWED_QUANTS}", default="fast_quantized"
    # ),
):
    """Run Training"""
    # set up CUDA device
    from geomas.core.learning.continued_pretrain import cpt_train

    model_name = ALLOWED_MODELS.get(model, None)
    if not model:
        logger.error(f"Model <{model}> is wrong. Available: {ALLOWED_MODELS.keys()}")
        return

    dataset_name = dataset_path.split("/")[-1]

    logger.info(f"Training model '{model_name}' on dataset '{dataset_name}'")

    try:
        logger.info(f"CUDA device <{os.environ['CUDA_VISIBLE_DEVICES']}> is selected")
    except Exception:
        logger.error(
            "No CUDA_VISIBLE_DEVICES env variable is set. Do `export CUDA_VISIBLE_DEVICES=1`"
        )
        return

    cpt_train(
        model_name=model_name,
        dataset_path=dataset_path,
        # quantization_mode=quantization_mode,
        tag=tag,
    )

    logger.info(">>>>>> Training finished <<<<<<<")


@app.command()
def makedataset(
    source: str = typer.Argument(help="Path to folder with data files"),
    destination: str = typer.Argument(
        help="Directory to save processed docs", default=PROJECT_PATH
    ),
):
    """Convert PDF files to JSON format for training."""
    if not os.path.exists(source):
        logger.error(f"Source path <{source}> doesn't exist")
        raise ValueError

    os.makedirs(destination, exist_ok=True)

    logger.info(f"Processing folder: <{source}>")
    process_folder(folder_path=source, output_folder=destination)
    logger.info(f"Saved to: <{destination}>")


@app.command()
def convert_annotations(
    source: str = typer.Argument(help="Path to annotations directory"),
    destination: str = typer.Argument(
        help="Directory to save BERT training data", default="./bert_training_data"
    ),
):
    """Convert Label Studio annotations to BERT training format."""
    from pathlib import Path
    from geomas.core.data.annotation_converter import LabelStudioConverter

    source_path = Path(source)
    dest_path = Path(destination)

    if not source_path.exists():
        logger.error(f"Source directory not found: {source}")
        return

    logger.info(f"Converting annotations from {source} to {destination}")
    
    converter = LabelStudioConverter()
    converter.process_annotations_directory(source_path, dest_path)
    
    logger.info("Annotation conversion completed")


@app.command()
def train_bert_ner(
    data_path: str = typer.Argument(help="Path to BERT training data directory"),
    model_name: str = typer.Option(
        "DeepPavlov/rubert-base-cased", 
        help="BERT model name to fine-tune"
    ),
    output_dir: str = typer.Option(
        "./bert_ner_output", 
        help="Directory to save the trained model"
    ),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Training batch size"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate"),
):
    """Train BERT NER model for geological entity recognition."""
    from pathlib import Path
    from geomas.core.learning.bert_ner_trainer import train_bert_ner

    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        logger.error(f"Training data directory not found: {data_path}")
        return

    logger.info(f"Training BERT NER model on data from: {data_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        trainer = train_bert_ner(
            data_path=data_path,
            model_name=model_name,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        logger.info("BERT NER training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


@app.command()
def extract_entities(
    model_path: str = typer.Argument(help="Path to trained BERT NER model"),
    text: str = typer.Option("", help="Text to extract entities from"),
    input_file: str = typer.Option("", help="Input file with text"),
    output_file: str = typer.Option("", help="Output file for results"),
):
    """Extract geological entities using trained BERT NER model."""
    from pathlib import Path
    from geomas.core.inference.bert_ner_inference import load_bert_ner_model

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.error(f"Model directory not found: {model_path}")
        return

    # Load model
    logger.info(f"Loading BERT NER model from: {model_path}")
    ner_model = load_bert_ner_model(model_path)
    
    # Get input text
    input_text = ""
    if text:
        input_text = text
    elif input_file:
        input_file_obj = Path(input_file)
        if input_file_obj.exists():
            with open(input_file_obj, 'r', encoding='utf-8') as f:
                input_text = f.read()
        else:
            logger.error(f"Input file not found: {input_file}")
            return
    else:
        logger.error("Either --text or --input-file must be provided")
        return
    
    # Extract entities
    logger.info("Extracting entities...")
    result = ner_model.extract_entities(input_text)
    
    # Format output
    output = {
        "text": result.text,
        "entities": [
            {
                "text": entity.text,
                "label": entity.label,
                "start": entity.start,
                "end": entity.end,
                "confidence": entity.confidence
            }
            for entity in result.entities
        ],
        "entity_types_found": ner_model.get_all_entity_types(result)
    }
    
    # Save or print results
    if output_file:
        output_file_obj = Path(output_file)
        with open(output_file_obj, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_file}")
    else:
        print("\n=== Extracted Entities ===")
        print(ner_model.format_entities_as_text(result))
        print(f"\nEntity types found: {', '.join(output['entity_types_found'])}")


@app.command()
def generate_qa_pairs(
    chunks_path: str = typer.Argument(help="Path to chunks.json file"),
    model_path: str = typer.Argument(help="Path to trained BERT NER model"),
    output_path: str = typer.Argument(
        help="Output path for QA pairs JSON", default="./qa_pairs.json"
    ),
    num_pairs: int = typer.Option(2, help="Number of QA pairs per entity"),
    no_context: bool = typer.Option(False, help="Don't add context to answers"),
):
    """Generate question-answer pairs from chunks using BERT NER model."""
    from pathlib import Path
    from geomas.core.data.qa_generator import generate_qa_from_chunks

    chunks_path_obj = Path(chunks_path)
    model_path_obj = Path(model_path)
    output_path_obj = Path(output_path)

    if not chunks_path_obj.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return

    if not model_path_obj.exists():
        logger.error(f"Model directory not found: {model_path}")
        return

    logger.info(f"Generating QA pairs from: {chunks_path}")
    logger.info(f"Using BERT NER model: {model_path}")
    logger.info(f"Output: {output_path}")

    try:
        num_generated = generate_qa_from_chunks(
            chunks_path=chunks_path_obj,
            model_path=model_path_obj,
            output_path=output_path_obj,
            num_pairs_per_entity=num_pairs,
            add_context=not no_context,
        )
        logger.info(f"✅ Successfully generated {num_generated} QA pairs!")
    except Exception as e:
        logger.error(f"QA generation failed: {e}")
        raise


@app.command()
def train_sft(
    model: str = typer.Argument(help="Model to fine-tune (e.g., 'mistral-7b-4bit')"),
    dataset_path: str = typer.Argument(help="Path to instruct dataset JSON"),
    tag: str = typer.Option("", help="Tag for MLflow experiment"),
    max_seq_length: int = typer.Option(2048, help="Maximum sequence length"),
):
    """Train model with supervised fine-tuning on instruct dataset."""
    from geomas.core.learning.sft_trainer import sft_train

    model_name = ALLOWED_MODELS.get(model, None)
    if not model_name:
        logger.error(f"Model <{model}> is not available. Available: {list(ALLOWED_MODELS.keys())}")
        return

    logger.info(f"Starting SFT training for model: {model_name}")
    logger.info(f"Dataset: {dataset_path}")

    try:
        save_path = sft_train(
            model_name=model_name,
            dataset_path=dataset_path,
            tag=tag,
            max_seq_length=max_seq_length
        )
        logger.info(f"✅ SFT training completed! Model saved to: {save_path}")
    except Exception as e:
        logger.error(f"SFT training failed: {e}")
        raise


@app.command()
def format_instruct_dataset(
    qa_pairs_path: str = typer.Argument(help="Path to QA pairs JSON file"),
    output_path: str = typer.Argument(
        help="Output path for formatted dataset", default="./instruct_dataset.json"
    ),
    format_type: str = typer.Option("alpaca", help="Format type: 'alpaca' or 'chat'"),
    split_ratio: float = typer.Option(0.9, help="Train/validation split ratio"),
    no_context: bool = typer.Option(False, help="Don't include context in examples"),
):
    """Format QA pairs for instruct fine-tuning."""
    from pathlib import Path
    from geomas.core.data.instruct_formatter import format_qa_dataset

    qa_pairs_path_obj = Path(qa_pairs_path)
    output_path_obj = Path(output_path)

    if not qa_pairs_path_obj.exists():
        logger.error(f"QA pairs file not found: {qa_pairs_path}")
        return

    logger.info(f"Formatting QA pairs from: {qa_pairs_path}")
    logger.info(f"Output format: {format_type}")
    logger.info(f"Output: {output_path}")

    try:
        format_qa_dataset(
            qa_pairs_path=qa_pairs_path_obj,
            output_path=output_path_obj,
            format_type=format_type,
            split_ratio=split_ratio,
            use_context=not no_context,
        )
        logger.info(f"✅ Successfully formatted dataset!")
    except Exception as e:
        logger.error(f"Dataset formatting failed: {e}")
        raise


@app.command()
def health():
    logger.info("Checking core libs...")
    try:
        import platform

        import torch
        import unsloth

        logger.info("Running sanity check...")
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Torch version: {torch.__version__}")
        logger.info(f"Unsloth version: {unsloth.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"Available devices: {torch.cuda.device_count()}")
            for device in range(torch.cuda.device_count()):
                logger.info(f"Device ###{device}: {torch.cuda.get_device_name(device)}")

                device = torch.device(f"cuda:{device}")
                free, total = torch.cuda.mem_get_info(device)
                mem_used_MB = (total - free) / 1024**2
                logger.info(f"Memory in use, MB: {mem_used_MB}")

    except Exception as e:
        logger.info("Caught exception:")
        logger.info(e)
    logger.info("Sanity check finished")


if __name__ == "__main__":
    health()
