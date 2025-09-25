import os
from pathlib import Path
from typing import List, Optional

import typer

from geomas.core.logger import get_logger
from geomas.core.pdf_to_json import process_folder
from geomas.core.utils import ALLOWED_MODELS, ALLOWED_QUANTS, PROJECT_PATH
from geomas.core.repository.ocr_repository import get_ocr_adapter_names, is_supported_file

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
    from geomas.core.continued_pretrain import cpt_train

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
    if not os.path.exists(source):
        logger.error(f"Source path <{source}> doesn't exist")
        raise ValueError

    os.makedirs(destination, exist_ok=True)

    logger.info(f"Processing folder: <{source}>")
    process_folder(folder_path=source, output_folder=destination)
    logger.info(f"Saved to: <{destination}>")


@app.command()
def ocr(
    source: str = typer.Argument(help="Path to file or directory for OCR processing"),
    adapter: str = typer.Option("marker", help="OCR adapter to use"),
    output_dir: str = typer.Option("output/markdown", help="Directory to save results"),
    work_dir: Optional[str] = typer.Option(None, help="Working directory (temporary if not specified)"),
    batch_size: int = typer.Option(8, help="Batch size for processing"),
    language: str = typer.Option("auto", help="Language for OCR processing"),
):
    """Run OCR processing on documents"""
    from geomas.core.api.ocr import process_document_ocr, process_documents_ocr
    
    source_path = Path(source)
    
    if not source_path.exists():
        logger.error(f"Path does not exist: {source}")
        raise typer.Exit(1)
    
    # Check adapter availability
    available_adapters = get_ocr_adapter_names()
    if adapter not in available_adapters:
        logger.error(f"Unknown adapter: {adapter}. Available: {available_adapters}")
        raise typer.Exit(1)
    
    try:
        if source_path.is_file():
            # Process single file
            if not is_supported_file(str(source_path)):
                logger.error(f"Unsupported file type: {source_path.suffix}")
                raise typer.Exit(1)
                
            logger.info(f"Processing file {source_path} with adapter {adapter}")
            result = process_document_ocr(
                source_path,
                adapter_name=adapter,
                output_dir=output_dir,
                work_dir=work_dir,
                batch_size=batch_size,
                language=language
            )
        else:
            # Process directory
            files = []
            for file_path in source_path.rglob("*"):
                if file_path.is_file() and is_supported_file(str(file_path)):
                    files.append(file_path)
            
            if not files:
                logger.info(f"No supported files found in {source_path}")
                return
                
            logger.info(f"Found {len(files)} files for processing")
            result = process_documents_ocr(
                files,
                adapter_name=adapter,
                output_dir=output_dir, 
                work_dir=work_dir,
                batch_size=batch_size,
                language=language
            )
            
        logger.info(f"OCR processing completed. Created {len(result)} files:")
        for file_path in result:
            logger.info(f"  {file_path}")
            
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}")
        raise typer.Exit(1)


@app.command()
def ocr_adapters():
    """Show available OCR adapters"""
    try:
        adapters = get_ocr_adapter_names()
        if adapters:
            logger.info("Available OCR adapters:")
            for adapter in sorted(adapters):
                logger.info(f"  - {adapter}")
        else:
            logger.info("No available OCR adapters found")
    except Exception as e:
        logger.error(f"Error getting adapters list: {e}")


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
