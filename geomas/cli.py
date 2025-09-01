import typer
from geomas.core.logger import get_logger
import os
from geomas.core.utils import ALLOWED_QUANTS, ALLOWED_MODELS
from geomas.core.utils import PROJECT_PATH
from geomas.core.pdf_to_json import process_folder


app = typer.Typer(help="GEOMAS: CLI tool for LLM Training")
logger = get_logger()

@app.command()
def train(
	model: str = typer.Argument(help=f"Model to train. Available: {ALLOWED_MODELS.keys()}"),
	dataset_path: str = typer.Argument(help="Path to dataset"),
	tag: str = typer.Argument(help="Any prefix to experiment name", default=""),
	quantization_mode: str = typer.Argument(help=f"Allowed methods: {ALLOWED_QUANTS}", default="fast_quantized"),
	):
	"""Run Training"""
	# set up CUDA device
	from geomas.core.continued_pretrain import cpt_train

	model_name = ALLOWED_MODELS.get(model, None)
	if not model:
		logger.error(f'Model <{model}> is wrong. Available: {ALLOWED_MODELS.keys()}')
		return
	
	dataset_name = dataset_path.split('/')[-1]
	
	logger.info(f"Training model '{model_name}' on dataset '{dataset_name}'")
	
	try:
		logger.info(f"CUDA device <{os.environ['CUDA_VISIBLE_DEVICES']}> is selected")
	except Exception:
		logger.error("No CUDA_VISIBLE_DEVICES env variable is set. Do `export CUDA_VISIBLE_DEVICES=1`")
		return
	
	cpt_train(
		model_name=model_name, 
		dataset_path=dataset_path,
		quantization_mode=quantization_mode,
		tag=tag)
	
	logger.info('>>>>>> Training finished <<<<<<<')


@app.command()
def makedataset(
	source: str = typer.Argument(help="Path to folder with data files"),
	destination: str = typer.Argument(help="Directory to save processed docs", default=PROJECT_PATH)
	):

	if not os.path.exists(source):
		logger.error(f"Source path <{source}> doesn't exist")
		raise ValueError
	
	os.makedirs(destination, exist_ok=True)

	logger.info(f"Processing folder: <{source}>")
	process_folder(
		folder_path=source,
		output_folder=destination
	)
	logger.info(f"Saved to: <{destination}>")



@app.command()
def health():

	logger.info("Checking core libs...")
	try:
		import torch, platform, unsloth
		logger.info("Running sanity check...")
		logger.info(f"Python version: {platform.python_version()}")
		logger.info(f"Torch version: {torch.__version__}")
		logger.info(f"Unsloth version: {unsloth.__version__}")
		logger.info(f"CUDA available: {torch.cuda.is_available()}")
		if torch.cuda.is_available():
			logger.info(f"Available devices: {torch.cuda.device_count()}")
			for device in range(torch.cuda.device_count()):
				logger.info(f"Device ###{device}: {torch.cuda.get_device_name(device)}")

				device = torch.device(f'cuda:{device}')
				free, total = torch.cuda.mem_get_info(device)
				mem_used_MB = (total - free) / 1024 ** 2
				logger.info(f'Memory in use, MB: {mem_used_MB}')

	except Exception as e:
		logger.info("Caught exception:")
		logger.info(e)
	logger.info("Sanity check finished")


if __name__ == "__main__":
	health()