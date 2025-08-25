import typer
from geomas.core.logger import get_logger
import os
from geomas.core.utils import ALLOWED_QUANTS
from geomas.core.utils import PROJECT_PATH
from geomas.core.pdf_to_json import process_folder


app = typer.Typer(help="GEOMAS: CLI tool for LLM Training")
logger = get_logger()

@app.command()
def train(
	model: str = typer.Argument(help="Model to train"),
	dataset_path: str = typer.Argument(help="Path to dataset"),
	device: int = typer.Argument(help="Number of GPU device to compute on", default=0),
	quantization_mode: str = typer.Argument(help=f"Allowed methods: {ALLOWED_QUANTS}", default="fast_quantized")
	):
	"""Run Training"""
	# set up CUDA device
	from geomas.core.continued_pretrain import cpt_train
	os.environ["CUDA_VISIBLE_DEVICES"] = device
	
	dataset_name = dataset_path.split('/')[-1]
	
	logger.info(f"Training model '{model}' on dataset '{dataset_name}'")
	logger.info(f"CUDA device <{device}> is selected")

	
	cpt_train(
		model_name=model, 
		dataset_path=dataset_path,
		quantization_mode=quantization_mode)
	
	logger.info('>>>>>> training finished <<<<<<<')


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
		logger.info("="*30)
		logger.info("Running sanity check...")
		logger.info("Python version:", platform.python_version())
		logger.info("Torch version:", torch.__version__)
		logger.info("Unsloth version:", unsloth.__version__)
		logger.info("="*30)

		logger.info("CUDA available:", torch.cuda.is_available())
		if torch.cuda.is_available():
			logger.info("Device:", torch.cuda.get_device_name(0))
	except Exception as e:
		logger.info("Caught exception:")
		logger.info(e)
	logger.info("="*30)
	logger.info("Sanity check finished")
