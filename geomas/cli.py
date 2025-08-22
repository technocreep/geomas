import typer
from geomas.core.logger import get_logger
from geomas.core.continued_pretrain import cpt_train

app = typer.Typer(help="GEOMAS: CLI tool for LLM Training")
logger = get_logger()

@app.command()
def train(
	model: str = typer.Argument(help="Model to train"),
	dataset_path: str = typer.Argument(help="Path to dataset"),

	):
	"""Run Training"""

	dataset_name = dataset_path.split('/')[-1]
	logger.info(f"Training model '{model}' on dataset '{dataset_name}'")

	result = cpt_train(model, dataset_path)
	logger.info('>>>>>> training finished <<<<<<<')


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
