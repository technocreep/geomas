import typer
from parser.core.utils.logger import Logger
from parser.core.gk import GKParser

app = typer.Typer(
    help="PARSER: Cli tool for parsing papers and books"
)


@app.command()
def run(
    source: str = typer.Argument(help="Source to parse"),
    page_limit: int = typer.Argument(help="Number of pages to parse", default=1),
    file_limit: int = typer.Argument(help="Limit of pdfs to download", default=5)   
):
    """Run the system over certain source of files"""
    logger = Logger.create(
        source=source
    )
    logger.info(f"Parser initiated for {source} with {page_limit} pages and {file_limit} files")

    parser = GKParser(
        max_pages=page_limit,
        file_limit=file_limit,
    )
    parser.run()


    # logger.save_json

