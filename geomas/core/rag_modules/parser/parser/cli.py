import typer

from parser.core.gk import GKParser
from parser.core.utils.logger import Logger

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


@app.command()
def search(
    source: str = typer.Argument(help="Source to parse"),
):
    """Run search for books"""
    logger = Logger.create(
        source=source
    )
    logger.info(f"Search in {source} has started")


@app.command()
def dload(
    search_results: str = typer.Argument(help="Source to search results"),
):
    """Run download of found books"""
    logger = Logger.create(
        source='source'
    )
    logger.info("Download initiated")
