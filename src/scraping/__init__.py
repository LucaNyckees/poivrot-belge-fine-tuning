import typer

from src.scraping.core import DikkenekScraper
from src.utils.log_handler import setup_log

step = "scraping"
app = typer.Typer(name=step)


@app.command("execute")
def make_scraping_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
) -> None:
    logger = setup_log("scraping")
    scraper = DikkenekScraper(logger=logger)
    scraper.execute()



