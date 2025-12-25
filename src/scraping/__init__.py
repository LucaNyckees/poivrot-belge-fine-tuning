import typer

from src.scraping.core import DikkenekScraper

step = "scraping"
app = typer.Typer(name=step)


@app.command("execute")
def make_scraping_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
) -> None:

    print(test)
    scraper = DikkenekScraper()
    scraper.execute()



