import typer

from src import scraping

app = typer.Typer()


app.add_typer(scraping.app)

if __name__ == "__main__":
    app()