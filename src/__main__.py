import typer

from src import scraping, lora

app = typer.Typer()


app.add_typer(scraping.app)
app.add_typer(lora.app)

if __name__ == "__main__":
    app()