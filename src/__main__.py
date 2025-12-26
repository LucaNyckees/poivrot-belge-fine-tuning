import typer

from src import scraping, lora, inference, visualization

app = typer.Typer()


app.add_typer(scraping.app)
app.add_typer(lora.app)
app.add_typer(inference.app)
app.add_typer(visualization.app)

if __name__ == "__main__":
    app()