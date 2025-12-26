import typer

from src.visualization.core import Visualizer

step = "visualization"
app = typer.Typer(name=step)


@app.command("execute")
def make_visualization_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
) -> None:

    print(test)
    visualizer = Visualizer()
    visualizer.execute()