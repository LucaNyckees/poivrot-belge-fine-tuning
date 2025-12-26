import typer

from src.visualization.core import Visualizer
from src.utils.log_handler import setup_log

step = "visualization"
app = typer.Typer(name=step)


@app.command("execute")
def make_visualization_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
) -> None:
    logger = setup_log("visualization")
    visualizer = Visualizer(logger=logger)
    visualizer.execute()