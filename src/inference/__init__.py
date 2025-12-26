import typer

from src.inference.core import InferenceEngine
from src.utils.log_handler import setup_log

step = "inference"
app = typer.Typer(name=step)


@app.command("execute")
def make_inference_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
    input: str = typer.Option(default="", help="Input prompt for inference"),
    tuned: bool = typer.Option(default=True, help="Use the fine-tuned model"),
) -> None:
    logger = setup_log("inference")
    inference_engine = InferenceEngine(logger=logger, input=input, tuned=tuned)
    inference_engine.execute()
