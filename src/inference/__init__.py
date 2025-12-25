import typer

from src.inference.core import InferenceEngine

step = "inference"
app = typer.Typer(name=step)


@app.command("execute")
def make_inference_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
    input: str = typer.Option(default="", help="Input prompt for inference"),
) -> None:
    inference_engine = InferenceEngine(input=input)
    inference_engine.execute()
