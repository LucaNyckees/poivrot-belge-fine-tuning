import typer

from src.inference.core import InferenceEngine

step = "inference"
app = typer.Typer(name=step)


@app.command("execute")
def make_inference_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
) -> None:

    print(test)
    inference_engine = InferenceEngine()
    inference_engine.execute()
