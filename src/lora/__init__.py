import typer

from src.lora.core import MistralLoraFineTuner

step = "lora"
app = typer.Typer(name=step)


@app.command("execute")
def make_lora_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
) -> None:

    print(test)
    lora_fine_tuner = MistralLoraFineTuner()
    lora_fine_tuner.execute()
