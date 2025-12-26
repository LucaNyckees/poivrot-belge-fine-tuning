import typer

from src.lora.core import MistralLoraFineTuner
from src.utils.log_handler import setup_log

step = "lora"
app = typer.Typer(name=step)


@app.command("execute")
def make_lora_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
) -> None:
    logger = setup_log("lora")
    lora_fine_tuner = MistralLoraFineTuner(logger=logger)
    lora_fine_tuner.execute()
