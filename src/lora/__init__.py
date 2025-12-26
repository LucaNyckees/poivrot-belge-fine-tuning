import typer
import json

from src.lora.core import MistralLoraFineTuner
from src.utils.log_handler import setup_log
from src.utils.paths import RESOURCES_PATH

step = "lora"
app = typer.Typer(name=step)

with open(RESOURCES_PATH / "misc.json", "r", encoding="utf-8") as f:
    misc_data = json.load(f)
    character_names = misc_data.get("character_names", [])


@app.command("execute")
def make_lora_cmd(
    test: bool = typer.Option(default=False, help="Run in test mode"),
) -> None:
    logger = setup_log("lora")
    for character_name in character_names:
        logger.info(f"Starting LoRA fine-tuning for character: {character_name}")
        lora_fine_tuner = MistralLoraFineTuner(logger=logger, character_name=character_name)
        lora_fine_tuner.execute()
        logger.info(f"Finished LoRA fine-tuning for character: {character_name}")
