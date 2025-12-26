import os
from pathlib import Path


BASE_PATH = Path(os.getenv("BASE_PATH", Path(__file__).parents[2]))
print("BASE_PATH:", BASE_PATH)
TRAINING_DATA_PATH = BASE_PATH / "scripts"
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", BASE_PATH / "results"))
MODELS_PATH = Path(os.getenv("MODELS_PATH", BASE_PATH / "poivrot_belge_lora"))
SRC_PATH = Path(os.getenv("SRC_PATH", BASE_PATH / "src"))