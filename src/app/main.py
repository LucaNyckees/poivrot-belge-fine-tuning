from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from typing import Dict
import json

from src.inference.core import generate_response
from src.utils.paths import MODELS_PATH, RESOURCES_PATH
from src.app.schemas import Question, Response

with open(RESOURCES_PATH / "misc.json", "r", encoding="utf-8") as f:
    misc_data = json.load(f)
    character_names = misc_data.get("character_names", [])

# Charger les modèles au démarrage (pour éviter de recharger à chaque requête)
models: Dict[str, AutoModelForCausalLM] = {}
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def load_models():
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        dtype=torch.float16,
        device_map="auto"
    )

    for character in character_names:
        model = PeftModel.from_pretrained(
            base_model,
            MODELS_PATH / f"dikkenek_{character}"
        )
        models[character] = model

app = FastAPI(title="Dikkenek API")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/characters", response_model=list[str])
async def list_characters():
    return character_names

@app.post("/ask/{character}", response_model=Response)
async def ask_character(character: str, question: Question):
    """
    Args:
        character (str): name of character (ex: "claudy").
        question (Question): Object containing the text of the question and generation parameters.
    """
    if character not in models:
        raise HTTPException(status_code=404, detail="Character not found")

    answer = generate_response(
        model=models[character],
        tokenizer=tokenizer,
        character=character,
        prompt=question.text
    )

    return Response(
        character=character,
        question=question.text,
        answer=answer
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}
