from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from logging import Logger
import re

from src.utils.paths import MODELS_PATH


class InferenceEngine:
    def __init__(self, logger: Logger, character: str, input: str = "", tuned: bool = True):
        self.logger = logger
        self.character = character
        base_model = "mistralai/Mistral-7B-v0.1"
        lora_path = MODELS_PATH / f"dikkenek_{self.character}"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            dtype=torch.float16
        )
        if tuned:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        else:
            pass
        self.model.eval()
        self.input = input

    def generate(self, prompt: str) -> str:
        return generate_response(
            model=self.model,
            tokenizer=self.tokenizer,
            character=self.character,
            prompt=prompt,
        )

    def execute(self):
        prompt = self.input
        response = self.generate(prompt)
        self.logger.info(response)


def generate_response(model, tokenizer, character: str, prompt: str) -> str:
    formatted_prompt = f"""<s>[INST] <<SYS>>
        Tu es {character}. Réponds en **une phrase courte et drôle**, comme dans le film Dikkenek.
        <</SYS>>
        Question: {prompt}
        Réponse: [/INST]"""
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("Réponse:")[-1].strip()
    match = re.search(r'^.*[.!?]', response)
    response = match.group(0) if match else ""
    return response