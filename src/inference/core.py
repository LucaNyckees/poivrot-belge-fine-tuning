from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from logging import Logger


class InferenceEngine:
    def __init__(self, logger: Logger, input: str = "", tuned: bool = True):
        self.logger = logger
        base_model = "mistralai/Mistral-7B-v0.1"
        lora_path = "./poivrot_belge_lora"
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

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to("cuda")
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()

    def execute(self):
        prompt = self.input
        response = self.generate(prompt)
        self.logger.info(f"Poivrot: {response}")
