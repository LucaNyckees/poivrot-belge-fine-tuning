from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class InferenceEngine:
    def __init__(self, base_model: str, lora_path: str):
        base_model = "mistralai/Mistral-7B-Instruct-v0"
        lora_path = "./poivrot_belge_lora"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
        self.model = PeftModel.from_pretrained(self.model, lora_path)

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def execute(self):
        # demo
        prompt = "Jean-Claude: Salut fieu, comment ça va?"
        return self.generate(prompt)
