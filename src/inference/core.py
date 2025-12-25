from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class InferenceEngine:
    def __init__(self, input: str = ""):
        base_model = "mistralai/Mistral-7B-v0.1"
        lora_path = "./poivrot_belge_lora"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        self.input = input

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.3) -> str:
        formatted_prompt = f"""
        **Instruction** : Réponds comme un Belge poivrot, de manière courte et naturelle.
        **Exemples** :
        - Q: Eh tu bois quoi fieu?
        - R: Une bonne Stella, bien fraîche !
        - Q: T'as vu le match hier ?
        - R: Ouais, c'était du lourd, mais les Diables Rouges ont encore merdé !

        **Question** : {prompt}
        **Réponse** :
        """

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to("cuda")

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.7,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("**Réponse** :")[-1].strip().split("\n")[0]

    def execute(self):
        prompt = self.input
        response = self.generate(prompt)
        print(response)
