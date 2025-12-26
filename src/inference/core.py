from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.helpers import select_first_in_multi_reponses_output


class InferenceEngine:
    def __init__(self, input: str = "", tuned: bool = True):
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
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        reponses_list = full_response.split("**Réponse** :")
        if len(reponses_list) > 1:
            print(f"There are {len(reponses_list)} parts in the response split by '**Réponse** :'")
            print("Returning the first one only.")
            response_part = reponses_list[1].strip()
        else:
            print("No '**Réponse** :' found in the response.")
            response_part = ""

        return response_part

    def execute(self):
        prompt = self.input
        response = self.generate(prompt)
        print("Voici la réponse générée :")
        print(response)
        processed_response = select_first_in_multi_reponses_output(response)
        print("Voici la réponse après traitement : ----------------")
        print(processed_response)
