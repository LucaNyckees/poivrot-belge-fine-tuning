from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
from logging import Logger

from src.utils.paths import MODELS_PATH


class MistralLoraFineTuner:
    def __init__(self, logger: Logger, character_name: str):
        self.logger = logger
        self.character_name = character_name
        self.model_name = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = load_dataset("json", data_files=f"scripts/dikkenek_{self.character_name}.jsonl")["train"]
        self.dataset = self.dataset.remove_columns([col for col in self.dataset.column_names if col not in ["prompt", "completion"]])
        self.logger.info(f"Nombre total de paires: {len(self.dataset)}")
        for i, ex in enumerate(self.dataset.select(range(3))):
            self.logger.info(f"Exemple {i+1}: Q={ex['prompt']} | A={ex['completion']}")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def execute(self):
        self.prepare_data()
        self.load_model()
        self.fine_tune()
        self.save_model()

    def tokenize(self, example):
        formatted_text = f"<s>[INST] {example['prompt']} [/INST] {example['completion']}</s>"
        tokens = self.tokenizer(
            formatted_text,
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "labels": tokens["input_ids"][0]
        }

    def prepare_data(self):
        self.tokenized_dataset = self.dataset.map(
            self.tokenize,
            batched=False,
            remove_columns=["prompt", "completion"]
        )
        self.tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def fine_tune(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=MODELS_PATH / f"dikkenek_{self.character_name}",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=50,
            num_train_epochs=3,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        trainer.train()

    def save_model(self):
        self.model.save_pretrained(MODELS_PATH / f"dikkenek_{self.character_name}")
