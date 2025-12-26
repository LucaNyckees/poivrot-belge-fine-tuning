import requests
from bs4 import BeautifulSoup
import json
import os
from logging import Logger


class DikkenekScraper:
    def __init__(self, logger: Logger):
        self.url = "https://dikkenek.ovh/?script=."
        self.logger = logger

    def execute(self):

        response = requests.get(self.url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        os.makedirs("scripts", exist_ok=True)
        output_file = "scripts/dikkenek_qa_dataset.jsonl"

        scenes = soup.find_all("div", class_="row d-flex border rounded mx-2 my-4")

        pairs = []

        for scene in scenes:
            dialogues = []

            rows = scene.find_all("div", class_="row my-2")
            for row in rows:
                character_tag = row.find("a", href=lambda x: x and "?character=" in x)
                dialogue_tag = row.find("div", class_="script")

                if not character_tag or not dialogue_tag:
                    continue

                text = dialogue_tag.get_text(separator=" ", strip=True)
                text = text.replace("\xa0", " ")
                text = " ".join(text.split())

                dialogues.append(text)

            for i in range(len(dialogues) - 1):
                question = dialogues[i]
                answer = dialogues[i + 1]

                if "?" in question:
                    pairs.append({
                        "prompt": question,
                        "completion": answer
                    })

        with open(output_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        self.logger.info(f"{len(pairs)} question/answer pairs saved to {output_file}")
