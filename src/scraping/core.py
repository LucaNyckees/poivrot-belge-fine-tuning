import requests
from bs4 import BeautifulSoup
import json
import os
from logging import Logger
from collections import defaultdict


class DikkenekScraper:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.url = "https://dikkenek.ovh/?script=."
        self.conversations = []  # Liste de tous les échanges

    def execute(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, "html.parser")
        os.makedirs("scripts", exist_ok=True)

        # Trouver toutes les scènes
        scenes = soup.find_all("div", class_="row d-flex border rounded mx-2 my-4")

        for scene in scenes:
            # Extraire les dialogues de la scène
            dialogues = []
            rows = scene.find_all("div", class_="row my-2")
            for row in rows:
                character_tag = row.find("a", href=lambda x: x and "?character=" in x)
                dialogue_tag = row.find("div", class_="script")

                if character_tag and dialogue_tag:
                    character_name = character_tag.get_text(strip=True)
                    dialogue_text = dialogue_tag.get_text(separator=" ", strip=True)
                    dialogue_text = " ".join(dialogue_text.split())
                    dialogues.append((character_name, dialogue_text))

            # Créer des paires question/réponse
            for i in range(len(dialogues) - 1):
                current_char, current_dialogue = dialogues[i]
                next_char, next_dialogue = dialogues[i + 1]

                # Format: "Jean-Claude: Tu viens d’où ?" → "Fabien: De Bruxelles."
                pair = {
                    "prompt": f"{current_char}: {current_dialogue}",
                    "completion": f"{next_char}: {next_dialogue}"
                }
                self.conversations.append(pair)

        # Sauvegarder un fichier JSONL unique avec toutes les paires
        output_file = "scripts/dikkenek_conversations.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for pair in self.conversations:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        # Sauvegarder des fichiers séparés par personnage
        characters = defaultdict(list)
        for pair in self.conversations:
            # Extraire le personnage qui répond
            response_char = pair["completion"].split(":")[0]
            pair_wo_response_char = {
                "prompt": pair["prompt"].split(":", 1)[1].strip(),
                "completion": pair["completion"].split(":", 1)[1].strip()
            }
            characters[response_char].append(pair_wo_response_char)

        # Un fichier JSONL par personnage
        for character, pairs in characters.items():
            if len(pairs) < 50:
                continue  # Ignorer les personnages avec moins de 5 réponses
            char_output_file = f"scripts/dikkenek_{character.lower().replace(' ', '_')}.jsonl"
            with open(char_output_file, "w", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            print(f"Saved {len(pairs)} pairs for {character} in {char_output_file}")

        self.logger.info(f"{len(pairs)} question/answer pairs saved to {output_file}")
