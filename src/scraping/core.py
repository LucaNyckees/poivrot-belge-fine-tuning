import requests
from bs4 import BeautifulSoup
import json
import os


class DikkenekScraper:
    def __init__(self):
        self.url = "https://dikkenek.ovh/?script=."

    def execute(self) -> str:
        # os.makedirs("scripts", exist_ok=True)
        # file_path = "scripts/dikkenek.txt"
        # response = requests.get(self.url)
        # if response.status_code == 200:
        #     text = response.text
        #     with open(file_path, "w", encoding="utf-8") as f:
        #         f.write(text)
        #     print(f"Script saved in {file_path}")
        # else:
        #     print("Got error :", response.status_code)

        # # Petit check
        # print("Premières lignes :")
        # print(text[:500])

        # URL du script
        url = "https://dikkenek.ovh/?script=."

        # Téléchargement
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Erreur lors du téléchargement : {response.status_code}")

        html = response.text

        # Parser HTML
        soup = BeautifulSoup(html, "html.parser")

        # Dossier de sortie
        os.makedirs("scripts", exist_ok=True)
        output_file = "scripts/dikkenek_dataset.jsonl"

        # On récupère toutes les scènes
        dataset = []
        scenes = soup.find_all("div", class_="row d-flex border rounded mx-2 my-4")

        for scene in scenes:
            # Dans chaque scène, récupérer les dialogues
            rows = scene.find_all("div", class_="row my-2")
            for row in rows:
                # Personnage
                character_tag = row.find("a", href=lambda x: x and "?character=" in x)
                if not character_tag:
                    continue  # skip si pas de personnage
                character = character_tag.get_text(strip=True)

                # Dialogue
                dialogue_tag = row.find("div", class_="script")
                if not dialogue_tag:
                    continue
                dialogue = dialogue_tag.get_text(separator=" ", strip=True)

                # Nettoyage simple
                dialogue = dialogue.replace("\xa0", " ")  # non-breaking spaces
                dialogue = " ".join(dialogue.split())  # enlever doubles espaces

                # Ajouter au dataset
                dataset.append({"character": character, "text": dialogue})

        # Sauvegarder en JSONL pour fine-tuning
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in dataset:
                # On peut créer prompt/completion pour LoRA
                json_line = {
                    "prompt": f"{entry['character']}: ",
                    "completion": f"{entry['text']}\n"
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

        print(f"Dataset créé avec {len(dataset)} lignes -> {output_file}")
