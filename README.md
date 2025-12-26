# 🍺 Dikkenek Movie Characters: Fine-Tuning Mistral-7B with LoRA

A parameter efficient **fine-tuning** (PEFT) project for the **Mistral-7B** model to generate dialogues in the style of characters from the movie **Dikkenek** (Belgian humor, local expressions). Uses **LoRA (Low-Rank Adaptation)** for efficient and lightweight training. The end product is an application in which the user can choose a character from the movie and interact with them.

---

## 📌 Table of Contents
- [📌 Table of Contents](#-table-of-contents)
- [🎯 Purpose](#-purpose)
- [🛠 Requirements](#-requirements)
- [🤖 Usage](#-usage)
- [🌐 API: Interact with Dikkenek Characters](#-api)
- [🔧 Model Parameters](#-model-parameters)
- [🚀 Installation](#-installation)
- [📂 Project Structure](#-project-structure)
- [💡 Example Responses](#-example-responses)
- [📊 Results and Evaluation](#-results-and-evaluation)

---

## 🎯 Purpose
Create a suite of models capable of **generating responses in the style of Dikkenek characters** (Belgian humor, local slang). This project uses:
- **Mistral-7B** as the base model.
- **LoRA** for efficient fine-tuning (less memory, faster training).
- A **dialogue dataset** scraped from [dikkenek.ovh](https://dikkenek.ovh).

---

## 🛠 Requirements
- **Hardware**:
  - **NVIDIA GPU** (tested with RTX A6000, 48GB VRAM).
  - **CUDA 12.3** and **cuDNN** installed.
- **Software**:
  - Python 3.10+
  - `pip` for dependency installation.

---

## 🤖 Usage
```
cd poivrot-belge-fine-tuning/
```
### 1. Scrape Dialogues
```
python -m src scraping execute
```
### 2. Fine-Tune the Model
```
python -m src lora execute
```
### 3. Inference (Generate Responses)
```
python -m src inference execute --input="Eh tu bois quoi fieu ?"
```

---

## 🔧 Model Parameters
The base model is Hugging Face's `mistralai/Mistral-7B-v0.1` (information here: https://mistral.ai/news/announcing-mistral-7b).

---

## 🌐 API: Interact with Poivrot Belge Characters
The project includes a FastAPI-based REST API to interact with the fine-tuned LoRA models. The API allows you to:

Select a character (Claudy, Natasha, Jean-Claude, etc.).
Ask a question and get a response in the character's unique style.
Deploy locally or online for easy integration with other applications.

### API Setup
Run the API locally with:
```
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
### API Endpoints
| Endpoint                     | Method | Description                                                                 | Example Request                                                                 |
|------------------------------|--------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `/characters`                | GET    | List all available characters.                                             | `curl http://localhost:8000/characters`                                         |
| `/ask/{character}`           | POST   | Ask a question to a specific character.                                    | `curl -X POST "http://localhost:8000/ask/claudy" -H "Content-Type: application/json" -d '{"text": "Quelle est la couleur du ciel ?"}'` |
| `/health`                    | GET    | Check if the API is running.                                               | `curl http://localhost:8000/health`                                             |

---

## 🚀 Installation

### Clone the Repository
```bash
git clone https://github.com/LucaNyckees/poivrot-belge-fine-tuning.git
cd poivrot-belge-fine-tuning
```

### Virtual environment
Use the following command lines to create and use venv python package:
```
python3.10 -m venv venv
```
Then use the following to activate the environment:
```
source venv/bin/activate
```
You can now use pip to install any packages you need for the project and run python scripts, usually through a `requirements.txt`:
```
python -m pip install -r requirements.txt
```
When you are finished, you can stop the environment by running:
```
deactivate
```

---

## Project Structure
```
├── LICENSE
|
├── config files (.env, .ini, ...)
|
├── README.md
│
├── scripts/               
│
├── notebooks/                              
│
├── requirements.txt  
|
├── __main__.py
│
├── src/                
|     ├── __init__.py
|     ├── _version.py
      ├── scraping/
             ├── __init__.py
             └── core.py
      ├── lora/
             ├── __init__.py
             └── core.py
      ├── inference/
             ├── __init__.py
             └── core.py
|
└── tests/
```
