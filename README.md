# 🍺 Poivrot Belge: Fine-Tuning Mistral-7B with LoRA

A **fine-tuning** project for the **Mistral-7B** model to generate dialogues in the style of **Dikkenek** (Belgian humor, local expressions, and a "poivrot" tone). Uses **LoRA (Low-Rank Adaptation)** for efficient and lightweight training.

---

## 📌 Table of Contents
- [📌 Table of Contents](#-table-of-contents)
- [🎯 Purpose](#-purpose)
- [🛠 Requirements](#-requirements)
- [🚀 Installation](#-installation)
- [📂 Project Structure](#-project-structure)
- [🤖 Usage](#-usage)
  - [1. Scrape Dialogues](#1-scrape-dialogues)
  - [2. Fine-Tune the Model](#2-fine-tune-the-model)
  - [3. Inference (Generate Responses)](#3-inference-generate-responses)
- [🔧 Model Parameters](#-model-parameters)
- [💡 Example Responses](#-example-responses)
- [📊 Results and Evaluation](#-results-and-evaluation)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 🎯 Purpose
Create a model capable of **generating responses in the style of Dikkenek characters** (Belgian humor, local slang, and a "poivrot" tone). This project uses:
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

### Basic structure
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
