import re


def select_first_in_multi_reponses_output(text: str) -> str | None:
    match = re.search(r"1\.\s*(.*)", text)
    first_answer = match.group(1) if match else None
    return first_answer