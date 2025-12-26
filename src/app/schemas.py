from pydantic import BaseModel


class Question(BaseModel):
    text: str
    max_new_tokens: int = 30
    temperature: float = 0.3

class Response(BaseModel):
    character: str
    question: str
    answer: str