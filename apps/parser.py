from typing import Any, List, Mapping, Optional
from pydantic import  Field, BaseModel,validator


class QA(BaseModel):
    question: str = Field(description="question ")
    answer: str = Field(description="answer")

    # You can add custom validation logic easily with Pydantic.
    @validator("question")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field