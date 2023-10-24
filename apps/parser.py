from typing import Any, List, Mapping, Optional
from pydantic import  Field, BaseModel,validator


class QAItem(BaseModel):
    question: str = Field(description="question")
    answer: str = Field(description="answer")

    
class QAPackage(BaseModel):
    data: List[QAItem] = Field(..., description="data")