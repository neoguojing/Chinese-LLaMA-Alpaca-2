from typing import Any, List, Mapping, Optional
from pydantic import  Field, BaseModel,validator


class QAItem(BaseModel):
    question: str = Field(description="question")
    answer: str = Field(description="answer")

    
class QAPackage(BaseModel):
    data: List[QAItem] = Field(..., description="data")

class QwenConversationItem(BaseModel):
    from_: str = Field(..., alias="from", description="发送方")
    value: str = Field(..., description="消息内容")


class QwenItem(BaseModel):
    id: str = Field(..., description="标识")
    conversations: List[QwenConversationItem] = Field(..., description="对话列表")


class QwenPackage(BaseModel):
    data: List[QwenItem] = Field(..., description="问题答案列表")