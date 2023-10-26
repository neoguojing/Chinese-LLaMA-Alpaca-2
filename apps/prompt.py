
from langchain.prompts import PromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate

class PromptFactory:
    @staticmethod
    def qa_data_generate_prompt(format_instructions):
        return ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "将\n{text}转换中文问答对，格式如下：\n{format_instructions}"
            )
        ],
        input_variables=["text"],
        partial_variables={
            "format_instructions": format_instructions,
        },
    )
