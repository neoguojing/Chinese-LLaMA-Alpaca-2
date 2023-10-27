
from langchain.prompts import PromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate,SystemMessagePromptTemplate

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

    @staticmethod
    def caibao_analyse_prompt(format_instructions):
        template="您是一个专业的财务报表分析师,能够通过用户输入的财报片段，分析有价值的信息，并将分析结果转换为问答形式，输出{format_instructions}格式"
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        return ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "{text}"
                ),
                system_message_prompt
            ],
            input_variables=["text"],
            partial_variables={
                "format_instructions": format_instructions,
            },
        )
