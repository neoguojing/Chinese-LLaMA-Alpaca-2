

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Agent
from transformers.generation import GenerationConfig
from .inference import load_model


class QWenAgent(Agent):
    """
    Agent that uses QWen model and tokenizer to generate code.

    Args:
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    agent = QWenAgent()
    agent.run("Draw me a picture of rivers and lakes.")
    ```
    """
    def __init__(self, chat_prompt_template=None, run_prompt_template=None, additional_tools=None,**kwargs):
        model_path = kwargs.pop("model_path")
        self.model,self.tokenizer = load_model(model_path=model_path,llama=False)
        self.model.generation_config.do_sample = False  # greedy
        
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_one(self, prompt, stop):
        # "Human:" 和 "Assistant:" 曾为通义千问的特殊保留字，需要替换为 "_HUMAN_:" 和 "_ASSISTANT_:"。这一问题将在未来版本修复。
        prompt = prompt.replace("Human:", "_HUMAN_:").replace("Assistant:", "_ASSISTANT_:")
        stop = [item.replace("Human:", "_HUMAN_:").replace("Assistant:", "_ASSISTANT_:") for item in stop]

        result, _ = self.model.chat(self.tokenizer, prompt, history=None)
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]

        result = result.replace("_HUMAN_:", "Human:").replace("_ASSISTANT_:", "Assistant:")
        return result
