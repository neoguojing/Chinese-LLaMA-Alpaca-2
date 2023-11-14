# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import  Field
from apps.base import Task
# BCP47 code https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
class Translate(LLM):
    model_path: str = Field(None, alias='model_path')
    model: Any = None 
    tokenizer: Any = None
    src_lang: str = "eng_Latn" 
    dst_lang: str = "zho_Hans"

    def __init__(self, model_path: str,**kwargs):
        super(Translate, self).__init__()
        self.model_path = model_path
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",cache_dir="../../model/nllb")
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M",cache_dir="../../model/nllb")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        src_lang = kwargs.pop("src_lang")
        if src_lang is not None:
            self.src_lang = src_lang

        dst_lang = kwargs.pop("dst_lang")
        if dst_lang is not None:
            self.dst_lang = dst_lang

    @property
    def _llm_type(self) -> str:
        return "facebook/nllb-200-distilled-600M"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # translate Hindi to French
        self.tokenizer.src_lang = self.src_lang
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded.to(self.device)
        generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.dst_lang],max_new_tokens=512)
        output = self.tokenizer.decode(generated_tokens[0].cpu(), skip_special_tokens=True)

        return ''.join(output).strip('</s>')

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}

class TranslateTask(Task):
    excurtor: Any = None
    def __init__(self,llm: LLM):
        self.excurtor = llm

    def run(self,input: str=None):
        output = self.excurtor.predict(input)
        return output


# if __name__ == '__main__':
#     input = '''
#     Step 1: Choose a topic. I'll select geography as the topic for this question, as it is a subject rich with factual information. Step 2: Decide on a specific geographical aspect to focus on. I'll concentrate on capital cities, which are concrete data points within the field of geography. Step 3: Pick a country for the question. I'll choose Australia for its unique geography and its status as both a continent and a country. Step 4: Formulate the question, ensuring that it seeks a factual answer. My question will ask about the capital city of Australia. Step 5: Verify that a factual answer to the question exists. In this case, I'll confirm that Australia does have a capital city. The question I generated is: "What is the capital city of Australia?" The factual answer to this question is: "Canberra."
#     '''
#     out = translate(input)
#     print(out)

    