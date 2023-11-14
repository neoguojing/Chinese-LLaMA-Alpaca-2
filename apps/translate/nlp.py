# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')


# model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M",cache_dir="../../model/m2m")
# tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M",cache_dir="../../model/m2m")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",cache_dir="../../model/nllb")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M",cache_dir="../../model/nllb")
model.to(device)

# BCP47 code https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
def translate(input: str,src_lang: str="eng_Latn",dst_lang: str="zho_Hans"):
    # translate Hindi to French
    tokenizer.src_lang = src_lang
    encoded = tokenizer(input, return_tensors="pt")
    encoded.to(device)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[dst_lang],max_new_tokens=512)
    output = tokenizer.decode(generated_tokens[0].cpu(), skip_special_tokens=True)

    return ''.join(output).strip('</s>')

if __name__ == '__main__':
    input = '''
    Step 1: Choose a topic. I'll select geography as the topic for this question, as it is a subject rich with factual information. Step 2: Decide on a specific geographical aspect to focus on. I'll concentrate on capital cities, which are concrete data points within the field of geography. Step 3: Pick a country for the question. I'll choose Australia for its unique geography and its status as both a continent and a country. Step 4: Formulate the question, ensuring that it seeks a factual answer. My question will ask about the capital city of Australia. Step 5: Verify that a factual answer to the question exists. In this case, I'll confirm that Australia does have a capital city. The question I generated is: "What is the capital city of Australia?" The factual answer to this question is: "Canberra."
    '''
    out = translate(input)
    print(out)