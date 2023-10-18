import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', required=True, type=str)
parser.add_argument('--embedding_path', required=True, type=str)
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--gpu_id', default="0", type=str)
parser.add_argument('--chain_type', default="refine", type=str)
args = parser.parse_args()

os.environ['OPENAI_API_KEY'] = 'value'

from langchain.chat_models import ChatOpenAI

if __name__ == '__main__':
    print("start")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        streaming=True,
        temperature=0,
    )