import os
import json
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper
from langchain.agents import Tool,tool
from apps.tasks import TASK_IMAGE_GEN,TASK_SPEECH,TaskFactory

os.environ['SERPAPI_API_KEY'] = 'f765e0536e1a72c2f353bb1946875937b3ac7bed0270881f966d4147ac0a7943'
os.environ['WOLFRAM_ALPHA_APPID'] = 'QTJAQT-UPJ2R3KP89'

search = SerpAPIWrapper()
WolframAlpha = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()


@tool("image generate", return_direct=True)
def image_gen(input:str) ->str:
    """Useful for when you need to generate or draw a picture by input text.Text to image diffusion model capable of generating photo-realistic images given any text input."""
    task = TaskFactory.create_task(TASK_IMAGE_GEN)
    return task.run()

@tool("speech or audio generate", return_direct=True)
def text2speech(input:str) ->str:
    """Useful for when you need to transfer text to speech or audio.Speech to speech translation.Speech to text translation.Text to speech translation.Text to text translation.Automatic speech recognition."""
    task = TaskFactory.create_task(TASK_SPEECH)
    return task.run()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Math",
        func=WolframAlpha.run,
        description="Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life."
    ),
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org."
    ),
    image_gen
    # SpeechText(),

]

def tool_wrapper_for_qwen(tool):
    def tool_(query):
        query = json.loads(query)["query"]
        return tool.run(query)
    return tool_

# 以下是给千问看的工具描述：
TOOLS = [
    {
        'name_for_human':
            'google search',
        'name_for_model':
            'Search',
        'description_for_model':
            'useful for when you need to answer questions about current events.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "search query of google",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(search)
    },
    {
        'name_for_human':
            'Wolfram Alpha',
        'name_for_model':
            'Math',
        'description_for_model':
            'Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "the problem to solved by Wolfram Alpha",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(WolframAlpha)
    },  
    {
        'name_for_human':
            'arxiv',
        'name_for_model':
            'Arxiv',
        'description_for_model':
            'A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "the document id of arxiv to search",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(arxiv)
    }

]
