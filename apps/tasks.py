from langchain.tools import tool
from langchain.chains.llm import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper
from langchain.agents import Tool
import os
import sys
import time
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
import threading
from apps.base import Task,function_stats
from apps.model_factory import ModelFactory
from apps.prompt import QwenAgentPromptTemplate
from apps.parser import QwenAgentOutputParser

TASK_AGENT = 100
TASK_TRANSLATE = 200
TASK_DATA_HANDLER = 300
TASK_IMAGE_GEN = 400
TASK_SPEECH = 500


os.environ['SERPAPI_API_KEY'] = 'f765e0536e1a72c2f353bb1946875937b3ac7bed0270881f966d4147ac0a7943'
os.environ['WOLFRAM_ALPHA_APPID'] = 'QTJAQT-UPJ2R3KP89'

search = SerpAPIWrapper()
WolframAlpha = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()


@tool("image generate", return_direct=True)
def image_gen(input:str) ->str:
    """Useful for when you need to generate or draw a picture by input text.Text to image diffusion model capable of generating photo-realistic images given any text input."""
    task = TaskFactory.create_task(TASK_IMAGE_GEN)
    return task.run(input)

@tool("speech or audio generate", return_direct=True)
def text2speech(input:str) ->str:
    """Useful for when you need to transfer text to speech or audio.Speech to speech translation.Speech to text translation.Text to speech translation.Text to text translation.Automatic speech recognition."""
    task = TaskFactory.create_task(TASK_SPEECH)
    return task.run(input)

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
    image_gen,
    text2speech,

]

class Agent(Task):
    def __init__(self):
        prompt = QwenAgentPromptTemplate(
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )

        output_parser = QwenAgentOutputParser()
        llm_chain = LLMChain(llm=self.excurtor, prompt=prompt)

        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        self._executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    @function_stats
    def run(self,input: str=None):
        if input is None or input == "":
            return ""
        output = self._executor.run(input)
        return output
    
    def init_model(self):
        model = ModelFactory.get_model("qwen")
        return model
    
    def destroy(self):
        print("Agent model should not be destroy ")
    

class ImageGenTask(Task):

    def init_model(self):
        model = ModelFactory.get_model("text2image")
        return model

class Text2Speech(Task):
  def init_model(self):
        model = ModelFactory.get_model("speech")
        return model

class TranslateTask(Task):
    def init_model(self):
        model = ModelFactory.get_model("translate")
        return model
    
class TaskFactory:
    _instances = {}
    _lock = threading.Lock()  # 异步锁

    @staticmethod
    def create_task(task_type) -> Task:
        if task_type not in TaskFactory._instances:
            with TaskFactory._lock:
                if task_type not in TaskFactory._instances:
                    if task_type == TASK_AGENT:
                        instance = Agent()
                    elif task_type == TASK_TRANSLATE:
                        instance = TranslateTask()
                    elif task_type == TASK_IMAGE_GEN:
                        instance = ImageGenTask()
                    elif task_type == TASK_SPEECH:
                        instance = Text2Speech()

                    TaskFactory._instances[task_type] = instance

        return TaskFactory._instances[task_type]

    @staticmethod
    def release():
        with TaskFactory._lock:
            for k,task in TaskFactory._instances.items():
                last_call_time = task.get_last_call_time
                if last_call_time is None:
                    continue

                elapsed_time = time.time() - last_call_time
                elapsed_minutes = int(elapsed_time / 60)
                # print("elapsed_time:",elapsed_time)
                if elapsed_minutes >= 10:
                    model_name = task.bind_model_name()
                    print("ready to release ",model_name)
                    if model_name is not None:
                        task.destroy()
                        ModelFactory.destroy(model_name)


        