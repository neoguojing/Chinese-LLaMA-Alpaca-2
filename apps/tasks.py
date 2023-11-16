
from apps.base import Task
from apps.agent import Agent
from apps.model_factory import ModelFactory
from langchain.tools import BaseTool,tool
import asyncio
TASK_AGENT = 100
TASK_TRANSLATE = 200
TASK_DATA_HANDLER = 300
TASK_IMAGE_GEN = 400
TASK_SPEECH = 500

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
    _lock = asyncio.Lock()  # 异步锁

    @staticmethod
    def create_task(task_type) -> Task:
        if task_type not in TaskFactory._instances:
            with TaskFactory._lock:
                if task_type not in TaskFactory._instances:
                    if task_type == TASK_AGENT:
                        # model = ModelFactory.get_model("qwen")
                        instance = Agent()
                    elif task_type == TASK_TRANSLATE:
                        # model = ModelFactory.get_model("translate")
                        instance = TranslateTask()
                    elif task_type == TASK_IMAGE_GEN:
                        instance = ImageGenTask()
                    elif task_type == TASK_SPEECH:
                        instance = Text2Speech()

                    TaskFactory._instances[task_type] = instance

        return TaskFactory._instances[task_type]
