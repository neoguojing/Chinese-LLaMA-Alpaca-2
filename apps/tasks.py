
from apps.base import Task
from apps.agent import Agent
from apps.model_factory import ModelFactory
from apps.translate.nllb import TranslateTask

TASK_AGENT = 100
TASK_TRANSLATE = 200
TASK_DATA_HANDLER = 300

class TaskFactory:
    models = []
    @staticmethod
    def create_task(task_type) -> Task:
        if task_type == TASK_AGENT:
            model = ModelFactory().get_model("qwen")
            return Agent(model)
        elif task_type == TASK_TRANSLATE:
            model = ModelFactory().get_model("translate")
            return TranslateTask(model)
