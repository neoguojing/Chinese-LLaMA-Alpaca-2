import abc
import asyncio
import torch



class ITask(abc.ABC):
    
    @abc.abstractmethod
    def run(self):
        pass

class Task(ITask):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')

    def run(self):
        print("task")