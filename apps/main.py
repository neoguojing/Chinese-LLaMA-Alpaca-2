
import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
import time
import asyncio
from abc import ABC, abstractmethod
import aioconsole
import copy
from apps.tasks import TaskFactory,TASK_TRANSLATE,TASK_AGENT,TASK_SPEECH
from apps.model_factory import ModelFactory
from apps.recorder import AudioRecorder
from apps.config import message
# 创建一个共享的队列
input = asyncio.Queue()
output = asyncio.Queue()

def to_agent(input:str,_from:str):
    msg = copy.deepcopy(message)
    msg["data"] = input
    msg["to"] = TASK_AGENT
    msg["from"] = _from
    return msg

def to_speech(input:str,_from:str):
    msg = copy.deepcopy(message)
    msg["data"] = input
    msg["to"] = TASK_SPEECH
    msg["from"] = _from
    return msg

async def keyboard():
    while True:
        input_text = await aioconsole.ainput("Enter : ")
        msg = to_agent(input_text,"keyboard")
        input.put_nowait(msg)
        await aioconsole.aprint(f"Produced: {msg}")
async def output_loop():
    while True:
        item = await output.get()
        await aioconsole.aprint("Output:",item+"\n")

# 消费者协程函数
async def message_bus():
    translator = None
    agent = None
    translator = TaskFactory.create_task(TASK_TRANSLATE)
    # agent = TaskFactory.create_task(TASK_AGENT)
    speech = TaskFactory.create_task(TASK_SPEECH)
    while True:
        item = await input.get()
        await aioconsole.aprint(f"Consumed: {item}")
        _from = item["from"]
        # 模拟消费延迟
        if item["to"] == TASK_AGENT:
            out = await agent.arun(item["data"])
            output.put_nowait(out)
            if _from == TASK_SPEECH:
                msg = to_speech(out,"agent")
                input.put_nowait(msg)
        elif item["to"] == TASK_TRANSLATE:
            out = await translator.arun(item["data"])
            output.put_nowait(out)
        elif item["to"] == TASK_SPEECH:
            out = await speech.arun(item["data"])
            output.put_nowait(out)

async def garbage_collection():
    while True:
        await asyncio.sleep(60)
        TaskFactory.release()
        ModelFactory.release()

async def audio_input():
    recorder = AudioRecorder(input)
    await recorder.record()

async def main():
    # 并发运行多个异步任务
    await asyncio.gather(keyboard(),audio_input(), message_bus(),output_loop(),garbage_collection())

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping the event loop")
    finally:
        loop.close()