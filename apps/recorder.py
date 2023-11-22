import sounddevice as sd
import numpy as np
import datetime
import os
import asyncio
import queue
import pdb
from scipy.io import wavfile
from scipy.signal import medfilt
from apps.tasks import TaskFactory,TASK_SPEECH,TASK_AGENT
from apps.config import message
import copy

class AudioRecorder:
    def __init__(self,output_queue:asyncio.Queue, duration_per_file=5, silence_threshold=0.001, output_directory="./"):
        self.duration_per_file = duration_per_file
        self.silence_threshold = silence_threshold
        self.output_directory = output_directory
        self.stop_recording = False

        self.frames = None
        self.sample_rate = 16000
        self.gain_factor = 2.0  # 增益因子
        self.input_queue = queue.Queue()
        self.output_queue = output_queue
        self.need_save_audio = True
        self.speeh2text = TaskFactory.create_task(TASK_SPEECH)

    async def record(self):
        channels = 1
        # indata 二维数组,每一列代表一个channel的数据,2个channel则有两列
        # frames 帧数
        def callback(indata, frames, time, status):
            # 
            # print("indata:",indata)
            audio_data = indata.copy()
            # 计算每个通道的能量
            energy_per_channel = np.sum(np.square(audio_data), axis=0)
            # print("energy_per_channel:",energy_per_channel)
            # 过滤较低能量的通道
            is_silent_channel = energy_per_channel < self.silence_threshold
            # print("is_silent_channel:",is_silent_channel)
            #选择非静音的通道
            filtered_audio_data = audio_data[:, ~is_silent_channel]
            # print("filtered_audio_data:",filtered_audio_data)

            if filtered_audio_data.size == 0:
                # print("silence")
                return 
            
            if self.frames is None:
                self.frames = filtered_audio_data
            else:
                self.frames = np.append(self.frames, filtered_audio_data, axis=0)

            num_samples = self.frames.shape[0] 
            duration = num_samples / self.sample_rate
            # print("num_samples:",num_samples,self.sample_rate,duration)
            # pdb.set_trace()
            if duration >= self.duration_per_file:
                self.input_queue.put_nowait(self.frames)
                self.frames = None

            # 中值滤波器 表示在每个通道上应用大小为 3 的窗口进行中值滤波
            # filtered_audio_data = medfilt(filtered_audio_data, kernel_size=(3, 1))

        print("开始录音...")
        self.frames = None

        with sd.InputStream(callback=callback, channels=channels, samplerate=self.sample_rate):
            while not self.stop_recording:
                try:
                    frames = self.input_queue.get_nowait()
                except Exception:
                    await asyncio.sleep(0.1)
                    continue

                if self.need_save_audio:
                    await self.save_audio_file(frames)
                    
                if self.output_queue is not None:
                    text =  await self.speeh2text.arun(frames,generate_speech=False,tgt_lang="cmn")
                    msg = copy.deepcopy(message)
                    print("audio to text:",text)
                    msg["data"] = text
                    msg["to"] = TASK_AGENT
                    msg["from"] = "recorder"
                    await self.output_queue.put(msg)
                else:
                    await asyncio.sleep(0.1)

        print("录音退出...")

    async def save_audio_file(self,frames):
        
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
        file_path = os.path.join(self.output_directory, filename)
        wavfile.write(file_path,rate=self.sample_rate, data=frames)

        print(f"已保存音频文件: {file_path}")

    def stop_recording(self):
        self.stop_recording = True