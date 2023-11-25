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
import time
import threading
import noisereduce as nr
import librosa
from pynput.keyboard import Key

class AudioRecorder:
    def __init__(self,output_queue:asyncio.Queue, duration_per_file=10, silence_threshold=0.001, output_directory="./"):
        self.duration_per_file = duration_per_file
        self.silence_threshold = silence_threshold
        self.output_directory = output_directory
        self.record_swich = True

        self.frames = None
        self.sample_rate = 16000
        self.gain_factor = 2.0  # 增益因子
        self.input_queue = queue.Queue()
        self.output_queue = output_queue
        self.need_save_audio = True
        self.speeh2text = TaskFactory.create_task(TASK_SPEECH)
        self.pause = True

    async def record(self):
        channels = 1
        # indata 二维数组,每一列代表一个channel的数据,2个channel则有两列
        # frames 帧数
        def callback(indata, frames, _time, status):
            # 暂停录音的情况下，输出录音
            if self.pause:
                if self.frames is not None and self.frames.size > 0:
                    self.input_queue.put_nowait(self.frames)
                    self.frames = None
                    print("中止录音")
                return 
            
            print("继续录音")
            audio_data = indata.copy() 
            print("audio_data",audio_data.shape)

            # 计算每个通道的能量
            energy_per_channel = np.sum(np.square(audio_data), axis=0)
            print("energy_per_channel:",energy_per_channel)
            # 过滤较低能量的通道
            is_silent_channel = energy_per_channel < self.silence_threshold
            print("energy_per_channel:",is_silent_channel)
            #选择非静音的通道
            filtered_audio_data = audio_data[:, ~is_silent_channel]
            # 去除静音
            # filtered_audio_data, _ = librosa.effects.trim(audio_data,top_db=50)
            if filtered_audio_data.size == 0:
                return 
            
            # 降噪
            filtered_audio_data = nr.reduce_noise(y=filtered_audio_data.T, sr=self.sample_rate).T
            print(filtered_audio_data.shape)

            if self.frames is None:
                self.frames = filtered_audio_data
            else:
                self.frames = np.append(self.frames, filtered_audio_data, axis=0)

            num_samples = self.frames.shape[0] 
            duration = num_samples / self.sample_rate
            if duration >= self.duration_per_file:
                self.input_queue.put_nowait(self.frames)
                self.frames = None
                self.pause = True
                print("录音超过10s,中止录音")

        print("开始录音...")
        self.frames = None

        with sd.InputStream(callback=callback, channels=channels, samplerate=self.sample_rate):
            while self.record_swich:
                try:
                    frames = self.input_queue.get_nowait()
                    
                except Exception:
                    await asyncio.sleep(0.1)
                    continue

                if self.need_save_audio:
                    await self.save_audio_file(frames)
                    
                if self.output_queue is not None:
                    # 将语音转文字，并输出到agent
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
        self.record_swich = False

    def on_keypress(self,event):
        self.pause = not self.pause
        