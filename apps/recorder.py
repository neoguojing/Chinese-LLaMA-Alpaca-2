import sounddevice as sd
import numpy as np
import datetime
import os
import asyncio
import pdb

class AudioRecorder:
    def __init__(self, duration_per_file=5, silence_threshold=0, output_directory="./audio_files"):
        self.duration_per_file = duration_per_file
        self.silence_threshold = silence_threshold
        self.output_directory = output_directory
        self.stop_recording = False

        self.frames = []
        self.sample_rate = 16000

    async def record(self):
        duration = int(self.duration_per_file)
        channels = 1

        def callback(indata, frames, time, status):
            # pdb.set_trace()
            self.frames.append(indata.copy())

            audio_data = np.abs(indata).mean()
            if audio_data < self.silence_threshold:
                self.frames = []

            elapsed_time = len(self.frames) / self.sample_rate
            if elapsed_time >= duration:
                self.save_audio_file()

        print("开始录音...")
        self.frames = []

        with sd.InputStream(callback=callback, channels=channels, samplerate=self.sample_rate):
            while not self.stop_recording:
                await asyncio.sleep(0.1)

        await self.save_audio_file()
        print("录音退出...")

    async def save_audio_file(self):
        if len(self.frames) > 0:
            filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
            file_path = os.path.join(self.output_directory, filename)

            audio_data = np.concatenate(self.frames)
            sd.write(file_path, audio_data, self.sample_rate)

            print(f"已保存音频文件: {file_path}")

            self.frames = []

    def stop_recording(self):
        self.stop_recording = True