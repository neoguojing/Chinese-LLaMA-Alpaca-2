import pyaudio
import wave
import datetime
import numpy as np
import asyncio
import os

class AudioRecorder:
    def __init__(self, duration_per_file=5, silence_threshold=50, output_directory="./audio_files"):
        self.duration_per_file = duration_per_file
        self.silence_threshold = silence_threshold
        self.output_directory = output_directory
        self.stop_recording = False

        self.p = pyaudio.PyAudio()
        self.frames = []
        self.stream = None

    async def record(self):
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 44100

        self.stream = self.p.open(format=format,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=chunk)

        print("开始录音...")
        self.frames = []
        start_time = datetime.datetime.now()

        while not self.stop_recording:
            data = self.stream.read(chunk)
            self.frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.abs(audio_data).mean() < self.silence_threshold:
                self.frames = []
                start_time = datetime.datetime.now()

            elapsed_time = datetime.datetime.now() - start_time
            if elapsed_time.total_seconds() >= self.duration_per_file:
                await self.save_audio_file()

        await self.save_audio_file()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    async def save_audio_file(self):
        if len(self.frames) > 0:
            filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
            file_path = os.path.join(self.output_directory, "/",filename)

            wf = wave.open(file_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))
            wf.close()

            print(f"已保存音频文件: {file_path}")

            self.frames = []
