import os

import sounddevice as sd
import numpy as np
import queue
import time
from scipy.io.wavfile import write

# Arguments
SAMPLE_RATE = 16000
CHANNELS = 1            # single channel
BLOCK_SIZE = 1024
THRESHOLD = 0.01        # threshold of sound to stop recording
SILENCE_TIMEOUT = 3     # in second
OUTPUT_FILENAME = os.path.dirname(__file__) + "/output/recording.wav"  # 输出文件名

audio_queue = queue.Queue()     # store recording data
is_recording = False            # record status
last_voice_time = 0.0
recording_buffer = []


def audio_callback(indata, frames, tm, status):
    global is_recording, last_voice_time, recording_buffer

    if status:
        print(f"Status Error: {status}")

    # calculate RMS
    volume = np.linalg.norm(indata) / np.sqrt(len(indata))

    # detect start time to record
    if volume > THRESHOLD:
        if not is_recording:
            print("Start recording...")
            is_recording = True
        last_voice_time = time.time()

    if is_recording:
        recording_buffer.append(indata.copy())
        audio_queue.put(indata.copy())


def audio_record():
    global is_recording, last_voice_time, recording_buffer

    print("Waiting for recording...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCK_SIZE, callback=audio_callback):
        while True:
            # 如果正在录音，检查是否超时
            if is_recording:
                if last_voice_time and time.time() - last_voice_time > SILENCE_TIMEOUT:
                    print("End recording and save file.")
                    break

    # 将录音缓冲区保存为 WAV 文件
    recorded_audio = np.concatenate(recording_buffer, axis=0)
    write(OUTPUT_FILENAME, SAMPLE_RATE, (recorded_audio * 32767).astype(np.int16))
    print(f"Audio has been saved to {OUTPUT_FILENAME}")


if __name__ == "__main__":
    audio_record()
