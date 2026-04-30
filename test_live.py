import pyaudio
import numpy as np
import threading
import time

# Simulate BUTLER's exact listen loop setup
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
ENERGY_THRESHOLD = 0.01
SILENCE_LIMIT = int((RATE / CHUNK) * 2.0)  # 2 seconds

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

print(f"Energy threshold: {ENERGY_THRESHOLD}")
print(f"Silence limit (chunks): {SILENCE_LIMIT}")
print(f"\nStreaming from mic continuously (like BUTLER's listen_loop)...")
print(f"Press Ctrl+C to stop.\n")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio_np**2)))
        above = "SPEECH" if rms > ENERGY_THRESHOLD else "silent"
        bar_len = min(int(rms * 300), 80)
        bar = "#" * bar_len
        print(f"\rRMS: {rms:.6f} [{above}] |{bar:<80}", end="", flush=True)
except KeyboardInterrupt:
    print("\n\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
