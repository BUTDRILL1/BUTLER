import pyaudio
import wave
import numpy as np
import pygame

# Initialize pygame.mixer FIRST (like BUTLER does)
pygame.mixer.init()
print("pygame.mixer initialized")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DURATION = 3

p = pyaudio.PyAudio()
default = p.get_default_input_device_info()
print(f"Default input device: [{default['index']}] {default['name']}")
print(f"\n>>> SPEAK NOW! Recording for {DURATION} seconds... <<<\n")

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

frames = []
for i in range(int(RATE / CHUNK * DURATION)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)
    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(audio_np**2)))
    bar = "#" * int(rms * 500)
    print(f"\rRMS: {rms:.6f} |{bar}", end="", flush=True)

stream.stop_stream()
stream.close()

all_audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
print(f"\n\nMax amplitude: {np.max(np.abs(all_audio)):.6f}")
print(f"Average RMS:   {np.sqrt(np.mean(all_audio**2)):.6f}")
p.terminate()
pygame.mixer.quit()
