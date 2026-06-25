"""Quick diagnostic: run BUTLER's exact voice engine in isolation with debug prints."""
import pyaudio
import numpy as np
import time
import threading
from collections import deque

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
ENERGY_THRESHOLD = 0.01
SILENCE_SECONDS = 2.0
PREBUFFER_SECONDS = 0.75

chunks_per_sec = RATE / CHUNK
silence_limit = int(chunks_per_sec * SILENCE_SECONDS)
min_command_frames = int(chunks_per_sec * 0.4)
prebuffer_limit = int(chunks_per_sec * max(PREBUFFER_SECONDS, 0.1))

print(f"Config: threshold={ENERGY_THRESHOLD}, silence_limit={silence_limit} chunks, min_frames={min_command_frames}")

p = pyaudio.PyAudio()

# Also init pygame like BUTLER does
import pygame
pygame.mixer.init()

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

is_listening = False
command_started = False
has_spoken = False
command_frames = []
silence_frames = 0
prebuffer = deque(maxlen=prebuffer_limit)

def simulate_mic_click():
    global is_listening, command_started, has_spoken, command_frames, silence_frames
    print("\n>>> MIC CLICKED! Listening... <<<")
    is_listening = True
    command_started = False
    has_spoken = False
    command_frames = []
    silence_frames = 0

# Simulate a mic click after 2 seconds
def delayed_click():
    time.sleep(2)
    simulate_mic_click()

threading.Thread(target=delayed_click, daemon=True).start()
print("Mic will auto-click in 2 seconds. Then SPEAK!")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        prebuffer.append(data)
        rms = float(np.sqrt(np.mean(
            (np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)**2
        )))

        if not is_listening:
            continue

        if not command_started:
            command_started = True
            has_spoken = False
            command_frames = list(prebuffer)
            silence_frames = 0
            print(f"  [command_started] prebuffer frames={len(command_frames)}")

        command_frames.append(data)

        if rms < ENERGY_THRESHOLD:
            silence_frames += 1
        else:
            has_spoken = True
            silence_frames = 0

        current_limit = silence_limit if has_spoken else int(silence_limit * 2.5)
        bar = "#" * min(int(rms * 300), 50)
        status = "SPEECH" if rms >= ENERGY_THRESHOLD else "silent"
        print(f"\r  RMS={rms:.4f} [{status}] spoken={has_spoken} silence={silence_frames}/{current_limit} frames={len(command_frames)} |{bar:<50}", end="", flush=True)

        if silence_frames > current_limit and len(command_frames) > min_command_frames:
            print(f"\n\n>>> COMMAND FINISHED! Total frames={len(command_frames)}, has_spoken={has_spoken} <<<")
            is_listening = False
            command_started = False
            break

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.mixer.quit()
