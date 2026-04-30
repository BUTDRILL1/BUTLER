import pyaudio
import numpy as np

p = pyaudio.PyAudio()
print('--- PyAudio Input Devices ---')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get('maxInputChannels') > 0:
        print(f"[{i}] {info.get('name')}")

try:
    print('Testing default device for 2 seconds...')
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    chunks = []
    for _ in range(int(16000/1024 * 2)):
        data = stream.read(1024, exception_on_overflow=False)
        chunks.append(data)
    stream.stop_stream()
    stream.close()
    
    audio_np = np.frombuffer(b''.join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
    print(f"Max amplitude: {np.max(np.abs(audio_np)):.6f}")
    print(f"Average RMS: {np.sqrt(np.mean(audio_np**2)):.6f}")
except Exception as e:
    print(f"Error reading from default device: {e}")
finally:
    p.terminate()
