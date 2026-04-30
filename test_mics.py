import pyaudio
import numpy as np

p = pyaudio.PyAudio()
print('Testing all microphones...')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get('maxInputChannels') > 0 and 'mic' in info.get('name', '').lower():
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=i)
            chunks = []
            for _ in range(int(16000/1024 * 0.5)):
                chunks.append(stream.read(1024, exception_on_overflow=False))
            stream.stop_stream()
            stream.close()
            audio_np = np.frombuffer(b''.join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_np**2))
            print(f"[{i}] {info.get('name')} -> RMS: {rms:.6f}")
        except Exception as e:
            print(f"[{i}] {info.get('name')} -> ERROR: {e}")
p.terminate()
