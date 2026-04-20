import os
import io
import queue
import threading
import time
import wave
import numpy as np

import pyaudio
import pygame
from faster_whisper import WhisperModel
import edge_tts
import asyncio

class VoiceEngine:
    def __init__(self, on_command_callback, status_callback):
        self.on_command_callback = on_command_callback
        self.status_callback = status_callback
        
        # Audio constants
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.p = pyaudio.PyAudio()
        
        # Whisper model 
        self.status_callback("Loading Whisper...")
        # using tiny.en for ultra fast, low RAM
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        
        pygame.mixer.init()
        
        self.is_listening = False
        self.wake_word_enabled = False
        self.audio_queue = queue.Queue()
        
        self._stop_event = threading.Event()
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
        self.status_callback("Ready")

    def toggle_wake_word(self, enabled: bool):
        self.wake_word_enabled = enabled
        if enabled:
            self.status_callback("Wake word ON")
        else:
            self.status_callback("Wake word OFF")

    def trigger_manual_listen(self):
        if not self.is_listening:
            self.status_callback("Listening...")
            self.is_listening = True

    def _listen_loop(self):
        stream = self.p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK)
        try:
            while not self._stop_event.is_set():
                if self.is_listening:
                    self._record_command(stream)
                elif self.wake_word_enabled:
                    self._poll_wake_word(stream)
                else:
                    time.sleep(0.1)
        finally:
            stream.stop_stream()
            stream.close()

    def _poll_wake_word(self, stream):
        # A lightweight RMS detection before passing to Whisper
        frames = []
        for _ in range(0, int(self.RATE / self.CHUNK * 2)): # Listen in 2 sec chunks
            if not self.wake_word_enabled or self.is_listening:
                return
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
            
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Energy threshold to avoid empty transcriptions
        if np.sqrt(np.mean(audio_np**2)) < 0.01:
            return

        segments, _ = self.model.transcribe(audio_np, beam_size=1, vad_filter=False)
        text = " ".join([s.text for s in segments]).strip().lower()
        
        # If "hi butler" or "butler" is heard, wake up
        if "butler" in text:
            self.trigger_manual_listen()

    def _record_command(self, stream):
        # Record until silence
        frames = []
        silence_threshold = 0.01
        silence_frames = 0
        max_silence = int(self.RATE / self.CHUNK * 1.5) # 1.5s of silence
        
        while self.is_listening:
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if np.sqrt(np.mean(audio_np**2)) < silence_threshold:
                silence_frames += 1
            else:
                silence_frames = 0
                
            if silence_frames > max_silence and len(frames) > max_silence:
                break
                
        self.is_listening = False
        self.status_callback("Thinking...")
        
        # Transcribe
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_np, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        
        if text:
            # Pass to Butler 
            self.on_command_callback(text)
        else:
            self.status_callback("Ready")

    def play_tts(self, text: str):
        self.status_callback("Speaking...")
        # Run async edge-tts in a target thread
        threading.Thread(target=self._run_tts, args=(text,), daemon=True).start()

    def _run_tts(self, text: str):
        # Premium Female voice: en-US-AriaNeural or en-GB-SoniaNeural
        voice = "en-US-AriaNeural"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            communicate = edge_tts.Communicate(text, voice)
            loop.run_until_complete(communicate.save("response.mp3"))
            
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            pygame.mixer.music.unload()
            try:
                os.remove("response.mp3")
            except:
                pass
        finally:
            loop.close()
            self.status_callback("Ready")
