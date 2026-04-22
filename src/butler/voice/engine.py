import asyncio
import os
import threading
import time
from collections import deque

import numpy as np

import pyaudio
import pygame
from faster_whisper import WhisperModel
import edge_tts

from butler.config import ButlerConfig
from butler.voice.normalize import normalize_text


def _rms(audio_bytes: bytes) -> float:
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio_np.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_np**2)))


def _contains_phrase(text: str, phrase: str) -> bool:
    haystack = f" {text.casefold()} "
    needle = f" {phrase.casefold().strip()} "
    return needle in haystack


class VoiceEngine:
    def __init__(self, on_command_callback, status_callback, config: ButlerConfig | None = None):
        self.on_command_callback = on_command_callback
        self.status_callback = status_callback
        self.config = config or ButlerConfig()
        
        # Audio constants
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self._chunks_per_second = self.RATE / self.CHUNK
        self._prebuffer_seconds = max(0.0, float(self.config.voice_prebuffer_seconds))
        self._silence_seconds = max(0.1, float(self.config.voice_silence_seconds))
        self._energy_threshold = max(0.0, float(self.config.voice_energy_threshold))
        self._wake_detection_seconds = 2.0
        self._silence_limit = max(1, int(self._chunks_per_second * self._silence_seconds))
        self._min_command_frames = max(1, int(self._chunks_per_second * 0.4))
        self._prebuffer_limit = max(1, int(self._chunks_per_second * max(self._prebuffer_seconds, 0.1)))
        self._wake_window_limit = max(1, int(self._chunks_per_second * self._wake_detection_seconds))
        self._wake_word_aliases = [alias.casefold().strip() for alias in self.config.voice_wake_word_aliases if alias.strip()]
        self._transcript_aliases = self.config.transcript_aliases or {}

        self.p = pyaudio.PyAudio()
        
        # Whisper model 
        self.status_callback("Loading Whisper...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = WhisperModel(self.config.stt_model, device="cpu", compute_type="int8")
        
        pygame.mixer.init()
        
        self.is_listening = False
        self.wake_word_enabled = False
        self._command_frames: list[bytes] = []
        self._wake_frames: deque[bytes] = deque(maxlen=self._wake_window_limit)
        self._prebuffer: deque[bytes] = deque(maxlen=self._prebuffer_limit)
        self._silence_frames = 0
        self._command_started = False
        
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
            self._command_started = False
            self._command_frames = []
            self._silence_frames = 0

    def _listen_loop(self):
        stream = self.p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK)
        try:
            while not self._stop_event.is_set():
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self._prebuffer.append(data)

                if self.is_listening:
                    self._handle_listening_chunk(data)
                    continue

                if self.wake_word_enabled:
                    self._handle_wake_word_chunk(data)
                    continue

                time.sleep(0.01)
        finally:
            stream.stop_stream()
            stream.close()

    def _handle_wake_word_chunk(self, data: bytes) -> None:
        self._wake_frames.append(data)
        if len(self._wake_frames) < self._wake_window_limit:
            return

        audio_data = b"".join(self._wake_frames)
        if _rms(audio_data) < self._energy_threshold:
            return

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_np, beam_size=1, vad_filter=False, language=self.config.stt_language or "en")
        text = normalize_text(" ".join(s.text for s in segments), aliases=self._transcript_aliases)

        if any(_contains_phrase(text, alias) for alias in self._wake_word_aliases):
            self.trigger_manual_listen()
        self._wake_frames.clear()

    def _handle_listening_chunk(self, data: bytes) -> None:
        if not self._command_started:
            self._command_started = True
            self._command_frames = list(self._prebuffer)
            self._silence_frames = 0

        self._command_frames.append(data)
        if _rms(data) < self._energy_threshold:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        if self._silence_frames > self._silence_limit and len(self._command_frames) > self._min_command_frames:
            self._finish_command()

    def _finish_command(self) -> None:
        frames = self._command_frames[:]
        self._command_frames = []
        self._silence_frames = 0
        self._command_started = False
        self.is_listening = False

        self.status_callback("Thinking...")

        audio_data = b"".join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_np,
            beam_size=5,
            language=self.config.stt_language or "en",
        )
        text = normalize_text(" ".join(s.text for s in segments), aliases=self._transcript_aliases)

        if text:
            self.on_command_callback(text)
        else:
            self.status_callback("Ready")

    def play_tts(self, text: str):
        self.status_callback("Speaking...")
        # Run async edge-tts in a target thread
        threading.Thread(target=self._run_tts, args=(text,), daemon=True).start()

    def _run_tts(self, text: str):
        voice = self.config.voice or "en-IE-EmilyNeural"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        success = False
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Increased talking speed by 25% for a snappier response
                communicate = edge_tts.Communicate(text, voice, rate="+15%")
                loop.run_until_complete(communicate.save("response.mp3"))
                success = True
                break
            except Exception as e:
                time.sleep(1.0)
                
        loop.close()
        
        if success:
            try:
                pygame.mixer.music.load("response.mp3")
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                pygame.mixer.music.unload()
                os.remove("response.mp3")
            except Exception:
                pass
        else:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 170)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Fallback Failed: {e}")
                
        self.status_callback("Ready")
