import asyncio
import os
import threading
import time
import queue
import tempfile
from collections import deque
import logging

import numpy as np

logger = logging.getLogger(__name__)

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
    def __init__(self, on_command_callback, status_callback, config: ButlerConfig | None = None, mic_enabled: bool = True):
        self.on_command_callback = on_command_callback
        self.status_callback = status_callback
        self.config = config or ButlerConfig()
        self._mic_enabled = mic_enabled
        
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

        # IMPORTANT: pygame.mixer (SDL2) MUST init before PyAudio (PortAudio)
        # on Windows AMD audio drivers, otherwise the mic stream returns silence.
        pygame.mixer.init()
        
        self.is_listening = False
        self.wake_word_enabled = False
        self._command_frames: list[bytes] = []
        self._wake_frames: deque[bytes] = deque(maxlen=self._wake_window_limit)
        self._prebuffer: deque[bytes] = deque(maxlen=self._prebuffer_limit)
        self._silence_frames = 0
        self._command_started = False
        self.live_mode = False
        self._interrupted = False
        
        self._stop_event = threading.Event()
        
        if self._mic_enabled:
            self.p = pyaudio.PyAudio()
            
            # Whisper model 
            self.status_callback("Loading Whisper...")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = WhisperModel(self.config.stt_model, device="cpu", compute_type="int8")
            
            try:
                import static_ffmpeg
                static_ffmpeg.add_paths()
            except:
                pass
            
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()
        else:
            self.p = None
            self.model = None
            self.listen_thread = None
        
        self._tts_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        
        self._fetcher_thread = threading.Thread(target=self._tts_fetcher, daemon=True)
        self._player_thread = threading.Thread(target=self._tts_player, daemon=True)
        
        self._fetcher_thread.start()
        self._player_thread.start()
        
        self.status_callback("Ready")

    def stop(self):
        self._stop_event.set()
        self._tts_queue.put(None)
        try:
            if hasattr(self, 'p') and self.p:
                self.p.terminate()
        except Exception:
            pass
        self._audio_queue.put(None)
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2.0)
        if self._fetcher_thread.is_alive():
            self._fetcher_thread.join(timeout=2.0)
        if self._player_thread.is_alive():
            self._player_thread.join(timeout=2.0)
        try:
            if self.p:
                self.p.terminate()
        except Exception:
            pass

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
            self._has_spoken = False
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

                # Barge-in detection: If she is speaking, check for user voice
                if pygame.mixer.music.get_busy() and not self.is_listening:
                    rms = _rms(data)
                    # Use a slightly higher threshold when she is talking to avoid self-interruption
                    barge_in_threshold = self._energy_threshold * 1.5
                    if rms > barge_in_threshold:
                        self._interrupted = True
                        pygame.mixer.music.stop()

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
        if not getattr(self, '_command_started', False):
            self._command_started = True
            self._has_spoken = False
            self._command_frames = list(self._prebuffer)
            self._silence_frames = 0

        self._command_frames.append(data)
        if _rms(data) < self._energy_threshold:
            self._silence_frames += 1
        else:
            self._has_spoken = True
            self._silence_frames = 0

        # Allow more time (e.g., 5 seconds) to start speaking, but 2 seconds of silence AFTER speaking
        current_limit = self._silence_limit if getattr(self, '_has_spoken', False) else self._silence_limit * 2.5

        if self._silence_frames > current_limit and len(self._command_frames) > self._min_command_frames:
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
            # In live mode, restart listening if no command was detected
            if self.live_mode:
                self.trigger_manual_listen()

    def play_tts(self, text: str):
        self._tts_queue.put(text)

    def _tts_fetcher(self):
        """Thread 1: Downloads audio files in the background."""
        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            
            path, provider = self._fetch_audio(text)
            if path:
                self._audio_queue.put((path, provider))
            self._tts_queue.task_done()

    def _tts_player(self):
        """Thread 2: Plays audio files as soon as they are ready."""
        while True:
            item = self._audio_queue.get()
            if item is None:
                break
            
            path, provider = item
            self._play_audio_file(path, actual_provider=provider)
            self._audio_queue.task_done()

    def _fetch_audio(self, text: str) -> tuple[str | None, str]:
        # Switch based on user-selected provider
        provider = (self.config.tts_provider or "edge-tts").lower()
        
        if provider == "nvidia" and self.config.nvidia_api_key:
            path = self._fetch_audio_nvidia(text)
            if path:
                return path, "nvidia"
        
        if provider == "freetts":
            path = self._fetch_audio_freetts(text)
            if path:
                return path, "freetts"
        
        # Primary Edge TTS if selected
        if provider == "edge-tts":
            path = self._fetch_audio_edge(text)
            if path:
                return path, "edge-tts"

        # Final Universal Fallback: Google TTS (unlimited, stable)
        # Skip Edge TTS retries if it wasn't the primary choice
        path = self._fetch_audio_google(text)
        return path, "google"

    def _fetch_audio_google(self, text: str) -> str | None:
        """Emergency fallback using Google TTS (gTTS), with FFmpeg speed adjustment."""
        try:
            from gtts import gTTS
            import subprocess
            
            ffmpeg_path = "ffmpeg"
            
            tts = gTTS(text=text, lang=self.config.stt_language or "en")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                raw_path = tmp.name
                tts.save(raw_path)
                
            # Get rate from config (e.g., "+15%" -> 1.15)
            rate_str = self.config.rate or "+0%"
            try:
                # Convert "+15%" to 1.15 factor
                factor = 1.0 + (float(rate_str.strip("%").strip("+").strip("-")) / 100.0)
                if "-" in rate_str:
                    factor = 1.0 - (float(rate_str.strip("%").strip("+").strip("-")) / 100.0)
            except:
                factor = 1.0
            
            if abs(factor - 1.0) < 0.01:
                return raw_path # No speed adjustment needed
                
            # Use FFmpeg to adjust speed (atempo filter)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as out_tmp:
                out_path = out_tmp.name
                
            cmd = [
                ffmpeg_path, "-y", "-i", raw_path, 
                "-filter:a", f"atempo={factor}", 
                "-vn", out_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Clean up raw file
            if os.path.exists(raw_path):
                os.remove(raw_path)
                
            return out_path
        except Exception as e:
            logger.error(f"Google TTS Fallback Failed: {e}", exc_info=True)
            return None

    def _fetch_audio_freetts(self, text: str) -> str | None:
        """Fetch audio using FreeTTS.org REST API."""
        try:
            import requests
            voice = self.config.voice or "en-US-AvaNeural"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Content-Type": "application/json"
            }
            
            # Step 1: Request synthesis
            url_tts = "https://freetts.org/api/tts"
            payload = {
                "text": text,
                "voice": voice,
                "rate": "+0%",
                "pitch": "+0Hz"
            }
            res_tts = requests.post(url_tts, json=payload, headers=headers, timeout=15)
            if res_tts.status_code != 200:
                logger.error(f"FreeTTS Request Error: {res_tts.status_code}")
                return None
            
            file_id = res_tts.json().get("file_id")
            if not file_id:
                logger.error("FreeTTS Error: No file_id returned")
                return None
                
            # Step 2: Download the audio
            url_audio = f"https://freetts.org/api/audio/{file_id}"
            res_audio = requests.get(url_audio, headers={"User-Agent": headers["User-Agent"]}, timeout=15)
            if res_audio.status_code != 200:
                logger.error(f"FreeTTS Download Error: {res_audio.status_code}")
                return None
                
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
                with open(tmp_path, "wb") as f:
                    f.write(res_audio.content)
                return tmp_path
        except Exception as e:
            logger.error(f"FreeTTS Fetch Failed: {e}", exc_info=True)
            return None

    def _fetch_audio_edge(self, text: str) -> str | None:
        voice = self.config.voice or "en-IE-EmilyNeural"
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        success = False
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Force a fresh connection by creating a new Communicate object each time
                # and using a cleaner execution block
                async def _task():
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(tmp_path)
                
                asyncio.run(_task())
                
                # Verify file was actually written and has content
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    success = True
                    break
                else:
                    raise ValueError("Empty audio file received")
            except Exception as e:
                logger.warning(f"Edge TTS Attempt {attempt+1} Failed: {e}")
                time.sleep(1.0) # Longer pause between retries
        
        return tmp_path if success else None

    def _fetch_audio_nvidia(self, text: str) -> str | None:
        """Fetch audio using Nvidia Riva API over gRPC."""
        try:
            import grpc
            import riva.client

            # Connect to Nvidia's Cloud gRPC server
            auth = riva.client.Auth(
                use_ssl=True,
                uri="grpc.nvcf.nvidia.com:443",
                metadata_args=[
                    ("authorization", f"Bearer {self.config.nvidia_api_key}"),
                    ("function-id", "55cf67bf-600f-4b04-8eac-12ed39537a08")
                ]
            )
            tts_service = riva.client.SpeechSynthesisService(auth)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            # The exact voice name or model ID depends on the NIM. Usually left empty or generic for zero-shot
            # if the endpoint defaults to the zero-shot model, or we specify English-US.Female-1.
            req = {
                "text": text,
                "language_code": "en-US",
                "voice_name": "Magpie-ZeroShot",
                "sample_rate_hz": 22050,
                "encoding": riva.client.AudioEncoding.LINEAR_PCM
            }
            
            if getattr(self.config, "nvidia_tts_reference_audio", "") and os.path.exists(self.config.nvidia_tts_reference_audio):
                from pathlib import Path
                req["zero_shot_audio_prompt_file"] = Path(self.config.nvidia_tts_reference_audio)

            response = tts_service.synthesize(**req)

            if response and response.audio:
                import wave
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(22050)
                    wf.writeframes(response.audio)
                return tmp_path
            else:
                logger.error("Nvidia TTS Error: Empty response from Riva.")
                return None
        except Exception as e:
            logger.error(f"Nvidia Riva TTS Fetch Failed: {e}", exc_info=True)
            return None

    def _play_audio_file(self, path: str, actual_provider: str = ""):
        self.status_callback("Speaking...")
        self._interrupted = False
        try:
            # Measure duration
            sound = pygame.mixer.Sound(path)
            duration = sound.get_length()
            
            # ONLY snip if the audio actually came from FreeTTS
            stop_early_seconds = 2.8 if actual_provider == "freetts" else 0.0
            play_duration = max(0.1, duration - stop_early_seconds)
            
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                if self._interrupted:
                    break
                    
                elapsed = time.time() - start_time
                if elapsed >= play_duration:
                    pygame.mixer.music.stop()
                    break
                    
                if self.live_mode and self.is_listening:
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.01)
                
            pygame.mixer.music.unload()
            
            # If interrupted, immediately trigger listening if in live mode
            if self._interrupted and self.live_mode:
                self.trigger_manual_listen()
            # If finished naturally and in live mode, restart listening
            elif not self._interrupted and self.live_mode and self._audio_queue.empty():
                self.trigger_manual_listen()

        except Exception as e:
            logger.error(f"Audio Playback Error: {e}", exc_info=True)
        finally:
            self.status_callback("Ready")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
