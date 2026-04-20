from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests


class ModelProviderError(Exception):
    pass


@dataclass
class OllamaProvider:
    base_url: str
    model: str
    timeout_seconds: int = 60
    retry_count: int = 0
    total_timeout_seconds: int = 0
    retry_backoff_seconds: float = 0.3
    _session: requests.Session = field(init=False, repr=False, compare=False)
    _chat_url: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._chat_url = self.base_url.rstrip("/") + "/api/chat"

    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> str:
        active_model = model or self.model
        payload = {
            "model": active_model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": temperature},
        }

        max_attempts = 1 + max(0, int(self.retry_count))
        start = time.time()
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            if self.total_timeout_seconds and (time.time() - start) > float(self.total_timeout_seconds):
                break
            try:
                resp = self._session.post(self._chat_url, json=payload, timeout=self.timeout_seconds)
                try:
                    if resp.status_code >= 400:
                        raise ModelProviderError(f"Ollama error {resp.status_code}: {resp.text}")
                    data = resp.json()
                finally:
                    resp.close()
                msg = data.get("message") or {}
                content = msg.get("content")
                if not isinstance(content, str):
                    raise ModelProviderError(f"Unexpected Ollama response: {json.dumps(data)[:500]}")
                return content.strip()
            except requests.RequestException as e:
                last_error = e
            except ModelProviderError as e:
                # Do not retry semantic/provider errors (bad model tag, HTTP 4xx, malformed JSON).
                raise

            if attempt < max_attempts:
                time.sleep(self.retry_backoff_seconds)

        elapsed = time.time() - start
        budget = self.total_timeout_seconds
        attempts_used = min(max_attempts, attempt)
        base = f"Model call failed after {attempts_used} attempt(s) in {elapsed:.1f}s"
        if budget:
            base += f" (budget {budget}s)"
        if last_error is not None:
            raise ModelProviderError(f"{base}: {last_error}") from last_error
        raise ModelProviderError(base)

    def chat_stream(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> Any:
        active_model = model or self.model
        payload = {
            "model": active_model,
            "stream": True,
            "messages": messages,
            "options": {"temperature": temperature},
        }

        resp = None
        try:
            resp = self._session.post(self._chat_url, json=payload, stream=True, timeout=(10, self.timeout_seconds))
            if resp.status_code >= 400:
                raise ModelProviderError(f"Ollama error {resp.status_code}: {resp.text}")
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
        except requests.RequestException as e:
            raise ModelProviderError(f"Model stream failed: {e}") from e
        finally:
            if resp is not None:
                resp.close()


@dataclass
class GeminiProvider:
    api_key: str
    model: str = "gemini-2.5-flash"
    timeout_seconds: int = 60
    retry_count: int = 0
    total_timeout_seconds: int = 0
    retry_backoff_seconds: float = 0.3
    _session: requests.Session = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._session = requests.Session()

    def _format_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        contents = []
        system_instruction = None

        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if msg["role"] == "system":
                system_instruction = {"parts": [{"text": content}]}
            else:
                contents.append({"role": role, "parts": [{"text": content}]})

        return contents, system_instruction

    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> str:
        active_model = model or self.model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent?key={self.api_key}"
        
        contents, system_instruction = self._format_messages(messages)
        
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature}
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
            
        max_attempts = 1 + max(0, int(self.retry_count))
        start = time.time()
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            if self.total_timeout_seconds and (time.time() - start) > float(self.total_timeout_seconds):
                break
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout_seconds)
                try:
                    if resp.status_code >= 400:
                        raise ModelProviderError(f"Gemini error {resp.status_code}: {resp.text}")
                    data = resp.json()
                finally:
                    resp.close()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except requests.RequestException as e:
                last_error = e
            except ModelProviderError as e:
                raise
            except KeyError as e:
                raise ModelProviderError(f"Unexpected Gemini response structure: {e}")

            if attempt < max_attempts:
                time.sleep(self.retry_backoff_seconds)

        elapsed = time.time() - start
        budget = self.total_timeout_seconds
        attempts_used = min(max_attempts, attempt)
        base = f"Model call failed after {attempts_used} attempt(s) in {elapsed:.1f}s"
        if budget:
            base += f" (budget {budget}s)"
        if last_error:
            raise ModelProviderError(f"{base}: {last_error}") from last_error
        raise ModelProviderError(base)

    def chat_stream(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> Any:
        active_model = model or self.model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:streamGenerateContent?alt=sse&key={self.api_key}"
        
        contents, system_instruction = self._format_messages(messages)
        
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature}
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        resp = None
        try:
            resp = self._session.post(url, json=payload, stream=True, timeout=(10, self.timeout_seconds))
            if resp.status_code >= 400:
                raise ModelProviderError(f"Gemini error {resp.status_code}: {resp.text}")
            for line in resp.iter_lines():
                if line.startswith(b"data: "):
                    line_data = line[6:]
                    if line_data.strip() == b"[DONE]":
                        continue
                    chunk = json.loads(line_data)
                    parts = chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    if parts:
                        token = parts[0].get("text", "")
                        if token:
                            yield token
        except (requests.RequestException, json.JSONDecodeError) as e:
            raise ModelProviderError(f"Model stream failed: {e}") from e
        finally:
            if resp is not None:
                resp.close()
