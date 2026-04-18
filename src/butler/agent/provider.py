from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


class ModelProviderError(Exception):
    pass


@dataclass(frozen=True)
class OllamaProvider:
    base_url: str
    timeout_seconds: int = 60
    retry_count: int = 0
    total_timeout_seconds: int = 0
    retry_backoff_seconds: float = 0.3

    def chat(self, model: str, messages: list[dict[str, Any]], *, temperature: float = 0.2) -> str:
        url = self.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
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
                resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
                if resp.status_code >= 400:
                    raise ModelProviderError(f"Ollama error {resp.status_code}: {resp.text}")
                data = resp.json()
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

    def chat_stream(self, model: str, messages: list[dict[str, Any]], *, temperature: float = 0.2) -> Any:
        url = self.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "stream": True,
            "messages": messages,
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(url, json=payload, stream=True, timeout=(10, self.timeout_seconds))
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
