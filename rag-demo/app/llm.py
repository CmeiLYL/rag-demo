from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from app.config import settings


class LLMClient:
    """Simple Ollama HTTP client."""

    def __init__(self) -> None:
        self.base_url = settings.ollama_url.rstrip("/")
        self.model = settings.ollama_model
        self.connect_timeout = settings.ollama_connect_timeout_seconds
        self.read_timeout = settings.ollama_read_timeout_seconds
        self.session = requests.Session()
        # Ollama usually runs on local/LAN endpoints; bypass global HTTP(S)_PROXY
        # to avoid proxy-induced 502 errors.
        self.session.trust_env = False

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> Optional[str]:
        """Extract response text from Ollama native or OpenAI-compatible payload."""
        # Ollama native /api/chat format
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

        # OpenAI-compatible chat completions format
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content.strip()

        return None

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> requests.Response:
        return self.session.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=(self.connect_timeout, self.read_timeout),
        )

    def chat(self, prompt: str) -> str:
        """Send prompt to Ollama and return the model response."""
        native_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.1,
        }

        # Prefer Ollama native API first.
        try:
            response = self._post("/api/chat", native_payload)
            response.raise_for_status()
            data = response.json()
            text = self._extract_text(data)
            if text:
                return text
            raise RuntimeError("Ollama returned an unexpected response format")
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            body = (exc.response.text[:500] if exc.response is not None else "")

            # Fallback to OpenAI-compatible endpoint for gateway/proxy deployments.
            fallback_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            fallback_resp = self._post("/v1/chat/completions", fallback_payload)
            if fallback_resp.ok:
                data = fallback_resp.json()
                text = self._extract_text(data)
                if text:
                    return text
                raise RuntimeError("Ollama fallback returned an unexpected response format")

            raise RuntimeError(
                f"Ollama request failed. endpoint=/api/chat status={status_code} body={body} "
                f"fallback_status={fallback_resp.status_code} fallback_body={fallback_resp.text[:500]}"
            ) from exc
        except requests.ReadTimeout as exc:
            raise RuntimeError(
                f"Ollama response timed out at {self.base_url}. "
                f"Current read timeout is {self.read_timeout}s. "
                "You can increase OLLAMA_READ_TIMEOUT_SECONDS in .env."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
