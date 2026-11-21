"""Local vision LLM via Ollama. Everything stays on this machine."""

from __future__ import annotations

import base64
from typing import Any

from sphura.infrastructure.ai.http import AIProviderError, get_json, post_json
from sphura.ports.ai import LLMRequest, LLMResponse, ProviderInfo


class OllamaProvider:
    max_concurrency = 1  # one model on one GPU: parallel requests just queue

    def __init__(self, host: str, model: str, num_ctx: int = 16384) -> None:
        self._host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(name=f"ollama/{self.model}", runs_locally=True)

    def problem(self) -> str | None:
        """None if ready; otherwise what's wrong and how to fix it."""
        try:
            tags = get_json(f"{self._host}/api/tags")
        except Exception:
            return f"Ollama isn't running at {self._host}. Start the Ollama app, then retry."
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return None
        have = ", ".join(sorted(names)) or "none"
        return (
            f"model '{self.model}' isn't installed (installed: {have}). "
            f"Run: ollama pull {self.model}"
        )

    def available(self) -> bool:
        return self.problem() is None

    def complete(self, request: LLMRequest) -> LLMResponse:
        # Ollama has no interleaved text/image parts: list frame labels in the prompt instead
        labels = "\n".join(f"Image {i + 1}: {img.label}" for i, img in enumerate(request.images))
        msg: dict[str, Any] = {
            "role": "user",
            "content": f"{labels}\n\n{request.prompt}" if labels else request.prompt,
            "images": [base64.b64encode(img.jpeg).decode() for img in request.images],
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system}, msg],
            "stream": False,
            # Thinking models (e.g. gemma4) otherwise spend the whole budget reasoning and
            # return an empty answer; some only drop to a minimal level, so leave headroom
            "think": False,
            "options": {
                "temperature": 0.0,  # same input -> same verdict on re-runs
                "num_predict": request.max_tokens + 1024,
                "num_ctx": self.num_ctx,  # Ollama defaults to 4k; frames need more
            },
        }
        if request.json_schema:
            body["format"] = request.json_schema
        url = f"{self._host}/api/chat"
        try:
            resp = post_json(url, body, timeout_s=600, retries=1)
        except AIProviderError as e:
            if e.status != 400 or "think" not in str(e).lower():
                raise
            body.pop("think")  # model without thinking support rejected the flag
            resp = post_json(url, body, timeout_s=600, retries=1)
        message = resp.get("message") or {}
        text = str(message.get("content") or "").strip()
        if not text:
            thinking = len(str(message.get("thinking") or ""))
            raise AIProviderError(
                f"{self.model} returned an empty answer "
                f"(done_reason={resp.get('done_reason')}, thinking={thinking} chars)"
            )
        return LLMResponse(
            text, int(resp.get("prompt_eval_count", 0)), int(resp.get("eval_count", 0))
        )
