"""Claude via the Anthropic Messages API. Cloud: frames + transcript excerpts leave the machine."""

from __future__ import annotations

import base64
import json
from typing import Any

from sphura.infrastructure.ai.http import AIProviderError, post_json
from sphura.ports.ai import LLMRequest, LLMResponse, ProviderInfo

_URL = "https://api.anthropic.com/v1/messages"
_TOOL = "submit_result"


class AnthropicProvider:
    max_concurrency = 4

    def __init__(self, api_key: str, model: str) -> None:
        self._key = api_key
        self.model = model

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(name=f"anthropic/{self.model}", runs_locally=False)

    def complete(self, request: LLMRequest) -> LLMResponse:
        content: list[dict[str, Any]] = []
        for img in request.images:
            if img.label:
                content.append({"type": "text", "text": img.label})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(img.jpeg).decode(),
                    },
                }
            )
        content.append({"type": "text", "text": request.prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": content}],
        }
        if request.json_schema:
            # Forced tool call = reliably structured output
            body["tools"] = [
                {
                    "name": _TOOL,
                    "description": "Submit the result",
                    "input_schema": request.json_schema,
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": _TOOL}

        resp = post_json(
            _URL,
            body,
            headers={"x-api-key": self._key, "anthropic-version": "2023-06-01"},
        )
        usage = resp.get("usage") or {}
        blocks = resp.get("content") or []
        if request.json_schema:
            tool = next((b for b in blocks if b.get("type") == "tool_use"), None)
            if tool is None:
                raise AIProviderError(f"model returned no structured result: {resp}")
            text = json.dumps(tool.get("input", {}))
        else:
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return LLMResponse(
            text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        )
