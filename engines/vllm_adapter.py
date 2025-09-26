import time
import uuid

import httpx
import orjson


class VLLMAdapter:
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=None)

    async def prefill(self, model: str, prompt: str, grammar: str | None):
        # OpenAI-compatible vLLM endpoints do not expose a dedicated prefill API.
        # We synthesize a session identifier so the caller can track state locally.
        return {"session_id": str(uuid.uuid4()), "tokens": 0}

    async def continue_decode(
        self,
        session_id: str,
        obs: str,
        max_new: int,
        grammar: str | None,
        speculative: bool,
        prompt: str | None = None,
        model: str | None = None,
    ):
        if prompt is None:
            raise ValueError("prompt is required for vLLMAdapter.continue_decode")

        payload = {
            "model": model or "",
            "prompt": prompt,
            "max_tokens": max_new,
            "stream": True,
            "temperature": 0.0,
        }
        async with self.client.stream("POST", "/v1/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = orjson.loads(data)
                choice = chunk["choices"][0]
                text = choice.get("text") or ""
                if not text:
                    continue
                yield {
                    "token": text,
                    "t_us": int(time.time() * 1e6),
                    "kv_bytes": 0,
                    "boundary": choice.get("finish_reason") is not None,
                }
