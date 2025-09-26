import httpx
import orjson


class SGLangAdapter:
    """Adapter for SGLang's stateful decode HTTP interface."""

    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=None)

    async def prefill(self, model: str, prompt: str, grammar: str | None):
        resp = await self.client.post(
            "/prefill",
            json={"model": model, "prompt": prompt, "grammar": grammar},
        )
        resp.raise_for_status()
        return resp.json()

    async def continue_decode(
        self,
        session_id: str,
        obs: str,
        max_new: int,
        grammar: str | None,
        speculative: bool,
    ):
        payload = {
            "session_id": session_id,
            "obs": obs,
            "max_new_tokens": max_new,
            "grammar": grammar,
            "speculative": speculative,
        }
        async with self.client.stream("POST", "/decode", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                yield orjson.loads(line)
