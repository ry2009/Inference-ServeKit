from __future__ import annotations

from typing import Any


class GRPOSampler:
    """Sample K completions from the PrimeRL bridge for GRPO."""

    def __init__(self, client: Any, k: int, grammar: str | None, speculative: bool = True):
        self.client = client
        self.k = k
        self.grammar = grammar
        self.speculative = speculative

    async def sample_group(self, prompt: str, max_new: int, model: str):
        resp = await self.client.start_episode(
            env_id="grpo", model=model, prompt=prompt, pin_prefill=True
        )
        session_id = resp.session_id
        samples = []
        for _ in range(self.k):
            responses = await self.client.step(
                {
                    "session_id": session_id,
                    "obs": "",
                    "max_new_tokens": max_new,
                    "grammar_id": self.grammar or "",
                    "speculative": self.speculative,
                }
            )
            tokens = [
                {
                    "token": r.token,
                    "accepted": r.accepted,
                    "boundary": r.boundary,
                    "kv_bytes": r.kv_bytes,
                }
                for r in responses
            ]
            samples.append({"tokens": tokens})
        await self.client.end_episode(session_id)
        return samples
