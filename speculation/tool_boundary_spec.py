from __future__ import annotations

from typing import Any, List, Tuple


class ToolBoundarySpec:
    """Draft+verify speculation helper that respects grammar boundaries."""

    def __init__(self, draft_engine: Any, target_engine: Any, boundary_token: str):
        self.draft = draft_engine
        self.target = target_engine
        self.boundary = boundary_token

    async def generate(
        self,
        session_id: str,
        obs: str,
        max_new: int,
        grammar: str,
        prompt: str | None = None,
    ) -> Tuple[List[dict], List[bool]]:
        draft_tokens: list[dict] = []
        async for token in self.draft.continue_decode(
            session_id=session_id,
            obs=obs,
            max_new=max_new,
            grammar=grammar,
            speculative=False,
            prompt=prompt,
        ):
            draft_tokens.append(token)
            if token.get("boundary") or len(draft_tokens) >= max_new:
                break

        target_tokens: list[dict] = []
        async for token in self.target.continue_decode(
            session_id=session_id,
            obs=obs,
            max_new=len(draft_tokens),
            grammar=grammar,
            speculative=False,
            prompt=prompt,
        ):
            target_tokens.append(token)
            if len(target_tokens) >= len(draft_tokens):
                break

        accepted_mask: list[bool] = []
        for idx, token in enumerate(draft_tokens):
            if idx >= len(target_tokens):
                accepted_mask.append(False)
                draft_tokens = draft_tokens[:idx]
                accepted_mask = accepted_mask[:idx]
                break
            accepted = token.get("token") == target_tokens[idx].get("token")
            accepted_mask.append(accepted)
            if not accepted:
                draft_tokens = draft_tokens[: idx + 1]
                accepted_mask = accepted_mask[: idx + 1]
                break

        if not accepted_mask:
            accepted_mask = [True] * len(draft_tokens)

        return draft_tokens, accepted_mask
