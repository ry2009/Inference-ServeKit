import asyncio

import pytest

from speculation.tool_boundary_spec import ToolBoundarySpec


class DummyEngine:
    def __init__(self, tokens):
        self._tokens = tokens

    async def continue_decode(self, **_):
        for token in self._tokens:
            yield token


@pytest.mark.asyncio
async def test_speculation_masks_on_divergence():
    draft_tokens = [{"token": "a", "boundary": False}, {"token": "b", "boundary": True}]
    target_tokens = [{"token": "a"}, {"token": "c"}]
    spec = ToolBoundarySpec(DummyEngine(draft_tokens), DummyEngine(target_tokens), boundary_token="")
    tokens, mask = await spec.generate("sid", "obs", 4, "grammar")
    assert tokens[0]["token"] == "a"
    assert mask == [True, False]
