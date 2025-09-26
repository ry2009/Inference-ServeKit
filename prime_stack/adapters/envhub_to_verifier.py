from __future__ import annotations

from typing import Dict


def build_trace(episode: Dict, rewards: Dict, policy_meta: Dict) -> Dict:
    return {
        "episode_id": episode.get("episode_id"),
        "model": episode.get("model"),
        "prompt_fp": episode.get("prompt_fp"),
        "tokens": episode.get("tokens"),
        "accepted_mask": episode.get("accepted_mask", []),
        "tools": episode.get("tools", []),
        "metrics": rewards,
        "policy_meta": policy_meta,
        "meta": episode.get("meta", {}),
    }
