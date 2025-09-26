from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

import httpx

from grpo.advantage import group_relative
from grpo.dataset import Experience, ExperienceBuffer
from grpo.sampler import GRPOSampler
from prime_stack.adapters import build_trace
from rl_client.grpc_client import PrimeRLGrpcClient


@dataclass
class LearnerConfig:
    verifier_url: str | None
    prompt: str
    max_new: int
    steps: int
    env: str
    k: int
    grammar: str | None
    speculative: bool
    target: str
    model: str


async def run(cfg: LearnerConfig):
    client = PrimeRLGrpcClient(cfg.target)
    sampler = GRPOSampler(
        client=client,
        k=cfg.k,
        grammar=cfg.grammar,
        speculative=cfg.speculative,
    )
    buffer = ExperienceBuffer()
    verifier = httpx.AsyncClient(timeout=30) if cfg.verifier_url else None

    try:
        for step in range(cfg.steps):
            samples = await sampler.sample_group(
                prompt=cfg.prompt, max_new=cfg.max_new, model=cfg.model
            )
            if verifier and cfg.verifier_url:
                rewards = []
                for sample in samples:
                    trace = build_trace(
                        episode={
                            "episode_id": f"{cfg.env}-{step}",
                            "model": "llama3-8b",
                            "prompt_fp": None,
                            "tokens": " ".join(tok["token"] for tok in sample["tokens"]),
                            "accepted_mask": [tok.get("accepted", True) for tok in sample["tokens"]],
                            "tools": [],
                            "meta": {},
                        },
                        rewards={"ttfb_ms": 0},
                        policy_meta={"sandbox_profile": "default", "egress_blocked": True},
                    )
                    resp = await verifier.post(f"{cfg.verifier_url}/verify", json=trace)
                    resp.raise_for_status()
                    rewards.append(resp.json()["reward"])
            else:
                rewards = [
                    sum(1 for tok in sample["tokens"] if tok.get("accepted", True)) / max(len(sample["tokens"]), 1)
                    for sample in samples
                ]

            advantages = group_relative(rewards)
            for sample, reward, adv in zip(samples, rewards, advantages):
                buffer.extend(
                    [
                        Experience(
                            prompt=cfg.prompt,
                            tokens=[tok["token"] for tok in sample["tokens"]],
                            reward=reward,
                            advantage=adv,
                            accepted_mask=[tok.get("accepted", True) for tok in sample["tokens"]],
                        )
                    ]
                )
            print(f"step={step} rewards={rewards} buffer_size={len(list(buffer))}")
    finally:
        await client.close()
        if verifier:
            await verifier.aclose()


def main():
    parser = argparse.ArgumentParser(description="PrimeRL GRPO learner")
    parser.add_argument("--verifier", default=None)
    parser.add_argument("--prompt", default="Summarize a SQL log")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--env", default="sql_qa")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--grammar", default="sql_v1")
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--model", default="llama3-8b-instruct")
    args = parser.parse_args()

    cfg = LearnerConfig(
        verifier_url=args.verifier,
        prompt=args.prompt,
        max_new=args.max_new,
        steps=args.steps,
        env=args.env,
        k=args.k,
        grammar=args.grammar,
        speculative=args.speculative,
        target=args.target,
        model=args.model,
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
