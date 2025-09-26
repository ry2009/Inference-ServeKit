"""Run a PPO-style sampling loop against the PrimeRL gRPC service."""

from __future__ import annotations

import argparse
import asyncio
import time

import grpc

from api import primerl_pb2, primerl_pb2_grpc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PrimeRL PPO summarization demo")
    parser.add_argument("--prompt", default="Summarize the benefits of global prefix caching.")
    parser.add_argument("--grammar", default="sql_v1")
    parser.add_argument("--model", default="llama3-8b-instruct")
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--prefix-cache", choices=["on", "off"], default="on")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--verifier", default=None)
    return parser


async def run_episode(args: argparse.Namespace):
    channel = grpc.aio.insecure_channel(args.target)
    client = primerl_pb2_grpc.PrimeRLStub(channel)

    start_req = primerl_pb2.StartReq(
        env_id="summary",
        model=args.model,
        prompt=args.prompt,
        pin_prefill=args.prefix_cache == "on",
    )
    start_resp = await client.StartEpisode(start_req)
    session_id = start_resp.session_id
    print(f"session={session_id} cache_hit={start_resp.cache_hit}")

    tokens = []
    accepted = []
    for _ in range(args.steps):
        req = primerl_pb2.StepReq(
            session_id=session_id,
            obs="",
            max_new_tokens=args.max_new,
            grammar_id=args.grammar,
            speculative=args.speculative,
        )

        async def request_stream():
            yield req

        call = client.Step(request_stream())
        async for resp in call:
            tokens.append(resp.token)
            accepted.append(resp.accepted)
            if resp.boundary:
                break
        if tokens and tokens[-1].endswith("</tool>"):
            break

    end_resp = await client.EndEpisode(primerl_pb2.EndReq(session_id=session_id))
    print(f"episode closed evicted={end_resp.evicted}")
    await channel.close()

    accepted_ratio = sum(1 for v in accepted if v) / len(accepted) if accepted else 1.0
    print("generated tokens:", " ".join(tokens))
    print(f"accepted mask ratio={accepted_ratio:.2f}")

    if args.verifier:
        print("verifier_url:", args.verifier)


def main():
    parser = build_parser()
    args = parser.parse_args()

    start = time.time()
    asyncio.run(run_episode(args))
    print(f"elapsed={time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
