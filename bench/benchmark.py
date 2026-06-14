import asyncio
import httpx
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

RESULTS_DIR = Path(__file__).parent / "results"

async def send_request(client: httpx.AsyncClient, url: str, prompt: str) -> dict:
    payload = {
        "prompt": prompt,
        "max_tokens": 128,
        "stream": True,
    }

    ttft = None
    start = time.perf_counter()

    async with client.stream("POST", url, json=payload, timeout=60.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            if ttft is None:
                # in ms
                ttft = (time.perf_counter() - start)*1000

    total = (time.perf_counter() - start)*1000
    return {"ttft_ms": ttft, "total_ms": total}

async def run_benchmark(url: str, prompt: str, concurrency: int) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, url, prompt) for _ in range(concurrency)]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=["tark", "llamacpp"], required=True)
    parser.add_argument("--url", default="http://localhost:8080/v1/completions/stream")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--reason", default="unspecified")
    parser.add_argument("--prompt", default="The history of the Roman Empire is")

    args = parser.parse_args()

    print(f"Running benchmark: {args.concurrency} concurrent requests -> {args.url}")
    results = asyncio.run(run_benchmark(args.url, args.prompt, args.concurrency))

    ok = [r for r in results if isinstance(r, dict)]
    failed = len(results) - len(ok)

    if not ok:
        print("All requests failed.")
        return
    
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    totals = [r["total_ms"] for r in ok]

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server": args.server,
        "reason": args.reason,
        "concurrency": args.concurrency,
        "requests_ok": len(ok),
        "requests_failed": failed,
        "mean_ttft_ms": round(mean(ttfts), 2) if ttfts else None,
        "median_ttft_ms": round(median(ttfts), 2) if ttfts else None,
        "mean_total_ms": round(mean(totals), 2),
        "median_total_ms": round(median(totals), 2),
    }

    print(json.dumps(entry, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.server}.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(entry) + '\n')
    print(f"Appended to {out}")

if __name__ == "__main__":
    main()