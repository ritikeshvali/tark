import json
import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

def load_latest(server: str) -> dict[int, dict]:
    path = RESULTS_DIR / f"{server}.jsonl"
    if not path.exists():
        return {}
    
    latest = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            c = entry["concurrency"]
            latest[c] = entry
    return latest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, nargs="+", help="filter to specific concurrency levels")
    args = parser.parse_args()

    tark = load_latest("tark")
    llamacpp = load_latest("llamacpp")

    all_concurrency = sorted(set(tark.keys()) | set(llamacpp.keys()))
    if args.concurrency:
        all_concurrency = [c for c in all_concurrency if c in args.concurrency]

    if not all_concurrency:
        print("No results found.")
        return

    header = f"{'conc': >6} {'metric': <20} {'tark': >12} {'llamacpp': >12}"
    print(header)
    print("-"*len(header))

    for c in all_concurrency:
        t = tark.get(c)
        l = llamacpp.get(c)

        def fmt(entry, key):
            if entry is None:
                return "n/a"
            val = entry.get(key)
            return f"{val:.2f}" if val is not None else "n/a"
        
        metrics = [
            ("mean_ttft_ms", "mean TTFT (ms)"),
            ("median_ttft_ms", "median TTFT (ms)"),
            ("mean_total_ms", "mean total (ms)"),
            ("median_total_ms", "median total (ms)"),
        ]

        for key, label in metrics:
            print(f"{c:>6} {label:<20} {fmt(t, key):>12} {fmt(l, key):>12}")

        t_reason = t["reason"] if t else "n/a"
        l_reason = l["reason"] if l else "n/a"
        print(f"{'':>6} {'reason':<20} {t_reason:>12} {l_reason:>12}")
        print()

if __name__ == "__main__":
    main()