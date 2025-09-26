"""Compare benchmark results against baseline and flag regressions."""

import csv
from pathlib import Path


def read_csv(path: Path):
    with path.open() as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def compare(baseline_rows, current_rows, threshold: float = 0.05):
    regressions = []
    for base, curr in zip(baseline_rows, current_rows):
        base_tps = float(base["tokens_per_sec"])
        curr_tps = float(curr["tokens_per_sec"])
        delta = (curr_tps - base_tps) / base_tps
        if delta < -threshold:
            regressions.append((base, curr, delta))
    return regressions


def emit_summary(regressions):
    if not regressions:
        print("✅ perf stable")
        return 0
    print("::error::Performance regressions detected")
    for base, curr, delta in regressions:
        print(
            f"case ctx={curr['context_tokens']} batch={curr['batch_size']} precision={curr['precision']}"\
            f" delta={delta:.2%}"
        )
    return 1


def main(baseline: str, current: str):
    baseline_rows = read_csv(Path(baseline))
    current_rows = read_csv(Path(current))
    regressions = compare(baseline_rows, current_rows)
    exit_code = emit_summary(regressions)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    args = parser.parse_args()
    main(args.baseline, args.current)
