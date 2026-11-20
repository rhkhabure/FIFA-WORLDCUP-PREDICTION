"""Run a small dependency-free concurrency smoke test against an HTTP URL."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def request_once(url: str, timeout: float) -> tuple[float, bool]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read()
            return (time.perf_counter() - started) * 1000, 200 <= response.status < 400
    except (OSError, urllib.error.URLError):
        return (time.perf_counter() - started) * 1000, False


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * fraction))
    return values[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/health/live")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(
            executor.map(
                lambda _: request_once(args.url, args.timeout),
                range(args.requests),
            )
        )
    durations = [duration for duration, _ in results]
    successes = sum(success for _, success in results)
    print(f"url={args.url}")
    print(f"requests={args.requests} successes={successes} failures={args.requests - successes}")
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    print(f"p50_ms={percentile(durations, 0.50):.2f}")
    print(f"p95_ms={percentile(durations, 0.95):.2f}")
    print(f"max_ms={max(durations):.2f}")
    return 0 if successes == args.requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
