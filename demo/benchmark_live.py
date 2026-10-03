"""Measure sequential live-scoring latency over the authenticated HTTP API."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import uuid
from typing import Any

from demo.live_stream import _request, build_event

MINIMUM_API_KEY_LENGTH = 32


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return round(ordered[index], 3)


def dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_benchmark(
    *,
    api_url: str,
    api_key: str,
    sample_count: int,
    run_id: str,
) -> dict[str, Any]:
    status, _ = _request(f"{api_url}/live/status", api_key=api_key)
    first_step = max(
        int(status["reference_max_step"]),
        int(status["latest_step"] or 0),
    ) + 1
    samples: dict[str, list[float]] = {
        "feature_ms": [],
        "inference_ms": [],
        "explanation_ms": [],
        "state_update_ms": [],
        "processing_ms": [],
        "end_to_end_ms": [],
    }
    started_at = time.perf_counter()
    for index in range(sample_count):
        event = build_event(index, first_step=first_step, run_id=run_id)
        result, end_to_end_ms = _request(
            f"{api_url}/score_transaction",
            api_key=api_key,
            payload=event,
        )
        timings = result["timings"]
        for key in (
            "feature_ms",
            "inference_ms",
            "explanation_ms",
            "state_update_ms",
            "processing_ms",
        ):
            samples[key].append(float(timings[key]))
        samples["end_to_end_ms"].append(end_to_end_ms)
    elapsed_seconds = time.perf_counter() - started_at

    summary = {
        key: {
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "mean": round(statistics.fmean(values), 3),
        }
        for key, values in samples.items()
    }
    return {
        "benchmark": "sequential authenticated POST /score_transaction",
        "sample_count": sample_count,
        "run_id": run_id,
        "methodology": (
            "One HTTP request per event, no precomputed scores, one producer, "
            "strictly increasing steps, and no intentional delay. A preceding "
            "status request initializes the API/model; API startup is excluded, "
            "but there are no scoring warm-up requests. End-to-end latency is "
            "measured client-side around each complete HTTP request."
        ),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "events_per_second": round(sample_count / elapsed_seconds, 3),
        "latency_ms": summary,
        "environment": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor() or "not-reported",
            "logical_cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "numpy": dependency_version("numpy"),
            "pandas": dependency_version("pandas"),
            "xgboost": dependency_version("xgboost"),
            "fastapi": dependency_version("fastapi"),
            "model": "B",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the real sequential HTTP scoring path."
    )
    parser.add_argument("--events", type=int, default=50)
    parser.add_argument("--run-id", default=f"bench-{uuid.uuid4().hex[:12]}")
    args = parser.parse_args(argv)
    if args.events < 1:
        raise SystemExit("--events must be at least one")
    if (
        not args.run_id.strip()
        or len(args.run_id) > 80
        or any(ord(character) < 32 for character in args.run_id)
    ):
        raise SystemExit("--run-id must contain between 1 and 80 characters")
    api_key = os.environ.get("SYNDICAI_API_KEY", "")
    if len(api_key) < MINIMUM_API_KEY_LENGTH:
        raise SystemExit(
            "Set SYNDICAI_API_KEY to the same 32-character-or-longer key as the API."
        )
    api_url = os.environ.get("SYNDICAI_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        result = run_benchmark(
            api_url=api_url,
            api_key=api_key,
            sample_count=args.events,
            run_id=args.run_id,
        )
    except RuntimeError as error:
        print(f"Benchmark failed: {error}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
