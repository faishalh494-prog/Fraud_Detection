"""Send a deterministic PaySim-style event sequence through the live API."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Iterator

DEMO_TRANSACTIONS = (
    # 1-4: Canonical evaluated four-event scenario (validates new history -> alert progression)
    {"type": "PAYMENT", "amount": 5.0, "nameOrig": "DEMO-SENDER-001", "nameDest": "DEMO-RECEIVER-001"},
    {"type": "PAYMENT", "amount": 8.0, "nameOrig": "DEMO-SENDER-001", "nameDest": "DEMO-RECEIVER-001"},
    {"type": "CASH_OUT", "amount": 5_000.0, "nameOrig": "DEMO-SENDER-001", "nameDest": "DEMO-RECEIVER-001"},
    {"type": "CASH_OUT", "amount": 1_000_000.0, "nameOrig": "DEMO-SENDER-001", "nameDest": "DEMO-RECEIVER-001"},
    # 5-8: Legitimate commercial, peer-to-peer, and merchant operational flows
    {"type": "PAYMENT", "amount": 42.50, "nameOrig": "C104928190", "nameDest": "M882194012"},
    {"type": "TRANSFER", "amount": 1_250.0, "nameOrig": "C391024819", "nameDest": "C401928371"},
    {"type": "DEBIT", "amount": 180.0, "nameOrig": "C581920381", "nameDest": "C991823741"},
    {"type": "CASH_IN", "amount": 3_500.0, "nameOrig": "C771829301", "nameDest": "C882710394"},
    # 9-12: High-velocity suspected mule drain and large anomalous transfer
    {"type": "TRANSFER", "amount": 75_000.0, "nameOrig": "C661928301", "nameDest": "C228192041"},
    {"type": "CASH_OUT", "amount": 75_000.0, "nameOrig": "C228192041", "nameDest": "C449102831"},
    {"type": "TRANSFER", "amount": 450_000.0, "nameOrig": "C991028341", "nameDest": "C338192019"},
    {"type": "PAYMENT", "amount": 15.20, "nameOrig": "C112938401", "nameDest": "M192837461"},
)
SENDER = "DEMO-SENDER-001"
RECEIVER = "DEMO-RECEIVER-001"
MINIMUM_API_KEY_LENGTH = 32


def build_event(
    sequence: int,
    *,
    first_step: int,
    run_id: str,
) -> dict[str, Any]:
    """Build the next label-free event; sequence numbers map to unique steps."""
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    spec = DEMO_TRANSACTIONS[sequence % len(DEMO_TRANSACTIONS)]
    return {
        "step": first_step + sequence,
        "type": spec["type"],
        "amount": spec["amount"],
        "nameOrig": spec["nameOrig"],
        "nameDest": spec["nameDest"],
        "event_id": f"{run_id}-{sequence + 1:06d}",
        "model": "B",
        "operating_point": "max_f1",
    }


def _request(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="GET" if payload is None else "POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(detail).get("detail", detail)
        except (json.JSONDecodeError, AttributeError):
            message = detail
        raise RuntimeError(f"Live API returned HTTP {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot reach the live API at {url}") from error
    return result, (time.perf_counter() - started_at) * 1000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously submit deterministic PaySim-style events to "
            "POST /score_transaction."
        )
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Minimum interval in seconds between event submissions (default: 1.0).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after this many events; omit to stream until interrupted.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable ID prefix; use a fresh value on each run to avoid duplicates.",
    )
    return parser


def stream_events(
    *,
    api_url: str,
    api_key: str,
    first_step: int,
    run_id: str,
    interval: float,
    max_events: int | None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any], float]]:
    """Yield each real API result in order; no score is computed in this process."""
    sequence = 0
    while max_events is None or sequence < max_events:
        event = build_event(sequence, first_step=first_step, run_id=run_id)
        result, end_to_end_ms = _request(
            f"{api_url.rstrip('/')}/score_transaction",
            api_key=api_key,
            payload=event,
        )
        yield event, result, end_to_end_ms
        sequence += 1
        if max_events is None or sequence < max_events:
            time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        raise SystemExit("--interval must be a finite number greater than zero")
    if args.max_events is not None and args.max_events < 1:
        raise SystemExit("--max-events must be at least one")
    api_key = os.environ.get("SYNDICAI_API_KEY", "")
    if len(api_key) < MINIMUM_API_KEY_LENGTH:
        raise SystemExit(
            "Set SYNDICAI_API_KEY to the same 32-character-or-longer key as the API."
        )
    api_url = os.environ.get("SYNDICAI_API_URL", "http://127.0.0.1:8000").rstrip("/")
    run_id = args.run_id or f"demo-{uuid.uuid4().hex[:12]}"
    if (
        not run_id.strip()
        or len(run_id) > 80
        or any(ord(character) < 32 for character in run_id)
    ):
        raise SystemExit("--run-id must contain between 1 and 80 characters")

    try:
        status, _ = _request(f"{api_url}/live/status", api_key=api_key)
        first_step = max(
            int(status["reference_max_step"]),
            int(status["latest_step"] or 0),
        ) + 1
        print(
            f"Live stream connected · first step {first_step} · "
            f"interval {args.interval:g}s · run {run_id}"
        )
        for sequence, (event, result, end_to_end_ms) in enumerate(
            stream_events(
                api_url=api_url,
                api_key=api_key,
                first_step=first_step,
                run_id=run_id,
                interval=args.interval,
                max_events=args.max_events,
            )
        ):
            risk = result["risk"]
            print(
                json.dumps(
                    {
                        "event": sequence + 1,
                        "step": event["step"],
                        "type": event["type"],
                        "risk_score": risk["score"],
                        "review_priority": risk["review_priority"],
                        "flagged_for_review": risk["flagged_for_review"],
                        "history_status": result["evidence_strength"]["status"],
                        "processing_ms": result["timings"]["processing_ms"],
                        "end_to_end_ms": round(end_to_end_ms, 3),
                    },
                    separators=(",", ":"),
                )
            )
    except KeyboardInterrupt:
        print("\nLive stream stopped.")
    except RuntimeError as error:
        print(f"Live stream failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
