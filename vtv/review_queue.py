"""
review_queue.py - Storage for human-in-the-loop (C3) reviews.

When a C3 run pauses at the gate, an item is added to the review queue. Each
item carries a random review identifier so that the review interface can
present items without revealing the trial number. Completed reviews are
appended to a separate file; the resume step joins the two.

    results/review_queue.jsonl   one line per paused run
    results/reviews.jsonl        one line per completed review
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone

from vtv import config

QUEUE_FILE = config.RESULTS_DIR / "review_queue.jsonl"
REVIEWS_FILE = config.RESULTS_DIR / "reviews.jsonl"


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def enqueue(task_id: str, trial: int, extraction: dict | None, machine_seconds: float) -> str:
    """Add a paused run to the queue and return its review identifier."""
    review_id = secrets.token_hex(4)
    _append(QUEUE_FILE, {
        "review_id": review_id, "task_id": task_id, "trial": trial,
        "extraction": extraction, "machine_seconds": round(machine_seconds, 3),
        "queued_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return review_id


def queue_items() -> list[dict]:
    return _read(QUEUE_FILE)


def completed_reviews() -> dict[str, dict]:
    return {row["review_id"]: row for row in _read(REVIEWS_FILE)}


def pending_items() -> list[dict]:
    done = completed_reviews()
    return [item for item in queue_items() if item["review_id"] not in done]


def record_review(review_id: str, action: str, corrected: dict | None,
                  seconds: float, session: int) -> None:
    _append(REVIEWS_FILE, {
        "review_id": review_id, "action": action, "corrected": corrected,
        "seconds": round(seconds, 2), "session": session,
        "reviewed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
