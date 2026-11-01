"""
traces.py - Saving run traces and loading ground truth.

A trace records everything that happened in one run (intermediate outputs,
gate evaluations and per-call measurements). Traces are the raw data from
which the scoring step derives every reported result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from vtv import config

TRACE_KEYS = ["task_id", "condition", "trial", "seed", "experiment", "fault", "initial_extraction", "extraction", "gate_history",
              "revisions", "flagged", "candidates", "final_record", "purchase_order",
              "decision", "justification", "calls", "summary"]


@lru_cache(maxsize=1)
def _document_truth() -> dict[str, dict]:
    rows = json.loads((config.DATA_DIR / "ground_truth" / "documents.json").read_text(encoding="utf-8"))
    return {row["doc_id"]: row for row in rows}


def ground_truth(doc_id: str) -> dict:
    return _document_truth()[doc_id]


def document_text(doc_id: str) -> str:
    return (config.DATA_DIR / "documents" / f"{doc_id}.txt").read_text(encoding="utf-8")


def document_images(doc_id: str) -> list[str]:
    """Paths of the scanned page images for a document, in page order."""
    return [str(p) for p in sorted((config.DATA_DIR / "images").glob(f"{doc_id}_p*.jpg"))]


def trace_path(task_id: str, condition: str, trial: int) -> Path:
    return config.RESULTS_DIR / "traces" / f"{task_id}_{condition}_t{trial}.json"


def write_trace(final: dict) -> Path:
    trace = {key: final.get(key) for key in TRACE_KEYS}
    trace["model_id"] = "mock" if config.MOCK_MODE else config.MODEL_ID
    trace["temperature"] = config.TEMPERATURE
    trace["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = trace_path(final["task_id"], final["condition"], final["trial"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
