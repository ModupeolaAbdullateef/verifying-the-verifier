"""
faults.py - Controlled fault injection (Experiment 2).

In Experiment 2 the verification gate does not receive the extractor's own
first output. It receives the ground-truth record into which exactly one
known fault has been seeded, or, for a control share of runs, the unmodified
ground-truth record. Every condition receives the same seeded record for a
given document and trial, so the conditions are compared on identical inputs
and the detection rate of each verification mechanism can be measured
directly against a known answer.

The seeded records are generated in advance by generate_faults.py and stored
in data/faults/faults.json; this module only loads them at run time.
"""

from __future__ import annotations

import json
from functools import lru_cache

from vtv import config

FAULT_TYPES = ["omission", "substitution", "calculation", "hallucination", "format"]
CATEGORIES = FAULT_TYPES + ["none"]           # "none" = unmodified control record
FAULT_TRIALS = list(range(201, 225))          # 24 trials: 4 per category per document
FAULT_FILE = config.DATA_DIR / "faults" / "faults.json"


@lru_cache(maxsize=1)
def _plan() -> dict[tuple[str, int], dict]:
    if not FAULT_FILE.exists():
        return {}
    rows = json.loads(FAULT_FILE.read_text(encoding="utf-8"))
    return {(row["doc_id"], row["trial"]): row for row in rows}


def is_fault_trial(trial: int) -> bool:
    return trial in FAULT_TRIALS


def fault_for(doc_id: str, trial: int) -> dict | None:
    """The seeded fault for a document and trial, or None outside Experiment 2."""
    if not is_fault_trial(trial):
        return None
    try:
        return _plan()[(doc_id, trial)]
    except KeyError:
        raise RuntimeError(f"No seeded fault for {doc_id} trial {trial}: "
                           f"run generate_faults.py first") from None


def experiment_for(trial: int) -> str:
    return "fault" if is_fault_trial(trial) else "natural"
