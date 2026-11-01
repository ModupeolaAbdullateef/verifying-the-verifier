"""
check_review_queue.py - Detect and remove stale human-review (C3) items.

A C3 item is stale if it was created from an earlier version of the dataset,
for example during testing before the dataset was regenerated. Staleness is
detected objectively: the document text stored in the item's suspended
workflow state is compared with the current document file.

Removing a stale item deletes its queue entry, any recorded review, any
trace and its checkpoint thread, so that run_batch.py re-creates the item
from the current dataset on its next execution.

Usage:
    python check_review_queue.py            # report only
    python check_review_queue.py --remove   # remove stale items
"""

from __future__ import annotations

import argparse
import json
import sqlite3

from vtv import config, review_queue
from vtv.traces import document_text, trace_path
from vtv.workflow import _human_graph, thread_id


def stored_text(task_id: str, trial: int) -> str | None:
    state = _human_graph().get_state({"configurable": {"thread_id": thread_id(task_id, trial)}})
    return (state.values or {}).get("document_text")


def rewrite_without(path, review_ids: set[str]) -> None:
    if not path.exists():
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and json.loads(line)["review_id"] not in review_ids]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and remove stale C3 review items.")
    parser.add_argument("--remove", action="store_true", help="remove the stale items found")
    args = parser.parse_args()

    items = review_queue.queue_items()
    stale = [i for i in items
             if stored_text(i["task_id"], i["trial"]) != document_text(i["task_id"])]
    print(f"{len(items)} queued item(s); {len(stale)} stale.")
    for item in stale:
        print(f"  stale: {item['task_id']} trial {item['trial']} "
              f"(review id {item['review_id']}, queued {item['queued_utc']})")
    if not stale or not args.remove:
        return

    ids = {i["review_id"] for i in stale}
    rewrite_without(review_queue.QUEUE_FILE, ids)
    rewrite_without(review_queue.REVIEWS_FILE, ids)
    connection = sqlite3.connect(config.RESULTS_DIR / "checkpoints.sqlite")
    for item in stale:
        trace = trace_path(item["task_id"], "C3", item["trial"])
        if trace.exists():
            trace.unlink()
        for table in ("checkpoints", "writes"):
            connection.execute(f"DELETE FROM {table} WHERE thread_id = ?",
                               (thread_id(item["task_id"], item["trial"]),))
    connection.commit()
    connection.close()
    print(f"Removed {len(stale)} stale item(s). Run run_batch.py again to re-create them.")


if __name__ == "__main__":
    main()
