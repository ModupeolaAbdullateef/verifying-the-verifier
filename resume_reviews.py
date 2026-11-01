"""
resume_reviews.py - Complete C3 runs whose human review has been recorded.

For every reviewed item in the queue that does not yet have a trace, the
suspended workflow is resumed from the checkpoint with the reviewer's
response, the normaliser and decision agents run, and the trace is saved.

Usage:
    python resume_reviews.py
"""

from __future__ import annotations

from vtv import review_queue, runner
from vtv.traces import trace_path, write_trace


def main() -> None:
    reviews = review_queue.completed_reviews()
    resumed = waiting = skipped = 0
    for item in review_queue.queue_items():
        if trace_path(item["task_id"], "C3", item["trial"]).exists():
            skipped += 1
            continue
        review = reviews.get(item["review_id"])
        if review is None:
            waiting += 1
            continue
        final = runner.resume_human(
            item["task_id"], item["trial"],
            {"action": review["action"], "corrected": review["corrected"],
             "seconds": review["seconds"]},
            item["machine_seconds"],
        )
        write_trace(final)
        resumed += 1
        print(f"Completed {item['task_id']} trial {item['trial']}: "
              f"{review['action']} -> decision {final.get('decision')}")
    print(f"Resumed {resumed} run(s); {waiting} still awaiting review; {skipped} already complete.")


if __name__ == "__main__":
    main()
