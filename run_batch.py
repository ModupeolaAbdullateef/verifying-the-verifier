"""
run_batch.py - Execute an experimental plan in batch.

Plans:
    pilot         Experiment 1 pilot: 6 documents (2 per complexity level) x C0, C1, C2
                  x 5 trials (101-105) = 90 runs. Results in results_pilot/.
    pilot_faults  Experiment 2 pilot: the same 6 documents x C0, C1, C2 x 6 trials
                  (201-206) = 108 runs. Results in results_pilot/.
    main          Experiment 1 (natural errors): 21 documents x C0, C1, C2 x 30 trials
                  (1-30) = 1,890 automated runs, plus the machine phase of C3 for the
                  10 human-review documents x 6 trials (1-6) = 60 runs. Results in results/.
    faults        Experiment 2 (seeded faults): 21 documents x C0, C1, C2 x 24 trials
                  (201-224) = 1,512 automated runs, plus the machine phase of C3 for the
                  10 human-review documents x 24 trials = 240 runs. Results in results/.
    tickets       Replication (ticket triage): 10 tickets x C0, C1, C2 x 10 trials (1-10)
                  = 300 automated runs, plus the machine phase of C3 for the 5 human-review
                  tickets x 10 trials = 50 runs. Results in results/.

Experiment 1 and Experiment 2 share the results folder and are distinguished by
trial number, so that all C3 items enter a single blinded review queue.

Runs are executed in a fixed random order that interleaves documents,
conditions and trials, so that drift in API latency over time does not bias
any one condition. Completed runs (existing traces or queued reviews) are
skipped, so an interrupted batch can be restarted with the same command.
Execution stops submitting new runs once the cumulative token cost reaches
the cost cap.

Usage:
    python run_batch.py --plan pilot
    python run_batch.py --plan pilot_faults
    python run_batch.py --plan main --workers 4 --cost-cap 25
    python run_batch.py --plan faults --workers 4 --cost-cap 25
    python run_batch.py --plan tickets
"""

from __future__ import annotations

import argparse
import os
import sys

PILOT_DOCUMENTS = ["DOC-L01", "DOC-L05", "DOC-M02", "DOC-M06", "DOC-H01", "DOC-H05"]
PLANS = {
    "pilot": {
        "results": "results_pilot", "documents": PILOT_DOCUMENTS,
        "conditions": ["C0", "C1", "C2"], "trials": list(range(101, 106)),
        "human_documents": [], "human_trials": [],
    },
    "pilot_faults": {
        "results": "results_pilot", "documents": PILOT_DOCUMENTS,
        "conditions": ["C0", "C1", "C2"], "trials": list(range(201, 207)),
        "human_documents": [], "human_trials": [],
    },
    "main": {
        "results": "results", "documents": "all",
        "conditions": ["C0", "C1", "C2"], "trials": list(range(1, 31)),
        "human_documents": "hitl_subset", "human_trials": list(range(1, 7)),
    },
    "tickets": {
        "results": "results", "documents": "tickets",
        "conditions": ["C0", "C1", "C2"], "trials": list(range(1, 11)),
        "human_documents": "hitl_subset", "human_trials": list(range(1, 11)),
    },
    "faults": {
        "results": "results", "documents": "all",
        "conditions": ["C0", "C1", "C2"], "trials": list(range(201, 225)),
        "human_documents": "hitl_subset", "human_trials": list(range(201, 225)),
    },
}
ORDER_SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute an experimental plan in batch.")
    parser.add_argument("--plan", required=True, choices=sorted(PLANS))
    parser.add_argument("--workers", type=int, default=4, help="parallel runs (default 4)")
    parser.add_argument("--cost-cap", type=float, default=25.0,
                        help="stop submitting runs once token cost reaches this many USD")
    parser.add_argument("--limit", type=int, default=None,
                        help="run at most this many outstanding runs (for testing)")
    return parser.parse_args()


args = parse_args()
plan = PLANS[args.plan]
# The results folder must be chosen before the package configuration is imported.
os.environ["VTV_RESULTS"] = plan["results"]

import json                                                    # noqa: E402
import random                                                  # noqa: E402
import threading                                               # noqa: E402
import time                                                    # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from datetime import datetime, timezone                        # noqa: E402

from vtv import config, review_queue, runner                   # noqa: E402
from vtv.traces import trace_path, write_trace                 # noqa: E402

_human_lock = threading.Lock()     # C3 checkpoint database and queue file: one writer at a time
_cost_lock = threading.Lock()


def build_jobs() -> list[tuple[str, str, int]]:
    if plan["documents"] == "tickets":
        truth = json.loads((config.DATA_DIR / "ground_truth" / "tickets.json").read_text(encoding="utf-8"))
        truth = [{**row, "doc_id": row["ticket_id"]} for row in truth]
    else:
        truth = json.loads((config.DATA_DIR / "ground_truth" / "documents.json").read_text(encoding="utf-8"))
    documents = ([row["doc_id"] for row in truth] if plan["documents"] in ("all", "tickets")
                 else plan["documents"])
    human_docs = ([row["doc_id"] for row in truth if row["hitl_subset"]]
                  if plan["human_documents"] == "hitl_subset" else plan["human_documents"])
    jobs = [(doc, cond, trial) for doc in documents for cond in plan["conditions"]
            for trial in plan["trials"]]
    jobs += [(doc, "C3", trial) for doc in human_docs for trial in plan["human_trials"]]
    random.Random(ORDER_SEED).shuffle(jobs)
    return jobs


def is_done(job: tuple[str, str, int], queued: set[tuple[str, int]]) -> bool:
    doc, cond, trial = job
    if trace_path(doc, cond, trial).exists():
        return True
    return cond == "C3" and (doc, trial) in queued


def spent_so_far() -> float:
    """Token cost of all traces already in the results folder."""
    total = 0.0
    for path in (config.RESULTS_DIR / "traces").glob("*.json"):
        total += json.loads(path.read_text(encoding="utf-8"))["summary"]["token_cost_usd"]
    return total


def execute(job: tuple[str, str, int]) -> float:
    """Run one job and return its token cost."""
    doc, cond, trial = job
    if cond == "C3":
        with _human_lock:
            payload, seconds = runner.start_human(doc, trial)
            review_queue.enqueue(doc, trial, payload["extraction"], seconds)
        return 0.0   # counted when the run is completed after review
    final = runner.run(doc, cond, trial)
    write_trace(final)
    return final["summary"]["token_cost_usd"]


def log_error(job, error: Exception) -> None:
    path = config.RESULTS_DIR / "batch_errors.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"job": list(job), "error": str(error)[:500],
                                 "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")


def main() -> None:
    jobs = build_jobs()
    queued = {(i["task_id"], i["trial"]) for i in review_queue.queue_items()}
    outstanding = [j for j in jobs if not is_done(j, queued)]
    complete = len(jobs) - len(outstanding)
    if args.limit is not None:
        outstanding = outstanding[:args.limit]
    spent = spent_so_far()

    print(f"Plan '{args.plan}': {len(jobs)} runs in total, {complete} already complete, "
          f"{len(outstanding)} to run now. Results folder: {config.RESULTS_DIR.name}/")
    print(f"Model: {'mock' if config.MOCK_MODE else config.MODEL_ID} | workers: {args.workers} | "
          f"spent so far: ${spent:.4f} | cost cap: ${args.cost_cap:.2f}")
    if not outstanding:
        return

    done = failed = 0
    stopped_for_cost = False
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        queue = iter(outstanding)

        def submit_next() -> bool:
            nonlocal stopped_for_cost
            if spent >= args.cost_cap:
                stopped_for_cost = True
                return False
            job = next(queue, None)
            if job is None:
                return False
            futures[pool.submit(execute, job)] = job
            return True

        for _ in range(args.workers):
            submit_next()
        while futures:
            for future in as_completed(list(futures)):
                job = futures.pop(future)
                try:
                    cost = future.result()
                    with _cost_lock:
                        spent += cost
                    done += 1
                except Exception as error:   # record and continue with the remaining runs
                    failed += 1
                    log_error(job, error)
                    print(f"  FAILED {job}: {str(error).splitlines()[0][:120]}", file=sys.stderr)
                finished = done + failed
                if finished % 10 == 0 or finished == len(outstanding):
                    rate = (time.time() - started) / finished
                    remaining = (len(outstanding) - finished) * rate / 60
                    print(f"  {finished}/{len(outstanding)} runs | ${spent:.4f} spent | "
                          f"~{remaining:.0f} min remaining")
                submit_next()
                break

    print(f"Finished: {done} completed, {failed} failed"
          + (" (see batch_errors.jsonl)" if failed else "") + f". Total token cost: ${spent:.4f}")
    if stopped_for_cost:
        print(f"Stopped early: cost cap of ${args.cost_cap:.2f} reached. Re-run with a higher "
              f"--cost-cap to continue from where it stopped.")


if __name__ == "__main__":
    main()
