"""
run_one.py - Execute a single workflow run.

Usage:
    python run_one.py --doc DOC-L01 --condition C1 --trial 1
    python run_one.py --doc TKT-H01 --condition C2 --trial 1     (ticket triage)

Automated conditions (C0, C1, C2) run to completion and write a trace to
results/traces/<doc>_<condition>_t<trial>.json. A C3 run stops at the human
checkpoint and is added to the review queue; it is completed by
review_app.py followed by resume_reviews.py.

Set VTV_MOCK=1 to run offline against the ground truth at zero cost.
"""

from __future__ import annotations

import argparse

from vtv import config, review_queue, runner
from vtv.traces import ground_truth, is_ticket, ticket_truth, write_trace


def report(final: dict, truth: dict, path) -> None:
    if is_ticket(final["task_id"]):
        report_ticket(final, truth, path)
        return
    s = final["summary"]
    print(f"Run: {final['task_id']} | condition {final['condition']} | trial {final['trial']} | "
          f"model {'mock' if config.MOCK_MODE else config.MODEL_ID}")
    for h in final.get("gate_history", []):
        state = "passed" if h["passed"] else f"{len(h['issues'])} issue(s)"
        print(f"Gate (revision {h['revision']}): {state}")
        for issue in h["issues"]:
            print(f"    - {issue}")
    print(f"Decision: {final.get('decision')}   (correct answer: {truth['decision']})")
    print(f"Supplier matched: {(final.get('final_record') or {}).get('supplier_id')}   "
          f"(correct: {truth['final']['supplier_id']})")
    print(f"Total extracted: {(final.get('extraction') or {}).get('total_amount')}   "
          f"(correct: {truth['extraction']['total_amount']})")
    print(f"Model calls: {s['model_calls']} | tokens in/out/thinking: "
          f"{s['input_tokens']}/{s['output_tokens']}/{s['thinking_tokens']} | "
          f"cost: ${s['token_cost_usd']:.6f} | time: {s['wall_clock_s']} s")
    print(f"Revisions: {s['revisions']} | flagged: {s['flagged']} | API retries: {s['api_retries']} "
          f"| parse failures: {s['parse_failures']}")
    print(f"Trace saved to {path.relative_to(config.ROOT)}")


def report_ticket(final: dict, truth: dict, path) -> None:
    s, labels = final["summary"], truth["labels"]
    record = final.get("final_record") or {}
    print(f"Run: {final['task_id']} | condition {final['condition']} | trial {final['trial']} | "
          f"model {'mock' if config.MOCK_MODE else config.MODEL_ID}")
    for h in final.get("gate_history", []):
        state = "passed" if h["passed"] else f"{len(h['issues'])} issue(s)"
        print(f"Gate (revision {h['revision']}): {state}")
        for issue in h["issues"]:
            print(f"    - {issue}")
    for key in ("category", "affected_service", "assignment_group"):
        print(f"{key}: {record.get(key)}   (correct: {labels[key]})")
    print(f"priority: {final.get('decision')}   (correct: {labels['priority']})")
    print(f"Model calls: {s['model_calls']} | cost: ${s['token_cost_usd']:.6f} | "
          f"time: {s['wall_clock_s']} s | revisions: {s['revisions']} | flagged: {s['flagged']}")
    print(f"Trace saved to {path.relative_to(config.ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a single workflow run.")
    parser.add_argument("--doc", required=True, help="document or ticket id, e.g. DOC-L01, TKT-H01")
    parser.add_argument("--condition", default="C0", choices=["C0", "C1", "C2", "C3"])
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    truth = ticket_truth(args.doc) if is_ticket(args.doc) else ground_truth(args.doc)

    if args.condition == "C3":
        payload, seconds = runner.start_human(args.doc, args.trial)
        review_id = review_queue.enqueue(args.doc, args.trial, payload["extraction"], seconds)
        print(f"Run paused for human review (review id {review_id}). "
              f"Complete it in review_app.py, then run resume_reviews.py.")
        return

    final = runner.run(args.doc, args.condition, args.trial)
    report(final, truth, write_trace(final))


if __name__ == "__main__":
    main()
