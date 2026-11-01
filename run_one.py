"""
run_one.py - Execute a single workflow run and save its trace.

Usage:
    python run_one.py --doc DOC-L01 --condition C0 --trial 1

The trace (all intermediate outputs, gate evaluations and per-call
measurements) is written to results/traces/<doc>_<condition>_t<trial>.json.
Set VTV_MOCK=1 to run offline against the ground truth at zero cost.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from vtv import config
from vtv.workflow import run_workflow

TRACE_KEYS = ["task_id", "condition", "trial", "seed", "extraction", "gate_history",
              "revisions", "flagged", "candidates", "final_record", "purchase_order",
              "decision", "justification", "calls", "summary"]


def load_ground_truth(doc_id: str) -> dict:
    rows = json.loads((config.DATA_DIR / "ground_truth" / "documents.json").read_text(encoding="utf-8"))
    return next(row for row in rows if row["doc_id"] == doc_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--doc", required=True, help="document id, e.g. DOC-L01")
    parser.add_argument("--condition", default="C0", help="C0, C1, C2 or C3")
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    text = (config.DATA_DIR / "documents" / f"{args.doc}.txt").read_text(encoding="utf-8")
    truth = load_ground_truth(args.doc)
    final = run_workflow(args.doc, args.condition, args.trial, text,
                         mock=truth if config.MOCK_MODE else None)

    trace = {key: final.get(key) for key in TRACE_KEYS}
    trace["model_id"] = "mock" if config.MOCK_MODE else config.MODEL_ID
    trace["temperature"] = config.TEMPERATURE
    trace["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out_dir = config.RESULTS_DIR / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.doc}_{args.condition}_t{args.trial}.json"
    out_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

    s = trace["summary"]
    print(f"Run: {args.doc} | condition {args.condition} | trial {args.trial} | model {trace['model_id']}")
    print(f"Decision: {trace['decision']}   (correct answer: {truth['decision']})")
    print(f"Supplier matched: {(trace['final_record'] or {}).get('supplier_id')}   "
          f"(correct: {truth['final']['supplier_id']})")
    print(f"Total extracted: {(trace['extraction'] or {}).get('total_amount')}   "
          f"(correct: {truth['extraction']['total_amount']})")
    print(f"Model calls: {s['model_calls']} | tokens in/out/thinking: "
          f"{s['input_tokens']}/{s['output_tokens']}/{s['thinking_tokens']} | "
          f"cost: ${s['token_cost_usd']:.6f} | time: {s['wall_clock_s']} s")
    print(f"Revisions: {s['revisions']} | flagged: {s['flagged']} | API retries: {s['api_retries']} "
          f"| parse failures: {s['parse_failures']}")
    print(f"Trace saved to {out_path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
