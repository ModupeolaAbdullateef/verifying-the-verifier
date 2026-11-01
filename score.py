"""
score.py - Score run traces against the ground truth.

Each trace is scored at three stages:
    initial   the extractor's first output, before verification (gate input)
    gated     the extraction after the verification gate (revised or corrected)
    final     the record after the normaliser, which the decision agent uses

Every field is classified as correct or as one of five error types:
    omission       null although the document contains the value
    substitution   wrong, but the value appears in the document (a misread or distractor)
    calculation    a derived or arithmetic field with a wrong value not found in the document
    hallucination  wrong, not in the document and not derivable from it (fabricated)
    format         correct in substance but not in the required format

Experiment 2 runs (seeded faults) additionally record the seeded fault type
and variant, whether the fault was visible to the C1 rules, whether the gate
detected it, and whether it was removed before reaching the normaliser.

Outputs (in the results folder):
    runs.csv     one row per run: outcomes, error counts, gate behaviour, cost and time
    fields.csv   one row per field per run and stage: truth, prediction, error type

Usage:
    python score.py                  # scores results/
    python score.py --results results_pilot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

HEADER_FIELDS = ["invoice_number", "invoice_date", "due_date", "po_number", "supplier_name",
                 "supplier_vat_number", "currency", "subtotal", "vat_amount", "total_amount",
                 "payment_terms_days", "customer_reference"]
ITEM_FIELDS = ["description", "quantity", "unit_price", "line_total"]
DATE_FIELDS = {"invoice_date", "due_date"}
MONEY_FIELDS = {"subtotal", "vat_amount", "total_amount", "unit_price", "line_total"}
NUMBER_FIELDS = MONEY_FIELDS | {"quantity", "payment_terms_days"}
CALCULATED_FIELDS = {"due_date", "subtotal", "vat_amount", "total_amount", "line_total"}
ERROR_TYPES = ["omission", "substitution", "calculation", "hallucination", "format"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


# --------------------------------------------------------------------------
# Value normalisation
# --------------------------------------------------------------------------

def squash(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def as_number(value):
    """Numeric value of a field, accepting strings such as '£1,284.50' or '1 284,50 EUR'."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"[^\d,.\-]", "", str(value))
    if re.search(r",\d{2}$", text) and "." not in text:          # continental decimal comma
        text = text.replace(",", ".")
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def as_date(value) -> date | None:
    """Parse ISO and common invoice date formats (UK day-first)."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d %B %Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    match = re.match(r"^(\d{1,2})(st|nd|rd|th)\s+(\w+)\s+(\d{4})$", text, re.IGNORECASE)
    if match and match.group(3).lower() in MONTHS:
        return date(int(match.group(4)), MONTHS.index(match.group(3).lower()) + 1, int(match.group(1)))
    return None


def date_renderings(d: date) -> list[str]:
    month = MONTHS[d.month - 1].capitalize()
    return [d.isoformat(), d.strftime("%d/%m/%Y"), f"{d.day} {month} {d.year}",
            f"{d.day:02d}-{month[:3]}-{d.strftime('%y')}"]


def number_renderings(x: float) -> list[str]:
    plain = f"{x:.2f}"
    whole, pence = plain.split(".")
    return [plain, f"{x:,.2f}", f"{int(float(whole)):,}".replace(",", " ") + f",{pence}"]


def appears_in(value, field: str, document: str) -> bool:
    """Whether a (wrong) predicted value is present in the source document."""
    source = squash(document)
    if field in DATE_FIELDS:
        d = as_date(value)
        return d is not None and any(squash(r) in source for r in date_renderings(d))
    if field in NUMBER_FIELDS:
        x = as_number(value)
        if x is None:
            return False
        if field in ("quantity", "payment_terms_days"):
            return bool(re.search(rf"(?<![\d.]){int(x)}(?![\d.])", document))
        return any(squash(r) in source for r in number_renderings(x))
    return squash(value) in source


# --------------------------------------------------------------------------
# Field comparison
# --------------------------------------------------------------------------

def compare(field: str, truth, pred, document: str) -> str:
    """Return 'correct' or one of ERROR_TYPES."""
    if truth is None:
        return "correct" if pred in (None, "") else "hallucination"
    if pred is None or pred == "":
        return "omission"

    if field in DATE_FIELDS:
        target = date.fromisoformat(truth)
        parsed = as_date(pred)
        if parsed == target:
            return "correct" if pred == truth else "format"
    elif field in NUMBER_FIELDS:
        x = as_number(pred)
        if x is not None and abs(x - float(truth)) < 0.011:
            exact_type = isinstance(pred, (int, float)) and not isinstance(pred, bool)
            return "correct" if exact_type else "format"
    elif field == "currency":
        if str(pred).strip().upper() == truth:
            return "correct"
    elif field in ("supplier_name", "description"):
        if " ".join(str(pred).lower().split()) == " ".join(truth.lower().split()):
            return "correct"
    elif squash(pred) == squash(truth):
        return "correct"

    if appears_in(pred, field, document):
        return "substitution"
    return "calculation" if field in CALCULATED_FIELDS else "hallucination"


def match_items(truth_items: list[dict], pred_items: list) -> list[tuple[dict | None, dict | None]]:
    """
    Pair predicted with true line items: exact (normalised) description matches
    first, then the most similar remaining description (threshold 0.6).
    """
    preds = [p for p in pred_items if isinstance(p, dict)] if isinstance(pred_items, list) else []
    norm = lambda text: " ".join(str(text or "").lower().split())
    assigned: dict[int, int] = {}
    used: set[int] = set()
    for i, t in enumerate(truth_items):
        for j, p in enumerate(preds):
            if j not in used and norm(p.get("description")) == norm(t["description"]):
                assigned[i] = j
                used.add(j)
                break
    for i, t in enumerate(truth_items):
        if i in assigned:
            continue
        best, best_score = None, 0.6
        for j, p in enumerate(preds):
            if j in used:
                continue
            score = SequenceMatcher(None, norm(t["description"]), norm(p.get("description"))).ratio()
            if score > best_score:
                best, best_score = j, score
        if best is not None:
            assigned[i] = best
            used.add(best)
    pairs = [(t, preds[assigned[i]] if i in assigned else None) for i, t in enumerate(truth_items)]
    pairs += [(None, p) for j, p in enumerate(preds) if j not in used]
    return pairs


def score_record(truth: dict, pred: dict | None, document: str,
                 extra_fields: tuple[str, ...] = ()) -> list[dict]:
    """Field-level comparison of one record; returns one row per field."""
    pred = pred if isinstance(pred, dict) else {}
    rows = []
    for field in HEADER_FIELDS + list(extra_fields):
        outcome = compare(field, truth.get(field), pred.get(field), document)
        rows.append({"field": field, "truth": truth.get(field), "pred": pred.get(field),
                     "outcome": outcome})
    for n, (t, p) in enumerate(match_items(truth["line_items"], pred.get("line_items")), start=1):
        if t is None:      # an extra line item the document does not contain
            rows.append({"field": f"line_item_extra_{n}", "truth": None,
                         "pred": p.get("description"), "outcome": "hallucination"})
            continue
        for field in ITEM_FIELDS:
            value = p.get(field) if p else None
            rows.append({"field": f"line_{n}.{field}", "truth": t[field], "pred": value,
                         "outcome": compare(field, t[field], value, document)})
    return rows


def summarise_fields(rows: list[dict]) -> dict:
    expected = sum(1 for r in rows if not r["field"].startswith("line_item_extra"))
    counts = {e: sum(1 for r in rows if r["outcome"] == e) for e in ERROR_TYPES}
    errors = sum(counts.values())
    return {"fields": expected, "errors": errors,
            "error_rate": round(errors / expected, 4) if expected else None,
            **counts}


def record_hash(record) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# Run scoring
# --------------------------------------------------------------------------

def score_trace(trace: dict, truth: dict, document: str) -> tuple[dict, list[dict]]:
    initial = trace.get("initial_extraction", trace.get("extraction"))
    stages = {
        "initial": score_record(truth["extraction"], initial, document),
        "gated": score_record(truth["extraction"], trace.get("extraction"), document),
        "final": score_record(truth["final"], trace.get("final_record"), document,
                              extra_fields=("supplier_id",)),
    }
    summaries = {stage: summarise_fields(rows) for stage, rows in stages.items()}
    history = trace.get("gate_history") or []
    first = history[0] if history else {}
    gate_flagged = bool(first) and (not first.get("passed", True) or first.get("action") == "correct")
    s = trace["summary"]
    decision_correct = trace.get("decision") == truth["decision"]

    fault = trace.get("fault") or {}
    experiment = trace.get("experiment") or ("fault" if 201 <= trace["trial"] <= 299 else "natural")
    fault_present = experiment == "fault" and fault.get("fault_type") not in (None, "none")
    run = {
        "run_id": f"{trace['task_id']}_{trace['condition']}_t{trace['trial']}",
        "experiment": experiment,
        "task_id": trace["task_id"], "complexity": truth["complexity"], "layout": truth["layout"],
        "hitl_subset": truth["hitl_subset"], "condition": trace["condition"],
        "trial": trace["trial"], "seed": trace["seed"], "model_id": trace.get("model_id"),
        "fault_type": fault.get("fault_type"), "fault_variant": fault.get("variant"),
        "fault_c1_visible": fault.get("c1_visible"),
        "initial_extraction_hash": record_hash(initial),
    }
    for stage, summary in summaries.items():
        for key, value in summary.items():
            run[f"{stage}_{key}"] = value
    run.update({
        "initial_has_error": summaries["initial"]["errors"] > 0,
        "final_hallucination_present": summaries["final"]["hallucination"] > 0,
        "supplier_id_correct": (trace.get("final_record") or {}).get("supplier_id")
                               == truth["final"]["supplier_id"],
        "decision": trace.get("decision"), "decision_true": truth["decision"],
        "decision_correct": decision_correct,
        "task_success": decision_correct and summaries["final"]["errors"] == 0,
        "gate_flagged": gate_flagged,
        "fault_present": fault_present if experiment == "fault" else None,
        "fault_detected": (gate_flagged if fault_present else None),
        "false_alarm": (gate_flagged if experiment == "fault" and not fault_present else None),
        "fault_removed_at_gate": (summaries["gated"]["errors"] == 0 if fault_present else None),
        "gate_issue_count": len(first.get("issues") or []),
        "human_action": first.get("action"),
        "revisions": s["revisions"], "flagged_unresolved": s["flagged"],
        "model_calls": s["model_calls"], "input_tokens": s["input_tokens"],
        "output_tokens": s["output_tokens"], "thinking_tokens": s["thinking_tokens"],
        "token_cost_usd": s["token_cost_usd"],
        "machine_seconds": s.get("machine_seconds", s["wall_clock_s"]),
        "human_seconds": s.get("human_seconds", 0.0), "wall_clock_s": s["wall_clock_s"],
        "api_retries": s["api_retries"], "parse_failures": s["parse_failures"],
        "timestamp_utc": trace.get("timestamp_utc"),
    })
    field_rows = [{"run_id": run["run_id"], "task_id": run["task_id"],
                   "complexity": run["complexity"], "condition": run["condition"],
                   "trial": run["trial"], "stage": stage, **row}
                  for stage, rows in stages.items() for row in rows]
    return run, field_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                             for k, v in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Score run traces against the ground truth.")
    parser.add_argument("--results", default="results", help="results folder (default: results)")
    args = parser.parse_args()
    results = ROOT / args.results

    truth = {row["doc_id"]: row for row in
             json.loads((DATA / "ground_truth" / "documents.json").read_text(encoding="utf-8"))}
    runs, fields = [], []
    for path in sorted((results / "traces").glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        document = (DATA / "documents" / f"{trace['task_id']}.txt").read_text(encoding="utf-8")
        run, field_rows = score_trace(trace, truth[trace["task_id"]], document)
        runs.append(run)
        fields.extend(field_rows)

    write_csv(results / "runs.csv", runs)
    write_csv(results / "fields.csv", fields)
    print(f"Scored {len(runs)} runs -> {args.results}/runs.csv and {args.results}/fields.csv")
    if not runs:
        return

    natural = [r for r in runs if r["experiment"] == "natural"]
    seeded = [r for r in runs if r["experiment"] == "fault"]
    if natural:
        print_natural(natural)
    if seeded:
        print_faults(seeded)


def rate(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r[key] is not None]
    return f"{sum(values) / len(values):.1%}" if values else "-"


def print_natural(runs: list[dict]) -> None:
    print("\nExperiment 1 - natural errors")
    print(f"{'Condition':<10}{'Runs':>6}{'Initial errors':>16}{'Final error rate':>18}"
          f"{'Task success':>14}{'Decision correct':>18}{'Mean cost $':>13}{'Mean time s':>13}")
    for cond in sorted({r["condition"] for r in runs}):
        rs = [r for r in runs if r["condition"] == cond]
        n = len(rs)
        print(f"{cond:<10}{n:>6}"
              f"{sum(r['initial_has_error'] for r in rs) / n:>15.1%} "
              f"{sum(r['final_error_rate'] for r in rs) / n:>17.1%} "
              f"{sum(r['task_success'] for r in rs) / n:>13.1%} "
              f"{sum(r['decision_correct'] for r in rs) / n:>17.1%} "
              f"{sum(r['token_cost_usd'] for r in rs) / n:>12.5f} "
              f"{sum(r['wall_clock_s'] for r in rs) / n:>12.2f}")
    print("'Initial errors' = share of runs whose first extraction contained at least one error.")


def print_faults(runs: list[dict]) -> None:
    conditions = sorted({r["condition"] for r in runs})
    print("\nExperiment 2 - seeded faults")
    print(f"{'Condition':<10}{'Runs':>6}{'Detected':>10}{'False alarm':>13}{'Removed':>9}"
          f"{'Task success':>14}{'Decision correct':>18}{'Mean cost $':>13}{'Mean time s':>13}")
    for cond in conditions:
        rs = [r for r in runs if r["condition"] == cond]
        n = len(rs)
        print(f"{cond:<10}{n:>6}{rate(rs, 'fault_detected'):>10}{rate(rs, 'false_alarm'):>13}"
              f"{rate(rs, 'fault_removed_at_gate'):>9}{rate(rs, 'task_success'):>14}"
              f"{rate(rs, 'decision_correct'):>18}"
              f"{sum(r['token_cost_usd'] for r in rs) / n:>13.5f}"
              f"{sum(r['wall_clock_s'] for r in rs) / n:>13.2f}")
    print("\nDetection rate by seeded fault type")
    types = ERROR_TYPES + ["C1-visible", "semantic"]
    print(f"{'Condition':<10}" + "".join(f"{t:>15}" for t in types))
    for cond in conditions:
        rs = [r for r in runs if r["condition"] == cond and r["fault_present"]]
        cells = [rate([r for r in rs if r["fault_type"] == t], "fault_detected") for t in ERROR_TYPES]
        cells += [rate([r for r in rs if r["fault_c1_visible"]], "fault_detected"),
                  rate([r for r in rs if not r["fault_c1_visible"]], "fault_detected")]
        print(f"{cond:<10}" + "".join(f"{c:>15}" for c in cells))
    print("'Detected' = faulty records flagged by the gate; 'False alarm' = clean control records "
          "flagged;\n'Removed' = faulty records that left the gate with no errors.")


if __name__ == "__main__":
    main()
