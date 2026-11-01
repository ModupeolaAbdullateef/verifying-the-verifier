"""
generate_faults.py - Generate the seeded-fault records for Experiment 2.

For every invoice and every Experiment 2 trial, one category is assigned:
one of the five error types of the taxonomy (omission, substitution,
calculation, hallucination, format) or "none" (an unmodified control record).
Each document receives every category equally often (four times over 24
trials), in a random order fixed by the seed.

For a fault category, a variant is drawn at random from those applicable to
the document and applied to a copy of the ground-truth extraction. The
mutated record is then scored with the same scorer used for the experiment
(score.py); a variant is accepted only if the scorer classifies the seeded
error as the intended type, so the labels are consistent with how results
are measured. Each record also states whether the fault is visible to the
deterministic rules of C1 (constraint-visible) or not (semantic), determined
by running the C1 validator on the record.

Output: data/faults/faults.json

Usage:
    python generate_faults.py
"""

from __future__ import annotations

import copy
import json
import random
import re
from datetime import date, timedelta

from score import as_date, score_record, summarise_fields
from vtv import config
from vtv.faults import CATEGORIES, FAULT_FILE, FAULT_TRIALS
from vtv.mechanisms import validate_record

SEED = 20261102
VERSION = "1.0.0"
REPEATS = len(FAULT_TRIALS) // len(CATEGORIES)


# --------------------------------------------------------------------------
# Values present in the source document (used for substitution faults)
# --------------------------------------------------------------------------

DATE_PATTERNS = [r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}", r"\d{2}-[A-Z][a-z]{2}-\d{2}",
                 r"\d{1,2}(?:st|nd|rd|th)? [A-Z][a-z]+ \d{4}"]
REFERENCE = re.compile(r"\b(?:[A-Z]{1,4}\d*-\d{3,}|\d{5,6})\b")
VAT = re.compile(r"GB \d{3} \d{4} \d{2}")
AMOUNT = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})*\.\d{2}(?![\d])")


def document_dates(text: str) -> list[str]:
    found = set()
    for pattern in DATE_PATTERNS:
        for match in re.findall(pattern, text):
            parsed = as_date(re.sub(r"(\d)(st|nd|rd|th)", r"\1", match))
            if parsed:
                found.add(parsed.isoformat())
    return sorted(found)


def document_amounts(text: str) -> list[float]:
    return sorted({float(m.replace(",", "")) for m in AMOUNT.findall(text)})


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def money(x: float) -> float:
    return round(x + 1e-9, 2)


def retotal(record: dict) -> None:
    """Recompute subtotal, VAT and total from the line items (internally consistent)."""
    subtotal = money(sum(item["line_total"] for item in record["line_items"]))
    rate = 0.0 if abs(record["vat_amount"]) < 0.005 else 0.20
    record["subtotal"] = subtotal
    record["vat_amount"] = money(subtotal * rate)
    record["total_amount"] = money(record["subtotal"] + record["vat_amount"])


def other(values: list, exclude: set, rng: random.Random):
    pool = [v for v in values if v not in exclude]
    return rng.choice(pool) if pool else None


def digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


# --------------------------------------------------------------------------
# Fault variants. Each takes (record, document_text, rng) and mutates the
# record in place, returning a short description, or None if not applicable.
# --------------------------------------------------------------------------

def omit_required_identifier(r, text, rng):
    field = rng.choice(["po_number", "invoice_number"])
    r[field] = None
    return f"{field} set to null"


def omit_optional_identifier(r, text, rng):
    fields = [f for f in ("supplier_vat_number", "customer_reference") if r.get(f)]
    if not fields:
        return None
    field = rng.choice(fields)
    r[field] = None
    return f"{field} set to null"


def omit_line_item(r, text, rng):
    if len(r["line_items"]) < 2:
        return None
    n = rng.randrange(len(r["line_items"]))
    r["line_items"].pop(n)
    return f"line item {n + 1} removed; totals unchanged"


def omit_line_item_consistently(r, text, rng):
    candidates = [i for i, item in enumerate(r["line_items"]) if item["line_total"] > 0]
    if len(r["line_items"]) < 2 or not candidates:
        return None
    n = rng.choice(candidates)
    r["line_items"].pop(n)
    retotal(r)
    return f"line item {n + 1} removed; subtotal, VAT and total recomputed to match"


def substitute_invoice_date(r, text, rng):
    value = other(document_dates(text), {r["invoice_date"], r["due_date"]}, rng)
    if value is None:
        return None
    r["invoice_date"] = value
    return f"invoice_date replaced by another date in the document ({value})"


def substitute_invoice_number(r, text, rng):
    exclude = {r["invoice_number"], r["po_number"], r.get("customer_reference")}
    value = other(sorted(set(REFERENCE.findall(text))), exclude, rng)
    if value is None:
        return None
    r["invoice_number"] = value
    return f"invoice_number replaced by another reference in the document ({value})"


def substitute_vat_number(r, text, rng):
    value = other(sorted(set(VAT.findall(text))), {r.get("supplier_vat_number")}, rng)
    if value is None:
        return None
    r["supplier_vat_number"] = value
    return f"supplier_vat_number replaced by another VAT number in the document ({value})"


def substitute_total(r, text, rng):
    exclude = {r["total_amount"], r["subtotal"], r["vat_amount"]}
    exclude |= {item["line_total"] for item in r["line_items"]}
    value = other([a for a in document_amounts(text) if a > r["total_amount"] * 0.5],
                  exclude, rng)
    if value is None:
        return None
    r["total_amount"] = value
    return f"total_amount replaced by another amount in the document ({value})"


def substitute_quantity(r, text, rng):
    items = r["line_items"]
    pairs = [(i, j) for i in range(len(items)) for j in range(len(items))
             if i != j and items[i]["quantity"] != items[j]["quantity"]]
    if not pairs:
        return None
    i, j = rng.choice(pairs)
    items[i]["quantity"] = items[j]["quantity"]
    return f"line item {i + 1} quantity replaced by that of line item {j + 1}"


def miscalculate_line_total(r, text, rng):
    candidates = [i for i, item in enumerate(r["line_items"]) if item["quantity"] > 1]
    n = rng.choice(candidates) if candidates else rng.randrange(len(r["line_items"]))
    item = r["line_items"][n]
    item["line_total"] = money(item["line_total"] + rng.choice([-1, 1]) * rng.choice([1, 10, 100]))
    return f"line item {n + 1} line_total miscalculated"


def miscalculate_vat(r, text, rng):
    if abs(r["vat_amount"]) < 0.005:
        return None
    rate = rng.choice([0.175, 0.15, 0.05])
    r["vat_amount"] = money(r["subtotal"] * rate)
    r["total_amount"] = money(r["subtotal"] + r["vat_amount"])
    return f"vat_amount calculated at {rate:.1%} and total_amount recomputed"


def miscalculate_due_date(r, text, rng):
    invoice = date.fromisoformat(r["invoice_date"])
    shift = rng.choice([-1, 1, 7, 14])
    r["due_date"] = (invoice + timedelta(days=r["payment_terms_days"] + shift)).isoformat()
    return f"due_date derived incorrectly (offset {shift:+d} days)"


def hallucinate_customer_reference(r, text, rng):
    if r.get("customer_reference") is not None:
        return None
    r["customer_reference"] = f"CC-{digits(rng, 4)}"
    return "customer_reference fabricated although the document has none"


def hallucinate_po_number(r, text, rng):
    r["po_number"] = f"PO-{digits(rng, 6)}"
    return "po_number fabricated in the valid format"


def hallucinate_vat_number(r, text, rng):
    r["supplier_vat_number"] = f"GB {digits(rng, 3)} {digits(rng, 4)} {digits(rng, 2)}"
    return "supplier_vat_number fabricated in the valid format"


def hallucinate_line_item(r, text, rng):
    quantity = rng.randint(1, 4)
    price = money(rng.uniform(20, 250))
    r["line_items"].insert(rng.randint(0, len(r["line_items"])), {
        "description": rng.choice(["Administration fee", "Expedited delivery surcharge",
                                   "Packaging and handling", "Site attendance charge"]),
        "quantity": quantity, "unit_price": price, "line_total": money(quantity * price)})
    retotal(r)
    return "fabricated line item added; subtotal, VAT and total recomputed to match"


def format_invoice_date(r, text, rng):
    r["invoice_date"] = date.fromisoformat(r["invoice_date"]).strftime("%d/%m/%Y")
    return "invoice_date in DD/MM/YYYY instead of ISO format"


def format_due_date(r, text, rng):
    d = date.fromisoformat(r["due_date"])
    r["due_date"] = f"{d.day} {d.strftime('%B')} {d.year}"
    return "due_date written out in words instead of ISO format"


def format_total_as_string(r, text, rng):
    r["total_amount"] = f"{r['total_amount']:,.2f}"
    return "total_amount as a string with a thousands separator"


def format_quantity_as_string(r, text, rng):
    n = rng.randrange(len(r["line_items"]))
    r["line_items"][n]["quantity"] = str(r["line_items"][n]["quantity"])
    return f"line item {n + 1} quantity as a string"


# variant name -> (function, error types the scorer may report for it)
VARIANTS = {
    "omission": {
        "omit_required_identifier": (omit_required_identifier, {"omission"}),
        "omit_optional_identifier": (omit_optional_identifier, {"omission"}),
        "omit_line_item": (omit_line_item, {"omission"}),
        "omit_line_item_consistently": (omit_line_item_consistently, {"omission", "calculation"}),
    },
    "substitution": {
        "substitute_invoice_date": (substitute_invoice_date, {"substitution"}),
        "substitute_invoice_number": (substitute_invoice_number, {"substitution"}),
        "substitute_vat_number": (substitute_vat_number, {"substitution"}),
        "substitute_total": (substitute_total, {"substitution"}),
        "substitute_quantity": (substitute_quantity, {"substitution"}),
    },
    "calculation": {
        "miscalculate_line_total": (miscalculate_line_total, {"calculation"}),
        "miscalculate_vat": (miscalculate_vat, {"calculation"}),
        "miscalculate_due_date": (miscalculate_due_date, {"calculation"}),
    },
    "hallucination": {
        "hallucinate_customer_reference": (hallucinate_customer_reference, {"hallucination"}),
        "hallucinate_po_number": (hallucinate_po_number, {"hallucination"}),
        "hallucinate_vat_number": (hallucinate_vat_number, {"hallucination"}),
        "hallucinate_line_item": (hallucinate_line_item, {"hallucination", "calculation"}),
    },
    "format": {
        "format_invoice_date": (format_invoice_date, {"format"}),
        "format_due_date": (format_due_date, {"format"}),
        "format_total_as_string": (format_total_as_string, {"format"}),
        "format_quantity_as_string": (format_quantity_as_string, {"format"}),
    },
}


def scored_types(truth: dict, record: dict, text: str) -> dict[str, int]:
    summary = summarise_fields(score_record(truth, record, text))
    return {k: summary[k] for k in VARIANTS if summary[k]}


def seed_fault(category: str, truth: dict, text: str, rng: random.Random) -> dict:
    """Apply one accepted variant of the category to a copy of the truth record."""
    names = list(VARIANTS[category])
    rng.shuffle(names)
    for name in names:
        function, allowed = VARIANTS[category][name]
        record = copy.deepcopy(truth)
        description = function(record, text, rng)
        if description is None:
            continue
        types = scored_types(truth, record, text)
        if category in types and set(types) <= allowed:
            return {"variant": name, "description": description,
                    "record": record, "scored_errors": types}
    raise RuntimeError(f"No applicable {category} variant")


def main() -> None:
    truth_rows = json.loads((config.DATA_DIR / "ground_truth" / "documents.json")
                            .read_text(encoding="utf-8"))
    rows = []
    for doc in truth_rows:
        doc_id, truth = doc["doc_id"], doc["extraction"]
        text = (config.DATA_DIR / "documents" / f"{doc_id}.txt").read_text(encoding="utf-8")
        schedule = CATEGORIES * REPEATS
        random.Random(f"{SEED}-{doc_id}").shuffle(schedule)
        for trial, category in zip(FAULT_TRIALS, schedule):
            rng = random.Random(f"{SEED}-{doc_id}-{trial}")
            if category == "none":
                seeded = {"variant": "none", "description": "unmodified ground-truth record",
                          "record": copy.deepcopy(truth), "scored_errors": {}}
            else:
                seeded = seed_fault(category, truth, text, rng)
            c1_issues = validate_record(seeded["record"], None)
            rows.append({"doc_id": doc_id, "trial": trial, "complexity": doc["complexity"],
                         "fault_type": category, **seeded,
                         "c1_visible": bool(c1_issues), "c1_issues": c1_issues})

    FAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAULT_FILE.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"generate_faults.py v{VERSION}: {len(rows)} seeded records -> "
          f"{FAULT_FILE.relative_to(config.ROOT)}")
    print(f"\n{'Category':<15}{'Records':>8}{'C1-visible':>12}  Variants")
    for category in CATEGORIES:
        rs = [r for r in rows if r["fault_type"] == category]
        variants = sorted({r["variant"] for r in rs})
        print(f"{category:<15}{len(rs):>8}{sum(r['c1_visible'] for r in rs):>12}  "
              f"{len(variants)} used")
    hitl = [r for r in rows if r["doc_id"] in {d["doc_id"] for d in truth_rows if d["hitl_subset"]}]
    print(f"\nHuman-review subset: {len(hitl)} records.")


if __name__ == "__main__":
    main()
