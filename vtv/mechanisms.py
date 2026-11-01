"""
mechanisms.py - The verification mechanisms evaluated in the study.

Each mechanism is a function that receives the workflow state at the
verification gate and returns a GateResult. The gate sits between the
extractor and the normaliser, so a mechanism inspects the extractor's output
before any downstream agent can act on it.

    C0  gate_unverified   No verification (baseline).
    C1  gate_schema       Deterministic schema and constraint validation.
    C2  gate_checker      A separate LLM checker agent with an explicit rubric.
    C3  gate_human        A human reviewer, via a LangGraph interrupt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from vtv import config, prompts
from vtv.llm import CallRecord, call_llm


@dataclass
class GateResult:
    """Outcome of one evaluation by a verification mechanism."""
    passed: bool
    issues: list[str] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    corrected: dict | None = None      # record corrected directly by a human reviewer
    human_seconds: float = 0.0
    action: str | None = None          # human reviewer's action: "approve" or "correct"


# --------------------------------------------------------------------------
# C0 - unverified baseline
# --------------------------------------------------------------------------

def gate_unverified(state: dict) -> GateResult:
    return GateResult(passed=True)


# --------------------------------------------------------------------------
# C1 - schema and constraint validation
# --------------------------------------------------------------------------

# Field types. "num" accepts integers or floats (but not booleans).
SCHEMA = {
    "invoice_number": ("str", False), "invoice_date": ("str", False),
    "due_date": ("str", False), "po_number": ("str", False),
    "supplier_name": ("str", False), "supplier_vat_number": ("str", True),
    "currency": ("str", False), "subtotal": ("num", False),
    "vat_amount": ("num", False), "total_amount": ("num", False),
    "payment_terms_days": ("int", False), "customer_reference": ("str", True),
    "line_items": ("list", False),
}
LINE_ITEM_SCHEMA = {"description": "str", "quantity": "num",
                    "unit_price": "num", "line_total": "num"}

# Formats defined by the buyer's own systems (purchase orders and cost centres).
PO_PATTERN = re.compile(r"^PO-\d{6}$")
CUSTOMER_REF_PATTERN = re.compile(r"^CC-\d{4}$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CURRENCIES = {"GBP", "EUR", "USD"}
VAT_RATE = 0.20


def _is_type(value, kind: str) -> bool:
    if kind == "str":
        return isinstance(value, str) and value.strip() != ""
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "num":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "list":
        return isinstance(value, list)
    return False


def _parse_date(value) -> date | None:
    if not isinstance(value, str) or not ISO_DATE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _squash(text: str) -> str:
    """Lower-case and remove whitespace, for grounding checks."""
    return re.sub(r"\s+", "", text).lower()


def validate_record(record: dict | None, document_text: str | None = None) -> list[str]:
    """
    Return a list of human-readable rule violations (empty if the record is valid).

    Source-grounding checks need machine-readable document text. When the input
    is a scanned image (document_text is None) they are skipped, as they would be
    in a scan-based pipeline without an OCR layer.
    """
    if not isinstance(record, dict):
        return ["The output is not a valid JSON object."]
    issues: list[str] = []

    # 1. Structure and types
    for key in record:
        if key not in SCHEMA:
            issues.append(f"Unexpected field '{key}'.")
    for key, (kind, nullable) in SCHEMA.items():
        if key not in record:
            issues.append(f"Missing field '{key}'.")
        elif record[key] is None:
            if not nullable:
                issues.append(f"'{key}' is null but is required.")
        elif not _is_type(record[key], kind):
            issues.append(f"'{key}' must be of type {kind}; got {record[key]!r}.")

    items = record.get("line_items") if isinstance(record.get("line_items"), list) else []
    if isinstance(record.get("line_items"), list) and not items:
        issues.append("'line_items' is empty.")
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            issues.append(f"Line item {n} is not an object.")
            continue
        for key, kind in LINE_ITEM_SCHEMA.items():
            if not _is_type(item.get(key), kind):
                issues.append(f"Line item {n}: '{key}' missing or not of type {kind}.")

    # 2. Formats
    inv_date, due_date = _parse_date(record.get("invoice_date")), _parse_date(record.get("due_date"))
    for key, parsed in (("invoice_date", inv_date), ("due_date", due_date)):
        if record.get(key) is not None and parsed is None:
            issues.append(f"'{key}' is not a valid ISO date (YYYY-MM-DD): {record.get(key)!r}.")
    if isinstance(record.get("currency"), str) and record["currency"] not in CURRENCIES:
        issues.append(f"'currency' must be an ISO 4217 code; got {record['currency']!r}.")
    if isinstance(record.get("po_number"), str) and not PO_PATTERN.match(record["po_number"]):
        issues.append(f"'po_number' does not match the buyer format PO-nnnnnn: {record['po_number']!r}.")
    ref = record.get("customer_reference")
    if isinstance(ref, str) and not CUSTOMER_REF_PATTERN.match(ref):
        issues.append(f"'customer_reference' does not match the buyer format CC-nnnn: {ref!r}.")
    terms = record.get("payment_terms_days")
    if _is_type(terms, "int") and not 0 < terms <= 120:
        issues.append(f"'payment_terms_days' is outside the plausible range 1-120: {terms}.")

    # 3. Date logic
    if inv_date and due_date and _is_type(terms, "int"):
        expected = inv_date + timedelta(days=terms)
        if due_date != expected:
            issues.append(f"'due_date' {due_date} does not equal invoice_date plus "
                          f"{terms} days ({expected}).")

    # 4. Arithmetic
    line_sum = 0.0
    arithmetic_ok = True
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not all(_is_type(item.get(k), "num")
                                                 for k in ("quantity", "unit_price", "line_total")):
            arithmetic_ok = False
            continue
        if abs(item["quantity"] * item["unit_price"] - item["line_total"]) > 0.011:
            issues.append(f"Line item {n}: quantity x unit_price ({item['quantity']} x "
                          f"{item['unit_price']}) does not equal line_total {item['line_total']}.")
        line_sum += item["line_total"]
    sub, vat, total = record.get("subtotal"), record.get("vat_amount"), record.get("total_amount")
    if arithmetic_ok and items and _is_type(sub, "num") and abs(line_sum - sub) > 0.02:
        issues.append(f"Line totals sum to {line_sum:.2f} but 'subtotal' is {sub}.")
    if all(_is_type(v, "num") for v in (sub, vat, total)):
        if abs(sub + vat - total) > 0.011:
            issues.append(f"'subtotal' + 'vat_amount' ({sub} + {vat}) does not equal "
                          f"'total_amount' {total}.")
        if abs(vat) > 0.005 and abs(sub * VAT_RATE - vat) > 0.02:
            issues.append(f"'vat_amount' {vat} is neither zero nor 20% of the subtotal "
                          f"({sub * VAT_RATE:.2f}).")

    # 5. Source grounding: identifiers must appear in the document (text input only)
    if document_text is None:
        return issues
    source = _squash(document_text)
    for key in ("invoice_number", "po_number", "supplier_vat_number",
                "customer_reference", "supplier_name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip() and _squash(value) not in source:
            issues.append(f"'{key}' value {value!r} does not appear in the invoice.")

    return issues


def gate_schema(state: dict) -> GateResult:
    """C1: deterministic validation; no model calls and no token cost."""
    text = None if config.INPUT_MODE == "image" else state["document_text"]
    issues = validate_record(state.get("extraction"), text)
    return GateResult(passed=not issues, issues=issues)


# --------------------------------------------------------------------------
# C2 - checker agent
# --------------------------------------------------------------------------

def gate_checker(state: dict) -> GateResult:
    """C2: an independent LLM agent compares the extraction with the source document."""
    record = state.get("extraction")
    if not isinstance(record, dict):
        return GateResult(passed=False, issues=["The output is not a valid JSON object."])
    mock = {"passed": True, "issues": []} if state.get("mock") else None
    image_mode = config.INPUT_MODE == "image"
    verdict, call = call_llm("checker", prompts.CHECKER_SYSTEM,
                             prompts.checker_prompt(None if image_mode else state["document_text"],
                                                    record),
                             state["seed"], mock,
                             images=state.get("image_paths") if image_mode else None)
    if not isinstance(verdict, dict) or not isinstance(verdict.get("passed"), bool):
        # An unusable verdict is treated as a pass: the checker failed to object.
        return GateResult(passed=True, issues=["checker returned no usable verdict"],
                          calls=[call])
    issues = [str(i) for i in verdict.get("issues") or []]
    passed = verdict["passed"] and not issues
    return GateResult(passed=passed, issues=issues, calls=[call])


# --------------------------------------------------------------------------
# C3 - human-in-the-loop checkpoint
# --------------------------------------------------------------------------

def gate_human(state: dict) -> GateResult:
    """
    C3: pause the workflow for human review using a LangGraph interrupt.

    The run is suspended and its state saved by the checkpointer. The review
    application later resumes it with the reviewer's response:
        {"action": "approve" | "correct", "corrected": {...} | None, "seconds": float}
    The reviewer is the final authority, so the gate always passes after review.
    """
    from langgraph.types import interrupt

    review = interrupt({"task_id": state["task_id"], "trial": state["trial"],
                        "extraction": state.get("extraction")})
    seconds = float(review.get("seconds", 0.0))
    if review.get("action") == "correct" and isinstance(review.get("corrected"), dict):
        before, after = state.get("extraction") or {}, review["corrected"]
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        if not changed:
            return GateResult(passed=True, human_seconds=seconds, action="approve")
        return GateResult(passed=True, issues=[f"reviewer corrected: {', '.join(changed)}"],
                          corrected=after, human_seconds=seconds, action="correct")
    return GateResult(passed=True, human_seconds=seconds, action="approve")


MECHANISMS = {
    "C0": gate_unverified,
    "C1": gate_schema,
    "C2": gate_checker,
    "C3": gate_human,
}
