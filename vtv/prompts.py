"""
prompts.py - System instructions and prompt builders for the three agents.

The same prompts are used in every experimental condition; only the
verification gate differs between conditions.
"""

from __future__ import annotations

import json

INVOICE_FIELDS_SPEC = """\
Return one JSON object with exactly these keys:
  invoice_number       string, exactly as printed
  invoice_date         string, ISO format YYYY-MM-DD
  due_date             string, ISO format YYYY-MM-DD; if not stated, invoice_date plus payment_terms_days
  po_number            string, the buyer's purchase-order number
  supplier_name        string, exactly as printed on the invoice
  supplier_vat_number  string, exactly as printed, or null
  currency             string, ISO 4217 code (for example GBP, EUR)
  subtotal             number, net amount before VAT
  vat_amount           number
  total_amount         number, the amount currently payable on this invoice
  payment_terms_days   integer
  customer_reference   string, the buyer's own reference, or null
  line_items           list of objects with keys description, quantity, unit_price, line_total
Amounts are plain numbers with two decimal places and no currency symbols or thousands separators.
Use null for any value that the document does not contain. Do not invent values."""

EXTRACTOR_SYSTEM = (
    "You are an accounts-payable extraction agent. You read supplier invoices and "
    "convert them into structured records for the finance system.\n\n" + INVOICE_FIELDS_SPEC
)

NORMALISER_SYSTEM = """\
You are an accounts-payable normalisation agent. You receive an extracted invoice record
and a list of candidate suppliers returned by the supplier-master search tool.

Tasks:
1. Choose the candidate that is the same legal entity as the supplier on the invoice.
   Prefer an exact VAT-number match. If no candidate is the same entity, use null.
2. Return the full record with every original key, plus:
     supplier_id    the chosen candidate's supplier_id, or null
     supplier_name  replaced by the chosen candidate's registered name (unchanged if none)
3. Standardise formats only: ISO dates, numbers with two decimal places, ISO currency codes.
   Do not change any other values.
Return one JSON object only."""

DECISION_SYSTEM_TEMPLATE = """\
You are an accounts-payable approval agent. Apply the business rules below, in order,
to the invoice record and its purchase-order register entry. The first rule that
applies determines the decision.

Business rules:
{rules}

Return one JSON object with exactly these keys:
  decision       one of: {codes}
  justification  one or two sentences citing the values that determined the decision"""


def decision_system(rules: list[dict]) -> str:
    lines = "\n".join(f"{r['order']}. {r['code']}: {r['rule']}" for r in rules)
    codes = ", ".join(r["code"] for r in rules)
    return DECISION_SYSTEM_TEMPLATE.format(rules=lines, codes=codes)


def extractor_prompt(document_text: str, feedback: str | None = None,
                     previous: dict | None = None) -> str:
    prompt = f"Invoice:\n<<<\n{document_text}\n>>>"
    if feedback:
        prompt += (
            "\n\nA reviewer checked your previous extraction and reported these issues:\n"
            f"{feedback}\n\nPrevious extraction:\n{json.dumps(previous, indent=2)}\n\n"
            "Correct the issues and return the complete JSON object again."
        )
    return prompt


def normaliser_prompt(record: dict, candidates: list[dict]) -> str:
    return (f"Extracted record:\n{json.dumps(record, indent=2)}\n\n"
            f"Supplier-master search results:\n{json.dumps(candidates, indent=2)}")


def decision_prompt(record: dict, purchase_order: dict | None) -> str:
    po_text = (json.dumps(purchase_order, indent=2) if purchase_order
               else "NOT FOUND: no purchase order with this number exists in the register.")
    return (f"Invoice record:\n{json.dumps(record, indent=2)}\n\n"
            f"Purchase-order register entry:\n{po_text}")
