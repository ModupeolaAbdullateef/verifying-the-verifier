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
  total_amount         number, the gross total of this invoice (subtotal plus VAT); exclude balances
                       brought forward from other invoices and any superseded amounts
  payment_terms_days   integer
  customer_reference   string, the buyer's own reference, or null
  line_items           list of objects with keys description, quantity, unit_price, line_total
Amounts are JSON numbers, never strings (for example 1284.5, not "1,284.50"), with no currency
symbols or thousands separators. Discounts and credits are negative numbers.
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
3. Standardise formats only: ISO dates, amounts as JSON numbers (never strings), ISO currency codes.
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


CHECKER_SYSTEM = (
    "You are an independent accounts-payable verification agent. Another agent produced "
    "the extracted record below; your task is to check it against the source invoice "
    "before it enters the finance system.\n\n"
    "Check every field against the invoice:\n"
    "- each value appears in the invoice or is correctly derived from it;\n"
    "- values that the invoice does not contain are null, not invented;\n"
    "- total_amount is this invoice's gross total, not a superseded amount, a previous invoice or an account balance;\n"
    "- dates are the correct dates (not order or delivery dates) and due_date follows the rule below;\n"
    "- line items match the invoice, and quantity x unit_price, the line totals, subtotal, "
    "VAT and total are arithmetically consistent;\n"
    "- formats follow the specification.\n\n"
    "Specification the extraction must follow:\n" + INVOICE_FIELDS_SPEC + "\n\n"
    "Return one JSON object with exactly these keys:\n"
    "  passed  true only if you found no issues, otherwise false\n"
    "  issues  list of strings, each naming the field, the problem and, where visible in the "
    "invoice, the correct value; an empty list if passed is true"
)


def checker_prompt(document_text: str | None, record: dict) -> str:
    return (f"{_source(document_text)}\n\n"
            f"Extracted record to verify:\n{json.dumps(record, indent=2)}")


def decision_system(rules: list[dict]) -> str:
    lines = "\n".join(f"{r['order']}. {r['code']}: {r['rule']}" for r in rules)
    codes = ", ".join(r["code"] for r in rules)
    return DECISION_SYSTEM_TEMPLATE.format(rules=lines, codes=codes)


IMAGE_SOURCE = "The invoice is provided as the attached scanned page image(s), in page order."


def _source(document_text: str | None) -> str:
    return IMAGE_SOURCE if document_text is None else f"Invoice:\n<<<\n{document_text}\n>>>"


def extractor_prompt(document_text: str | None, feedback: str | None = None,
                     previous: dict | None = None) -> str:
    prompt = _source(document_text)
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
