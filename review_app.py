"""
review_app.py - Blinded review interface for the human-in-the-loop condition (C3).

The reviewer sees only the source document (the same scanned page images
the extractor received) and the extracted record. The
ground truth, the trial number and the run identifier are never displayed.
Items are presented in a fixed random order across documents and complexity
levels. Review time is measured from the moment an item is displayed until
it is submitted. A session ends after 40 items or 45 minutes.

Usage:
    streamlit run review_app.py
"""

from __future__ import annotations

import json
import math
import random
import time

import pandas as pd
import streamlit as st

from vtv import config, review_queue
from vtv.traces import document_images, document_text

SESSION_ITEM_LIMIT = 40
SESSION_MINUTES_LIMIT = 45
ORDER_SEED = 7

HEADER_FIELDS = ["invoice_number", "invoice_date", "due_date", "po_number", "supplier_name",
                 "supplier_vat_number", "currency", "subtotal", "vat_amount", "total_amount",
                 "payment_terms_days", "customer_reference"]
NUMBER_FIELDS = {"subtotal", "vat_amount", "total_amount"}
ITEM_COLUMNS = ["description", "quantity", "unit_price", "line_total"]


# --------------------------------------------------------------------------
# Conversion between records and form values
# --------------------------------------------------------------------------

def _as_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _parse(value: str, kind: str):
    """Convert a form string back to a typed value; unparsable input is kept as text."""
    text = (value or "").strip()
    if text == "":
        return None
    try:
        if kind == "int":
            return int(text)
        if kind == "num":
            return round(float(text.replace(",", "")), 2)
    except ValueError:
        return text
    return text


def to_form(record: dict | None) -> tuple[dict, list[dict]]:
    record = record if isinstance(record, dict) else {}
    header = {key: _as_text(record.get(key)) for key in HEADER_FIELDS}
    items = record.get("line_items") if isinstance(record.get("line_items"), list) else []
    rows = [{col: _as_text(item.get(col) if isinstance(item, dict) else None)
             for col in ITEM_COLUMNS} for item in items]
    return header, rows


def from_form(header: dict, rows: list[dict]) -> dict:
    record = {}
    for key in HEADER_FIELDS:
        kind = "num" if key in NUMBER_FIELDS else "int" if key == "payment_terms_days" else "str"
        record[key] = _parse(header.get(key, ""), kind)
    items = []
    for row in rows:
        if not any(_as_text(row.get(col)).strip() for col in ITEM_COLUMNS):
            continue
        items.append({"description": _parse(_as_text(row.get("description")), "str"),
                      "quantity": _parse(_as_text(row.get("quantity")), "num"),
                      "unit_price": _parse(_as_text(row.get("unit_price")), "num"),
                      "line_total": _parse(_as_text(row.get("line_total")), "num")})
    record["line_items"] = items
    return record


# --------------------------------------------------------------------------
# Session handling
# --------------------------------------------------------------------------

def ordered_pending() -> list[dict]:
    """All queue items in a fixed random order, filtered to those not yet reviewed."""
    items = sorted(review_queue.queue_items(), key=lambda i: i["review_id"])
    random.Random(ORDER_SEED).shuffle(items)
    done = review_queue.completed_reviews()
    return [item for item in items if item["review_id"] not in done]


def init_session() -> None:
    if "session" not in st.session_state:
        previous = [r.get("session", 0) for r in review_queue.completed_reviews().values()]
        st.session_state.session = (max(previous) if previous else 0) + 1
        st.session_state.session_start = time.time()
        st.session_state.session_count = 0
        st.session_state.item_start = {}


def main() -> None:
    st.set_page_config(page_title="Invoice review", layout="wide")
    init_session()

    elapsed_min = (time.time() - st.session_state.session_start) / 60
    pending = ordered_pending()
    st.title("Invoice extraction review")
    st.caption(f"Session {st.session_state.session} · reviewed this session: "
               f"{st.session_state.session_count}/{SESSION_ITEM_LIMIT} · "
               f"{elapsed_min:.0f}/{SESSION_MINUTES_LIMIT} min · items remaining: {len(pending)}")

    if not pending:
        st.success("No items awaiting review.")
        return
    if (st.session_state.session_count >= SESSION_ITEM_LIMIT
            or elapsed_min >= SESSION_MINUTES_LIMIT):
        st.warning("Session limit reached. Take a break, then restart the app to begin a new session.")
        return

    item = pending[0]
    rid = item["review_id"]
    st.session_state.item_start.setdefault(rid, time.time())
    header, rows = to_form(item["extraction"])

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Source document")
        if config.INPUT_MODE == "image":
            for page, path in enumerate(document_images(item["task_id"]), start=1):
                st.image(path, caption=f"Page {page}", width="stretch")
        else:
            st.code(document_text(item["task_id"]), language=None)
    with right:
        st.subheader("Extracted record")
        st.write("Check every value against the document. Correct any field that is wrong; "
                 "leave a field empty if the document does not contain it.")
        if item["extraction"] is None:
            st.error("The extractor did not return a usable record. Enter the values manually.")
        with st.form(key=f"form-{rid}"):
            edited = {key: st.text_input(key, value=header[key], key=f"{rid}-{key}")
                      for key in HEADER_FIELDS}
            st.markdown("**Line items**")
            table = st.data_editor(pd.DataFrame(rows, columns=ITEM_COLUMNS, dtype="string"),
                                   num_rows="dynamic", width="stretch",
                                   key=f"{rid}-items")
            submitted = st.form_submit_button("Submit review", type="primary")

    if submitted:
        seconds = time.time() - st.session_state.item_start[rid]
        original = from_form(header, rows)
        reviewed = from_form(edited, table.astype(object).where(table.notna(), None).to_dict("records"))
        if json.dumps(original, sort_keys=True) == json.dumps(reviewed, sort_keys=True):
            review_queue.record_review(rid, "approve", None, seconds, st.session_state.session)
        else:
            review_queue.record_review(rid, "correct", reviewed, seconds, st.session_state.session)
        st.session_state.session_count += 1
        st.rerun()


if __name__ == "__main__":
    main()
