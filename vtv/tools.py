"""
tools.py - Deterministic tools available to the agents.

search_suppliers() is the reference-table lookup used by the normaliser
agent; get_purchase_order() is the register lookup used by the decision
agent. Both read the reference data produced by generate_data.py.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache

from vtv import config

_LEGAL_SUFFIXES = re.compile(r"\b(ltd|limited|plc|ab|llc|co)\b\.?", re.IGNORECASE)


@lru_cache(maxsize=1)
def _suppliers() -> list[dict]:
    return json.loads((config.DATA_DIR / "reference" / "suppliers.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _purchase_orders() -> dict[str, dict]:
    rows = json.loads((config.DATA_DIR / "reference" / "purchase_orders.json").read_text(encoding="utf-8"))
    return {row["po_number"]: row for row in rows}


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads((config.DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def _normalise_name(name: str) -> str:
    name = name.lower().replace("&", " and ")
    name = _LEGAL_SUFFIXES.sub(" ", name)
    return re.sub(r"[^a-z0-9 ]", " ", " ".join(name.split())).strip()


def _normalise_vat(vat: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (vat or "").upper())


def search_suppliers(name: str | None, vat_number: str | None, limit: int = 3) -> list[dict]:
    """
    Return the closest supplier-master entries to a printed name and VAT number.

    Candidates are ranked by string similarity of the normalised names; an
    exact VAT-number match adds a strong bonus. The agent receives the ranked
    candidates and must choose one (or none), so the final match remains a
    model decision that can be right or wrong.
    """
    query, vat = _normalise_name(name or ""), _normalise_vat(vat_number)
    scored = []
    for s in _suppliers():
        score = SequenceMatcher(None, query, _normalise_name(s["name"])).ratio()
        if vat and vat == _normalise_vat(s["vat_number"]):
            score += 1.0
        scored.append((score, s))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{"supplier_id": s["supplier_id"], "name": s["name"],
             "vat_number": s["vat_number"], "currency": s["currency"],
             "similarity": round(score, 3)} for score, s in scored[:limit]]


def get_purchase_order(po_number: str | None) -> dict | None:
    """Return the purchase-order register entry, or None if it does not exist."""
    if not po_number:
        return None
    return _purchase_orders().get(po_number.strip().upper())


# --------------------------------------------------------------------------
# Ticket-triage replication
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "my", "our",
              "is", "it", "with", "by", "at", "after", "including"}


@lru_cache(maxsize=1)
def _services() -> list[dict]:
    return json.loads((config.DATA_DIR / "reference" / "services.json").read_text(encoding="utf-8"))


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOPWORDS}


def search_services(description: str | None, limit: int = 5) -> list[dict]:
    """
    Return the service-catalogue entries closest to a free-text service description.

    Entries are ranked by the share of the description's words found in the
    entry's name and description, with string similarity of the names as a
    tie-breaker. The routing agent receives the ranked candidates and must
    choose one (or none), so the final match remains a model decision.
    """
    query = _tokens(description or "")
    scored = []
    for s in _services():
        words = _tokens(s["name"] + " " + s["description"])
        overlap = len(query & words) / len(query) if query else 0.0
        similarity = SequenceMatcher(None, (description or "").lower(), s["name"].lower()).ratio()
        scored.append((overlap + 0.5 * similarity, s))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{"service_id": s["service_id"], "name": s["name"], "description": s["description"],
             "score": round(score, 3)} for score, s in scored[:limit]]


def assignment_group(category: str | None) -> str | None:
    """Return the resolver group for a ticket category, or None if the category is unknown."""
    return manifest()["ticket_categories"].get(category or "")
