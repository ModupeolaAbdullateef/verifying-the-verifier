"""
ticket_prompts.py - System instructions and prompt builders for the ticket-triage replication.

The replication mirrors the invoice workflow with three agents: a triage agent
(the extractor's counterpart), a routing agent that matches the service
catalogue and resolver group through tools (the normaliser's counterpart), and
a priority agent that applies the priority rules (the decision agent's
counterpart). As in the main study, the same prompts are used in every
condition; only the verification gate differs.
"""

from __future__ import annotations

import json

from vtv import tools


def _rules_text() -> str:
    m = tools.manifest()
    categories = "\n".join(f"  - {c}" for c in m["ticket_categories"])
    impact = "\n".join(f"  {k} = {v}" for k, v in m["ticket_impact_levels"].items())
    urgency = "\n".join(f"  {k} = {v}" for k, v in m["ticket_urgency_levels"].items())
    special = "\n".join(f"  - {s}" for s in m["ticket_priority_rules"]["special_cases"])
    return (f"Categories:\n{categories}\n\nImpact:\n{impact}\n\nUrgency:\n{urgency}\n\n"
            f"Triage rules:\n{special}")


def triage_spec() -> str:
    return f"""\
Return one JSON object with exactly these keys:
  category          string, exactly one of the categories listed below
  impact            integer 1, 2 or 3
  urgency           integer 1, 2 or 3
  affected_service  string, a short description of the service or item affected by the main issue
  summary           string, one sentence describing the main issue

{_rules_text()}"""


def triage_system() -> str:
    return ("You are an IT service-desk triage agent. You read incoming tickets and classify "
            "them for routing.\n\n" + triage_spec())


ROUTER_SYSTEM = """\
You are an IT service-desk routing agent. You receive a triaged ticket record, the
candidate services returned by the service-catalogue search tool, and the resolver
group returned by the assignment lookup tool for the ticket's category.

Tasks:
1. Choose the catalogue service that matches the record's affected_service and summary.
   If no candidate matches, use null.
2. Return the full record with every original key, with:
     service_id         the chosen candidate's service_id, or null
     affected_service   replaced by the chosen candidate's name (unchanged if none)
     assignment_group   the resolver group returned by the lookup tool
   Do not change any other values.
Return one JSON object only."""


def priority_system() -> str:
    rule = tools.manifest()["ticket_priority_rules"]["rule"]
    return f"""\
You are an IT service-desk prioritisation agent. Apply the priority rule below to the
routed ticket record.

Priority rule: {rule}

Return one JSON object with exactly these keys:
  priority       one of: P1, P2, P3, P4
  justification  one sentence citing the impact and urgency that determined the priority"""


def checker_system() -> str:
    return (
        "You are an independent IT service-desk verification agent. Another agent triaged the "
        "ticket below; your task is to check its record against the ticket before the ticket "
        "is routed.\n\n"
        "Check every field against the ticket and the rules:\n"
        "- category reflects the underlying cause of the main issue the reporter asks to be fixed;\n"
        "- impact and urgency follow the definitions and triage rules, not the reporter's wording alone;\n"
        "- affected_service names the service or item affected by the main issue;\n"
        "- formats follow the specification.\n\n"
        "Specification the record must follow:\n" + triage_spec() + "\n\n"
        "Return one JSON object with exactly these keys:\n"
        "  passed  true only if you found no issues, otherwise false\n"
        "  issues  list of strings, each naming the field, the problem and the correct value; "
        "an empty list if passed is true"
    )


def _source(ticket_text: str) -> str:
    return f"Ticket:\n<<<\n{ticket_text}\n>>>"


def triage_prompt(ticket_text: str, feedback: str | None = None,
                  previous: dict | None = None) -> str:
    prompt = _source(ticket_text)
    if feedback:
        prompt += (
            "\n\nA reviewer checked your previous triage and reported these issues:\n"
            f"{feedback}\n\nPrevious record:\n{json.dumps(previous, indent=2)}\n\n"
            "Correct the issues and return the complete JSON object again."
        )
    return prompt


def checker_prompt(ticket_text: str, record: dict) -> str:
    return f"{_source(ticket_text)}\n\nTriage record to verify:\n{json.dumps(record, indent=2)}"


def router_prompt(record: dict, candidates: list[dict], group: str | None) -> str:
    return (f"Triage record:\n{json.dumps(record, indent=2)}\n\n"
            f"Service-catalogue search results:\n{json.dumps(candidates, indent=2)}\n\n"
            f"Assignment lookup result for category {record.get('category')!r}: "
            f"{json.dumps(group)}")


def priority_prompt(record: dict) -> str:
    return f"Routed ticket record:\n{json.dumps(record, indent=2)}"
