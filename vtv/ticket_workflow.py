"""
ticket_workflow.py - The IT ticket-triage workflow used for the replication study.

    START -> triage -> gate --(issues found, revision allowed)--> triage
                          \\--(passed, or revision used)------> route -> prioritise -> END

The structure mirrors the invoice workflow: the triage agent classifies the
ticket (category, impact, urgency, affected service), the verification gate
applies the condition's mechanism, the routing agent matches the service
catalogue and resolver group through tools, and the priority agent applies
the priority rule. The gate behaviour, revision cap, human checkpoint and
measurement are shared with the invoice workflow.

Ticket runs use the same state keys as invoice runs, so that traces are
written and stored in the same way: "extraction" holds the triage record,
"final_record" the routed record and "decision" the priority.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from vtv import ticket_prompts, tools
from vtv.llm import call_llm
from vtv.mechanisms import TICKET_MECHANISMS
from vtv.workflow import (RunState, checkpointer, make_gate_node, route_after_gate,
                          summarise, thread_id)


def mock_triage(truth: dict) -> dict:
    """Ground-truth triage record, used only in offline test mode."""
    return {"category": truth["labels"]["category"], "impact": truth["reasoning"]["impact"],
            "urgency": truth["reasoning"]["urgency"],
            "affected_service": truth["labels"]["affected_service"],
            "summary": "offline test mode"}


def triage_node(state: RunState) -> dict:
    is_revision = state.get("feedback") is not None
    prompt = ticket_prompts.triage_prompt(
        state["document_text"], feedback=state.get("feedback"),
        previous=state.get("extraction") if is_revision else None)
    mock = mock_triage(state["mock"]) if state.get("mock") else None
    parsed, record = call_llm("extract_revision" if is_revision else "extract",
                              ticket_prompts.triage_system(), prompt, state["seed"], mock)
    update = {"extraction": parsed, "feedback": None, "calls": [record.to_dict()]}
    if not is_revision:
        update["initial_extraction"] = parsed
    return update


def route_node(state: RunState) -> dict:
    record = state.get("extraction") or {}
    candidates = tools.search_services(
        f"{record.get('affected_service') or ''} {record.get('summary') or ''}")
    group = tools.assignment_group(record.get("category"))
    mock = None
    if state.get("mock"):
        labels = state["mock"]["labels"]
        service = next((c for c in tools.search_services(labels["affected_service"], limit=25)
                        if c["name"] == labels["affected_service"]), None)
        mock = {**mock_triage(state["mock"]), "service_id": service and service["service_id"],
                "assignment_group": labels["assignment_group"]}
    parsed, call = call_llm("normalise", ticket_prompts.ROUTER_SYSTEM,
                            ticket_prompts.router_prompt(record, candidates, group),
                            state["seed"], mock)
    return {"candidates": candidates, "final_record": parsed, "calls": [call.to_dict()]}


def prioritise_node(state: RunState) -> dict:
    record = state.get("final_record") or {}
    mock = ({"priority": state["mock"]["labels"]["priority"], "justification": "offline test mode"}
            if state.get("mock") else None)
    parsed, call = call_llm("decide", ticket_prompts.priority_system(),
                            ticket_prompts.priority_prompt(record), state["seed"], mock)
    parsed = parsed or {}
    return {"decision": parsed.get("priority"), "justification": parsed.get("justification"),
            "calls": [call.to_dict()]}


def build_ticket_graph(saver=None):
    graph = StateGraph(RunState)
    graph.add_node("extract", triage_node)
    graph.add_node("gate", make_gate_node(TICKET_MECHANISMS))
    graph.add_node("normalise", route_node)
    graph.add_node("decide", prioritise_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "gate")
    graph.add_conditional_edges("gate", route_after_gate,
                                {"extract": "extract", "normalise": "normalise"})
    graph.add_edge("normalise", "decide")
    graph.add_edge("decide", END)
    return graph.compile(checkpointer=saver)


_GRAPH = None
_HUMAN_GRAPH = None


def _automated_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_ticket_graph()
    return _GRAPH


def human_graph():
    global _HUMAN_GRAPH
    if _HUMAN_GRAPH is None:
        _HUMAN_GRAPH = build_ticket_graph(checkpointer())
    return _HUMAN_GRAPH


def _initial_state(ticket_id: str, condition: str, trial: int, ticket_text: str,
                   mock: dict | None) -> RunState:
    from vtv import config
    return {"task_id": ticket_id, "condition": condition, "trial": trial,
            "seed": config.SEED_BASE + trial, "document_text": ticket_text,
            "image_paths": [], "experiment": "replication", "fault": None,
            "seeded_extraction": None, "mock": mock, "revisions": 0, "flagged": False,
            "human_seconds": 0.0, "calls": [], "gate_history": []}


def run_ticket_workflow(ticket_id: str, condition: str, trial: int, ticket_text: str,
                        mock: dict | None = None) -> dict:
    """Execute one complete ticket run in an automated condition (C0, C1 or C2)."""
    if condition == "C3":
        raise ValueError("C3 runs pause for review: use start_ticket_human_run")
    started = time.perf_counter()
    final = _automated_graph().invoke(_initial_state(ticket_id, condition, trial,
                                                     ticket_text, mock))
    final["summary"] = summarise(final, time.perf_counter() - started)
    return final


def start_ticket_human_run(ticket_id: str, trial: int, ticket_text: str,
                           mock: dict | None = None) -> tuple[dict, float]:
    """Run a C3 ticket workflow up to the human checkpoint."""
    run_config = {"configurable": {"thread_id": thread_id(ticket_id, trial)}}
    started = time.perf_counter()
    result = human_graph().invoke(_initial_state(ticket_id, "C3", trial, ticket_text, mock),
                                  run_config)
    seconds = time.perf_counter() - started
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        raise RuntimeError("C3 run finished without pausing for review")
    return interrupts[0].value, seconds


def resume_ticket_human_run(ticket_id: str, trial: int, review: dict,
                            machine_seconds_before: float) -> dict:
    """Resume a suspended C3 ticket run with the reviewer's response and finish it."""
    from langgraph.types import Command

    run_config = {"configurable": {"thread_id": thread_id(ticket_id, trial)}}
    started = time.perf_counter()
    final = human_graph().invoke(Command(resume=review), run_config)
    final["summary"] = summarise(final, machine_seconds_before + (time.perf_counter() - started))
    return final
