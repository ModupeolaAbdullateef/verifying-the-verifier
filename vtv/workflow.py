"""
workflow.py - The invoice-processing multi-agent workflow, built with LangGraph.

    START -> extract -> gate --(issues found, revision allowed)--> extract
                            \\--(passed, or revision used)------> normalise -> decide -> END

The extractor, normaliser and decision agents are identical in every
experimental condition. The only component that changes is the verification
gate, selected from GATES by the run's condition code:

    C0  unverified baseline (gate always passes, no cost)
    C1  schema and constraint validation
    C2  checker agent
    C3  human-in-the-loop checkpoint

A gate that finds issues may send the extraction back once for revision
(config.MAX_REVISIONS). If issues remain after the revision, the record
proceeds and the run is flagged.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from vtv import config, prompts, tools
from vtv.llm import CallRecord, call_llm


class RunState(TypedDict, total=False):
    """Shared state passed between the nodes of one workflow run."""
    task_id: str
    condition: str
    trial: int
    seed: int
    document_text: str
    mock: dict | None                 # ground truth, used only in offline test mode
    extraction: dict | None
    revisions: int
    feedback: str | None
    gate_passed: bool | None
    flagged: bool
    candidates: list
    final_record: dict | None
    purchase_order: dict | None
    decision: str | None
    justification: str | None
    human_seconds: float
    calls: Annotated[list, operator.add]          # one entry per model call
    gate_history: Annotated[list, operator.add]   # one entry per gate evaluation


@dataclass
class GateResult:
    """Outcome of one evaluation by a verification mechanism."""
    passed: bool
    issues: list[str] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    corrected: dict | None = None      # record corrected directly (human reviewer)
    human_seconds: float = 0.0


# --------------------------------------------------------------------------
# Verification mechanisms
# --------------------------------------------------------------------------

def gate_unverified(state: RunState) -> GateResult:
    """C0: no verification."""
    return GateResult(passed=True)


GATES: dict[str, Callable[[RunState], GateResult]] = {
    "C0": gate_unverified,
}


def register_gate(condition: str, mechanism: Callable[[RunState], GateResult]) -> None:
    """Add a verification mechanism under a condition code."""
    GATES[condition] = mechanism


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def extract_node(state: RunState) -> dict:
    is_revision = state.get("feedback") is not None
    prompt = prompts.extractor_prompt(
        state["document_text"],
        feedback=state.get("feedback"),
        previous=state.get("extraction") if is_revision else None,
    )
    mock = state["mock"]["extraction"] if state.get("mock") else None
    parsed, record = call_llm("extract_revision" if is_revision else "extract",
                              prompts.EXTRACTOR_SYSTEM, prompt, state["seed"], mock)
    return {"extraction": parsed, "feedback": None, "calls": [record.to_dict()]}


def gate_node(state: RunState) -> dict:
    mechanism = GATES[state["condition"]]
    started = time.perf_counter()
    result = mechanism(state)
    gate_seconds = time.perf_counter() - started

    update: dict = {
        "gate_passed": result.passed,
        "calls": [c.to_dict() for c in result.calls],
        "gate_history": [{
            "revision": state.get("revisions", 0),
            "passed": result.passed,
            "issues": result.issues,
            "gate_seconds": round(gate_seconds, 3),
        }],
    }
    if result.corrected is not None:
        update["extraction"] = result.corrected
    if result.human_seconds:
        update["human_seconds"] = state.get("human_seconds", 0.0) + result.human_seconds

    revisions = state.get("revisions", 0)
    if not result.passed and result.corrected is None and revisions < config.MAX_REVISIONS:
        update["feedback"] = "\n".join(f"- {issue}" for issue in result.issues)
        update["revisions"] = revisions + 1
    else:
        update["flagged"] = not result.passed and result.corrected is None
    return update


def route_after_gate(state: RunState) -> str:
    return "extract" if state.get("feedback") else "normalise"


def normalise_node(state: RunState) -> dict:
    record = state.get("extraction") or {}
    candidates = tools.search_suppliers(record.get("supplier_name"),
                                        record.get("supplier_vat_number"))
    mock = state["mock"]["final"] if state.get("mock") else None
    parsed, call = call_llm("normalise", prompts.NORMALISER_SYSTEM,
                            prompts.normaliser_prompt(record, candidates), state["seed"], mock)
    return {"candidates": candidates, "final_record": parsed, "calls": [call.to_dict()]}


def decide_node(state: RunState) -> dict:
    record = state.get("final_record") or {}
    purchase_order = tools.get_purchase_order(record.get("po_number"))
    system = prompts.decision_system(tools.manifest()["invoice_business_rules"])
    mock = ({"decision": state["mock"]["decision"], "justification": "offline test mode"}
            if state.get("mock") else None)
    parsed, call = call_llm("decide", system,
                            prompts.decision_prompt(record, purchase_order), state["seed"], mock)
    parsed = parsed or {}
    return {"purchase_order": purchase_order,
            "decision": parsed.get("decision"),
            "justification": parsed.get("justification"),
            "calls": [call.to_dict()]}


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(RunState)
    graph.add_node("extract", extract_node)
    graph.add_node("gate", gate_node)
    graph.add_node("normalise", normalise_node)
    graph.add_node("decide", decide_node)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "gate")
    graph.add_conditional_edges("gate", route_after_gate,
                                {"extract": "extract", "normalise": "normalise"})
    graph.add_edge("normalise", "decide")
    graph.add_edge("decide", END)
    return graph.compile()


_GRAPH = None


def run_workflow(task_id: str, condition: str, trial: int, document_text: str,
                 mock: dict | None = None) -> dict:
    """Execute one run and return the final state with summary measurements."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    if condition not in GATES:
        raise ValueError(f"No verification mechanism registered for condition {condition}")

    initial: RunState = {
        "task_id": task_id, "condition": condition, "trial": trial,
        "seed": config.SEED_BASE + trial, "document_text": document_text,
        "mock": mock, "revisions": 0, "flagged": False, "human_seconds": 0.0,
        "calls": [], "gate_history": [],
    }
    started = time.perf_counter()
    final = _GRAPH.invoke(initial)
    wall = time.perf_counter() - started

    calls = final.get("calls", [])
    final["summary"] = {
        "wall_clock_s": round(wall, 3),
        "model_calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "thinking_tokens": sum(c["thinking_tokens"] for c in calls),
        "token_cost_usd": round(sum(c["cost_usd"] for c in calls), 8),
        "api_retries": sum(c["attempt_retries"] for c in calls),
        "parse_failures": sum(1 for c in calls if not c["parse_ok"]),
        "revisions": final.get("revisions", 0),
        "flagged": final.get("flagged", False),
        "human_seconds": round(final.get("human_seconds", 0.0), 3),
    }
    return final
