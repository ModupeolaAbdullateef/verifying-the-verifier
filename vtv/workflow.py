"""
workflow.py - The invoice-processing multi-agent workflow, built with LangGraph.

    START -> extract -> gate --(issues found, revision allowed)--> extract
                            \\--(passed, or revision used)------> normalise -> decide -> END

The extractor, normaliser and decision agents are identical in every
experimental condition. The only component that changes is the verification
mechanism applied at the gate (see mechanisms.py):

    C0  unverified baseline          C2  checker agent
    C1  schema and constraints       C3  human-in-the-loop checkpoint

In Experiment 2 (controlled fault injection, trials 201-224) the first
extraction is replaced by a pre-generated record containing one seeded fault
(see faults.py), so that every condition verifies an identical input.

An automated mechanism that finds issues may send the extraction back once
for revision (config.MAX_REVISIONS); if issues remain, the record proceeds
and the run is flagged. In C3 the workflow is suspended at the gate with a
LangGraph interrupt, its state is persisted by a SQLite checkpointer, and it
is resumed once the human review has been recorded.
"""

from __future__ import annotations

import operator
import sqlite3
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from vtv import config, faults, prompts, tools
from vtv.llm import call_llm
from vtv.mechanisms import MECHANISMS, GateResult

GATES = dict(MECHANISMS)


class RunState(TypedDict, total=False):
    """Shared state passed between the nodes of one workflow run."""
    task_id: str
    condition: str
    trial: int
    seed: int
    document_text: str
    image_paths: list[str]            # scanned page images (input mode "image")
    mock: dict | None                 # ground truth, used only in offline test mode
    experiment: str                   # "natural" (Experiment 1) or "fault" (Experiment 2)
    fault: dict | None                # seeded fault description (Experiment 2 only)
    seeded_extraction: dict | None    # record presented to the gate in place of the first extraction
    initial_extraction: dict | None   # extractor output before any verification
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


def register_gate(condition: str, mechanism) -> None:
    """Add a verification mechanism under a condition code (used in testing)."""
    GATES[condition] = mechanism


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def extract_node(state: RunState) -> dict:
    is_revision = state.get("feedback") is not None
    if not is_revision and state.get("seeded_extraction") is not None:
        # Experiment 2: the gate receives the seeded record; no extraction call is made.
        seeded = state["seeded_extraction"]
        return {"extraction": seeded, "initial_extraction": seeded, "feedback": None, "calls": []}
    image_mode = config.INPUT_MODE == "image"
    prompt = prompts.extractor_prompt(
        None if image_mode else state["document_text"],
        feedback=state.get("feedback"),
        previous=state.get("extraction") if is_revision else None,
    )
    mock = state["mock"]["extraction"] if state.get("mock") else None
    parsed, record = call_llm("extract_revision" if is_revision else "extract",
                              prompts.EXTRACTOR_SYSTEM, prompt, state["seed"], mock,
                              images=state.get("image_paths") if image_mode else None)
    update = {"extraction": parsed, "feedback": None, "calls": [record.to_dict()]}
    if not is_revision:
        update["initial_extraction"] = parsed
    return update


def gate_node(state: RunState) -> dict:
    started = time.perf_counter()
    result: GateResult = GATES[state["condition"]](state)
    gate_seconds = time.perf_counter() - started

    update: dict = {
        "gate_passed": result.passed,
        "calls": [c.to_dict() for c in result.calls],
        "gate_history": [{
            "revision": state.get("revisions", 0),
            "passed": result.passed,
            "issues": result.issues,
            "action": result.action,
            "gate_seconds": round(gate_seconds, 4),
            "human_seconds": round(result.human_seconds, 3),
        }],
    }
    if result.corrected is not None:
        update["extraction"] = result.corrected
    if result.human_seconds:
        update["human_seconds"] = state.get("human_seconds", 0.0) + result.human_seconds

    revisions = state.get("revisions", 0)
    if not result.passed and revisions < config.MAX_REVISIONS:
        update["feedback"] = "\n".join(f"- {issue}" for issue in result.issues)
        update["revisions"] = revisions + 1
    else:
        update["flagged"] = not result.passed
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
# Graph construction
# --------------------------------------------------------------------------

def build_graph(checkpointer=None):
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
    return graph.compile(checkpointer=checkpointer)


_GRAPH = None          # automated conditions: no persistence needed
_HUMAN_GRAPH = None    # C3: state persisted between pause and resume


def _automated_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _human_graph():
    global _HUMAN_GRAPH
    if _HUMAN_GRAPH is None:
        from langgraph.checkpoint.sqlite import SqliteSaver
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(config.RESULTS_DIR / "checkpoints.sqlite",
                                     check_same_thread=False)
        _HUMAN_GRAPH = build_graph(checkpointer=SqliteSaver(connection))
    return _HUMAN_GRAPH


def _initial_state(task_id: str, condition: str, trial: int, document_text: str,
                   mock: dict | None, image_paths: list[str] | None = None) -> RunState:
    seeded = faults.fault_for(task_id, trial)
    return {"task_id": task_id, "condition": condition, "trial": trial,
            "seed": config.SEED_BASE + trial, "document_text": document_text,
            "image_paths": image_paths or [],
            "experiment": faults.experiment_for(trial),
            "fault": ({k: seeded[k] for k in ("fault_type", "variant", "description",
                                              "c1_visible", "scored_errors")}
                      if seeded else None),
            "seeded_extraction": seeded["record"] if seeded else None,
            "mock": mock, "revisions": 0, "flagged": False, "human_seconds": 0.0,
            "calls": [], "gate_history": []}


def summarise(final: dict, machine_seconds: float) -> dict:
    """Aggregate per-call measurements into the run-level summary."""
    calls = final.get("calls", [])
    human = final.get("human_seconds", 0.0)
    return {
        "machine_seconds": round(machine_seconds, 3),
        "human_seconds": round(human, 3),
        "wall_clock_s": round(machine_seconds + human, 3),
        "model_calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "thinking_tokens": sum(c["thinking_tokens"] for c in calls),
        "token_cost_usd": round(sum(c["cost_usd"] for c in calls), 8),
        "api_retries": sum(c["attempt_retries"] for c in calls),
        "parse_failures": sum(1 for c in calls if not c["parse_ok"]),
        "revisions": final.get("revisions", 0),
        "flagged": final.get("flagged", False),
    }


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def run_workflow(task_id: str, condition: str, trial: int, document_text: str,
                 mock: dict | None = None, image_paths: list[str] | None = None) -> dict:
    """Execute one complete run in an automated condition (C0, C1 or C2)."""
    if condition not in GATES:
        raise ValueError(f"No verification mechanism registered for condition {condition}")
    if condition == "C3":
        raise ValueError("C3 runs pause for review: use start_human_run and resume_human_run")
    started = time.perf_counter()
    final = _automated_graph().invoke(_initial_state(task_id, condition, trial,
                                                     document_text, mock, image_paths))
    final["summary"] = summarise(final, time.perf_counter() - started)
    return final


def thread_id(task_id: str, trial: int) -> str:
    return f"{task_id}_C3_t{trial}"


def start_human_run(task_id: str, trial: int, document_text: str,
                    mock: dict | None = None,
                    image_paths: list[str] | None = None) -> tuple[dict, float]:
    """
    Run a C3 workflow up to the human checkpoint.

    Returns (interrupt_payload, machine_seconds_so_far). The suspended state is
    stored in results/checkpoints.sqlite under thread_id(task_id, trial).
    """
    graph = _human_graph()
    run_config = {"configurable": {"thread_id": thread_id(task_id, trial)}}
    started = time.perf_counter()
    result = graph.invoke(_initial_state(task_id, "C3", trial, document_text, mock, image_paths),
                          run_config)
    seconds = time.perf_counter() - started
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        raise RuntimeError("C3 run finished without pausing for review")
    return interrupts[0].value, seconds


def resume_human_run(task_id: str, trial: int, review: dict,
                     machine_seconds_before: float) -> dict:
    """Resume a suspended C3 run with the reviewer's response and finish it."""
    from langgraph.types import Command

    graph = _human_graph()
    run_config = {"configurable": {"thread_id": thread_id(task_id, trial)}}
    started = time.perf_counter()
    final = graph.invoke(Command(resume=review), run_config)
    machine = machine_seconds_before + (time.perf_counter() - started)
    final["summary"] = summarise(final, machine)
    return final
