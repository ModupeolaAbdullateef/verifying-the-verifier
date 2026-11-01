"""
runner.py - Single entry point for executing a run of either task.

Invoice identifiers (DOC-...) are executed by the invoice workflow and ticket
identifiers (TKT-...) by the ticket-triage workflow. The batch runner, the
single-run script and the review tools call these functions so that both
tasks are handled identically.
"""

from __future__ import annotations

from vtv import config
from vtv.traces import (document_images, document_text, ground_truth, is_ticket,
                        ticket_text, ticket_truth)


def _mock(task_id: str):
    if not config.MOCK_MODE:
        return None
    return ticket_truth(task_id) if is_ticket(task_id) else ground_truth(task_id)


def run(task_id: str, condition: str, trial: int) -> dict:
    """Execute one automated run (C0, C1 or C2) and return its final state."""
    if is_ticket(task_id):
        from vtv.ticket_workflow import run_ticket_workflow
        return run_ticket_workflow(task_id, condition, trial, ticket_text(task_id), _mock(task_id))
    from vtv.workflow import run_workflow
    return run_workflow(task_id, condition, trial, document_text(task_id), _mock(task_id),
                        document_images(task_id))


def start_human(task_id: str, trial: int) -> tuple[dict, float]:
    """Run a C3 workflow up to the human checkpoint."""
    if is_ticket(task_id):
        from vtv.ticket_workflow import start_ticket_human_run
        return start_ticket_human_run(task_id, trial, ticket_text(task_id), _mock(task_id))
    from vtv.workflow import start_human_run
    return start_human_run(task_id, trial, document_text(task_id), _mock(task_id),
                           document_images(task_id))


def resume_human(task_id: str, trial: int, review: dict, machine_seconds: float) -> dict:
    """Resume a suspended C3 run with the reviewer's response and finish it."""
    if is_ticket(task_id):
        from vtv.ticket_workflow import resume_ticket_human_run
        return resume_ticket_human_run(task_id, trial, review, machine_seconds)
    from vtv.workflow import resume_human_run
    return resume_human_run(task_id, trial, review, machine_seconds)


def human_state(task_id: str, trial: int) -> dict:
    """The stored state of a suspended or completed C3 run."""
    from vtv.workflow import thread_id
    if is_ticket(task_id):
        from vtv.ticket_workflow import human_graph
        graph = human_graph()
    else:
        from vtv.workflow import _human_graph
        graph = _human_graph()
    state = graph.get_state({"configurable": {"thread_id": thread_id(task_id, trial)}})
    return state.values or {}
