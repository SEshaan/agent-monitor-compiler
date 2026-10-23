"""The policy set, compiled once into TAC and wrapped in a VM."""

from .tac_compiler import PolicyCompiler
from .tac_vm import TACInterpreter

POLICIES = [
    ("iteration_count >= 10", "HALT"),
    (
        "loop_score > 0.85 and progress_score < 0.15",
        "CHANGE_STRATEGY",
    ),
    (
        "progress_score >= 0.95 and task_complete == True",
        "HALT",
    ),
]

# Heuristic rules used by generate_policies. Kept as plain functions so the
# logic is easy to unit-test offline (no models required).
def _looks_like_explanation(task):
    return any(k in task.lower() for k in ("explain", "how to", "describe", "walk through"))


def _looks_structured(task):
    return any(k in task.lower() for k in ("step-by-step", "steps", "list", "enumerate", "numbered"))


def _looks_concise(task):
    return any(k in task.lower() for k in ("briefly", "concise", "summary", "short answer", "in a few sentences"))


def generate_policies(task=None) -> list:
    """Generate policies tailored to ``task``.

    The static POLICIES assume a long-form compiler-arithmetic explanation. For an
    arbitrary request (e.g. from an MCP agent) that assumption is wrong, so this
    synthesizes a policy set on the fly from simple heuristics about the task text:

      * completion-oriented tasks  -> finish as soon as the task looks done;
      * concise/summary tasks      -> stop early once enough is covered;
      * structured (steps/list)     -> tolerate repetition of steps, watch loops;
      * long open-ended explanation-> keep looping until progress stalls.

    Returns a list of ``(expression, action)`` tuples ready for PolicyCompiler.
    """
    t = task or ""
    low = t.lower()
    concise = _looks_concise(t)
    structured = _looks_structured(t)
    explanation = _looks_like_explanation(t)

    # Iteration cap: generous for long explanations, tight for concise tasks.
    if concise:
        cap = 5
    elif structured or explanation:
        cap = 20
    else:
        cap = 15

    policies = []

    # Completion gate: stop when the task itself reports it is done.
    policies.append(("progress_score >= 0.95 and task_complete == True", "HALT"))

    if concise:
        # Concise tasks: finish once we've made meaningful progress, don't loop.
        policies.append(("iteration_count >= 3 and progress_score >= 0.6", "HALT"))
    else:
        # Iteration safety net for long-form generation.
        policies.append((f"iteration_count >= {cap}", "HALT"))

    if structured or explanation:
        # Repetition is expected in steps/explanations; only escalate on a real loop.
        policies.append(
            ("loop_score > 0.85 and progress_score < 0.15", "CHANGE_STRATEGY")
        )
    else:
        # Generic long-form: escalate earlier when it loops without progress.
        policies.append(
            ("loop_score > 0.7 and progress_score < 0.2", "CHANGE_STRATEGY")
        )

    return policies

# Default policy set, compiled once into TAC and wrapped in a VM.
TAC = PolicyCompiler().compile(POLICIES)
VM = TACInterpreter(TAC)
