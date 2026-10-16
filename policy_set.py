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

TAC = PolicyCompiler().compile(POLICIES)
VM = TACInterpreter(TAC)
