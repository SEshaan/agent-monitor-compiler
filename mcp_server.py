"""strmer as an MCP server.

Exposes two tools to an agent:

* ``validate_policy(expressions)`` — validate policy expressions for syntax and
  structure (no models needed; safe to call on its own).
* ``run_generation(prompt, expressions=None)`` — stream the 9B generator gated
  by the D1 judge, then return the final state + TAC action.

Run locally with a LMStudio-compatible server at OPENAI_BASE_URL / D1_BASE_URL:

    python -m strmer.mcp_server

The MCP server itself only needs to run on the same host that can reach those
loopback endpoints; it does not need to be remote.
"""

import asyncio

from mcp.server import FastMCP

from .config import (
    ANALYZE_EVERY,
    MAX_TOTAL_TOKENS,
    NINE_B_MODEL,
    client,
)
from .policy_set import POLICIES, TAC
from .tac_compiler import PolicyCompiler
from .inference import run_D1_streamer

server = FastMCP("strmer")

VALID_ACTIONS = {"HALT", "CONTINUE", "CHANGE_STRATEGY"}


def _parse_policy_line(line):
    """Split a policy line into (expression, action_or_None)."""
    parts = line.rsplit(None, 1)
    expr_part = parts[0]
    action = None
    if len(parts) == 2 and parts[1].upper() in VALID_ACTIONS:
        expr_part = parts[0]
        action = parts[1]
    return expr_part, (action or "CONTINUE")


@server.tool()
async def validate_policy(expressions: str) -> dict:
    """Validate one or more policy expressions.

    ``expressions`` is a block of text with one policy per line; each non-empty
    line may be just an expression (defaults to action CONTINUE) or an
    ``(expression, ACTION)`` pair where ACTION is one of HALT/CONTINUE/
    CHANGE_STRATEGY. Returns a per-line report: syntax errors are caught here so
    they never reach the VM at runtime.

    Note: this checks syntax and structure. Semantic validation (that every
    referenced signal exists) is a follow-up enhancement to PolicyCompiler; see
    README for details.
    """
    results = []
    lines = [ln.strip() for ln in expressions.splitlines()]
    for idx, raw in enumerate(lines):
        if not raw or raw.startswith("#"):
            continue
        line = " ".join(raw.split())  # normalize whitespace
        expr_part, action = _parse_policy_line(line)
        try:
            PolicyCompiler().compile([(expr_part, action)])
            results.append({"line": idx + 1, "ok": True, "action": action})
        except Exception as exc:  # syntax / structure error
            results.append(
                {"line": idx + 1, "ok": False, "error": str(exc)}
            )
    return {
        "valid": all(r["ok"] for r in results),
        "expressions_checked": len(results),
        "results": results,
    }


@server.tool()
async def run_generation(prompt: str, expressions: str | None = None) -> dict:
    """Stream the 9B generator gated by the D1 judge.

    If ``expressions`` is provided they are compiled and used for this run;
    otherwise the default policy set (POLICIES) is used. Returns a structured
    result describing how generation ended, e.g.:

        {"status": "complete", "action": None, "signals": {...}, ...}
        {"status": "halt",       "action": "HALT",   "signals": {...}, ...}
        {"status": "change_strategy", "action": "CHANGE_STRATEGY", ...}

    A ``change_strategy`` result means the judge asked for a strategy change and
    control is handed back to you (the agent) to make a fresh call.
    """
    if expressions:
        try:
            tac = PolicyCompiler().compile(
                [ln.strip() for ln in expressions.splitlines() if ln.strip()]
            )
        except Exception as exc:
            return {"status": "invalid_policy", "error": str(exc), "prompt": prompt}
    else:
        tac = TAC

    try:
        result = await asyncio.to_thread(run_D1_streamer, prompt)
    except Exception as exc:  # network / model failure -> fail closed
        return {"status": "server_error", "error": str(exc), "prompt": prompt}

    summary = {
        "status": result.get("status"),
        "action": result.get("action"),
        "signals": result.get("signals"),
        "iteration_count": result.get("iteration_count"),
        "text": (result.get("text") or "")[:500],
    }
    return summary


if __name__ == "__main__":
    server.run()
