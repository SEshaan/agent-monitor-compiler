"""Run a small end-to-end MCP proof of concept and append results to JSON.

Run from the repository root with its virtual environment:

    .venv/Scripts/python.exe poc_test_client.py

The client starts ``strmer.mcp_server`` over stdio, exercises policy validation,
three model-backed termination paths, and history retrieval. Every call is
persisted to ``poc_runs.json`` as it completes so partial sessions survive a
later failure.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPO = Path(__file__).resolve().parent
WORKSPACE = REPO.parent
OUTPUT = REPO / "poc_runs.json"

CASES = [
    {
        "name": "halt",
        "prompt": "Reply with exactly MCP_POC_HALT. Do not explain.",
        "expressions": "iteration_count >= 1 HALT",
        "expected_status": "halt",
    },
    {
        "name": "change_strategy",
        "prompt": "Reply with exactly MCP_POC_STRATEGY. Do not explain.",
        "expressions": "iteration_count >= 1 CHANGE_STRATEGY",
        "expected_status": "change_strategy",
    },
    {
        "name": "natural_completion",
        "prompt": "Reply with exactly MCP_POC_COMPLETE. Do not explain.",
        "expressions": "iteration_count >= 100 HALT",
        "expected_status": "complete",
    },
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_log():
    if OUTPUT.exists():
        return json.loads(OUTPUT.read_text(encoding="utf-8"))
    return {"schema_version": 1, "sessions": []}


def save_log(log):
    log["last_updated_utc"] = utc_now()
    OUTPUT.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unpack_result(result):
    contents = []
    parsed = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text is None:
            contents.append({"type": getattr(item, "type", "unknown")})
            continue
        contents.append({"type": getattr(item, "type", "text"), "text": text})
        try:
            parsed.append(json.loads(text))
        except (TypeError, json.JSONDecodeError):
            pass
    return {
        "is_error": result.isError,
        "parsed": parsed[0] if len(parsed) == 1 else parsed,
        "content": contents,
    }


async def main():
    log = load_log()
    session_record = {
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "started_at_utc": utc_now(),
        "transport": "stdio",
        "server_command": [sys.executable, "-m", "strmer.mcp_server"],
        "calls": [],
        "history_snapshot": None,
    }
    log["sessions"].append(session_record)
    save_log(log)

    child_env = os.environ.copy()
    child_env.setdefault("ANALYZE_EVERY", "1024")
    child_env.setdefault("MAX_TOTAL_TOKENS", "1024")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "strmer.mcp_server"],
        cwd=str(WORKSPACE),
        env=child_env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            tool_list = await client.list_tools()
            session_record["discovered_tools"] = [tool.name for tool in tool_list.tools]
            save_log(log)

            calls = [
                (
                    "validate_policy",
                    {"expressions": "iteration_count >= 1 HALT"},
                )
            ]
            calls.extend(
                ("run_generation", {"prompt": case["prompt"], "expressions": case["expressions"]})
                for case in CASES
            )

            for index, (name, arguments) in enumerate(calls):
                started = utc_now()
                tick = perf_counter()
                call_record = {
                    "tool": name,
                    "case": CASES[index - 1]["name"] if index > 0 else "policy_validation",
                    "arguments": arguments,
                    "started_at_utc": started,
                }
                if index == 0:
                    call_record["expected"] = {"valid": True}
                else:
                    call_record["expected"] = {"status": CASES[index - 1]["expected_status"]}
                try:
                    result = await asyncio.wait_for(
                        client.call_tool(name, arguments), timeout=180
                    )
                    call_record["result"] = unpack_result(result)
                    parsed = call_record["result"]["parsed"]
                    expected_key, expected_value = next(iter(call_record["expected"].items()))
                    call_record["passed"] = (
                        not result.isError
                        and isinstance(parsed, dict)
                        and parsed.get(expected_key) == expected_value
                    )
                except Exception as exc:  # retain failures as useful PoC evidence
                    call_record["passed"] = False
                    call_record["exception"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                call_record["duration_seconds"] = round(perf_counter() - tick, 3)
                call_record["finished_at_utc"] = utc_now()
                session_record["calls"].append(call_record)
                save_log(log)
                print(
                    f"{name} ({call_record['case']}): "
                    f"{call_record.get('result', call_record.get('exception'))}",
                    flush=True,
                )

            try:
                history_result = await asyncio.wait_for(
                    client.call_tool("get_history", {"limit": 100}), timeout=30
                )
                session_record["history_snapshot"] = unpack_result(history_result)
            except Exception as exc:
                session_record["history_exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            session_record["finished_at_utc"] = utc_now()
            save_log(log)

    print(f"Saved MCP proof-of-concept log: {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
