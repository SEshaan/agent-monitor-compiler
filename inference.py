"""Inference middleman: stream the 9B model and gate it with D1 signals."""

import requests

from .config import ANALYZE_EVERY, MAX_TOTAL_TOKENS, NINE_B_MODEL, client
from .D1_monitor import analyze_with_D1
from .policy_set import VM


def run_D1_streamer(prompt):
    messages = [{"role": "user", "content": prompt}]
    previous_window = ""
    total_tokens_requested = 0
    iteration = 0

    while total_tokens_requested < MAX_TOTAL_TOKENS:
        budget = min(
            ANALYZE_EVERY,
            MAX_TOTAL_TOKENS - total_tokens_requested,
        )

        stream = client.chat.completions.create(
            model=NINE_B_MODEL,
            messages=messages,
            max_tokens=budget,
            stream=True,
        )

        window_parts = []
        finish_reason = None

        print("\n[9B] ", end="", flush=True)

        for chunk in stream:
            if not chunk.choices:  # e.g. trailing usage-only chunk
                continue

            choice = chunk.choices[0]
            delta = choice.delta.content or ""

            if delta:
                print(delta, end="", flush=True)
                window_parts.append(delta)

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        window = "".join(window_parts)

        if not window:
            print(
                "\n[Middleman] No text received; stopping. "
                "(If this is a thinking model, the token budget may have "
                "been spent on reasoning - disable thinking or raise "
                "ANALYZE_EVERY.)"
            )
            break

        iteration += 1
        total_tokens_requested += budget

        # Preserve the conversation between generation segments.
        messages.append({
            "role": "assistant",
            "content": window,
        })

        print(f"\n\n[Middleman] Analysis checkpoint #{iteration}")

        try:
            state = analyze_with_D1(
                window,
                previous_window,
                iteration,
            )
        except (requests.RequestException, ValueError, TypeError) as exc:
            print(f"[D1 error] {exc}")
            print("[Middleman] Failing closed: stopping generation.")
            break

        print(f"[Signals] {state}")

        # Deterministic control: D1 signals -> compiled TAC -> action.
        try:
            action = VM.run(state)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            print(f"[TAC error] {exc}")
            print("[Middleman] Stopping because policy evaluation failed.")
            break

        print(f"[TAC decision] {action}")

        if action == "HALT":
            print("[Middleman] Generation terminated by D1 policy.")
            break

        if action == "CHANGE_STRATEGY":
            print("[Middleman] Strategy change requested.")
            print("[Middleman] Stopping; replanning is not implemented.")
            break

        # A natural end from the 9B model also ends the run.
        if finish_reason == "stop":
            print("[Middleman] 9B model finished naturally.")
            break

        previous_window = window

        messages.append({
            "role": "user",
            "content": (
                "Continue the original task from where you left off. "
                "Do not repeat the previous segment."
            ),
        })

    print("\n[Middleman] Session ended.")
