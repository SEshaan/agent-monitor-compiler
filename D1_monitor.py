"""D1 (System One) monitor: rubrics, answer extraction, and scoring."""

import requests

from .config import D1_API_KEY, D1_ENDPOINT, D1_MODEL, D1_TIMEOUT

# System One "score" questions use an ordered rubric: `criteria` is a list,
# and the API answers with a number between 0 and len(criteria) - 1
# (it can fall between levels). We rescale that to 0.0-1.0 for the policies.
LOOP_LEVELS = [
    "No repetition; the text keeps moving forward",
    "Slight repetition of phrases or ideas",
    "Noticeable repetition with little new content",
    "Heavy repetition; mostly restating earlier text",
    "Severe loop; essentially the same text again",
]

PROGRESS_LEVELS = [
    "No progress toward the requested task",
    "Minimal progress; mostly preamble or off-topic",
    "Some progress; part of the task is addressed",
    "Substantial progress; most of the task is addressed",
    "Task fully addressed",
]


def get_answer(result, name, expected_type):
    """Return one answer from a System One response, or raise ValueError."""
    answers = result.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("D1 response has no 'answers' object")

    answer = answers.get(name)
    if not isinstance(answer, dict):
        raise ValueError(f"D1 response is missing answer '{name}'")

    if answer.get("type") == "refusal":
        raise ValueError(f"D1 refused question '{name}'")

    if answer.get("type") != expected_type or expected_type not in answer:
        raise ValueError(
            f"D1 answer '{name}' is not a usable '{expected_type}' answer: "
            f"{answer}"
        )

    return answer[expected_type]


def rescale(score, levels):
    """Map a score in [0, len(levels) - 1] onto 0.0-1.0."""
    top = len(levels) - 1
    return max(0.0, min(1.0, float(score) / top))


def analyze_with_D1(text, previous_text, iteration):
    payload = {
        "model": D1_MODEL,
        "state": {
            "recent_output": text,
            "previous_output": previous_text or "(none; this is the first segment)",
            "iteration": iteration,
        },
        "questions": {
            "loop_score": {
                "type": "score",
                "instructions": (
                    "How repetitive is recent_output compared with "
                    "previous_output, without adding anything new?"
                ),
                "criteria": LOOP_LEVELS,
            },
            "progress_score": {
                "type": "score",
                "instructions": (
                    "How much meaningful progress does recent_output make "
                    "toward explaining how a compiler translates "
                    "arithmetic expressions?"
                ),
                "criteria": PROGRESS_LEVELS,
            },
            # "noul" answers with the probability (0-1) that the reply is yes.
            "task_complete": {
                "type": "noul",
                "instructions": (
                    "Is the requested task fully complete, so that no more "
                    "output is needed?"
                ),
                "criteria": {
                    "true": "The task is fully complete; no more output is needed",
                    "false": "More work is required to finish the task",
                },
            },
        },
    }

    headers = {"Content-Type": "application/json"}
    if D1_API_KEY:
        headers["Authorization"] = f"Bearer {D1_API_KEY}"

    response = requests.post(
        D1_ENDPOINT, json=payload, headers=headers, timeout=D1_TIMEOUT
    )
    if not response.ok:
        # Surface the server's explanation (e.g. which field it rejected).
        raise requests.HTTPError(
            f"{response.status_code} from {D1_ENDPOINT}: {response.text[:1000]}",
            response=response,
        )
    result = response.json()

    print(f"\n[D1 raw response #{iteration}] {result}")

    return {
        "loop_score": rescale(
            get_answer(result, "loop_score", "score"), LOOP_LEVELS
        ),
        "progress_score": rescale(
            get_answer(result, "progress_score", "score"), PROGRESS_LEVELS
        ),
        "task_complete": float(get_answer(result, "task_complete", "noul")) >= 0.5,
        "iteration_count": iteration,
    }
