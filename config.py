"""Shared configuration and the OpenAI client used by the 9B model.

Every endpoint, model, and key is overridable through environment variables so
you can plug in any OpenAI-compatible server (LMStudio, a self-hosted vLLM /
Ollama instance, Azure, etc.) without editing code. LMStudio defaults keep the
local workflow working out of the box.

Environment variables
----------------------
OPENAI_BASE_URL      chat endpoint for the 9B generator   (default: http://127.0.0.1:1234/v1)
NINE_B_MODEL         model name sent to OPENAI_BASE_URL    (default: qwen3.5-0.8b)
OPENAI_API_KEY       bearer token for the chat endpoint    (optional; "lm-studio" if unset)

D1_BASE_URL          judge ("System One") endpoint         (default: http://127.0.0.1:1234/v1/systemone)
D1_MODEL             model name sent to D1_BASE_URL        (default: d1-3b)
D1_API_KEY           bearer token for the judge endpoint   (optional; empty if unset)

ANALYZE_EVERY        tokens generated per analysis checkpoint  (default: 200)
MAX_TOTAL_TOKENS     hard cap on total tokens requested        (default: 4096)
D1_TIMEOUT           timeout in seconds for a judge request    (default: 30)
"""

import os

from openai import OpenAI

# ---------------- CONFIG ----------------

ANALYZE_EVERY = int(os.environ.get("ANALYZE_EVERY", "200"))
MAX_TOTAL_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", "4096"))
D1_TIMEOUT = int(os.environ.get("D1_TIMEOUT", "30"))

# Generator (9B) — any OpenAI-compatible chat endpoint.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
NINE_B_MODEL = os.environ.get("NINE_B_MODEL", "qwen3.5-0.8b")

# Judge ("System One") — any OpenAI-compatible decisions endpoint.
D1_BASE_URL = os.environ.get("D1_BASE_URL", "http://127.0.0.1:1234/v1/systemone")
D1_ENDPOINT = D1_BASE_URL
D1_MODEL = os.environ.get("D1_MODEL", "d1-omni-600m")
D1_API_KEY = os.environ.get("D1_API_KEY", "")  # optional; sent as a Bearer token

# The OpenAI client is only for the 9B generator. JEV/D1 are called with plain
# HTTP POST, because the SDK has no `decisions` API.
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
)
