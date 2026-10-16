"""Shared configuration and the OpenAI client used by the 9B model."""

import os

from openai import OpenAI

# ---------------- CONFIG ----------------

ANALYZE_EVERY = 200
MAX_TOTAL_TOKENS = 4096
JEV_TIMEOUT = 30  # seconds

LMSTUDIO_URL = "http://127.0.0.1:1234/v1"
NINE_B_MODEL = "qwen/qwen3.5-0.8b"

JEV_ENDPOINT = "http://127.0.0.1:1234/v1/systemone"
JEV_MODEL = "d1-3b"
JEV_API_KEY = os.environ.get("JEV_API_KEY")  # optional; sent as a Bearer token

# The OpenAI client is only for the 9B model. JEV is called with plain HTTP,
# because the OpenAI SDK has no `decisions` API.
client = OpenAI(
    base_url=LMSTUDIO_URL,
    api_key="lm-studio",
)
