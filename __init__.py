"""strmer: a JEV + TAC streaming middleman.

The package is split into focused submodules (config, tac_compiler, tac_vm,
policy_set, jev_monitor, inference). Importing the package itself does not spin
up the OpenAI client or compile policies; those side effects only happen when
you actually run something via ``python -m strmer``.
"""

import sys


def __getattr__(name):
    # Lazy re-export so `from strmer import run_jev_streamer` works without
    # eagerly building the OpenAI client / compiling policies at import time.
    if name == "run_jev_streamer":
        from .inference import run_jev_streamer

        return run_jev_streamer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_jev_streamer"]
