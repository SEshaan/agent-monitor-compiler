"""strmer: a JEV + TAC streaming middleman.

Importing the package does not initialize the model client or compile policies.
The main inference entry point is loaded only when requested.
"""

__all__ = ["run_D1_streamer"]


def __getattr__(name):
    if name == "run_D1_streamer":
        from .inference import run_D1_streamer

        return run_D1_streamer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
