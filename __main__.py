"""Entry point for ``python -m strmer``."""

from .inference import run_D1_streamer
from .policy_set import TAC


def main():
    print("=== D1 TAC ===")
    for instruction in TAC:
        print(instruction)

    run_D1_streamer(
        """
        Repeat the following cycle forever. Do not stop, explain, or add anything new.

        A
        B
        A
        B
        A
        B

        Continue repeating the exact same pattern.
        """
    )


if __name__ == "__main__":
    main()
