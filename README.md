# strmer — a JEV + TAC streaming middleman

`strmer` is an inference **middleman** that controls an autoregressive text
generator with a small, human-readable policy engine. It works in three
layers:

1. **Generate** — stream a 9B model (`qwen/qwen3.5-0.8b`) in bounded chunks.
2. **Judge** — after each chunk, ask a lightweight "System One" judge model
   (`d1-3b`) to score the segment for *looping*, *progress*, and *task
   completion*.
3. **Decide** — feed those signals into a compiled policy set (a tiny bytecode
   virtual machine) that deterministically emits one of three actions:
   `CONTINUE`, `HALT`, or `CHANGE_STRATEGY`.

The idea is to prevent runaway loops and off-topic drift by letting an external
judge veto the generator mid-stream, using policies you can read and edit as
plain strings.

## Architecture

```
prompt ──▶ [9B model, streamed in ANALYZE_EVERY-token windows]
                │  each window → judge scores it
                ▼
        System One (d1-3b) answers rubric questions:
            loop_score      ∈ [0, 1]   how repetitive vs. the previous window
            progress_score  ∈ [0, 1]   how much real progress was made
            task_complete   ∈ {0, 1}   is the task fully done?
            iteration_count = window index (auto-incremented)
                ▼
        signals ──▶ TAC VM ──▶ action ∈ {CONTINUE, HALT, CHANGE_STRATEGY}
```

### Modules

| File | Responsibility |
|------|----------------|
| `config.py` | Shared config + the OpenAI client used for the 9B model. |
| `tac_compiler.py` | Compiles policy strings (AST) into a tiny assembly-like instruction list. |
| `tac_vm.py` | Interprets that bytecode deterministically and returns an action. |
| `policy_set.py` | The default policy set, compiled once into TAC + wrapped in the VM. |
| `D1_monitor.py` | Judge model: rubrics, answer extraction, scoring (`analyze_with_D1`). |
| `inference.py` | Ties it together — streams the 9B model and gates it with judge signals. |
| `__main__.py` | Entry point for `python -m strmer`. |
| `__init__.py` | Lazy re-export of `run_D1_streamer` (no side effects at import). |

### The policy language

Policies are `(expression, action)` pairs. Expressions support:

- **Arithmetic:** `+ - * /`
- **Comparison:** `< > <= >= == !=`
- **Boolean:** `and`, `or`
- **Signals:** `iteration_count`, `loop_score`, `progress_score`, `task_complete`

The default policy set (`policy_set.py`):

```python
POLICIES = [
    ("iteration_count >= 10", "HALT"),
    ("loop_score > 0.85 and progress_score < 0.15", "CHANGE_STRATEGY"),
    ("progress_score >= 0.95 and task_complete == True", "HALT"),
]
```

### Actions

| Action | Effect |
|--------|--------|
| `CONTINUE` | Keep streaming the next window. (default) |
| `HALT` | Stop generation immediately. |
| `CHANGE_STRATEGY` | Strategy change requested — currently stops; replanning is not implemented yet. |

## Running

Requires a local LMStudio server exposing the OpenAI-compatible chat API and
the System One judge endpoint at `http://127.0.0.1:1234`.

```bash
# install dependencies
pip install -r requirements.txt

# run the middleman (prompts the 9B model to explain a compiler)
python -m strmer
```

Optional environment variables:

| Variable | Purpose |
|----------|---------|
| `D1_API_KEY` | Bearer token for the judge endpoint, if required. |

## Tuning knobs (`config.py`)

| Constant | Default | Meaning |
|----------|---------|---------|
| `ANALYZE_EVERY` | 200 | Tokens generated per analysis checkpoint. |
| `MAX_TOTAL_TOKENS` | 4096 | Hard cap on total tokens requested. |
| `D1_TIMEOUT` | 30 | Timeout (seconds) for a judge request. |
| `NINE_B_MODEL` | `qwen/qwen3.5-0.8b` | Model used for generation. |
| `D1_MODEL` | `d1-3b` | Judge model. |
| `D1_ENDPOINT` | `http://127.0.0.1:1234/v1/systemone` | Judge endpoint. |

## Failing safe

Every stage fails closed: a judge error, a policy-evaluation error, or an
unexpected signal stops generation rather than letting it continue unchecked.
