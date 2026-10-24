# agent-loop

Lead → worker → check → reviewer loop with a 3-rejection cap, built on LangGraph
and a local Ollama model (`gemma3:4b`).

- `lead` writes a short plan.
- `worker` writes a solution (code in a fenced block).
- `check` extracts that code and runs it in a subprocess against fixed asserts,
  storing `tests_passed` / `test_output`.
- `reviewer` sees the task, solution, and real test results. A model "APPROVED"
  cannot override a failing run — `route` requires `tests_passed` to finish.
- After 3 rejections the graph pauses at `ask_tai`. A reply that is *exactly*
  `stop` or `accept` (case and trailing punctuation ignored) ends the run and
  records an `outcome`; anything longer is guidance, resets the counter, and
  sends it back for another lap.

## Run

```powershell
.\.venv\Scripts\python.exe loop.py                      # normal task, interactive owner call
.\.venv\Scripts\python.exe loop.py --cap --resume stop   # force the cap, then end as owner
.\.venv\Scripts\python.exe loop.py --max-pauses 1        # pause once and exit
.\.venv\Scripts\python.exe _smoke_test.py                # offline checks, no Ollama needed
```

`_smoke_test.py` is also pytest-ready (`def test_*()` functions), so `pytest`
will collect it in CI.

Every run prints token usage per node (with call counts) from
`response.usage_metadata`, summed across retries.

## The `check` node is not a sandbox

`check` runs model-written code via `subprocess` (`python -I`), and `-I` only
isolates Python's import paths — that code still runs as your user with access
to your files, environment, and network. Fine for a toy like `is_palindrome`;
before giving it real tasks, run `check` inside WSL or Docker, or at minimum in
a throwaway temp directory under a restricted user.
