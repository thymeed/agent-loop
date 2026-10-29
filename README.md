# agent-loop

Lead → worker → check → reviewer loop with a 3-rejection cap, built on LangGraph
and a local Ollama model (`gemma3:4b`).

- `lead` writes a short plan.
- `worker` writes a solution (code in a fenced block).
- `check` extracts that code and runs it in a Docker sandbox (see below) against fixed
  asserts, storing `tests_passed` / `test_output`.
- `reviewer` sees the task, solution, and real test results. A model "APPROVED"
  cannot override a failing run — `route` requires `tests_passed` to finish.
- After 3 rejections the graph pauses at `ask_tai`. A reply that is *exactly*
  `stop` or `accept` (case and trailing punctuation ignored) ends the run and
  records an `outcome`; anything longer is guidance, resets the counter, and
  sends it back for another lap.

## Run

```powershell
.\.venv\Scripts\python.exe loop.py                      # built-in palindrome task, interactive owner call
.\.venv\Scripts\python.exe loop.py tasks/fizzbuzz.yaml   # your own task (see below)
.\.venv\Scripts\python.exe loop.py --cap --resume stop   # force the cap, then end as owner
.\.venv\Scripts\python.exe loop.py --max-pauses 1        # pause once and exit
.\.venv\Scripts\python.exe _smoke_test.py                # offline checks, no Ollama needed
```

`_smoke_test.py` is also pytest-ready (`def test_*()` functions), so `pytest`
will collect it in CI.

Every run prints token usage per node (with call counts) from
`response.usage_metadata`, summed across retries.

## Your own tasks

A task is a YAML file with a description and Python asserts. The worker's code is
placed above the asserts, so they call the function by name:

```yaml
task: Write fizzbuzz(n) in Python that returns a list of strings for 1..n ...
tests:
  - 'assert fizzbuzz(0) == []'
  - 'assert fizzbuzz(15)[-1] == "FizzBuzz"'
```

See `tasks/fizzbuzz.yaml`. `--cap` only works with the built-in task.

## Research mode

`type: research` tasks answer from recent arXiv papers instead of writing code
(`tasks/ssm_moe.yaml`). `query` is passed to arXiv verbatim (quoted phrases,
`AND`/`OR`, `abs:`/`ti:` prefixes); `max_papers` is 1-30, newest first.

```yaml
type: research
task: What open problems do these papers identify for SSMs and MoEs?
query: '(abs:"state space model" OR abs:"mamba") AND abs:"mixture of experts"'
```

Flow: `lead` → `search` (fetch papers) → `worker` (answers only from them) →
`check` → `reviewer`. The `check` node is deterministic: every cited `[arXiv:ID]`
must be a fetched paper, quoted spans of 4+ words must appear verbatim in the
abstract of a paper cited in the same paragraph (`...` / `…` split a quote into
fragments), and every cited paper needs at least one verified quote. A model
"APPROVED" cannot override a failed check, same as for code.

What this does and does not prove: citations and quotes are real. Whether the
conclusion is right is still a judgment. Only abstracts are used. Small local
models often paraphrase instead of quoting and will hit the rejection cap; use a
stronger `worker` model in `config.yaml`. Research mode needs network access for
arXiv; it runs no model-written code, so the Docker sandbox is not involved.

## The `check` sandbox

By default (`sandbox: docker` in `config.yaml`) `check` runs model-written code in
a throwaway container: no network, read-only filesystem, 16 MB tmpfs, memory/CPU/pid
limits, all capabilities dropped, non-root user, script passed over stdin (nothing
mounted). On timeout the container is killed by name. If Docker is missing the
check fails closed with an install hint.

`sandbox: local` runs a plain `python -I` subprocess instead. `-I` only isolates
import paths: that code runs as your user with your files, environment and
network. Use it only for trusted toy tasks.

The image is pulled on first use; to pre-pull: `docker pull python:3.12-slim`.
The sandbox is exercised against real Docker by `_smoke_test.py` whenever the
daemon is running (skipped otherwise).

## Layout

- `loop.py` — entry point.
- `config.yaml` — per-role models (`provider:model`), rejection cap, test timeout.
- `agent_loop/` — `state.py`, `checks.py`, `nodes.py`, `graph.py`, `cli.py`, `config.py`.
