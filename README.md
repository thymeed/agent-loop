# agent-loop

Lead → worker → check → reviewer loop with a 3-rejection cap, built on LangGraph
and local Ollama models: `gemma3:4b` plans, `qwen2.5-coder:7b` writes and reviews
(set per role in `config.yaml`).

- `lead` writes a short plan.
- `worker` writes a solution (code in a fenced block).
- `check` extracts that code and runs it in a Docker sandbox (see below) against fixed
  asserts, storing `tests_passed` / `test_output`.
- `reviewer` sees the task, solution, and real test results. A model "APPROVED"
  cannot override a failing run — `route` requires `tests_passed` to finish.
- After 3 rejections an optional stronger model gets one attempt (see Escalation),
  then the graph pauses at `ask_tai`. A reply that is *exactly*
  `stop` or `accept` (case and trailing punctuation ignored) ends the run and
  records an `outcome`; anything longer is guidance, resets the counter, and
  sends it back for another lap.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Needs Ollama (with the models above pulled) and Docker running.

## Run

```powershell
.\.venv\Scripts\python.exe loop.py                      # built-in palindrome task, interactive owner call
.\.venv\Scripts\python.exe loop.py tasks/fizzbuzz.yaml   # your own task (see below)
.\.venv\Scripts\python.exe loop.py --cap --resume stop   # force the cap, then end as owner
.\.venv\Scripts\python.exe loop.py --max-pauses 1        # pause once and exit
.\.venv\Scripts\python.exe _smoke_test.py                # offline checks, no Ollama needed
```

More than one task, or a directory, runs them one after another and prints a summary:

```powershell
.\.venv\Scripts\python.exe loop.py tasks --resume stop
```

## Saved runs and resuming

Everything lands in `runs/` (gitignored). `runs/checkpoints.sqlite` holds the graph
state after every node, and `runs/<thread>.json` is the final record of a run: result,
answer, review, test output and the token table. The thread id defaults to
`<task>-<timestamp>`; set your own with `--thread`.

If a run is killed, or stopped with `--max-pauses`, pick it up later, even from a new
process (it asks the owner call again if it was paused at one):

```powershell
.\.venv\Scripts\python.exe loop.py --continue <thread>
```

## Escalation

When a task hits the rejection cap, the model under `escalation:` in `config.yaml`
gets one attempt, with the same prompt and the same
`check`, before the owner is asked. Its tokens show on their own `escalate` row. An
owner reply that is guidance resets the cap and earns another escalation. Delete the
`escalation:` line to turn it off. It ships commented out: pull a model that differs
from the worker (`ollama pull qwen3:8b`) and uncomment it.

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

## Project tasks

`type: project` points the loop at a real directory and gates the result on the
project's own tests (`tasks/fix_slugify.yaml`, demo in `projects/slugify_demo/`):

```yaml
type: project
task: Make slugify() fold accents and collapse repeated dashes.
project: ../projects/slugify_demo   # relative to the task file
test_command: pytest -q             # default
requirements: ['six==1.17.0']       # optional; installed at image build time only
```

- Your project is never written. It is copied to `runs/<thread>/workspace`; the worker
  replies with whole files (` ```path/to/file.py ` blocks), which are applied only if the
  path is relative and inside the project (no `..`, drive letters, symlinks or `.git`).
  A bad path fails the lap; nothing from that reply is tested.
- Tests run with `pytest` in the Docker sandbox: workspace mounted read-only, no network.
  Packages in `requirements:` are installed while the image is built, never at test time.
- **The worker cannot edit the tests.** `test_*.py`, `tests/`, `conftest.py`, `pytest.ini`,
  `pyproject.toml` and similar are protected: edits to them are ignored (the rest of the
  reply still counts) and checked again by content before every test run. Add globs with
  `protected: [...]`, or turn it off with `allow_test_edits: true` for tasks that need
  new tests.
- The result is `runs/<thread>/changes.patch`. Review it, then `git apply` it yourself
  from the project directory.
- Limits: Python and pytest only; small models often fail the whole-file format on
  large files; the worker sees about `max_context_chars` of the project per prompt.

## Dashboard

See what the models are doing, live:

```powershell
.\.venv\Scripts\python.exe -m agent_loop.dashboard    # prints a URL with a one-time token
```

Open that URL in your browser. **Runs** lists every traced run and shows, as it happens,
each model call with its full prompt and reply, tokens and seconds, the tests' output,
reviews, pauses and the final patch. **Evals** shows eval reports (including one still
running), per-attempt detail and a side-by-side comparison of any two.

It is read-only and local: it listens on `127.0.0.1` only (there is no option to change
that), requires the random token, rejects unexpected `Host` headers, answers `GET` only,
sends no CORS headers, serves a page whose content-security-policy forbids inline script,
makes no outside requests, and shows model output as plain text (a test fails if the page
code ever uses `innerHTML` and friends). It serves only files under `runs/`.

Runs write `runs/<thread>/trace.jsonl` (prompts and replies, with key-shaped strings
redacted). Those files contain your project code, so they stay local and gitignored.
Runs made before this feature have no trace; they still appear under Evals.

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

`sandbox: local` runs a plain `python -I` subprocess instead. That code runs as your
user with your files, environment and network, so the config refuses to load unless
you also set the environment variable `AGENT_LOOP_ALLOW_LOCAL=1`. Project tasks never
run locally. The offline test suite sets it internally and only ever runs fixed code.

The image is pinned by digest in `config.yaml`. To refresh it on purpose:
`docker pull python:3.12-slim`, then copy the digest from
`docker image inspect python:3.12-slim` into `sandbox_image`.
The sandbox is exercised against real Docker by `_smoke_test.py` whenever the
daemon is running (skipped otherwise, so a green run without Docker proves less).

## Safety

Model-written code is the threat. These layers apply on every run:

1. **Container.** No network or IPC, read-only root, a small `noexec` tmpfs, no
   swap, memory/CPU/process/file limits, all capabilities dropped,
   `no-new-privileges`, non-root user, no image pull at run time, nothing mounted
   except (project tasks) the workspace copy, read-only.
2. **Self-check.** Before the first model code of a process runs, a probe runs inside
   the real container and verifies root, capabilities, `no-new-privileges`, network,
   root-filesystem writes, visible host paths and cgroup limits. If anything is off
   the run stops and says the model code was not run.
3. **Pinned image.** `sandbox_image` is a digest, not a tag.
4. **No silent unisolated mode.** `sandbox: local` needs `AGENT_LOOP_ALLOW_LOCAL=1`.
5. **Your project is a copy.** Files that look like credentials (`.env*`, `*.pem`,
   `*.key`, `id_rsa*`, `.npmrc`, ...) are not copied, and a project containing a
   key-shaped string or private key block is refused. Worker file paths are validated
   (no `..`, absolute paths, drive letters, symlinks, `.git`), and the tests are
   read-only to the worker.
6. **Plain names.** Thread ids (`--thread`, `--continue`) become folders and mount
   paths, so they allow only letters, digits, `.`, `_` and `-`.
7. **Bounded.** A rejection cap, one escalation per cap, test timeouts, and
   `--budget` tokens per eval attempt.

What this does not give you: a container shares a kernel with its host (Docker
Desktop's WSL2 VM, here), so an escape is unlikely but not impossible. The image
build step has network access (that is where `requirements:` are installed). Stronger
isolation is outside this repo: run everything in a dedicated WSL2 distro with the
Windows drives and interop turned off, ideally with rootless Docker.

## Layout

- `loop.py` — entry point.
- `config.yaml` — per-role models (`provider:model`), rejection cap, test timeout.
- `agent_loop/` — `state.py`, `checks.py`, `nodes.py`, `graph.py`, `cli.py`, `config.py`.
