# agent-loop

Lead → worker → check → reviewer loop with a 3-rejection cap, built on LangGraph
and local Ollama models: `gemma3:4b` plans, `qwen2.5-coder:7b` writes and reviews, and
`qwen3:8b` gets one escalation attempt at the cap (set per role in `config.yaml`).
Three kinds of task share the loop: **code** (a function checked by asserts), **project**
(changes to a real directory, checked by its own pytest) and **research** (answers checked
against fetched arXiv abstracts). Model output is never trusted: everything it writes runs in
a locked-down Docker sandbox, and deterministic checks, not the models, decide what passes.

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
.\.venv\Scripts\python.exe -m pytest _smoke_test.py -q   # tests: fake models, no Ollama; Docker tests run if Docker is up
node jstests\logic.test.js                               # the dashboard page's logic (needs Node)
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
(`qwen3:8b`) gets one attempt, with the same prompt and the same `check`, before the owner is
asked. Its tokens show on their own `escalate` row. An owner reply that is guidance resets the
cap and earns another escalation. Delete the `escalation:` line to turn it off; pull the model
first with `ollama pull qwen3:8b`, and it should differ from the worker to be worth anything.

Its thinking is **off** (`model_options: escalation: reasoning: false`), which follows from the
experiment below. When a thinking model does think, its reasoning is stripped from the reply
before anything is parsed, so draft file blocks inside it can never reach the workspace; the
dashboard still shows the full text.

### Escalation findings (2026-10-29)

Question: is `qwen3:8b`'s thinking worth its cost as the escalation model? Three variants ran one
after another on the three tasks that actually reach escalation (`csvrow`, `slugify`, `intervals`),
5 attempts each (15 per variant), `--budget 40000`. The rule was fixed before running: adopt a
variant only if its pass count is at most 2 below the control's **and** its mean seconds are at
least 30% lower.

| variant | passes | mean seconds | mean tokens | escalated, then rescued | rule |
|---|---|---|---|---|---|
| thinking default (control) | 10/15 | 90.4 | 7,257 | 5, then 1 | |
| thinking off | 9/15 | 30.0 | 5,081 | 7, then 2 | adopted |
| thinking on, capped at 4,096 tokens | 7/15 | 62.3 | 6,869 | 8, then 0 | not adopted: 3 below |

What it shows, and what it does not:

- The saving is almost all one task. `csvrow` escalations took 233 s with default thinking and
  7 s with it off, and none rescued anything in any variant: 0 of 15 `csvrow` attempts passed
  (0 of 3 in the earlier baseline too). That task is beyond both models here.
- On `slugify` and `intervals`, thinking off rescued as many attempts as the other settings
  (2, 1 and 0 rescues for off, default and capped). Those counts are tiny: no evidence that
  thinking helps, not proof that it does not.
- Thinking off's missing pass is one attempt lost to an Ollama abort ("token repeat limit
  reached") before escalation was even reached; the control lost one the same way. Such aborts
  are now retried once.
- Only 10 attempts per variant can tell them apart (`csvrow` is 0 for all), so one flipped
  attempt is 10 points. Read this as "about three times faster at no visible cost", not as a
  ranking of the settings.
- The control ran after the reasoning-stripping fix, so it is not the setup of the earlier
  baseline: compare variants with each other, not with the baseline.

Reports are in `runs/evals/` (`control`, `think-off`, `think-capped`) and the patches in
`experiments/`. `experiments/think-default.yaml` re-creates the old control now that thinking is
off in `config.yaml`.

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
- Tests run with `pytest` (`python -I -m pytest`, pinned in the image) in the Docker sandbox:
  workspace mounted read-only, no network. Packages in `requirements:` are installed while the
  image is built, never at test time, and they are code that runs inside the image. `pytest` and
  its dependencies (`pluggy`, `iniconfig`, `packaging`, `pygments`) are reserved there (any
  spelling is refused), since the image holds them fixed. Only `pytest` is pinned to a version;
  its dependencies are frozen at the versions pip resolved right after installing it, and the
  requirements install is held to those, so a requirement that needs another one fails the build
  rather than replacing it. The build context is made under `.scratch/`, never the system temp.
  The build also runs `sandbox/verify_pytest.py`: it records pytest's files right after that
  install and fails the build if a requirement, by accident, overwrites pytest's or its
  dependencies' files, or adds a file to their own package or dist-info folders (pip overwrites
  silently). A new file at the top level of site-packages is not seen, including a new top-level
  module and its `.pyc` in the shared `__pycache__`. It is NOT a defence against a hostile
  package, which can run code in every test process through a `.pth` file (even under `-I`) and
  can shadow pytest from the interpreter's stdlib directory. Only `pytest` commands get the `-I`
  treatment; another `test_command` still runs, but without that protection.
- **The worker cannot edit the tests.** This exists because it happened: in live runs the worker
  rewrote the tests it was being graded on, so a "pass" meant nothing. Do not loosen it to save a
  retry. `test_*.py`, `tests/`, `conftest.py`, `pytest.ini`, `pyproject.toml` and similar are
  protected: edits to them are ignored (the rest of the reply still counts) and checked again by
  content before every test run. Add globs with `protected: [...]`, or turn it off with
  `allow_test_edits: true` for tasks that need new tests.
- The result is `runs/<thread>/changes.patch`. Review it, then `git apply` it yourself
  from the project directory.
- Limits: Python and pytest only; small models often fail the whole-file format on
  large files; the worker sees about `max_context_chars` of the project per prompt.

## Evals

Measure instead of guessing. `evals/` holds six fixed tasks with deterministic tests: two code
tasks and four small projects. Ollama must be running with the configured models pulled (the
harness checks first).

```powershell
.\.venv\Scripts\python.exe -m agent_loop.evals --attempts 3 --label baseline --budget 40000
.\.venv\Scripts\python.exe -m agent_loop.evals --compare runs\evals\<a>.json runs\evals\<b>.json
```

- Each attempt runs without a terminal (the owner call at the cap is `stop`), in its own thread,
  and the report `runs/evals/<label>-<timestamp>.json` is saved after every attempt, with a trace
  per attempt beside it. A failed attempt is recorded and the rest continue.
- `--only a,b` picks tasks, `--attempts N` repeats each, `--budget TOKENS` stops an attempt at the
  first node boundary after its total passes that number.
- **Experiments are small patches.** `AGENT_LOOP_CONFIG` names a YAML file layered over
  `config.yaml` (models and `model_options` merge key by key), so it holds only the difference.
  `experiments/` has `control` (the current `config.yaml`), `think-default` (the escalation
  model's own thinking), `think-off` and `think-capped`. In PowerShell:
  `$env:AGENT_LOOP_CONFIG = "experiments\think-off.yaml"`. A patch cannot turn the sandbox off or
  use an unknown option. Per-role `model_options` allow `reasoning`, `num_predict` and
  `temperature` only; a thinking model's reasoning is stripped from its answer before use.
  A patch may also set `prompt_variants` (`real_path_example`, `no_plan`), which switch on prompt
  changes for experiments; an unknown name is an error, and the default is none.
- **Fixtures are frozen on purpose.** `evals/projects/slugify` is a copy of `projects/slugify_demo`
  so editing the demo cannot change eval numbers. `evals/FIXTURES.sha256` pins everything under
  `evals/`; the test suite fails if a fixture changes until you run
  `python -m agent_loop.fixtures --update`, because earlier reports are no longer comparable after
  a change.
- Read the numbers with care: at 3-5 attempts per task, one flipped attempt moves a task by 20-33
  points. Treat differences of that size as noise and repeat before trusting them.

## Dashboard

See what the models are doing, live, and answer a paused run from the page:

```powershell
.\.venv\Scripts\python.exe -m agent_loop.dashboard    # prints a URL with a one-time token
```

Open that URL in your browser. Press `?` on the page for the keyboard shortcuts.

**Runs.** Search and filter by status. For one run you get a timeline of every model call
(full prompt and reply, tokens, seconds), the tests' output, reviews, pauses and the final
result, with filters (model calls, node results, pauses, only failures) and a search box.
*Charts* shows time and tokens per role and node and a call-by-call strip. *Patch* shows the
change as a unified or side-by-side diff. *Export report* downloads the run as Markdown.
Eval attempts are hidden from the list by default (use the "eval attempts" chip).

**Evals.** Every report with its pass rate, per-task bars, a trend across reports, every
attempt, and a comparison of any two ticked reports.

**Deep links.** `#run=<id>`, `#run=<id>&view=charts` (or `patch`), `#eval=<report>.json`.

**Answering a paused run.** When a run stops at the rejection cap and waits for you, the page
shows the last review and test output with **Stop run**, **Accept anyway** (a second click
confirms, and says if the tests failed) and a box for guidance. Your answer is used exactly
like a line typed in the terminal (the exact-word `stop` / `accept` rule is unchanged), and
whichever of the terminal or the page answers first wins. Evals never wait.

### What the dashboard can and cannot do

It is read-only with one exception, the owner reply:

- Listens on `127.0.0.1` only (no option to change that); the port cannot be shared with
  another program; every request needs the random per-launch token; unexpected `Host`
  headers are refused; no CORS headers are ever sent.
- The page makes no outside requests, loads no outside code, has a strict content-security
  policy (no inline script or style), and shows model output as plain text. Tests fail if the
  page code uses `innerHTML` and similar, references a remote address, or has more than the
  single write below.
- **The one write.** `POST /api/owner_reply` leaves a small file, `runs/<thread>/owner_reply.json`.
  It is accepted only when ALL of these hold: login cookie, exact `Origin`, the per-launch CSRF
  header, a JSON body of at most 4 KB, a run that is paused right now with that pause's random
  nonce, reply text of 1-2000 characters with no control characters, no reply already waiting,
  and at most one attempt per second. The run uses the reply once and deletes it. A reply for
  another pause is discarded.
- It cannot start or stop runs, run code, change `config.yaml`, or write anywhere else.
- It serves only files under `runs/`, with names checked against a strict pattern.

Runs write `runs/<thread>/trace.jsonl` (prompts and replies, key-shaped strings redacted).
Those files contain your project code, so they stay local and gitignored. Runs made before
tracing existed have no trace; they still appear under Evals. The page refreshes by polling
every second or two. Stop the dashboard with Ctrl+C.

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
8. **Project code cannot replace pytest.** The test image pins pytest, puts the project on
   `sys.path` after the standard library (`work.pth`), occupies the `sitecustomize` name, and
   runs `python -I -m pytest`, so a worker-written `pytest.py` or `sitecustomize.py` cannot load
   first and force a pass. Those names, `_pytest/` and `*.pth` are also protected. A test plants
   exactly those files in a real container, checks the verdict is still the real one, and checks
   the same files DO fool the old command, so the test can fail. The image tag includes the
   Dockerfile and its verifier, so editing either rebuilds the image instead of reusing a stale one.

Known limit of item 8: a hostile requirement can still shadow pytest from the interpreter's stdlib
directory, or run code in every test through a `.pth` file. The build check catches accidents only.

What this does not give you: a container shares a kernel with its host (Docker
Desktop's WSL2 VM, here), so an escape is unlikely but not impossible. The image
build step has network access (that is where `requirements:` are installed). Stronger
isolation is outside this repo: run everything in a dedicated WSL2 distro with the
Windows drives and interop turned off, ideally with rootless Docker. The code under
test runs in the same process as pytest and can patch it, so a pass is evidence only
for code you trust enough to run; the sandbox and protected tests limit this but do
not remove it.

## Layout

- `loop.py` — entry point.
- `config.yaml` — per-role models (`provider:model`), per-role `model_options`, caps, timeouts, sandbox.
- `agent_loop/` — the graph and its nodes (`state.py`, `nodes.py`, `graph.py`, `cli.py`), the checks
  (`checks.py`, `research.py`, `sandbox.py`, `workspace.py`), `config.py`, `tasks.py`, run records and
  traces (`runs.py`, `trace.py`, `mailbox.py`), evals (`evals.py`, `fixtures.py`) and the dashboard
  (`dashboard.py` with `web/`).
- `tasks/`, `projects/` — example tasks and the demo project. `evals/` — frozen eval fixtures.
  `experiments/` — config patches for evals. `sandbox/Dockerfile` — the project test image, and
  `sandbox/verify_pytest.py` — snapshots pytest's files after its install and checks them at the
  end of the build.
- `_smoke_test.py`, `jstests/` — Python and page-logic tests.
