"""Offline checks: owner stop/accept, guidance retries, lap counting, tokens.

Run with pytest (`python -m pytest _smoke_test.py -q`) or directly
(`python _smoke_test.py`).
"""
import atexit
import contextlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

# Keep every scratch file inside this project (gitignored) and delete it on exit. Left to
# the default, the suite scatters hundreds of folders through the system temp directory.
_REPO = pathlib.Path(__file__).resolve().parent
_SCRATCH = _REPO / ".scratch" / f"tests-{os.getpid()}"


def _force_remove(path):
    def writable(func, p, _exc):  # a read-only file would otherwise block the cleanup on Windows
        os.chmod(p, 0o700)
        func(p)
    shutil.rmtree(path, onexc=writable)


for _old in (_REPO / ".scratch").glob("tests-*"):  # leftovers of a crashed run, never a live one
    if _old != _SCRATCH and time.time() - _old.stat().st_mtime > 3600:
        _force_remove(_old)
_SCRATCH.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_SCRATCH)
atexit.register(lambda: _SCRATCH.exists() and _force_remove(_SCRATCH))

from langgraph.types import Command

from agent_loop import checks, cli, graph as graph_mod, nodes as agent_nodes, evals, research, sandbox, workspace
from agent_loop.tasks import Task, load_task

NL = chr(10)
checks.SANDBOX = "local"  # offline suite never needs Docker

GOOD = """```python
def is_palindrome(s):
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```"""

BROKEN = """```python
def is_palindrome(s):
    return True
```"""


class FakeLLM:
    """Says the same thing every call; reports fixed token usage."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, _msg):
        self.calls += 1
        return SimpleNamespace(
            content=self.reply,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def install(*, code=GOOD, review="APPROVED: fine", escalation=None):
    """Point the graph at fake models; returns the (worker, reviewer) fakes.

    Escalation is off unless `escalation` code is given; its fake is MODELS["escalation"].
    """
    agent_nodes.MODELS["lead"] = FakeLLM("1. Do it")
    agent_nodes.MODELS["worker"] = FakeLLM(code)
    agent_nodes.MODELS["reviewer"] = FakeLLM(review)
    if escalation is None:
        agent_nodes.MODELS.pop("escalation", None)
    else:
        agent_nodes.MODELS["escalation"] = FakeLLM(escalation)
    return agent_nodes.MODELS["worker"], agent_nodes.MODELS["reviewer"]


def drive(thread, *, resume=None, stop_after_pauses=None, tests=None):
    """Run the graph, resuming with `resume` each time it pauses."""
    cfg = {"configurable": {"thread_id": thread}}
    inputs = {
        "task": "is_palindrome",
        "rejections": 0,
        "tests": checks.IMPOSSIBLE_TESTS if tests is None else tests,
    }
    nodes = []
    while True:
        for step in graph_mod.graph.stream(inputs, cfg):
            for node in step:
                nodes.append(node)
        state = graph_mod.graph.get_state(cfg)
        if not state.next:
            return nodes, state
        if stop_after_pauses is not None and nodes.count("__interrupt__") >= stop_after_pauses:
            return nodes, state
        inputs = Command(resume=resume)


def test_extract_code_and_run_tests():
    code = checks.extract_code(GOOD)
    assert code.startswith("def is_palindrome"), code
    ok, out = checks.run_tests(code)
    assert ok and "ALL TESTS PASSED" in out, (ok, out)

    ok, out = checks.run_tests(checks.extract_code(BROKEN))
    assert not ok and "AssertionError" in out, (ok, out)

    ok, out = checks.run_tests("")
    assert not ok and "no Python code" in out, (ok, out)


def test_guidance_reply_retries():
    """A whole sentence must be treated as guidance, not a decision."""
    worker, _ = install()
    nodes, state = drive("guidance",
                         resume="Accept the assert for now; explain the reasoning",
                         stop_after_pauses=2)
    assert nodes.count("__interrupt__") == 2, nodes
    assert "ask_tai" in nodes, nodes
    assert state.values.get("outcome") is None, state.values
    assert worker.calls == 2 * agent_nodes.MAX_REJECTIONS, worker.calls


def test_whole_reply_stop_and_accept_end_run():
    cases = [
        ("stop", "stopped by owner"),
        ("Stop.", "stopped by owner"),
        ("  accept  ", "accepted by owner"),
        ("SHIP!", "accepted by owner"),
    ]
    for i, (cmd, expected) in enumerate(cases):
        thread = f"end-{i}"
        worker, _ = install()
        nodes, state = drive(thread, resume=cmd)
        assert nodes[-1] == "ask_tai", (cmd, nodes)
        assert state.values["outcome"].startswith(expected), (cmd, state.values)
        assert worker.calls == agent_nodes.MAX_REJECTIONS, (cmd, worker.calls)
        assert graph_mod.graph.get_state({"configurable": {"thread_id": thread}}).next == ()


def test_stop_does_not_add_a_lap():
    worker, reviewer = install()
    nodes, state = drive("stop-once", resume="stop")
    assert nodes.count("__interrupt__") == 1, nodes
    assert worker.calls == agent_nodes.MAX_REJECTIONS, worker.calls
    assert reviewer.calls == agent_nodes.MAX_REJECTIONS, reviewer.calls
    assert state.values["rejections"] == agent_nodes.MAX_REJECTIONS
    assert state.values["tokens"]["worker"]["calls"] == agent_nodes.MAX_REJECTIONS
    assert state.values["tokens"]["lead"]["calls"] == 1


def test_token_counts_sum_across_retries():
    worker, reviewer = install()
    laps = 2 * agent_nodes.MAX_REJECTIONS
    nodes, state = drive("retry", resume="more", stop_after_pauses=2)
    assert nodes.count("__interrupt__") == 2, nodes
    assert worker.calls == laps and reviewer.calls == laps
    assert state.values["tokens"]["worker"] == {
        "input": 10 * laps, "output": 5 * laps, "total": 15 * laps, "calls": laps,
    }, state.values["tokens"]


def test_cap_run_does_not_leak_tests_into_later_runs():
    before = list(checks.TESTS)
    install()
    stopped = cli.run("is_palindrome", tests=checks.IMPOSSIBLE_TESTS,
                       resume_with="stop", thread="leak")
    assert stopped["outcome"].startswith("stopped by owner")
    assert checks.TESTS == before, checks.TESTS

    install()
    passed = cli.run("is_palindrome", thread="leak-after")
    assert passed["approved"] is True and passed["tests_passed"] is True


def test_load_task_file():
    import pathlib, tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    ok = d / 'ok.yaml'
    ok.write_text(NL.join(['task: Add two numbers', 'tests:', "  - 'assert add(1, 2) == 3'"]), encoding='utf-8')
    assert load_task(str(ok)) == Task('Add two numbers', 'code', tests=['assert add(1, 2) == 3'])
    for name, body in [('empty.yaml', 'task: x' + NL + 'tests: []'),
                       ('notests.yaml', 'task: x'),
                       ('list.yaml', '- a')]:
        bad = d / name
        bad.write_text(body, encoding='utf-8')
        try:
            load_task(str(bad))
        except ValueError:
            continue
        raise AssertionError(name + ' should be rejected')

    # the shipped example is valid and a correct solution passes its asserts
    tests = load_task('tasks/fizzbuzz.yaml').tests
    code = ('def fizzbuzz(n):' + NL + '    return ["FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 '
            'else "Buzz" if i % 5 == 0 else str(i) for i in range(1, n + 1)]')
    passed, out = checks.run_tests(code, tests)
    assert passed, out


FEED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2312.00752v2</id>
    <published>2023-12-01T17:00:00Z</published>
    <title>Mamba:  Linear-Time Sequence
      Modeling</title>
    <summary>  Selective state spaces scale linearly in sequence length.
      Open problems remain for in-context recall.  </summary>
    <author><name>Albert Gu</name></author>
    <author><name>Tri Dao</name></author>
  </entry>
</feed>"""

PAPERS = research.parse_feed(FEED)


def test_parse_feed():
    assert len(PAPERS) == 1
    p = PAPERS[0]
    assert p["id"] == "2312.00752", p
    assert p["title"] == "Mamba: Linear-Time Sequence Modeling", p
    assert p["authors"] == ["Albert Gu", "Tri Dao"] and p["published"] == "2023-12-01"
    assert "Open problems remain for in-context recall." in p["abstract"], p


def test_citation_check():
    inline = 'Recall is open [arXiv:2312.00752v2 "Open problems remain for in-context recall"].'
    ok, out = research.check_citations(inline, PAPERS)
    assert ok and "1 CITED PAPERS" in out, out

    # a quote next to the citation instead of inside the bracket, with curly quotes
    prose = 'Selective state spaces [arXiv:2312.00752] note “scale linearly in sequence length”.'
    ok, out = research.check_citations(prose, PAPERS)
    assert ok, out

    cases = {
        "no citations at all": "no citations found",
        'x [arXiv:9999.99999] "scale linearly in sequence length"': "not one of the fetched papers",
        'x [arXiv:2312.00752] "SSMs beat transformers on everything"': "quote not found",
        'x [arXiv:2312.00752] "scale linearly"': "cited but no verified quote",
        "x [arXiv:2312.00752]": "cited but no verified quote",
        # right quote, wrong paragraph: it is not next to a citation of that paper
        'Quote: "scale linearly in sequence length".' + NL + NL + "Cites [arXiv:2312.00752].":
            "cited but no verified quote",
    }
    for answer, expected in cases.items():
        ok, out = research.check_citations(answer, PAPERS)
        assert not ok and expected in out, (answer, out)

    # ellipses and trailing commas inside the quote marks are tolerated, per fragment
    elided = 'Papers [arXiv:2312.00752] say "...Selective state spaces scale linearly ... for in-context recall,"'
    ok, out = research.check_citations(elided, PAPERS)
    assert ok, out

    mixed = inline + ' But [arXiv:2312.00752] "made up words that are not here".'
    ok, out = research.check_citations(mixed, PAPERS)
    assert not ok and "quote not found" in out, out


def test_research_task_validation():
    import pathlib, tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    ok = d / "ok.yaml"
    ok.write_text(NL.join(["type: research", "task: Survey SSMs", "query: ssm"]), encoding="utf-8")
    t = load_task(str(ok))
    assert (t.kind, t.query, t.max_papers, t.tests) == ("research", "ssm", 8, None), t
    for name, lines in {
        "noquery.yaml": ["type: research", "task: x"],
        "bigmax.yaml": ["type: research", "task: x", "query: q", "max_papers: 500"],
        "badtype.yaml": ["type: poem", "task: x"],
    }.items():
        bad = d / name
        bad.write_text(NL.join(lines), encoding="utf-8")
        try:
            load_task(str(bad))
        except ValueError:
            continue
        raise AssertionError(name + " should be rejected")
    shipped = load_task("tasks/ssm_moe.yaml")
    assert shipped.kind == "research" and shipped.query


def test_research_run_end_to_end_with_fakes():
    cited = 'Recall is open [arXiv:2312.00752] "Open problems remain for in-context recall".'
    fabricated = 'Transformers are dead [arXiv:2312.00752] "SSMs beat transformers on everything".'
    searched = []

    def fake_fetch(query, max_papers):
        searched.append((query, max_papers))
        return PAPERS

    with mock.patch.object(research, "fetch_papers", fake_fetch):
        install(code=cited, review="APPROVED: grounded")
        run = cli.run("Survey SSMs", kind="research", query="ssm", max_papers=3, thread="research-ok")
        assert run["approved"] and run["tests_passed"] and run["rejections"] == 0, run
        assert searched == [("ssm", 3)], searched

        # An approving reviewer cannot rescue a fabricated quote.
        worker, _ = install(code=fabricated, review="APPROVED: looks great")
        run = cli.run("Survey SSMs", kind="research", query="ssm",
                      resume_with="stop", thread="research-bad")
        assert not run["approved"] and run["outcome"].startswith("stopped by owner"), run
        assert "quote not found" in run["test_output"], run
        assert worker.calls == agent_nodes.MAX_REJECTIONS, worker.calls


def test_run_survives_restart_via_sqlite():
    import json, pathlib, tempfile
    from langgraph.checkpoint.sqlite import SqliteSaver
    d = pathlib.Path(tempfile.mkdtemp())
    db = str(d / "c.sqlite")

    install()
    with SqliteSaver.from_conn_string(db) as saver:
        app = graph_mod.compile_graph(saver)
        cli.run("is_palindrome", tests=checks.IMPOSSIBLE_TESTS, max_pauses=1,
                thread="t", app=app)

    # fresh connection, fresh graph, fresh fake models: everything must come off disk
    install()
    with SqliteSaver.from_conn_string(db) as saver:
        app = graph_mod.compile_graph(saver)
        done = cli.run(None, thread="t", app=app, continue_thread=True,
                       resume_with="stop", runs_dir=d / "runs")
        assert done["outcome"].startswith("stopped by owner"), done
        try:
            cli.run(None, thread="t", app=app, continue_thread=True)
        except SystemExit as exc:
            assert "nothing to resume" in str(exc), exc
        else:
            raise AssertionError("a finished thread must not resume")

    rec = json.loads((d / "runs" / "t.json").read_text(encoding="utf-8"))
    assert rec["result"] == "stopped by owner" and rec["rejections"] == agent_nodes.MAX_REJECTIONS, rec
    assert rec["tokens"]["lead"]["calls"] == 1, rec["tokens"]  # the first run's lead call survived
    assert rec["tokens"]["worker"]["calls"] == agent_nodes.MAX_REJECTIONS, rec["tokens"]


def test_run_record_labels_and_batch_expansion():
    import pathlib, tempfile
    from agent_loop import runs
    assert runs.result_label({"approved": True, "tests_passed": True}) == "approved"
    assert runs.result_label({"outcome": "accepted by owner despite failing checks: ship"}) == "accepted by owner despite failing checks"
    assert runs.result_label({"approved": False}) == "unfinished"
    assert runs.new_thread_id("my task!") .startswith("my-task-")

    d = pathlib.Path(tempfile.mkdtemp())
    (d / "b.yaml").write_text("x", encoding="utf-8")
    (d / "a.yaml").write_text("x", encoding="utf-8")
    (d / "skip.txt").write_text("x", encoding="utf-8")
    assert [pathlib.Path(f).name for f in cli.expand_task_files([str(d)])] == ["a.yaml", "b.yaml"]
    assert cli.expand_task_files(["tasks/fizzbuzz.yaml"]) == ["tasks/fizzbuzz.yaml"]
    try:
        cli.expand_task_files([str(tempfile.mkdtemp())])
    except SystemExit:
        pass
    else:
        raise AssertionError("an empty task directory must be an error")


def test_escalation_rescues_a_stuck_run():
    worker, _ = install(code=BROKEN, escalation=GOOD)
    nodes, state = drive("esc-win", tests=checks.TESTS)
    assert nodes.count("__interrupt__") == 0, nodes   # the owner was never bothered
    assert nodes.count("escalate") == 1 and nodes[-1] == "reviewer", nodes
    assert state.values["approved"] and state.values["tests_passed"], state.values
    assert worker.calls == agent_nodes.MAX_REJECTIONS, worker.calls
    tokens = state.values["tokens"]
    assert tokens["escalate"]["calls"] == 1 and tokens["worker"]["calls"] == agent_nodes.MAX_REJECTIONS, tokens


def test_escalation_runs_once_per_cap_then_asks_owner():
    worker, _ = install(code=BROKEN, escalation=BROKEN)
    esc = agent_nodes.MODELS["escalation"]
    nodes, state = drive("esc-lose", resume="stop", tests=checks.TESTS)
    assert nodes.count("escalate") == 1 and nodes.count("__interrupt__") == 1, nodes
    assert nodes.index("escalate") < nodes.index("__interrupt__"), nodes
    assert esc.calls == 1 and worker.calls == agent_nodes.MAX_REJECTIONS, (esc.calls, worker.calls)
    assert state.values["outcome"].startswith("stopped by owner"), state.values

    # guidance resets the cap, so the next cap earns the stronger model another attempt
    worker, _ = install(code=BROKEN, escalation=BROKEN)
    esc = agent_nodes.MODELS["escalation"]
    nodes, _ = drive("esc-guided", resume="try harder", stop_after_pauses=2, tests=checks.TESTS)
    assert nodes.count("escalate") == 2 and esc.calls == 2, (nodes, esc.calls)


def test_no_escalation_without_a_configured_model():
    worker, _ = install(code=BROKEN)
    nodes, _ = drive("no-esc", resume="stop", tests=checks.TESTS)
    assert "escalate" not in nodes, nodes


def _project(tmp, files):
    for name, text in files.items():
        f = tmp / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(text.encode("utf-8"))
    return tmp


def test_copy_project_skips_junk_and_refuses_symlinks_and_size():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    src = _project(root / "src", {"a.py": "x = 1\n", "pkg/b.py": "y = 2\n",
                                  ".git/config": "secret", "__pycache__/a.pyc": "junk",
                                  ".venv/lib.py": "junk"})
    dest = root / "ws"
    workspace.copy_project(src, dest, max_bytes=10_000)
    assert workspace.project_files(dest) == ["a.py", "pkg/b.py"], workspace.project_files(dest)
    assert not (dest / ".git").exists() and not (dest / ".venv").exists()
    for bad, kwargs in [(dest, {"max_bytes": 10_000}), (root / "big", {"max_bytes": 3})]:
        try:
            workspace.copy_project(src, bad, **kwargs)
        except workspace.WorkspaceError:
            continue
        raise AssertionError("expected WorkspaceError")


def test_apply_blocks_writes_files_and_rejects_hostile_paths():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    ws = _project(root / "ws", {"keep.py": "old\n", "sub/x.py": "x\n"})
    outside = root / "evil.py"
    fence = "`" * 3
    reply = NL.join([
        "Here you go.",
        fence + "pkg/new.py", "print('hi')", fence,
        fence + "keep.py", "new content", fence,
        fence + "../evil.py", "owned", fence,
        fence + "/etc/passwd", "owned", fence,
        fence + "C:/Windows/evil.py", "owned", fence,
        fence + "back" + chr(92) + "slash.py", "owned", fence,
        fence + ".git/hooks/pre-commit", "owned", fence,
        fence + "python", "print('a bare language fence is not a file')", fence,
    ])
    result = workspace.apply_blocks(ws, reply)
    assert result.written == ["pkg/new.py", "keep.py"], result
    assert len(result.errors) == 5, result.errors
    assert (ws / "keep.py").read_bytes() == b"new content\n"
    assert b"\r" not in (ws / "pkg" / "new.py").read_bytes(), "must not write CRLF"
    assert not outside.exists() and not (ws / ".git").exists()

    empty = workspace.apply_blocks(ws, "I refuse to write any blocks")
    assert not empty.written and "no file blocks" in empty.errors[0], empty

    too_many = NL.join(fence + f"f{i}.py" + NL + "x" + NL + fence for i in range(workspace.MAX_BLOCKS + 1))
    assert not workspace.apply_blocks(ws, too_many).written


def test_apply_blocks_rejects_symlink_escape():
    import os, pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    ws = _project(root / "ws", {"a.py": "x\n"})
    outside = root / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, ws / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        print("  (symlinks not permitted here - skipped)")
        return
    fence = "`" * 3
    result = workspace.apply_blocks(ws, fence + "link/pwn.py" + NL + "owned" + NL + fence)
    assert not result.written and result.errors, result
    assert not (outside / "pwn.py").exists()


def test_context_and_patch():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    orig = _project(root / "orig", {"app.py": "def f():\n    return 1\n", "test_app.py": "from app import f\n",
                                    "notes.bin": "binary-ish"})
    ws = root / "ws"
    workspace.copy_project(orig, ws, max_bytes=10_000)
    ctx = workspace.build_context(ws, budget=10_000)
    assert "### app.py" in ctx and "### test_app.py" in ctx and "notes.bin" not in ctx, ctx
    assert ctx.index("### test_app.py") < ctx.index("### app.py"), "tests come first"
    small = workspace.build_context(ws, budget=40)
    assert "not shown" in small, small

    fence = "`" * 3
    workspace.apply_blocks(ws, NL.join([fence + "app.py", "def f():", "    return 2", fence,
                                       fence + "new.py", "z = 1", fence]))
    patch = workspace.make_patch(orig, ws)
    assert "diff --git a/app.py b/app.py" in patch and "-    return 1" in patch and "+    return 2" in patch, patch
    assert "new file mode 100644" in patch and "+++ b/new.py" in patch, patch
    assert "test_app.py" not in patch
    # the patch is real git-apply input, and applies cleanly to the original
    import shutil, subprocess
    if shutil.which("git"):
        (root / "p.patch").write_bytes(patch.encode("utf-8"))
        check = subprocess.run(["git", "apply", "--check", str(root / "p.patch")], cwd=orig,
                               capture_output=True, text=True)
        assert check.returncode == 0, check.stderr


def test_no_personal_data_in_repo_files():
    """Privacy gate: no home-directory paths, emails or key-shaped strings in committed files."""
    import re, shutil, pathlib
    if not shutil.which("git"):
        return
    BS = chr(92)
    repo = pathlib.Path(__file__).resolve().parent
    files = (subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                            cwd=repo, capture_output=True, text=True).stdout.split(NL))
    patterns = {
        "home directory path": re.compile(
            "[A-Za-z]:" + BS * 2 + "Users" + BS * 2 + "|/Users/[A-Za-z]|/home/[a-z]"),
        "email address": re.compile(r"[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.[a-z]{2,}"),
        "key-shaped string": re.compile(r"sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"),
    }
    hits = []
    for name in filter(None, files):
        path = repo / name
        if not path.is_file() or path.suffix in {".pyc", ".sqlite"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, rx in patterns.items():
            if name == "_smoke_test.py":
                continue  # this file contains the patterns themselves
            for m in rx.finditer(text):
                hits.append(f"{name}: {label}: {m.group(0)!r}")
    assert not hits, hits


@contextlib.contextmanager
def _docker_mode(verified=True):
    """Pretend Docker is the sandbox. `verified` skips the one-per-process self-check probe,
    which tests that mock subprocess would otherwise have to answer."""
    with mock.patch.object(checks, "SANDBOX", "docker"), mock.patch.object(checks, "_VERIFIED", verified):
        yield


def test_requirement_validation_rejects_anything_but_plain_specifiers():
    ok = ["six==1.17.0", "requests[socks]>=2,<3", "numpy", "Flask~=3.0", "a.b-c_d != 1.0"]
    assert sandbox.validate_requirements(ok) == [r.strip() for r in ok]
    for bad in ["-r other.txt", "-e .", "--index-url http://evil", "https://evil/x.whl",
                "git+https://github.com/x/y", "pkg @ https://evil/x.whl", "pkg; python_version<'3'",
                "pkg==1.0 && rm -rf /", "pkg==$(id)", "", "./local", "pkg==1.0\nevil"]:
        try:
            sandbox.validate_requirements([bad])
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    a, b = sandbox.image_tag(["b==1", "a==2"]), sandbox.image_tag(["a==2", "b==1"])
    assert a == b and a != sandbox.image_tag(["a==2"]), "tag depends on the set, not the order"


def test_requirement_validation_rejects_reserved_pytest_names():
    assert sandbox.validate_requirements(["six==1.17.0", "pytest-cov", "pytest_mock>=3"])
    for bad in ["pytest==8.0", "PyTest>=8", "pytest[testing]==8", "Pluggy==1.0", "_pytest",
                "iniconfig==2.0", "Packaging>=20", "pygments", "pytest.", "pytest_", "Pluggy-==1",
                "pygments.._-"]:
        try:
            sandbox.validate_requirements([bad])
        except ValueError:
            continue
        raise AssertionError(f"accepted reserved {bad!r}")


def test_project_docker_command_is_locked_down_with_a_readonly_mount():
    import pathlib, tempfile
    ws = pathlib.Path(tempfile.mkdtemp())
    cmd = sandbox.project_docker_cmd("n1", ws, "img", "pytest -q tests")
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges",
                 "--pids-limit", "--memory", "--user 65534", "target=/work,readonly", "-w /work"):
        assert flag in joined, (flag, joined)
    assert cmd[cmd.index("img") + 1:] == ["python", "-I", "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", "tests"], cmd
    assert cmd.count("--mount") == 1 and "-v" not in cmd and "--volume" not in cmd
    assert "/var/run/docker.sock" not in joined
    try:
        sandbox.project_docker_cmd("n", ws, "img", "   ")
    except ValueError:
        pass
    else:
        raise AssertionError("empty test command must be rejected")


def test_project_tests_build_once_fail_closed_and_kill_on_timeout():
    import pathlib, tempfile
    ws = pathlib.Path(tempfile.mkdtemp())
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")      # not built yet
        if cmd[:2] == ["docker", "build"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["docker", "run"]:
            return SimpleNamespace(returncode=0, stdout="3 passed", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with _docker_mode(), mock.patch.object(sandbox.subprocess, "run", fake_run):
        ok, out = sandbox.run_project_tests(ws, "pytest", ["six==1.17.0"])
    assert ok and out == "3 passed", out
    kinds = [c[1] for c in calls]
    assert kinds == ["image", "build", "run"], kinds
    build = calls[1]
    base = f"BASE={checks.CONFIG['sandbox_image']}"
    assert base in build and "@sha256:" in base, build      # the sandbox base is pinned by digest
    assert build.index("-t") < len(build) - 1, build

    # hostile requirement never reaches docker
    calls.clear()
    with _docker_mode(), mock.patch.object(sandbox.subprocess, "run", fake_run):
        ok, out = sandbox.run_project_tests(ws, "pytest", ["--index-url http://evil"])
    assert not ok and "plain requirement" in out and not calls, (out, calls)

    # failed build fails closed, no run
    def failing_build(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0 if cmd[1] != "build" and cmd[1] != "image" else 1,
                               stdout="", stderr="boom")
    calls.clear()
    with _docker_mode(), mock.patch.object(sandbox.subprocess, "run", failing_build):
        ok, out = sandbox.run_project_tests(ws, "pytest", [])
    assert not ok and "build failed" in out and all(c[1] != "run" for c in calls), (out, calls)

    # timeout kills the container by name
    calls.clear()

    def hang(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 1)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    with _docker_mode(), mock.patch.object(sandbox.subprocess, "run", hang):
        ok, out = sandbox.run_project_tests(ws, "pytest", [])
    run = next(c for c in calls if c[:2] == ["docker", "run"])
    assert not ok and "timeout" in out and calls[-1] == ["docker", "kill", run[run.index("--name") + 1]], calls

    # local mode is refused for real projects; missing docker fails closed
    with mock.patch.object(checks, "SANDBOX", "local"):
        assert "need `sandbox: docker`" in sandbox.run_project_tests(ws, "pytest", [])[1]

    def no_docker(cmd, **kw):
        raise FileNotFoundError("docker")
    with _docker_mode(), mock.patch.object(sandbox.subprocess, "run", no_docker):
        ok, out = sandbox.run_project_tests(ws, "pytest", [])
    assert not ok and "install Docker" in out, out


def test_build_context_is_made_under_the_project_scratch():
    import pathlib
    seen = {}

    def fake_run(cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")      # not built yet
        ctx = pathlib.Path(cmd[-1])                                         # docker build ... <context>
        seen["ctx"] = ctx
        seen["present"] = [n for n in ("Dockerfile", "verify_pytest.py", "requirements.txt") if (ctx / n).is_file()]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with mock.patch.object(sandbox.subprocess, "run", fake_run):
        sandbox.ensure_image(["six==1.17.0"])
    assert seen["present"] == ["Dockerfile", "verify_pytest.py", "requirements.txt"], seen
    assert seen["ctx"].parent == sandbox.SCRATCH and sandbox.SCRATCH == _REPO / ".scratch", seen
    assert not seen["ctx"].exists(), "the build context is removed after the build"


class SeqLLM:
    """Replies in order, repeating the last one; remembers every prompt it was given."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def invoke(self, msg):
        self.prompts.append(msg)
        reply = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return SimpleNamespace(content=reply, usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                                             "total_tokens": 15})

    @property
    def calls(self):
        return len(self.prompts)


FENCE = "`" * 3
BUGGY = {"calc.py": "def add(a, b):" + NL + "    return a - b" + NL,
         "test_calc.py": "from calc import add" + NL + "def test_add():" + NL + "    assert add(2, 3) == 5" + NL}
FIX = FENCE + "calc.py" + NL + "def add(a, b):" + NL + "    return a + b" + NL + FENCE
HOSTILE = NL.join([FENCE + "../evil.py", "owned", FENCE, FENCE + "/abs/evil.py", "owned", FENCE])


def _fake_project_tests(calls):
    """Stands in for the Docker run: passes once calc.py adds instead of subtracts."""
    def run(ws, command, requirements):
        import pathlib
        calls.append((str(ws), command, requirements))
        fixed = "a + b" in (pathlib.Path(ws) / "calc.py").read_text(encoding="utf-8")
        return fixed, "1 passed" if fixed else "FAILED test_calc.py::test_add - assert -1 == 5"
    return run


def _project_run(root, *, worker, escalation=None, thread="proj", **kw):
    import pathlib
    orig = _project(pathlib.Path(root) / "orig", BUGGY)
    install()
    agent_nodes.MODELS["worker"] = worker
    if escalation is not None:
        agent_nodes.MODELS["escalation"] = escalation
    agent_nodes.MODELS["reviewer"] = FakeLLM("APPROVED: looks right")
    values = cli.run("Fix add()", kind="project", project=str(orig), thread=thread,
                     runs_dir=pathlib.Path(root) / "runs", **kw)
    return orig, values


def test_project_task_file_validation():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "proj").mkdir()

    def task(*lines):
        f = root / "t.yaml"
        f.write_text(NL.join(lines), encoding="utf-8")
        return str(f)

    t = load_task(task("type: project", "task: Fix it", "project: proj"))
    assert (t.kind, t.test_command, t.requirements) == ("project", "pytest -q", []), t
    assert pathlib.Path(t.project) == (root / "proj").resolve(), t.project  # relative to the task file
    t = load_task(task("type: project", "task: x", "project: proj", "test_command: pytest -x",
                       "requirements: ['six==1.17.0']"))
    assert t.test_command == "pytest -x" and t.requirements == ["six==1.17.0"], t
    for bad in [["type: project", "task: x"],
                ["type: project", "task: x", "project: nope"],
                ["type: project", "task: x", "project: proj", "requirements: ['-r evil.txt']"],
                ["type: project", "task: x", "project: proj", "requirements: six"],
                ["type: project", "task: x", "project: proj", "test_command: ''"]]:
        try:
            load_task(task(*bad))
        except ValueError:
            continue
        raise AssertionError(bad)


def test_project_run_rejects_hostile_blocks_then_fixes_and_leaves_original_alone():
    import json, pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    calls = []
    worker = SeqLLM([HOSTILE, FIX])
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _fake_project_tests(calls)):
        orig, values = _project_run(root, worker=worker)

    assert values["approved"] and values["tests_passed"] and values["rejections"] == 1, values
    assert len(calls) == 1, calls                      # the unappliable lap never reached the tests
    assert "could not be applied" in worker.prompts[1] and "../evil.py" in worker.prompts[1], worker.prompts[1]
    assert values["changed"] == ["calc.py"], values["changed"]
    assert not (root / "evil.py").exists()
    assert (orig / "calc.py").read_bytes() == BUGGY["calc.py"].encode(), "original must be untouched"

    patch_path = root / "runs" / "proj" / "changes.patch"
    patch = patch_path.read_bytes()
    assert b"-    return a - b" in patch and b"+    return a + b" in patch and b"\r" not in patch, patch
    check = subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=orig,
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr

    text = (root / "runs" / "proj.json").read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["patch"] == "proj/changes.patch" and record["changed"] == ["calc.py"], record
    assert "workspace" not in record and "project_dir" not in record
    assert str(root) not in text and str(root).replace(chr(92), "/") not in text, "no absolute paths in the record"


def test_project_escalation_goes_through_apply():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    calls = []
    wrong = FENCE + "calc.py" + NL + "def add(a, b):" + NL + "    return 0" + NL + FENCE
    worker, esc = SeqLLM([wrong]), SeqLLM([FIX])
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _fake_project_tests(calls)):
        orig, values = _project_run(root, worker=worker, escalation=esc)
    assert values["approved"] and esc.calls == 1 and worker.calls == agent_nodes.MAX_REJECTIONS, values
    assert values["tokens"]["escalate"]["calls"] == 1, values["tokens"]
    fixed = (root / "runs" / "proj" / "workspace" / "calc.py").read_text(encoding="utf-8")
    assert "a + b" in fixed, fixed
    install()  # drop the escalation fake so later tests stay offline


def test_project_prepare_is_idempotent_and_keeps_worker_changes():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    orig = _project(root / "orig", BUGGY)
    state = {"type": "project", "project_dir": str(orig), "workspace": str(root / "ws")}
    agent_nodes.prepare(state)
    (root / "ws" / "calc.py").write_text("changed by the worker" + NL, encoding="utf-8")
    agent_nodes.prepare(state)                                   # e.g. a resumed run
    assert (root / "ws" / "calc.py").read_text(encoding="utf-8").startswith("changed by the worker")
    assert not (root / "ws.partial").exists()
    assert agent_nodes.prepare({"type": "code"}) == {} and agent_nodes.apply({"type": "code"}) == {}


def test_project_hard_gate_survives_an_approving_reviewer():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    still_broken = FENCE + "calc.py" + NL + "def add(a, b):" + NL + "    return a * b" + NL + FENCE
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _fake_project_tests([])):
        _, values = _project_run(root, worker=SeqLLM([still_broken]), resume_with="stop")
    assert not values["approved"] and values["outcome"].startswith("stopped by owner"), values
    assert values["rejections"] == agent_nodes.MAX_REJECTIONS


def test_real_docker_project_run_end_to_end_with_fake_models():
    import pathlib, tempfile
    if not _require_docker():
        return
    root = pathlib.Path(tempfile.mkdtemp())
    with _docker_mode():
        orig, values = _project_run(root, worker=SeqLLM([HOSTILE, FIX]), thread="real")
    assert values["approved"] and values["tests_passed"] and values["rejections"] == 1, values
    assert "1 passed" in values["test_output"], values["test_output"]
    assert (orig / "calc.py").read_bytes() == BUGGY["calc.py"].encode()
    left = subprocess.run(["docker", "ps", "-a", "--filter", "name=agent-loop-", "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.split()
    assert not left, left


def test_protected_globs_match_tests_and_test_config_only():
    p = workspace.DEFAULT_PROTECTED
    for protected in ["test_calc.py", "pkg/test_calc.py", "calc_test.py", "conftest.py", "pkg/conftest.py",
                      "tests/test_x.py", "tests/helpers.py", "tests/sub/data.py", "pkg/tests/helpers.py",
                      "test/x.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
                      "TESTS/Test_X.py", "pkg/PYTEST.INI"]:
        assert workspace.is_protected(protected, p), protected
    for free in ["calc.py", "textutil/slug.py", "src/app.py", "latest.py", "contest.py", "docs/tests.md",
                 "pkg/attest.py"]:
        assert not workspace.is_protected(free, p), free
    assert not workspace.is_protected("tests/test_x.py", [])


def test_apply_blocks_refuses_protected_files_unless_allowed():
    import pathlib, tempfile
    ws = _project(pathlib.Path(tempfile.mkdtemp()) / "ws", dict(BUGGY))
    reply = NL.join(FENCE + name + NL + "x = 1" + NL + FENCE for name in
                    ["test_calc.py", "conftest.py", "tests/test_new.py", "pkg/tests/helpers.py",
                     "pytest.ini", "calc.py"])
    result = workspace.apply_blocks(ws, reply)
    assert result.written == ["calc.py"], result
    assert not result.errors, result.errors                       # not fatal: the lap still gets tested
    assert len(result.skipped) == 5 and all("protected" in e for e in result.skipped), result.skipped
    assert (ws / "test_calc.py").read_bytes() == BUGGY["test_calc.py"].encode()
    assert not (ws / "conftest.py").exists() and not (ws / "tests").exists()

    allowed = workspace.apply_blocks(ws, FENCE + "test_calc.py" + NL + "x = 2" + NL + FENCE, protected=[])
    assert allowed.written == ["test_calc.py"] and not allowed.errors, allowed


def test_protected_changes_catches_edit_delete_and_new_files():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    orig = _project(root / "orig", {**BUGGY, "tests/test_b.py": "def test_b():" + NL + "    pass" + NL})
    ws = root / "ws"
    workspace.copy_project(orig, ws, max_bytes=100_000)
    p = workspace.DEFAULT_PROTECTED
    assert workspace.protected_changes(orig, ws, p) == []
    (ws / "calc.py").write_text("changed" + NL, encoding="utf-8")            # unprotected: fine
    assert workspace.protected_changes(orig, ws, p) == []
    (ws / "test_calc.py").write_text("def test_add():" + NL + "    assert True" + NL, encoding="utf-8")
    (ws / "tests" / "test_b.py").unlink()
    (ws / "conftest.py").write_text("collect_ignore = ['test_calc.py']" + NL, encoding="utf-8")
    assert workspace.protected_changes(orig, ws, p) == ["conftest.py", "test_calc.py", "tests/test_b.py"]
    assert workspace.protected_changes(orig, ws, []) == []


def _runner_that_honours_the_tests(calls):
    """Passes when the code is fixed OR when the test file no longer demands the real answer."""
    def run(ws, command, requirements):
        import pathlib
        calls.append(str(ws))
        code = (pathlib.Path(ws) / "calc.py").read_text(encoding="utf-8")
        tests = (pathlib.Path(ws) / "test_calc.py").read_text(encoding="utf-8")
        ok = "a + b" in code or "assert add(2, 3) == 5" not in tests
        return ok, "1 passed" if ok else "FAILED test_calc.py::test_add"
    return run


def test_worker_cannot_pass_by_rewriting_the_tests():
    import pathlib, tempfile
    cheat = FENCE + "test_calc.py" + NL + "def test_add():" + NL + "    assert True" + NL + FENCE
    root = pathlib.Path(tempfile.mkdtemp())
    calls, worker = [], SeqLLM([cheat, FIX])
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _runner_that_honours_the_tests(calls)):
        orig, values = _project_run(root, worker=worker, thread="honest")
    # lap 1 is tested on the ORIGINAL tests (edit ignored) and fails; lap 2 fixes the code
    assert values["approved"] and values["rejections"] == 1 and len(calls) == 2, (values, calls)
    assert "protected" in worker.prompts[1], worker.prompts[1]               # the worker is told why
    ws = root / "runs" / "honest" / "workspace"
    assert (ws / "test_calc.py").read_bytes() == BUGGY["test_calc.py"].encode()
    assert "a + b" in (ws / "calc.py").read_text(encoding="utf-8")           # passed by fixing the code

    # The same cheat DOES pass when protection is off, so the test above measures something real.
    calls2 = []
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _runner_that_honours_the_tests(calls2)):
        _, values = _project_run(pathlib.Path(tempfile.mkdtemp()), worker=SeqLLM([cheat]),
                                 thread="cheat", protected=[])
    assert values["approved"] and values["rejections"] == 0, values


def test_a_good_fix_survives_a_forbidden_test_edit_in_the_same_reply():
    """Seen live: qwen3 fixed slug.py correctly but also tried to edit the tests."""
    import pathlib, tempfile
    cheat = FENCE + "test_calc.py" + NL + "def test_add():" + NL + "    assert True" + NL + FENCE
    root = pathlib.Path(tempfile.mkdtemp())
    calls = []
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _runner_that_honours_the_tests(calls)):
        orig, values = _project_run(root, worker=SeqLLM([FIX + NL + cheat]), thread="mixed")
    assert values["approved"] and values["rejections"] == 0 and len(calls) == 1, (values, calls)
    assert "ignored edits to protected files" in values["test_output"], values["test_output"]
    ws = root / "runs" / "mixed" / "workspace"
    assert (ws / "test_calc.py").read_bytes() == BUGGY["test_calc.py"].encode()

    # A hostile path in the same reply is still fatal: nothing from that lap is tested.
    calls.clear()
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _runner_that_honours_the_tests(calls)):
        _, values = _project_run(pathlib.Path(tempfile.mkdtemp()), worker=SeqLLM([FIX + NL + HOSTILE, FIX]),
                                 thread="hostile")
    assert values["rejections"] == 1 and len(calls) == 1, (values, calls)


def test_check_fails_closed_if_a_protected_file_slipped_through():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    orig = _project(root / "orig", BUGGY)
    ws = root / "ws"
    workspace.copy_project(orig, ws, max_bytes=100_000)
    (ws / "test_calc.py").write_text("def test_add():" + NL + "    assert True" + NL, encoding="utf-8")
    ran = []
    state = {"type": "project", "project_dir": str(orig), "workspace": str(ws), "apply_errors": []}
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _fake_project_tests(ran)):
        out = agent_nodes.check(state)
    assert out["tests_passed"] is False and "test_calc.py" in out["test_output"] and not ran, (out, ran)


def test_task_file_protection_keys():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "proj").mkdir()

    def task(*lines):
        f = root / "t.yaml"
        f.write_text(NL.join(["type: project", "task: x", "project: proj", *lines]), encoding="utf-8")
        return str(f)

    assert load_task(task()).protected == list(workspace.DEFAULT_PROTECTED)
    extended = load_task(task("protected: ['docs/*', 'data.json']")).protected
    assert extended == [*workspace.DEFAULT_PROTECTED, "docs/*", "data.json"], extended
    assert load_task(task("allow_test_edits: true")).protected == []
    for bad in (["protected: docs"], ["protected: [1]"], ["allow_test_edits: 'yes'"]):
        try:
            load_task(task(*bad))
        except ValueError:
            continue
        raise AssertionError(bad)


REFERENCE_SOLUTIONS = {
    "slugify": ("textutil/slug.py", '''"""Turn a title into a URL slug."""
import re
import unicodedata


def slugify(text):
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
'''),
    "wordwrap": ("wrapper/wrap.py", '''def wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines
'''),
    "intervals": ("ivl/merge.py", '''def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
'''),
    "csvrow": ("csvrow/parse.py", '''def parse_row(line):
    fields, cur, i, quoted = [], [], 0, False
    while i < len(line):
        c = line[i]
        if quoted:
            if c == '"':
                if line[i + 1:i + 2] == '"':
                    cur.append('"')
                    i += 1
                else:
                    quoted = False
            else:
                cur.append(c)
        elif c == '"':
            quoted = True
        elif c == ",":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    fields.append("".join(cur))
    return fields
'''),
}


def _pytest_in(project_copy):
    import os
    env = {**os.environ, "PYTHONPATH": str(project_copy), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=project_copy, env=env, capture_output=True, encoding="utf-8",
                          errors="replace")


def test_shipped_eval_tasks_are_satisfiable_but_not_already_solved():
    """Each eval project fails as shipped and passes with a reference solution, so a
    failing attempt means the model failed and not that the task is broken."""
    import pathlib, shutil, tempfile
    repo = pathlib.Path(__file__).resolve().parent
    names = sorted(p.stem for p in (repo / "evals").glob("*.yaml"))
    assert names == ["csvrow", "fizzbuzz", "intervals", "palindrome", "slugify", "wordwrap"], names

    for name, (rel, solution) in REFERENCE_SOLUTIONS.items():
        spec = load_task(str(repo / "evals" / f"{name}.yaml"))
        assert spec.kind == "project" and pathlib.Path(spec.project).is_dir(), spec
        copy = pathlib.Path(tempfile.mkdtemp()) / name
        shutil.copytree(spec.project, copy)
        shipped = _pytest_in(copy)
        assert shipped.returncode != 0, f"{name}: already passes as shipped\n{shipped.stdout}"
        (copy / rel).write_bytes(solution.encode("utf-8"))
        fixed = _pytest_in(copy)
        assert fixed.returncode == 0, f"{name}: reference solution fails\n{fixed.stdout}"

    good = {"palindrome": "def is_palindrome(s):" + NL + "    t = [c.lower() for c in s if c.isalnum()]"
                          + NL + "    return t == t[::-1]",
            "fizzbuzz": "def fizzbuzz(n):" + NL + "    return ['FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 "
                        "else 'Buzz' if i % 5 == 0 else str(i) for i in range(1, n + 1)]"}
    for name, code in good.items():
        spec = load_task(str(repo / "evals" / f"{name}.yaml"))
        ok, out = checks.run_tests(code, spec.tests)
        assert ok, (name, out)
        ok, _ = checks.run_tests("def is_palindrome(s): return True" + NL + "def fizzbuzz(n): return []", spec.tests)
        assert not ok, f"{name}: a stub passes the asserts"


def test_summarize_format_and_compare_math():
    def r(task, passed, laps=1, tokens=100, seconds=10.0, esc=False, result=None):
        return {"task": task, "attempt": 1, "passed": passed, "laps": laps, "escalated": esc,
                "total_tokens": tokens, "seconds": seconds,
                "result": result or ("approved" if passed else "stopped by owner")}

    a = {"label": "A", "attempts": 3, "results": [
        r("t1", True, laps=1, tokens=100, seconds=10), r("t1", True, laps=3, tokens=300, seconds=30),
        r("t1", False, tokens=200, seconds=20, esc=True),
        r("t2", False), r("t2", False), r("t2", False, result="error")]}
    rows = {x["task"]: x for x in evals.summarize(a["results"])}
    assert rows["t1"]["passes"] == 2 and abs(rows["t1"]["rate"] - 2 / 3) < 1e-9, rows["t1"]
    assert rows["t1"]["mean_laps_to_pass"] == 2.0 and rows["t1"]["escalated"] == 1, rows["t1"]
    assert rows["t1"]["mean_tokens"] == 200 and rows["t1"]["mean_seconds"] == 20, rows["t1"]
    assert rows["t2"]["passes"] == 0 and rows["t2"]["mean_laps_to_pass"] is None and rows["t2"]["errors"] == 1
    assert rows["ALL"]["n"] == 6 and rows["ALL"]["passes"] == 2, rows["ALL"]
    table = evals.format_table(list(rows.values()))
    assert "2/3" in table and "0/3" in table and "ALL" in table and "-" in table, table

    b = {"label": "B", "attempts": 3, "results": [r("t1", True)] * 3 + [r("t2", True), r("t2", False), r("t2", False)]
         + [r("t3", True)]}
    text = evals.compare(a, b)
    assert "A = A   B = B" in text and "2/3 -> 3/3" in text and "+33%" in text, text
    assert "0/3 -> 1/3" in text and "t3" in text and "only in B" in text, text
    assert "noise" in text, text


class PromptLLM:
    """Replies by looking at the prompt, so one fake serves tasks that should differ."""

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def invoke(self, msg):
        self.calls += 1
        return SimpleNamespace(content=self.fn(msg), usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                                                    "total_tokens": 15})


def _eval_tasks(tmp):
    asserts = NL.join("  - '" + t.replace("'", "''") + "'" for t in checks.TESTS)
    (tmp / "good.yaml").write_text("task: Write is_palindrome(s)." + NL + "tests:" + NL + asserts + NL, encoding="utf-8")
    (tmp / "bad.yaml").write_text("task: BADTASK write is_palindrome(s)." + NL + "tests:" + NL + asserts + NL,
                                  encoding="utf-8")
    return [str(tmp / "bad.yaml"), str(tmp / "good.yaml")]


def test_run_eval_end_to_end_with_fake_models_and_no_leaked_paths():
    import json, pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    files = _eval_tasks(tmp)
    install()
    agent_nodes.MODELS["worker"] = PromptLLM(lambda m: BROKEN if "BADTASK" in m else GOOD)
    messages = []
    report, path = evals.run_eval(files, 2, "unit test!", out_root=tmp / "out", progress=messages.append)

    assert report["finished"] and len(report["results"]) == 4 and len(messages) == 4, report
    rows = {x["task"]: x for x in evals.summarize(report["results"])}
    assert rows["good"]["passes"] == 2 and rows["good"]["mean_laps_to_pass"] == 1, rows["good"]
    assert rows["bad"]["passes"] == 0 and rows["bad"]["n"] == 2, rows["bad"]
    good = next(x for x in report["results"] if x["task"] == "good")
    assert good["total_tokens"] == 45 and good["result"] == "approved" and not good["escalated"], good
    assert {"lead", "worker", "check", "reviewer"} <= set(good["node_seconds"]), good["node_seconds"]
    bad = next(x for x in report["results"] if x["task"] == "bad")
    assert bad["result"] == "stopped by owner" and bad["laps"] == agent_nodes.MAX_REJECTIONS, bad

    assert path.parent == tmp / "out" and path.name.startswith("unit-test--"), path
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["results"] == report["results"] and saved["config"]["models"], saved
    text = path.read_text(encoding="utf-8")
    assert str(tmp) not in text and str(pathlib.Path.home()) not in text, "no local paths in the report"

    try:
        evals.run_eval(files, 1, "x", out_root=tmp / "out", only={"nope"})
    except SystemExit as exc:
        assert "nope" in str(exc), exc
    else:
        raise AssertionError("an unknown --only name must be an error")


def test_run_eval_budget_stops_an_attempt_and_errors_do_not_stop_the_eval():
    import pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    files = _eval_tasks(tmp)[1:]                       # good.yaml only
    install()
    report, _ = evals.run_eval(files, 1, "budget", out_root=tmp / "out", budget=20, progress=lambda *_: None)
    r = report["results"][0]
    assert not r["passed"] and r["result"] == "over token budget", r
    assert r["total_tokens"] == 30, r                  # lead + worker; the budget is checked per node

    class Boom:
        def __init__(self):
            self.calls = 0

        def invoke(self, msg):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model fell over at " + str(pathlib.Path.home()) + "/secret")
            return SimpleNamespace(content=GOOD, usage_metadata={"input_tokens": 1, "output_tokens": 1,
                                                                 "total_tokens": 2})

    install()
    agent_nodes.MODELS["worker"] = Boom()
    report, _ = evals.run_eval(files, 2, "boom", out_root=tmp / "out", progress=lambda *_: None)
    first, second = report["results"]
    assert first["result"] == "error" and not first["passed"] and "RuntimeError" in first["error"], first
    assert str(pathlib.Path.home()) not in first["error"] and "~" in first["error"], first["error"]
    assert second["passed"], second                    # the next attempt still ran
    assert evals.summarize(report["results"])[-1]["errors"] == 1


def test_config_override_and_model_preflight():
    import pathlib, tempfile
    from agent_loop import config
    assert config.override_path({}) is None
    alt = pathlib.Path(tempfile.mkdtemp()) / "alt.yaml"
    alt.write_text("models:" + NL + "  worker: ollama:other:1b" + NL, encoding="utf-8")
    assert config.override_path({"AGENT_LOOP_CONFIG": str(alt)}) == alt.resolve()
    try:
        config.override_path({"AGENT_LOOP_CONFIG": str(alt.parent / "typo.yaml")})
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing override must not silently fall back to the default models")
    loaded = config.load_config(alt)
    assert loaded["models"]["worker"] == "ollama:other:1b" and loaded["models"]["lead"] == "ollama:gemma3:4b"

    tags = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "gemma3:4b"}, {"name": "tiny:latest"}]}
    models = {"lead": "ollama:gemma3:4b", "worker": "ollama:qwen2.5-coder:7b",
              "escalation": "ollama:qwen3:8b", "reviewer": "ollama:tiny"}
    assert evals.missing_models(models, tags) == ["qwen3:8b"]
    assert evals.missing_models({"w": "ollama:gemma3:4b"}, tags) == []


def test_every_container_gets_the_full_lockdown():
    import pathlib, tempfile
    ws = pathlib.Path(tempfile.mkdtemp())
    assert_cmd, project_cmd = checks._docker_cmd("n"), sandbox.project_docker_cmd("n", ws, "img", "pytest")
    for cmd in (assert_cmd, project_cmd):
        joined = " ".join(cmd)
        for flag in ("--network none", "--ipc none", "--read-only", "noexec,nosuid,nodev", "--cap-drop ALL",
                     "no-new-privileges", "--user 65534:65534", "--pull never", "--pids-limit",
                     "nofile=1024", "core=0", "--label agent-loop=sandbox"):
            assert flag in joined, (flag, joined)
        assert cmd[cmd.index("--memory-swap") + 1] == cmd[cmd.index("--memory") + 1], "swap must equal memory"
        assert "--privileged" not in cmd and "--pid" not in cmd and "docker.sock" not in joined, joined
    # nothing is mounted into the assert sandbox; the project one mounts only the workspace, read-only
    assert "--mount" not in assert_cmd and "-v" not in assert_cmd and "--volume" not in assert_cmd
    mounts = [project_cmd[i + 1] for i, a in enumerate(project_cmd) if a in ("--mount", "-v", "--volume")]
    assert len(mounts) == 1 and mounts[0].endswith("target=/work,readonly"), mounts


def test_project_mount_rejects_paths_that_could_inject_mount_options():
    import pathlib
    for bad in ("C:/tmp/a,readonly=false", 'C:/tmp/a"b', "C:/tmp/a'b"):
        try:
            sandbox.project_docker_cmd("n", pathlib.Path(bad), "img", "pytest")
        except ValueError as exc:
            assert "unsafe" in str(exc), exc
        else:
            raise AssertionError(bad)


def test_unisolated_sandbox_needs_an_explicit_override():
    import pathlib, tempfile
    from agent_loop import config
    d = pathlib.Path(tempfile.mkdtemp())
    local, junk = d / "local.yaml", d / "junk.yaml"
    local.write_text("sandbox: local" + NL, encoding="utf-8")
    junk.write_text("sandbox: nothing" + NL, encoding="utf-8")
    try:
        config.load_config(local, env={})
    except ValueError as exc:
        assert "NO isolation" in str(exc) and "AGENT_LOOP_ALLOW_LOCAL" in str(exc), exc
    else:
        raise AssertionError("a bare 'sandbox: local' must not load")
    assert config.load_config(local, env={"AGENT_LOOP_ALLOW_LOCAL": "1"})["sandbox"] == "local"
    for value in ("true", "yes", "0", ""):
        try:
            config.load_config(local, env={"AGENT_LOOP_ALLOW_LOCAL": value})
        except ValueError:
            continue
        raise AssertionError(f"override accepted {value!r}")
    try:
        config.load_config(junk, env={})
    except ValueError as exc:
        assert "docker" in str(exc), exc
    else:
        raise AssertionError("an unknown sandbox value must be an error")
    assert config.load_config(d / "missing.yaml", env={})["sandbox"] == "docker"
    assert config.CONFIG["sandbox"] == "docker", "the shipped config.yaml must stay on docker"


def test_self_check_runs_first_once_and_blocks_model_code_when_broken():
    secret_code = "def f():" + NL + "    return 'MODEL_CODE_MARKER'"
    seen = []

    def healthy(cmd, **kw):
        seen.append(kw.get("input"))
        if kw.get("input") == checks.PROBE:
            return SimpleNamespace(returncode=0, stdout="SANDBOX OK", stderr="")
        return SimpleNamespace(returncode=0, stdout="ALL TESTS PASSED", stderr="")

    with _docker_mode(verified=False), mock.patch.object(checks.subprocess, "run", healthy):
        assert checks.run_tests(secret_code, ["assert f()"])[0]
        assert seen[0] == checks.PROBE and "MODEL_CODE_MARKER" in seen[1], seen   # probe, then the code
        assert checks.run_tests(secret_code, ["assert f()"])[0]
        assert seen.count(checks.PROBE) == 1 and len(seen) == 3, "the probe runs once per process"

    seen.clear()

    def broken(cmd, **kw):
        seen.append(kw.get("input"))
        return SimpleNamespace(returncode=1, stdout="SANDBOX BROKEN: network reachable", stderr="")

    with _docker_mode(verified=False), mock.patch.object(checks.subprocess, "run", broken):
        ok, out = checks.run_tests(secret_code, ["assert f()"])
        assert not ok and "NOT run" in out and "network reachable" in out, out
        assert seen == [checks.PROBE], "model code must never reach a sandbox that failed its check"
        assert not checks.verify_sandbox()[0] and len(seen) == 2, "a failed check is not cached as a pass"

    seen.clear()
    ws = __import__("pathlib").Path(__import__("tempfile").mkdtemp())
    with _docker_mode(verified=False), mock.patch.object(checks.subprocess, "run", broken):
        ok, out = sandbox.run_project_tests(ws, "pytest", [])
        assert not ok and "NOT run" in out and seen == [checks.PROBE], (out, seen)

    def hangs(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 1)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with _docker_mode(verified=False), mock.patch.object(checks.subprocess, "run", hangs):
        ok, out = checks.run_tests(secret_code, ["assert f()"])
    assert not ok and "timed out" in out and "NOT run" in out, out


def test_real_docker_self_check_passes_and_really_detects_a_weak_container():
    if not _require_docker():
        return
    with _docker_mode(verified=False):
        ok, msg = checks.verify_sandbox()
        assert ok, msg
    # The probe is only worth anything if it fails on a container that is NOT locked down.
    weak = subprocess.run(
        ["docker", "run", "--rm", "-i", "--pull", "never", checks.CONFIG["sandbox_image"], "python", "-I", "-"],
        input=checks.PROBE, capture_output=True, encoding="utf-8", errors="replace")
    assert weak.returncode == 1 and "SANDBOX BROKEN" in weak.stdout, weak.stdout
    assert "runs as root" in weak.stdout and "capabilities not dropped" in weak.stdout, weak.stdout
    assert "no-new-privileges not set" in weak.stdout, weak.stdout


def test_credential_files_are_not_copied_and_key_shaped_content_refuses_the_project():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    fake_key = "sk-" + "A" * 24
    src = _project(root / "src", {"app.py": "x = 1" + NL, ".env": "TOKEN=abc" + NL, ".env.local": "A=1" + NL,
                                  "deploy.pem": "pem", "id_rsa": "rsa", "sub/credentials.json": "{}",
                                  "tests/test_app.py": "def test(): pass" + NL})
    skipped = workspace.copy_project(src, root / "ws", max_bytes=100_000)
    assert skipped == [".env", ".env.local", "deploy.pem", "id_rsa", "sub/credentials.json"], skipped
    assert workspace.project_files(root / "ws") == ["app.py", "tests/test_app.py"]
    assert not (root / "ws" / ".env").exists() and not (root / "ws" / "id_rsa").exists()

    leaky = _project(root / "leaky", {"config.py": f'API_KEY = "{fake_key}"' + NL})
    try:
        workspace.copy_project(leaky, root / "ws2", max_bytes=100_000)
    except workspace.WorkspaceError as exc:
        assert "config.py" in str(exc) and fake_key not in str(exc), exc    # names the file, never the secret
    else:
        raise AssertionError("a key-shaped string must refuse the project")
    assert not (root / "ws2").exists(), "nothing is copied when the project is refused"

    pem = _project(root / "pem", {"notes.txt": "-----BEGIN RSA PRIVATE KEY-----" + NL + "abc" + NL})
    try:
        workspace.copy_project(pem, root / "ws3", max_bytes=100_000)
    except workspace.WorkspaceError:
        pass
    else:
        raise AssertionError("a private key block must refuse the project")


def test_thread_ids_stay_plain():
    from agent_loop import runs
    for good in ("live-slug2", "a.b_c-1", "x", "eval-20261029-164049-baba-slugify-1", "A" * 80):
        assert runs.valid_thread(good), good
    for bad in ("", "../x", "a/b", "a" + chr(92) + "b", "a,b", ".hidden", "a b", "A" * 81, "a..b", "x;y", "-x"):
        assert not runs.valid_thread(bad), bad
    assert runs.valid_thread(runs.new_thread_id("some task!! name")), "generated ids are always valid"


FAKE_KEY = "sk-" + "B" * 24


def _traced_run(root, *, reply=GOOD, thread="demo", resume="stop", tests=None):
    install(code=reply)
    return cli.run("is_palindrome", tests=tests or checks.TESTS, thread=thread,
                   runs_dir=root, resume_with=resume)


def _events(root, thread="demo"):
    import json
    lines = (root / thread / "trace.jsonl").read_text(encoding="utf-8").split(NL)
    return [json.loads(x) for x in lines if x]


def test_trace_records_what_each_model_was_asked_and_said():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    _traced_run(root)
    ev = _events(root)
    kinds = [e["event"] for e in ev]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end", kinds
    for role in ("lead", "worker", "reviewer"):
        starts = [e for e in ev if e["event"] == "llm_start" and e["role"] == role]
        ends = [e for e in ev if e["event"] == "llm_end" and e["role"] == role]
        assert len(starts) == len(ends) == 1, (role, kinds)
        assert starts[0]["prompt"] and ends[0]["reply"] and ends[0]["tokens"]["total_tokens"] == 15
        assert kinds.index("llm_start") <= ev.index(starts[0])
    worker_end = next(e for e in ev if e["event"] == "llm_end" and e["role"] == "worker")
    assert "def is_palindrome" in worker_end["reply"], worker_end
    check = next(e for e in ev if e["event"] == "node" and e["node"] == "check")
    assert check["update"]["tests_passed"] is True and "ALL TESTS PASSED" in check["update"]["test_output"]
    assert "work" not in check["update"] and "tokens" not in check["update"], "bulky fields are not repeated"
    end = ev[-1]
    assert end["result"] == "approved" and end["approved"] and end["tokens"]["worker"]["calls"] == 1, end
    assert all(isinstance(e["ts"], float) for e in ev) and ev == sorted(ev, key=lambda e: e["ts"])


def test_trace_covers_pauses_owner_calls_errors_and_turns_off_afterwards():
    import pathlib, tempfile
    from agent_loop import trace
    root = pathlib.Path(tempfile.mkdtemp())
    _traced_run(root, reply=BROKEN, thread="capped", tests=checks.TESTS)
    kinds = [e["event"] for e in _events(root, "capped")]
    assert "pause" in kinds and "owner_call" in kinds and kinds[-1] == "run_end", kinds
    assert kinds.index("pause") < kinds.index("owner_call") < kinds.index("run_end")
    assert _events(root, "capped")[-1]["result"] == "stopped by owner"
    assert trace._sink is None, "the sink is released after a run"

    class Boom:
        def invoke(self, msg):
            raise RuntimeError("model fell over")

    install()
    agent_nodes.MODELS["worker"] = Boom()
    try:
        cli.run("is_palindrome", thread="crash", runs_dir=root, resume_with="stop")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the error must still propagate")
    crash = _events(root, "crash")
    assert any(e["event"] == "llm_error" for e in crash) and crash[-1]["event"] == "run_error", crash
    assert trace._sink is None, "the sink is released even when the run crashes"

    install()
    cli.run("is_palindrome", thread="untraced", resume_with="stop")          # no runs_dir: no trace
    assert not (root / "untraced").exists()


def test_trace_redacts_key_shaped_strings_and_cuts_huge_text():
    import pathlib, tempfile
    from agent_loop import trace
    root = pathlib.Path(tempfile.mkdtemp())
    leaky = GOOD + NL + f"API_KEY = '{FAKE_KEY}'" + NL + "x" * (trace.MAX_TEXT + 500)
    _traced_run(root, reply=leaky, thread="leaky")
    text = (root / "leaky" / "trace.jsonl").read_text(encoding="utf-8")
    assert FAKE_KEY not in text and "[redacted]" in text, "key-shaped strings never reach the trace"
    reply = next(e for e in _events(root, "leaky") if e["event"] == "llm_end" and e["role"] == "worker")["reply"]
    assert len(reply) < trace.MAX_TEXT + 100 and "chars, cut" in reply, len(reply)


class _Client:
    """A bare HTTP client against the dashboard, with full control over headers."""

    def __init__(self, server, token="testtoken"):
        self.server, self.token = server, token

    def request(self, method, path, headers=None, cookie=True, host=None, body=None):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        hdrs = {"Host": host or f"127.0.0.1:{self.server.port}"}
        if cookie:
            hdrs["Cookie"] = f"dash={self.token}"
        hdrs.update(headers or {})
        conn.request(method, path, body=body, headers=hdrs)
        resp = conn.getresponse()
        body = resp.read()
        out = (resp.status, {k.lower(): v for k, v in resp.getheaders()}, body)
        conn.close()
        return out

    def get_json(self, path, **kw):
        import json
        status, headers, body = self.request("GET", path, **kw)
        return status, json.loads(body or b"null"), headers


@contextlib.contextmanager
def _dashboard(root):
    import threading
    from agent_loop import dashboard
    server = dashboard.Dashboard(root, port=0, token="testtoken")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Client(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_listens_on_loopback_only_and_nobody_else_can_take_the_port():
    import pathlib, tempfile
    from agent_loop import dashboard
    with _dashboard(pathlib.Path(tempfile.mkdtemp())) as c:
        assert c.server.server_address[0] == "127.0.0.1", c.server.server_address
        try:
            dashboard.Dashboard(pathlib.Path(tempfile.mkdtemp()), port=c.server.port)
        except OSError:
            pass
        else:
            raise AssertionError("a second program must not be able to bind the same port")


def test_dashboard_needs_the_token_the_right_host_and_only_allows_get():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    _traced_run(root)
    with _dashboard(root) as c:
        for path in ("/", "/api/runs", "/api/trace?id=demo&after=0", "/app.js"):
            assert c.request("GET", path, cookie=False)[0] == 403, path           # no token
            assert c.request("GET", path, headers={"Cookie": "dash=wrong"}, cookie=False)[0] == 403, path
        assert c.request("GET", "/?t=wrong", cookie=False)[0] == 403
        assert c.request("GET", "/api/runs?t=testtoken", cookie=False)[0] == 403, "the URL token only works on /"

        status, headers, _ = c.request("GET", "/?t=testtoken", cookie=False)
        assert status == 302 and headers["location"] == "/" and "testtoken" not in headers["location"], headers
        cookie = headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie, cookie
        assert c.request("GET", "/")[0] == 200

        # DNS-rebinding defence: right cookie, wrong Host
        for host in ("evil.example", "evil.example:80", "127.0.0.1:1", "127.0.0.1", "0.0.0.0:%d" % c.server.port):
            assert c.request("GET", "/api/runs", host=host)[0] == 403, host
        assert c.request("GET", "/api/runs", host=f"localhost:{c.server.port}")[0] == 200

        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS"):
            status, headers, _ = c.request(method, "/api/runs")
            assert status == 405 and headers["allow"] == "GET", (method, status)
            assert not any(k.startswith("access-control") for k in headers), "no CORS, ever"


def test_dashboard_responses_carry_a_strict_policy_and_never_allow_cross_origin_reads():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    _traced_run(root)
    with _dashboard(root) as c:
        for path in ("/", "/app.js", "/app.css", "/api/runs", "/api/evals", "/missing"):
            status, headers, _ = c.request("GET", path)
            csp = headers["content-security-policy"]
            assert "script-src 'self'" in csp and "default-src 'none'" in csp and "frame-ancestors 'none'" in csp, csp
            assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "http" not in csp, csp
            assert headers["x-content-type-options"] == "nosniff" and headers["cache-control"] == "no-store"
            assert headers["x-frame-options"] == "DENY" and headers["referrer-policy"] == "no-referrer"
            assert headers["cross-origin-resource-policy"] == "same-origin"
            assert not any(k.startswith("access-control") for k in headers), path
        assert c.request("GET", "/api/runs")[1]["content-type"].startswith("application/json")


def test_dashboard_page_files_cannot_inject_markup_or_pull_in_anything_remote():
    import pathlib, re
    web = pathlib.Path(__file__).resolve().parent / "agent_loop" / "web"
    scripts = {n: (web / n).read_text(encoding="utf-8") for n in ("app.js", "logic.js")}
    js = NL.join(scripts.values())
    html, css = [(web / n).read_text(encoding="utf-8") for n in ("index.html", "app.css")]
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
                   "srcdoc", "javascript:", ".href", "setAttribute(\"href\"", "setAttribute(\"src\"",
                   "setAttribute(\"style\"", "style:", "XMLHttpRequest", "WebSocket", "import("):
        for name, source in scripts.items():
            assert banned not in source, f"{name} uses {banned}"
    # The SVG namespace is an identifier the browser never fetches; it is the only address allowed.
    everything = (js + html + css).replace("http://www.w3.org/2000/svg", "")
    assert not re.search(r"https?://", everything), "the page must not reference any remote address"
    assert "style=" not in html and not re.search(r"<script(?![^>]*\ssrc=)", html), "no inline script or style"
    assert not re.search(r"\son[a-z]+\s*=", html), "no inline event handlers"
    assert "@import" not in css and "url(" not in css


def test_dashboard_lists_runs_pages_traces_and_refuses_hostile_ids():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    html_attack = '<script>alert(1)</script></pre><img src=x onerror=alert(1)>'
    _traced_run(root, reply=html_attack + NL + GOOD)
    (root / "demo" / "changes.patch").write_bytes(("--- a/x" + NL + "+++ b/x" + NL).encode("utf-8"))
    secret = root.parent / "outside-secret"
    secret.mkdir(exist_ok=True)
    (secret / "trace.jsonl").write_text('{"event":"llm_end","reply":"TOP SECRET"}' + NL, encoding="utf-8")
    with _dashboard(root) as c:
        status, data, _ = c.get_json("/api/runs")
        run = next(r for r in data["runs"] if r["id"] == "demo")
        assert run["status"] == "finished" and run["result"] == "approved" and run["calls"] == 3, run
        assert run["kind"] == "code" and run["models"], run

        status, first, headers = c.get_json("/api/trace?id=demo&after=0")
        assert status == 200 and first["total"] == len(_events(root)) and first["next"] == first["total"], first
        status, rest, _ = c.get_json("/api/trace?id=demo&after=4")
        assert rest["events"] == first["events"][4:], "paging by index returns exactly the later events"
        status, tail, _ = c.get_json(f"/api/trace?id=demo&after={first['total']}")
        assert status == 200 and tail["events"] == [], tail

        # hostile model output is only ever JSON data, never a page
        worker = next(e for e in first["events"] if e["event"] == "llm_end" and e["role"] == "worker")
        assert html_attack in worker["reply"] and headers["content-type"].startswith("application/json")
        assert headers["x-content-type-options"] == "nosniff"

        assert c.get_json("/api/patch?id=demo")[1]["patch"].startswith("--- a/x")
        assert c.get_json("/api/trace?id=demo&after=abc")[0] == 400
        assert c.get_json("/api/trace?id=demo&after=-1")[0] == 400

        outside = "../" + secret.name
        for ident in (outside, "demo/../../" + secret.name, "%2e%2e/" + secret.name, "..%2f" + secret.name,
                      "/etc/passwd", "C:/Windows/win.ini", "demo%5C..%5Cx", "demo/..", ".", "", "evals",
                      "demo/trace.jsonl", "a/b/c/d/e/f", "demo;x", "demo%00", "x" * 400):
            status, body, _ = c.get_json("/api/trace?id=" + ident + "&after=0")
            assert status in (400, 404) and "TOP SECRET" not in str(body), (ident, status)
        assert c.request("GET", "/../agent_loop/config.py")[0] == 404
        assert c.request("GET", "/app.js/../../config.yaml")[0] == 404
        assert b"TOP SECRET" not in c.request("GET", "/api/trace?id=" + outside)[2]


def test_dashboard_shows_a_run_that_is_still_being_written():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    d = root / "live"
    d.mkdir()
    start = '{"ts": 1.0, "event": "run_start", "kind": "code", "task": "t", "models": {}}'
    ask = '{"ts": 2.0, "event": "llm_start", "role": "worker", "model": "m", "prompt": "p"}'
    (d / "trace.jsonl").write_bytes((start + NL + ask + NL + '{"ts": 3.0, "event": "llm_en').encode("utf-8"))
    with _dashboard(root) as c:
        run = next(r for r in c.get_json("/api/runs")[1]["runs"] if r["id"] == "live")
        assert run["status"] == "running" and run["thinking"] is True and run["events"] == 2, run
        assert c.get_json("/api/trace?id=live&after=0")[1]["total"] == 2, "a half-written line is not served"
        with open(d / "trace.jsonl", "ab") as f:
            f.write(b'd", "role": "worker", "reply": "r", "seconds": 1.0, "tokens": {}}' + b"\n")
        run = next(r for r in c.get_json("/api/runs")[1]["runs"] if r["id"] == "live")
        assert run["events"] == 3 and run["thinking"] is False, run
        (d / "trace.jsonl").write_bytes((start + NL + "this is not json" + NL).encode("utf-8"))
        assert c.get_json("/api/trace?id=live&after=0")[1]["total"] == 1, "a corrupt line is skipped, not fatal"


def test_dashboard_serves_eval_reports_and_comparisons_by_validated_name():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    tmp = pathlib.Path(tempfile.mkdtemp())
    files = _eval_tasks(tmp)
    install()
    agent_nodes.MODELS["worker"] = PromptLLM(lambda m: BROKEN if "BADTASK" in m else GOOD)
    first, p1 = evals.run_eval(files, 1, "alpha", out_root=root / "evals", progress=lambda *_: None)
    second, p2 = evals.run_eval(files, 2, "beta", out_root=root / "evals", progress=lambda *_: None)
    with _dashboard(root) as c:
        reports = c.get_json("/api/evals")[1]["reports"]
        assert {r["name"] for r in reports} == {p1.name, p2.name}, reports
        assert all(r["finished"] for r in reports)
        status, data, _ = c.get_json("/api/eval?name=" + p2.name)
        rows = {r["task"]: r for r in data["summary"]}
        assert rows["good"]["passes"] == 2 and rows["bad"]["passes"] == 0 and data["report"]["label"] == "beta"
        status, data, _ = c.get_json(f"/api/compare?a={p1.name}&b={p2.name}")
        assert status == 200 and "A = alpha   B = beta" in data["text"], data
        for bad in ("../" + p1.name, "nope.json", "..%2fx.json", p1.name + "/", "a.json%00", "", "x" * 300 + ".json"):
            assert c.get_json("/api/eval?name=" + bad)[0] == 404, bad
        assert c.get_json("/api/compare?a=" + p1.name + "&b=nope.json")[0] == 404


def _outside_fences(markdown: str) -> str:
    """The Markdown that is NOT inside a code fence. Model text must never appear here."""
    import re
    outside, fence = [], None
    for line in markdown.split(NL):
        m = re.fullmatch(r"(`{3,})[A-Za-z]*", line)
        if fence is None:
            if m:
                fence = len(m.group(1))
            else:
                outside.append(line)
        elif re.fullmatch(r"`{%d,}" % fence, line):
            fence = None
    assert fence is None, "a code fence was never closed"
    return NL.join(outside)


def test_export_keeps_every_model_written_byte_inside_a_code_fence():
    from agent_loop import dashboard
    assert dashboard.fence_for("plain") == "`" * 3
    assert dashboard.fence_for("a " + "`" * 3 + " b") == "`" * 4
    assert dashboard.fence_for("`" * 7 + " x `") == "`" * 8
    assert dashboard.export_filename("evals/a b/../c") == "evals_a_b_._c.md"
    assert dashboard.export_filename("../../x") == "x.md" and dashboard.export_filename("...") == "run.md"
    hostile = "HOSTILE<script>alert(1)</script>" + NL + "`" * 3 + NL + "# fake heading" + NL + "`" * 9 + NL + "[x](javascript:1)"
    events = [
        {"ts": 1.0, "event": "run_start", "kind": "code", "task": hostile, "models": {"worker": "m" + hostile}},
        {"ts": 2.0, "event": "llm_start", "role": "worker", "model": "m", "prompt": hostile},
        {"ts": 3.0, "event": "llm_end", "role": "worker", "seconds": 1.0, "reply": hostile, "tokens": {"total_tokens": 5}},
        {"ts": 4.0, "event": "node", "node": "check", "seconds": 1.0, "update": {"test_output": hostile}},
        {"ts": 5.0, "event": "pause", "number": 1, "info": {"review": hostile}},
        {"ts": 6.0, "event": "owner_call", "text": hostile, "source": "dashboard"},
        {"ts": 7.0, "event": "llm_error", "role": "reviewer", "seconds": 1.0, "error": hostile},
        {"ts": 8.0, "event": "run_end", "result": "stopped by owner", "rejections": 3, "pauses": 1},
    ]
    report = dashboard.render_report("demo/x", events, "diff --git a/x b/x" + NL + hostile)
    assert report.count("HOSTILE") >= 8, "the content is all there"
    outside = _outside_fences(report)
    for marker in ("HOSTILE", "<script>", "fake heading", "javascript:", "alert(1)"):
        assert marker not in outside, f"{marker!r} escaped its code fence"
    assert "# Run demo_x" in outside and "stopped by owner" in outside


def test_dashboard_export_session_and_eval_stats_endpoints():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    _traced_run(root, reply="<script>1</script>" + NL + GOOD)
    tmp = pathlib.Path(tempfile.mkdtemp())
    install()
    agent_nodes.MODELS["worker"] = PromptLLM(lambda m: BROKEN if "BADTASK" in m else GOOD)
    evals.run_eval(_eval_tasks(tmp), 2, "stats", out_root=root / "evals", progress=lambda *_: None)
    with _dashboard(root) as c:
        status, headers, body = c.request("GET", "/api/export?id=demo")
        text = body.decode("utf-8")
        assert status == 200 and headers["content-type"].startswith("text/markdown"), headers
        assert headers["content-disposition"] == 'attachment; filename="demo.md"', headers
        assert headers["x-content-type-options"] == "nosniff" and "<script>1</script>" in text
        assert "<script>" not in _outside_fences(text)
        for bad in ("../x", "nope", "", "a/b/c/d/e/f", "%2e%2e/x"):
            assert c.request("GET", "/api/export?id=" + bad)[0] == 404, bad
        assert c.request("GET", "/api/export?id=demo", cookie=False)[0] == 403

        assert c.request("GET", "/api/session", cookie=False)[0] == 403
        csrf = c.get_json("/api/session")[1]["csrf"]
        assert isinstance(csrf, str) and len(csrf) >= 20 and csrf == c.get_json("/api/session")[1]["csrf"]
        assert csrf != "testtoken", "the CSRF value is separate from the login token"

        report = c.get_json("/api/evals")[1]["reports"][0]
        assert (report["passes"], report["n"], report["rate"]) == (2, 4, 0.5), report
        assert report["mean_tokens"] and report["mean_seconds"] is not None

        status, headers, body = c.request("GET", "/logic.js")
        assert status == 200 and headers["content-type"].startswith("text/javascript") and b"parseDiff" in body


def test_page_logic_passes_its_node_tests():
    import shutil
    node = shutil.which("node")
    if not node:
        print("  (node not installed - page logic tests skipped)")
        return
    done = subprocess.run([node, str(_REPO / "jstests" / "logic.test.js")], capture_output=True,
                          encoding="utf-8", errors="replace", cwd=_REPO)
    assert done.returncode == 0 and "logic checks passed" in done.stdout, done.stdout + done.stderr


def _wait_for(predicate, timeout=20.0, step=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    raise AssertionError("timed out waiting")


def _pause_event(root, thread):
    path = root / thread / "trace.jsonl"
    if not path.exists():
        return None
    return next((e for e in _events(root, thread) if e["event"] == "pause"), None)


class _RunInThread:
    """cli.run in a background thread, so a test can play the part of the dashboard or the terminal."""

    def __init__(self, **kw):
        import threading
        self.result, self.error = None, None

        def go():
            try:
                self.result = cli.run(**kw)
            except BaseException as exc:  # reported by join()
                self.error = exc

        self.thread = threading.Thread(target=go, daemon=True)
        self.thread.start()

    def join(self, timeout=30):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the run is still waiting"
        if self.error:
            raise self.error
        return self.result


def test_mailbox_accepts_only_clean_replies_for_the_right_pause_and_only_once():
    import json, pathlib, tempfile
    from agent_loop import mailbox
    for ok in ("stop", "  accept  ", "try the other approach" + NL + "and add a test", "a" * 2000, "tab" + chr(9) + "ok"):
        assert mailbox.clean_text(ok) == ok.strip(), ok
    for bad in ("", "   ", "a" * 2001, "nul" + chr(0), "esc" + chr(27) + "[31m", "del" + chr(127), None, 5, ["stop"], {"a": 1}):
        assert mailbox.clean_text(bad) is None, repr(bad)

    d = pathlib.Path(tempfile.mkdtemp())
    nonce = "abcdefgh1234"
    mailbox.write_reply(d, nonce, "stop")
    assert sorted(p.name for p in d.iterdir()) == ["owner_reply.json"], "no temp file may be left behind"
    try:
        mailbox.write_reply(d, nonce, "accept")
    except FileExistsError:
        pass
    else:
        raise AssertionError("a second reply must not overwrite the first")
    assert mailbox.consume_reply(d, "some-other-nonce") is None and not (d / "owner_reply.json").exists(), \
        "a reply for another pause is discarded"
    mailbox.write_reply(d, nonce, "stop")
    assert mailbox.consume_reply(d, nonce) == "stop"
    assert mailbox.consume_reply(d, nonce) is None, "a reply is used once"
    assert list(d.iterdir()) == []

    for bad_nonce, bad_text in (("short", "stop"), ("x" * 65, "stop"), (nonce, ""), (nonce, "a" * 2001), (nonce, "x" + chr(0)), (None, "stop")):
        try:
            mailbox.write_reply(d, bad_nonce, bad_text)
        except ValueError:
            pass
        else:
            raise AssertionError((bad_nonce, bad_text))
    assert list(d.iterdir()) == [], "a rejected reply leaves nothing behind"

    path = d / "owner_reply.json"
    for body in (b"x" * 9000, b"not json", b"[1, 2]", json.dumps({"nonce": nonce}).encode(),
                 json.dumps({"nonce": nonce, "text": "x" + chr(0)}).encode(), json.dumps({"nonce": 7, "text": "stop"}).encode(),
                 b"\xff\xfe\x00bad utf8"):
        path.write_bytes(body)
        assert mailbox.consume_reply(d, nonce) is None, body[:20]
        assert not path.exists(), "an unusable reply is deleted"
    mailbox.write_reply(d, nonce, "stop")
    mailbox.clear(d)
    mailbox.clear(d)  # clearing nothing is fine
    assert list(d.iterdir()) == []


def test_pause_nonce_is_traced_and_a_stale_reply_is_cleared():
    import pathlib, tempfile
    from agent_loop import mailbox
    root = pathlib.Path(tempfile.mkdtemp())
    thread_dir = root / "t"
    thread_dir.mkdir()
    mailbox.write_reply(thread_dir, "oldnonce1234", "accept")
    cli.trace.begin(thread_dir / "trace.jsonl")
    try:
        first = cli._open_pause(thread_dir, 1, {"review": "r", "rejections": 3})
        second = cli._open_pause(thread_dir, 2, {"review": "r"})
    finally:
        cli.trace.begin(None)
    assert not (thread_dir / "owner_reply.json").exists(), "a reply from an earlier pause is dropped"
    assert first != second and len(first) >= 12
    pauses = [e for e in _events(root, "t") if e["event"] == "pause"]
    assert [e["nonce"] for e in pauses] == [first, second] and pauses[0]["info"]["rejections"] == 3


def test_a_paused_run_can_be_answered_from_the_dashboard():
    import pathlib, tempfile, threading
    from agent_loop import mailbox
    root = pathlib.Path(tempfile.mkdtemp())
    forever = threading.Event()

    def silent_terminal():
        forever.wait(60)
        raise EOFError

    install(code=BROKEN)
    with mock.patch.object(cli, "_terminal", cli._Terminal(silent_terminal)):
        run = _RunInThread(task="is_palindrome", tests=checks.TESTS, thread="web", runs_dir=root)
        try:
            pause = _wait_for(lambda: _pause_event(root, "web"))
            assert pause["nonce"] and pause["info"]["rejections"] == agent_nodes.MAX_REJECTIONS, pause
            assert run.thread.is_alive(), "the run waits for an answer"
            mailbox.write_reply(root / "web", pause["nonce"], "stop")
            values = run.join()
        finally:
            forever.set()
    assert values["outcome"] == "stopped by owner: stop", values["outcome"]
    ev = _events(root, "web")
    owner = next(e for e in ev if e["event"] == "owner_call")
    assert owner["source"] == "dashboard" and owner["text"] == "stop", owner
    assert [e["event"] for e in ev].index("pause") < ev.index(owner) < len(ev) - 1 and ev[-1]["event"] == "run_end"
    assert not (root / "web" / "owner_reply.json").exists()


def test_terminal_answer_still_works_and_wrong_or_stale_dashboard_replies_are_ignored():
    import pathlib, tempfile, threading
    from agent_loop import mailbox
    root = pathlib.Path(tempfile.mkdtemp())
    typed = threading.Event()
    calls = []

    def terminal():
        calls.append(1)
        if len(calls) == 1:
            typed.wait(60)
            return "stop"
        raise EOFError

    install(code=BROKEN)
    with mock.patch.object(cli, "_terminal", cli._Terminal(terminal)):
        run = _RunInThread(task="is_palindrome", tests=checks.TESTS, thread="term", runs_dir=root)
        pause = _wait_for(lambda: _pause_event(root, "term"))
        time.sleep(0.6)  # the wait has started and dropped anything typed before the question
        mailbox.write_reply(root / "term", "not-this-pause-1", "accept")       # wrong nonce
        _wait_for(lambda: not (root / "term" / "owner_reply.json").exists())
        assert run.thread.is_alive(), "a reply for another pause must not end the run"
        typed.set()
        values = run.join()
    assert values["outcome"] == "stopped by owner: stop", values["outcome"]
    owner = next(e for e in _events(root, "term") if e["event"] == "owner_call")
    assert owner["source"] == "terminal", owner

    # --resume (what evals use) never waits and never reads the mailbox
    install(code=BROKEN)
    cli.run("is_palindrome", tests=checks.TESTS, thread="flagged", runs_dir=root, resume_with="stop")
    assert next(e for e in _events(root, "flagged") if e["event"] == "owner_call")["source"] == "flag"


def test_lines_typed_while_no_question_is_open_are_not_answers():
    term = cli._Terminal(read_line=lambda: (_ for _ in ()).throw(EOFError()))
    term.lines.put("early")
    term.lines.put("typing")
    assert term.drain() == 2 and term.drain() == 0
    term.start()
    _wait_for(lambda: term.eof)
    term.start()  # EOF is final: no new reader is started in a loop
    assert term.thread is not None and not term.thread.is_alive()


NONCE = "nonce-abcdef12"


def _write_trace(root, name, events):
    import json
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.jsonl").write_bytes("".join(json.dumps(e) + NL for e in events).encode("utf-8"))


def _paused_fixture(root):
    """Runs in each state the POST endpoint has to tell apart."""
    start = {"ts": 1.0, "event": "run_start", "kind": "code", "task": "t", "models": {}}
    pause = {"ts": 2.0, "event": "pause", "number": 1, "nonce": NONCE, "info": {"review": "r", "rejections": 3}}
    _write_trace(root, "waiting", [start, pause])
    _write_trace(root, "nopause", [start])
    _write_trace(root, "answered", [start, pause, {"ts": 3.0, "event": "owner_call", "text": "stop", "source": "terminal"}])
    _write_trace(root, "finished", [start, pause, {"ts": 3.0, "event": "owner_call", "text": "stop", "source": "terminal"},
                                    {"ts": 4.0, "event": "run_end", "result": "stopped by owner"}])
    _write_trace(root, "crashed", [start, pause, {"ts": 3.0, "event": "run_error", "error": "boom"}])
    second = dict(pause, ts=5.0, number=2, nonce="second-nonce-99")
    _write_trace(root, "repaused", [start, pause, {"ts": 3.0, "event": "owner_call", "text": "try again", "source": "terminal"}, second])


def _tree(root):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size if p.is_file() else -1) for p in root.rglob("*"))


def _post(c, payload=None, *, raw=None, headers=None, cookie=True, host=None, origin="same", csrf="same",
          content_type="application/json", path="/api/owner_reply", method="POST"):
    import json
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    hdrs = {}
    if content_type is not None:
        hdrs["Content-Type"] = content_type
    if origin == "same":
        hdrs["Origin"] = f"http://127.0.0.1:{c.server.port}"
    elif origin is not None:
        hdrs["Origin"] = origin
    if csrf == "same":
        hdrs["X-CSRF"] = c.server.csrf
    elif csrf is not None:
        hdrs["X-CSRF"] = csrf
    hdrs.update(headers or {})
    status, resp_headers, text = c.request(method, path, headers=hdrs, cookie=cookie, host=host, body=body)
    return status, resp_headers, text


GOOD_POST = {"id": "waiting", "nonce": NONCE, "text": "stop"}


def test_owner_reply_endpoint_writes_exactly_one_small_file_and_nothing_else():
    import json, pathlib, tempfile
    from agent_loop import dashboard, mailbox
    root = pathlib.Path(tempfile.mkdtemp())
    _paused_fixture(root)
    before = _tree(root)
    with mock.patch.object(dashboard, "POST_INTERVAL", 0), _dashboard(root) as c:
        status, headers, body = _post(c, GOOD_POST)
        assert status == 200 and json.loads(body) == {"ok": True}, (status, body)
        assert headers["x-content-type-options"] == "nosniff" and "access-control-allow-origin" not in headers
        added = [row for row in _tree(root) if row not in before]
        assert added == [("waiting/owner_reply.json", added[0][1])], added
        assert json.loads((root / "waiting" / "owner_reply.json").read_text(encoding="utf-8")) == {"nonce": NONCE, "text": "stop"}
        assert mailbox.consume_reply(root / "waiting", NONCE) == "stop"

        # localhost is an equally valid origin; guidance text passes through unchanged
        status, _, _ = _post(c, dict(GOOD_POST, text="Try a regex" + NL + "and keep the tests"), origin=f"http://localhost:{c.server.port}")
        assert status == 200
        assert mailbox.consume_reply(root / "waiting", NONCE) == "Try a regex" + NL + "and keep the tests"
        # a later pause needs its own nonce
        assert _post(c, {"id": "repaused", "nonce": "second-nonce-99", "text": "stop"})[0] == 200
        assert (root / "repaused" / "owner_reply.json").exists()


def test_owner_reply_endpoint_rejects_everything_else_and_changes_nothing():
    import json, pathlib, tempfile
    from agent_loop import dashboard, mailbox
    root = pathlib.Path(tempfile.mkdtemp())
    _paused_fixture(root)
    big = json.dumps(dict(GOOD_POST, text="x" * 5000)).encode("utf-8")
    cases = {
        # who is asking
        "no cookie": (dict(cookie=False), 403),
        "wrong cookie": (dict(cookie=False, headers={"Cookie": "dash=wrong"}), 403),
        "wrong host": (dict(host="evil.example"), 403),
        "host without port": (dict(host="127.0.0.1"), 403),
        "no origin": (dict(origin=None), 403),
        "foreign origin": (dict(origin="http://evil.example"), 403),
        "null origin": (dict(origin="null"), 403),
        "origin with another port": (dict(origin="http://127.0.0.1:1"), 403),
        "origin without port": (dict(origin="http://127.0.0.1"), 403),
        "https origin": (dict(origin="https://127.0.0.1"), 403),
        "no csrf header": (dict(csrf=None), 403),
        "wrong csrf header": (dict(csrf="wrong"), 403),
        "the login token is not the csrf token": (dict(csrf="testtoken"), 403),
        # what is being sent
        "text/plain body": (dict(content_type="text/plain"), 415),
        "form body": (dict(content_type="application/x-www-form-urlencoded"), 415),
        "no content type": (dict(content_type=None), 415),
        "oversize body": (dict(raw=big), 413),
        "empty body": (dict(raw=b""), 413),
        "not json": (dict(raw=b"stop"), 400),
        "json list": (dict(raw=b'["stop"]'), 400),
        "json string": (dict(raw=b'"stop"'), 400),
        "missing text": (dict(raw=json.dumps({"id": "waiting", "nonce": NONCE}).encode()), 400),
        "number as nonce": (dict(raw=json.dumps({"id": "waiting", "nonce": 5, "text": "stop"}).encode()), 400),
        "list as id": (dict(raw=json.dumps({"id": ["waiting"], "nonce": NONCE, "text": "stop"}).encode()), 400),
        # which run, and whether it is waiting
        "unknown run": (dict(payload=dict(GOOD_POST, id="nope")), 404),
        "path traversal id": (dict(payload=dict(GOOD_POST, id="../waiting")), 404),
        "encoded traversal id": (dict(payload=dict(GOOD_POST, id="%2e%2e/waiting")), 404),
        "absolute id": (dict(payload=dict(GOOD_POST, id="/etc")), 404),
        "empty id": (dict(payload=dict(GOOD_POST, id="")), 404),
        "run that never paused": (dict(payload=dict(GOOD_POST, id="nopause")), 409),
        "wrong nonce": (dict(payload=dict(GOOD_POST, nonce="wrong-nonce-123")), 409),
        "nonce of an older pause": (dict(payload={"id": "repaused", "nonce": NONCE, "text": "stop"}), 409),
        "pause already answered": (dict(payload=dict(GOOD_POST, id="answered")), 409),
        "run already finished": (dict(payload=dict(GOOD_POST, id="finished")), 409),
        "run that crashed": (dict(payload=dict(GOOD_POST, id="crashed")), 409),
        # what is being said
        "empty text": (dict(payload=dict(GOOD_POST, text="")), 400),
        "blank text": (dict(payload=dict(GOOD_POST, text="   ")), 400),
        "text too long": (dict(payload=dict(GOOD_POST, text="x" * 2001)), 400),
        "control character": (dict(payload=dict(GOOD_POST, text="stop" + chr(0))), 400),
        "escape sequence": (dict(payload=dict(GOOD_POST, text=chr(27) + "[2J")), 400),
        # other doors stay shut
        "other POST path": (dict(payload=GOOD_POST, path="/api/runs"), 405),
        "POST to a static file": (dict(payload=GOOD_POST, path="/app.js"), 405),
        "PUT": (dict(payload=GOOD_POST, method="PUT"), 405),
        "DELETE": (dict(payload=GOOD_POST, method="DELETE"), 405),
        "PATCH": (dict(payload=GOOD_POST, method="PATCH"), 405),
    }
    before = _tree(root)
    with mock.patch.object(dashboard := __import__("agent_loop.dashboard", fromlist=["x"]), "POST_INTERVAL", 0), _dashboard(root) as c:
        for name, (kwargs, expected) in cases.items():
            kwargs = dict(kwargs)
            payload = kwargs.pop("payload", GOOD_POST if "raw" not in kwargs else None)
            status, headers, body = _post(c, payload, **kwargs)
            assert status == expected, (name, status, body[:80])
            assert _tree(root) == before, f"{name}: something was written"
            assert not any(k.startswith("access-control") for k in headers), name

        # a reply that is already waiting is never overwritten
        assert _post(c, GOOD_POST)[0] == 200
        first = (root / "waiting" / "owner_reply.json").read_bytes()
        assert _post(c, dict(GOOD_POST, text="accept"))[0] == 409
        assert (root / "waiting" / "owner_reply.json").read_bytes() == first
        mailbox.clear(root / "waiting")
        assert _tree(root) == before


def test_owner_reply_endpoint_is_rate_limited_and_never_needs_a_get():
    import pathlib, tempfile
    from agent_loop import dashboard
    root = pathlib.Path(tempfile.mkdtemp())
    _paused_fixture(root)
    with _dashboard(root) as c:
        assert _post(c, dict(GOOD_POST, text=""))[0] == 400            # reaches validation: uses the slot
        assert _post(c, dict(GOOD_POST, text=""))[0] == 429            # one attempt per second, valid or not
        assert _post(c, GOOD_POST)[0] == 429
        time.sleep(dashboard.POST_INTERVAL + 0.1)
        assert _post(c, GOOD_POST)[0] == 200
        assert c.request("GET", "/api/owner_reply")[0] == 404, "reading is not a way to write"
        assert c.request("GET", "/api/owner_reply?id=waiting&nonce=" + NONCE + "&text=stop")[0] == 404
        assert not any(p.name == "owner_reply.json" for p in (root / "nopause").iterdir())


def test_answering_from_the_page_end_to_end_with_a_real_paused_run():
    import json, pathlib, tempfile, threading
    from agent_loop import dashboard
    root = pathlib.Path(tempfile.mkdtemp())
    forever = threading.Event()

    def silent_terminal():
        forever.wait(60)
        raise EOFError

    install(code=BROKEN)
    with mock.patch.object(dashboard, "POST_INTERVAL", 0), mock.patch.object(cli, "_terminal", cli._Terminal(silent_terminal)), \
            _dashboard(root) as c:
        run = _RunInThread(task="is_palindrome", tests=checks.TESTS, thread="live", runs_dir=root)
        try:
            summary = _wait_for(lambda: next((r for r in c.get_json("/api/runs")[1]["runs"] if r["id"] == "live" and r["status"] == "paused"), None))
            pause = next(e for e in c.get_json("/api/trace?id=live&after=0")[1]["events"] if e["event"] == "pause")
            assert summary["status"] == "paused" and pause["nonce"]
            assert _post(c, {"id": "live", "nonce": "not-the-nonce-00", "text": "stop"})[0] == 409
            time.sleep(0.05)
            assert run.thread.is_alive(), "a wrong answer does not end the run"
            status, _, body = _post(c, {"id": "live", "nonce": pause["nonce"], "text": "stop"})
            assert status == 200, body
            values = run.join()
        finally:
            forever.set()
        assert values["outcome"] == "stopped by owner: stop"
        trace_events = c.get_json("/api/trace?id=live&after=0")[1]["events"]
        assert next(e for e in trace_events if e["event"] == "owner_call")["source"] == "dashboard"
        assert _post(c, {"id": "live", "nonce": pause["nonce"], "text": "stop"})[0] == 409, "an answered pause cannot be answered again"
        assert next(r for r in c.get_json("/api/runs")[1]["runs"] if r["id"] == "live")["status"] == "finished"


def test_the_page_has_exactly_one_way_to_write_and_it_is_the_owner_reply():
    import pathlib, re
    js = (pathlib.Path(__file__).resolve().parent / "agent_loop" / "web" / "app.js").read_text(encoding="utf-8")
    assert len(re.findall(r"method:\s*\"POST\"", js)) == 1, "one POST in the whole page"
    assert not re.search(r"method:\s*\"(PUT|DELETE|PATCH)\"", js)
    assert js.count("/api/owner_reply") == 1 and "X-CSRF" in js and "same-origin" in js
    assert len(re.findall(r"\bfetch\(", js)) == 2, "a read helper and the single write"
    assert "localStorage" not in js and "sessionStorage" not in js and "document.cookie" not in js


class _ServerError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class _FlakyLLM:
    """Fails with the given errors in order, then answers."""

    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def invoke(self, msg):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(content="fine", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})


def test_a_transient_server_error_is_retried_exactly_once():
    import pathlib, tempfile
    from agent_loop import trace
    path = pathlib.Path(tempfile.mkdtemp()) / "t" / "trace.jsonl"

    def attempt(errors):
        model = _FlakyLLM(errors)
        trace.begin(path)
        try:
            with mock.patch.object(trace, "RETRY_WAIT", 0):
                try:
                    return model, trace.llm_call("worker", model, "prompt"), None
                except Exception as exc:
                    return model, None, exc
        finally:
            trace.begin(None)

    model, resp, error = attempt([_ServerError("CUDA error: an illegal memory access", 500)])
    assert error is None and resp.content == "fine" and model.calls == 2
    kinds = [e["event"] for e in _events(path.parent.parent, "t")]
    assert kinds == ["llm_start", "llm_retry", "llm_end"], kinds        # one call, visibly retried
    assert "CUDA" in next(e for e in _events(path.parent.parent, "t") if e["event"] == "llm_retry")["error"]

    path.unlink()
    model, resp, error = attempt([_ServerError("boom", 500), _ServerError("boom again", 503)])
    assert resp is None and "boom again" in str(error) and model.calls == 2, "a second failure is not retried"
    assert [e["event"] for e in _events(path.parent.parent, "t")] == ["llm_start", "llm_retry", "llm_error"]

    for errors in ([_ServerError("bad request", 400)], [_ServerError("not found", 404)], [ValueError("plain")],
                   [_ServerError("odd", True)], [_ServerError("odd", "500")]):
        path.unlink()
        model, resp, error = attempt(errors)
        assert resp is None and error is not None and model.calls == 1, errors      # never retried
        assert "llm_retry" not in [e["event"] for e in _events(path.parent.parent, "t")]


def test_the_project_is_kept_off_the_front_of_the_python_path():
    import pathlib, re, tempfile
    ws = pathlib.Path(tempfile.mkdtemp())
    cmd = sandbox.project_docker_cmd("n", ws, "img", "pytest -q tests")
    assert cmd[cmd.index("img") + 1:] == ["python", "-I", "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", "tests"], cmd
    assert not any("PYTHONPATH" in part for part in cmd), "PYTHONPATH would put project files ahead of site-packages"
    assert sandbox._test_argv("python -m unittest")[:3] == ["python", "-m", "unittest"], "only pytest commands are rewritten"
    docker = pathlib.Path(sandbox.DOCKERFILE).read_text(encoding="utf-8")
    assert re.search(r"install --no-cache-dir pytest==\d+\.\d+\.\d+", docker), "pytest must be pinned"
    assert "work.pth" in docker and "/work/src" in docker, "the project goes on the path through the image"
    assert "sitecustomize.py" in docker, "the image must occupy the sitecustomize name"


def test_the_requirements_install_is_held_to_the_frozen_pytest_set():
    import pathlib
    docker = pathlib.Path(sandbox.DOCKERFILE).read_text(encoding="utf-8")
    pin, freeze, snapshot, install = (docker.find(s) for s in (
        "install --no-cache-dir pytest==9.2.0", "pip freeze > /tmp/constraints.txt",
        "python -I -S /tmp/verify_pytest.py snapshot",
        "pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt"))
    assert -1 not in (pin, freeze, snapshot, install) and pin < freeze < snapshot < install, (
        "freeze and snapshot the pytest set right after the pytest install, then install requirements with -c")


def test_the_image_build_ends_by_verifying_the_pytest_set():
    import pathlib, re
    docker = pathlib.Path(sandbox.DOCKERFILE).read_text(encoding="utf-8")
    steps = [line for line in docker.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert steps[-1] == "RUN python -I -S /tmp/verify_pytest.py check", (
        "the last build step must check the pytest set against its snapshot, under -I -S", steps[-3:])
    assert "python -I -S /tmp/verify_pytest.py snapshot" in docker, "the pytest set must be recorded after its install"
    assert re.search(r"python -I -S -c \"import pathlib, site", docker), "the work.pth line must run under -I -S"
    assert "COPY verify_pytest.py /tmp/verify_pytest.py" in docker, "the verifier must be copied into the image"
    assert pathlib.Path(sandbox.VERIFIER).is_file(), "the build context is built from this file"


def _copy_pytest_set(root, dest=None):
    """A copy of this venv's pytest set laid out as RECORD describes it, plus a constraints file for it."""
    import csv, importlib.metadata as md, os, pathlib, shutil
    # RECORD rows may climb out of site-packages (console scripts), so the default leaves room for that
    site = pathlib.Path(dest) if dest else root / "d" / "d" / "d" / "site"
    constraints, lines = root / "constraints.txt", []
    for name in ["pytest", "pluggy", "iniconfig", "packaging", "pygments"]:
        dist = md.distribution(name)
        lines.append(f"{name}=={dist.version}")
        for row in csv.reader(dist.read_text("RECORD").splitlines()):
            src = pathlib.Path(dist.locate_file(row[0]))
            if src.is_file():
                dst = pathlib.Path(os.path.normpath(site / row[0]))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    constraints.parent.mkdir(parents=True, exist_ok=True)
    constraints.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return site, constraints


def _verifier(mode, constraints, snapshot, site, python=None):
    """The verifier the way the Dockerfile runs it (-I -S), against an explicit site directory."""
    cmd = [python or sys.executable, "-I", "-S", str(sandbox.VERIFIER), mode, "--constraints", str(constraints),
           "--snapshot", str(snapshot), "--site", str(site)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _recorded(root, prepare=None):
    """A copy of the pytest set with a snapshot of it, taken as the Dockerfile takes it: right after the
    pytest install and before the requirements. prepare(site) may change the copy before the snapshot."""
    site, constraints = _copy_pytest_set(root)
    if prepare:
        prepare(site)
    snapshot = root / "snapshot.json"
    made = _verifier("snapshot", constraints, snapshot, site)
    assert made.returncode == 0, made.stderr
    return site, constraints, snapshot


def _pytest_dist_info(site):
    return next(site.glob("pytest-*.dist-info"))


def test_the_pytest_set_verifier_passes_clean_and_names_what_a_requirement_changed():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())

    site, constraints, snap = _recorded(root / "clean")
    ok = _verifier("check", constraints, snap, site)
    assert ok.returncode == 0, ok.stderr

    site, constraints, snap = _recorded(root / "clobbered")
    (site / "_pytest" / "__init__.py").write_text('raise ImportError("CLOBBERED")' + NL, encoding="utf-8")
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and "_pytest/__init__.py: changed" in bad.stderr, bad.stderr

    site, constraints, snap = _recorded(root / "added")
    (site / "_pytest" / "extra_evil.py").write_text("x = 1" + NL, encoding="utf-8")
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and "_pytest/extra_evil.py: added" in bad.stderr, bad.stderr

    site, constraints, snap = _recorded(root / "deleted")
    (site / "pluggy" / "__init__.py").unlink()
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and "pluggy/__init__.py: missing" in bad.stderr, bad.stderr

    site, constraints = _copy_pytest_set(root / "nopin")
    constraints.write_text("pluggy==1.0\n", encoding="utf-8")
    bad = _verifier("snapshot", constraints, root / "nopin" / "snapshot.json", site)
    assert bad.returncode == 1 and "pins no pytest" in bad.stderr, bad.stderr


def test_a_file_added_to_the_dist_info_folder_is_caught():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    site, constraints, snap = _recorded(root / "dist")
    (_pytest_dist_info(site) / "entry_points_extra.txt").write_text("[console_scripts]" + NL, encoding="utf-8")
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and "entry_points_extra.txt: added" in bad.stderr, bad.stderr


def test_a_forged_record_does_not_hide_a_changed_file():
    """A wheel can ship its own RECORD. The check compares the disk with the snapshot, so a RECORD that
    matches the altered file changes nothing."""
    import base64, csv, hashlib, io, pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    site, constraints, snap = _recorded(root / "forged")
    target = site / "_pytest" / "__init__.py"
    target.write_text('raise ImportError("FORGED")' + NL, encoding="utf-8")
    data = target.read_bytes()
    sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    record = _pytest_dist_info(site) / "RECORD"
    rows = [["_pytest/__init__.py", f"sha256={sha}", str(len(data))] if row[:1] == ["_pytest/__init__.py"] else row
            for row in csv.reader(record.read_text(encoding="utf-8").splitlines())]
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    record.write_text(text.getvalue(), encoding="utf-8")
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and "_pytest/__init__.py: changed" in bad.stderr, bad.stderr


def test_bytecode_is_checked_like_any_other_file():
    """A .pyc has no hash in RECORD, so it is checked against the snapshot; a new one is an addition."""
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    pyc = pathlib.PurePosixPath("_pytest/__pycache__/__init__.cpython-312.pyc")

    def plant(site):
        (site / pyc.parent).mkdir(exist_ok=True)
        (site / pyc).write_bytes(b"genuine")

    site, constraints, snap = _recorded(root / "replaced", plant)
    (site / pyc).write_bytes(b"CLOBBERED")
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and f"{pyc}: changed" in bad.stderr, bad.stderr

    site, constraints, snap = _recorded(root / "added")
    (site / "_pytest" / "__pycache__").mkdir(exist_ok=True)
    (site / "_pytest" / "__pycache__" / "junk.cpython-312.pyc").write_bytes(b"junk")
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and "_pytest/__pycache__/junk.cpython-312.pyc: added" in bad.stderr, bad.stderr


def _plant_listed_top_level_pyc(site):
    """pytest ships a top-level py.py, so its RECORD lists __pycache__/py.cpython-312.pyc."""
    (site / "__pycache__").mkdir(exist_ok=True)
    (site / "__pycache__" / "py.cpython-312.pyc").write_bytes(b"genuine")
    record = _pytest_dist_info(site) / "RECORD"
    record.write_text(record.read_text(encoding="utf-8").rstrip(NL) + NL + "__pycache__/py.cpython-312.pyc,," + NL,
                      encoding="utf-8")


def test_a_top_level_module_does_not_make_the_shared_pycache_a_watched_folder():
    """The shared top-level __pycache__ is no dist's own folder, so a requirement's top-level module and its
    .pyc there are not additions to pytest's set."""
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    site, constraints, snap = _recorded(root / "shared", _plant_listed_top_level_pyc)
    (site / "six.py").write_text("__version__ = '1.17.0'" + NL, encoding="utf-8")
    (site / "__pycache__" / "six.cpython-312.pyc").write_bytes(b"six")
    ok = _verifier("check", constraints, snap, site)
    assert ok.returncode == 0, ok.stderr


def test_a_listed_pyc_in_the_shared_pycache_is_still_checked():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    site, constraints, snap = _recorded(root / "changed", _plant_listed_top_level_pyc)
    (site / "__pycache__" / "py.cpython-312.pyc").write_bytes(b"CLOBBERED")
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and "__pycache__/py.cpython-312.pyc: changed" in bad.stderr, bad.stderr


def test_a_listed_file_outside_site_packages_is_checked_too():
    import csv, os, pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    site, constraints, snap = _recorded(root / "script")
    listed = [os.path.normpath(site / row[0]) for row in csv.reader(
        _pytest_dist_info(site).joinpath("RECORD").read_text(encoding="utf-8").splitlines())
        if row and row[0].startswith("..")]
    victim = next(path for path in listed if os.path.isfile(path))  # a console script, e.g. Scripts/pytest.exe
    pathlib.Path(victim).write_bytes(b"CLOBBERED")
    key = pathlib.PurePath(os.path.relpath(victim, site)).as_posix()
    bad = _verifier("check", constraints, snap, site)
    assert bad.returncode == 1 and f"{key}: changed" in bad.stderr, bad.stderr


def test_a_pth_file_cannot_fake_a_pass_under_dash_s():
    """Python runs .pth files in site-packages at start-up unless -S is given. A .pth in a venv that exits 0
    fakes a pass for a check run without -S, and the same check run with -S still fails."""
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    venv = root / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = subprocess.run([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                             capture_output=True, text=True, check=True).stdout.strip()
    site, constraints = _copy_pytest_set(root / "pth", pathlib.Path(purelib))
    snap = root / "pth" / "snapshot.json"
    made = _verifier("snapshot", constraints, snap, site, python=str(python))
    assert made.returncode == 0, made.stderr
    (site / "_pytest" / "__init__.py").write_text('raise ImportError("CLOBBERED")' + NL, encoding="utf-8")
    (site / "evil.pth").write_text(
        'import os, sys; sys.argv[0].endswith("verify_pytest.py") and os._exit(0)' + NL, encoding="utf-8")
    fooled = subprocess.run([str(python), "-I", str(sandbox.VERIFIER), "check", "--constraints", str(constraints),
                             "--snapshot", str(snap), "--site", str(site)], capture_output=True, text=True)
    assert fooled.returncode == 0, "the planted .pth must fake a pass without -S, or this test proves nothing"
    bad = _verifier("check", constraints, snap, site, python=str(python))
    assert bad.returncode == 1 and "_pytest/__init__.py: changed" in bad.stderr, bad.stderr


def test_the_image_tag_follows_the_verifier_script():
    import pathlib, tempfile
    script = pathlib.Path(tempfile.mkdtemp()) / "verify_pytest.py"
    text = pathlib.Path(sandbox.VERIFIER).read_text(encoding="utf-8").replace("\r\n", "\n")
    script.write_bytes(text.encode("utf-8"))
    with mock.patch.object(sandbox, "VERIFIER", script):
        tag = sandbox.image_tag(["six==1.17.0"])
        script.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        assert sandbox.image_tag(["six==1.17.0"]) == tag, "a Windows checkout (CRLF) must name the same image"
        script.write_bytes((text + "# an edit" + NL).encode("utf-8"))
        assert sandbox.image_tag(["six==1.17.0"]) != tag, "editing the verifier must build a new image"


def test_the_image_tag_follows_the_dockerfile():
    import pathlib, tempfile
    recipe = pathlib.Path(tempfile.mkdtemp()) / "Dockerfile"
    text = pathlib.Path(sandbox.DOCKERFILE).read_text(encoding="utf-8").replace("\r\n", "\n")
    recipe.write_bytes(text.encode("utf-8"))
    with mock.patch.object(sandbox, "DOCKERFILE", recipe):
        tag = sandbox.image_tag(["six==1.17.0"])
        assert sandbox.image_tag(["six==1.17.0"]) == tag and tag.startswith("agent-loop-proj-")
        recipe.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        assert sandbox.image_tag(["six==1.17.0"]) == tag, "a Windows checkout (CRLF) must name the same image"
        recipe.write_bytes((text + "# an edit" + NL).encode("utf-8"))
        edited = sandbox.image_tag(["six==1.17.0"])
        assert edited != tag, "editing the Dockerfile must build a new image, not reuse the stale one"
        assert sandbox.image_tag(["six==1.16.0"]) != edited, "the requirements still matter"


def test_names_that_could_replace_pytest_or_run_at_startup_are_protected():
    import pathlib, tempfile
    p = workspace.DEFAULT_PROTECTED
    for name in ["pytest.py", "pkg/pytest.py", "_pytest/__init__.py", "_pytest/python.py", "sitecustomize.py",
                 "usercustomize.py", "SiteCustomize.py", "evil.pth", "pkg/x.pth"]:
        assert workspace.is_protected(name, p), name
    for free in ["mypytest.py", "pytesting.py", "sitecustomize_helpers.py", "my_pytest/util.py", "pth.py", "src/app.py"]:
        assert not workspace.is_protected(free, p), free
    ws = _project(pathlib.Path(tempfile.mkdtemp()) / "ws", {"app.py": "x = 0" + NL})
    reply = NL.join(FENCE + name + NL + "import os" + NL + FENCE for name in
                    ["sitecustomize.py", "pytest.py", "_pytest/hook.py", "evil.pth", "app.py"])
    result = workspace.apply_blocks(ws, reply)
    assert result.written == ["app.py"] and len(result.skipped) == 4 and not result.errors, result
    assert not (ws / "sitecustomize.py").exists() and not (ws / "pytest.py").exists()


def test_a_repeat_limit_abort_is_retried_once_but_other_unknown_errors_are_not():
    from agent_loop import trace
    abort = "prediction aborted, token repeat limit reached"
    assert trace._transient(_ServerError(abort, -1))
    assert not trace._transient(_ServerError("some other failure", -1))
    assert not trace._transient(_ServerError(abort, 400)), "only the -1 abort counts"
    assert not trace._transient(_ServerError(abort, None)) and not trace._transient(ValueError(abort))
    model = _FlakyLLM([_ServerError(abort, -1)])
    with mock.patch.object(trace, "RETRY_WAIT", 0):
        resp = trace.llm_call("worker", model, "prompt")
    assert resp.content == "fine" and model.calls == 2
    model = _FlakyLLM([_ServerError(abort, -1), _ServerError(abort, -1)])
    with mock.patch.object(trace, "RETRY_WAIT", 0):
        try:
            trace.llm_call("worker", model, "prompt")
        except _ServerError:
            pass
        else:
            raise AssertionError("a second abort must propagate")
    assert model.calls == 2


def test_real_docker_planted_startup_code_cannot_fake_a_pass_but_the_old_command_could():
    """The attack: files in the project that load before pytest and force a green result."""
    import pathlib, tempfile
    if not _require_docker():
        return
    root = pathlib.Path(tempfile.mkdtemp())
    files = {
        "calc.py": "def add(a, b):" + NL + "    return a - b" + NL,
        "test_calc.py": "from calc import add" + NL + "def test_add():" + NL + "    assert add(2, 3) == 5" + NL,
        "src/mylib/__init__.py": "VALUE = 7" + NL,
        "test_lib.py": "from mylib import VALUE" + NL + "def test_value():" + NL + "    assert VALUE == 7" + NL,
        # planted: any of these, if loaded ahead of the real ones, makes the run exit 0
        "sitecustomize.py": "import atexit, os" + NL + "atexit.register(lambda: os._exit(0))" + NL,
        "pytest.py": "raise SystemExit(0)" + NL,
        "_pytest/__init__.py": "raise SystemExit(0)" + NL,
    }
    ws = _project(root / "ws", files)
    with _docker_mode():
        ok, out = sandbox.run_project_tests(ws, "pytest -q", [])
        assert not ok and "1 failed" in out and "1 passed" in out, out      # the real verdict, src-layout import works
        honest = _project(root / "honest", {**files, "calc.py": "def add(a, b):" + NL + "    return a + b" + NL})
        ok, out = sandbox.run_project_tests(honest, "pytest -q", [])
        assert ok and "2 passed" in out, out                                 # the planted files do not break a good run

        # the same planted files DO fool the old command (project first on PYTHONPATH, bare `pytest`)
        image = sandbox.ensure_image([])
        new = sandbox.project_docker_cmd("planted-old", ws, image, "pytest -q")
        i = new.index(image)
        old = new[:i] + ["-e", "PYTHONPATH=/work:/work/src", image, "pytest", "-p", "no:cacheprovider", "-q"]
        fooled = subprocess.run(old, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
        assert fooled.returncode == 0, "the attack must work against the old command, or this test proves nothing"
    left = subprocess.run(["docker", "ps", "-a", "--filter", "name=agent-loop-", "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.split()
    assert not left, left


def test_eval_fixtures_match_their_pin():
    from agent_loop import fixtures
    assert fixtures.pinned() is not None, "evals/FIXTURES.sha256 is missing"
    assert fixtures.digest() == fixtures.pinned(), (
        "an eval fixture changed, so earlier eval reports are no longer comparable; if that is intended run "
        "python -m agent_loop.fixtures --update")


def test_fixture_digest_notices_real_edits_but_not_line_endings_or_caches():
    import pathlib, shutil, tempfile
    from agent_loop import fixtures
    root = pathlib.Path(tempfile.mkdtemp()) / "evals"
    shutil.copytree(fixtures.EVALS, root, ignore=shutil.ignore_patterns("__pycache__"))
    base = fixtures.digest(root)
    assert base == fixtures.digest(fixtures.EVALS), "a copy of the fixtures hashes the same"

    target = root / "projects" / "csvrow" / "tests" / "test_parse.py"
    original = target.read_bytes()
    target.write_bytes(original.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert fixtures.digest(root) == base, "a Windows checkout (CRLF) must hash like a Linux one"
    target.write_bytes(original)

    (root / "projects" / "csvrow" / "__pycache__").mkdir()
    (root / "projects" / "csvrow" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"junk")
    (root / "projects" / "csvrow" / "stray.pyc").write_bytes(b"junk")
    (root / "FIXTURES.sha256").write_text("anything", encoding="utf-8")
    assert fixtures.digest(root) == base, "caches, bytecode and the pin file itself are ignored"

    target.write_bytes(original + b"# edit" + b"\n")
    assert fixtures.digest(root) != base, "a changed test must change the digest"
    target.write_bytes(original)
    (root / "new.yaml").write_text("task: x", encoding="utf-8")
    assert fixtures.digest(root) != base, "a new task must change the digest"
    (root / "new.yaml").unlink()
    shutil.move(str(root / "fizzbuzz.yaml"), str(root / "fizzbuzz2.yaml"))
    assert fixtures.digest(root) != base, "a renamed file must change the digest"
    shutil.move(str(root / "fizzbuzz2.yaml"), str(root / "fizzbuzz.yaml"))
    assert fixtures.digest(root) == base, "undoing every change restores the digest"


def _cfg(tmp, name, text):
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def test_model_options_are_whitelisted_and_type_checked():
    import pathlib, tempfile
    from agent_loop import config
    tmp = pathlib.Path(tempfile.mkdtemp())
    base = "models:" + NL + "  escalation: ollama:qwen3:8b" + NL + "  worker: ollama:qwen2.5-coder:7b" + NL

    ok = config.load_config(_cfg(tmp, "ok.yaml", base + "model_options:" + NL + "  escalation:" + NL +
                                 "    reasoning: false" + NL + "    num_predict: 4096" + NL + "    temperature: 0.2" + NL), env={})
    assert ok["model_options"] == {"escalation": {"reasoning": False, "num_predict": 4096, "temperature": 0.2}}
    assert config.load_config(_cfg(tmp, "none.yaml", base), env={})["model_options"] == {}
    default = config.load_config(_cfg(tmp, "null.yaml", base + "model_options:" + NL + "  escalation:" + NL +
                                      "    reasoning: null" + NL), env={})
    assert default["model_options"] == {"escalation": {"reasoning": None}}, "null means the model's own default"

    for bad in (
        "    think: false",                      # not an allowed option (typo for reasoning)
        "    reasoning: 'yes'", "    reasoning: 1", "    reasoning: 0", "    reasoning: low",
        "    num_predict: 0", "    num_predict: 99999", "    num_predict: true", "    num_predict: '5'", "    num_predict: 1.5",
        "    temperature: -1", "    temperature: 3", "    temperature: hot",
        "    base_url: http://evil.example", "    num_ctx: 99999999", "    keep_alive: -1", "    seed: 1",
    ):
        try:
            config.load_config(_cfg(tmp, "bad.yaml", base + "model_options:" + NL + "  escalation:" + NL + bad + NL), env={})
        except ValueError as exc:
            assert "escalation" in str(exc), exc
        else:
            raise AssertionError(f"accepted {bad.strip()}")
    for body in ("model_options:" + NL + "  ghost:" + NL + "    reasoning: false",       # role with no model
                 "model_options:" + NL + "  escalation: nope",                           # options must be a mapping
                 "models:" + NL + "  escalation: other:model" + NL + "model_options:" + NL + "  escalation:" + NL + "    reasoning: false"):
        try:
            config.load_config(_cfg(tmp, "bad2.yaml", (base if "models:" not in body else "") + body + NL), env={})
        except ValueError:
            continue
        raise AssertionError(body)


def test_an_experiment_patch_layers_over_the_real_config_key_by_key():
    import pathlib, tempfile
    from agent_loop import config
    tmp = pathlib.Path(tempfile.mkdtemp())
    base = _cfg(tmp, "base.yaml", NL.join([
        "models:", "  escalation: ollama:qwen3:8b", "  worker: ollama:qwen2.5-coder:7b",
        "model_options:", "  escalation:", "    reasoning: true", "    num_predict: 100", "max_rejections: 5", ""]))
    patch = _cfg(tmp, "patch.yaml", NL.join(["model_options:", "  escalation:", "    reasoning: false", ""]))
    plain = config.load_config(base, env={})
    layered = config.load_config(base, env={"AGENT_LOOP_CONFIG": str(patch)})
    assert plain["override"] is None and layered["override"] == "patch.yaml"
    assert layered["model_options"] == {"escalation": {"reasoning": False, "num_predict": 100}}, "merged per option"
    assert layered["models"] == plain["models"] and layered["max_rejections"] == 5, "everything else comes from config.yaml"
    assert plain["model_options"]["escalation"]["reasoning"] is True, "the base is not changed by layering"

    swap = _cfg(tmp, "swap.yaml", "models:" + NL + "  escalation: ollama:other:3b" + NL)
    assert config.load_config(base, env={"AGENT_LOOP_CONFIG": str(swap)})["models"]["escalation"] == "ollama:other:3b"
    empty = _cfg(tmp, "empty.yaml", "# nothing\\n")
    assert config.load_config(base, env={"AGENT_LOOP_CONFIG": str(empty)})["model_options"] == plain["model_options"]
    for text in ("- a list", "just text"):
        try:
            config.load_config(base, env={"AGENT_LOOP_CONFIG": str(_cfg(tmp, "odd.yaml", text))})
        except (ValueError, AttributeError):
            continue
        raise AssertionError(text)
    # a patch cannot slip past the safety gates
    unsafe = _cfg(tmp, "unsafe.yaml", "sandbox: local" + NL)
    try:
        config.load_config(base, env={"AGENT_LOOP_CONFIG": str(unsafe)})
    except ValueError as exc:
        assert "NO isolation" in str(exc), exc
    else:
        raise AssertionError("an experiment patch must not be able to turn the sandbox off")
    try:
        config.load_config(base, env={"AGENT_LOOP_CONFIG": str(tmp / "missing.yaml")})
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing patch must be an error, not a silent control run")


def test_prompt_variants_are_whitelisted_and_checked():
    import pathlib, tempfile
    from agent_loop import config
    tmp = pathlib.Path(tempfile.mkdtemp())
    base = "models:" + NL + "  worker: ollama:qwen2.5-coder:7b" + NL
    assert config.load_config(_cfg(tmp, "pv0.yaml", base), env={})["prompt_variants"] == [], "the default is none"
    ok = config.load_config(_cfg(tmp, "pv1.yaml", base + "prompt_variants: [real_path_example, no_plan]" + NL), env={})
    assert ok["prompt_variants"] == ["real_path_example", "no_plan"], ok["prompt_variants"]
    replaced = config.load_config(_cfg(tmp, "pv2.yaml", base + "prompt_variants: [no_plan]" + NL),
                                  env={"AGENT_LOOP_CONFIG": str(_cfg(tmp, "pv3.yaml", "prompt_variants: [real_path_example]" + NL))})
    assert replaced["prompt_variants"] == ["real_path_example"], "a patch's list replaces the file's list"
    assert config.DEFAULTS["prompt_variants"] == [], "layering must never change the shared defaults"
    for bad in ("prompt_variants: [no_plna]", "prompt_variants: [real_path_example, typo]",
                "prompt_variants: real_path_example", "prompt_variants:", "prompt_variants: [1]",
                "prompt_variants: [[no_plan]]", "prompt_variants: [null]"):
        try:
            config.load_config(_cfg(tmp, "pv-bad.yaml", base + bad + NL), env={})
        except ValueError as exc:
            assert "real_path_example" in str(exc) and "no_plan" in str(exc), exc
        else:
            raise AssertionError(f"accepted {bad}")


def test_csvrow_experiment_patches_load_with_their_values():
    import pathlib
    from agent_loop import config
    experiments = pathlib.Path(__file__).resolve().parent / "experiments"
    base = config.load_config(config.CONFIG_PATH, env={})
    want = {
        "csvrow-path": {"prompt_variants": ["real_path_example"]},
        "csvrow-noplan": {"prompt_variants": ["no_plan"]},
        "csvrow-qwen3-worker": {"models": {"worker": "ollama:qwen3:8b"},
                                "model_options": {"worker": {"reasoning": False}}},
    }
    for name, changes in want.items():
        cfg = config.load_config(config.CONFIG_PATH, env={"AGENT_LOOP_CONFIG": str(experiments / f"{name}.yaml")})
        assert cfg["override"] == f"{name}.yaml", name
        assert cfg["prompt_variants"] == changes.get("prompt_variants", []), name
        assert cfg["models"] == {**base["models"], **changes.get("models", {})}, name
        assert cfg["model_options"] == {**base["model_options"], **changes.get("model_options", {})}, name
        rest = {k: v for k, v in cfg.items() if k not in ("models", "model_options", "prompt_variants", "override")}
        assert rest == {k: v for k, v in base.items() if k not in ("models", "model_options", "prompt_variants", "override")}, name


def test_options_reach_only_the_roles_that_list_them_and_reports_record_them():
    from agent_loop import config
    thinker = config.make_llm("ollama:qwen3:8b", {"reasoning": False, "num_predict": 4096, "temperature": 0.1})
    assert thinker.reasoning is False and thinker.num_predict == 4096 and thinker.temperature == 0.1
    plain = config.make_llm("ollama:gemma3:4b")
    assert plain.reasoning is None and plain.num_predict is None, "a model with no options gets no think setting"
    assert plain.num_ctx == config.CONFIG["num_ctx"]
    assert config.make_llm("ollama:x:1b", {}).reasoning is None
    assert set(agent_nodes.MODELS) >= {"lead", "worker", "reviewer"}
    report = evals.new_report("opts", 1, None)
    assert report["config"]["model_options"] == config.CONFIG["model_options"]
    assert "override" in report["config"]


def test_reports_record_the_prompt_variants_that_ran():
    with _variants():
        assert evals.new_report("plain", 1, None)["config"]["prompt_variants"] == [], "the default records none"
    with _variants("no_plan"):
        assert evals.new_report("csvrow", 1, None)["config"]["prompt_variants"] == ["no_plan"]


def test_the_pre_run_line_names_prompt_variants_and_is_otherwise_unchanged():
    nothing = {"model_options": {}, "override": None, "prompt_variants": []}
    assert evals.config_line(nothing) == "", "nothing set: nothing printed"
    patched = {"model_options": {"escalation": {"reasoning": False}}, "override": "think-off.yaml",
               "prompt_variants": []}
    assert evals.config_line(patched) == "options: {'escalation': {'reasoning': False}}   patch: think-off.yaml", \
        "without variants the line is the old text exactly"
    assert evals.config_line({"model_options": {}, "override": "x.yaml", "prompt_variants": []}) == \
        "options: {}   patch: x.yaml"
    variant = evals.config_line({"model_options": {}, "override": None, "prompt_variants": ["no_plan"]})
    assert "prompt variants: ['no_plan']" in variant, variant
    assert variant == "options: {}   patch: None   prompt variants: ['no_plan']", variant
    both = evals.config_line({**patched, "prompt_variants": ["real_path_example", "no_plan"]})
    assert both.endswith("   prompt variants: ['real_path_example', 'no_plan']"), both


def test_the_shipped_experiment_files_change_only_what_they_say():
    import pathlib
    from agent_loop import config
    experiments = pathlib.Path(__file__).resolve().parent / "experiments"
    base = config.load_config(config.CONFIG_PATH, env={})
    results = {p.stem: config.load_config(config.CONFIG_PATH, env={"AGENT_LOOP_CONFIG": str(p)})
               for p in sorted(experiments.glob("*.yaml"))}
    assert sorted(results) == ["control", "csvrow-noplan", "csvrow-path", "csvrow-qwen3-worker",
                               "think-capped", "think-default", "think-off"], sorted(results)
    # the adopted setting: the experiment found thinking off as good and about three times faster
    assert base["model_options"] == {"escalation": {"reasoning": False}}, base["model_options"]
    assert base["prompt_variants"] == [], "the default has no prompt variant"
    for name, cfg in results.items():
        if name not in ("csvrow-path", "csvrow-noplan"):  # pinned exactly by the csvrow test above
            assert cfg["prompt_variants"] == [], f"{name} must not switch on a prompt variant"
        if name != "csvrow-qwen3-worker":
            assert cfg["models"] == base["models"], f"{name} must not change models"
        assert {k: v for k, v in cfg.items() if k not in ("model_options", "models", "prompt_variants", "override")} == \
            {k: v for k, v in base.items() if k not in ("model_options", "models", "prompt_variants", "override")}, \
            f"{name} changes more than model options, models and prompt variants"
    assert results["control"]["model_options"] == base["model_options"]
    assert results["think-off"]["model_options"]["escalation"] == {"reasoning": False}
    assert results["think-default"]["model_options"]["escalation"] == {"reasoning": None}, "re-creates the old control"
    assert results["think-capped"]["model_options"]["escalation"] == {"reasoning": True, "num_predict": 4096}
    assert all(set(cfg["model_options"]) <= {"escalation"} for name, cfg in results.items()
               if not name.startswith("csvrow-")), "only the escalation role is touched"
    assert set(results["csvrow-qwen3-worker"]["model_options"]) == {"escalation", "worker"}


def test_strip_think_removes_reasoning_but_nothing_else():
    strip = agent_nodes.strip_think
    assert strip("plain answer") == "plain answer"
    assert strip("<think>hmm</think>answer") == "answer"
    assert strip("<think>a" + NL + "b</think>" + NL + NL + "answer" + NL) == "answer"
    assert strip("<think>one</think>mid<think>two</think>end") == "midend"
    assert strip("<THINK>x</THINK>ok") == "ok"
    assert strip("") == "" and strip(None) == ""
    fenced = "`" * 3 + "calc.py" + NL + "x = 1" + NL + "`" * 3
    assert strip("<think>never finished " + fenced) == "", "a reply cut off mid-thought has no answer"
    assert strip("reasoning only</think>the answer") == "the answer", "a lone closing tag ends the reasoning"
    assert strip("<think>" + fenced + "</think>" + fenced) == fenced, "only the answer's blocks survive"
    assert strip("a < b and c > d, no tags here") == "a < b and c > d, no tags here"


def test_file_blocks_inside_reasoning_are_never_applied():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    draft = FENCE + "scratch_notes.py" + NL + "print('draft')" + NL + FENCE
    wrong = FENCE + "calc.py" + NL + "def add(a, b):" + NL + "    return 0" + NL + FENCE
    reply = "<think>First I will try this:" + NL + draft + NL + wrong + NL + "no, that is wrong</think>" + NL + FIX
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _fake_project_tests([])):
        orig, values = _project_run(root, worker=SeqLLM([reply]), thread="thinks")
    ws = root / "runs" / "thinks" / "workspace"
    assert values["approved"] and values["rejections"] == 0, values
    assert not (ws / "scratch_notes.py").exists(), "a draft inside the reasoning must not be written"
    assert "a + b" in (ws / "calc.py").read_text(encoding="utf-8")
    assert values["changed"] == ["calc.py"] and "<think>" not in values["work"], values["changed"]
    raw = next(e for e in _events(root / "runs", "thinks") if e["event"] == "llm_end" and e["role"] == "worker")["reply"]
    assert raw.startswith("<think>"), "the trace keeps the full reply so the thinking stays visible"


def test_a_reply_cut_off_inside_its_reasoning_is_an_ordinary_rejection():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    worker = SeqLLM(["<think>still thinking about it " + FIX, FIX])          # lap 1 never closed its <think>
    with mock.patch.object(agent_nodes.sandbox, "run_project_tests", _fake_project_tests([])):
        _, values = _project_run(root, worker=worker, thread="cutoff")
    assert values["approved"] and values["rejections"] == 1, values
    assert "no file blocks" in worker.prompts[1], "the worker is told it produced no files"


def test_lead_and_reviewer_replies_are_read_without_their_reasoning():
    install(review="<think>the tests passed, so</think>APPROVED: fine")
    agent_nodes.MODELS["lead"] = FakeLLM("<think>plan the plan</think>1. The real plan")
    values = cli.run("is_palindrome", tests=checks.TESTS, thread="clean-roles")
    assert values["approved"] is True and values["plan"] == "1. The real plan", values
    assert values["review"].startswith("APPROVED"), values["review"]


def _variants(*names):
    """Pin the prompt switches for one block, so the suite never depends on AGENT_LOOP_CONFIG."""
    return mock.patch.dict(agent_nodes.CONFIG, {"prompt_variants": list(names)})


# The default prompts exactly as HEAD builds them (the same workspace and state as the test below).
# A change to any character, including a trailing space, must show up here.
PROJECT_WORKER_PROMPT = """Task: Fix parse_row
Plan: 1. Fix it

Project files:
- tests/test_parse.py
- csvrow/parse.py

### tests/test_parse.py
```
from csvrow.parse import parse_row

```

### csvrow/parse.py
```
def parse_row(line):
    return []

```

Last review: REJECTED: quoted fields are split at the comma
Last test run:
FAILED tests/test_parse.py::test_quoted - assert [] == ['a,b']

Change the project so the task is done and its tests pass. Reply with one fenced block per file you change or create. The fence's info string is the file path and the block holds the COMPLETE new file, for example:
```pkg/module.py
<entire file>
```
Use paths relative to the project root, only files inside the project, and no other fenced blocks.
Test files and test configuration (test_*.py, tests/, conftest.py, pytest.ini, pyproject.toml, ...) are read-only: fix the code so the existing tests pass."""

CODE_WORKER_PROMPT = ("Task: is_palindrome" + NL + "Plan: 1. Clean the string" + NL + "2. Compare it reversed" + NL
                      + "Last review: REJECTED: fails on the empty string" + NL + "Write the solution.")

RESEARCH_WORKER_PROMPT = """Task: What do the papers show?
Plan: 1. Read the abstracts
Papers (your ONLY allowed source):
[arXiv:2401.00001] A Paper (2024-01-02; A. One, B. Two, C. Three)
Abstract: We show a result that is quoted exactly here.

Last review: REJECTED: one claim has no quote
Last citation check: unsupported claim
Answer the task using only these papers. Support every claim by citing the paper as [arXiv:ID] and copying an exact phrase of at least 4 words from its abstract in double quotes in the same paragraph. Every paper you cite needs at least one such quote. Do not use outside knowledge and do not cite other papers."""


def test_default_worker_prompt_and_lead_call_are_unchanged():
    import pathlib, tempfile
    ws = _project(pathlib.Path(tempfile.mkdtemp()) / "ws",
                  {"csvrow/parse.py": "def parse_row(line):" + NL + "    return []" + NL,
                   "tests/test_parse.py": "from csvrow.parse import parse_row" + NL})
    state = {"task": "Fix parse_row", "plan": "1. Fix it", "type": "project", "workspace": str(ws),
             "review": "REJECTED: quoted fields are split at the comma",
             "test_output": "FAILED tests/test_parse.py::test_quoted - assert [] == ['a,b']"}
    lead = SeqLLM(["1. Fix it"])
    agent_nodes.MODELS["lead"] = lead
    with _variants():
        out = agent_nodes.lead({"task": "Fix parse_row"})
        prompt = agent_nodes._worker_prompt(state)
    assert lead.prompts == ["Task: Fix parse_row" + NL
                            + "Write a short numbered plan (max 6 steps) to solve the task. No code, no preamble."], \
        lead.prompts
    assert lead.calls == 1 and out["plan"] == "1. Fix it" and out["tokens"]["lead"]["calls"] == 1, out
    assert prompt == PROJECT_WORKER_PROMPT, prompt


def test_code_and_research_worker_prompts_are_unchanged():
    code ={"task": "is_palindrome", "plan": "1. Clean the string" + NL + "2. Compare it reversed",
            "review": "REJECTED: fails on the empty string"}
    sources = [{"id": "2401.00001", "title": "A Paper", "published": "2024-01-02",
                "authors": ["A. One", "B. Two", "C. Three", "D. Four"],
                "abstract": "We show a result that is quoted exactly here."}]
    research_state = {"task": "What do the papers show?", "plan": "1. Read the abstracts", "type": "research",
                      "review": "REJECTED: one claim has no quote", "test_output": "unsupported claim",
                      "sources": sources}
    with _variants():
        assert agent_nodes._worker_prompt(code) == CODE_WORKER_PROMPT, agent_nodes._worker_prompt(code)
        assert agent_nodes._worker_prompt(research_state) == RESEARCH_WORKER_PROMPT, \
            agent_nodes._worker_prompt(research_state)


def test_real_path_example_names_a_workspace_file_the_worker_may_edit():
    import pathlib, tempfile
    root = pathlib.Path(tempfile.mkdtemp())
    ws = _project(root / "ws", {"csvrow/parse.py": "def parse_row(line):" + NL + "    return []" + NL,
                                "tests/test_parse.py": "from csvrow.parse import parse_row" + NL})
    state = {"task": "Fix parse_row", "plan": "1. Fix it", "type": "project", "workspace": str(ws)}
    with _variants():
        default = agent_nodes._worker_prompt(state)
    with _variants("real_path_example"):
        prompt = agent_nodes._worker_prompt(state)
    assert "```csvrow/parse.py" + NL in prompt and "pkg/module.py" not in prompt, prompt
    assert "```tests/" not in prompt, "a test file is never the example"
    assert prompt == default.replace("```pkg/module.py", "```csvrow/parse.py"), "only the example path may change"

    # no non-protected .py file: the example falls back to the old placeholder
    for i, files in enumerate([
        {"tests/test_parse.py": "x" + NL, "conftest.py": "y" + NL, "README.md": "z" + NL},
        {"tests/test_parse.py": "x" + NL, "notes.txt": "z" + NL},
    ]):
        empty = _project(root / f"fallback{i}", files)
        with _variants("real_path_example"):
            fallback = agent_nodes._worker_prompt({**state, "workspace": str(empty)})
        assert "```pkg/module.py" + NL in fallback, fallback
    # several candidates: the first in sorted order wins
    many = _project(root / "many", {"zeta/app.py": "z = 1" + NL, "alpha/app.py": "a = 1" + NL})
    with _variants("real_path_example"):
        assert "```alpha/app.py" + NL in agent_nodes._worker_prompt({**state, "workspace": str(many)})
    # an __init__.py is never the example, even when it sorts first (pkg/__init__.py < pkg/module.py)
    pkg = _project(root / "pkg", {"pkg/__init__.py": "", "pkg/module.py": "x = 1" + NL})
    with _variants("real_path_example"):
        prompt = agent_nodes._worker_prompt({**state, "workspace": str(pkg)})
    assert "```pkg/module.py" + NL in prompt and "```pkg/__init__.py" not in prompt, prompt
    only_init = _project(root / "only-init", {"pkg/__init__.py": ""})
    with _variants("real_path_example"):
        assert "```pkg/module.py" + NL in agent_nodes._worker_prompt({**state, "workspace": str(only_init)})
    # the real csvrow fixture: csvrow/__init__.py sorts before csvrow/parse.py and must not be the example
    fixture = root / "csvrow-fixture"
    shutil.copytree(_REPO / "evals" / "projects" / "csvrow", fixture)
    with _variants("real_path_example"):
        prompt = agent_nodes._worker_prompt({**state, "workspace": str(fixture)})
    assert "```csvrow/parse.py" + NL in prompt and "```csvrow/__init__.py" not in prompt, prompt
    with _variants():
        assert "```pkg/module.py" + NL in agent_nodes._worker_prompt({**state, "workspace": str(fixture)})


def test_no_plan_makes_no_lead_call_and_the_worker_is_told_none():
    lead = SeqLLM(["1. Fix it"])
    agent_nodes.MODELS["lead"] = lead
    with _variants("no_plan"):
        out = agent_nodes.lead({"task": "Fix it"})
    assert lead.calls == 0, lead.calls
    assert out["plan"] == "none" and out["tokens"] == {"lead": {"input": 0, "output": 0, "total": 0, "calls": 0}}, out
    with _variants():
        out = agent_nodes.lead({"task": "Fix it"})
    assert lead.calls == 1 and out["plan"] == "1. Fix it", out
    # a whole code task with the variant on: no lead call, the run still goes through and is approved
    install()
    lead = agent_nodes.MODELS["lead"]
    with _variants("no_plan"):
        values = cli.run("is_palindrome", tests=checks.TESTS, thread="no-plan")
    assert lead.calls == 0 and values["plan"] == "none" and values["approved"] is True, values
    assert values["tokens"]["lead"]["calls"] == 0, values["tokens"]


def test_tests_keep_their_scratch_files_inside_the_project():
    made = pathlib.Path(tempfile.mkdtemp())
    with tempfile.TemporaryDirectory() as other:
        assert _SCRATCH in made.parents and _SCRATCH in pathlib.Path(other).parents, (made, other)
    assert _REPO in _SCRATCH.parents and _SCRATCH.parent.name == ".scratch"
    ignored = subprocess.run(["git", "check-ignore", "-q", str(_SCRATCH / "x")], cwd=_REPO)
    assert ignored.returncode in (0, 128), "the scratch folder must be gitignored"  # 128: git absent / no repo


def test_project_runs_refuse_to_make_a_workspace_nobody_looks_after():
    install()
    try:
        cli.run("Fix add()", kind="project", project=str(_REPO), thread="orphan")
    except ValueError as exc:
        assert "runs_dir" in str(exc), exc
    else:
        raise AssertionError("a project run without a runs_dir must not invent a temp workspace")


def test_non_cp1252_code_and_output_do_not_crash_the_check():
    """Windows decodes subprocess text with cp1252; model code can print anything."""
    snowman = chr(0x2603)
    code = "def f():" + NL + f"    print('{snowman} caf' + chr(0xe9) + ' ' + chr(0x1F600))" + NL + "    return 1"
    ok, out = checks.run_tests(code, ["assert f() == 1", "assert f() == 2"])
    assert not ok and "AssertionError" in out, out          # failed cleanly, did not raise
    ok, out = checks.run_tests(f"def f():{NL}    return '{snowman}'", ["assert f() == chr(0x2603)"])
    assert ok, out                                           # unicode source also survives stdin


def _docker_up():
    import shutil
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "info"], capture_output=True).returncode == 0


def _require_docker():
    """True when Docker is up. Down is a failure, so a green run means the sandbox was exercised;
    AGENT_LOOP_SKIP_DOCKER=1 is the explicit offline opt-out."""
    if _docker_up():
        return True
    if os.environ.get("AGENT_LOOP_SKIP_DOCKER") == "1":
        print("  (docker not available - skipped)")
        return False
    raise AssertionError("Docker is not running: the real-Docker tests would be skipped, not passed "
                         "(set AGENT_LOOP_SKIP_DOCKER=1 to skip on purpose)")


def test_real_docker_project_sandbox():
    """Fixture project: fails, passes after a fix, cannot write to /work, needs no network."""
    import pathlib, tempfile
    if not _require_docker():
        return
    root = pathlib.Path(tempfile.mkdtemp())
    ws = _project(root / "ws", {
        "calc.py": "def add(a, b):" + NL + "    return a - b" + NL,
        "test_calc.py": "from calc import add" + NL + "def test_add():" + NL + "    assert add(2, 3) == 5" + NL,
        "test_sandbox.py": NL.join([
            "import socket, pathlib",
            "def test_cannot_write_to_workspace():",
            "    try:",
            "        pathlib.Path('/work/pwned.txt').write_text('x')",
            "    except OSError:",
            "        return",
            "    raise AssertionError('workspace is writable')",
            "def test_no_network():",
            "    try:",
            "        socket.create_connection(('1.1.1.1', 53), timeout=2)",
            "    except OSError:",
            "        return",
            "    raise AssertionError('network reachable')", ""]),
    })
    with _docker_mode():
        ok, out = sandbox.run_project_tests(ws, "pytest -q", [])
        assert not ok and "1 failed" in out and "test_add" in out, out     # buggy add() is caught
        assert "test_cannot_write_to_workspace" not in out and "test_no_network" not in out, out

        workspace.apply_blocks(ws, "`" * 3 + "calc.py" + NL + "def add(a, b):" + NL + "    return a + b" + NL + "`" * 3)
        ok, out = sandbox.run_project_tests(ws, "pytest -q", [])
        assert ok and "3 passed" in out, out
    assert not (ws / "pwned.txt").exists() and not (ws / ".pytest_cache").exists()
    left = subprocess.run(["docker", "ps", "-a", "--filter", "name=agent-loop-", "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.split()
    assert not left, left


def test_docker_command_is_locked_down():
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(returncode=0, stdout="ALL TESTS PASSED", stderr="")

    with _docker_mode(), mock.patch.object(checks.subprocess, "run", fake_run):
        ok, out = checks.run_tests(checks.extract_code(GOOD))
    assert ok and "ALL TESTS PASSED" in out
    cmd, kw = calls[0]
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL",
                 "no-new-privileges", "--pids-limit", "--memory", "--user 65534"):
        assert flag in joined, (flag, joined)
    assert "-v" not in cmd and "--volume" not in cmd, "nothing may be mounted"
    assert "def is_palindrome" in kw["input"], "script goes in over stdin"


def test_docker_timeout_kills_container():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 1)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with _docker_mode(), mock.patch.object(checks.subprocess, "run", fake_run):
        ok, out = checks.run_tests("while True: pass")
    assert not ok and "timeout" in out, out
    run_name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "kill", run_name], calls


def test_missing_docker_fails_closed():
    def fake_run(cmd, **kw):
        raise FileNotFoundError("docker")

    with _docker_mode(), mock.patch.object(checks.subprocess, "run", fake_run):
        ok, out = checks.run_tests(checks.extract_code(GOOD))
    assert not ok and "install Docker" in out, out


def test_real_docker_sandbox_blocks_network_and_files():
    """Real Docker: no network, no writes. Fails when Docker is down unless AGENT_LOOP_SKIP_DOCKER=1."""
    if not _require_docker():
        return
    probe = (
        "import socket, os\n"
        "def net():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
        "def write():\n"
        "    try:\n"
        "        open('/etc/pwned', 'w').write('x')\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
    )
    with _docker_mode():
        ok, out = checks.run_tests(probe, ["assert net() is False", "assert write() is False"])
    assert ok, out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"{fn.__name__}: ok")
    print(f"{len(tests)} offline checks passed")
