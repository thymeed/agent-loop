"""Offline checks: owner stop/accept, guidance retries, lap counting, tokens.

Run with pytest (`python -m pytest _smoke_test.py -q`) or directly
(`python _smoke_test.py`).
"""
import contextlib
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

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


def test_project_docker_command_is_locked_down_with_a_readonly_mount():
    import pathlib, tempfile
    ws = pathlib.Path(tempfile.mkdtemp())
    cmd = sandbox.project_docker_cmd("n1", ws, "img", "pytest -q tests")
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges",
                 "--pids-limit", "--memory", "--user 65534", "target=/work,readonly", "-w /work"):
        assert flag in joined, (flag, joined)
    assert cmd[-6:] == ["pytest", "-p", "no:cacheprovider", "-q", "tests"][-6:] or \
        cmd[cmd.index("img") + 1:] == ["pytest", "-p", "no:cacheprovider", "-q", "tests"], cmd
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
    if not _docker_up():
        print("  (docker not available - skipped)")
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
    assert config.config_path({}) == config.DEFAULT_CONFIG_PATH
    alt = pathlib.Path(tempfile.mkdtemp()) / "alt.yaml"
    alt.write_text("models:" + NL + "  worker: ollama:other:1b" + NL, encoding="utf-8")
    assert config.config_path({"AGENT_LOOP_CONFIG": str(alt)}) == alt.resolve()
    try:
        config.config_path({"AGENT_LOOP_CONFIG": str(alt.parent / "typo.yaml")})
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
    if not _docker_up():
        print("  (docker not available - skipped)")
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


def test_real_docker_project_sandbox():
    """Fixture project: fails, passes after a fix, cannot write to /work, needs no network."""
    import pathlib, tempfile
    if not _docker_up():
        print("  (docker not available - skipped)")
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
    """Runs only when Docker is up; otherwise a no-op so the suite stays offline."""
    import shutil
    if not shutil.which("docker") or subprocess.run(
            ["docker", "info"], capture_output=True).returncode != 0:
        print("  (docker not available - skipped)")
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
