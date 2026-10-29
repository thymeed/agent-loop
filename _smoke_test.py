"""Offline checks: owner stop/accept, guidance retries, lap counting, tokens.

Run with pytest (`python -m pytest _smoke_test.py -q`) or directly
(`python _smoke_test.py`).
"""
import subprocess
from types import SimpleNamespace
from unittest import mock

from langgraph.types import Command

from agent_loop import checks, cli, graph as graph_mod, nodes as agent_nodes, research
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


def install(*, code=GOOD, review="APPROVED: fine"):
    """Point the graph at fake models; returns the (worker, reviewer) fakes."""
    agent_nodes.MODELS["lead"] = FakeLLM("1. Do it")
    agent_nodes.MODELS["worker"] = FakeLLM(code)
    agent_nodes.MODELS["reviewer"] = FakeLLM(review)
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


def _docker_mode():
    return mock.patch.object(checks, "SANDBOX", "docker")


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
