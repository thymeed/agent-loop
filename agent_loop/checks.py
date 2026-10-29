import re
import subprocess
import sys
import uuid

from .config import CONFIG

TEST_TIMEOUT = CONFIG["test_timeout"]
SANDBOX = CONFIG["sandbox"]  # "docker" (default) or "local" (unisolated subprocess)

# Fixed asserts the worker's code has to satisfy. `--cap` swaps these out for an
# impossible one so the 3-rejection path can be exercised on purpose.
TESTS = [
    'assert is_palindrome("A man a plan a canal Panama") is True',
    'assert is_palindrome("") is True',
    'assert is_palindrome("ab") is False',
]

IMPOSSIBLE_TESTS = [
    'assert is_palindrome("ab") is True  # impossible: "ab" is not a palindrome',
]

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """First ```python fenced block, else everything from the first `def `."""
    match = _FENCE.search(text or "")
    if match:
        return match.group(1).strip()
    idx = (text or "").find("def ")
    return (text[idx:] if idx >= 0 else (text or "")).strip()


def _docker_cmd(name: str) -> list[str]:
    """Locked-down container: no network, read-only fs, capped resources, non-root."""
    return [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:size=16m",
        "--memory", CONFIG["sandbox_memory"], "--cpus", str(CONFIG["sandbox_cpus"]),
        "--pids-limit", "64",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        CONFIG["sandbox_image"], "python", "-I", "-",
    ]


def _execute(script: str) -> subprocess.CompletedProcess:
    """Run the script in the configured sandbox. May raise TimeoutExpired/OSError."""
    if SANDBOX == "local":
        return subprocess.run(
            [sys.executable, "-X", "utf8", "-I", "-c", script],
            capture_output=True, encoding="utf-8", errors="replace", timeout=TEST_TIMEOUT, check=False,
        )
    name = f"agent-loop-{uuid.uuid4().hex[:12]}"
    try:
        return subprocess.run(
            _docker_cmd(name), input=script,
            capture_output=True, encoding="utf-8", errors="replace", timeout=TEST_TIMEOUT, check=False,
        )
    except subprocess.TimeoutExpired:
        # Killing the docker client leaves the container running; stop it by name.
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)
        raise


def run_tests(code: str, tests: list[str] | None = None) -> tuple[bool, str]:
    """Run `code` plus the asserts in the sandbox; never raises."""
    if not code.strip():
        return False, "no Python code found in the worker output"
    body = "\n".join(tests if tests is not None else TESTS)
    script = f"{code}\n\n\n{body}\nprint('ALL TESTS PASSED')\n"
    try:
        proc = _execute(script)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TEST_TIMEOUT}s"
    except OSError as exc:  # interpreter or docker missing / not executable
        hint = " (install Docker or set sandbox: local in config.yaml)" if SANDBOX == "docker" else ""
        return False, f"could not start sandbox: {exc}{hint}"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, (output[-1500:] or "(no output)")
