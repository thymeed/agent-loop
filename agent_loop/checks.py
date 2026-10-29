import re
import subprocess
import sys

from .config import CONFIG

TEST_TIMEOUT = CONFIG["test_timeout"]

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


def run_tests(code: str, tests: list[str] | None = None) -> tuple[bool, str]:
    """Run `code` plus the asserts in a throwaway subprocess; never raises."""
    if not code.strip():
        return False, "no Python code found in the worker output"
    body = "\n".join(tests if tests is not None else TESTS)
    script = f"{code}\n\n\n{body}\nprint('ALL TESTS PASSED')\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
            check=False,  # a red test suite is a result, not an exception
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TEST_TIMEOUT}s"
    except OSError as exc:  # interpreter missing / not executable
        return False, f"could not start subprocess: {exc}"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, (output[-1500:] or "(no output)")
