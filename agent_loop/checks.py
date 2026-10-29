import re
import subprocess
import sys
import uuid

from .config import CONFIG

TEST_TIMEOUT = CONFIG["test_timeout"]
SANDBOX = CONFIG["sandbox"]  # "docker" (default) or "local" (unisolated; needs AGENT_LOOP_ALLOW_LOCAL=1)

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


def lockdown_flags(*, memory: str, cpus, pids: int, tmp_mb: int) -> list[str]:
    """Flags every sandbox container gets (assert runs and project test runs alike).

    No network or IPC, read-only root, a small no-exec tmpfs, capped memory/CPU/processes/
    files, no capabilities, no privilege escalation, a non-root user, and never an image
    pull at run time (the image is built or pulled on purpose, not by a test run)."""
    return [
        "--network", "none", "--ipc", "none",
        "--read-only", "--tmpfs", f"/tmp:size={tmp_mb}m,noexec,nosuid,nodev",
        "--memory", memory, "--memory-swap", memory,
        "--cpus", str(cpus), "--pids-limit", str(pids),
        "--ulimit", "nofile=1024:1024", "--ulimit", "core=0",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--pull", "never", "--label", "agent-loop=sandbox",
    ]


def _docker_cmd(name: str) -> list[str]:
    return [
        "docker", "run", "--rm", "-i", "--name", name,
        *lockdown_flags(memory=CONFIG["sandbox_memory"], cpus=CONFIG["sandbox_cpus"], pids=64, tmp_mb=16),
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


# Runs INSIDE the container. It checks what the container really is, not what we asked for.
PROBE = '''\
import os, socket
problems = []
if os.getuid() == 0:
    problems.append("runs as root")
status = dict(line.split(":", 1) for line in open("/proc/self/status").read().splitlines() if ":" in line)
if int(status.get("CapEff", "1").strip(), 16) != 0:
    problems.append("capabilities not dropped")
if status.get("NoNewPrivs", "0").strip() != "1":
    problems.append("no-new-privileges not set")
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2).close()
    problems.append("network reachable")
except OSError:
    pass
for path in ("/sandbox-probe", "/etc/sandbox-probe", "/usr/sandbox-probe"):
    try:
        open(path, "w").close()
        problems.append("root filesystem writable")
        break
    except OSError:
        pass
for path in ("/var/run/docker.sock", "/run/docker.sock", "/mnt/c", "/host"):
    if os.path.exists(path):
        problems.append("host path visible: " + path)
for limit in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/pids.max"):
    try:
        if open(limit).read().strip() == "max":
            problems.append("no limit in " + limit)
    except OSError:
        pass
print("SANDBOX OK" if not problems else "SANDBOX BROKEN: " + "; ".join(problems))
raise SystemExit(1 if problems else 0)
'''

_VERIFIED = False  # set once per process after the probe passes


def verify_sandbox() -> tuple[bool, str]:
    """Prove the container is locked down before any model code runs in it.

    A sandbox that silently lost a protection (a Docker change, a bad flag, a config edit)
    must stop the run, not just weaken it. Checked once per process."""
    global _VERIFIED
    if SANDBOX != "docker" or _VERIFIED:
        return True, "ok"
    try:
        proc = _execute(PROBE)
    except subprocess.TimeoutExpired:
        return False, "sandbox self-check timed out; model code was NOT run"
    except OSError as exc:
        return False, f"could not start sandbox: {exc} (install Docker and start it)"
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0 and out.endswith("SANDBOX OK"):
        _VERIFIED = True
        return True, "ok"
    return False, "sandbox self-check failed; model code was NOT run: " + out[-600:]


def run_tests(code: str, tests: list[str] | None = None) -> tuple[bool, str]:
    """Run `code` plus the asserts in the sandbox; never raises."""
    if not code.strip():
        return False, "no Python code found in the worker output"
    safe, why = verify_sandbox()
    if not safe:
        return False, why
    body = "\n".join(tests if tests is not None else TESTS)
    script = f"{code}\n\n\n{body}\nprint('ALL TESTS PASSED')\n"
    try:
        proc = _execute(script)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TEST_TIMEOUT}s"
    except OSError as exc:  # interpreter or docker missing / not executable
        hint = " (install Docker and start it)" if SANDBOX == "docker" else ""
        return False, f"could not start sandbox: {exc}{hint}"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, (output[-1500:] or "(no output)")
