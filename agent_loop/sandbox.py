"""Docker sandbox for `type: project` tasks: pytest on a read-only workspace.

Same lockdown as the assert sandbox in checks.py, plus the workspace mounted read-only.
Third-party packages are installed only while building the image, never at test time.
"""
import hashlib
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from . import checks
from .config import CONFIG

DOCKERFILE = Path(__file__).resolve().parent.parent / "sandbox" / "Dockerfile"
BUILD_TIMEOUT = 600
OUTPUT_TAIL = 3000

# One requirement: name, optional [extras], optional version clauses. No URLs, markers,
# flags (-r, -e, --index-url) or anything a shell or pip option parser would act on.
_VERSION = r"(?:==|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!_-]+"
_REQUIREMENT = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9,._-]+\])?(?:\s*{_VERSION}(?:\s*,\s*{_VERSION})*)?$")


class SandboxError(RuntimeError):
    """The sandbox image could not be prepared."""


def validate_requirements(requirements: list[str]) -> list[str]:
    bad = [r for r in requirements if not isinstance(r, str) or not _REQUIREMENT.match(r.strip())]
    if bad:
        raise ValueError(f"not a plain requirement specifier (name plus optional versions): {bad}")
    return [r.strip() for r in requirements]


def image_tag(requirements: list[str]) -> str:
    key = "\n".join([CONFIG["sandbox_image"], *sorted(requirements)])
    return "agent-loop-proj-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def ensure_image(requirements: list[str]) -> str:
    """Tag of the test image for these requirements, building it once if it is missing."""
    requirements = validate_requirements(requirements)
    tag = image_tag(requirements)
    try:
        if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
            return tag
        with tempfile.TemporaryDirectory() as ctx:
            shutil.copy(DOCKERFILE, Path(ctx) / "Dockerfile")
            (Path(ctx) / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
            print(f"building sandbox image {tag} (requirements: {requirements or 'none'}) ...")
            build = subprocess.run(
                ["docker", "build", "--build-arg", f"BASE={CONFIG['sandbox_image']}", "-t", tag, ctx],
                capture_output=True, encoding="utf-8", errors="replace", timeout=BUILD_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxError(f"could not build the sandbox image: {exc} "
                           "(install Docker and start it)") from exc
    if build.returncode != 0:
        raise SandboxError("sandbox image build failed:\n" + (build.stdout + build.stderr)[-1500:])
    return tag


def _test_argv(command: str) -> list[str]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("test_command is empty")
    if argv[0] == "pytest":
        argv[1:1] = ["-p", "no:cacheprovider"]  # /work is read-only; no .pytest_cache
    return argv


def project_docker_cmd(name: str, workspace: Path, image: str, command: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:size=64m",
        "--memory", CONFIG["project_memory"], "--cpus", str(CONFIG["sandbox_cpus"]),
        "--pids-limit", "256",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--mount", f"type=bind,source={Path(workspace).resolve()},target=/work,readonly",
        "-w", "/work",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPATH=/work:/work/src", "-e", "HOME=/tmp",
        image, *_test_argv(command),
    ]


def run_project_tests(workspace: Path, command: str, requirements: list[str]) -> tuple[bool, str]:
    """Run the project's tests in the sandbox; never raises."""
    if checks.SANDBOX != "docker":
        return False, "project tasks need `sandbox: docker` in config.yaml (no local mode for real projects)"
    timeout = CONFIG["project_test_timeout"]
    try:
        image = ensure_image(requirements)
        name = f"agent-loop-{uuid.uuid4().hex[:12]}"
        try:
            proc = subprocess.run(project_docker_cmd(name, workspace, image, command),
                                  capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            return False, f"timeout after {timeout}s"
    except (SandboxError, ValueError) as exc:
        return False, str(exc)
    except OSError as exc:
        return False, f"could not start sandbox: {exc} (install Docker and start it)"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, (output[-OUTPUT_TAIL:] or "(no output)")
