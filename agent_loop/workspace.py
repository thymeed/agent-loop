"""Project workspaces: copy a project, show it to the worker, apply its file blocks safely.

Everything here is pure file handling with no model calls. `apply_blocks` is the
security boundary for paths written by a model.
"""
import difflib
import fnmatch
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".tox", "runs"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".rst"}
# The tests are the grade, so the worker may not edit them or the files that steer
# test collection. fnmatch's `*` crosses `/`, so `tests/*` covers nested paths.
DEFAULT_PROTECTED = ("test_*.py", "*_test.py", "conftest.py", "tests/*", "*/tests/*",
                     "test/*", "*/test/*", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
                     # names that would replace pytest or run at interpreter start-up if they got
                     # onto the path (the sandbox also keeps the project off the front of it)
                     "pytest.py", "_pytest/*", "sitecustomize.py", "usercustomize.py", "*.pth")
MAX_BLOCKS = 20             # files the worker may write per lap
MAX_FILE_BYTES = 100_000    # one written file

# ```path/to/file.py  (fence info string is the path)  ...content...  ```
_BLOCK = re.compile(r"^```[ \t]*(?P<path>[^\s`]+)[ \t]*\r?\n(?P<body>.*?)^```[ \t]*$",
                    re.DOTALL | re.MULTILINE)
# Info strings that are a language, not a path: they carry no file name.
_LANGUAGES = {"python", "py", "text", "txt", "bash", "sh", "shell", "diff", "yaml", "json",
              "toml", "markdown", "md", "console", "output"}


class WorkspaceError(ValueError):
    """The project cannot be used as a workspace."""


# Files that look like credentials are never copied into a workspace: the workspace is
# mounted into a container that runs model-written code.
SECRET_NAMES = (".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.kdbx", "id_rsa*", "id_dsa*",
                "id_ecdsa*", "id_ed25519*", ".npmrc", ".pypirc", ".netrc", ".git-credentials",
                "credentials", "credentials.*", "secrets.*", "*.token")
# Key-shaped strings inside ordinary files. A hit refuses the project instead of copying it.
SECRET_CONTENT = re.compile(r"sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
                            r"|-----BEGIN [A-Z ]*PRIVATE KEY-----")
MAX_SCAN_BYTES = 1_000_000


def copy_project(src: Path, dest: Path, *, max_bytes: int) -> list[str]:
    """Copy `src` to a fresh `dest`, skipping junk dirs; refuse symlinks and huge projects.

    Credential-looking files are skipped (their names are returned); a file that contains
    a key-shaped string makes the whole copy fail, naming the file but never the secret."""
    src = Path(src)
    if not src.is_dir():
        raise WorkspaceError(f"project directory not found: {src}")
    if dest.exists():
        raise WorkspaceError(f"workspace already exists: {dest}")
    total = 0
    files: list[Path] = []
    skipped: list[str] = []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            raise WorkspaceError(f"symlinks are not allowed in a project: {rel}")
        if path.is_file():
            if any(fnmatch.fnmatchcase(path.name.lower(), pat) for pat in SECRET_NAMES):
                skipped.append(rel.as_posix())
                continue
            size = path.stat().st_size
            total += size
            if total > max_bytes:
                raise WorkspaceError(f"project is over the {max_bytes}-byte limit")
            if path.suffix in TEXT_SUFFIXES and size <= MAX_SCAN_BYTES:
                if SECRET_CONTENT.search(path.read_text(encoding="utf-8", errors="ignore")):
                    raise WorkspaceError(
                        f"{rel.as_posix()} contains what looks like a secret (API key or private key); "
                        "remove it from the project before using it with this tool")
            files.append(rel)
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
    dest.mkdir(parents=True, exist_ok=True)  # an empty project still gets a workspace
    return sorted(skipped)


def project_files(root: Path) -> list[str]:
    """Text files of the workspace as posix paths, sorted."""
    return sorted(
        p.relative_to(root).as_posix() for p in Path(root).rglob("*")
        if p.is_file() and not p.is_symlink() and p.suffix in TEXT_SUFFIXES
        and not any(part in SKIP_DIRS for part in p.relative_to(root).parts)
    )


def build_context(root: Path, *, budget: int, first: tuple[str, ...] = ()) -> str:
    """File tree plus file contents that fit `budget` characters; tests first, then `first`."""
    names = project_files(root)
    names.sort(key=lambda n: (not Path(n).name.startswith("test_") and not n.startswith("tests/"),
                              n not in first, n))
    parts, used, omitted = [], 0, []
    for name in names:
        text = (Path(root) / name).read_text(encoding="utf-8", errors="replace")
        block = f"### {name}\n```\n{text}\n```\n"
        if used + len(block) > budget:
            omitted.append(name)
            continue
        parts.append(block)
        used += len(block)
    out = "Project files:\n" + "\n".join(f"- {n}" for n in names) + "\n\n" + "\n".join(parts)
    if omitted:
        out += "\n(not shown, over the context budget: " + ", ".join(omitted) + ")\n"
    return out


@dataclass
class Applied:
    written: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)   # fatal: the lap is not tested
    skipped: list[str] = field(default_factory=list)  # edits to protected files, ignored


def parse_blocks(reply: str) -> list[tuple[str, str]]:
    """(path, content) for each fenced block whose info string is a file path."""
    blocks = []
    for m in _BLOCK.finditer(reply or ""):
        path = m["path"].strip()
        if path.lower() in _LANGUAGES:
            continue
        blocks.append((path, m["body"]))
    return blocks


def _check_path(root: Path, raw: str) -> tuple[Path | None, str | None]:
    """The resolved target inside `root`, or an error string."""
    if "\\" in raw or ":" in raw:
        return None, f"{raw}: use plain relative posix paths (no backslashes or drive letters)"
    posix = PurePosixPath(raw)
    if posix.is_absolute() or ".." in posix.parts:
        return None, f"{raw}: path must be relative and stay inside the project"
    if any(part in SKIP_DIRS for part in posix.parts):
        return None, f"{raw}: writing into {sorted(set(posix.parts) & SKIP_DIRS)[0]} is not allowed"
    root_real = Path(root).resolve()
    target = (root_real / posix).resolve()
    if root_real != target and root_real not in target.parents:
        return None, f"{raw}: resolves outside the project"
    # A symlink anywhere on the path could point out of the workspace.
    probe = root_real
    for part in posix.parts:
        probe = probe / part
        if probe.is_symlink():
            return None, f"{raw}: path goes through a symlink"
    return target, None


def is_protected(path: str, patterns) -> bool:
    """True if `path` (posix, relative) matches a protected glob; case-insensitive, like Windows."""
    path = PurePosixPath(path).as_posix().lower()
    name = path.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatchcase(path, p.lower()) or fnmatch.fnmatchcase(name, p.lower())
               for p in patterns)


def protected_changes(original: Path, root: Path, patterns) -> list[str]:
    """Protected files that were edited, deleted or added in `root` compared to `original`.

    Independent of apply_blocks, so a path that slipped past it is still caught."""
    if not patterns:
        return []
    names = set(project_files(original)) | set(project_files(root))
    changed = []
    for name in sorted(n for n in names if is_protected(n, patterns)):
        old, new = Path(original) / name, Path(root) / name
        if old.exists() != new.exists() or (old.exists() and old.read_bytes() != new.read_bytes()):
            changed.append(name)
    return changed


def apply_blocks(root: Path, reply: str, protected=DEFAULT_PROTECTED) -> Applied:
    """Write the reply's file blocks into `root`. Bad blocks are reported, never written."""
    result = Applied()
    blocks = parse_blocks(reply)
    if not blocks:
        result.errors.append("no file blocks found; reply with one ```path/to/file.py block "
                             "per changed file, containing the complete new file")
        return result
    if len(blocks) > MAX_BLOCKS:
        result.errors.append(f"{len(blocks)} files in one reply; the limit is {MAX_BLOCKS}")
        return result
    for raw, content in blocks:
        target, error = _check_path(root, raw)
        if error is None and len(content.encode("utf-8")) > MAX_FILE_BYTES:
            error = f"{raw}: file is over {MAX_FILE_BYTES} bytes"
        if error is None and is_protected(raw, protected):
            # Not fatal: the rest of the reply may be a fine fix. The tests stay untouched,
            # and `protected_changes` re-checks that before they run.
            result.skipped.append(f"{raw}: protected test or config file, edit ignored")
            continue
        if error is None and target.is_dir():
            error = f"{raw}: is a directory"
        if error:
            result.errors.append(error)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # bytes, not write_text: Windows would turn every LF into CRLF and pollute the diff
        target.write_bytes((content if content.endswith("\n") else content + "\n").encode("utf-8"))
        result.written.append(PurePosixPath(raw).as_posix())
    return result


def _lines(path: Path) -> list[str]:
    """Lines with their original endings (CRLF files must diff and apply as CRLF)."""
    if not path.exists():
        return []
    return path.read_bytes().decode("utf-8", errors="replace").splitlines(keepends=True)


def make_patch(original: Path, workspace: Path) -> str:
    """Unified diff (git-apply style, a/ and b/ prefixes) from `original` to `workspace`."""
    names = set(project_files(original)) | set(project_files(workspace))
    chunks = []
    for name in sorted(names):
        old, new = Path(original) / name, Path(workspace) / name
        old_lines, new_lines = _lines(old), _lines(new)
        if old_lines == new_lines:
            continue
        header = ["--- /dev/null\n" if not old.exists() else f"--- a/{name}\n",
                  "+++ /dev/null\n" if not new.exists() else f"+++ b/{name}\n"]
        diff = list(difflib.unified_diff(old_lines, new_lines, n=3))[2:]
        if diff and not diff[-1].endswith("\n"):
            diff[-1] += "\n\\ No newline at end of file\n"
        mode = "new file mode 100644\n" if not old.exists() else ""
        chunks.append(f"diff --git a/{name} b/{name}\n{mode}" + "".join(header + diff))
    return "".join(chunks)
