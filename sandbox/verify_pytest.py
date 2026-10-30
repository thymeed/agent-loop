"""Build-time check for sandbox/Dockerfile: pytest and its dependencies must be as pip installed them.

pip overwrites files silently. A requirement whose wheel ships a `_pytest/__init__.py` replaces pytest
without changing any version, and the constraints file cannot see it. This check catches that kind of
accident. It is not a defence against a deliberately hostile package: such a package can run code in
every test process through a .pth file, and can shadow pytest from the interpreter's stdlib directory.

The Dockerfile runs the script twice, both times as `python -I -S`. -S skips site, so neither run
imports a .pth file or a sitecustomize.py from a requirement (-I alone does not stop that):

    python -I -S /tmp/verify_pytest.py snapshot   # right after `pip install pytest==9.2.0` and the freeze
    python -I -S /tmp/verify_pytest.py check      # the last build step, after the requirements install

Options: --constraints (default /tmp/constraints.txt), --site (default: the interpreter's site-packages,
from site.getsitepackages(), because -S leaves site-packages off sys.path), --snapshot (default
/tmp/verify_pytest_snapshot.json).

For every dist named in the constraints file, snapshot records the sha256 and size of each file in its
scope: every file its RECORD lists, and every file under its top-level package directories and its
dist-info directory, __pycache__ included. The shared top-level __pycache__ is not one of those
directories: a .pyc in it is in scope only when RECORD lists it. check compares the disk with that
record, not with RECORD, because a wheel can ship its own RECORD. It fails on a changed or missing
file, and on a file added inside one of those directories. A file added at the top level of
site-packages, or unlisted in the shared top-level __pycache__, is outside the scope.

Nothing imports these packages between the two steps, so a .pyc that changes or appears there was put
there by a requirement. The snapshot is written under /tmp, which a plain wheel does not write to.

Each problem is printed with its name and file; any problem means exit 1.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import site
import sys

CONSTRAINTS = "/tmp/constraints.txt"
SNAPSHOT = "/tmp/verify_pytest_snapshot.json"


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_names(constraints):
    names = []
    for line in pathlib.Path(constraints).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(canonical(re.split(r"[\s=<>!~;@\[]", line, maxsplit=1)[0]))
    return names


def installed(search):
    found = {}
    for dist in importlib.metadata.distributions(path=search):
        found.setdefault(canonical(dist.metadata["Name"] or ""), []).append(dist)
    return found


def digest(path):
    data = path.read_bytes()
    return [hashlib.sha256(data).hexdigest(), len(data)]


def relative(path, base):
    return pathlib.PurePath(os.path.relpath(path, base)).as_posix()


def scope_of(dist):
    """{relative path: [sha256, size]} for the dist's files, the relative names of its directories, and
    problems with its RECORD. Paths are relative to the directory the dist is installed in."""
    base = pathlib.Path(dist.locate_file(""))
    text = dist.read_text("RECORD")
    if text is None:
        return {}, [], ["RECORD is missing"]
    problems, listed, dirs = [], [], set()
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) != 3:
            problems.append(f"malformed RECORD row {row!r}")
            continue
        parts = pathlib.PurePosixPath(row[0]).parts
        # the shared top-level __pycache__ holds every top-level module's .pyc, so it is no dist's own folder
        if len(parts) > 1 and parts[0] not in ("..", "__pycache__") and not parts[0].endswith(".data"):
            dirs.add(parts[0])  # a package directory, or the dist-info directory
        listed.append(pathlib.Path(os.path.normpath(base / row[0])))
    files = {}
    for path in listed:
        if path.is_file():
            files[relative(path, base)] = digest(path)
        elif "__pycache__" not in path.parts:  # a missing .pyc is harmless: it is regenerated on import
            problems.append(f"{relative(path, base)}: listed in RECORD but missing")
    for name in sorted(dirs):
        if (base / name).is_dir():
            for path in (base / name).rglob("*"):
                if path.is_file():
                    files[relative(path, base)] = digest(path)
    return files, sorted(dirs), problems


def changes(record, base):
    """What differs from the snapshot: changed or missing files, and files added in a scope directory."""
    out, files = [], record["files"]
    for key, (sha, size) in files.items():
        path = base / key
        if not path.is_file():
            out.append(f"{key}: missing")
            continue
        now = digest(path)
        if now != [sha, size]:
            out.append(f"{key}: changed (snapshot {sha[:12]}, {size} bytes; now {now[0][:12]}, {now[1]} bytes)")
    for name in record["dirs"]:
        if (base / name).is_dir():
            for path in (base / name).rglob("*"):
                if path.is_file() and relative(path, base) not in files:
                    out.append(f"{relative(path, base)}: added")
    return out


def take_snapshot(names, found):
    out, record = [], {}
    for name in names:
        dists = found.get(name, [])
        if len(dists) != 1:
            out.append(f"{name}: expected exactly one installed copy, found {len(dists)}")
            continue
        files, dirs, problems = scope_of(dists[0])
        out.extend(f"{name}: {p}" for p in problems)
        record[name] = {"files": files, "dirs": dirs}
    return record, out


def check(names, found, record):
    out = []
    for name in names:
        dists = found.get(name, [])
        if len(dists) != 1:
            out.append(f"{name}: expected exactly one installed copy, found {len(dists)}")
        elif name not in record:
            out.append(f"{name}: not in the snapshot")
        else:
            base = pathlib.Path(dists[0].locate_file(""))
            out.extend(f"{name}: {p}" for p in changes(record[name], base))
    return out


def report(out, verdict):
    for line in out:
        print("pytest set problem: " + line, file=sys.stderr)
    print(verdict, file=sys.stderr)
    return 1


def main(argv):
    parser = argparse.ArgumentParser(description="Snapshot or check the pytest set. See the module docstring.")
    parser.add_argument("mode", choices=["snapshot", "check"])
    parser.add_argument("--constraints", default=CONSTRAINTS)
    parser.add_argument("--site", default=None)
    parser.add_argument("--snapshot", default=SNAPSHOT)
    args = parser.parse_args(argv[1:])
    try:
        names = pinned_names(args.constraints)
    except OSError as exc:
        return report([f"cannot read {args.constraints}: {exc}"], "sandbox image build FAILS")
    if "pytest" not in names:
        return report([f"{args.constraints} pins no pytest, so there is nothing to verify"],
                      "sandbox image build FAILS")
    found = installed([args.site] if args.site else site.getsitepackages())
    if args.mode == "snapshot":
        record, out = take_snapshot(names, found)
        if out:
            return report(out, "sandbox image build FAILS: pytest or its dependencies are not as pip installed them")
        pathlib.Path(args.snapshot).write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        print(f"pytest set recorded: {sum(len(r['files']) for r in record.values())} files in {len(record)} dists")
        return 0
    try:
        record = json.loads(pathlib.Path(args.snapshot).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return report([f"no usable snapshot at {args.snapshot}: {exc}"], "sandbox image build FAILS")
    out = check(names, found, record)
    if out:
        return report(out, "sandbox image build FAILS: a requirement altered pytest or its dependencies")
    print("pytest set verified: every file matches the snapshot and nothing was added")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
