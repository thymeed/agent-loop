"""Fingerprint of the eval fixtures (everything under evals/).

    python -m agent_loop.fixtures             # check the fixtures against evals/FIXTURES.sha256
    python -m agent_loop.fixtures --update    # rewrite the pin after an intentional change

Eval reports are only comparable while the tasks and projects behind them are unchanged, so an
edit to a fixture has to be a deliberate act: the test suite fails until the pin is updated.
"""
import hashlib
import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent.parent / "evals"
PIN = EVALS / "FIXTURES.sha256"
_SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def digest(root: Path = EVALS) -> str:
    """One hash over every fixture file's path and content.

    Line endings are normalised so a Windows checkout and a Linux one agree, and caches,
    bytecode and the pin file itself are ignored."""
    files = sorted(p for p in Path(root).rglob("*")
                   if p.is_file() and p.name != PIN.name and p.suffix != ".pyc"
                   and not (set(p.relative_to(root).parts) & _SKIP_DIRS))
    total = hashlib.sha256()
    for path in files:
        content = path.read_bytes().replace(b"\r\n", b"\n")
        total.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + hashlib.sha256(content).digest())
    return total.hexdigest()


def pinned() -> str | None:
    return PIN.read_text(encoding="utf-8").strip() if PIN.is_file() else None


def main() -> int:
    current = digest()
    if "--update" in sys.argv[1:]:
        PIN.write_text(current + "\n", encoding="utf-8", newline="\n")
        print(f"pinned {current[:16]}... in {PIN.name}")
        return 0
    if pinned() == current:
        print("eval fixtures match their pin")
        return 0
    print("eval fixtures differ from evals/FIXTURES.sha256: reports made before this change are no "
          "longer comparable. If the change is intended, run: python -m agent_loop.fixtures --update")
    return 1


if __name__ == "__main__":
    sys.exit(main())
