"""Eval harness: fixed tasks, several attempts each, a pass-rate table you can compare.

    python -m agent_loop.evals evals --attempts 3 --label baseline
    python -m agent_loop.evals --compare runs/evals/baseline-*.json runs/evals/idea-*.json

One lone run says little (the same task passed and failed on consecutive live runs), so
change the models or the prompts only when this table says it helps.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

from . import runs
from .cli import expand_task_files, run
from .config import CONFIG
from .tasks import load_task

EVALS_DIR = runs.RUNS_DIR / "evals"


def _scrub(text: str) -> str:
    """Keep local paths out of the saved report."""
    for path, shown in ((str(runs.RUNS_DIR.parent), "<repo>"), (str(Path.home()), "~")):
        text = text.replace(path, shown).replace(path.replace("\\", "/"), shown)
    return text


def run_attempt(name: str, spec, attempt: int, stamp: str, out_dir: Path,
                max_tokens: int | None) -> dict:
    """One attempt at one task, non-interactive (the owner call at the cap is 'stop')."""
    timings: dict[str, float] = {}
    started = time.monotonic()
    error = None
    values: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            values = run(
                spec.task, tests=spec.tests, kind=spec.kind, query=spec.query,
                max_papers=spec.max_papers, project=spec.project,
                test_command=spec.test_command, requirements=spec.requirements,
                protected=spec.protected, resume_with="stop",
                thread=f"{stamp}-{name}-{attempt}", runs_dir=out_dir,
                timings=timings, max_tokens=max_tokens)
    except Exception as exc:  # one broken attempt (Ollama hiccup, Docker down) must not lose the rest
        error = _scrub(f"{type(exc).__name__}: {exc}")[:300]
    tokens = {n: c.get("total", 0) for n, c in values.get("tokens", {}).items()}
    calls = {n: c.get("calls", 0) for n, c in values.get("tokens", {}).items()}
    passed = bool(values.get("approved") and values.get("tests_passed") and not values.get("outcome"))
    return {
        "task": name,
        "attempt": attempt,
        "passed": passed,
        "result": "error" if error else runs.result_label(values),
        "laps": calls.get("worker", 0) + calls.get("escalate", 0),
        "escalated": calls.get("escalate", 0) > 0,
        "tokens": tokens,
        "total_tokens": sum(tokens.values()),
        "seconds": round(time.monotonic() - started, 1),
        "node_seconds": {n: round(s, 1) for n, s in timings.items()},
        **({"error": error} if error else {}),
    }


def new_report(label: str, attempts: int, budget: int | None) -> dict:
    return {
        "label": label,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"models": dict(CONFIG["models"]), "model_options": dict(CONFIG["model_options"]),
                   "override": CONFIG["override"], "num_ctx": CONFIG["num_ctx"],
                   "max_rejections": CONFIG["max_rejections"], "sandbox": CONFIG["sandbox"],
                   "prompt_variants": list(CONFIG["prompt_variants"])},
        "attempts": attempts,
        "budget": budget,
        "finished": False,
        "results": [],
    }


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # a crash mid-write never leaves half a report


def run_eval(task_files: list[str], attempts: int, label: str, *, out_root: Path = EVALS_DIR,
             budget: int | None = None, progress=print, only: set[str] | None = None) -> tuple[dict, Path]:
    """Run every task `attempts` times. The report is saved after each attempt."""
    # The suffix keeps two evals started in the same second from sharing a thread (and its state).
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label)
    report_path = out_root / f"{slug}-{stamp}.json"
    out_dir = out_root / f"{slug}-{stamp}"
    jobs = [(Path(f).stem, load_task(f)) for f in task_files]
    if only:
        unknown = only - {name for name, _ in jobs}
        if unknown:
            raise SystemExit(f"no such eval task(s): {', '.join(sorted(unknown))}")
        jobs = [(n, s) for n, s in jobs if n in only]
    report = new_report(label, attempts, budget)
    total = len(jobs) * attempts
    try:
        for name, spec in jobs:
            for attempt in range(1, attempts + 1):
                result = run_attempt(name, spec, attempt, stamp, out_dir, budget)
                report["results"].append(result)
                save_report(report, report_path)
                progress(f"[{len(report['results'])}/{total}] {name} #{attempt}: "
                         f"{'PASS' if result['passed'] else 'FAIL'} ({result['result']}) "
                         f"laps={result['laps']} tokens={result['total_tokens']} {result['seconds']}s")
        report["finished"] = True
    except KeyboardInterrupt:
        progress("interrupted: the report so far is saved")
    save_report(report, report_path)
    return report, report_path


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: list[dict]) -> list[dict]:
    """One row per task (in first-seen order) plus an ALL row."""
    order: list[str] = []
    for r in results:
        if r["task"] not in order:
            order.append(r["task"])
    rows = []
    for name, group in [*((n, [r for r in results if r["task"] == n]) for n in order), ("ALL", results)]:
        passes = [r for r in group if r["passed"]]
        rows.append({
            "task": name,
            "n": len(group),
            "passes": len(passes),
            "rate": len(passes) / len(group) if group else 0.0,
            "mean_laps_to_pass": _mean([r["laps"] for r in passes]),
            "escalated": sum(1 for r in group if r["escalated"]),
            "errors": sum(1 for r in group if r["result"] == "error"),
            "mean_tokens": _mean([r["total_tokens"] for r in group]),
            "mean_seconds": _mean([r["seconds"] for r in group]),
        })
    return rows


def _fmt(value, spec="{:.1f}") -> str:
    return "-" if value is None else spec.format(value)


def format_table(rows: list[dict]) -> str:
    head = f"{'task':16s} {'pass':>7s} {'rate':>6s} {'laps*':>6s} {'esc':>4s} {'err':>4s} {'tokens':>8s} {'secs':>7s}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['task']:16s} {str(r['passes']) + '/' + str(r['n']):>7s} {r['rate'] * 100:>5.0f}% "
            f"{_fmt(r['mean_laps_to_pass']):>6s} {r['escalated']:>4d} {r['errors']:>4d} "
            f"{_fmt(r['mean_tokens'], '{:.0f}'):>8s} {_fmt(r['mean_seconds']):>7s}")
    lines.append("* mean worker/escalation laps among the attempts that passed")
    return "\n".join(lines)


def compare(a: dict, b: dict) -> str:
    """Per-task change from report `a` to report `b`."""
    rows_a = {r["task"]: r for r in summarize(a["results"])}
    rows_b = {r["task"]: r for r in summarize(b["results"])}
    head = f"{'task':16s} {'pass rate':>17s} {'change':>8s} {'tokens':>14s} {'secs':>12s}"
    lines = [f"A = {a['label']}   B = {b['label']}", head, "-" * len(head)]
    for name in [*rows_a, *(n for n in rows_b if n not in rows_a)]:
        ra, rb = rows_a.get(name), rows_b.get(name)
        if ra is None or rb is None:
            lines.append(f"{name:16s} only in {'A' if rb is None else 'B'}")
            continue
        delta = (rb["rate"] - ra["rate"]) * 100
        lines.append(
            f"{name:16s} {f'{ra['passes']}/{ra['n']} -> {rb['passes']}/{rb['n']}':>17s} {delta:>+7.0f}% "
            f"{_fmt(ra['mean_tokens'], '{:.0f}'):>6s}->{_fmt(rb['mean_tokens'], '{:.0f}'):>6s} "
            f"{_fmt(ra['mean_seconds']):>5s}->{_fmt(rb['mean_seconds']):>5s}")
    n = min(a.get("attempts") or 1, b.get("attempts") or 1)
    lines.append(f"With {n} attempt(s) per task one flipped attempt moves a task by {100 // n}%; "
                 "treat differences of that size as noise and repeat before trusting them.")
    return "\n".join(lines)


def missing_models(config_models: dict[str, str], tags: dict) -> list[str]:
    """Ollama models named in the config that are not pulled."""
    have = {m["name"] for m in tags.get("models", [])}
    have |= {n.removesuffix(":latest") for n in have}
    wanted = {spec.partition(":")[2] for spec in config_models.values()
              if spec.partition(":")[0] == "ollama"}
    return sorted(m for m in wanted if m not in have)


def preflight() -> None:
    """Fail fast, before a long run, if Ollama is down or a configured model is missing."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if "://" not in host:
        host = "http://" + host
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=5).json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(f"Ollama is not reachable at {host} ({type(exc).__name__}); start it first")
    missing = missing_models(CONFIG["models"], tags)
    if missing:
        raise SystemExit("models not pulled: " + ", ".join(missing)
                         + "  (ollama pull <name>, or edit the config)")


def config_line(cfg: dict) -> str:
    """The pre-run line naming the options, patch and prompt variants in force; "" when none is set."""
    if not (cfg["model_options"] or cfg["override"] or cfg["prompt_variants"]):
        return ""
    variants = f"   prompt variants: {cfg['prompt_variants']}" if cfg["prompt_variants"] else ""
    return f"options: {cfg['model_options']}   patch: {cfg['override']}{variants}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tasks", nargs="*", help="eval task files or directories (default: evals)")
    ap.add_argument("--attempts", type=int, default=3, help="attempts per task (default 3)")
    ap.add_argument("--label", default="eval", help="name for this configuration")
    ap.add_argument("--budget", type=int, default=None, metavar="TOKENS",
                    help="stop an attempt once it has used this many tokens")
    ap.add_argument("--only", default=None, help="comma-separated task names to run")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="print the change from report A to report B and exit")
    args = ap.parse_args()

    if args.compare:
        a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)
        print(compare(a, b))
        return
    if args.attempts < 1:
        ap.error("--attempts must be at least 1")

    files = expand_task_files(args.tasks or [str(Path(__file__).resolve().parent.parent / "evals")])
    preflight()
    only = {n.strip() for n in args.only.split(",") if n.strip()} if args.only else None
    print(f"models: {CONFIG['models']}")
    line = config_line(CONFIG)
    if line:
        print(line)
    report, path = run_eval(files, args.attempts, args.label, budget=args.budget, only=only)
    print("\n" + format_table(summarize(report["results"])))
    print(f"\nreport: {path.relative_to(runs.RUNS_DIR.parent)}")


if __name__ == "__main__":
    main()
