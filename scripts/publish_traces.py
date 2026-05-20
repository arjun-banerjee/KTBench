"""Publish KTBench traces and the leaderboard to gh-pages.

Three jobs per invocation. First, walk ``traces/`` for every ``.jsonl``
trace (single-run files at the top level plus per-cell traces under
``traces/<sweep>/<cell_slug>/``) and materialise a per-run viewer at
``<slug>/viewer.html`` on the gh-pages branch, copying the ensemble
static viewer next to each trace. Second, parse every published trace
into a summary record (timestamp, scenario, model, persona, src_dsl,
tgt_dsl, problem_id, outcome, final_score, sol_score, correctness_rate,
stress_pass_rate, cost_gpu_seconds) and write ``runs.json`` at the
gh-pages root, which is what the leaderboard and the run index page
fetch on load. Third, copy the top-level site assets (``style.css``,
``app.js``, ``index.html``, ``runs.html``) from ``site/`` to the
gh-pages root.

The wipe-before-publish step keeps gh-pages reflecting exactly what
the current ``traces/`` and ``site/`` directories say. Old content
stays in git history on the gh-pages branch.

Usage
-----
    uv run python scripts/publish_traces.py --ensemble-root ~/Documents/ensemble

    uv run python scripts/publish_traces.py \\
        --ensemble-root ~/Documents/ensemble --watch 300

    # dry run: build the worktree, skip the commit and push
    uv run python scripts/publish_traces.py --dry-run

Summary-record extraction
-------------------------
For each trace the script extracts:

- ``timestamp``: first event's ``ts_ms`` converted to UTC ISO 8601, or
  file mtime as fallback.
- ``scenario``: parsed from a ``grader:`` system event when present,
  otherwise the trace file stem.
- ``model`` and ``persona``: parsed from the ``agent_spawn:`` system
  note the scenarios emit at startup (first spawn wins, which is the
  author for judge_translate).
- ``problem_id``, ``src_dsl``, ``tgt_dsl``: parsed from the
  ``problem_prompt`` note when emitted, plus from sibling
  ``runs.jsonl`` entries when the trace was driven by ``run_sweep``.
- ``final_score``, ``sol_score``, ``correctness_rate``,
  ``stress_pass_rate``: parsed from the most recent ``state_diff``
  whose ``field == "ktbench_submissions"`` (the wrapped
  ``submit_kernel`` populates this).
- ``outcome``: ``passed`` when ``final_score > 0``, ``failed`` when a
  submission was recorded with ``final_score == 0``, ``incomplete``
  otherwise.
- ``cost_gpu_seconds``: sum of every event's ``costs.gpu_seconds``.
- ``viewer_path``: ``<run_slug>/viewer.html`` relative to gh-pages
  root.

GitHub Pages side (one-time)
----------------------------
    Settings -> Pages -> Source: Deploy from a branch
                         Branch: gh-pages, Folder: / (root)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_TOP = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_TOP / "traces"
SITE_DIR = REPO_TOP / "site"


def _run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def _git(args: List[str], cwd: Path = REPO_TOP, check: bool = True) -> str:
    proc = _run(["git", *args], cwd=cwd, check=check)
    return proc.stdout.strip()


def _ensure_gh_pages_branch_exists() -> None:
    branches = _git(["branch", "-a"]).splitlines()
    flat = [b.strip().lstrip("* ").replace("remotes/origin/", "") for b in branches]
    if "gh-pages" in flat:
        return
    print("creating gh-pages orphan branch", file=sys.stderr)
    _git(["checkout", "--orphan", "gh-pages"])
    _git(["rm", "-rf", "--quiet", "."], check=False)
    seed = REPO_TOP / "README.md"
    seed.write_text("# KTBench leaderboard\n\nPublished by scripts/publish_traces.py.\n")
    _git(["add", "README.md"])
    _git(["commit", "-m", "seed gh-pages branch"])
    _git(["checkout", "-"])


def _worktree(scratch: Path, *, fetch_remote: bool = True, remote: str = "origin") -> Path:
    worktree = scratch / "gh-pages-worktree"
    if worktree.exists():
        _git(["worktree", "remove", "--force", str(worktree)], check=False)
        shutil.rmtree(worktree, ignore_errors=True)
    if fetch_remote:
        _git(["fetch", remote, "gh-pages"], check=False)
        _git(["worktree", "add", "-B", "gh-pages", str(worktree), f"{remote}/gh-pages"], check=False)
    if not worktree.exists():
        _git(["worktree", "add", "-b", "gh-pages-local", str(worktree)], check=False)
        if not worktree.exists():
            _git(["worktree", "add", str(worktree)], check=False)
    return worktree


def _wipe_worktree(worktree: Path) -> None:
    for entry in worktree.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink()


def _copy_ensemble_viewer(ensemble_root: Path, dest: Path) -> None:
    src = ensemble_root / "site"
    if not src.exists():
        raise FileNotFoundError(
            f"could not find ensemble site at {src}; pass --ensemble-root"
        )
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {"viewer.html", "viewer.js", "style.css"}
    for entry in src.iterdir():
        if entry.name in wanted:
            shutil.copy2(entry, dest / entry.name)


def _copy_local_site(dest: Path) -> None:
    if not SITE_DIR.exists():
        raise FileNotFoundError(f"local site dir not found at {SITE_DIR}")
    for entry in SITE_DIR.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, dest / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest / entry.name)


def _all_traces() -> List[Path]:
    if not TRACES_DIR.exists():
        return []
    return sorted(TRACES_DIR.rglob("*.jsonl"))


def _slug_for(trace: Path) -> str:
    rel = trace.relative_to(TRACES_DIR)
    parts = list(rel.with_suffix("").parts)
    # Sweep cells: <sweep>/<cell_slug>/<scenario>.jsonl. Drop the
    # scenario component (redundant with the cell slug) and join
    # with __ so the slug is filename-safe.
    if len(parts) >= 3 and parts[-1] in {"ktbench.translate_problem", "ktbench.judge_translate"}:
        parts = parts[:-1]
    return "__".join(parts)


@dataclass
class Summary:
    slug: str
    trace_relpath: str
    timestamp: Optional[str] = None
    scenario: Optional[str] = None
    model: Optional[str] = None
    persona: Optional[str] = None
    problem_id: Optional[str] = None
    src_dsl: Optional[str] = None
    tgt_dsl: Optional[str] = None
    src_hw: Optional[str] = None
    tgt_hw: Optional[str] = None
    outcome: str = "incomplete"
    final_score: Optional[float] = None
    sol_score: Optional[float] = None
    correctness_rate: Optional[float] = None
    stress_pass_rate: Optional[float] = None
    speedup_vs_ref: Optional[float] = None
    cost_gpu_seconds: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    viewer_path: str = ""


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _load_runs_jsonl_index(trace: Path) -> Optional[Dict[str, Any]]:
    sweep_dir = trace.parent.parent
    runs_jsonl = sweep_dir / "runs.jsonl"
    if not runs_jsonl.exists():
        return None
    trace_str = str(trace)
    last = None
    for rec in _read_jsonl(runs_jsonl):
        if rec.get("trace_path") == trace_str or rec.get("trace_path", "").endswith(str(trace.relative_to(REPO_TOP))):
            last = rec
    return last


_GRADER_NOTE = re.compile(r"^grader:\s*(\{.*\})$")
_SPAWN_NOTE = re.compile(
    r"^agent_spawn:\s+id=(?P<id>\S+)\s+persona=(?P<persona>\S+)\s+model=(?P<model>\S+)"
)


def _parse_trace(trace: Path) -> Summary:
    summary = Summary(
        slug=_slug_for(trace),
        trace_relpath=str(trace.relative_to(REPO_TOP)),
    )
    last_submission: Optional[Dict[str, Any]] = None
    grader_payload: Optional[Dict[str, Any]] = None
    first_ts_ms: Optional[int] = None

    for ev in _read_jsonl(trace):
        ts_ms = ev.get("ts_ms")
        if first_ts_ms is None and isinstance(ts_ms, int):
            first_ts_ms = ts_ms
        payload = ev.get("payload") or {}
        kind = payload.get("kind")

        if kind == "state_diff":
            diffs = payload.get("diff") or []
            items = diffs if isinstance(diffs, list) else [diffs]
            for item in items:
                if isinstance(item, dict) and item.get("field") == "ktbench_submissions":
                    new = item.get("new")
                    if isinstance(new, dict):
                        last_submission = new
        elif kind == "system":
            note = payload.get("note") or ""
            m = _GRADER_NOTE.match(note)
            if m:
                try:
                    grader_payload = json.loads(m.group(1))
                except json.JSONDecodeError:
                    grader_payload = None
            first_line = note.splitlines()[0] if note else ""
            sm = _SPAWN_NOTE.match(first_line)
            if sm:
                if summary.persona is None:
                    summary.persona = sm.group("persona")
                if summary.model is None:
                    summary.model = sm.group("model")

        costs = ev.get("costs") or payload.get("costs") or {}
        if isinstance(costs, dict):
            gpu = costs.get("gpu_seconds")
            if isinstance(gpu, (int, float)):
                summary.cost_gpu_seconds += float(gpu)

    if first_ts_ms is not None:
        summary.timestamp = _dt.datetime.utcfromtimestamp(first_ts_ms / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        mtime = trace.stat().st_mtime
        summary.timestamp = _dt.datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%SZ")

    if last_submission:
        for key in (
            "final_score",
            "sol_score",
            "correctness_rate",
            "stress_pass_rate",
            "speedup_vs_ref",
        ):
            val = last_submission.get(key)
            if isinstance(val, (int, float)):
                setattr(summary, key, float(val))

    if grader_payload:
        if isinstance(grader_payload.get("scenario"), str):
            summary.scenario = grader_payload["scenario"]
        scores = grader_payload.get("scores") or {}
        if isinstance(scores, dict):
            summary.scores = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}

    if summary.scenario is None:
        summary.scenario = trace.stem

    # Outcome from final_score: > 0 passed, == 0 (with a submission)
    # failed, no submission at all incomplete. The 0.5-cap fallback in
    # ktbench/score.py for SOL-not-measured is still > 0 so it counts
    # as a pass for ranking purposes; the leaderboard's mean_final
    # column shows the actual value.
    if summary.final_score is not None and summary.final_score > 0:
        summary.outcome = "passed"
    elif summary.final_score is not None:
        summary.outcome = "failed"
    elif last_submission:
        summary.outcome = "failed"
    else:
        summary.outcome = "incomplete"

    # Sibling runs.jsonl carries the problem metadata for sweep cells.
    # Prefer it over best-effort trace parsing because the runner knows
    # exactly what cell this trace belongs to.
    runs_rec = _load_runs_jsonl_index(trace)
    if runs_rec:
        for key in ("model", "persona", "scenario"):
            val = runs_rec.get(key)
            if isinstance(val, str):
                setattr(summary, key, val)
        for key in ("problem_id", "src_dsl", "tgt_dsl", "src_hw", "tgt_hw"):
            val = runs_rec.get(key)
            if isinstance(val, str):
                setattr(summary, key, val)
        if isinstance(runs_rec.get("scores"), dict):
            for k, v in runs_rec["scores"].items():
                if isinstance(v, (int, float)):
                    summary.scores.setdefault(k, float(v))

    summary.viewer_path = f"{summary.slug}/viewer.html"
    return summary


def _summarize_all() -> List[Summary]:
    summaries: List[Summary] = []
    for trace in _all_traces():
        try:
            summaries.append(_parse_trace(trace))
        except Exception as e:
            print(f"parse failed for {trace}: {e}", file=sys.stderr)
    summaries.sort(key=lambda s: s.timestamp or "", reverse=True)
    return summaries


def _write_runs_json(worktree: Path, summaries: List[Summary]) -> None:
    payload = {
        "generated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": [
            {
                "slug": s.slug,
                "timestamp": s.timestamp,
                "scenario": s.scenario,
                "model": s.model,
                "persona": s.persona,
                "problem_id": s.problem_id,
                "src_dsl": s.src_dsl,
                "tgt_dsl": s.tgt_dsl,
                "src_hw": s.src_hw,
                "tgt_hw": s.tgt_hw,
                "outcome": s.outcome,
                "final_score": s.final_score,
                "sol_score": s.sol_score,
                "correctness_rate": s.correctness_rate,
                "stress_pass_rate": s.stress_pass_rate,
                "speedup_vs_ref": s.speedup_vs_ref,
                "cost_gpu_seconds": round(s.cost_gpu_seconds, 3) if s.cost_gpu_seconds else 0.0,
                "scores": s.scores,
                "viewer_path": s.viewer_path,
            }
            for s in summaries
        ],
    }
    (worktree / "runs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def publish(ensemble_root: Path, scratch: Path, dry_run: bool = False, remote: str = "origin") -> None:
    _ensure_gh_pages_branch_exists()
    worktree = _worktree(scratch, fetch_remote=not dry_run, remote=remote)
    _wipe_worktree(worktree)

    _copy_local_site(worktree)

    summaries = _summarize_all()
    for trace in _all_traces():
        slug = _slug_for(trace)
        target = worktree / slug
        target.mkdir(parents=True, exist_ok=True)
        _copy_ensemble_viewer(ensemble_root, target)
        shutil.copy2(trace, target / "trace.jsonl")

    _write_runs_json(worktree, summaries)

    if dry_run:
        print(f"dry-run: built worktree at {worktree} with {len(summaries)} runs", file=sys.stderr)
        return

    _git(["add", "."], cwd=worktree)
    diff = _git(["status", "--porcelain"], cwd=worktree)
    if not diff:
        print("nothing to publish", file=sys.stderr)
        return
    stamp = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    _git(["commit", "-m", f"publish {len(summaries)} runs {stamp}"], cwd=worktree)
    push = _run(["git", "push", remote, "gh-pages"], cwd=worktree, check=False)
    if push.returncode != 0:
        print(push.stderr, file=sys.stderr)
        print("publish: push failed (see stderr above)", file=sys.stderr)
        return
    print(f"published {len(summaries)} runs to {remote}/gh-pages", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="publish_traces")
    parser.add_argument(
        "--ensemble-root",
        type=Path,
        default=os.environ.get("ENSEMBLE_ROOT", str(Path.home() / "Documents" / "ensemble")),
        help="Path to the ensemble checkout that holds site/.",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=REPO_TOP.parent / ".ktbench-publish",
        help="Scratch directory for the gh-pages worktree.",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="If > 0, repeat the publish every N seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the worktree and runs.json but skip the commit and push.",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote to fetch and push the gh-pages branch from/to (default: origin).",
    )
    args = parser.parse_args(argv)

    ensemble_root = Path(args.ensemble_root).expanduser().resolve()
    scratch = Path(args.scratch).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            publish(ensemble_root, scratch, dry_run=args.dry_run, remote=args.remote)
        except Exception as e:
            print(f"publish failed: {e}", file=sys.stderr)
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
