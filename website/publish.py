#!/usr/bin/env python3
"""
website/publish.py — publish KTBench trace files to gh-pages.

One script to run.  Reads trace JSONL files written by ensemble, builds
runs.json, then commits + pushes to the gh-pages branch via git worktree.
The viewer HTML / CSS / JS that already lives on gh-pages is left untouched.

USAGE
    python website/publish.py               # publish once from traces/
    python website/publish.py --watch 60    # re-publish every 60 s
    python website/publish.py --traces DIR  # use a different traces dir
    python website/publish.py --dry-run     # parse + preview, no git push
    python website/publish.py --teardown    # remove the worktree and exit

HOW TRACES ARE GENERATED (install ensemble first)
    pip install ensemble-ai

    Then define a world in integrations/ensemble/ (see world.toml for
    scenario config) and run:

        ensemble run integrations/ensemble/world.toml \\
            --traces traces/

    Each completed run writes one *.jsonl file to traces/.  Running
    this publish script (or --watch) picks them up automatically.

WHAT GETS PUSHED TO GH-PAGES
    runs.json                    — rebuilt on every publish
    {slug}/trace.jsonl           — copied from traces/ (idempotent)

    The viewer (viewer.html?trace={slug}) and all static assets are
    already on gh-pages and are not modified by this script.

FIRST-TIME SETUP
    The script creates a git worktree at .ghpages-worktree/ on first
    run.  Make sure gh-pages branch exists:

        git checkout --orphan gh-pages
        git reset --hard
        git commit --allow-empty -m "init gh-pages"
        git push -u origin gh-pages
        git checkout -          # switch back to your branch

    Then run this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent.parent
TRACES_DIR = REPO / "traces"
WORKTREE_PATH = Path(
    os.environ.get("GHPAGES_WORKTREE",
                   str(REPO / ".ghpages-worktree"))
)


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

def _all_strings(obj) -> list[str]:
    """Recursively collect every string value in a parsed JSON object."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_all_strings(v))
        return out
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_all_strings(item))
        return out
    return []


def _extract_meta_block(text: str) -> dict | None:
    """Extract the [meta] JSON block that submit_kernel appends to its output."""
    idx = text.find("[meta]")
    if idx == -1:
        return None
    rest = text[idx + 6:].lstrip()
    if not rest.startswith("{"):
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(rest)
        if "final_score" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _parse_trace(path: Path) -> dict | None:
    """Return a runs.json entry dict parsed from a trace JSONL file."""
    slug = path.stem

    meta: dict = {
        "slug": slug,
        "timestamp": None,
        "scenario": None,
        "model": None,
        "persona": "normal_translation",
        "problem_id": None,
        "src_dsl": "cuda",
        "tgt_dsl": "cuda",
        "src_hw": "nvidia_a100_sxm",
        "tgt_hw": "nvidia_h100_sxm",
        "outcome": "incomplete",
        "final_score": 0.0,
        "sol_score": 0.0,
        "correctness_rate": 0.0,
        "stress_pass_rate": 0.0,
        "speedup_vs_ref": None,
        "cost_gpu_seconds": 0.0,
        "scores": {},
    }

    last_score_meta: dict | None = None

    try:
        with path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                ts_ms = ev.get("ts_ms")
                payload = ev.get("payload") or {}
                kind = payload.get("kind", "")

                # First timestamp → run timestamp
                if meta["timestamp"] is None and ts_ms:
                    meta["timestamp"] = datetime.fromtimestamp(
                        ts_ms / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")

                # System events carry stringified JSON in payload.note
                if kind == "system":
                    note = payload.get("note", "")
                    try:
                        note_obj = json.loads(note)
                    except (json.JSONDecodeError, TypeError):
                        note_obj = {}

                    note_kind = note_obj.get("kind", "")

                    if note_kind == "agent_spawned" and not meta["model"]:
                        meta["model"] = note_obj.get("model")
                        meta["persona"] = note_obj.get("persona") or meta["persona"]

                    elif note_kind == "scenario":
                        meta["scenario"] = note_obj.get("id") or note_obj.get("name")

                    elif note_kind == "problem_prompt":
                        text = note_obj.get("text", "")
                        # **Source DSL:** `cuda` on `nvidia_a100_sxm`
                        m = re.search(r"Source DSL[^`]*`([^`]+)`[^`]*`([^`]+)`", text)
                        if m:
                            meta["src_dsl"], meta["src_hw"] = m.group(1), m.group(2)
                        m = re.search(r"Target DSL[^`]*`([^`]+)`[^`]*`([^`]+)`", text)
                        if m:
                            meta["tgt_dsl"], meta["tgt_hw"] = m.group(1), m.group(2)
                        # problem_id line (if the prompt embeds it)
                        m = re.search(r"problem_id[`:\s]+([a-z0-9_]+)", text, re.I)
                        if m:
                            meta["problem_id"] = m.group(1)

                # Scan every string value in the event for a [meta] score block
                for s in _all_strings(payload):
                    block = _extract_meta_block(s)
                    if block:
                        last_score_meta = block

    except OSError as exc:
        print(f"[publish] warning: could not read {path.name}: {exc}")
        return None

    # Apply score metadata from the last submit_kernel call
    if last_score_meta:
        for key in ("final_score", "sol_score", "correctness_rate",
                    "stress_pass_rate", "speedup_vs_ref", "cost_gpu_seconds"):
            if key in last_score_meta:
                meta[key] = last_score_meta[key]
        if "scores" in last_score_meta:
            meta["scores"] = last_score_meta["scores"]
        meta["outcome"] = (
            "passed" if meta["correctness_rate"] >= 1.0 else "failed"
        )

    # Fall back: derive problem_id + scenario from the slug
    # Convention: {model}__{scenario}  e.g. gpt4o__ktbench_softmax_a100_to_h100
    if not meta["problem_id"] or not meta["scenario"]:
        parts = slug.split("__", 1)
        scenario_part = parts[1] if len(parts) == 2 else slug
        if not meta["scenario"]:
            meta["scenario"] = scenario_part
        if not meta["problem_id"]:
            meta["problem_id"] = re.sub(r"^ktbench_", "", scenario_part)

    meta["viewer_path"] = f"viewer.html?trace={slug}"
    return meta


# ---------------------------------------------------------------------------
# gh-pages worktree management
# ---------------------------------------------------------------------------

def _ensure_worktree() -> Path:
    wt = WORKTREE_PATH
    if not wt.exists():
        print(f"[publish] creating gh-pages worktree at {wt} …")
        subprocess.run(
            ["git", "worktree", "add", str(wt), "gh-pages"],
            cwd=REPO, check=True,
        )
    return wt


def _teardown_worktree() -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(WORKTREE_PATH)],
        cwd=REPO, check=False,
    )
    print(f"[publish] removed worktree {WORKTREE_PATH}")


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

def publish(traces_dir: Path, dry_run: bool = False) -> None:
    trace_files = sorted(traces_dir.glob("*.jsonl"))
    if not trace_files:
        print(f"[publish] no *.jsonl files in {traces_dir}")
        return

    wt = _ensure_worktree()

    runs: list[dict] = []
    for tf in trace_files:
        rec = _parse_trace(tf)
        if rec is None:
            continue
        runs.append(rec)

        # Copy trace file into its slug dir on gh-pages
        slug_dir = wt / rec["slug"]
        slug_dir.mkdir(exist_ok=True)
        dest = slug_dir / "trace.jsonl"
        if not dest.exists() or dest.stat().st_mtime < tf.stat().st_mtime:
            shutil.copy2(tf, dest)

    # Sort by timestamp descending
    runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)

    runs_json = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": runs,
    }
    (wt / "runs.json").write_text(json.dumps(runs_json, indent=2))
    print(f"[publish] runs.json: {len(runs)} runs")

    if dry_run:
        print("[publish] dry-run — skipping git commit/push")
        return

    subprocess.run(["git", "add", "-A"], cwd=wt, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=wt).returncode == 0:
        print("[publish] gh-pages is up to date")
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subprocess.run(
        ["git", "commit", "-m", f"publish traces {ts}"],
        cwd=wt, check=True,
    )
    subprocess.run(["git", "push", "origin", "gh-pages"], cwd=wt, check=True)
    print("[publish] pushed to gh-pages")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish KTBench ensemble traces to gh-pages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--traces", default=str(TRACES_DIR), metavar="DIR",
        help=f"Traces directory (default: {TRACES_DIR})",
    )
    parser.add_argument(
        "--watch", type=int, metavar="SECONDS",
        help="Re-publish every N seconds (Ctrl-C to stop)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse traces and preview runs.json without pushing",
    )
    parser.add_argument(
        "--teardown", action="store_true",
        help="Remove the gh-pages worktree and exit",
    )
    args = parser.parse_args()

    if args.teardown:
        _teardown_worktree()
        return

    traces_dir = Path(args.traces)
    if not traces_dir.exists():
        sys.exit(f"[publish] traces directory not found: {traces_dir}")

    publish(traces_dir, dry_run=args.dry_run)

    if args.watch:
        print(f"[publish] watching every {args.watch}s — Ctrl-C to stop")
        try:
            while True:
                time.sleep(args.watch)
                publish(traces_dir, dry_run=args.dry_run)
        except KeyboardInterrupt:
            print("\n[publish] stopped")


if __name__ == "__main__":
    main()
