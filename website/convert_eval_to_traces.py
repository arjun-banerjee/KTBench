#!/usr/bin/env python3
"""
website/convert_eval_to_traces.py — convert inspect_ai .eval files to
ensemble-format .jsonl traces consumable by website/publish.py.

USAGE
    python website/convert_eval_to_traces.py                        # all evals → traces/
    python website/convert_eval_to_traces.py --log-dir results/inspect_logs
    python website/convert_eval_to_traces.py --out-dir my_traces/
    python website/convert_eval_to_traces.py --dry-run              # preview, no files written
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import timezone
from pathlib import Path

REPO = Path(__file__).parent.parent


def _slug(model: str, problem_id: str, created: str) -> str:
    """Build a filesystem-safe slug: {model}__{problem_id}__{created}."""
    def clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    # created e.g. "2026-05-25T06:50:30+00:00" → "20260525t065030"
    ts = re.sub(r"[^0-9t]", "", created.lower().replace(":", "").split("+")[0])
    return f"{clean(model)}__{clean(problem_id)}__{ts}"


def _resolve(value: object, attachments: dict[str, bytes]) -> str:
    """Resolve an attachment:// reference to its text content."""
    if not isinstance(value, str):
        return str(value)
    if value.startswith("attachment://"):
        key = value[len("attachment://"):]
        raw = attachments.get(key, b"")
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    return value


def convert_sample(sample, model: str, attachments: dict[str, bytes]) -> list[dict]:
    """Convert one EvalSample to a list of ensemble trace event dicts."""
    events: list[dict] = []
    tick = 0

    def ts(dt) -> int:
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)

    def emit(dt, actor, payload):
        events.append({
            "tick": tick,
            "ts_ms": ts(dt),
            "actor": actor,
            "message_id": None,
            "payload": payload,
        })

    first_event = sample.events[0] if sample.events else None
    base_ts = first_event.timestamp if first_event else None

    # Opening system event
    if base_ts:
        emit(base_ts, None, {
            "kind": "system",
            "note": json.dumps({
                "kind": "scenario",
                "id": sample.id,
                "model": model,
            }),
        })

    for ev in sample.events:
        etype = type(ev).__name__

        if etype == "SampleInitEvent":
            prompt_text = _resolve(
                ev.sample.input if isinstance(ev.sample.input, str)
                else (ev.sample.input[0].content if ev.sample.input else ""),
                attachments,
            )
            emit(ev.timestamp, None, {
                "kind": "system",
                "note": json.dumps({"kind": "problem_prompt", "text": prompt_text}),
            })

        elif etype == "ModelEvent":
            out_msg = ev.output.choices[0].message
            tool_calls = out_msg.tool_calls or []

            # Resolve assistant text content (may be an attachment reference)
            raw_content = out_msg.content
            if isinstance(raw_content, list):
                content_str = " ".join(
                    _resolve(getattr(c, "text", str(c)), attachments)
                    for c in raw_content
                )
            else:
                content_str = _resolve(raw_content or "", attachments)

            if content_str.strip():
                emit(ev.timestamp, model, {
                    "kind": "message",
                    "role": "assistant",
                    "content": content_str,
                })

            for tc in tool_calls:
                tick += 1
                emit(ev.timestamp, model, {
                    "kind": "tool_call",
                    "id": tc.id,
                    "name": tc.function,
                    "args": tc.arguments or {},
                })

        elif etype == "ToolEvent":
            result_text = _resolve(ev.result, attachments) if ev.result is not None else ""
            emit(ev.timestamp, model, {
                "kind": "tool_result",
                "id": ev.id,
                "name": ev.function,
                "result": result_text,
                "is_error": ev.error is not None,
            })
            tick += 1

        elif etype == "ScoreEvent":
            emit(ev.timestamp, None, {
                "kind": "system",
                "note": json.dumps({
                    "kind": "score",
                    "value": ev.score.value,
                    "explanation": ev.score.explanation or "",
                }),
            })

    return events


def convert_eval(path: Path, out_dir: Path, dry_run: bool = False) -> list[Path]:
    """Convert one .eval file; returns list of written .jsonl paths."""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        sys.exit("inspect-ai is not installed. Run: pip install inspect-ai")

    log = read_eval_log(str(path))
    model = log.eval.model
    written: list[Path] = []

    for sample in log.samples or []:
        attachments = sample.attachments or {}
        events = convert_sample(sample, model, attachments)

        # Skip aborted/empty runs (only system events, no tool interactions)
        real_events = [e for e in events if e["payload"]["kind"] not in ("system",)]
        if not real_events:
            print(f"  skip {path.name} / {sample.id!r} (no tool events)")
            continue

        slug = _slug(model, sample.id, log.eval.created)
        out_path = out_dir / f"{slug}.jsonl"

        if dry_run:
            print(f"  [dry-run] would write {out_path} ({len(events)} events)")
            written.append(out_path)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        print(f"  wrote {out_path} ({len(events)} events)")
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert inspect_ai .eval files to ensemble .jsonl traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir", default=str(REPO / "results" / "inspect_logs"),
        metavar="DIR", help="Directory (searched recursively) for *.eval files",
    )
    parser.add_argument(
        "--out-dir", default=str(REPO / "traces"),
        metavar="DIR", help="Output directory for *.jsonl traces (default: traces/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written without creating files",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    eval_files = sorted(log_dir.rglob("*.eval"))
    if not eval_files:
        sys.exit(f"No *.eval files found under {log_dir}")

    print(f"Found {len(eval_files)} .eval file(s) under {log_dir}")
    total = 0
    for ef in eval_files:
        written = convert_eval(ef, out_dir, dry_run=args.dry_run)
        total += len(written)

    print(f"\n{'Would write' if args.dry_run else 'Wrote'} {total} trace(s) to {out_dir}")
    if not args.dry_run and total:
        print(f"\nNext: python website/publish.py --traces {out_dir}")


if __name__ == "__main__":
    main()
