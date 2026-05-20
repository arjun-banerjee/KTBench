"""
Subprocess isolation for eval runs.

Candidate code runs in a child process so it cannot:
  - access the grader's Python namespace or call stack
  - kill the parent evaluator (SIGKILL is caught and scored as failure)
  - snoop oracle tensors or test_suite.toml from the filesystem
    (the caller must NOT pass oracle tensor paths to the child)

The child process receives its task as JSON on stdin and returns
its result as a single JSON line on stdout. Stderr is captured for logging.

For official leaderboard runs the container layer (Docker/nspawn) provides
stronger isolation — this module handles the Python-level boundary.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass


@dataclass
class IsolatedResult:
    success: bool
    payload: dict
    stderr: str
    timed_out: bool = False
    killed: bool = False


_WORKER_TEMPLATE = textwrap.dedent("""\
import sys, json, os, traceback

# Remove current dir from path so child cannot import from the grader's workspace
if '' in sys.path: sys.path.remove('')
if '.' in sys.path: sys.path.remove('.')

def main():
    try:
        task = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({{"error": f"bad task json: {{e}}"}}, flush=True))
        return

    fn_src  = task["fn_src"]
    fn_name = task["fn_name"]
    args    = task.get("args", {{}})

    ns = {{}}
    try:
        exec(compile(fn_src, "<worker_fn>", "exec"), ns)
    except Exception as e:
        print(json.dumps({{"error": f"worker fn compile error: {{e}}"}}))
        return

    fn = ns.get(fn_name)
    if fn is None:
        print(json.dumps({{"error": f"{{fn_name}} not defined in worker fn"}}))
        return

    try:
        result = fn(**args)
        print(json.dumps({{"ok": True, "result": result}}), flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({{"error": str(e), "traceback": tb}}), flush=True)

main()
""")


def run_isolated(
    fn_src: str,
    fn_name: str,
    args: dict,
    timeout: int = 300,
    extra_env: dict[str, str] | None = None,
) -> IsolatedResult:
    """
    Run fn_name(**args) defined in fn_src inside a child Python process.

    fn_src must be a complete Python source string that defines fn_name at
    the module level. args must be JSON-serialisable.

    The child receives the task via stdin and returns a JSON result on stdout.
    If it times out or is killed, the result is marked accordingly.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)   # don't leak parent's import paths
    if extra_env:
        env.update(extra_env)

    task = json.dumps({"fn_src": fn_src, "fn_name": fn_name, "args": args})

    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER_TEMPLATE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        # New process group so we can kill the whole tree on timeout
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(
            input=task.encode(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # Kill the entire process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return IsolatedResult(
            success=False,
            payload={"error": f"eval timed out after {timeout}s"},
            stderr="",
            timed_out=True,
        )

    stderr_text = stderr.decode(errors="replace").strip()

    # Parse the last JSON line from stdout (worker may print debug before it)
    stdout_text = stdout.decode(errors="replace").strip()
    json_line = ""
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            json_line = line
            break

    if not json_line:
        return IsolatedResult(
            success=False,
            payload={"error": "no JSON output from worker", "stdout": stdout_text[:500]},
            stderr=stderr_text,
        )

    try:
        payload = json.loads(json_line)
    except json.JSONDecodeError as e:
        return IsolatedResult(
            success=False,
            payload={"error": f"JSON decode failed: {e}", "raw": json_line[:200]},
            stderr=stderr_text,
        )

    success = payload.get("ok", False) and "error" not in payload
    return IsolatedResult(success=success, payload=payload, stderr=stderr_text)
