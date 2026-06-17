"""Shared Claude (`claude -p`) backend for the alignment experiments.

The 7 experiments were originally GPT/Gemini only (Claude was deny-listed under
the sponsorship). This adds the Claude models as a SEPARATE, caveated probe,
elicited through the Claude Code agent exactly like the 140-dilemma probe
(`gen_responses_claude.py`): even with the system prompt replaced, dynamic
context excluded, and tools disabled, the harness injects agentic context and
temperature/thinking are not controllable. So every Claude row is tagged
`elicitation="claude_code_cli"` and is kept OUT of the 11-model headline numbers
— it surfaces only as a flagged side-series on the experiments page.

Each `claude_exp*.py` module reproduces its runner's EXACT prompts and appends
Claude rows to that experiment's responses.jsonl (resume-safe). Multi-turn
protocols are packed into a single prompt as a labeled transcript, since
`claude -p` has no conversation-resume.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The 3 Claude models that represent "Claude" on the site (Fable 5 is currently
# unavailable via the CLI). Matches CLAUDE_LOGICAL_MODELS minus fable.
CLAUDE_EXP_MODELS = ["claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"]

CALL_TIMEOUT_S = 240
MAX_WORKERS = 4   # gentle on the subscription rate-limit window
_DISALLOWED_TOOLS = [
    "Bash", "Read", "Edit", "Write", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "TodoWrite", "NotebookEdit",
]


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def pack(messages: list[dict]) -> tuple[str | None, str]:
    """Split an OpenAI-style messages list into (system, single_prompt).

    Single user turn -> the prompt is that content verbatim. Multi-turn (an
    assistant turn is present) -> pack the conversation as a labeled transcript
    so the model continues from it (the agent-path caveat already covers the
    fidelity gap vs a true multi-turn API call)."""
    system = None
    convo = []
    for m in messages:
        if m.get("role") == "system":
            system = m["content"]
        else:
            convo.append(m)
    if len(convo) == 1 and convo[0]["role"] == "user":
        return system, convo[0]["content"]
    parts = []
    for m in convo:
        who = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{who}: {m['content']}")
    return system, "\n\n".join(parts)


def claude_chat(model: str, messages: list[dict], timeout: int = CALL_TIMEOUT_S) -> dict:
    """One `claude -p` call from an OpenAI-style messages list. Returns a dict
    shaped like the runners' call_one() output (response/finish_reason/tokens),
    plus cost_usd/duration_ms, or {"error": ...}."""
    system, prompt = pack(messages)
    cmd = ["claude", "-p", prompt, "--model", model,
           "--max-turns", "1", "--output-format", "json"]
    if system:
        cmd += ["--system-prompt", system, "--exclude-dynamic-system-prompt-sections"]
    cmd += ["--disallowedTools", *_DISALLOWED_TOOLS]
    # Run in its own process group so a wedged call (claude spawns children that
    # inherit the stdout pipe — subprocess.run's timeout then hangs on the pipe)
    # can be killed group-wide. This is the fix for the 30-min zombie calls.
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
    except Exception as e:
        return {"error": f"spawn: {e!r}"}
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        return {"error": f"timeout>{timeout}s"}
    # claude -p sometimes exits non-zero even when it produced a valid success
    # result (observed: rc=1 with a complete {"subtype":"success",...} payload).
    # Trust the JSON first; only fall back to an rc error if it won't parse.
    try:
        d = json.loads(out)
    except Exception:
        return {"error": f"rc={proc.returncode}: {(err or out)[:300]}"}
    if d.get("is_error") or d.get("subtype") != "success":
        return {"error": f"cli-error: rc={proc.returncode} subtype={d.get('subtype')} api={d.get('api_error_status')}"}
    text = (d.get("result") or "").strip()
    if not text:
        return {"error": "empty_result"}
    u = d.get("usage", {}) or {}
    in_tok = (u.get("input_tokens", 0) or 0) + (u.get("cache_creation_input_tokens", 0) or 0) \
        + (u.get("cache_read_input_tokens", 0) or 0)
    return {
        "response": text,
        "finish_reason": (d.get("stop_reason") or "stop"),
        "prompt_tokens": in_tok,
        "completion_tokens": u.get("output_tokens", 0) or 0,
        "cost_usd": d.get("total_cost_usd"),
        "duration_ms": d.get("duration_ms"),
    }


def run_batch(work: list, build_row, out_path: Path, label: str,
              max_workers: int = MAX_WORKERS) -> dict:
    """Run build_row(item)->row over `work` in a thread pool, appending each row
    to out_path as it lands (so a mid-run rate-limit pause loses nothing).
    build_row must call claude_chat and return a fully-formed dict row."""
    n_ok = n_err = 0
    cost = 0.0
    out_f = open(out_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(build_row, it): it for it in work}
            for fut in as_completed(futs):
                rec = fut.result()
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                if rec.get("error"):
                    n_err += 1
                    print(f"  [{label}] ERR {rec.get('model')} {rec.get('error','')[:90]}", flush=True)
                else:
                    n_ok += 1
                    cost += rec.get("cost_usd") or 0.0
                    if (n_ok + n_err) % 10 == 0:
                        print(f"  [{label}] {n_ok} ok, {n_err} err, ~${cost:.2f}", flush=True)
    finally:
        out_f.close()
    print(f"[{label}] DONE: {n_ok} ok, {n_err} err, ~${cost:.2f} (notional)", flush=True)
    return {"ok": n_ok, "err": n_err, "cost": cost}
