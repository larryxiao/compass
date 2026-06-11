"""Shared config: endpoints, env loading, helpers."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the cost guard from wsb_synthdata when present (internal monorepo);
# fall back to no-op stubs in the public export, which ships without it.
ROOT = Path(__file__).resolve().parent.parent.parent  # .../code
sys.path.insert(0, str(ROOT / "wsb_synthdata"))
try:
    from safety import assert_deployment_allowed, CostGuard, SafetyError  # noqa: E402
except ImportError:  # public export: no internal cost-guard module
    class SafetyError(RuntimeError):
        pass

    def assert_deployment_allowed(*_a, **_k):
        return None

    class CostGuard:  # no-op stand-in; SoftCostGuard tolerates its absence
        def __init__(self, *_a, **_k):
            raise SafetyError("internal cost guard unavailable in public export")

DENY_PREFIXES = ["claude-"]
MTD_EMERGENCY_STOP = 11000.00

# GCP project for Vertex AI calls — supply via env (or a repo-root .env.local
# with VERTEX_PROJECT_ID=...). Never hardcoded so the public export stays
# free of account identifiers.
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
if not VERTEX_PROJECT_ID:
    _envf = Path(__file__).resolve().parents[3] / ".env.local"
    if _envf.exists():
        for _ln in _envf.read_text().splitlines():
            _ln = _ln.strip().removeprefix("export ")
            if _ln.startswith("VERTEX_PROJECT_ID="):
                VERTEX_PROJECT_ID = _ln.split("=", 1)[1].strip().strip('"')
                break


# --- Endpoint table -----------------------------------------------------------

@dataclass
class Endpoint:
    name: str               # logical model name surfaced in records ("gpt-5.5", "gemini-3.5-flash", ...)
    base_url: str
    api_version: str
    deployment: str         # the actual Azure deployment / Vertex model id
    api_key_env: str
    tpm_cap: int            # rough cap for self-throttling
    region: str             # for metadata (Azure region or Vertex location)
    provider: str = "azure"  # "azure" (legacy, no new calls) or "vertex"
    tokens_used_window: list = field(default_factory=list)

    def admit(self, tokens_estimated: int) -> float:
        now = time.time()
        self.tokens_used_window = [(t, c) for (t, c) in self.tokens_used_window if now - t < 60]
        used = sum(c for _, c in self.tokens_used_window)
        budget = int(self.tpm_cap * 0.7)
        if used + tokens_estimated <= budget:
            return 0.0
        if self.tokens_used_window:
            oldest_t = self.tokens_used_window[0][0]
            return max(0.1, 60 - (now - oldest_t))
        return 1.0

    def record(self, tokens: int) -> None:
        self.tokens_used_window.append((time.time(), tokens))


# Decision-model endpoints: each (model, region) is a separate Endpoint object.
# Pooling for gpt-5.5: we balance across the 3 regions but label all of them as
# logical model "gpt-5.5".
DECISION_ENDPOINTS: list[Endpoint] = [
    # gpt-5.5: restricted to southcentralus only for the 140-dilemma extension run
    # (avoid triple-counting). The other two regions stayed in-rotation for the
    # original 20-dilemma precompute; the responses.jsonl 'region' field records
    # which region answered each historical row.
    Endpoint("gpt-5.5", "https://southcentralus.api.cognitive.microsoft.com/",
             "2024-10-21", "aoai-southcentralus", "AOAI_KEY_AOAI_2_SOUTHCENTRALUS", 10000, "southcentralus"),
    Endpoint("gpt-5.4", "https://your-aoai-resource-2.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-5.4", "AOAI_KEY_AOAI_RESOURCE_2", 5000, "eastus2"),
    Endpoint("gpt-5.4-nano", "https://your-aoai-resource-2.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-5.4-nano", "AOAI_KEY_AOAI_RESOURCE_2", 75000, "eastus2"),
    Endpoint("gpt-4o", "https://your-aoai-resource-1.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-4o", "AOAI_KEY_AOAI_RESOURCE_1", 450, "eastus"),
    Endpoint("gpt-4o-mini", "https://your-aoai-resource-1.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-4o-mini", "AOAI_KEY_AOAI_RESOURCE_1", 1021, "eastus"),
]

# Judge endpoints — single instance per judge model.
# tpm_cap here is the SELF-throttle.
#
# Post-Azure-sunset (2026-05-20): judges moved from Azure (gpt-4o + gpt-5.4) to
# Gemini (gemini-2.5-pro + gemini-3.5-flash) on Vertex. The original Azure-judged
# rows for the 900 GPT responses are preserved in mapped_options.gpt_judges.jsonl
# as a historical record / for cross-judge agreement analysis. All current
# analysis uses the consistent Gemini judges across the full 11-model matrix.
JUDGE_ENDPOINTS: dict[str, Endpoint] = {
    # Two mid-tier flash judges from different generations — mirrors the original
    # Azure pair (gpt-4o + gpt-5.4, both mid-tier from different generations).
    # 2.5-flash chosen over 2.5-pro because the classification task doesn't need
    # pro-tier reasoning, and 2.5-pro's project quota throttled us to ~2 rows/min
    # in trial — see mapped_options.partial_2_5_pro_judge.jsonl for the partial run.
    "gemini-2.5-flash": Endpoint("gemini-2.5-flash", "", "", "gemini-2.5-flash",
                                  "VERTEX_PROJECT", 100000, "us-central1", provider="vertex"),
    "gemini-3.5-flash": Endpoint("gemini-3.5-flash", "", "", "gemini-3.5-flash",
                                  "VERTEX_PROJECT", 100000, "global", provider="vertex"),
}

# Historical Azure judges (NO new calls; preserved for record/replay tooling).
_AZURE_JUDGE_ENDPOINTS_HISTORICAL: dict[str, Endpoint] = {
    "gpt-4o": Endpoint("gpt-4o", "https://your-aoai-resource-1.cognitiveservices.azure.com/",
                       "2024-10-21", "gpt-4o", "AOAI_KEY_AOAI_RESOURCE_1", 30000, "eastus"),
    "gpt-5.4": Endpoint("gpt-5.4", "https://your-aoai-resource-2.cognitiveservices.azure.com/",
                        "2024-10-21", "gpt-5.4", "AOAI_KEY_AOAI_RESOURCE_2", 30000, "eastus2"),
}

# Gemini lineup — direct tier mapping to the GPT 5-model structure.
# Generation × tier matrix:
#   3.1-pro-preview  ≈ gpt-5.5  (flagship)
#   3.5-flash        ≈ gpt-5.4  (frontier mid)
#   3.1-flash-lite   ≈ gpt-5.4-nano (nano/lite)
#   2.5-pro          ≈ gpt-4o   (legacy flagship)
#   2.5-flash        ≈ gpt-4o-mini (legacy mid)
# Region notes: Gemini 3.x lives on the Vertex "global" location;
# Gemini 2.5 lives on standard regional endpoints (us-central1 is GA).
_ALL_GEMINI_ENDPOINTS: list[Endpoint] = [
    Endpoint("gemini-3.1-pro-preview", "", "", "gemini-3.1-pro-preview",
             "VERTEX_PROJECT", 50000, "global", provider="vertex"),
    Endpoint("gemini-3.5-flash", "", "", "gemini-3.5-flash",
             "VERTEX_PROJECT", 100000, "global", provider="vertex"),
    Endpoint("gemini-3.1-flash-lite", "", "", "gemini-3.1-flash-lite",
             "VERTEX_PROJECT", 200000, "global", provider="vertex"),
    Endpoint("gemini-2.5-pro", "", "", "gemini-2.5-pro",
             "VERTEX_PROJECT", 50000, "us-central1", provider="vertex"),
    Endpoint("gemini-2.5-flash", "", "", "gemini-2.5-flash",
             "VERTEX_PROJECT", 100000, "us-central1", provider="vertex"),
    Endpoint("gemini-2.5-flash-lite", "", "", "gemini-2.5-flash-lite",
             "VERTEX_PROJECT", 200000, "us-central1", provider="vertex"),
]

# GEMINI_SKIP env var (comma-separated logical names) filters the active lineup
# at import time. Lets us run flash-tier-only fast, then backfill the slow
# pro-tier models separately, without editing any runner. Example:
#   GEMINI_SKIP="gemini-2.5-pro,gemini-3.1-pro-preview" python3 run_exp1.py
# Read from the shell env (set BEFORE python starts), not .env.local.
_GEMINI_SKIP = {s.strip() for s in os.environ.get("GEMINI_SKIP", "").split(",") if s.strip()}
GEMINI_ENDPOINTS: list[Endpoint] = [e for e in _ALL_GEMINI_ENDPOINTS if e.name not in _GEMINI_SKIP]

GPT_LOGICAL_MODELS = ["gpt-5.5", "gpt-5.4", "gpt-5.4-nano", "gpt-4o", "gpt-4o-mini"]
GEMINI_LOGICAL_MODELS = [ep.name for ep in GEMINI_ENDPOINTS]
LOGICAL_MODELS = GPT_LOGICAL_MODELS + GEMINI_LOGICAL_MODELS

# Claude family — a SEPARATE, caveated probe. Elicited through the Claude Code
# CLI (`gen_responses_claude.py`), NOT the raw provider API the other 11 models
# used: even with the system prompt replaced and tools disabled, the harness
# injects ~11.5k tokens of agentic context and the model stays tool-aware. So
# Claude is NOT comparable to the API columns and is DELIBERATELY kept out of
# LOGICAL_MODELS — the canonical cross-family analysis (REPORT_140, the 11-model
# B-attractor / family-disagreement numbers) must stay GPT-vs-Gemini only.
# WEB_MODELS is the superset that reaches the website (where Claude is shown as a
# clearly-labelled, statistics-excluded side panel).
CLAUDE_LOGICAL_MODELS = ["claude-fable-5", "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"]
WEB_MODELS = LOGICAL_MODELS + CLAUDE_LOGICAL_MODELS
PERTURBATION_KINDS = ["original", "gender_swap", "reversed_rapport"]


def is_gemini(name: str) -> bool:
    return name.lower().startswith("gemini-")


def is_claude(name: str) -> bool:
    return name.lower().startswith("claude-")


def is_temp_locked(deployment: str, logical_model: str | None = None) -> bool:
    """gpt-5.x family + reasoning models only allow default temperature.

    We test BOTH the deployment string and (if provided) the logical model name,
    because some Azure deployment slugs (e.g. "aoai-eastus2" for gpt-5.5)
    don't carry the model name.
    """
    needles = ("gpt-5", "5.4", "5.5")
    d = deployment.lower()
    if any(n in d for n in needles):
        return True
    if d.startswith("o1") or d.startswith("o3"):
        return True
    if logical_model:
        m = logical_model.lower()
        if any(n in m for n in needles):
            return True
    return False


# --- Vertex helpers (shared by gen_responses + all experiment runners) -------
#
# Provider dispatch is INLINE in each runner: branch on ep.provider, then call
# build_vertex_client / vertex_chat for the Gemini branch. Keep the runner's own
# Azure code path untouched. This avoids a provider registry (per user choice)
# while still avoiding duplicated google-genai boilerplate across 8 files.

def build_vertex_client(ep):
    """Lazy import to keep common.py importable without google-genai installed.

    Sets an HTTP timeout (ms) so a hung request can't block forever — without
    this, a stalled pro-tier call under quota pressure hangs the whole asyncio
    gather indefinitely (observed: a 22h hang on exp3).
    """
    from google import genai  # noqa: PLC0415
    from google.genai import types as gt  # noqa: PLC0415
    return genai.Client(
        vertexai=True, project=VERTEX_PROJECT_ID, location=ep.region,
        http_options=gt.HttpOptions(timeout=180_000),  # 180s in ms
    )


# Map Gemini FinishReason enum names to the OpenAI vocabulary the rest of the
# code assumes. Anything unknown passes through lowercased.
_VERTEX_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "recitation",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "OTHER": "other",
    "MALFORMED_FUNCTION_CALL": "other",
}


async def vertex_chat(ep, client, *,
                      system_prompt: str | None,
                      messages: list[dict],
                      max_output_tokens: int,
                      thinking_budget: int = 4000,
                      temperature: float = 0.7,
                      max_retries: int = 6) -> dict:
    """Call Gemini via Vertex AI; return a dict with the SAME shape Azure returns.

    `messages` follows OpenAI's role/content format: [{"role": "user"|"assistant", "content": "..."}].
    Multi-turn is preserved. The "assistant" role is mapped to Gemini's "model" role.
    Empty/system messages in the list are ignored — pass system via system_prompt.

    Retries on 429 RESOURCE_EXHAUSTED with exponential backoff + jitter.
    Other exceptions pass through to the {"error": ...} return.
    """
    import asyncio as _asyncio  # noqa: PLC0415
    import random as _random  # noqa: PLC0415
    from google.genai import types as gt  # noqa: PLC0415

    contents = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            continue  # use the dedicated system_instruction below
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    cfg = gt.GenerateContentConfig(
        system_instruction=system_prompt or None,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        thinking_config=gt.ThinkingConfig(thinking_budget=thinking_budget),
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            # Hard wall-clock timeout per attempt as a belt-and-suspenders guard
            # on top of the client http_options timeout — a hung call becomes a
            # retryable TimeoutError instead of blocking the gather forever.
            r = await _asyncio.wait_for(
                client.aio.models.generate_content(
                    model=ep.deployment, contents=contents, config=cfg,
                ),
                timeout=200.0,
            )
            break
        except Exception as e:
            last_err = e
            msg = repr(e)
            is_429 = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            is_503 = "UNAVAILABLE" in msg or "503" in msg
            is_timeout = isinstance(e, _asyncio.TimeoutError) or "TimeoutError" in msg
            if not (is_429 or is_503 or is_timeout) or attempt == max_retries - 1:
                return {"error": msg}
            # Exponential backoff: 2, 4, 8, 16, 32, 64 seconds + jitter
            wait = (2 ** (attempt + 1)) + _random.uniform(0, 1.5)
            await _asyncio.sleep(wait)
    else:
        return {"error": repr(last_err)}

    text = (r.text or "").strip() if r.text else ""
    finish = r.candidates[0].finish_reason if r.candidates else None
    finish_str = _VERTEX_FINISH_MAP.get(
        getattr(finish, "name", str(finish)) if finish else "",
        str(finish).lower() if finish else None,
    )
    um = r.usage_metadata
    prompt_tok = (um.prompt_token_count or 0) if um else 0
    cand_tok = (um.candidates_token_count or 0) if um else 0
    think_tok = (getattr(um, "thoughts_token_count", None) or 0) if um else 0
    return {
        "response": text,
        "finish_reason": finish_str,
        "prompt_tokens": prompt_tok,
        # Include thinking in completion_tokens (billed-as-output, matches
        # OpenAI's reasoning-token convention). Also surface separately for analysis.
        "completion_tokens": cand_tok + think_tok,
        "thinking_tokens": think_tok,
    }


# --- Cost estimation ---------------------------------------------------------

# Conservative blended $/1K-token estimates (Azure pricing as of 2026-05).
# Used purely for the local running-cost log. Real billing is via az.
COST_TABLE = {
    # logical_name: (in_per_1k, out_per_1k)
    "gpt-5.5":       (0.005, 0.015),  # estimate
    "gpt-5.4":       (0.0025, 0.010),
    "gpt-5.4-nano":  (0.00025, 0.001),
    "gpt-4o":        (0.0025, 0.010),
    "gpt-4o-mini":   (0.00015, 0.0006),
    # Gemini on Vertex AI (per $/1K tokens; thinking tokens count as output).
    "gemini-3.1-pro-preview": (0.0025, 0.0125),  # preview, estimate parity with 2.5-pro paid tier
    "gemini-3.5-flash":       (0.0015, 0.009),   # GA 2026-05-19, $1.50/$9 per Mtok
    "gemini-3.1-flash-lite":  (0.0001, 0.0004),
    "gemini-2.5-pro":         (0.00125, 0.010),
    "gemini-2.5-flash":       (0.0003, 0.0025),
    "gemini-2.5-flash-lite":  (0.0001, 0.0004),
    # Claude via Claude Code CLI — actual cost is captured per-call from the
    # CLI's total_cost_usd (dominated by ~11.5k cached harness tokens/call);
    # these list-rates are only a fallback for estimate_cost().
    "claude-opus-4-8":   (0.005, 0.025),
    "claude-opus-4-7":   (0.005, 0.025),
    "claude-sonnet-4-6": (0.003, 0.015),
}


def is_gpt5(name: str) -> bool:
    return name.lower().startswith("gpt-5")


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    rate = COST_TABLE.get(model, (0.003, 0.01))
    return (in_tok / 1000.0) * rate[0] + (out_tok / 1000.0) * rate[1]


# --- Env loading -------------------------------------------------------------

def load_env_local(path: str | None = None) -> None:
    p = Path(path or os.environ.get("ENV_LOCAL_PATH")
             or Path(__file__).resolve().parents[3] / ".env.local")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# --- JSONL utils -------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- Cost guard wrapper that doesn't fail closed on az issues ---------------

class SoftCostGuard:
    """Wraps CostGuard with a try/except so a flaky az CLI doesn't kill the run.

    Still raises SafetyError when MTD exceeds cap. Tolerates az errors silently
    (since this is a short run and we also track local cost).
    """

    def __init__(self, mtd_cap: float, poll_seconds: int = 300):
        sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "0288d7a3-cf1a-40a6-b8b3-c40ca8e13eee")
        try:
            self.inner = CostGuard(mtd_cap_usd=mtd_cap, poll_seconds=poll_seconds, sub_id=sub_id)
        except Exception as e:
            print(f"[SoftCostGuard] cost guard init failed, disabling: {e!r}", file=sys.stderr)
            self.inner = None

    def maybe_check(self) -> None:
        if self.inner is None:
            return
        try:
            self.inner.maybe_check()
        except SafetyError:
            raise
        except Exception as e:
            print(f"[SoftCostGuard] az poll failed (continuing): {e!r}", file=sys.stderr)
