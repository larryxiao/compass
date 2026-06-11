"""WS-F factory orchestrator: self-improving dilemma generator.

Per iteration:
  1. Pick axis-targets and category-targets (round-robin across the 6 axes × 7 cats).
  2. Setup agent generates N candidates from seed dilemmas + current prompt template.
  3. Schema-validate; rejects go to errors.jsonl.
  4. Decision agents (5 models) answer each survivor cold.
  5. Evaluator ensemble (2+ judges) scores each (dilemma, decisions) tuple.
  6. Refiner ranks → keeps top-K → optionally rewrites the setup prompt.
  7. Checkpoint: state/iter_NNN.json
  8. Append kept candidates to output/dilemmas_factory.jsonl.

CLI:
  python factory.py --iterations 5 --candidates-per-iter 20 --keep-top 10

For the validation run (cost <$1):
  python factory.py --iterations 1 --candidates-per-iter 2 --keep-top 1 \
      --decision-models gpt-4o,gpt-5.4,gpt-4o-mini
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

import yaml
from openai import AsyncAzureOpenAI

# Re-use safety utilities from WS-B.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "wsb_synthdata"))
from safety import assert_deployment_allowed, CostGuard, SafetyError  # noqa: E402

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import (  # noqa: E402
    AXES, CATEGORIES, validate_dilemma,
)
from agents.setup import INITIAL_PROMPT_TEMPLATE, generate_candidate  # noqa: E402
from agents.decide import decide_all  # noqa: E402
from agents.evaluate import evaluate_ensemble  # noqa: E402
from agents.refine import (  # noqa: E402
    rank_and_select, compute_diagnostics, refine_prompt,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def get_endpoint(cfg: dict, model_condition: str) -> dict:
    for e in cfg["endpoints"]:
        if e["model_condition"] == model_condition:
            return e
    raise KeyError(f"no endpoint with model_condition={model_condition!r}")


def build_client(ep: dict) -> AsyncAzureOpenAI:
    api_key = os.environ.get(ep["api_key_env"])
    if not api_key:
        raise SystemExit(
            f"missing env {ep['api_key_env']} for endpoint {ep['name']!r}"
        )
    return AsyncAzureOpenAI(
        api_key=api_key,
        api_version=ep["api_version"],
        azure_endpoint=ep["base_url"],
        max_retries=2,
        timeout=180.0,
    )


def axis_target_str(combo: tuple[str, ...]) -> str:
    """Convert ('loyalty_vs_honesty', 'rules_vs_outcomes') to natural language."""
    if len(combo) == 1:
        return f"{combo[0]} heavy"
    if len(combo) == 2:
        return f"{combo[0]} heavy, {combo[1]} moderate"
    return f"{combo[0]} heavy, {combo[1]} moderate, {combo[2]} light"


def plan_targets(n: int, seed_dilemmas: list[dict]) -> list[tuple[str, tuple[str, ...]]]:
    """Plan (category, axis_combo) targets to span gaps in the seed pool.

    We round-robin across categories, and for each category we pick an
    axis combo that's UNDER-represented in the seed pool.
    """
    cat_counts = {c: 0 for c in CATEGORIES}
    axis_combo_counts: dict[tuple, int] = {}
    for d in seed_dilemmas:
        cat_counts[d["category"]] = cat_counts.get(d["category"], 0) + 1
        combo = tuple(sorted(d["axes_in_play"]))
        axis_combo_counts[combo] = axis_combo_counts.get(combo, 0) + 1

    # Generate all axis 2-combos & 3-combos
    from itertools import combinations
    all_combos: list[tuple[str, ...]] = []
    for k in (2, 3):
        for combo in combinations(AXES, k):
            all_combos.append(tuple(sorted(combo)))
    # Rank combos by least-used in seed (then deterministic order)
    all_combos.sort(key=lambda c: (axis_combo_counts.get(c, 0), c))

    # Round-robin categories, ordered by least-used in seed.
    cats_sorted = sorted(CATEGORIES, key=lambda c: (cat_counts.get(c, 0), c))
    combo_cycle = cycle(all_combos)
    targets = []
    cat_iter = cycle(cats_sorted)
    for _ in range(n):
        targets.append((next(cat_iter), next(combo_cycle)))
    return targets


async def run_iteration(
    iter_idx: int,
    seed_dilemmas: list[dict],
    setup_prompt_template: str,
    setup_client: AsyncAzureOpenAI,
    setup_deployment: str,
    decide_clients: dict,        # model_condition -> (client, deployment)
    judge_endpoints: list[tuple[str, AsyncAzureOpenAI, str]],
    cost_guard: CostGuard,
    cfg: dict,
    candidates_per_iter: int,
    keep_top: int,
    next_id_start: int,
    runs_jsonl: Path,
    errors_jsonl: Path,
) -> dict:
    """Run one factory iteration. Returns checkpoint dict."""
    print(f"\n========== ITER {iter_idx} ==========")
    print(f"  seeds: {len(seed_dilemmas)}  candidates: {candidates_per_iter}  keep_top: {keep_top}")

    targets = plan_targets(candidates_per_iter, seed_dilemmas)
    axes_list = ", ".join(AXES)
    categories_list = " | ".join(CATEGORIES)
    token_budgets = cfg["token_budgets"]

    # ── Step 1: generate candidates ─────────────────────────────────────
    print("[setup] generating candidates...")
    setup_results = []
    for i, (cat, combo) in enumerate(targets):
        next_id = f"F{(next_id_start + i):03d}"
        try:
            cost_guard.maybe_check()
        except SafetyError as e:
            print(f"  cost guard tripped: {e}")
            break
        res = await generate_candidate(
            setup_client, setup_deployment, setup_prompt_template,
            seed_dilemmas, axis_target_str(combo), cat, next_id,
            axes_list, categories_list,
            max_completion_tokens=token_budgets["setup_max_completion"],
        )
        res["iter"] = iter_idx
        setup_results.append(res)
        _append_jsonl(runs_jsonl, res)
        ok = "OK" if "dilemma" in res else "FAIL"
        print(f"  {next_id} [{cat}, {'+'.join(combo)}] -> {ok}")

    # ── Step 2: validate ────────────────────────────────────────────────
    candidates: list[dict] = []
    for res in setup_results:
        if "dilemma" not in res:
            _append_jsonl(errors_jsonl, {
                "iter": iter_idx, "stage": "setup",
                "error": res.get("error"), "raw": res.get("raw", "")[:2000],
                "axis_target": res.get("axis_target"),
                "category_target": res.get("category_target"),
            })
            continue
        d = res["dilemma"]
        # If the model failed to set id, patch it from the request.
        if "id" not in d:
            d["id"] = res.get("next_id")
        errs = validate_dilemma(d)
        if errs:
            _append_jsonl(errors_jsonl, {
                "iter": iter_idx, "stage": "schema",
                "dilemma_id": d.get("id"),
                "errors": errs,
                "dilemma_partial": {k: d.get(k) for k in ("id", "title", "category", "axes_in_play")},
            })
            print(f"  [schema FAIL] {d.get('id')}: {errs[0]}")
            continue
        candidates.append(d)
    print(f"[setup] {len(candidates)}/{len(setup_results)} passed schema")

    if not candidates:
        return {
            "iter": iter_idx,
            "ts": utc_now(),
            "n_candidates": 0,
            "kept": [], "rejected": [], "metrics": {},
            "setup_prompt_template": setup_prompt_template,
            "note": "no candidates passed schema validation",
        }

    # ── Step 3: decision agents ─────────────────────────────────────────
    print(f"[decide] fanning out across {len(decide_clients)} models...")
    decisions_by_dilemma: dict[str, list[dict]] = {}
    for d in candidates:
        try:
            cost_guard.maybe_check()
        except SafetyError as e:
            print(f"  cost guard tripped: {e}")
            break
        decs = await decide_all(
            decide_clients, d,
            max_completion_tokens=token_budgets["decide_max_completion"],
            concurrency=4,
        )
        for rec in decs:
            rec["iter"] = iter_idx
            _append_jsonl(runs_jsonl, rec)
        decisions_by_dilemma[d["id"]] = decs
        n_ok = sum(1 for r in decs if not r.get("error") and r.get("response"))
        print(f"  {d['id']}: {n_ok}/{len(decs)} agents responded")

    # ── Step 4: evaluator ensemble ─────────────────────────────────────
    print(f"[evaluate] running {len(judge_endpoints)}-judge ensemble per dilemma...")
    evaluated = []
    for d in candidates:
        decs = decisions_by_dilemma.get(d["id"], [])
        try:
            cost_guard.maybe_check()
        except SafetyError as e:
            print(f"  cost guard tripped: {e}")
            break
        eval_result = await evaluate_ensemble(
            judge_endpoints, d, decs,
            max_completion_tokens=token_budgets["evaluate_max_completion"],
            concurrency=2,
        )
        for jr in eval_result["per_judge"]:
            jr["iter"] = iter_idx
            _append_jsonl(runs_jsonl, jr)
        evaluated.append({"dilemma": d, "evaluation": eval_result})
        agg = eval_result["aggregated"]
        if "mean_score" in agg:
            print(
                f"  {d['id']}: mean={agg['mean_score']:.2f} "
                f"(real={agg.get('real_dilemma'):.1f} "
                f"bal={agg.get('options_balanced'):.1f} "
                f"rel={agg.get('relatable'):.1f} "
                f"char={agg.get('character_revealing'):.1f} "
                f"split={agg.get('model_split'):.1f}) "
                f"axis_align={agg.get('axis_alignment')}"
            )
        else:
            print(f"  {d['id']}: EVAL FAIL ({agg.get('error')})")

    # ── Step 5: rank + refine ──────────────────────────────────────────
    selection = rank_and_select(evaluated, keep_top=keep_top, min_mean_score=3.5)
    diagnostics = compute_diagnostics(
        evaluated, decisions_by_dilemma,
        seed_axes=[d["axes_in_play"] for d in seed_dilemmas],
        seed_categories=[d["category"] for d in seed_dilemmas],
    )

    print(f"[rank] kept {len(selection['kept'])}, rejected {len(selection['rejected'])}")
    print(f"[metrics] quality={diagnostics['quality']} pass_rate={diagnostics['pass_rate']} "
          f"split={diagnostics['split']} diversity={diagnostics['diversity']}")

    print("[refine] checking if prompt needs revision...")
    refine_res = await refine_prompt(
        setup_client, setup_deployment,
        setup_prompt_template, evaluated, keep_top,
        max_completion_tokens=token_budgets["refine_max_completion"],
    )
    refine_res["iter"] = iter_idx
    _append_jsonl(runs_jsonl, refine_res)

    new_prompt = refine_res.get("new_prompt_template", setup_prompt_template)
    if refine_res.get("adopted"):
        print(f"[refine] adopted new prompt. Diagnosis: {refine_res.get('diagnosis', '')[:160]}")
    elif refine_res.get("error"):
        print(f"[refine] error, keeping current prompt: {refine_res.get('error')}")
    else:
        print("[refine] no change needed.")

    # ── Step 6: assemble checkpoint ────────────────────────────────────
    kept_dilemmas = []
    rejected_dilemmas = []
    for item in evaluated:
        d = item["dilemma"]
        agg = (item.get("evaluation") or {}).get("aggregated") or {}
        record = {
            "dilemma": d,
            "evaluation_aggregated": agg,
            "iter": iter_idx,
        }
        if any(k["id"] == d["id"] for k in selection["kept"]):
            kept_dilemmas.append(record)
        else:
            rejected_dilemmas.append(record)

    checkpoint = {
        "iter": iter_idx,
        "ts": utc_now(),
        "n_targets": len(targets),
        "n_setup_results": len(setup_results),
        "n_passed_schema": len(candidates),
        "kept": selection["kept"],
        "rejected": selection["rejected"],
        "kept_full": kept_dilemmas,
        "rejected_full": rejected_dilemmas,
        "metrics": diagnostics,
        "refiner": {
            "adopted": refine_res.get("adopted", False),
            "diagnosis": refine_res.get("diagnosis"),
            "changes": refine_res.get("changes"),
            "error": refine_res.get("error"),
        },
        "setup_prompt_template_before": setup_prompt_template,
        "setup_prompt_template_after": new_prompt,
    }
    return checkpoint


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--seed-dilemmas",
                    default=str(Path(__file__).parent.parent / "dilemmas" / "dilemmas.jsonl"),
                    help="JSONL of seed dilemmas (the bar)")
    ap.add_argument("--n-seeds", type=int, default=20,
                    help="How many seed dilemmas to load (use 2 for validation)")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--candidates-per-iter", type=int, default=20)
    ap.add_argument("--keep-top", type=int, default=10)
    ap.add_argument("--decision-models", default=None,
                    help="Override comma-separated list, e.g. gpt-4o,gpt-5.4,gpt-4o-mini")
    ap.add_argument("--evaluator-models", default=None,
                    help="Override comma-separated list, e.g. gpt-4o,gpt-5.4")
    ap.add_argument("--setup-model", default=None,
                    help="Override setup deployment model_condition")
    ap.add_argument("--run-tag", default=None,
                    help="Tag for state/output filenames (default: timestamp)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    deny = cfg["deny_deployment_prefixes"]
    for e in cfg["endpoints"]:
        assert_deployment_allowed(e["deployment"], deny)

    # Resolve model assignments
    setup_mc = args.setup_model or cfg["defaults"]["setup_model"]
    decision_mcs = (
        [s.strip() for s in args.decision_models.split(",")] if args.decision_models
        else cfg["defaults"]["decision_models"]
    )
    judge_mcs = (
        [s.strip() for s in args.evaluator_models.split(",")] if args.evaluator_models
        else cfg["defaults"]["evaluator_models"]
    )
    for mc in [setup_mc] + decision_mcs + judge_mcs:
        # raises KeyError if unknown
        get_endpoint(cfg, mc)
    print(f"setup_model: {setup_mc}")
    print(f"decision_models: {decision_mcs}")
    print(f"evaluator_models: {judge_mcs}")

    # Cost guard (orchestrator-level)
    sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "0288d7a3-cf1a-40a6-b8b3-c40ca8e13eee")
    cost_guard = CostGuard(
        mtd_cap_usd=cfg["global"]["mtd_emergency_stop"],
        poll_seconds=cfg["global"]["cost_poll_seconds"],
        sub_id=sub_id,
    )

    # Build clients (one per unique deployment)
    setup_ep = get_endpoint(cfg, setup_mc)
    setup_client = build_client(setup_ep)
    decide_clients: dict[str, tuple[AsyncAzureOpenAI, str]] = {}
    for mc in decision_mcs:
        ep = get_endpoint(cfg, mc)
        decide_clients[mc] = (build_client(ep), ep["deployment"])
    judge_endpoints: list[tuple[str, AsyncAzureOpenAI, str]] = []
    for mc in judge_mcs:
        ep = get_endpoint(cfg, mc)
        judge_endpoints.append((mc, build_client(ep), ep["deployment"]))

    # Load seeds
    seed_path = Path(args.seed_dilemmas)
    all_seeds = [json.loads(ln) for ln in seed_path.read_text().splitlines() if ln.strip()]
    seed_dilemmas = all_seeds[: args.n_seeds]
    # Sanity: all seeds must pass validator
    for s in seed_dilemmas:
        errs = validate_dilemma(s)
        if errs:
            raise SystemExit(f"seed {s.get('id')} fails validator: {errs[:3]}")
    print(f"loaded {len(seed_dilemmas)} seed dilemmas")

    # Output paths
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    here = Path(__file__).parent
    state_dir = here / "state" / run_tag
    out_dir = here / "output"
    state_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_jsonl = out_dir / f"runs_{run_tag}.jsonl"
    errors_jsonl = out_dir / f"errors_{run_tag}.jsonl"
    accepted_jsonl = out_dir / "dilemmas_factory.jsonl"

    # Iterate
    setup_prompt_template = INITIAL_PROMPT_TEMPLATE
    current_seeds = list(seed_dilemmas)
    # Reserve id range F001+
    next_id = 1 + sum(1 for ln in accepted_jsonl.read_text().splitlines()
                      if ln.strip()) if accepted_jsonl.exists() else 1

    iter_summaries = []
    for it in range(args.iterations):
        checkpoint = await run_iteration(
            iter_idx=it,
            seed_dilemmas=current_seeds,
            setup_prompt_template=setup_prompt_template,
            setup_client=setup_client,
            setup_deployment=setup_ep["deployment"],
            decide_clients=decide_clients,
            judge_endpoints=judge_endpoints,
            cost_guard=cost_guard,
            cfg=cfg,
            candidates_per_iter=args.candidates_per_iter,
            keep_top=args.keep_top,
            next_id_start=next_id,
            runs_jsonl=runs_jsonl,
            errors_jsonl=errors_jsonl,
        )
        ckpt_path = state_dir / f"iter_{it:03d}.json"
        ckpt_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2, default=str))
        print(f"[checkpoint] -> {ckpt_path}")

        # Append kept dilemmas to the accumulating pool.
        for rec in checkpoint.get("kept_full", []):
            d = rec["dilemma"]
            with accepted_jsonl.open("a") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # Update for next iter
        setup_prompt_template = checkpoint["setup_prompt_template_after"]
        next_id += checkpoint["n_targets"]
        # Seeds for next iter: previous-iter kept ⊕ original (capped).
        kept_only = [rec["dilemma"] for rec in checkpoint.get("kept_full", [])]
        if kept_only:
            # Mix: half new, half original (so we don't drift away from the bar).
            cap = max(len(seed_dilemmas), len(kept_only) + len(seed_dilemmas) // 2)
            current_seeds = (kept_only + seed_dilemmas)[:cap]

        iter_summaries.append({
            "iter": it,
            "metrics": checkpoint["metrics"],
            "refiner_adopted": checkpoint["refiner"].get("adopted"),
            "n_kept": len(checkpoint["kept"]),
        })

    # Final summary
    print("\n========== RUN SUMMARY ==========")
    for s in iter_summaries:
        m = s["metrics"]
        print(f"iter {s['iter']}: quality={m.get('quality')} pass_rate={m.get('pass_rate')} "
              f"split={m.get('split')} diversity={m.get('diversity')} "
              f"kept={s['n_kept']} prompt_changed={s['refiner_adopted']}")
    summary_path = state_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "iter_summaries": iter_summaries,
        "args": vars(args),
        "final_setup_prompt_template": setup_prompt_template,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote run summary -> {summary_path}")
    print(f"wrote runs -> {runs_jsonl}")
    print(f"wrote errors -> {errors_jsonl}")
    print(f"appended accepted dilemmas -> {accepted_jsonl}")

    # Close clients
    await setup_client.close()
    for client, _ in decide_clients.values():
        await client.close()
    for (_, client, _) in judge_endpoints:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
