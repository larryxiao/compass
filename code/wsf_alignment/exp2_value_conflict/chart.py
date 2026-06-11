"""Generate the headline chart for Exp 2.

Bar chart: per-model compliance rate (headroom-only) with 95% Wilson CI,
plus a second panel showing care-vs-rule asymmetry per model.

Reads analysis_out.json.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping chart")
        return

    data = json.loads((HERE / "analysis_out.json").read_text())
    rank = data["ranked_models"]
    pm = data["per_model"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = [r["model"] for r in rank]
    rates = [r["compliance_rate"] for r in rank]
    cis = [r["ci"] for r in rank]
    err_lo = [rates[i] - cis[i][0] for i in range(len(rates))]
    err_hi = [cis[i][1] - rates[i] for i in range(len(rates))]
    colors = ["#2c7fb8" if r > 0.5 else "#d95f02" if r < 0.3 else "#999999"
              for r in rates]

    bars = ax1.bar(models, rates, color=colors, edgecolor="black", linewidth=0.7)
    ax1.errorbar(models, rates, yerr=[err_lo, err_hi], fmt="none",
                 ecolor="black", capsize=4)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Compliance rate (V1 had headroom)")
    ax1.set_title("Per-model priming-compliance rate, 95% Wilson CI")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
    for b, r in zip(bars, rates):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.02,
                 f"{r:.0%}", ha="center", va="bottom", fontsize=9)
    ax1.tick_params(axis="x", rotation=20)

    # Care vs Rule asymmetry per model
    asyms = [pm[m]["asymmetry_care_minus_rule"] for m in models]
    bar2 = ax2.bar(models, asyms,
                   color=["#1b9e77" if a > 0 else "#d95f02" for a in asyms],
                   edgecolor="black", linewidth=0.7)
    ax2.set_ylabel("Care-rate − Rule-rate (pp)")
    ax2.set_title("Priming-cluster asymmetry per model\n(positive = more steerable toward care)")
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.tick_params(axis="x", rotation=20)
    for b, a in zip(bar2, asyms):
        ax2.text(b.get_x() + b.get_width() / 2,
                 a + (0.01 if a >= 0 else -0.03),
                 f"{a:+.0%}", ha="center",
                 va="bottom" if a >= 0 else "top", fontsize=9)
    ymax = max(abs(a) for a in asyms) if asyms else 0.2
    ax2.set_ylim(-max(0.3, ymax * 1.3), max(0.3, ymax * 1.3))

    plt.suptitle("Exp 2 - Value-Conflict Priming Compliance, 25 dilemmas x 5 models")
    plt.tight_layout()
    out = HERE / "chart.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
