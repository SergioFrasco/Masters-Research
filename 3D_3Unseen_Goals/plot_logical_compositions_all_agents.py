"""
Plot Logical Composition Results from Stored JSON

Reads all_algo_logical_results.json and generates:
  1. Box plots per category (AND, OR, NOT, COMPLEX)
  2. Overall box plot across all tasks
  3. Grouped bar chart per task

Usage:
    python plot_logical_compositions.py [path/to/results.json]
"""

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# TASK DEFINITIONS (for ordering and category separators)
# ============================================================================

AND_TASKS = [
    "red AND box", "red AND sphere", "blue AND box",
    "blue AND sphere", "green AND box", "green AND sphere",
]
OR_TASKS = [
    "red OR blue", "red OR green", "blue OR green",
    "box OR sphere", "red OR box", "blue OR sphere",
]
NOT_TASKS = ["NOT red", "NOT blue", "NOT green", "NOT box", "NOT sphere"]
COMPLEX_TASKS = [
    "(red AND sphere) OR (blue AND box)",
    "(red OR blue) AND sphere",
    "red AND (sphere OR box)",
    "(green AND box) OR (NOT red AND sphere)",
]
ALL_TASK_NAMES = AND_TASKS + OR_TASKS + NOT_TASKS + COMPLEX_TASKS
CATEGORIES = ["AND", "OR", "NOT", "COMPLEX"]

ALGO_COLORS = {
    "SR": "#2ecc71", "DQN": "#3498db",
    "LSTM": "#e67e22", "WVF": "#9b59b6",
}


# ============================================================================
# PLOTTING
# ============================================================================

def plot_all(results, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    algos = list(results.keys())

    # ── 1. Box plots per category ────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=True)
    fig.suptitle("Logical Composition — All Algorithms",
                 fontsize=40, fontweight="bold", y=1.02)

    for ax, cat in zip(axes, CATEGORIES):
        data, labels, cols = [], [], []
        for algo in algos:
            rates = [v["success_rate"]
                     for v in results[algo].values() if v["category"] == cat]
            if rates:
                data.append(rates)
                labels.append(algo)
                cols.append(ALGO_COLORS.get(algo, "#95a5a6"))
        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2))
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c)
            p.set_alpha(0.7)
        for i, (d, c) in enumerate(zip(data, cols)):
            j = np.random.default_rng(42).normal(0, 0.04, len(d))
            ax.scatter(np.full(len(d), i + 1) + j, d, color=c,
                       edgecolor="black", lw=.5, s=40, zorder=5, alpha=.8)
        ax.set_xticklabels(labels, fontsize=30, fontweight="bold")
        ax.set_title(f"{cat}", fontsize=35, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Success Rate" if cat == "AND" else "", fontsize=30, fontweight="bold")
        ax.tick_params(axis='y', labelsize=28)
        ax.axhline(.5, color="gray", ls="--", lw=1, alpha=.6)
        ax.grid(axis="y", alpha=.25, ls="--")

    plt.tight_layout()
    p1 = out / "boxplot_by_category.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p1}")

    # ── 2. Overall box plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12))
    data, labels, cols = [], [], []
    for algo in algos:
        data.append([v["success_rate"] for v in results[algo].values()])
        labels.append(algo)
        cols.append(ALGO_COLORS.get(algo, "#95a5a6"))

    bp = ax.boxplot(data, patch_artist=True, widths=.55,
                    medianprops=dict(color="black", linewidth=2))
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c)
        p.set_alpha(0.7)
    for i, (d, c) in enumerate(zip(data, cols)):
        j = np.random.default_rng(42).normal(0, 0.05, len(d))
        ax.scatter(np.full(len(d), i + 1) + j, d, color=c,
                   edgecolor="black", lw=.5, s=50, zorder=5, alpha=.8)
    ax.set_xticklabels(labels, fontsize=32, fontweight="bold")
    ax.set_ylabel("Success Rate", fontsize=30, fontweight="bold")
    ax.set_title("Overall Logical Composition Performance\n"
                 "(all task types combined)", fontsize=36, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='y', labelsize=28)
    ax.axhline(.5, color="gray", ls="--", lw=1.5, alpha=.6, label="50 % baseline")
    ax.legend(fontsize=28, loc="lower right")
    ax.grid(axis="y", alpha=.25, ls="--")
    plt.tight_layout()
    p2 = out / "boxplot_overall.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p2}")

    # ── 3. Grouped bar chart per task ────────────────────────────────
    fig, ax = plt.subplots(figsize=(24, 14))
    tnames = [t for t in ALL_TASK_NAMES if any(t in results[a] for a in algos)]
    n_t, n_a = len(tnames), len(algos)
    bw = 0.8 / n_a
    xb = np.arange(n_t)

    for i, algo in enumerate(algos):
        rates = [results[algo].get(t, {}).get("success_rate", 0) for t in tnames]
        ax.bar(xb + (i - n_a / 2 + .5) * bw, rates, bw,
               color=ALGO_COLORS.get(algo, "#95a5a6"), alpha=.8,
               edgecolor="black", lw=.5, label=algo)

    ax.set_xticks(xb)
    ax.set_xticklabels(tnames, rotation=90, ha="center", fontsize=26)
    ax.set_ylabel("Success Rate", fontsize=30, fontweight="bold")
    ax.set_title("Per-Task Success Rate — All Algorithms",
                 fontsize=36, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelsize=28)
    ax.axhline(.5, color="gray", ls="--", lw=1, alpha=.6)
    ax.legend(fontsize=28, loc="upper right")
    ax.grid(axis="y", alpha=.25, ls="--")

    # Category separators
    cum = 0
    for group in [AND_TASKS, OR_TASKS, NOT_TASKS, COMPLEX_TASKS]:
        count = sum(1 for t in group if t in tnames)
        if cum > 0 and count > 0:
            ax.axvline(cum - .5, color="black", lw=1.5, alpha=.4)
        cum += count

    plt.tight_layout()
    p3 = out / "bar_per_task.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p3}")


def print_summary(results):
    algos = list(results.keys())
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cat in CATEGORIES:
        print(f"\n{cat}:")
        for algo in algos:
            rates = [v["success_rate"]
                     for v in results[algo].values() if v["category"] == cat]
            if rates:
                print(f"  {algo:5s}  mean={np.mean(rates):.3f}  "
                      f"std={np.std(rates):.3f}")
    print("\nOverall:")
    for algo in algos:
        rates = [v["success_rate"] for v in results[algo].values()]
        if rates:
            print(f"  {algo:5s}  mean={np.mean(rates):.3f}  "
                  f"std={np.std(rates):.3f}")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Accept path as CLI arg, otherwise use default
    default_path = "logical_composition_all_algos/all_algo_logical_results.json"
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_path)

    if not json_path.exists():
        print(f"ERROR: Results file not found: {json_path}")
        sys.exit(1)

    print(f"Loading results from {json_path} …")
    with open(json_path) as f:
        results = json.load(f)

    print(f"Algorithms found: {list(results.keys())}")
    print(f"Tasks per algo:   {[len(v) for v in results.values()]}")

    out_dir = json_path.parent
    print(f"\nGenerating plots → {out_dir}/")
    plot_all(results, out_dir)
    print_summary(results)