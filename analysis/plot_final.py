import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# Paper-level plotting style (journal-friendly)
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Clean axes for papers
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        # Grid (subtle)
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        # Lines
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
    }
)


OUTPUT_DIR = "paper_figures_final/qwen2.5"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# Figure size policy:
# Use Figure 1's *single subplot* aspect ratio for all standalone figures.
FIG1_FIGSIZE = (12.6, 3.1)
SINGLE_FIGSIZE = (FIG1_FIGSIZE[0] / 4.0, FIG1_FIGSIZE[1])


# Experiment directory mapping (4 datasets × 2 methods)
# EXPERIMENTS = {
#     "gsm8k": {
#         "baseline": "outputs_gsm8k/baseline",
#         "fdt": "outputs_gsm8k/alpha0.6",
#     },
#     "cmmlu": {
#         "baseline": "outputs_cmmlu/baseline",
#         "fdt": "outputs_cmmlu/alpha0.6",
#     },
#     "sharegpt": {
#         "baseline": "outputs_sharegpt/baseline",
#         "fdt": "outputs_sharegpt/alpha0.6",
#     },
#     "mbpp": {
#         "baseline": "outputs_mbpp/baseline",
#         "fdt": "outputs_mbpp/alpha0.6",
#     },
# }

EXPERIMENTS = {
    "gsm8k": {
        "baseline": "outputs_qwen2.5_gsm8k/baseline",
        "fdt": "outputs_qwen2.5_gsm8k/alpha0.6",
    },
    "cmmlu": {
        "baseline": "outputs_qwen2.5_cmmlu/baseline",
        "fdt": "outputs_qwen2.5_cmmlu/alpha0.6",
    },
    "sharegpt": {
        "baseline": "outputs_qwen2.5_sharegpt/baseline",
        "fdt": "outputs_qwen2.5_sharegpt/alpha0.6",
    },
    "mbpp": {
        "baseline": "outputs_qwen2.5_mbpp/baseline",
        "fdt": "outputs_qwen2.5_mbpp/alpha0.6",
    },
}


def _discover_alpha_experiments(dataset: str) -> list[tuple[float, str]]:
    """Discover alpha experiment directories for a dataset.

    Looks for directories like: outputs_<dataset>/alpha0.6
    Returns a list of (alpha_value, exp_dir) sorted by alpha.
    """

    root_dir = Path(f"outputs_{dataset}")
    if not root_dir.exists():
        return []

    alpha_pattern = re.compile(r"^alpha(?P<alpha>[0-9]+(?:\.[0-9]+)?)$")
    found: list[tuple[float, str]] = []

    for child in root_dir.iterdir():
        if not child.is_dir():
            continue
        match = alpha_pattern.match(child.name)
        if not match:
            continue
        try:
            alpha_value = float(match.group("alpha"))
        except ValueError:
            continue
        found.append((alpha_value, str(child)))

    found.sort(key=lambda x: x[0])
    return found


def _smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average smoothing.

    If `values` is shorter than `window`, returns the original array.
    """

    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _prepare_smoothed_series(
    losses: np.ndarray,
    max_steps: int,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, smoothed_losses) clipped to max_steps.

    Aligns the smoothed series with the *end* of the smoothing window.
    For window W, the first smoothed point corresponds to step W.
    """

    if losses.ndim != 1:
        losses = losses.reshape(-1)

    steps = np.arange(1, len(losses) + 1)
    losses_smooth = _smooth_curve(losses, window=smooth_window)

    if smooth_window > 1 and len(losses) >= smooth_window:
        steps_smooth = steps[smooth_window - 1 :]
    else:
        steps_smooth = steps

    mask = steps_smooth <= max_steps
    return steps_smooth[mask], losses_smooth[mask]


def plot_4dataset_training_curves(max_steps: int = 500, smooth_window: int = 20) -> None:
    """Figure 1: 4 datasets training loss curves (1×4)."""

    fig, axes = plt.subplots(1, 4, figsize=FIG1_FIGSIZE, sharey=True)

    datasets = ["gsm8k", "cmmlu", "sharegpt", "mbpp"]
    dataset_names = {
        "gsm8k": "GSM8K (Math)",
        "cmmlu": "CMMLU (Chinese)",
        "sharegpt": "ShareGPT (Dialogue)",
        "mbpp": "MBPP (Code)",
    }

    # Consistent, paper-friendly colors
    colors = {"Baseline": "#D62728", "FDA (α=0.6)": "#1F77B4"}

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        for method_name, method_key in [("Baseline", "baseline"), ("FDA (α=0.6)", "fdt")]:
            exp_dir = EXPERIMENTS[dataset][method_key]
            losses_file = os.path.join(exp_dir, "training_losses.npy")

            if not os.path.exists(losses_file):
                print(f"⚠️  未找到: {losses_file}")
                continue

            losses = np.load(losses_file)
            steps_smooth, losses_smooth = _prepare_smoothed_series(
                losses,
                max_steps=max_steps,
                smooth_window=smooth_window,
            )

            ax.plot(
                steps_smooth,
                losses_smooth,
                label=method_name,
                color=colors[method_name],
                alpha=0.95,
            )

        ax.set_xlabel("Training Steps", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Training Loss", fontsize=9)

        ax.xaxis.set_major_locator(mticker.MultipleLocator(100))

        # Title under each subplot
        ax.text(
            0.5,
            -0.28,
            f"({chr(97 + idx)}) {dataset_names[dataset]}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        # Reduce repeated legend clutter: show legend once
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, max_steps)

    # Leave space for the titles under subplots
    fig.subplots_adjust(bottom=0.30, wspace=0.22)

    # out_base = "fig01_trainloss_4datasets"
    out_base = "qwen2.5_trainloss_4datasets"
    for fmt in ["pdf", "png"]:
        plt.savefig(f"{OUTPUT_DIR}/{out_base}.{fmt}", format=fmt)

    print(f"✓ 图 1: 4 数据集训练曲线 → {OUTPUT_DIR}/{out_base}.pdf")
    plt.close()


def plot_alpha_sweep_for_dataset(
    dataset: str,
    max_steps: int = 500,
    smooth_window: int = 20,
) -> None:
    """Per-dataset alpha sweep training curves.

    Generates one figure per dataset. Style and smoothing match Figure 1.
    Curves included:
    - Baseline (if exists)
    - All discovered alpha experiments under outputs_<dataset>/alpha*
    """

    dataset_names = {
        "gsm8k": "GSM8K (Math)",
        "cmmlu": "CMMLU (Chinese)",
        "sharegpt": "ShareGPT (Dialogue)",
        "mbpp": "MBPP (Code)",
    }

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)

    # Baseline
    baseline_dir = EXPERIMENTS.get(dataset, {}).get("baseline")
    if baseline_dir:
        losses_file = os.path.join(baseline_dir, "training_losses.npy")
        if os.path.exists(losses_file):
            losses = np.load(losses_file)
            steps_smooth, losses_smooth = _prepare_smoothed_series(
                losses,
                max_steps=max_steps,
                smooth_window=smooth_window,
            )
            ax.plot(
                steps_smooth,
                losses_smooth,
                label="Baseline",
                color="#D62728",
                alpha=0.95,
            )
        else:
            print(f"⚠️  未找到: {losses_file}")

    # Alpha experiments (only selected alpha points)
    alpha_exps_all = _discover_alpha_experiments(dataset)
    if not alpha_exps_all:
        print(f"⚠️  未找到 alpha 实验目录: outputs_{dataset}/alpha*")

    selected_alphas = [0.6, 0.8, 1.0, 1.4]
    alpha_exps: list[tuple[float, str]] = []
    for alpha_value, exp_dir in alpha_exps_all:
        if any(np.isclose(alpha_value, a, atol=1e-9) for a in selected_alphas):
            alpha_exps.append((alpha_value, exp_dir))

    # Report missing requested alphas
    present = {round(a, 6) for a, _ in alpha_exps}
    missing = [a for a in selected_alphas if round(a, 6) not in present]
    if missing:
        missing_str = ", ".join(f"{a:g}" for a in missing)
        print(f"⚠️  {dataset}: 缺少指定 alpha 目录: {missing_str}")

    # Use a qualitative palette with good separability (baseline red is fixed)
    tab10 = sns.color_palette("tab10", n_colors=10)
    # Fixed alpha -> color mapping for consistency across runs/plots
    alpha_color_map: dict[float, tuple[float, float, float]] = {
        0.6: tab10[0],  # blue
        0.8: tab10[9],  # cyan (avoid red-like hue)
        1.0: tab10[2],  # green
        1.4: tab10[4],  # purple
    }

    for (alpha_value, exp_dir) in alpha_exps:
        # pick mapped color; fallback to a neutral tab color if unexpected
        color = None
        for a, c in alpha_color_map.items():
            if np.isclose(alpha_value, a, atol=1e-9):
                color = c
                break
        if color is None:
            color = tab10[7]

        losses_file = os.path.join(exp_dir, "training_losses.npy")
        if not os.path.exists(losses_file):
            print(f"⚠️  未找到: {losses_file}")
            continue

        losses = np.load(losses_file)
        steps_smooth, losses_smooth = _prepare_smoothed_series(
            losses,
            max_steps=max_steps,
            smooth_window=smooth_window,
        )

        ax.plot(
            steps_smooth,
            losses_smooth,
            label=f"FDA (α={alpha_value:g})",
            color=color,
            alpha=0.95,
        )

    ax.set_xlabel("Training Steps", fontsize=9)
    ax.set_ylabel("Training Loss", fontsize=9)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))

    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, max_steps)

    # No title for non-Fig.1 plots; reduce bottom whitespace accordingly.
    fig.subplots_adjust(bottom=0.18)

    out_base = f"fig02_alpha_sweep_{dataset}"
    for fmt in ["pdf", "png"]:
        plt.savefig(f"{OUTPUT_DIR}/{out_base}.{fmt}", format=fmt)

    print(f"✓ {dataset} alpha 曲线 → {OUTPUT_DIR}/{out_base}.pdf")
    plt.close()


def _pick_losses_file_from_experiment_dir(exp_dir: str) -> str | None:
    """Pick a training_losses.npy path from an experiment directory.

    Supports two layouts:
    - <exp_dir>/training_losses.npy
    - <exp_dir>/alpha0.6/training_losses.npy (preferred if exists)
    """

    direct = os.path.join(exp_dir, "training_losses.npy")
    if os.path.exists(direct):
        return direct

    preferred = os.path.join(exp_dir, "alpha0.6", "training_losses.npy")
    if os.path.exists(preferred):
        return preferred

    return None


def plot_cmmlu_lora_rank_sweep(
    max_steps: int = 500,
    smooth_window: int = 20,
) -> None:
    """GSM8K: training curves for different LoRA ranks.

    Requirement:
    - lora_r16 uses the experiment results from outputs_gsm8k/alpha0.6
    """

    dataset = "cmmlu"
    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)

    # Baseline
    baseline_dir = EXPERIMENTS.get(dataset, {}).get("baseline")
    if baseline_dir:
        losses_file = os.path.join(baseline_dir, "training_losses.npy")
        if os.path.exists(losses_file):
            losses = np.load(losses_file)
            steps_smooth, losses_smooth = _prepare_smoothed_series(
                losses,
                max_steps=max_steps,
                smooth_window=smooth_window,
            )
            ax.plot(
                steps_smooth,
                losses_smooth,
                label="Baseline",
                color="#D62728",
                alpha=0.95,
            )
        else:
            print(f"⚠️  未找到: {losses_file}")

    # Rank experiments
    rank_map: dict[int, str] = {
        # r16: must use alpha0.6 experiment
        16: "outputs_cmmlu/alpha0.6",
        8: "outputs_cmmlu/lora_r8",
        32: "outputs_cmmlu/lora_r32",
        64: "outputs_cmmlu/lora_r64",
    }

    ranks = [8, 16, 32, 64]
    # Use a qualitative palette with good separability (baseline red is fixed)
    tab10 = sns.color_palette("tab10", n_colors=10)
    rank_color_map: dict[int, tuple[float, float, float]] = {
        8: tab10[9],   # cyan (avoid red-like hue)
        16: tab10[0],  # blue
        32: tab10[2],  # green
        64: tab10[4],  # purple
    }

    for rank in ranks:
        color = rank_color_map.get(rank, tab10[7])
        exp_dir = rank_map[rank]
        losses_file = _pick_losses_file_from_experiment_dir(exp_dir)
        if not losses_file:
            print(f"⚠️  未找到 rank={rank} 的 training_losses.npy: {exp_dir}")
            continue

        losses = np.load(losses_file)
        steps_smooth, losses_smooth = _prepare_smoothed_series(
            losses,
            max_steps=max_steps,
            smooth_window=smooth_window,
        )

        ax.plot(
            steps_smooth,
            losses_smooth,
            label=f"FDA (r={rank}, α=0.6)",
            color=color,
            alpha=0.95,
        )

    ax.set_xlabel("Training Steps", fontsize=9)
    ax.set_ylabel("Training Loss", fontsize=9)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))

    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, max_steps)

    # No title for non-Fig.1 plots; reduce bottom whitespace accordingly.
    fig.subplots_adjust(bottom=0.18)

    out_base = "fig03_lora_rank_sweep_cmmlu"
    for fmt in ["pdf", "png"]:
        plt.savefig(f"{OUTPUT_DIR}/{out_base}.{fmt}", format=fmt)

    print(f"✓ GSM8K LoRA rank 曲线 → {OUTPUT_DIR}/{out_base}.pdf")
    plt.close()


def main() -> None:
    print("=" * 70)
    print("📊 仅生成图 1（4 数据集训练曲线）")
    print("=" * 70)
    print()
    print("输出目录:", OUTPUT_DIR)
    print()

    plot_4dataset_training_curves()

    # # Per-dataset alpha sweep figures
    # for dataset in ["gsm8k", "cmmlu", "sharegpt", "mbpp"]:
    #     plot_alpha_sweep_for_dataset(dataset)

    # # GSM8K LoRA rank sweep figure
    # plot_cmmlu_lora_rank_sweep()

    print()
    print("=" * 70)
    print("✅ 生成完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
