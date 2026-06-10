#!/usr/bin/env python3
"""
分析达到指定损失阈值所需的时间

对比 baseline vs FDA-SOC (alpha=0.6)
计算：
1. 达到某个损失阈值所需的步数
2. 实际时间（根据 step_time_ms 累加）
3. 加速比
"""

import os
import json
import pandas as pd
import numpy as np

# 数据集配置
DATASETS = {
    "cmmlu": {
        "base_dir": "/root/nvme0n1/Noneq_Neural_Network/FDT_Init/outputs_cmmlu",
        "thresholds": [8.0, 5.0, 3.0, 2.0, 1.0, 0.5],  # 根据实际损失范围调整
    },
    "gsm8k": {
        "base_dir": "/root/nvme0n1/Noneq_Neural_Network/FDT_Init/outputs_gsm8k",
        "thresholds": [8.0, 5.0, 3.0, 2.0, 1.5, 1.0],
    },
    "mbpp": {
        "base_dir": "/root/nvme0n1/Noneq_Neural_Network/FDT_Init/outputs_mbpp",
        "thresholds": [6.0, 4.0, 2.5, 1.5, 1.0],
    },
    "sharegpt": {
        "base_dir": "/root/nvme0n1/Noneq_Neural_Network/FDT_Init/outputs_sharegpt",
        "thresholds": [5.0, 3.0, 2.0, 1.0, 0.5],
    },
}

# 种子配置
SEEDS = {
    "1107": ("baseline", "alpha0.6"),
    "123": ("baseline_seed123", "alpha0.6_seed123"),
    "42": ("baseline_seed42", "alpha0.6_seed42"),
}


def find_time_to_threshold(csv_path, threshold, max_steps=2500):
    """
    找到达到指定损失阈值所需的时间（秒）

    Returns:
        steps: 达到阈值的步数（如果没达到返回None）
        time_sec: 达到阈值累计的时间（如果没达到返回None）
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Warning: 无法读取 {csv_path}: {e}")
        return None, None

    # 过滤掉空值
    df = df[df["train_loss"].notna()]

    # 找到第一个低于阈值的行
    below_threshold = df[df["train_loss"] <= threshold]

    if len(below_threshold) == 0:
        return None, None

    # 第一个达到阈值的步数
    first_idx = below_threshold.index[0]
    steps = int(df.loc[first_idx, "step"])

    # 累计时间（考虑warmup前期可能时间不同）
    cumulative_time_ms = df.loc[:first_idx, "step_time_ms"].sum()
    time_sec = cumulative_time_ms / 1000.0

    return steps, time_sec


def analyze_dataset(dataset_name, dataset_config):
    """分析单个数据集"""
    base_dir = dataset_config["base_dir"]
    thresholds = dataset_config["thresholds"]

    results = []

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"{'='*60}")

    for seed_name, (base_dirname, fda_dirname) in SEEDS.items():
        print(f"\n--- Seed {seed_name} ---")

        base_csv = os.path.join(base_dir, base_dirname, "training_log.csv")
        fda_csv = os.path.join(base_dir, fda_dirname, "training_log.csv")

        # 检查文件是否存在
        if not os.path.exists(base_csv) or not os.path.exists(fda_csv):
            print(f"  跳过: 文件不存在")
            continue

        for threshold in thresholds:
            base_steps, base_time = find_time_to_threshold(base_csv, threshold)
            fda_steps, fda_time = find_time_to_threshold(fda_csv, threshold)

            # 只有当两者都达到阈值时才记录
            if base_steps is not None and fda_steps is not None:
                speedup_steps = base_steps / fda_steps if fda_steps > 0 else 0
                speedup_time = base_time / fda_time if fda_time > 0 else 0
                time_saved_min = (base_time - fda_time) / 60.0

                results.append({
                    "dataset": dataset_name,
                    "seed": seed_name,
                    "threshold": threshold,
                    "base_steps": base_steps,
                    "fda_steps": fda_steps,
                    "base_time_sec": base_time,
                    "fda_time_sec": fda_time,
                    "speedup_steps": speedup_steps,
                    "speedup_time": speedup_time,
                    "time_saved_min": time_saved_min,
                })

                print(f"  Loss ≤ {threshold:.1f}: "
                      f"Baseline {base_steps:4d} steps ({base_time:6.1f}s) → "
                      f"FDA {fda_steps:4d} steps ({fda_time:6.1f}s) | "
                      f"Speedup: {speedup_time:.2f}× | "
                      f"Saved: {time_saved_min:.1f} min")
            else:
                if base_steps is None:
                    print(f"  Loss ≤ {threshold:.1f}: Baseline 未达到")
                if fda_steps is None:
                    print(f"  Loss ≤ {threshold:.1f}: FDA 未达到")

    return results


def print_summary_table(all_results):
    """打印汇总表格"""
    print(f"\n{'='*80}")
    print(f"汇总表格 (所有数据集 × 所有种子)")
    print(f"{'='*80}")

    # 按数据集和阈值分组
    df = pd.DataFrame(all_results)

    if df.empty:
        print("没有结果")
        return

    # 计算每个数据集-阈值组合的平均值
    summary = df.groupby(["dataset", "threshold"]).agg({
        "speedup_time": ["mean", "std"],
        "time_saved_min": ["mean", "std"],
    }).reset_index()

    summary.columns = ["dataset", "threshold", "speedup_mean", "speedup_std",
                       "time_saved_mean", "time_saved_std"]

    print("\n平均加速比和节省时间:")
    print(f"{'Dataset':<12} {'Threshold':<10} {'Speedup':<12} {'Time Saved':<15}")
    print(f"{'-'*60}")

    for _, row in summary.iterrows():
        print(f"{row['dataset']:<12} {row['threshold']:<10.1f} "
              f"{row['speedup_mean']:.2f}× (±{row['speedup_std']:.2f})     "
              f"{row['time_saved_mean']:.1f} min (±{row['time_saved_std']:.1f})")

    # 打印最显著的加速
    print(f"\n{'='*80}")
    print("最显著的加速效果:")
    best = df.nlargest(5, "speedup_time")
    for _, row in best.iterrows():
        print(f"  {row['dataset']} (Seed {row['seed']}, Loss ≤ {row['threshold']}): "
              f"{row['speedup_time']:.2f}×, 节省 {row['time_saved_min']:.1f} 分钟")

    # 计算总体统计
    print(f"\n{'='*80}")
    print("总体统计:")
    print(f"  配置数量: {len(df)} (数据集 × 种子 × 阈值)")
    print(f"  平均加速比: {df['speedup_time'].mean():.2f}× (±{df['speedup_time'].std():.2f})")
    print(f"  平均节省时间: {df['time_saved_min'].mean():.1f} 分钟 (±{df['time_saved_min'].std():.1f})")
    print(f"  中位加速比: {df['speedup_time'].median():.2f}×")
    print(f"  中位节省时间: {df['time_saved_min'].median():.1f} 分钟")

    # 按阈值分组统计
    print(f"\n按阈值的加速比统计:")
    threshold_summary = df.groupby("threshold").agg({
        "speedup_time": ["mean", "std", "min", "max"],
        "time_saved_min": ["mean", "std"],
    })
    print(threshold_summary.to_string())


def generate_latex_table(all_results):
    """生成LaTeX表格"""
    df = pd.DataFrame(all_results)

    if df.empty:
        return

    # 按数据集分组，选择代表性阈值
    print(f"\n{'='*80}")
    print("LaTeX 表格 (可用于论文):")
    print(f"{'='*80}")

    # 选择一个有代表性的阈值（比如中位数阈值）
    # 实际使用时可以根据需要调整

    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        print(f"\n% {dataset.upper()}")

        # 按阈值分组
        for threshold in dataset_df["threshold"].unique():
            threshold_df = dataset_df[dataset_df["threshold"] == threshold]

            if len(threshold_df) >= 2:  # 至少有2个种子的数据
                speedup_mean = threshold_df["speedup_time"].mean()
                speedup_std = threshold_df["speedup_time"].std()
                time_saved = threshold_df["time_saved_min"].mean()

                print(f"% Loss ≤ {threshold:.1f}: {speedup_mean:.2f}× speedup, "
                      f"{time_saved:.1f} min saved (±{speedup_std:.2f})")


def main():
    all_results = []

    for dataset_name, dataset_config in DATASETS.items():
        results = analyze_dataset(dataset_name, dataset_config)
        all_results.extend(results)

    print_summary_table(all_results)
    generate_latex_table(all_results)

    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = "/root/nvme0n1/Noneq_Neural_Network/FDT_Init/time_to_threshold_results.json"
        df.to_json(output_file, orient="records", indent=2)
        print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
