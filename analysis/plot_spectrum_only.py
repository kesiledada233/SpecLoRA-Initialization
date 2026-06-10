"""plot_spectrum_only.py

只做一件事：绘制 LoRA 权重在初始化前/后的功率谱密度（PSD）对比图（log-log）。

流程：
1) 加载模型并应用 LoRA；
2) 选择一个 LoRA 参数（默认第一个 lora_A.weight）；
3) 计算初始化前谱密度 + 幂律指数 alpha（复用 measure_alpha.py）；
4) （可选）对 LoRA 权重应用 FDA 初始化（实现仍来自 fdt_init.apply_fdt_to_lora；参数名保持兼容）；
5) 计算初始化后谱密度 + alpha，并绘制对比图。

输出（--out_dir，默认 outputs_spectrum_only）：
- lora_power_spectrum.pdf
- lora_power_spectrum.png
- lora_power_spectrum.npz
- config_spectrum_only.json
- init_info_spectrum_only.json

示例：
python plot_spectrum_only.py --use_fdt_init --fdt_alpha 0.6 --fdt_method fft --device npu:0

"""

import os
import sys
import argparse
import json
from typing import Any
import math

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ==================== 3) 工具函数 ====================
def convert_to_json_serializable(obj: Any):
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(v) for v in obj]
    # numpy 标量需要兼容；若 numpy 未安装，类型检查会抛 NameError，直接跳过即可
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None and isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        return float(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (int, float, str)):
        return obj
    return str(obj)


def _select_lora_param_name(model, layer_name: str | None) -> str:
    if layer_name:
        return layer_name

    first_any = None
    for name, _ in model.named_parameters():
        if "lora_" in name and name.endswith("weight"):
            if first_any is None:
                first_any = name
            if "lora_A" in name:
                return name

    if first_any is None:
        raise RuntimeError("未找到任何 LoRA 参数（请确认已成功应用 LoRA）")
    return first_any


def _get_param_by_name(model, name: str):
    params = dict(model.named_parameters())
    if name not in params:
        available = [k for k in params.keys() if "lora_" in k and k.endswith("weight")]
        raise KeyError(
            f"找不到参数: {name}\n可用 LoRA 参数示例(最多 10 个): {available[:10]}"
        )
    return params[name]


def spectrum_and_alpha_from_tensor(tensor, measure_alpha_func):
    alpha, freqs, power = measure_alpha_func(tensor, method="fft", return_full_spectrum=True)
    import numpy as np

    return float(alpha), freqs.astype(np.float64), power.astype(np.float64)


def _draw_spectrum_curve(
    ax,
    freqs: np.ndarray,
    power: np.ndarray,
    alpha: float,
    label: str,
    color: str,
    linewidth: float,
    alpha_val: float,
):
    ax.loglog(
        freqs,
        power,
        linestyle="-",
        linewidth=linewidth,
        alpha=alpha_val,
        label=label,
        color=color,
    )


def _style_axes_arrows(ax, axis_lw: float = 2.2):
    """去除图框/刻度，绘制加粗带箭头的坐标轴（使用 axes fraction 坐标，兼容 log-log）。"""

    # 1) 去掉图框（spines）
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 2) 去掉网格、刻度与刻度数值
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    # 3) 在 axes fraction 坐标系里画箭头坐标轴
    # x 轴：从 (0,0) 到 (1,0)
    ax.annotate(
        "",
        xy=(1.02, 0.0),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=axis_lw, color="black", shrinkA=0, shrinkB=0),
        clip_on=False,
    )
    # y 轴：从 (0,0) 到 (0,1)
    ax.annotate(
        "",
        xy=(0.0, 1.02),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=axis_lw, color="black", shrinkA=0, shrinkB=0),
        clip_on=False,
    )


# ==================== 4) 参数与主逻辑 ====================
def get_args():
    ap = argparse.ArgumentParser(description="OpenPangu FDA 初始化前后谱密度对比（仅谱图）")

    # 模型与 LoRA
    ap.add_argument(
        "--model_path",
        type=str,
        default="/opt/pangu/openPangu-Embedded-7B-V1.1",
        help="预训练模型路径",
    )
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
    )

    # FDA 初始化（参数名保持兼容）
    ap.add_argument("--use_fdt_init", action="store_true")
    ap.add_argument("--fdt_alpha", type=float, default=1.1)
    ap.add_argument("--fdt_method", type=str, default="fft", choices=["fft", "ar"])

    ap.add_argument(
        "--layer_name",
        type=str,
        default=None,
        help="要绘制谱密度对比的 LoRA 参数名；不填则自动选择第一个 lora_A.weight",
    )

    # 设备与输出
    ap.add_argument("--device", type=str, default="npu:0")
    ap.add_argument("--seed", type=int, default=1107)
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs_spectrum_only",
        help="输出目录（默认：outputs_spectrum_only）",
    )
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()


def main():
    args = get_args()

    # 延迟导入重依赖（保证 -h 可用）
    try:
        import numpy as np
        import torch
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("错误: 缺少绘图/数值依赖（numpy/torch/matplotlib）")
        print("详细错误:", e)
        return

    # 绘图风格（与 plot_init.py 保持一致）
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.figsize": (8, 6),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )

    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:
        print("错误: 未找到 transformers，请在正确的环境里运行（与 plot_init.py 同环境）")
        print("详细错误:", e)
        return

    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except Exception as e:
        print("错误: 未找到 peft 库，请先安装: pip install peft")
        print("详细错误:", e)
        return

    try:
        from measure_alpha import measure_alpha as measure_alpha_func
    except Exception as e:
        print("错误: measure_alpha.py 不可用，无法进行幂律拟合")
        print("详细错误:", e)
        return

    # FDA 初始化实现仍来自 fdt_init（只改展示名称，不改导入路径）
    try:
        from fdt_init import apply_fdt_to_lora

        fdt_init_available = True
    except Exception:
        apply_fdt_to_lora = None
        fdt_init_available = False

    os.makedirs(args.out_dir, exist_ok=True)

    # 保存配置
    config_file = os.path.join(args.out_dir, "config_spectrum_only.json")
    with open(config_file, "w") as f:
        json.dump(convert_to_json_serializable(vars(args)), f, indent=2)
    print(f"[配置] 已保存到: {config_file}")

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备
    device = torch.device(args.device)
    device_type = args.device.split(":")[0]
    if device_type == "npu":
        try:
            import torch_npu

            torch_npu.npu.set_device(device)
            torch_npu.npu.manual_seed_all(args.seed)
            print(f"[设备] ✓ NPU 初始化成功: {device}")
        except Exception as e:
            print(f"[设备] ❌ NPU 初始化失败: {e}")
            return
    else:
        print(f"[设备] 使用: {device}")

    # 1) 加载模型
    print("=" * 70)
    print("📦 步骤 1: 加载模型")
    print("=" * 70)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 2) 应用 LoRA
    print("=" * 70)
    print("🔧 步骤 2: 应用 LoRA")
    print("=" * 70)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # 3) 选择参数并计算 Before
    chosen_name = _select_lora_param_name(model, args.layer_name)
    chosen_param = _get_param_by_name(model, chosen_name)
    print(f"[谱密度] 选定参数: {chosen_name}")

    before_tensor = chosen_param.detach().clone().float().cpu()
    alpha_before, freqs_before, power_before = spectrum_and_alpha_from_tensor(
        before_tensor, measure_alpha_func
    )
    print(f"[alpha] 初始化前( measure_alpha ): {alpha_before:.3f}")

    # 4) （可选）FDA 初始化
    print("=" * 70)
    print("🎯 步骤 3: FDA 初始化（可选）")
    print("=" * 70)
    init_info = {
        "use_fdt": bool(args.use_fdt_init),
        "alpha": float(args.fdt_alpha) if args.use_fdt_init else None,
        "method": str(args.fdt_method) if args.use_fdt_init else "peft_default",
    }

    if args.use_fdt_init:
        if not fdt_init_available or apply_fdt_to_lora is None:
            print("[FDA] ✗ 未加载 fdt_init 模块，无法应用 FDA 初始化")
        else:
            print(f"[FDA] 应用 FDA 初始化: alpha={args.fdt_alpha:.2f}, method={args.fdt_method}")
            apply_fdt_to_lora(
                model,
                alpha=args.fdt_alpha,
                method=args.fdt_method,
                verbose=args.verbose,
            )
            print("[FDA] ✓ 初始化完成")
    else:
        print("[FDA] 使用 PEFT 默认初始化 (Kaiming Uniform + Zero)")

    # 5) After
    after_tensor = chosen_param.detach().clone().float().cpu()
    alpha_after, freqs_after, power_after = spectrum_and_alpha_from_tensor(
        after_tensor, measure_alpha_func
    )
    print(f"[alpha] 初始化后( measure_alpha ): {alpha_after:.3f}")

    # 保存 init_info
    init_info["measured"] = {
        "layer_name": chosen_name,
        "alpha_before": float(alpha_before),
        "alpha_after": float(alpha_after),
        "target_alpha": float(args.fdt_alpha) if args.use_fdt_init else None,
    }
    init_info_file = os.path.join(args.out_dir, "init_info_spectrum_only.json")
    with open(init_info_file, "w") as f:
        json.dump(convert_to_json_serializable(init_info), f, indent=2)
    print(f"[FDA] 初始化信息已保存到: {init_info_file}")

    # 6) 对齐频率并绘图
    print("=" * 70)
    print("📈 步骤 4: 绘制初始化前后谱密度对比 (log-log)")
    print("=" * 70)

    common_freqs = np.intersect1d(freqs_before, freqs_after)
    if common_freqs.size < 10:
        raise RuntimeError(
            f"初始化前后频率点交集过少，无法对比绘图: before={len(freqs_before)}, after={len(freqs_after)}"
        )

    before_map = {int(f): i for i, f in enumerate(freqs_before.astype(np.int64))}
    after_map = {int(f): i for i, f in enumerate(freqs_after.astype(np.int64))}
    idx_b = np.array([before_map[int(f)] for f in common_freqs], dtype=np.int64)
    idx_a = np.array([after_map[int(f)] for f in common_freqs], dtype=np.int64)

    freqs = common_freqs.astype(np.float64)
    p_before = power_before[idx_b]
    p_after = power_after[idx_a]

    fig_path_pdf = os.path.join(args.out_dir, "lora_power_spectrum.pdf")
    fig_path_png = os.path.join(args.out_dir, "lora_power_spectrum.png")
    npz_path = os.path.join(args.out_dir, "lora_power_spectrum.npz")

    fig, ax = plt.subplots()

    _draw_spectrum_curve(
        ax,
        freqs,
        p_before,
        alpha_before,
        label="Baseline",
        color="#D62728",
        linewidth=0.9,
        alpha_val=0.5,
    )

    after_label = f"FDA($\\alpha={args.fdt_alpha:.2f}$)" if args.use_fdt_init else "After"
    _draw_spectrum_curve(
        ax,
        freqs,
        p_after,
        alpha_after,
        label=after_label,
        color="#1F77B4",
        linewidth=4.0,
        alpha_val=1.0,
    )

    ax.set_xlabel("Frequency", fontweight="bold")
    ax.set_ylabel("Power Spectral Density (PSD)", fontweight="bold")
    # 需求：去除网格、图例、拟合线、坐标轴数值、图框；坐标轴加粗并带箭头
    _style_axes_arrows(ax, axis_lw=2.2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(fig_path_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        npz_path,
        layer_name=chosen_name,
        freqs=freqs,
        power_before=p_before,
        power_after=p_after,
        alpha_before=np.float32(alpha_before),
        alpha_after=np.float32(alpha_after),
        target_alpha=np.float32(args.fdt_alpha if args.use_fdt_init else np.nan),
    )

    print(f"[输出] ✓ 矢量图: {fig_path_pdf}")
    print(f"[输出] ✓ 预览图: {fig_path_png}")
    print(f"[输出] ✓ 数据: {npz_path}")
    print(f"[总结] α(初始化前 -> 初始化后): {alpha_before:.3f} -> {alpha_after:.3f}")


if __name__ == "__main__":
    main()
