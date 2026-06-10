"""plot_init.py

只做两件事：
1) 加载模型并应用 LoRA；
2) (可选) 对 LoRA 权重应用 FDA 初始化，然后绘制初始化前/后功率谱密度对比图（双对数），并用 measure_alpha.py 的同一逻辑拟合幂律指数 α。

输出（--out_dir，可选，默认 init_outputs）：
- lora_power_spectrum.pdf           : [新增] 学术级矢量图
- lora_power_spectrum.png           : 预览图
- lora_power_spectrum.npz           : 频率与 PSD 数值 + α
- lora_weight_heatmap_before.pdf    : [可选] LoRA 权重矩阵热力图（初始化前）
- lora_weight_heatmap_before.png    : [可选] 预览图
- lora_weight_heatmap_after.pdf     : [可选] LoRA 权重矩阵热力图（初始化后）
- lora_weight_heatmap_after.png     : [可选] 预览图
- config_init_only.json             : 命令行参数
- init_info_only.json               : 初始化信息（含目标 α、测得 α）

手动指定 FDA 参数（例如 α=1.1，fft；参数名保持 --use_fdt_init/--fdt_alpha/--fdt_method 不变）
python plot_init.py  --use_fdt_init --fdt_alpha 0.6 --fdt_method fft --device npu:0 --viz_lora_matrix

"""

import os
os.environ["DISABLE_NPU_FUSED_ATTENTION"] = "1"
os.environ["NPU_FUSED_INFER_ATTENTION"] = "0"

import sys
import argparse
import json
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== 1. 学术绘图全局配置 ====================
plt.rcParams.update({
    "font.family": "serif",             # 使用衬线体
    "font.serif": ['DejaVu Sans'],  
    "mathtext.fontset": "stix",         # LaTeX 公式字体风格
    "font.size": 12,                    # 全局字号
    "axes.labelsize": 14,               # 轴标签字号
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.figsize": (8, 6),           # 4:3 比例，适合单栏或半页图
    "axes.grid": True,                  # 开启网格
    "grid.alpha": 0.3,                  # 网格透明度
    "grid.linestyle": "--",             # 网格样式
})
# ==========================================================

from transformers import AutoModelForCausalLM

# ==== PEFT / LoRA ====
try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError as e:
    print("错误: 未找到 peft 库，请先安装: pip install peft")
    print("详细错误:", e)
    PEFT_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print("\n" + "=" * 70)
print("🔧 加载 FDA 初始化与 α 拟合模块")
print("=" * 70)

try:
    from fdt_init import apply_fdt_to_lora
    FDT_INIT_AVAILABLE = True
    print("[导入] ✓ fdt_init.apply_fdt_to_lora")
except ImportError as e:
    FDT_INIT_AVAILABLE = False
    apply_fdt_to_lora = None
    print("[导入] ✗ fdt_init 导入失败:", e)

try:
    # 关键要求：幂律指数拟合必须复用 measure_alpha.py
    from measure_alpha import measure_alpha
    MEASURE_ALPHA_AVAILABLE = True
    print("[导入] ✓ measure_alpha.measure_alpha")
except ImportError as e:
    MEASURE_ALPHA_AVAILABLE = False
    measure_alpha = None
    print("[导入] ✗ measure_alpha 导入失败:", e)

print("=" * 70 + "\n")


# ==================== 工具函数 ====================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (int, float, str)):
        return obj
    else:
        return str(obj)


def _select_lora_param_name(model: torch.nn.Module, layer_name: str | None) -> str:
    if layer_name:
        return layer_name

    # 默认优先选第一个 lora_A.weight（便于展示），找不到再退化到任意 lora_*.weight
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


def _get_param_by_name(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    params = dict(model.named_parameters())
    if name not in params:
        available = [k for k in params.keys() if "lora_" in k and k.endswith("weight")]
        raise KeyError(
            f"找不到参数: {name}\n"
            f"可用 LoRA 参数示例(最多 10 个): {available[:10]}"
        )
    return params[name]


def spectrum_and_alpha_from_tensor(tensor: torch.Tensor):
    """复用 measure_alpha.py：返回 (alpha, freqs, power)。"""
    if not MEASURE_ALPHA_AVAILABLE or measure_alpha is None:
        raise RuntimeError("measure_alpha 不可用：请确认 measure_alpha.py 可被导入")
    alpha, freqs, power = measure_alpha(tensor, method="fft", return_full_spectrum=True)
    return float(alpha), freqs.astype(np.float64), power.astype(np.float64)


def _draw_spectrum_curve(ax, freqs, power, alpha, label, color, style="line", linewidth=1.0, alpha_val=1.0):
    """
    绘制单条谱线及拟合线
    style: 'line' (实线, 无marker), 'marker' (带点), 'both'
    """
    # 1. 绘制数据
    marker = None
    if style == 'marker':
        marker = '.'
    
    ax.loglog(freqs, power, 
              linestyle='-' if style != 'marker' else 'None',
              marker=marker, 
              markersize=3, 
              linewidth=linewidth, 
              alpha=alpha_val, 
              label=label, 
              color=color)

    # 2. 绘制拟合线 (虚线)
    if np.isfinite(alpha):
        # 用一个参考点确定常数 C，使 C*f^{-alpha} 贴近曲线
        ref_idx = int(len(freqs) * 0.15) # 取低频偏后一点的位置，避开直流分量干扰
        ref_idx = max(0, min(ref_idx, len(freqs) - 1))
        
        # 为了让拟合线稍微浮在数据上方一点点以便观察，可以乘以一个小系数，或者直接取均值
        C = power[ref_idx] * (freqs[ref_idx] ** alpha)
        fit_line = C * (freqs ** (-alpha))
        
        # 拟合线颜色稍微加深一点或保持一致
        ax.loglog(freqs, fit_line, "--", linewidth=1.5, alpha=0.8, color=color)


def _tensor_to_2d_numpy(
    tensor: torch.Tensor,
    max_rows: int = 512,
    max_cols: int = 512,
) -> np.ndarray:
    """将任意 tensor 转为 2D numpy（用于 imshow）。

    - 2D: 原样
    - 1D: 变为 (1, N)
    - >=3D: 变为 (dim0, -1)
    过大时按等间隔下采样到 (<=max_rows, <=max_cols)。
    """
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim >= 3:
        arr = arr.reshape(arr.shape[0], -1)

    rows, cols = arr.shape
    if rows > max_rows:
        idx_r = np.linspace(0, rows - 1, max_rows).astype(np.int64)
        arr = arr[idx_r, :]
        rows = arr.shape[0]
    if cols > max_cols:
        idx_c = np.linspace(0, cols - 1, max_cols).astype(np.int64)
        arr = arr[:, idx_c]

    return arr


def _plot_single_weight_heatmap(
    matrix: np.ndarray,
    title: str,
    layer_name: str,
    out_pdf: str,
    out_png: str,
    vmin: float,
    vmax: float,
    cmap: str = "RdBu_r",
) -> None:
    """绘制单张 LoRA 权重矩阵热力图。

    约束：
    - 通过 set_box_aspect(1)（或降级方案）将子图图框固定为 1:1。
    - vmin/vmax 由外部传入，便于 Before/After 共用同一色标范围。
    """
    # 适当给 colorbar 留空间；axes 本身保持正方形
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")

    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")

    # ax.set_title(title)
    ax.set_xlabel("Input Dim")
    ax.set_ylabel("Rank")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # fig.suptitle(f"LoRA Weight Heatmap | {layer_name}")

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ==================== 参数配置 ====================
def get_args():
    ap = argparse.ArgumentParser(description="OpenPangu FDA 初始化 + 谱密度绘图（无训练）")

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

    # FDT 初始化
    ap.add_argument("--use_fdt_init", action="store_true")
    ap.add_argument("--fdt_alpha", type=float, default=1.1)
    ap.add_argument(
        "--fdt_method",
        type=str,
        default="fft",
        choices=["fft", "ar"],
    )
    ap.add_argument(
        "--init_preset",
        type=str,
        default=None,
        choices=["baseline", "soft", "medium", "strong"],
        help="方便实验的预设: baseline/soft/medium/strong",
    )

    ap.add_argument(
        "--layer_name",
        type=str,
        default=None,
        help="要绘制谱密度对比的 LoRA 参数名；不填则自动选择第一个 lora_A.weight",
    )

    # 可视化：LoRA 权重矩阵（Before/After/Diff）
    ap.add_argument(
        "--viz_lora_matrix",
        action="store_true",
        help="绘制所选 LoRA 参数矩阵热力图（前/后/差分）",
    )
    ap.add_argument(
        "--viz_max_rows",
        type=int,
        default=512,
        help="热力图最大行数（过大则等间隔下采样）",
    )
    ap.add_argument(
        "--viz_max_cols",
        type=int,
        default=512,
        help="热力图最大列数（过大则等间隔下采样）",
    )

    # 设备与输出
    ap.add_argument("--device", type=str, default="npu:0")
    ap.add_argument("--seed", type=int, default=1107)
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs_spectrum_init",
        help="输出目录（默认：outputs_spectrum_init）",
    )
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()


# ==================== 主逻辑 ====================
def main():
    if not PEFT_AVAILABLE:
        print("错误: PEFT 不可用，无法应用 LoRA")
        return

    if not MEASURE_ALPHA_AVAILABLE:
        print("错误: measure_alpha.py 不可用，无法进行幂律拟合")
        return

    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 保存配置
    config_file = os.path.join(args.out_dir, "config_init_only.json")
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

    # ===== 步骤 1: 加载模型 =====
    print("=" * 70)
    print("📦 步骤 1: 加载模型")
    print("=" * 70)

    print(f"[模型] 路径: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    total_params, _ = count_parameters(model)
    print(f"[模型] ✓ 加载完成: {total_params/1e9:.2f}B 参数\n")

    # ===== 步骤 2: 应用 LoRA =====
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

    print(
        f"[LoRA] r={args.lora_r}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}"
    )
    print(f"[LoRA] target_modules={args.lora_target_modules}")

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(
        f"[LoRA] ✓ 可训练参数: {trainable_params:,} "
        f"({trainable_params/total_params*100:.4f}%)\n"
    )

    # ===== 选定要画谱的 LoRA 参数，并记录“初始化前”的谱 =====
    chosen_name = _select_lora_param_name(model, args.layer_name)
    chosen_param = _get_param_by_name(model, chosen_name)
    print(f"[谱密度] 选定参数: {chosen_name}")

    before_tensor = chosen_param.detach().clone().float().cpu()
    alpha_before, freqs_before, power_before = spectrum_and_alpha_from_tensor(before_tensor)
    print(f"[alpha] 初始化前( measure_alpha ): {alpha_before:.3f}")

    # ===== 步骤 3: FDT 初始化（可选 + 预设） =====
    print("=" * 70)
    print("🎯 步骤 3: FDA 初始化")
    print("=" * 70)

    if args.init_preset:
        preset_configs = {
            "baseline": {
                "use_fdt": False,
                "alpha": None,
                "name": "PEFT Default (Kaiming+Zero)",
            },
            "soft": {
                "use_fdt": True,
                "alpha": 0.8,
                "name": "FDA-Soft (α=0.8)",
            },
            "medium": {
                "use_fdt": True,
                "alpha": 1.1,
                "name": "FDA-Medium (α=1.1)",
            },
            "strong": {
                "use_fdt": True,
                "alpha": 1.5,
                "name": "FDA-Strong (α=1.5)",
            },
        }
        cfg = preset_configs[args.init_preset]
        print(f"[预设] 使用预设: {cfg['name']}")
        if cfg["use_fdt"]:
            args.use_fdt_init = True
            args.fdt_alpha = cfg["alpha"]

    init_info = {
        "use_fdt": args.use_fdt_init,
        "preset": args.init_preset,
        "alpha": None,
        "method": "peft_default",
        "lora_a_init": "kaiming_uniform",
        "lora_b_init": "zero",
    }

    if args.use_fdt_init:
        if not FDT_INIT_AVAILABLE:
            print("[FDA] ✗ 未加载 fdt_init 模块，无法应用 FDA 初始化")
        else:
            print(
                f"[FDA] 应用 FDA 初始化: alpha={args.fdt_alpha:.2f}, "
                f"method={args.fdt_method}"
            )
            apply_fdt_to_lora(
                model,
                alpha=args.fdt_alpha,
                method=args.fdt_method,
                verbose=args.verbose,
            )
            init_info["alpha"] = args.fdt_alpha
            init_info["method"] = args.fdt_method
            print("[FDA] ✓ 初始化完成")
    else:
        print("[FDA] 使用 PEFT 默认初始化 (Kaiming Uniform + Zero)")

    init_info_file = os.path.join(args.out_dir, "init_info_only.json")
    # ===== 记录“初始化后”的谱（同一参数）=====
    after_tensor = chosen_param.detach().clone().float().cpu()
    alpha_after, freqs_after, power_after = spectrum_and_alpha_from_tensor(after_tensor)
    print(f"[alpha] 初始化后( measure_alpha ): {alpha_after:.3f}")

    # ===== 可视化：LoRA 权重矩阵热力图（前/后/差分） =====
    if getattr(args, "viz_lora_matrix", False):
        print("=" * 70)
        print("🧩 可视化: LoRA 权重矩阵热力图 (before/after 分开输出)")
        print("=" * 70)
        heat_pdf_before = os.path.join(args.out_dir, "lora_weight_heatmap_before.pdf")
        heat_png_before = os.path.join(args.out_dir, "lora_weight_heatmap_before.png")
        heat_pdf_after = os.path.join(args.out_dir, "lora_weight_heatmap_after.pdf")
        heat_png_after = os.path.join(args.out_dir, "lora_weight_heatmap_after.png")

        w_before = _tensor_to_2d_numpy(before_tensor, max_rows=args.viz_max_rows, max_cols=args.viz_max_cols)
        w_after = _tensor_to_2d_numpy(after_tensor, max_rows=args.viz_max_rows, max_cols=args.viz_max_cols)

        if w_before.shape != w_after.shape:
            raise RuntimeError(f"before/after 矩阵形状不一致: {w_before.shape} vs {w_after.shape}")

        max_abs = float(np.nanmax(np.abs(np.stack([w_before, w_after], axis=0))))
        if not np.isfinite(max_abs) or max_abs == 0.0:
            max_abs = 1.0

        _plot_single_weight_heatmap(
            w_before,
            title="Before",
            layer_name=chosen_name,
            out_pdf=heat_pdf_before,
            out_png=heat_png_before,
            vmin=-max_abs,
            vmax=max_abs,
        )
        _plot_single_weight_heatmap(
            w_after,
            title="After",
            layer_name=chosen_name,
            out_pdf=heat_pdf_after,
            out_png=heat_png_after,
            vmin=-max_abs,
            vmax=max_abs,
        )

        # 一点点数值摘要，帮助量化差异
        diff = w_after - w_before
        print(f"[heatmap] ✓ 保存: {heat_pdf_before}")
        print(f"[heatmap] ✓ 保存: {heat_png_before}")
        print(f"[heatmap] ✓ 保存: {heat_pdf_after}")
        print(f"[heatmap] ✓ 保存: {heat_png_after}")
        print(
            "[heatmap] 统计: "
            f"before(std={float(np.std(w_before)):.4g}), "
            f"after(std={float(np.std(w_after)):.4g}), "
            f"diff(std={float(np.std(diff)):.4g}, max_abs={float(np.max(np.abs(diff))):.4g})"
        )

    # 保存初始化信息（包含测得 α）
    init_info["measured"] = {
        "layer_name": chosen_name,
        "alpha_before": float(alpha_before),
        "alpha_after": float(alpha_after),
    }
    with open(init_info_file, "w") as f:
        json.dump(convert_to_json_serializable(init_info), f, indent=2)
    print(f"[FDA] 初始化信息已保存到: {init_info_file}\n")

    # ===== 步骤 4: 绘图（对比 + 拟合线） =====
    print("=" * 70)
    print("📈 步骤 4: 绘制初始化前后谱密度对比 (log-log)")
    print("=" * 70)

    # 频率索引通常一致；如果不一致就取交集对齐
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

    # 文件名
    fig_path_pdf = os.path.join(args.out_dir, "lora_power_spectrum.pdf") # 学术首选
    fig_path_png = os.path.join(args.out_dir, "lora_power_spectrum.png") # 快速预览
    npz_path = os.path.join(args.out_dir, "lora_power_spectrum.npz")

    # 绘制
    fig, ax = plt.subplots()

    # 4.1 绘制 Before (作为背景对比)
    # 策略：颜色稍淡，线条变细，去掉 marker，使其不抢眼但清晰可见
    _draw_spectrum_curve(
        ax, freqs, p_before, alpha_before,
        label=r"Baseline",
        color="#3b75af",  # Steel Blue 类颜色
        style="line",     # 不画点，只画线
        linewidth=0.8,
        alpha_val=0.6     # 半透明
    )

    # 4.2 绘制 After (重点展示)
    # 策略：颜色鲜艳(红色)，线条加粗
    after_label_text = r"FDA($\alpha=%.2f$)" % args.fdt_alpha if args.use_fdt_init else r"After"
    _draw_spectrum_curve(
        ax, freqs, p_after, alpha_after,
        label=f"{after_label_text}",
        color="#d62728",  # 砖红色
        style="line",
        linewidth=2.0,    # 加粗
        alpha_val=1.0
    )

    # 4.3 标签与修饰
    ax.set_xlabel(r"Frequency Index")
    ax.set_ylabel(r"Power Spectral Density (PSD)")
    
    # 标题简化，去掉冗余变量名
    # 如果必须保留层信息，建议放在副标题或文件名中，或者使用 text 标注在图内
    # ax.set_title("Power Spectrum of LoRA Weights", fontweight='bold')
    
    # 图例优化
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    
    # # 标注具体的层名称（小字，底部）
    # plt.figtext(0.5, 0.01, f"Layer: {chosen_name}", wrap=True, horizontalalignment='center', fontsize=8, color='gray')

    plt.tight_layout()
    # 留出一点底部空间给 figtext
    plt.subplots_adjust(bottom=0.15)

    # 保存
    plt.savefig(fig_path_pdf, format='pdf', bbox_inches="tight")
    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    plt.close()

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
    print("\n✅ 完成（仅初始化 + 绘图，无训练/数据集）")


if __name__ == "__main__":
    main()