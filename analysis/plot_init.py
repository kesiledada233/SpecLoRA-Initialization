"""plot_init.py


1)  LoRA
2) ()  LoRA  FDA / measure_alpha.py  α

--out_dir init_outputs
- lora_power_spectrum.pdf           : [] 
- lora_power_spectrum.png           : 
- lora_power_spectrum.npz           :  PSD  + α
- lora_weight_heatmap_before.pdf    : [] LoRA 
- lora_weight_heatmap_before.png    : [] 
- lora_weight_heatmap_after.pdf     : [] LoRA 
- lora_weight_heatmap_after.png     : [] 
- config_init_only.json             : 
- init_info_only.json               :  α α

 FDA  α=1.1fft --use_fdt_init/--fdt_alpha/--fdt_method 
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

plt.rcParams.update({
    "font.family": "serif",             # 
    "font.serif": ['DejaVu Sans'],  
    "mathtext.fontset": "stix",         # LaTeX 
    "font.size": 12,                    # 
    "axes.labelsize": 14,               # 
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.figsize": (8, 6),           # 4:3 
    "axes.grid": True,                  # 
    "grid.alpha": 0.3,                  # 
    "grid.linestyle": "--",             # 
})

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
    print(":  peft : pip install peft")
    print(":", e)
    PEFT_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print("\n" + "=" * 70)
print("  FDA  α ")
print("=" * 70)

try:
    from fdt_init import apply_fdt_to_lora
    FDT_INIT_AVAILABLE = True
    print("[]  fdt_init.apply_fdt_to_lora")
except ImportError as e:
    FDT_INIT_AVAILABLE = False
    apply_fdt_to_lora = None
    print("[]  fdt_init :", e)

try:
    #  measure_alpha.py
    from measure_alpha import measure_alpha
    MEASURE_ALPHA_AVAILABLE = True
    print("[]  measure_alpha.measure_alpha")
except ImportError as e:
    MEASURE_ALPHA_AVAILABLE = False
    measure_alpha = None
    print("[]  measure_alpha :", e)

print("=" * 70 + "\n")


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

    #  lora_A.weight lora_*.weight
    first_any = None
    for name, _ in model.named_parameters():
        if "lora_" in name and name.endswith("weight"):
            if first_any is None:
                first_any = name
            if "lora_A" in name:
                return name

    if first_any is None:
        raise RuntimeError(" LoRA  LoRA")
    return first_any


def _get_param_by_name(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    params = dict(model.named_parameters())
    if name not in params:
        available = [k for k in params.keys() if "lora_" in k and k.endswith("weight")]
        raise KeyError(
            f": {name}\n"
            f" LoRA ( 10 ): {available[:10]}"
        )
    return params[name]


def spectrum_and_alpha_from_tensor(tensor: torch.Tensor):
    """ measure_alpha.py (alpha, freqs, power)"""
    if not MEASURE_ALPHA_AVAILABLE or measure_alpha is None:
        raise RuntimeError("measure_alpha  measure_alpha.py ")
    alpha, freqs, power = measure_alpha(tensor, method="fft", return_full_spectrum=True)
    return float(alpha), freqs.astype(np.float64), power.astype(np.float64)


def _draw_spectrum_curve(ax, freqs, power, alpha, label, color, style="line", linewidth=1.0, alpha_val=1.0):
    """
    
    style: 'line' (, marker), 'marker' (), 'both'
    """
    # 1. 
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

    # 2.  ()
    if np.isfinite(alpha):
        #  C C*f^{-alpha} 
        ref_idx = int(len(freqs) * 0.15) # 
        ref_idx = max(0, min(ref_idx, len(freqs) - 1))
        
        # 
        C = power[ref_idx] * (freqs[ref_idx] ** alpha)
        fit_line = C * (freqs ** (-alpha))
        
        # 
        ax.loglog(freqs, fit_line, "--", linewidth=1.5, alpha=0.8, color=color)


def _tensor_to_2d_numpy(
    tensor: torch.Tensor,
    max_rows: int = 512,
    max_cols: int = 512,
) -> np.ndarray:
    """ tensor  2D numpy imshow

    - 2D: 
    - 1D:  (1, N)
    - >=3D:  (dim0, -1)
     (<=max_rows, <=max_cols)
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
    """ LoRA 

    
    -  set_box_aspect(1) 1:1
    - vmin/vmax  Before/After 
    """
    #  colorbar axes 
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


def get_args():
    ap = argparse.ArgumentParser(description="OpenPangu FDA  + ")

    #  LoRA
    ap.add_argument(
        "--model_path",
        type=str,
        default="/opt/pangu/openPangu-Embedded-7B-V1.1",
        help="",
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

    # FDT 
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
        help=": baseline/soft/medium/strong",
    )

    ap.add_argument(
        "--layer_name",
        type=str,
        default=None,
        help=" LoRA  lora_A.weight",
    )

    # LoRA Before/After/Diff
    ap.add_argument(
        "--viz_lora_matrix",
        action="store_true",
        help=" LoRA //",
    )
    ap.add_argument(
        "--viz_max_rows",
        type=int,
        default=512,
        help="",
    )
    ap.add_argument(
        "--viz_max_cols",
        type=int,
        default=512,
        help="",
    )

    # 
    ap.add_argument("--device", type=str, default="npu:0")
    ap.add_argument("--seed", type=int, default=1107)
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs_spectrum_init",
        help="outputs_spectrum_init",
    )
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()


def main():
    if not PEFT_AVAILABLE:
        print(": PEFT  LoRA")
        return

    if not MEASURE_ALPHA_AVAILABLE:
        print(": measure_alpha.py ")
        return

    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 
    config_file = os.path.join(args.out_dir, "config_init_only.json")
    with open(config_file, "w") as f:
        json.dump(convert_to_json_serializable(vars(args)), f, indent=2)
    print(f"[] : {config_file}")

    # 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 
    device = torch.device(args.device)
    device_type = args.device.split(":")[0]

    if device_type == "npu":
        try:
            import torch_npu
            torch_npu.npu.set_device(device)
            torch_npu.npu.manual_seed_all(args.seed)
            print(f"[]  NPU : {device}")
        except Exception as e:
            print(f"[]  NPU : {e}")
            return
    else:
        print(f"[] : {device}")

    print("=" * 70)
    print("  1: ")
    print("=" * 70)

    print(f"[] : {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    total_params, _ = count_parameters(model)
    print(f"[]  : {total_params/1e9:.2f}B \n")

    print("=" * 70)
    print("  2:  LoRA")
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
        f"[LoRA]  : {trainable_params:,} "
        f"({trainable_params/total_params*100:.4f}%)\n"
    )

    chosen_name = _select_lora_param_name(model, args.layer_name)
    chosen_param = _get_param_by_name(model, chosen_name)
    print(f"[] : {chosen_name}")

    before_tensor = chosen_param.detach().clone().float().cpu()
    alpha_before, freqs_before, power_before = spectrum_and_alpha_from_tensor(before_tensor)
    print(f"[alpha] ( measure_alpha ): {alpha_before:.3f}")

    print("=" * 70)
    print("  3: FDA ")
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
        print(f"[] : {cfg['name']}")
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
            print("[FDA]   fdt_init  FDA ")
        else:
            print(
                f"[FDA]  FDA : alpha={args.fdt_alpha:.2f}, "
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
            print("[FDA]  ")
    else:
        print("[FDA]  PEFT  (Kaiming Uniform + Zero)")

    init_info_file = os.path.join(args.out_dir, "init_info_only.json")
    after_tensor = chosen_param.detach().clone().float().cpu()
    alpha_after, freqs_after, power_after = spectrum_and_alpha_from_tensor(after_tensor)
    print(f"[alpha] ( measure_alpha ): {alpha_after:.3f}")

    if getattr(args, "viz_lora_matrix", False):
        print("=" * 70)
        print(" : LoRA  (before/after )")
        print("=" * 70)
        heat_pdf_before = os.path.join(args.out_dir, "lora_weight_heatmap_before.pdf")
        heat_png_before = os.path.join(args.out_dir, "lora_weight_heatmap_before.png")
        heat_pdf_after = os.path.join(args.out_dir, "lora_weight_heatmap_after.pdf")
        heat_png_after = os.path.join(args.out_dir, "lora_weight_heatmap_after.png")

        w_before = _tensor_to_2d_numpy(before_tensor, max_rows=args.viz_max_rows, max_cols=args.viz_max_cols)
        w_after = _tensor_to_2d_numpy(after_tensor, max_rows=args.viz_max_rows, max_cols=args.viz_max_cols)

        if w_before.shape != w_after.shape:
            raise RuntimeError(f"before/after : {w_before.shape} vs {w_after.shape}")

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

        # 
        diff = w_after - w_before
        print(f"[heatmap]  : {heat_pdf_before}")
        print(f"[heatmap]  : {heat_png_before}")
        print(f"[heatmap]  : {heat_pdf_after}")
        print(f"[heatmap]  : {heat_png_after}")
        print(
            "[heatmap] : "
            f"before(std={float(np.std(w_before)):.4g}), "
            f"after(std={float(np.std(w_after)):.4g}), "
            f"diff(std={float(np.std(diff)):.4g}, max_abs={float(np.max(np.abs(diff))):.4g})"
        )

    #  α
    init_info["measured"] = {
        "layer_name": chosen_name,
        "alpha_before": float(alpha_before),
        "alpha_after": float(alpha_after),
    }
    with open(init_info_file, "w") as f:
        json.dump(convert_to_json_serializable(init_info), f, indent=2)
    print(f"[FDA] : {init_info_file}\n")

    print("=" * 70)
    print("  4:  (log-log)")
    print("=" * 70)

    # 
    common_freqs = np.intersect1d(freqs_before, freqs_after)
    if common_freqs.size < 10:
        raise RuntimeError(
            f": before={len(freqs_before)}, after={len(freqs_after)}"
        )

    before_map = {int(f): i for i, f in enumerate(freqs_before.astype(np.int64))}
    after_map = {int(f): i for i, f in enumerate(freqs_after.astype(np.int64))}
    idx_b = np.array([before_map[int(f)] for f in common_freqs], dtype=np.int64)
    idx_a = np.array([after_map[int(f)] for f in common_freqs], dtype=np.int64)

    freqs = common_freqs.astype(np.float64)
    p_before = power_before[idx_b]
    p_after = power_after[idx_a]

    # 
    fig_path_pdf = os.path.join(args.out_dir, "lora_power_spectrum.pdf") # 
    fig_path_png = os.path.join(args.out_dir, "lora_power_spectrum.png") # 
    npz_path = os.path.join(args.out_dir, "lora_power_spectrum.npz")

    # 
    fig, ax = plt.subplots()

    # 4.1  Before ()
    #  marker
    _draw_spectrum_curve(
        ax, freqs, p_before, alpha_before,
        label=r"Baseline",
        color="#3b75af",  # Steel Blue 
        style="line",     # 
        linewidth=0.8,
        alpha_val=0.6     # 
    )

    # 4.2  After ()
    # ()
    after_label_text = r"FDA($\alpha=%.2f$)" % args.fdt_alpha if args.use_fdt_init else r"After"
    _draw_spectrum_curve(
        ax, freqs, p_after, alpha_after,
        label=f"{after_label_text}",
        color="#d62728",  # 
        style="line",
        linewidth=2.0,    # 
        alpha_val=1.0
    )

    # 4.3 
    ax.set_xlabel(r"Frequency Index")
    ax.set_ylabel(r"Power Spectral Density (PSD)")
    
    # 
    #  text 
    # ax.set_title("Power Spectrum of LoRA Weights", fontweight='bold')
    
    # 
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    
    # # 
    # plt.figtext(0.5, 0.01, f"Layer: {chosen_name}", wrap=True, horizontalalignment='center', fontsize=8, color='gray')

    plt.tight_layout()
    #  figtext
    plt.subplots_adjust(bottom=0.15)

    # 
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

    print(f"[]  : {fig_path_pdf}")
    print(f"[]  : {fig_path_png}")
    print(f"[]  : {npz_path}")
    print(f"[] α( -> ): {alpha_before:.3f} -> {alpha_after:.3f}")
    print("\n  + /")


if __name__ == "__main__":
    main()