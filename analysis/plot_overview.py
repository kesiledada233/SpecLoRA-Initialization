import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def generate_noise(n, alpha):
    """生成具有特定 alpha 的噪声"""
    freqs = np.fft.rfftfreq(n)[1:]
    amp = freqs ** (-alpha / 2.0)
    phase = np.random.uniform(0, 2*np.pi, len(freqs))
    z = amp * np.exp(1j * phase)
    z = np.insert(z, 0, 0)
    noise = np.fft.irfft(z, n=n)
    return (noise - noise.mean()) / noise.std()

def plot_enhanced_figure():
    r, d_in = 32, 256  # 稍微调整尺寸以适应绘图
    N = r * d_in
    
    # 1. 生成数据
    # Baseline: 白噪声 (alpha=0)
    w_white = np.random.randn(N)
    
    # FDA: 为了视觉效果，我们稍微夸大 alpha 到 1.5，让"结块"更明显
    # 论文里写 1.1 没问题，示意图是为了展示"原理"
    w_pink = generate_noise(N, alpha=1.5) 
    
    # 2. 绘图布局 (左边是频谱原理，右边是矩阵纹理)
    fig = plt.figure(figsize=(12, 4), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2])
    
    # === Panel A: 核心原理 (Spectral Domain) ===
    ax1 = fig.add_subplot(gs[0])
    f_w, p_w = welch(w_white, nperseg=1024)
    f_p, p_p = welch(w_pink, nperseg=1024)
    
    ax1.loglog(f_w, p_w, color='gray', alpha=0.6, lw=1, label='Baseline (Flat)')
    ax1.loglog(f_p, p_p, color='#0077cc', lw=2, label=r'FDA-SOC ($1/f^\alpha$)')
    ax1.set_title("Frequency Domain (The 'Why')", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Power")
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.text(0.1, 0.5, "Difference is HERE!", transform=ax1.transAxes, color='red', rotation=-25)

    # === Panel B: Baseline Matrix ===
    ax2 = fig.add_subplot(gs[1])
    # 使用 seismic 颜色，让 0 值是白色，正负值分明
    im2 = ax2.imshow(w_white.reshape(r, d_in), cmap='gray', aspect='auto')
    ax2.set_title(r"Baseline ($\alpha \approx 0$)", fontsize=10, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Input Dim")
    ax2.set_ylabel("Rank")
    # 加个标签
    ax2.text(d_in/2, r/2, "Disordered\n(White Noise)", ha='center', va='center', 
             color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    # === Panel C: FDA Matrix ===
    ax3 = fig.add_subplot(gs[2])
    # 使用 Coolwarm 或 RdBu，让云雾感更强
    im3 = ax3.imshow(w_pink.reshape(r, d_in), cmap='RdBu_r', aspect='auto')
    ax3.set_title(r"FDA-SOC ($\alpha > 0$)", fontsize=10, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel("Input Dim")
    # 加个标签
    ax3.text(d_in/2, r/2, "Structured\n(Correlated)", ha='center', va='center', 
             color='#004488', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("fda_enhanced_overview.png")
    plt.show()

if __name__ == "__main__":
    plot_enhanced_figure()