import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def generate_noise(n, alpha):
    """ alpha """
    freqs = np.fft.rfftfreq(n)[1:]
    amp = freqs ** (-alpha / 2.0)
    phase = np.random.uniform(0, 2*np.pi, len(freqs))
    z = amp * np.exp(1j * phase)
    z = np.insert(z, 0, 0)
    noise = np.fft.irfft(z, n=n)
    return (noise - noise.mean()) / noise.std()

def plot_enhanced_figure():
    r, d_in = 32, 256  # 
    N = r * d_in
    
    # 1. 
    # Baseline:  (alpha=0)
    w_white = np.random.randn(N)
    
    # FDA:  alpha  1.5""
    #  1.1 ""
    w_pink = generate_noise(N, alpha=1.5) 
    
    # 2.  ()
    fig = plt.figure(figsize=(12, 4), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2])
    
    # === Panel A:  (Spectral Domain) ===
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
    #  seismic  0 
    im2 = ax2.imshow(w_white.reshape(r, d_in), cmap='gray', aspect='auto')
    ax2.set_title(r"Baseline ($\alpha \approx 0$)", fontsize=10, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Input Dim")
    ax2.set_ylabel("Rank")
    # 
    ax2.text(d_in/2, r/2, "Disordered\n(White Noise)", ha='center', va='center', 
             color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    # === Panel C: FDA Matrix ===
    ax3 = fig.add_subplot(gs[2])
    #  Coolwarm  RdBu
    im3 = ax3.imshow(w_pink.reshape(r, d_in), cmap='RdBu_r', aspect='auto')
    ax3.set_title(r"FDA-SOC ($\alpha > 0$)", fontsize=10, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel("Input Dim")
    # 
    ax3.text(d_in/2, r/2, "Structured\n(Correlated)", ha='center', va='center', 
             color='#004488', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("fda_enhanced_overview.png")
    plt.show()

if __name__ == "__main__":
    plot_enhanced_figure()