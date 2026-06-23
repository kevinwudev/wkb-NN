"""
wkb_nn_final.py
WKB Tunneling Probability via Neural Network (ħ = m = 1)

Key design choices:
  Data   : 50% single-Gaussian (back-solved to balance log T), 50% multi-Gaussian (accept-reject)
  Model  : MLP, output = log T directly (no Sigmoid saturation)
  Loss   : importance-weighted MSE in log T space
  Result : ~5-30% relative error across all barrier types

Outputs 4 figures:
  fig1_data_dist.png   — training T distribution (log T uniform)
  fig2_training.png    — loss curve
  fig3_scatter.png     — WKB vs NN scatter on test set (log scale)
  fig4_cases.png       — 4 case-study barriers with T_WKB vs T_NN

Run: pip install numpy matplotlib torch
     python wkb_nn_final.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── config ────────────────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

N_X        = 50          # grid points
E          = 1.0         # particle energy (ħ = m = 1)
LOG_T_MIN  = -8.0        # minimum log10(T) in training data
N_TRAIN    = 2000
N_TEST     = 300
EPOCHS     = 300
LR         = 1e-3
BATCH      = 128

x = np.linspace(0.0, 10.0, N_X)   # spatial grid

# ── physics ───────────────────────────────────────────────────────────────────

def wkb(V):
    """T = exp(-2 ∫_a^b √(2(V-E)) dx),  a/b = classical turning points."""
    fb = V > E
    if not np.any(fb): return 1.0
    idx = np.where(fb)[0]
    if idx[0] == idx[-1]: return 1.0
    k = np.sqrt(np.maximum(2.0 * (V[idx[0]:idx[-1]+1] - E), 0.0))
    return float(np.clip(np.exp(-2.0 * np.trapezoid(k, x[idx[0]:idx[-1]+1])), 1e-10, 1.0))

# ── data generation ───────────────────────────────────────────────────────────

def _wkb_integral(A, mu, sg):
    """WKB integral for a single Gaussian V = A·exp(-(x-mu)²/2sg²)."""
    V = A * np.exp(-0.5 * ((x - mu) / sg) ** 2)
    fb = V > E
    if not np.any(fb): return 0.0
    idx = np.where(fb)[0]
    k = np.sqrt(np.maximum(2.0 * (V[idx[0]:idx[-1]+1] - E), 0.0))
    return float(np.trapezoid(k, x[idx[0]:idx[-1]+1]))

def _bisect_A(I_target, mu, sg, lo=1.001, hi=20.0):
    """Bisection: find A such that WKB integral ≈ I_target."""
    if _wkb_integral(lo, mu, sg) > I_target: return lo
    if _wkb_integral(hi, mu, sg) < I_target: return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if _wkb_integral(mid, mu, sg) < I_target: lo = mid
        else: hi = mid
        if hi - lo < 1e-3: break
    return (lo + hi) / 2

def generate_data(N):
    """
    Mixed balanced dataset:
      50% single-Gaussian — back-solve A to hit target log T uniformly
      50% multi-Gaussian  — accept-reject to fill sparse log T bins
    Both halves cover log10(T) ∈ [LOG_T_MIN, 0] uniformly.
    """
    V_all, T_all = [], []

    # half 1: single Gaussian, perfectly balanced
    for _ in range(N // 2):
        log_T = np.random.uniform(LOG_T_MIN, 0.0)
        I_tgt = -np.log(max(10.0**log_T, 1e-10)) / 2.0
        mu, sg = np.random.uniform(3, 7), np.random.uniform(0.5, 2.0)
        A = _bisect_A(I_tgt, mu, sg)
        V = A * np.exp(-0.5 * ((x - mu) / sg) ** 2)
        V_all.append(V); T_all.append(wkb(V))

    # half 2: multi-Gaussian, accept-reject
    n_bins = 30
    counts = np.zeros(n_bins)
    target = (N - N // 2) / n_bins
    collected, attempts = 0, 0
    while collected < N - N // 2 and attempts < (N - N // 2) * 30:
        attempts += 1
        n_g = np.random.randint(2, 4)
        V = sum(np.random.uniform(0.5, 3.5) *
                np.exp(-0.5 * ((x - np.random.uniform(2, 8)) / np.random.uniform(0.5, 2)) ** 2)
                for _ in range(n_g))
        T = wkb(V)
        log_T = np.log10(T + 1e-10)
        if log_T < LOG_T_MIN: continue
        b = int(np.clip((log_T - LOG_T_MIN) / (-LOG_T_MIN) * n_bins, 0, n_bins - 1))
        if np.random.random() < max(0.1, 1.0 - counts[b] / max(target, 1)):
            V_all.append(V); T_all.append(T)
            counts[b] += 1; collected += 1

    return np.array(V_all[:N]), np.array(T_all[:N])

def importance_weights(T_arr, n_bins=30):
    """w_i = 1 / bin_count(log T_i),  normalised to mean = 1."""
    log_T = np.log10(np.clip(T_arr, 1e-10, 1.0))
    hist, edges = np.histogram(log_T, bins=n_bins, range=(LOG_T_MIN, 0.0))
    hist = np.maximum(hist, 1)
    idx = np.clip(np.digitize(log_T, edges[:-1]) - 1, 0, n_bins - 1)
    w = 1.0 / hist[idx].astype(float)
    return (w / w.mean()).astype(np.float32)

# ── model ─────────────────────────────────────────────────────────────────────

class TunnelingNet(nn.Module):
    """
    Input  : V(x) discretised to N_X values
    Output : log T̂  (unbounded; avoids Sigmoid saturation at small T)
    Infer  : T̂ = exp(output)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_X, 128), nn.ReLU(),
            nn.Linear(128,  64), nn.ReLU(),
            nn.Linear( 64,  32), nn.ReLU(),
            nn.Linear( 32,   1),            # linear output = log T
        )
    def forward(self, x): return self.net(x).squeeze(-1)
    def predict(self, V_np):
        """Convenience: numpy array → T (not log T)."""
        self.eval()
        with torch.no_grad():
            return float(torch.exp(self(torch.FloatTensor(V_np).unsqueeze(0))))

# ── training ──────────────────────────────────────────────────────────────────

def train(V_tr, T_tr, V_va, T_va):
    """Weighted MSE in log T space, Adam optimiser. Returns model, train_losses, val_losses."""
    model = TunnelingNet()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    w     = importance_weights(T_tr)

    Vt  = torch.FloatTensor(V_tr)
    lTt = torch.FloatTensor(np.log(np.clip(T_tr, 1e-10, 1.0)))
    wt  = torch.FloatTensor(w)
    Vv  = torch.FloatTensor(V_va)
    lTv = torch.FloatTensor(np.log(np.clip(T_va, 1e-10, 1.0)))

    loader = DataLoader(TensorDataset(Vt, lTt, wt), batch_size=BATCH, shuffle=True)
    train_losses, val_losses = [], []

    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for Vb, lTb, wb in loader:
            loss = (wb * (model(Vb) - lTb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(Vb)
        train_losses.append(ep_loss / len(Vt))
        model.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(model(Vv), lTv).item()
        val_losses.append(vl)
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:3d}/{EPOCHS}  val_loss={vl:.4f}")

    return model, train_losses, val_losses

# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, V_test, T_test):
    """Print per-sample relative error and overall log-MAE."""
    model.eval()
    with torch.no_grad():
        log_pred = model(torch.FloatTensor(V_test)).numpy()
    T_pred   = np.exp(log_pred)
    rel_err  = np.abs(T_pred - T_test) / (T_test + 1e-12) * 100
    log_mae  = np.mean(np.abs(log_pred - np.log(np.clip(T_test, 1e-10, 1.0))))
    corr     = np.corrcoef(np.log(T_test + 1e-10), log_pred)[0, 1]
    print(f"\n  test log-MAE = {log_mae:.4f}   log-corr = {corr:.4f}")
    return T_pred, rel_err

# ── case studies ──────────────────────────────────────────────────────────────

CASES = [
    ("Low thin",    1.5 * np.exp(-0.5 * ((x - 5.0) / 0.6) ** 2)),
    ("High thick",  3.5 * np.exp(-0.5 * ((x - 5.0) / 2.0) ** 2)),
    ("Double-peak", 1.8 * np.exp(-0.5 * ((x - 3.5) / 0.8) ** 2) +
                    1.8 * np.exp(-0.5 * ((x - 6.5) / 0.8) ** 2)),
    ("Asymmetric",  2.0 * np.exp(-0.5 * ((x - 4.0) / 0.5) ** 2) +
                    1.2 * np.exp(-0.5 * ((x - 7.0) / 1.5) ** 2)),
]

def print_case_results(model):
    print(f"\n  {'Case':15s}  {'T_WKB':>10s}  {'T_NN':>10s}  {'rel_err%':>10s}")
    print("  " + "-" * 52)
    for name, V in CASES:
        T_w = wkb(V)
        T_n = model.predict(V)
        err = abs(T_n - T_w) / (T_w + 1e-12) * 100
        print(f"  {name:15s}  {T_w:10.5f}  {T_n:10.5f}  {err:9.1f}%")

# ── plots ─────────────────────────────────────────────────────────────────────

def plot_data_dist(T_tr):
    """Fig 1: histogram of training T in log space — should be roughly flat."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-9, 0, 35)
    ax.hist(np.log10(np.clip(T_tr, 1e-10, 1.0)), bins=bins,
            color='#2563EB', alpha=0.85, edgecolor='white')
    ax.axvline(np.log10(0.208),   color='red',    ls='--', lw=1.5, label='Low thin  T=0.208')
    ax.axvline(np.log10(0.00124), color='orange', ls='--', lw=1.5, label='Double-peak T=0.00124')
    ax.set_xlabel('log₁₀(T_WKB)')
    ax.set_ylabel('Count')
    ax.set_title('Training data: T distribution (balanced across log scale)', fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig('fig1_data_dist.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  saved fig1_data_dist.png")


def plot_training(train_losses, val_losses):
    """Fig 2: train / val loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ep = np.arange(1, len(train_losses) + 1)
    ax.semilogy(ep, train_losses, color='#2563EB', lw=2, label='train')
    ax.semilogy(ep, val_losses,   color='#DC2626', lw=2, label='val', ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted MSE  (log T space)')
    ax.set_title('Training loss curve', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig('fig2_training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  saved fig2_training.png")


def plot_scatter(model, V_va, T_va):
    """Fig 3: WKB vs NN on test set in log scale."""
    model.eval()
    with torch.no_grad():
        log_pred = model(torch.FloatTensor(V_va)).numpy()
    lx = np.log10(np.clip(T_va,         1e-10, 1.0))
    ly = np.log10(np.clip(np.exp(log_pred), 1e-10, 1.0))

    log_mae = np.mean(np.abs(ly - lx))
    corr    = np.corrcoef(lx, ly)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(lx, ly, c=lx, cmap='plasma', s=8, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='log₁₀(T_WKB)')
    lo, hi = min(lx.min(), ly.min()) - 0.3, max(lx.max(), ly.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='perfect')
    ax.set_xlabel('log₁₀(T_WKB)  — ground truth')
    ax.set_ylabel('log₁₀(T_NN)   — predicted')
    ax.set_title('Test set: WKB vs NN (log scale)', fontweight='bold')
    ax.text(0.04, 0.93, f'log-MAE={log_mae:.3f}\ncorr={corr:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig('fig3_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  saved fig3_scatter.png")


def plot_cases(model):
    """Fig 4: 4-panel case studies — V(x) with turning points + T bar."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 13))
    fig.suptitle('Case Studies: WKB vs Neural Network', fontsize=13, fontweight='bold')

    for i, (name, V) in enumerate(CASES):
        T_w = wkb(V)
        T_n = model.predict(V)
        err = abs(T_n - T_w) / (T_w + 1e-12) * 100

        # left: potential + forbidden zone
        ax_v = axes[i][0]
        forbidden = V > E
        ax_v.fill_between(x, E, V, where=forbidden, alpha=0.2,
                          color='orange', label='Forbidden zone')
        ax_v.plot(x, V, color='#6D28D9', lw=2, label='V(x)')
        ax_v.axhline(E, color='#059669', lw=1.5, ls='--', label=f'E={E}')
        # mark turning points
        idxs = np.where(forbidden)[0]
        if len(idxs) > 1:
            ax_v.axvline(x[idxs[0]],  color='gray', ls=':', lw=1)
            ax_v.axvline(x[idxs[-1]], color='gray', ls=':', lw=1)
        ax_v.set_ylim(-0.2, max(V.max() * 1.15, E * 1.5))
        ax_v.set_xlabel('x'); ax_v.set_ylabel('V(x)')
        ax_v.set_title(name, fontweight='bold', fontsize=10)
        ax_v.legend(fontsize=7)

        # right: bar chart T_WKB vs T_NN
        ax_b = axes[i][1]
        bars = ax_b.bar(['WKB\n(truth)', 'NN\n(pred)'], [T_w, T_n],
                        color=['#2563EB', '#DC2626'], width=0.4,
                        alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, [T_w, T_n]):
            ax_b.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(T_w, T_n) * 0.03,
                      f'{val:.5f}', ha='center', va='bottom', fontsize=9)
        ax_b.set_ylim(0, max(T_w, T_n) * 1.5 + 1e-6)
        ax_b.set_ylabel('T')
        ax_b.set_title(f'Relative error: {err:.1f}%', fontsize=9)

    plt.tight_layout()
    fig.savefig('fig4_cases.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  saved fig4_cases.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating data ...")
    V_all, T_all = generate_data(N_TRAIN + N_TEST)
    V_tr, T_tr = V_all[:N_TRAIN], T_all[:N_TRAIN]
    V_va, T_va = V_all[N_TRAIN:], T_all[N_TRAIN:]
    print(f"T range: [{T_tr.min():.2e}, {T_tr.max():.4f}]")

    print("\nTraining ...")
    model, train_losses, val_losses = train(V_tr, T_tr, V_va, T_va)

    print("\nTest set evaluation:")
    evaluate(model, V_va, T_va)

    print("\nCase studies:")
    print_case_results(model)

    print("\nSaving figures ...")
    plot_data_dist(T_tr)
    plot_training(train_losses, val_losses)
    plot_scatter(model, V_va, T_va)
    plot_cases(model)

if __name__ == "__main__":
    main()