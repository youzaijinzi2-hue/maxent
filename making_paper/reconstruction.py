import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Load data
# =========================
no  = np.load("data/no_entropy.npz", allow_pickle=True)
use = np.load("data/use_entropy.npz", allow_pickle=True)
inp = np.load("data/inputs.npz", allow_pickle=True)

omega = np.array(use["omega"], dtype=float)

A_use = np.array(use["A_est"], dtype=float)
A_use_low  = np.array(use["A_low"],  dtype=float)
A_use_high = np.array(use["A_high"], dtype=float)

A_no = np.array(no["A_est"], dtype=float)
A_no_low  = np.array(no["A_low"],  dtype=float)
A_no_high = np.array(no["A_high"], dtype=float)

# Input / Ground truth spectrum
D_w = np.array(inp["D_w"], dtype=float)

# =========================
# Color palette
# =========================
blue   = "#1f77b4"   # with entropy
orange = "#ff7f0e"   # no entropy
black  = "#111111"   # input / ground truth

# =========================
# Plot
# =========================
fig, ax1 = plt.subplots(figsize=(9, 5))
# Figure & Axes background
fig.patch.set_facecolor("#F7F7F7")

# --- Left axis: Input + With entropy ---
ax1.plot(omega, D_w, color=black, lw=3.0, label="True spectrum")
ax1.plot(omega, A_use, color=blue, lw=2.5, label="With entropy")
ax1.fill_between(
    omega, A_use_low, A_use_high,
    color=blue, alpha=0.18
)

ax1.set_xlabel(r"$\omega$ [eV]")
ax1.set_ylabel("Intensity (input / with entropy) [arb. units]", color=blue)
ax1.tick_params(axis="y", labelcolor=blue)

# --- Right axis: No entropy ---
ax2 = ax1.twinx()
ax2.plot(omega, A_no, color=orange, lw=2.5, ls="--", label="No entropy")
ax2.fill_between(
    omega, A_no_low, A_no_high,
    color=orange, alpha=0.18
)
ax2.set_ylabel("Intensity (no entropy) [arb. units]", color=orange)
ax2.tick_params(axis="y", labelcolor=orange)

ax1.set_facecolor("#F7F7F7")
ax2.set_facecolor("#F7F7F7")  # twin axis

for spine in ax1.spines.values():
    spine.set_alpha(0.6)
for spine in ax2.spines.values():
    spine.set_alpha(0.6)
    
# --- Legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
    frameon=False
)

# fig.subplots_adjust(bottom=0.5)
title = "Unregularized Reconstruction Distorts XAS $K_2 Cr_2 O_7$ Spectrum"
plt.title(title, y=-0.20, fontsize=14)
fig.tight_layout()

plt.savefig("./data/reconstruction.svg")
# plt.show()
