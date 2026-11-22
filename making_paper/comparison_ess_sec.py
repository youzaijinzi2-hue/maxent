import numpy as np
import matplotlib.pyplot as plt

npz_path = "data/use_entropy.npz"  # パス適宜
z = np.load(npz_path, allow_pickle=True)

# --- Values ---
mh_ess_per_sec = 0.91          # keep as-is (user request)
proposed_ess_per_sec = float(np.array(z["ess_per_sec"]).reshape(()))
ratio = proposed_ess_per_sec / mh_ess_per_sec
mh_note = f"× {ratio:.1f}"             # keep as-is (user request)

labels = ["MH", "Proposed"]
values = [mh_ess_per_sec, proposed_ess_per_sec]

# --- Plot ---
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
# Figure & Axes background
fig.patch.set_facecolor("#F7F7F7")
ax.set_facecolor("#F7F7F7")
# Match the example-like palette
colors = ["#4C72B0", "#DD8452"]
bars = ax.bar(labels, values, color=colors, width=0.6)

ax.set_yscale("log")
ax.set_ylabel(r"ESS/sec ($\log_{10}$ scale)")
ax.grid(True, which="both", axis="y", alpha=0.25)

# y-limits for annotation space
ymin = min(values) / 3
ymax = max(values) * 2.2
ax.set_ylim(max(ymin, 1e-3), ymax)

# --- Annotations ---
# MH: fixed text
mh_y = values[0]
ax.text(
    bars[0].get_x() + bars[0].get_width() / 2,
    mh_y * 1.35,
    f"{mh_ess_per_sec:.2f} ESS/sec\n{mh_note}",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
)

# Proposed: from npz
prop_y = values[1]
ax.text(
    bars[1].get_x() + bars[1].get_width() / 2,
    prop_y * 1.10,
    f"{proposed_ess_per_sec:.1f} ESS/sec",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
)

# Title under the plot (like sample)
fig.subplots_adjust(bottom=0.22)
fig.text(0.5, 0.05, "Comparison of ESS/sec Across Methods", ha="center", fontsize=13)

plt.savefig("./data/ess_per_sec_comparison_no_hmc.svg", bbox_inches="tight")
plt.show()
