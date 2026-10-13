"""Figure 1 (framework schematic) and Figure 2 (example simulated tissue)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget.simulate import simulate_tissue, CELL_TYPES  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(out, exist_ok=True)

# ---------------- Figure 1: framework -----------------
fig, ax = plt.subplots(figsize=(12, 3.0))
ax.set_xlim(0, 12)
ax.set_ylim(0.3, 3.1)
ax.axis("off")
boxes = [
    (0.2, "1  Inputs", "Spatial expression\nCell coordinates\nCell states / types\nLR pairs (CellChatDB,\nOmniPath)\nNormal reference"),
    (2.6, "2  Niche-aware graph", "k-NN cell graph\nEdge: distance + LR\nco-expression $x_{j,l}\\,x_{i,r}$\nNiche descriptor $\\nu_i$\n(neighbourhood\ncomposition)"),
    (5.0, "3  Encoder", "Edge-aware graph attention\nFiLM conditioning on $\\nu_i$\nRead-out genes never input\nMasked-expression training"),
    (7.4, "4  Counterfactual", "Inhibit gene $g$ ($x_g \\to 0$)\nRecompute LR edges\nPredict read-out shift\nEfficacy per niche $E_g^{(n)}$"),
    (9.8, "5  Ranking", "Niche-max efficacy\n× selectivity $\\sigma_g$\n→ ranked, niche-specific,\nselective targets\n→ drugs (ChEMBL/OT)"),
]
for x, title, body in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.45), 2.0, 2.55, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="#eef3fb", ec="#2a6fdb", lw=1.4))
    ax.text(x + 1.0, 2.72, title, ha="center", va="center", fontsize=10.5, weight="bold", color="#1b3f80")
    ax.text(x + 1.0, 1.55, body, ha="center", va="center", fontsize=7.8, linespacing=1.5)
for x in [2.2, 4.6, 7.0, 9.4]:
    ax.add_patch(FancyArrowPatch((x + 0.02, 1.6), (x + 0.38, 1.6), arrowstyle="-|>", mutation_scale=16, color="#1b3f80"))
fig.savefig(os.path.join(out, "Fig1_framework.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: example simulated tissue -----------------
sim = simulate_tissue(seed=1)
colors = ["#d1495b", "#edae49", "#00798c", "#30638e", "#8d8d8d"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for t, ct in enumerate(CELL_TYPES):
    m = sim.cell_type == t
    axes[0].scatter(sim.coords[m, 0], sim.coords[m, 1], s=3, c=colors[t], label=ct)
axes[0].legend(markerscale=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
axes[0].set_title("A  Cell types", loc="left", fontsize=10)
names = {0: "background", 1: "myeloid", 2: "stromal", 3: "immune"}
for n in range(4):
    m = sim.niche == n
    axes[1].scatter(sim.coords[m, 0], sim.coords[m, 1], s=3, label=names[n])
axes[1].legend(markerscale=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
axes[1].set_title("B  Planted niches", loc="left", fontsize=10)
mal = sim.cell_type == 0
sc = axes[2].scatter(sim.coords[mal, 0], sim.coords[mal, 1], s=3, c=sim.aggressiveness[mal], cmap="magma")
plt.colorbar(sc, ax=axes[2], fraction=0.046, label="aggressiveness $a_i$")
axes[2].set_title("C  Malignant-cell state", loc="left", fontsize=10)
for a in axes:
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(out, "Fig2_simulated_tissue.png"), dpi=300)
plt.close(fig)
print("ok")
