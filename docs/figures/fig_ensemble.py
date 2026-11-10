"""Two ways to ensemble three agents, and why only one is allowed (one ToolEnv per sample, every call counts).
figures4papers style, Track 1 deck v2. Run: python docs/figures/fig_ensemble.py -> ensemble.{png,pdf,svg}"""
from matplotlib.patches import FancyArrowPatch

from style import C, arrow, box, gbox, mix, rich, save, setup

W, H = 16.0, 6.0
fig, ax = setup(W, H)
B, G, I = {"weight": "bold"}, {"color": C["grey"]}, {"style": "italic"}
M = {"family": "Liberation Mono"}
GAP = 0.1
AMBER, AMBER_F, AMBER_T = "#B07A12", "#FBE3B6", "#7A520A"


def agent(x, y, name, w=1.55, h=0.5):
    gbox(ax, x, y, w, h, C["p_fill"], C["p_edge"], lw=1.3, r=0.12)
    ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=11.5, fontweight="bold", color=C["p_text"], zorder=5)


def env(x, y, w=1.55, h=0.5, label="ToolEnv"):
    gbox(ax, x, y, w, h, AMBER_F, AMBER, lw=1.3, r=0.12)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11.5, fontweight="bold", color=AMBER_T, zorder=5)


def cross(x, y, s=0.16):
    ax.plot([x - s, x + s], [y - s, y + s], color=C["r_edge"], lw=3, solid_capstyle="round", zorder=6)
    ax.plot([x - s, x + s], [y + s, y - s], color=C["r_edge"], lw=3, solid_capstyle="round", zorder=6)


def tick(x, y, s=0.16, col=None):
    ax.plot([x - s, x - s * 0.3, x + s], [y, y - s * 0.7, y + s * 0.8], color=col or C["g_edge"], lw=3, solid_capstyle="round", zorder=6)


# ---- panel a: vote on the final state -------------------------------------------------------------------------------------------------
box(ax, 0.2, 0.3, 7.5, 5.4, mix(C["r_fill"], 0.6), C["r_edge"], lw=1.4, r=0.25, z=0)
cross(0.62, 5.25)
ax.text(0.92, 5.25, "a  Vote on the final state", ha="left", va="center", fontsize=14.5, fontweight="bold", color=C["r_edge"])
for k, (name, h) in enumerate((("LoRA p1", "h1"), ("LoRA e1", "h1"), ("LoRA synth", "h2"))):
    yy = 4.15 - k * 1.05
    agent(0.55, yy, name)
    arrow(ax, 2.1 + GAP, yy + 0.25, 2.65 - GAP, yy + 0.25, C["p_edge"], width=0.14, head_w=0.3, head_l=0.13)
    env(2.65, yy, label=f"ToolEnv #{k + 1}")
    arrow(ax, 4.2 + GAP, yy + 0.25, 4.75 - GAP, yy + 0.25, AMBER, width=0.14, head_w=0.3, head_l=0.13)
    box(ax, 4.75, yy + 0.05, 1.0, 0.4, "white", C["grey"], lw=1.0, r=0.08, z=3)
    ax.text(5.25, yy + 0.25, f"hash {h}", ha="center", va="center", fontsize=10.5, family="Liberation Mono", color=C["dark"], zorder=5)
    ax.plot([5.75 + GAP, 6.15], [yy + 0.25, 3.12], color=C["grey"], lw=1.1, zorder=2)
gbox(ax, 6.15, 2.82, 1.25, 0.6, "white", C["grey"], lw=1.1, r=0.1)
ax.text(6.775, 3.12, "pick h1", ha="center", va="center", fontsize=11, fontweight="bold", color=C["dark"], zorder=5)
rich(ax, 3.95, 0.98, [("three ToolEnvs for one sample, two runs thrown away", {"color": C["r_edge"], "weight": "bold"})], size=11.5)
rich(ax, 3.95, 0.62, [("kit README: ", G), ("“one ToolEnv per sample, created once, every call counts”", I)], size=11)

# ---- panel b: vote per step on one ToolEnv --------------------------------------------------------------------------------------------
box(ax, 8.1, 0.3, 7.7, 5.4, mix(C["g_fill"], 0.55), C["g_edge"], lw=1.4, r=0.25, z=0)
tick(8.52, 5.22)
ax.text(8.82, 5.25, "b  Vote per step, one ToolEnv (ours)", ha="left", va="center", fontsize=14.5, fontweight="bold", color=C["g_text"])
props = [("LoRA p1", "pay(bank)", C["g_text"]), ("LoRA e1", "pay(bank)", C["g_text"]), ("LoRA synth", "pay(ewallet)", C["r_edge"])]
for k, (name, prop, col) in enumerate(props):
    yy = 4.15 - k * 0.75
    agent(8.45, yy, name)
    box(ax, 10.15, yy + 0.05, 1.55, 0.4, "white", col, lw=1.0, r=0.08, z=3)
    ax.text(10.925, yy + 0.25, prop, ha="center", va="center", fontsize=10.3, family="Liberation Mono", color=col, zorder=5)
    ax.plot([11.7 + GAP, 12.15], [yy + 0.25, 3.65], color=C["grey"], lw=1.1, zorder=2)
gbox(ax, 12.15, 3.25, 1.35, 0.8, mix(C["p_fill"], 0.2), C["p_edge"], lw=1.3, r=0.12)
ax.text(12.825, 3.82, "majority", ha="center", va="center", fontsize=11.5, fontweight="bold", color=C["p_text"], zorder=5)
ax.text(12.825, 3.5, "tie goes to p1", ha="center", va="center", fontsize=10, color=C["grey"], zorder=5)
arrow(ax, 13.5 + GAP, 3.65, 14.05 - GAP, 3.65, C["p_edge"], width=0.14, head_w=0.3, head_l=0.13)
env(14.05, 3.4, w=1.45)
# result goes back to all three for the next turn
ax.plot([14.775, 14.775, 9.225, 9.225], [3.4 - GAP, 2.0, 2.0, 2.65 - GAP], color=C["loop"], lw=1.6, ls=(0, (4, 2.5)), zorder=1)
ax.add_patch(FancyArrowPatch((9.225, 2.3), (9.225, 2.65 - GAP + 0.02), arrowstyle="-|>", mutation_scale=13, color=C["loop"], lw=1.6, zorder=3))
rich(ax, 12.0, 1.78, [("one call executed, its result goes to all three", {"color": C["loop"]}), (", then the next turn", {"color": C["loop"]})], size=11, style="italic")
rich(ax, 11.95, 0.85, [("validation: ", G), ("70.91", B), (" best single, ", G), ("72.71", {"weight": "bold", "color": C["g_text"]}), (" with the vote", G)], size=12)

save(fig, "ensemble")
