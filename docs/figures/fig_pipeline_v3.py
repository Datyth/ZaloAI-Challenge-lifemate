"""LifeMate inference pipeline, linear and sparse (figures4papers style, Track 1 deck v2): one block per stage, one line of detail,
one loop arrow. Running example: TRAIN validation sample train_0169.
Run: python docs/figures/fig_pipeline_v3.py -> docs/figures/pipeline_v3.{png,pdf,svg}"""
import numpy as np
from matplotlib.patches import FancyArrowPatch

from style import C, arrow, box, gbox, gpoly, mix, rich, save, setup

W, H = 16.0, 4.3
fig, ax = setup(W, H)
B, G, I = {"weight": "bold"}, {"color": C["grey"]}, {"style": "italic"}
M = {"family": "Liberation Mono"}
GAP = 0.13
AMBER, AMBER_F, AMBER_T = "#B07A12", "#FBE3B6", "#7A520A"
Y0, Y1 = 1.15, 2.95            # every block spans the same band
YM = (Y0 + Y1) / 2


def label(x, title, sub, color):
    ax.text(x, 0.72, title, ha="center", va="center", fontsize=15, fontweight="bold", color=color)
    ax.text(x, 0.38, sub, ha="center", va="center", fontsize=12, color=C["grey"], style="italic")


# audio
rng = np.random.default_rng(3)
env = np.sin(np.linspace(0.15, np.pi - 0.15, 22)) ** 0.7
for k, a in enumerate(env * (0.35 + 0.65 * rng.random(22))):
    xx = 0.3 + 1.2 * k / 21
    ax.plot([xx, xx], [YM - 0.42 * a, YM + 0.42 * a], color=C["b_edge"], lw=2.2, solid_capstyle="round")
label(0.9, "Audio", "request + history", C["dark"])

# ASR
XA, WA = 2.05, 2.0
gbox(ax, XA, Y0, WA, Y1 - Y0, C["b_fill"], C["b_edge"], r=0.22)
ax.text(XA + WA / 2, YM + 0.25, "Gipformer", ha="center", va="center", fontsize=15, fontweight="bold", color=C["b_text"], zorder=5)
ax.text(XA + WA / 2, YM - 0.18, "RNN-T, 68M", ha="center", va="center", fontsize=12.5, color=C["b_text"], zorder=5)
label(XA + WA / 2, "ASR", "WER 15.2% to 7.7%", C["b_text"])

# transcript
XT, WT = 4.6, 3.05
box(ax, XT, Y0, WT, Y1 - Y0, "white", C["dark"], lw=1.3, r=0.22, z=3)
rich(ax, XT + WT / 2, YM + 0.3, [("“… ví mà ", I), ("không được", {**I, **B, "color": C["r_edge"]}), (" thì", I)], size=13.5)
rich(ax, XT + WT / 2, YM - 0.12, [("trừ vào tài khoản ngân hàng”", I)], size=13.5)
label(XT + WT / 2, "Transcript", "said: “không đủ”", C["dark"])

# context (into the LLM from above)
XL, XLe = 8.2, 10.3
cL = (XL + XLe) / 2
ax.text(cL, 3.95, "context, history, tool schemas", ha="center", va="center", fontsize=12, color=C["grey"], style="italic")
arrow(ax, cL, 3.75, cL, Y1 + 0.05 + GAP, C["grey"], width=0.14, head_w=0.3, head_l=0.13)

# LLM
gpoly(ax, [[XL, Y0 - 0.05], [XL, Y1 + 0.05], [XLe, Y1 - 0.3], [XLe, Y0 + 0.3]], C["p_fill"], C["p_edge"], lw=1.8)
ax.text(XL + 0.95, YM + 0.25, "Qwen3.5-4B", ha="center", va="center", fontsize=14.5, fontweight="bold", color=C["p_text"], zorder=5)
ax.text(XL + 0.95, YM - 0.18, "3 LoRA, vote", ha="center", va="center", fontsize=12.5, color=C["p_text"], zorder=5)
label(cL, "LLM", "one call per turn", C["p_text"])

# ToolEnv
XE, WE = 10.85, 3.05
gbox(ax, XE, Y0, WE, Y1 - Y0, AMBER_F, AMBER, r=0.22)
for k, (call, col) in enumerate((("list_bills()", AMBER_T), ("get_balance()", AMBER_T), ('pay(source="bank")', C["g_text"]))):
    rich(ax, XE + 0.25, YM + 0.42 - k * 0.42, [(f"{k + 1}  ", {"weight": "bold", "color": AMBER_T}), (call, {**M, "color": col})], ha="left", size=12)
label(XE + WE / 2, "ToolEnv", "validate, then execute", AMBER_T)

# output
XO, WO = 14.45, 1.3
gbox(ax, XO, Y0 + 0.3, WO, Y1 - Y0 - 0.6, C["g_fill"], C["g_edge"], r=0.18)
ax.text(XO + WO / 2, YM + 0.12, "3 calls", ha="center", va="center", fontsize=13.5, fontweight="bold", color=C["g_text"], zorder=5)
ax.text(XO + WO / 2, YM - 0.22, "state_hash", ha="center", va="center", fontsize=10.5, color=C["grey"], family="Liberation Mono", zorder=5)
label(XO + WO / 2, "Output", "or ask back", C["g_text"])

# arrows
arrow(ax, 1.55 + GAP, YM, XA - GAP, YM, C["b_edge"])
arrow(ax, XA + WA + GAP, YM, XT - GAP, YM, C["b_edge"])
arrow(ax, XT + WT + GAP, YM, XL - GAP, YM, C["p_edge"])
arrow(ax, XLe + GAP, YM, XE - GAP, YM, C["p_edge"])
arrow(ax, XE + WE + GAP, YM, XO - GAP, YM, AMBER)
# loop: tool result back to the LLM, above the blocks
LY = 3.45
ax.plot([XE + WE / 2, XE + WE / 2, XLe - 0.35], [Y1 + GAP, LY, LY], color=C["loop"], lw=1.6, ls=(0, (4, 2.5)), zorder=1)
ax.add_patch(FancyArrowPatch((XLe - 0.35, LY), (XLe - 0.35, Y1 - 0.22 + GAP), arrowstyle="-|>", mutation_scale=13, color=C["loop"], lw=1.6, zorder=3))
ax.text((XE + WE / 2 + XLe - 0.35) / 2, LY + 0.2, "result back to the LLM", ha="center", va="center", fontsize=12, color=C["loop"], style="italic")

save(fig, "pipeline_v3")
