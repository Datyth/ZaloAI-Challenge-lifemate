"""Inference pipeline of the current best LifeMate system (q35e1: public 58.02, val 70.91), figures4papers style.
Running example: TRAIN validation sample train_0169 (solved exactly with ASR). Run: python docs/figures/fig_pipeline.py"""
import numpy as np
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch, Polygon, Rectangle

from style import C, arrow, box, flow, gbox, gpoly, logo, mix, rich, save, setup

fig, ax = setup(20.1, 6.6)
B, G, I = {"weight": "bold"}, {"color": C["grey"]}, {"style": "italic"}
M = {"family": "Liberation Mono"}
GAP = 0.12


def title(x, text, color, sub=None):
    ax.text(x, 0.8, text, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    if sub:
        rich(ax, x, 0.4, sub, size=13.5, color=C["grey"], style="italic")


def wave(x, y, w, amp, n, color, seed, lw=2.2):
    """Audio waveform: n vertical bars centred on y."""
    rng = np.random.default_rng(seed)
    env = np.sin(np.linspace(0.15, np.pi - 0.15, n)) ** 0.7
    for k, a in enumerate(env * (0.35 + 0.65 * rng.random(n))):
        xx = x + w * k / (n - 1)
        ax.plot([xx, xx], [y - amp * a, y + amp * a], color=color, lw=lw, solid_capstyle="round", zorder=5)


def result_line(x, y, parts, size=9.4):
    """Indented tool result: a small drawn arrow, then grey text."""
    hw = 0.1
    ax.add_patch(FancyArrow(x, y, 0.16, 0, width=hw * 0.36, head_width=hw, head_length=0.08, length_includes_head=True,
                            fc=C["grey"], ec="none", zorder=6))
    rich(ax, x + 0.24, y, parts, ha="left", size=size)


# ---- speech input ------------------------------------------------------------------------------------------------------------------
box(ax, 0.22, 2.98, 1.62, 0.78, "white", C["grey"], lw=1.3, r=0.16, z=3)
wave(0.38, 3.37, 1.3, 0.3, 26, C["b_edge"], seed=3)
ax.text(1.03, 2.68, "Request audio", ha="center", va="center", fontsize=14, fontweight="bold", color=C["dark"])
ax.text(1.03, 2.4, "6 s, 16 kHz", ha="center", va="center", fontsize=11, color=C["grey"], style="italic")
for k in (1, 0):
    box(ax, 0.42 + 0.1 * k, 1.38 + 0.1 * k, 1.22, 0.5, "white", mix(C["grey"], 0.3), lw=1.0, r=0.12, z=3)
wave(0.55, 1.63, 0.96, 0.16, 18, mix(C["b_edge"], 0.45), seed=8, lw=1.6)
ax.text(1.03, 1.12, "history audio", ha="center", va="center", fontsize=11.5, color=C["grey"], style="italic")

# ---- 1 ASR -------------------------------------------------------------------------------------------------------------------------
X1, W1 = 2.55, 2.2
c1 = X1 + W1 / 2
gbox(ax, X1, 1.2, W1, 3.1, C["b_fill"], C["b_edge"], r=0.3)
ax.text(c1, 3.95, "Gipformer", ha="center", va="center", fontsize=15, fontweight="bold", color=C["b_text"], zorder=5)
ax.text(c1, 3.65, "RNN-T, 68M", ha="center", va="center", fontsize=12, color=C["b_text"], zorder=5)
ax.text(c1, 3.25, "WER on val audio", ha="center", va="center", fontsize=10.5, color=C["b_text"], style="italic", zorder=5)
for k, (name, wer, fc) in enumerate((("base", 17.1, mix(C["grey"], 0.35)), ("ours", 10.7, C["b_edge"]))):
    yy = 2.9 - k * 0.36
    ax.text(X1 + 0.5, yy, name, ha="right", va="center", fontsize=11, color=C["b_text"], fontweight="bold" if k else "normal", zorder=5)
    ax.add_patch(Rectangle((X1 + 0.58, yy - 0.1), wer * 0.056, 0.2, fc=fc, ec="none", zorder=5))
    ax.text(X1 + 0.66 + wer * 0.056, yy, f"{wer}%", ha="left", va="center", fontsize=11, color=C["b_text"], fontweight="bold" if k else "normal", zorder=5)
box(ax, X1 + 0.32, 1.45, W1 - 0.64, 0.32, "white", C["b_edge"], lw=1.1, r=0.13, z=4)
ax.text(c1, 1.61, "sherpa-onnx, CPU", ha="center", va="center", fontsize=11.5, style="italic", color=C["b_text"], zorder=5)
title(c1, "ASR", C["b_text"], [("fine-tuned", B), (" on TRAIN audio", {})])

# ---- 2 transcript ------------------------------------------------------------------------------------------------------------------
X2, W2 = 5.35, 2.35
c2 = X2 + W2 / 2
box(ax, X2, 1.35, W2, 2.8, "white", C["dark"], lw=1.5, r=0.32, z=3)
RED = {**I, **B, "color": C["r_edge"]}
lines = [[("“", I), ("trả tiền điện", {**I, **B, "color": C["p_text"]}), (" bằng", I)],
         [("ví điện tử", {**I, **B, "color": C["y_text"]}), (" giúp mình", I)],
         [("ví mà ", I), ("không được", RED), (" thì", I)],
         [("trừ vào ", I), ("tài khoản", {**I, **B, "color": C["y_text"]})],
         [("ngân hàng", {**I, **B, "color": C["y_text"]}), (" nhé”", I)]]
for k, seg in enumerate(lines):
    rich(ax, c2, 3.72 - k * 0.37, seg, size=13.5)
rich(ax, c2, 1.65, [("gold: ", G), ("không đủ", {"weight": "bold", "color": C["g_text"]})], size=11.5)
title(c2, "Transcript", C["dark"], [("lower-case", B), (", no punctuation", {})])

# ---- context (enters the prompt from above) ----------------------------------------------------------------------------------------
X3, W3 = 8.35, 2.1
c3 = X3 + W3 / 2
box(ax, c3 - 2.2, 5.18, 4.4, 1.14, "white", C["dark"], lw=1.5, r=0.32)
rich(ax, c3, 6.04, [("current_time ", G), ("2026-11-08 07:10", {"color": C["y_text"], "weight": "bold"}), ("   user ", G), ("Chị Hoa", B)], size=12)
rich(ax, c3, 5.74, [("14", {"color": C["b_text"], "weight": "bold"}), (" installed apps and their tool schemas", G)], size=12)
rich(ax, c3, 5.44, [("history", {"color": C["g_text"], "weight": "bold"}), (" past tool calls and results", G)], size=12)

# ---- 3 prompt ----------------------------------------------------------------------------------------------------------------------
ax.add_patch(Polygon([[X3, 1.2], [X3 + W3, 1.2], [X3 + W3, 4.0], [X3 + W3 - 0.28, 4.3], [X3, 4.3]], closed=True, fc="white", ec=C["dark"], lw=1.6, zorder=2))
ax.add_patch(Polygon([[X3 + W3 - 0.28, 4.3], [X3 + W3 - 0.28, 4.0], [X3 + W3, 4.0]], closed=True, fc=C["light"], ec=C["dark"], lw=1.2, zorder=3))
parts = [("rules", "#E4E4E4", C["dark"]), ("tool schemas", C["b_fill"], C["b_text"]), ("context", C["y_fill"], C["y_text"]),
         ("history", C["g_fill"], C["g_text"]), ("request", C["p_fill"], C["p_text"])]
for k, (name, fc, tc) in enumerate(parts):
    yy = 3.6 - k * 0.5
    gbox(ax, X3 + 0.18, yy, W3 - 0.36, 0.38, fc, mix(tc, 0.35), lw=0.9, r=0.1, z=3)
    ax.text(c3, yy + 0.19, name, ha="center", va="center", fontsize=12, color=tc, zorder=5)
title(c3, "Prompt", C["dark"], [("≈ ", {}), ("6k", B), (" tokens, ≤ 16k", {})])

# ---- 4 LLM -------------------------------------------------------------------------------------------------------------------------
X4, X4e = 11.1, 12.85
c4 = X4 + 0.87
gpoly(ax, [[X4, 1.3], [X4, 4.2], [X4e, 3.62], [X4e, 1.88]], C["p_fill"], C["p_edge"])
logo(ax, "qwen_gh.png", X4 + 0.62, 3.22, 0.52)
ax.text(c4, 2.84, "Qwen3.5-4B", ha="center", va="center", fontsize=15, fontweight="bold", color=C["p_text"], zorder=5)
ax.text(c4, 2.5, "+ LoRA r16", ha="center", va="center", fontsize=14, color=C["p_text"], zorder=5)
w = logo(ax, "vllm.png", X4 + 0.42, 1.9, 0.3)
ax.text(X4 + 0.42 + w + 0.05, 2.05, "vLLM", ha="left", va="center", fontsize=11, color=C["p_text"], zorder=5)
title(c4, "LLM", C["p_text"], [("SFT", B), (" on gold and ASR text", {})])

# ---- 5 validate -------------------------------------------------------------------------------------------------------------------
XV, WV = 13.45, 1.6
cv = XV + WV / 2
gbox(ax, XV, 1.2, WV, 3.1, C["r_fill"], C["r_edge"], r=0.3)
ax.add_patch(Polygon([[cv - 0.24, 4.02], [cv, 4.1], [cv + 0.24, 4.02], [cv + 0.24, 3.74], [cv, 3.48], [cv - 0.24, 3.74]], closed=True,
                     fc=C["r_edge"], ec="none", zorder=5))
ax.plot([cv - 0.1, cv - 0.02, cv + 0.12], [3.8, 3.7, 3.92], color="white", lw=2.2, solid_capstyle="round", solid_joinstyle="round", zorder=6)
for k, name in enumerate(("schema", "x-ref IDs seen", "no repeats")):
    yy = 3.12 - k * 0.36
    ax.add_patch(Circle((XV + 0.25, yy), 0.08, fc=C["g_edge"], ec="none", zorder=5))
    ax.text(XV + 0.42, yy, name, ha="left", va="center", fontsize=11.5, color=C["dark"], zorder=5)
ax.text(cv, 1.84, "rejected:", ha="center", va="center", fontsize=11, fontweight="bold", color=C["r_edge"], zorder=5)
ax.text(cv, 1.56, "error to the LLM", ha="center", va="center", fontsize=10.5, style="italic", color=C["r_edge"], zorder=5)
title(cv, "Validate", C["r_edge"], [("before", B), (" env.call", {})])

# ---- 6 agent loop ------------------------------------------------------------------------------------------------------------------
X5, W5 = 15.6, 2.8
c5 = X5 + W5 / 2
gbox(ax, X5, 1.2, W5, 3.1, C["y_fill"], C["y_edge"], r=0.3)
rich(ax, c5, 4.0, [("ToolEnv", {"weight": "bold", "color": C["y_text"]}), ("  zaic_tools", {"color": C["y_text"], "style": "italic"})], size=14)
turns = [([("bill_payment.list_bills()", M)], [("điện ", G), ("845.000đ", {**B, "color": C["dark"]}), (", bill_02", G)]),
         ([("ewallet.get_balance()", M)], [("713.606đ", {**B, "color": C["r_edge"]}), (" < 845.000đ", G)]),
         ([("bill_payment.pay(", M)], [("bill_02, ", M), ('source="bank"', {**M, **B, "color": C["g_text"]}), (")", M)])]
for k, (call, res) in enumerate(turns):
    yy = 3.55 - k * 0.66
    ax.add_patch(Circle((X5 + 0.24, yy), 0.13, fc=C["g_edge"] if k == 2 else C["y_edge"], ec="white", lw=1.0, zorder=5))
    ax.text(X5 + 0.24, yy, str(k + 1), ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=6)
    rich(ax, X5 + 0.46, yy, call, ha="left", size=9.6)
    if k < 2:
        result_line(X5 + 0.5, yy - 0.29, res, size=10.5)
    else:
        rich(ax, X5 + 0.66, yy - 0.29, res, ha="left", size=9.6)
rich(ax, c5, 1.5, [("stop", {"weight": "bold", "color": C["y_text"]}), (" when the LLM calls nothing", {"color": C["y_text"]})], size=11, style="italic")
title(c5, "Agent loop", C["y_text"], [("≤ 10", B), (" turns, one env per sample", {})])

# ---- output ------------------------------------------------------------------------------------------------------------------------
X6, W6 = 18.9, 1.0
c6 = X6 + W6 / 2
gbox(ax, X6, 2.5, W6, 1.6, C["g_fill"], C["g_edge"], lw=1.8, r=0.2)
ax.plot([c6 - 0.17, c6 - 0.05, c6 + 0.18], [3.82, 3.68, 3.96], color=C["g_edge"], lw=3.2, solid_capstyle="round", solid_joinstyle="round", zorder=5)
ax.text(c6, 3.3, "3 calls", ha="center", va="center", fontsize=12.5, color=C["g_text"], fontweight="bold", zorder=5)
ax.text(c6, 2.95, "in order", ha="center", va="center", fontsize=11, color=C["g_text"], zorder=5)
ax.text(c6, 2.68, "state_hash", ha="center", va="center", fontsize=9.5, color=C["grey"], family="Liberation Mono", zorder=5)
gbox(ax, X6, 1.42, W6, 0.78, C["p_fill"], C["p_edge"], lw=1.4, r=0.2)
ax.text(c6, 1.93, "missing slot", ha="center", va="center", fontsize=10.5, color=C["p_text"], zorder=5)
ax.text(c6, 1.64, "ask back", ha="center", va="center", fontsize=12.5, color=C["p_text"], fontweight="bold", zorder=5)
title(c6, "Output", C["g_text"])

# ---- scores --------------------------------------------------------------------------------------------------------------------------
rich(ax, 19.95, 6.1, [("val ", G), ("70.91", {"weight": "bold", "color": C["g_text"]}), ("    public ", G), ("58.02", {"weight": "bold", "color": C["g_text"]})], ha="right", size=14)
rich(ax, 19.95, 5.75, [("3.2 s", B), (" per sample, limit 30 s", G)], ha="right", size=12)

# ---- arrows (a gap on both sides) ----------------------------------------------------------------------------------------------------
arrow(ax, 1.84 + GAP, 3.37, X1 - GAP, 3.37, C["b_edge"])
arrow(ax, 1.74 + GAP, 1.68, X1 - GAP, 1.68, C["b_edge"], width=0.2, head_w=0.42, head_l=0.18)
arrow(ax, X1 + W1 + GAP, 2.75, X2 - GAP, 2.75, C["b_edge"])
arrow(ax, X2 + W2 + GAP, 2.75, X3 - GAP, 2.75, C["p_edge"])
arrow(ax, c3, 5.18 - GAP, c3, 4.3 + GAP, C["y_edge"])
arrow(ax, X3 + W3 + GAP, 2.75, X4 - GAP, 2.75, C["dark"])
arrow(ax, X4e + GAP, 2.95, XV - GAP, 2.95, C["p_edge"])
ax.add_patch(FancyArrowPatch((XV - GAP, 2.12), (X4e + GAP, 2.12), arrowstyle="-|>", mutation_scale=12, color=C["r_edge"], lw=1.5, zorder=3))
arrow(ax, XV + WV + GAP, 2.75, X5 - GAP, 2.75, C["r_edge"])
arrow(ax, X5 + W5 + GAP, 3.3, X6 - GAP, 3.3, C["y_edge"], width=0.22, head_w=0.44, head_l=0.18)
ax.add_patch(FancyArrowPatch((X5 + W5 + GAP, 1.81), (X6 - GAP, 1.81), arrowstyle="-|>", mutation_scale=12, color=C["p_edge"], lw=1.5, zorder=3))
# agent loop: each tool result goes back to the LLM for the next turn
ax.plot([c5, c5, X4 + 1.35, X4 + 1.35], [4.3 + GAP, 4.72, 4.72, 3.95], color=C["loop"], lw=1.7, ls=(0, (4, 2.5)), zorder=1)
ax.add_patch(FancyArrowPatch((X4 + 1.35, 4.05), (X4 + 1.35, 3.72 + GAP), arrowstyle="-|>", mutation_scale=14, color=C["loop"], lw=1.7, zorder=3))
flow(ax, (X4 + 1.35 + c5) / 2, 4.97, [[("tool result", B)], [("next LLM turn", {})]], size=13, ha="center", color=C["loop"], style="italic")

save(fig, "pipeline")
