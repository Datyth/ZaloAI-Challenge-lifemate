"""LifeMate inference pipeline in three lanes (HEAR / DECIDE / ACT), figures4papers style, for the Track 1 deck v2.
Differs from the DataMate figure on purpose: lanes instead of one conveyor, the ACT lane runs back right to left, and the
agent loop climbs from ToolEnv to the voting LoRAs. Running example: TRAIN validation sample train_0169.
Run: python docs/figures/fig_pipeline_v2.py  ->  docs/figures/pipeline_v2.{png,pdf,svg}"""
import numpy as np
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from style import C, arrow, box, gbox, gpoly, logo, mix, rich, save, setup

W, H = 16.6, 8.5
fig, ax = setup(W, H)
B, G, I = {"weight": "bold"}, {"color": C["grey"]}, {"style": "italic"}
M = {"family": "Liberation Mono"}
GAP = 0.12
AMBER, AMBER_F, AMBER_T = "#B07A12", "#FBE3B6", "#7A520A"

# ---- lanes ---------------------------------------------------------------------------------------------------------------------------
LANES = [("HEAR", 5.95, 8.15, C["b_edge"]), ("DECIDE", 3.15, 5.75, C["p_edge"]), ("ACT", 0.25, 2.95, AMBER)]
for name, y0, y1, col in LANES:
    ax.add_patch(FancyBboxPatch((0.08, y0), W - 0.16, y1 - y0, boxstyle="round,pad=0,rounding_size=0.18", fc=mix(col, 0.94), ec=mix(col, 0.6), lw=1.0, zorder=0))
    ax.add_patch(FancyBboxPatch((0.08, y0), 0.52, y1 - y0, boxstyle="round,pad=0,rounding_size=0.18", fc=col, ec=col, lw=1.0, zorder=0.5))
    ax.text(0.34, (y0 + y1) / 2, name, rotation=90, ha="center", va="center", fontsize=15, fontweight="bold", color="white", zorder=1)


def wave(x, y, w, amp, n, color, seed, lw=2.2):
    rng = np.random.default_rng(seed)
    env = np.sin(np.linspace(0.15, np.pi - 0.15, n)) ** 0.7
    for k, a in enumerate(env * (0.35 + 0.65 * rng.random(n))):
        xx = x + w * k / (n - 1)
        ax.plot([xx, xx], [y - amp * a, y + amp * a], color=color, lw=lw, solid_capstyle="round", zorder=5)


def caption(x, y, text, color, sub=None):
    ax.text(x, y, text, ha="center", va="center", fontsize=14.5, fontweight="bold", color=color, zorder=6)
    if sub:
        rich(ax, x, y - 0.27, sub, size=11.5, color=C["grey"], style="italic")


# ==== HEAR ===============================================================================================================================
yh = 7.05
box(ax, 0.95, yh - 0.05, 1.55, 0.72, "white", C["grey"], lw=1.3, r=0.15, z=3)
wave(1.1, yh + 0.31, 1.25, 0.26, 24, C["b_edge"], seed=3)
for k in (1, 0):
    box(ax, 1.08 + 0.08 * k, 6.3 + 0.08 * k, 1.15, 0.42, "white", mix(C["grey"], 0.3), lw=1.0, r=0.1, z=3)
wave(1.2, 6.51, 0.9, 0.13, 16, mix(C["b_edge"], 0.45), seed=8, lw=1.5)
ax.text(1.72, 7.92, "request audio", ha="center", va="center", fontsize=12.5, fontweight="bold", color=C["dark"])
ax.text(1.72, 6.12, "+ history turns", ha="center", va="center", fontsize=10.5, color=C["grey"], style="italic")

XA, WA = 3.75, 2.7
gbox(ax, XA, 6.15, WA, 1.8, C["b_fill"], C["b_edge"], r=0.25)
cA = XA + WA / 2
rich(ax, cA, 7.68, [("Gipformer", {"weight": "bold", "color": C["b_text"]}), (" RNN-T 68M", {"color": C["b_text"]})], size=13.5)
for k, (name, wer, fc, bold) in enumerate((("base", 15.2, mix(C["grey"], 0.35), False), ("v2", 8.7, mix(C["b_edge"], 0.45), False),
                                           ("v3", 7.7, C["b_edge"], True))):
    yy = 7.3 - k * 0.3
    ax.text(XA + 0.55, yy, name, ha="right", va="center", fontsize=11, color=C["b_text"], fontweight="bold" if bold else "normal", zorder=5)
    ax.add_patch(Rectangle((XA + 0.63, yy - 0.09), wer * 0.095, 0.18, fc=fc, ec="none", zorder=5))
    ax.text(XA + 0.7 + wer * 0.095, yy, f"{wer}%", ha="left", va="center", fontsize=11, color=C["b_text"], fontweight="bold" if bold else "normal", zorder=5)
ax.text(cA, 6.33, "WER on val audio · sherpa-onnx, CPU", ha="center", va="center", fontsize=10, style="italic", color=C["b_text"], zorder=5)

XT, WT = 7.05, 4.3
box(ax, XT, 6.2, WT, 1.75, "white", C["dark"], lw=1.4, r=0.25, z=3)
cT = XT + WT / 2
RED = {**I, **B, "color": C["r_edge"]}
rich(ax, cT, 7.66, [("“", I), ("trả tiền điện", {**I, **B, "color": C["p_text"]}), (" bằng ", I), ("ví điện tử", {**I, **B, "color": AMBER_T}), (" giúp mình", I)], size=12.5)
rich(ax, cT, 7.33, [("ví mà ", I), ("không được", RED), (" thì trừ vào", I)], size=12.5)
rich(ax, cT, 7.0, [("tài khoản ngân hàng", {**I, **B, "color": AMBER_T}), (" nhé”", I)], size=12.5)
rich(ax, cT, 6.5, [("misheard ", G), ("không được", {"color": C["r_edge"], "weight": "bold"}), (" · said ", G), ("không đủ", {"color": C["g_text"], "weight": "bold"})], size=10.5)

XC = 11.85
rich(ax, XC, 7.68, [("ASR fine-tuning data", {"weight": "bold", "color": C["b_text"]}), (" (TRAIN audio only)", G)], ha="left", size=12)
for k, parts in enumerate(([("number labels", B), (" spelled from their value: 368 / 368 kept", G)],
                           [("augmentation", B), (": speed 0.9–1.1, noise, SpecAugment", G)],
                           [("TTS", B), (": 1,202 cloned-voice sentences, new names and numbers", G)],
                           [("final", B), (": + dev and validation audio (all in)", G)])):
    ax.add_patch(Circle((XC + 0.08, 7.3 - k * 0.34), 0.05, fc=C["b_edge"], ec="none", zorder=5))
    rich(ax, XC + 0.22, 7.3 - k * 0.34, parts, ha="left", size=10.5)

arrow(ax, 2.5 + GAP, yh + 0.31, XA - GAP, yh + 0.31, C["b_edge"])
arrow(ax, 2.33 + GAP, 6.55, XA - GAP, 6.55, C["b_edge"], width=0.16, head_w=0.34, head_l=0.15)
arrow(ax, XA + WA + GAP, 7.05, XT - GAP, 7.05, C["b_edge"])

# ==== DECIDE =============================================================================================================================
yd = 4.45
# prompt document, fed by the transcript (from above) and by context / apps / history (from the right, via the lane text)
XP, WP, YP, HP = 7.95, 2.7, 3.35, 2.25
ax.add_patch(Polygon([[XP, YP], [XP + WP, YP], [XP + WP, YP + HP - 0.25], [XP + WP - 0.25, YP + HP], [XP, YP + HP]], closed=True, fc="white", ec=C["dark"], lw=1.5, zorder=2))
ax.add_patch(Polygon([[XP + WP - 0.25, YP + HP], [XP + WP - 0.25, YP + HP - 0.25], [XP + WP, YP + HP - 0.25]], closed=True, fc=C["light"], ec=C["dark"], lw=1.1, zorder=3))
sec = [("rules + recovery hints", "#E4E4E4", C["dark"]), ("app names (vi) + schemas", C["b_fill"], C["b_text"]),
       ("context + weekday table", AMBER_F, AMBER_T), ("history", C["g_fill"], C["g_text"]), ("request (transcript)", C["p_fill"], C["p_text"])]
for k, (name, fc, tc) in enumerate(sec):
    yy = YP + HP - 0.55 - k * 0.37
    gbox(ax, XP + 0.14, yy, WP - 0.42, 0.29, fc, mix(tc, 0.35), lw=0.8, r=0.08, z=3)
    ax.text(XP + 0.14 + (WP - 0.42) / 2, yy + 0.145, name, ha="center", va="center", fontsize=10.2, color=tc, zorder=5)
ax.text(XP + WP / 2, YP - 0.0, "", fontsize=1)
arrow(ax, XP + WP / 2, 6.2 - GAP, XP + WP / 2, YP + HP + GAP, C["p_edge"], width=0.24, head_w=0.48, head_l=0.2)

# left part of the lane: what the prompt adds (E1P)
rich(ax, 0.95 + 3.25, 5.22, [("Prompt E1P", {"weight": "bold", "color": C["p_text"]}), (" (≈ 6k tokens, ≤ 16k)", G)], size=12.5)
notes = [[("context", B), (": time, user profile, installed apps", G)],
         [("weekday table", B), (": \"thứ bảy tuần sau\" → a date", G)],
         [("Vietnamese app names", B), (": \"bên ví\" → ewallet", G)],
         [("recovery", B), (": empty search → shorter query", G)],
         [("72%", B), (" of the tokens are tool schemas", G)]]
notes = notes[:4] + notes[4:]
for k, parts in enumerate(notes):
    ax.add_patch(Circle((1.12, 4.85 - k * 0.33), 0.05, fc=C["p_edge"], ec="none", zorder=5))
    rich(ax, 1.27, 4.85 - k * 0.33, parts, ha="left", size=11)

# LLM: shared base, three LoRA members, one proposal each
XL, XLe = 11.25, 13.85
gpoly(ax, [[XL, 3.3], [XL, 5.6], [XLe, 5.15], [XLe, 3.75]], C["p_fill"], C["p_edge"])
logo(ax, "qwen_gh.png", XL + 0.12, 5.02, 0.4)
ax.text(XL + 0.6, 5.22, "Qwen3.5-4B", ha="left", va="center", fontsize=12.5, fontweight="bold", color=C["p_text"], zorder=5)
members = [("LoRA p1", "E1P", "pay(bank)", C["g_text"]), ("LoRA e1", "E1", "pay(bank)", C["g_text"]), ("LoRA synth", "E1", "pay(ewallet)", C["r_edge"])]
for k, (name, orch, prop, pc) in enumerate(members):
    yy = 4.72 - k * 0.42
    box(ax, XL + 0.12, yy - 0.16, 1.15, 0.32, "white", C["p_edge"], lw=0.9, r=0.08, z=4)
    rich(ax, XL + 0.695, yy, [(name, {"weight": "bold", "color": C["p_text"]})], size=10)
    rich(ax, XL + 1.36, yy, [(prop, {**M, "color": pc})], ha="left", size=9.6)
logo(ax, "vllm.png", XL + 1.92, 4.93, 0.22)
arrow(ax, XP + WP + GAP, yd, XL - GAP, yd, C["p_edge"])

# vote
XV, WV = 14.35, 2.05
gbox(ax, XV, 3.45, WV, 2.0, mix(C["p_fill"], 0.2), C["p_edge"], r=0.22)
cV = XV + WV / 2
ax.text(cV, 5.15, "vote", ha="center", va="center", fontsize=14, fontweight="bold", color=C["p_text"], zorder=5)
ax.text(cV, 4.83, "per turn", ha="center", va="center", fontsize=11, style="italic", color=C["p_text"], zorder=5)
rich(ax, cV, 4.42, [("2 / 3", {"weight": "bold", "color": C["g_text"]}), (" → pay(bank)", {**M, "color": C["dark"]})], size=10.5)
ax.text(cV, 4.05, "tie → p1", ha="center", va="center", fontsize=10.5, color=C["grey"], zorder=5)
ax.text(cV, 3.72, "val 70.9 → 72.7", ha="center", va="center", fontsize=10.5, color=C["p_text"], fontweight="bold", zorder=5)
arrow(ax, XLe + GAP, yd, XV - GAP, yd, C["p_edge"])

# ==== ACT (right to left) ================================================================================================================
ya = 1.6
XG, WG = 14.35, 2.05
gbox(ax, XG, 0.45, WG, 2.3, C["r_fill"], C["r_edge"], r=0.22)
cG = XG + WG / 2
ax.add_patch(Polygon([[cG - 0.2, 2.5], [cG, 2.57], [cG + 0.2, 2.5], [cG + 0.2, 2.27], [cG, 2.05], [cG - 0.2, 2.27]], closed=True, fc=C["r_edge"], ec="none", zorder=5))
ax.plot([cG - 0.09, cG - 0.02, cG + 0.1], [2.32, 2.24, 2.42], color="white", lw=2.0, solid_capstyle="round", zorder=6)
for k, name in enumerate(("schema", "x-ref IDs seen", "no repeats", "retry cap: 2 empty")):
    yy = 1.78 - k * 0.3
    ax.add_patch(Circle((XG + 0.22, yy), 0.065, fc=C["g_edge"], ec="none", zorder=5))
    ax.text(XG + 0.37, yy, name, ha="left", va="center", fontsize=10.5, color=C["dark"], zorder=5)
arrow(ax, cV, 3.45 - GAP, cG, 2.75 + GAP, C["p_edge"], width=0.24, head_w=0.48, head_l=0.2)

XE, WE = 9.15, 4.75
gbox(ax, XE, 0.45, WE, 2.3, AMBER_F, AMBER, r=0.22)
cE = XE + WE / 2
rich(ax, cE, 2.48, [("ToolEnv", {"weight": "bold", "color": AMBER_T}), ("  one per sample, every call counts", {"color": AMBER_T, "style": "italic"})], size=12)
turns = [([("bill_payment.list_bills()", M)], [("  điện ", G), ("845.000đ", B), (", bill_02", G)]),
         ([("ewallet.get_balance()", M)], [("  ", {}), ("713.606đ", {**B, "color": C["r_edge"]}), (" < 845.000đ", G)]),
         ([("bill_payment.pay(bill_02, ", M), ('source="bank"', {**M, **B, "color": C["g_text"]}), (")", M)], [])]
for k, (call, res) in enumerate(turns):
    yy = 2.07 - k * 0.52
    ax.add_patch(Circle((XE + 0.25, yy), 0.13, fc=C["g_edge"] if k == 2 else AMBER, ec="white", lw=1.0, zorder=5))
    ax.text(XE + 0.25, yy, str(k + 1), ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=6)
    rich(ax, XE + 0.47, yy, call + res, ha="left", size=10.2)
ax.text(cE, 0.66, "stop when the vote says: no call", ha="center", va="center", fontsize=10.5, style="italic", color=AMBER_T, zorder=5)
arrow(ax, XG - GAP, ya, XE + WE + GAP, ya, C["r_edge"])

# agent loop: tool result climbs back to the LoRAs for the next turn
LX = XL + 2.05
ax.plot([LX, LX], [2.75 + GAP, 3.3 - 0.02], color=C["loop"], lw=1.8, ls=(0, (4, 2.5)), zorder=1)
ax.add_patch(FancyArrowPatch((LX, 3.05), (LX, 3.62), arrowstyle="-|>", mutation_scale=15, color=C["loop"], lw=1.8, zorder=3))
rich(ax, LX - 0.12, 3.05, [("tool result", B), (" → next turn", {})], ha="right", size=11, color=C["loop"], style="italic")

# output
XO, WO = 6.2, 2.45
gbox(ax, XO, 1.25, WO, 1.5, C["g_fill"], C["g_edge"], r=0.2)
cO = XO + WO / 2
ax.plot([cO - 0.17, cO - 0.05, cO + 0.18], [2.48, 2.34, 2.62], color=C["g_edge"], lw=3.0, solid_capstyle="round", zorder=5)
ax.text(cO, 2.0, "3 calls, in order", ha="center", va="center", fontsize=12, fontweight="bold", color=C["g_text"], zorder=5)
ax.text(cO, 1.62, "+ state_hash", ha="center", va="center", fontsize=10.5, color=C["grey"], family="Liberation Mono", zorder=5)
gbox(ax, XO, 0.45, WO, 0.62, C["p_fill"], C["p_edge"], lw=1.2, r=0.15)
rich(ax, cO, 0.76, [("missing slot → ", {"color": C["p_text"]}), ("ask back", {"weight": "bold", "color": C["p_text"]})], size=11)
arrow(ax, XE - GAP, 2.0, XO + WO + GAP, 2.0, AMBER, width=0.22, head_w=0.44, head_l=0.18)
ax.add_patch(FancyArrowPatch((XE - GAP, 0.76), (XO + WO + GAP, 0.76), arrowstyle="-|>", mutation_scale=12, color=C["p_edge"], lw=1.4, zorder=3))

# scoring rules of the kit (left of the lane)
ax.text(1.0, 2.45, "Scored on", ha="left", va="center", fontsize=12, fontweight="bold", color=C["dark"])
for k, parts in enumerate(([("final state", B), (" = gold state, then", G)], [("call F1", B), (" over the ordered calls", G)],
                           [("≤ 30 s", B), (" per sample, offline", G)], [("LLM < 5B", B), (", total < 6B params", G)])):
    ax.add_patch(Circle((1.12, 2.07 - k * 0.38), 0.05, fc=AMBER, ec="none", zorder=5))
    rich(ax, 1.27, 2.07 - k * 0.38, parts, ha="left", size=11)

save(fig, "pipeline_v2")
