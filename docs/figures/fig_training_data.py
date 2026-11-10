"""How one SFT row is built and which tokens carry loss (figures4papers style, Track 1 deck v2).
Real row: train_0001, ASR copy, with an injected failed search (scripts/inject_recovery.py) and the free-text weight of
train_agent_sft_q35.py --fuzzy-weight 0.3. Run: python docs/figures/fig_training_data.py -> training_data.{png,pdf,svg}"""
from matplotlib.patches import FancyBboxPatch, Rectangle

from style import C, arrow, box, gbox, mix, rich, save, setup

W, H = 16.0, 6.3
fig, ax = setup(W, H)
B, G, I = {"weight": "bold"}, {"color": C["grey"]}, {"style": "italic"}
M = {"family": "Liberation Mono"}
GAP = 0.12

# ---- top: from a TRAIN sample to three kinds of rows --------------------------------------------------------------------------------
steps = [("TRAIN sample", "1,545 requests", C["g_fill"], C["g_edge"], C["g_text"]),
         ("replay in ToolEnv", "run the correct calls, keep the results", "#FBE3B6", "#B07A12", "#7A520A"),
         ("chat example", "same format as at test time", C["p_fill"], C["p_edge"], C["p_text"])]
x = 0.3
for k, (t, sub, fc, ec, tc) in enumerate(steps):
    gbox(ax, x, 4.75, 2.85, 1.2, fc, ec, r=0.2)
    ax.text(x + 1.425, 5.55, t, ha="center", va="center", fontsize=14, fontweight="bold", color=tc, zorder=5)
    ax.text(x + 1.425, 5.18, sub, ha="center", va="center", fontsize=11, style="italic", color=tc, zorder=5)
    if k < 2:
        arrow(ax, x + 2.85 + GAP, 5.35, x + 3.45 - GAP, 5.35, ec, width=0.22, head_w=0.44, head_l=0.18)
    x += 3.45
copies = [("true text", "1,545 copies", C["g_text"]), ("ASR text", "1,545 copies", C["b_text"]),
          ("recovery", "184 copies with a failed search", C["r_edge"])]
X0 = 10.85
arrow(ax, 10.15 + GAP, 5.35, X0 - GAP, 5.35, C["p_edge"], width=0.22, head_w=0.44, head_l=0.18)
for k, (t, sub, col) in enumerate(copies):
    yy = 5.75 - k * 0.42
    box(ax, X0, yy - 0.17, 4.85, 0.34, "white", col, lw=1.1, r=0.1, z=3)
    rich(ax, X0 + 0.15, yy, [(t, {"weight": "bold", "color": col}), ("   " + sub, G)], ha="left", size=12)

# ---- bottom: one row as a token strip --------------------------------------------------------------------------------------------
ax.text(0.3, 4.25, "One example (train_0001): which parts the model learns from", ha="left", va="center",
        fontsize=13.5, fontweight="bold", color=C["dark"])
segs = [  # (label lines, width, kind)
    (["system + tool schemas", "≈ 5k tokens"], 3.05, "ctx"),
    (["request", "“nhắn xếp dũng …”"], 1.55, "ctx"),
    (["calendar + weather", "calls (3 turns)"], 1.55, "call"),
    (["search(", "\"xếp dũng\")"], 1.3, "wrong"),
    (["result", "contacts: []"], 1.05, "ctx"),
    (["search(", "\"dũng\")"], 1.1, "call"),
    (["result", "c_08 Sếp Dũng"], 1.15, "ctx"),
    (["sms.send(c_08,", "content=…)"], 1.55, "mixed"),
    (["result"], 0.75, "ctx"),
    (["stop"], 0.62, "call"),
]
style = {"ctx": ("#EDEDED", C["grey"], C["grey"]), "call": (C["g_fill"], C["g_edge"], C["g_text"]),
         "wrong": (C["r_fill"], C["r_edge"], C["r_edge"]), "mixed": (C["g_fill"], C["g_edge"], C["g_text"])}
x, y0, h = 0.3, 2.55, 1.2
for lines, w, kind in segs:
    fc, ec, tc = style[kind]
    if kind == "mixed":   # the ID part weight 1, the free-text content weight 0.3
        ax.add_patch(Rectangle((x, y0), w * 0.45, h, fc=C["g_fill"], ec="none", zorder=2))
        ax.add_patch(Rectangle((x + w * 0.45, y0), w * 0.55, h, fc=mix(C["g_fill"], 0.65), ec="none", zorder=2))
        ax.add_patch(Rectangle((x, y0), w, h, fc="none", ec=ec, lw=1.2, zorder=3))
    elif kind == "wrong":
        ax.add_patch(Rectangle((x, y0), w, h, fc=fc, ec=mix(ec, 0.55), lw=0, hatch="///", zorder=2))
        ax.add_patch(Rectangle((x, y0), w, h, fc="none", ec=ec, lw=1.2, zorder=3))
    else:
        ax.add_patch(Rectangle((x, y0), w, h, fc=fc, ec=ec if kind != "ctx" else mix(C["grey"], 0.4), lw=1.0, zorder=2))
    for k, line in enumerate(lines):
        yy = y0 + h / 2 + (0.17 if len(lines) == 2 else 0) - k * 0.34
        ax.text(x + w / 2, yy, line, ha="center", va="center", fontsize=12 if k == 0 else 11, color=tc,
                fontweight="bold" if k == 0 and kind != "ctx" else "normal", family="Liberation Mono" if kind in ("call", "wrong", "mixed") and "(" in line + lines[0] else None, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9) if kind == "wrong" else None)
    x += w + 0.04
# weight labels under the strip
marks = [(0.3, 0.3 + 3.05 + 1.55, "not learned: the input", C["grey"]), (6.75, 8.05, "not learned: the planted mistake", C["r_edge"]),
         (11.3, 12.85, "ID fully, message text at 0.3", C["g_text"])]
for a, b, text, col in marks:
    ax.plot([a, a, b, b], [2.45, 2.32, 2.32, 2.45], color=col, lw=1.3, zorder=3)
    ax.text((a + b) / 2, 2.1, text, ha="center", va="center", fontsize=12.5, color=col, style="italic")

# legend + numbers
legend = [("#EDEDED", "not learned: input, tool results", None), (C["g_fill"], "learned: calls, IDs, stop", None),
          (mix(C["g_fill"], 0.65), "learned at 0.3: message text", None), (C["r_fill"], "not learned: planted failed call", "///")]
for k, (fc, text, hatch) in enumerate(legend):
    xx = 0.3 + k * 3.95
    ax.add_patch(Rectangle((xx, 1.35), 0.42, 0.28, fc=fc, ec=C["grey"] if not hatch else C["r_edge"], lw=0.9, hatch=hatch or "", zorder=3))
    ax.text(xx + 0.55, 1.49, text, ha="left", va="center", fontsize=12.5, color=C["dark"])
ax.set_ylim(1.15, H)             # the facts rows moved to the slides; crop the empty band
fig.set_size_inches(W, H - 1.15)
save(fig, "training_data")
