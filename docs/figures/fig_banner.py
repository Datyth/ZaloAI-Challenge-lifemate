"""README banner: speech waveform, a checked tool call, the project name. Text is saved as paths, so GitHub renders it the same
everywhere. Run: python docs/figures/fig_banner.py -> docs/assets/banner.{svg,png}"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "assets"
INK, GREY, BLUE, PURPLE, AMBER, GREEN = "#1F2A37", "#6B7280", "#2F5D8A", "#5E3B6E", "#B07A12", "#3F7D2A"
plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans"})
W, H = 12.8, 3.2
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.08, 0.08), W - 0.16, H - 0.16, boxstyle="round,pad=0,rounding_size=0.28", fc="#F6F8FB", ec="#DDE3EC", lw=1.4))

# waveform
rng = np.random.default_rng(7)
env = np.sin(np.linspace(0.2, np.pi - 0.2, 26)) ** 0.8
for k, a in enumerate(env * (0.35 + 0.65 * rng.random(26))):
    x = 0.75 + k * 0.075
    ax.plot([x, x], [1.6 - 0.62 * a, 1.6 + 0.62 * a], color=BLUE, lw=3.2, solid_capstyle="round", alpha=0.55 + 0.45 * a)
ax.add_patch(FancyArrowPatch((2.85, 1.6), (3.5, 1.6), arrowstyle="-|>", mutation_scale=18, color=GREY, lw=1.6))

# a tool call card with a check
ax.add_patch(FancyBboxPatch((3.65, 0.9), 2.9, 1.4, boxstyle="round,pad=0,rounding_size=0.16", fc="white", ec=PURPLE, lw=1.6))
for k, (text, col) in enumerate((("contacts.search(\"Dũng\")", AMBER), ("sms.send(c_08, ...)", GREEN))):
    ax.text(3.88, 1.92 - k * 0.5, text, ha="left", va="center", fontsize=12.5, family="DejaVu Sans Mono", color=col)
ax.plot([6.12, 6.24, 6.45], [1.4, 1.27, 1.55], color=GREEN, lw=3.2, solid_capstyle="round")

# name and tagline
ax.text(7.05, 2.05, "LifeMate", ha="left", va="center", fontsize=44, fontweight="bold", color=INK)
ax.text(7.08, 1.28, "Spoken Vietnamese requests, turned into", ha="left", va="center", fontsize=15.5, color=INK)
ax.text(7.08, 0.92, "verified tool calls on the user's apps", ha="left", va="center", fontsize=15.5, color=INK)
ax.text(7.08, 0.5, "Zalo AI Challenge 2026  |  Track 1", ha="left", va="center", fontsize=11.5, color=GREY)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "banner.svg", transparent=True)
fig.savefig(OUT / "banner.png", dpi=200, transparent=True)
print("saved", OUT / "banner.svg")
