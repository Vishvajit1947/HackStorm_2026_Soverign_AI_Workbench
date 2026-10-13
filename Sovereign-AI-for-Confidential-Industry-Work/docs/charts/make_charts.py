"""Explanatory charts for the Kavach knowledge/retrieval + router internals.
Numbers are taken from the project: config/router.yaml (weights + worked example),
index.py (BM25 saturation/IDF behaviour), retrieval.py (RRF fusion)."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- dataviz skill: validated categorical palette (light mode) ---------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE = "#fcfcfb"
INK   = "#0b0b0b"
INK2  = "#52514e"
GRID  = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 11, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

def style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

# =============================================================== FIGURE 1
# BM25 term-frequency saturation vs naive counting  (the question just asked)
fig, ax = plt.subplots(figsize=(8.2, 5.0))
tf = np.linspace(0, 30, 300)
k1 = 1.5                       # BM25 default saturation
bm25_tf = (tf * (k1 + 1)) / (tf + k1)   # TF component of BM25 (b=0 simplification)
naive = tf / tf.max() * bm25_tf.max()   # linear "count every time" line, scaled

ax.plot(tf, naive, color=INK2, lw=2, ls=(0, (5, 3)), label="Naive: count every occurrence")
ax.plot(tf, bm25_tf, color=BLUE, lw=2.6, label="BM25: saturating (k1 = 1.5)")

for x in (1, 3, 10):
    y = (x * (k1 + 1)) / (x + k1)
    ax.plot([x], [y], "o", ms=8, color=BLUE, zorder=5)
    ax.annotate(f"{x}x -> {y:.2f}", (x, y), textcoords="offset points",
                xytext=(8, -14), color=INK, fontsize=10)

ax.set_title("BM25 gives repeated words diminishing returns",
             fontsize=14, fontweight="bold", color=INK, pad=30)
ax.text(0, 1.05, "A word repeated 10x is worth far less than 10x a single hit - the curve flattens",
        transform=ax.transAxes, fontsize=10.5, color=INK2)
ax.set_xlabel("Times the word appears in one chunk (term frequency)")
ax.set_ylabel("Score contribution")
ax.set_xlim(0, 30); ax.set_ylim(0, bm25_tf.max() * 1.15)
ax.legend(frameon=False, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig("chart1_bm25_saturation.png", dpi=150)
plt.close(fig)

# =============================================================== FIGURE 2
# IDF: a word in many documents is worth less (rare tag >> common word)
fig, ax = plt.subplots(figsize=(8.2, 5.0))
N = 11                                  # 11-page MRPL demo corpus
df = np.arange(1, N + 1)                 # document frequency
idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
bars = ax.bar(df, idf, color=BLUE, width=0.66)
# highlight the two ends with meaning
bars[0].set_color(AQUA)                 # rare -> valuable
bars[-1].set_color(ORANGE)              # common -> cheap
ax.annotate("rare tag e.g. PSLL-101\n(in 1 doc) -> most weight",
            (1, idf[0]), textcoords="offset points", xytext=(6, -4),
            fontsize=9.5, color=INK)
ax.annotate("common word e.g. 'pump'\n(in most docs) -> least weight",
            (N, idf[-1]), textcoords="offset points", xytext=(-8, 30),
            ha="right", fontsize=9.5, color=INK)
ax.set_title("IDF: the rarer the word, the more it counts",
             fontsize=14, fontweight="bold", color=INK, pad=30)
ax.text(0, 1.05, "Why BM25 finds the exact instrument tag an engineer typed, not the boilerplate around it",
        transform=ax.transAxes, fontsize=10.5, color=INK2)
ax.set_xlabel("Number of documents (of 11) containing the word")
ax.set_ylabel("IDF weight")
ax.set_xticks(df)
style(ax)
fig.tight_layout()
fig.savefig("chart2_idf.png", dpi=150)
plt.close(fig)

# =============================================================== FIGURE 3
# Router utility score breakdown (stacked +/- from router.yaml worked example)
# coder-7b (cold) 0.88q  vs  qwen3-8b (resident) 0.68q
fig, ax = plt.subplots(figsize=(8.6, 5.2))
models = ["coder-7b\n(cold, must load)", "qwen3-8b\n(already resident)"]
# terms as they appear in the worked example comment in router.yaml
quality   = [0.88, 0.68]
latency   = [-0.095, -0.106]
swap      = [-0.20, 0.00]
vram      = [-0.047, -0.055]
totals    = [0.538, 0.519]

x = np.arange(len(models))
w = 0.5
# positive base = quality
ax.bar(x, quality, w, color=AQUA, label="+ quality")
# stack the negatives downward from 0
neg_terms = [("- latency", latency, ORANGE),
             ("- swap/reload", swap, RED),
             ("- vram pressure", vram, VIOLET)]
bottom = np.zeros(len(models))
for name, vals, col in neg_terms:
    ax.bar(x, vals, w, bottom=bottom, color=col, label=name)
    bottom = bottom + np.array(vals)

# net score markers
ax.plot(x, totals, "D", ms=12, color=INK, zorder=6)
for i, t in enumerate(totals):
    ax.annotate(f"net {t:.3f}", (x[i], t), textcoords="offset points",
                xytext=(16, 0), va="center", fontweight="bold", color=INK)
# winner
ax.annotate("WINS", (0, quality[0]), textcoords="offset points", xytext=(0, 6),
            ha="center", va="bottom", fontweight="bold", color=AQUA, fontsize=13)

ax.axhline(0, color=INK2, lw=1)
ax.set_title("How the router scores two models for one request",
             fontsize=14, fontweight="bold", color=INK, pad=30)
ax.text(0, 1.05, "A 20-point quality lead beats a 4.2s reload penalty - so the cold coder model still wins",
        transform=ax.transAxes, fontsize=10.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Contribution to utility score")
ax.set_ylim(-0.35, 1.0)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
style(ax)
fig.tight_layout()
fig.savefig("chart3_router_score.png", dpi=150)
plt.close(fig)

# =============================================================== FIGURE 4
# Hybrid retrieval: RRF fuses two ranked lists by POSITION, not raw score
fig, ax = plt.subplots(figsize=(8.6, 5.2))
RRF_K = 60                               # retrieval.py default
# a small illustrative scenario: 5 chunks, their rank in each arm (None = absent)
chunks = ["C1", "C2", "C3", "C4", "C5"]
bm25_rank = {"C1": 1, "C2": 2, "C3": 4, "C4": None, "C5": 3}
dense_rank = {"C1": 3, "C2": 1, "C3": None, "C4": 2, "C5": 4}
def rrf(r):
    return 0 if r is None else 1.0 / (RRF_K + r)
fused = {c: rrf(bm25_rank[c]) + rrf(dense_rank[c]) for c in chunks}
order = sorted(chunks, key=lambda c: -fused[c])

y = np.arange(len(order))[::-1]
b_contrib = [rrf(bm25_rank[c]) for c in order]
d_contrib = [rrf(dense_rank[c]) for c in order]
ax.barh(y, b_contrib, color=BLUE, label="BM25 arm (lexical / exact tags)")
ax.barh(y, d_contrib, left=b_contrib, color=MAGENTA, label="Dense arm (meaning)")
for i, c in enumerate(order):
    br = bm25_rank[c]; dr = dense_rank[c]
    tag = f"BM25 #{br if br else '-'} | Dense #{dr if dr else '-'}"
    ax.annotate(tag, (fused[c], y[i]), textcoords="offset points",
                xytext=(8, 0), va="center", fontsize=9.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(order, fontweight="bold")
ax.set_title("Reciprocal Rank Fusion: two arms combined by rank, not score",
             fontsize=14, fontweight="bold", color=INK, pad=30)
ax.text(0, 1.06, "C2 wins: strong in both arms. Neither BM25's big raw numbers nor cosine can swamp the other",
        transform=ax.transAxes, fontsize=10.5, color=INK2)
ax.set_xlabel("Fused RRF score  =  1/(60+rank_bm25) + 1/(60+rank_dense)")
ax.set_xlim(0, max(fused.values()) * 1.35)
ax.legend(frameon=False, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig("chart4_rrf_fusion.png", dpi=150)
plt.close(fig)

print("wrote 4 charts")
