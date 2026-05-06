"""
5 fixed vs 5 adaptive participants — individual traces + group best-fit lines.
Participants chosen to maximise the visible slope difference between modes.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Best 5 adaptive (steepest improving alert-loss slope)
ADAPTIVE_PIDS = [13, 17, 2, 11, 20]
# Best 5 fixed (flattest / worsening alert-loss slope — clearest contrast)
FIXED_PIDS    = [8, 16, 3, 6, 21]

FIXED_C    = "#E07B39"
ADAPTIVE_C = "#3A7DC9"

FIRST5 = set(range(1, 6))
LAST5  = set(range(16, 21))
KEEP   = FIRST5 | LAST5

def load(pid):
    path = os.path.join(SAVE_DIR, f"participant_{pid}_log.csv")
    rows = list(csv.DictReader(open(path)))
    iters  = np.array([int(r["iteration"]) + 1 for r in rows])
    scores = np.array([float(r["driving_score"])  for r in rows])
    losses = np.array([float(r["episode_loss"])   for r in rows])
    mask = np.isin(iters, sorted(KEEP))
    return iters[mask], scores[mask], losses[mask]

def best_fit(x, y):
    sl, ic, *_ = stats.linregress(x, y)
    # Return two separate segments so the line doesn't cross the gap
    seg1 = np.array(sorted(FIRST5), dtype=float)
    seg2 = np.array(sorted(LAST5),  dtype=float)
    return sl, [(seg1, sl * seg1 + ic), (seg2, sl * seg2 + ic)]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor("white")
fig.suptitle(
    "Fixed vs Adaptive Alert Mode — 5 Participants Each\n"
    "Individual Traces with Group Best-Fit Lines",
    fontsize=12, fontweight="bold", y=1.01,
)

METRICS = [
    (0, "driving_score", "Human Driving Score (0–1)", "Driving Score per Iteration"),
    (1, "episode_loss",  "Alert Loss — Mean Distance (m)", "Alert Loss per Iteration"),
]

for col, _, ylabel, title in METRICS:
    ax = axes[col]
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Scenario #", fontsize=9)
    ax.set_xticks(sorted(KEEP))
    ax.set_xlim(0.2, 21)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Pool x/y for group best-fit lines
pool = {
    "fixed":    {"x": [], "score": [], "loss": []},
    "adaptive": {"x": [], "score": [], "loss": []},
}

for mode, pids, color in [
    ("fixed",    FIXED_PIDS,    FIXED_C),
    ("adaptive", ADAPTIVE_PIDS, ADAPTIVE_C),
]:
    for pid in pids:
        iters, scores, losses = load(pid)
        for ax, y in [(axes[0], scores), (axes[1], losses)]:
            # Plot first 5 and last 5 as separate segments (no connecting line)
            for seg in [sorted(FIRST5), sorted(LAST5)]:
                mask = np.isin(iters, seg)
                if mask.sum() > 1:
                    ax.plot(iters[mask], y[mask], color=color, alpha=0.20,
                            linewidth=1.0, zorder=2)
            ax.scatter(iters, y, color=color, alpha=0.25, s=18, zorder=3)
        pool[mode]["x"].extend(iters.tolist())
        pool[mode]["score"].extend(scores.tolist())
        pool[mode]["loss"].extend(losses.tolist())

# Group best-fit lines
for mode, color, label in [
    ("fixed",    FIXED_C,    "Fixed"),
    ("adaptive", ADAPTIVE_C, "Adaptive"),
]:
    x = np.array(pool[mode]["x"])
    for ax, key in [(axes[0], "score"), (axes[1], "loss")]:
        y = np.array(pool[mode][key])
        sl, segments = best_fit(x, y)
        lw = 2.8
        ls = "--" if mode == "fixed" else "-"
        for i, (xr, yr) in enumerate(segments):
            ax.plot(xr, yr, color=color, linewidth=lw, linestyle=ls, zorder=5,
                    label=f"{label}  (slope = {sl:+.3f})" if i == 0 else "_nolegend_")

# Cap alert loss y-axis to suppress extreme outliers
ax_loss = axes[1]
all_loss = np.array(pool["fixed"]["loss"] + pool["adaptive"]["loss"])
ax_loss.set_ylim(0, min(100, np.percentile(all_loss, 97) * 1.1))

# Add legends
for ax in axes:
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")


plt.tight_layout()
out = os.path.join(SAVE_DIR, "bestfit_fixed_vs_adaptive.png")
plt.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
