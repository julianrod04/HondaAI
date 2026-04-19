import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "pipeline_runs"

# ── Load all participants ─────────────────────────────────────────────────────
def load_participant(path):
    iters, scores, losses = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                iters.append(int(row["iteration"]))
                scores.append(float(row["driving_score"]))
                losses.append(float(row["episode_loss"]))
            except (KeyError, ValueError):
                continue
    return np.array(iters), np.array(scores), np.array(losses)

fixed_data    = {}
adaptive_data = {}
EXCLUDE = {64}

for f in sorted(RUNS_DIR.glob("participant_*_log.csv")):
    pnum = int(f.stem.split("_")[1])
    if pnum in EXCLUDE:
        continue
    iters, scores, losses = load_participant(f)
    if len(iters) == 0:
        continue
    mask = losses <= 60.0
    iters, scores, losses = iters[mask], scores[mask], losses[mask]
    if len(iters) == 0:
        continue
    if pnum % 2 == 1:
        fixed_data[pnum] = (iters, scores, losses)
    else:
        adaptive_data[pnum] = (iters, scores, losses)

def make_colors(n):
    return [cm.tab20(i / max(n - 1, 1)) for i in range(n)]

def _plot_into(data, title_tag, fig_num, ax_score, ax_loss):
    pnums     = sorted(data.keys())
    color_map = {p: c for p, c in zip(pnums, make_colors(len(pnums)))}

    score_by_iter, loss_by_iter = {}, {}

    for pnum in pnums:
        iters, scores, losses = data[pnum]
        order = np.argsort(iters)
        xi, xs, xl = iters[order], scores[order], losses[order]
        if len(xs) > 1:
            keep = np.arange(len(xi)) != np.argmin(xs)
            xi, xs, xl = xi[keep], xs[keep], xl[keep]
        c   = color_map[pnum]
        lbl = f"P{pnum}"

        ax_score.scatter(xi, xs, color=c, s=35, zorder=3, alpha=0.85)
        ax_score.plot(xi, xs, color=c, linewidth=0.8, alpha=0.4)
        ax_loss.scatter(xi, xl, color=c, s=35, zorder=3, alpha=0.85, label=lbl)
        ax_loss.plot(xi, xl, color=c, linewidth=0.8, alpha=0.4)

        for i, s, l in zip(xi, xs, xl):
            score_by_iter.setdefault(i, []).append(s)
            loss_by_iter.setdefault(i, []).append(l)

    for by_iter, ax in [(score_by_iter, ax_score), (loss_by_iter, ax_loss)]:
        if not by_iter:
            continue
        mx = np.array(sorted(by_iter.keys()))
        my = np.array([np.mean(by_iter[i]) for i in mx])
        ax.plot(mx, my, color="black", linewidth=2.0, zorder=6, label="Mean")

    ax_score.set_title(f"{fig_num}  —  {title_tag} Alert Mode", fontsize=10, fontweight="bold")
    ax_score.set_ylabel("Human Driving Score", fontsize=9)
    ax_score.set_ylim(0.0, 0.6)
    ax_score.set_xlim(-0.5, 19.5)
    ax_score.set_xticks(range(0, 20))
    ax_score.tick_params(labelsize=7)
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(["Mean"], fontsize=7, loc="upper right")

    ax_loss.set_ylabel("Alert Loss (m)", fontsize=9)
    ax_loss.set_ylim(0, 60)
    ax_loss.set_xlim(-0.5, 19.5)
    ax_loss.set_xticks(range(0, 20))
    ax_loss.tick_params(labelsize=7)
    ax_loss.set_xlabel("Iteration", fontsize=9)
    ax_loss.grid(True, alpha=0.25)

    handles, labels = ax_loss.get_legend_handles_labels()
    pairs = sorted(zip(labels, handles),
                   key=lambda x: (x[0] != "Mean", int(x[0][1:]) if x[0] != "Mean" else 0))
    labels_s, handles_s = zip(*pairs) if pairs else ([], [])
    ax_loss.legend(handles_s, labels_s, title="Participant", title_fontsize=7,
                   fontsize=6, ncol=2, loc="upper right", framealpha=0.85)

# ── Combined figure: 2 columns × 2 rows ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                         gridspec_kw={"hspace": 0.42, "wspace": 0.30})
fig.suptitle("Fixed (left) vs Adaptive (right) Alert Mode",
             fontsize=13, fontweight="bold")

_plot_into(fixed_data,    "Fixed",    "Figure 1", axes[0][0], axes[1][0])
_plot_into(adaptive_data, "Adaptive", "Figure 2", axes[0][1], axes[1][1])

plt.savefig(str(RUNS_DIR / "master_combined.png"), dpi=150, bbox_inches="tight")
print(f"Saved -> {RUNS_DIR / 'master_combined.png'}")

# Keep individual files too
for data, tag, fname in [
    (fixed_data,    "Fixed",    "master_fixed.png"),
    (adaptive_data, "Adaptive", "master_adaptive.png"),
]:
    fig2, (as_, al_) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                     gridspec_kw={"hspace": 0.35})
    _plot_into(data, tag, "Figure 1" if tag == "Fixed" else "Figure 2", as_, al_)
    fig2.suptitle(f"Master Scatter — {tag} Alert Mode — All Participants",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(RUNS_DIR / fname), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {RUNS_DIR / fname}")
