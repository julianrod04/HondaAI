import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

CSV_PATH  = Path(__file__).parent / "pipeline_runs" / "participant_20_log.csv"
OUT_PATH  = Path(__file__).parent / "pipeline_runs" / "participant_20_colorblind.png"

TYPE_STYLE = {
    "arrow": ("o", "#e06c3b"),
    "route": ("s", "#4a90d9"),
}

def load_visual_alerts(path):
    """Return (iters, cbs, markers, colors) for arrow/route rows in a CSV."""
    its, cbs_, mks, cls_ = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "gui_type" not in row or "gui_color" not in row:
                continue
            gt = row["gui_type"].strip().lower()
            if gt not in TYPE_STYLE:
                continue
            its.append(int(row["iteration"]))
            cbs_.append(1 if row["gui_color"].strip().lower() == "colorblind" else 0)
            mks.append(TYPE_STYLE[gt][0])
            cls_.append(TYPE_STYLE[gt][1])
    return its, cbs_, mks, cls_

iters, cbs, markers, colors = load_visual_alerts(CSV_PATH)

# ── Even-numbered comparison participants (excluding p20, skip empty files) ──
RUNS_DIR = CSV_PATH.parent
from collections import defaultdict
_other_by_iter = defaultdict(list)   # iter -> [cb_values from other participants]
for _pnum in range(0, 100, 2):
    if _pnum == 20:
        continue
    _p = RUNS_DIR / f"participant_{_pnum}_log.csv"
    if not _p.exists():
        continue
    _its, _cbs, _, _ = load_visual_alerts(_p)
    if not _its:
        continue
    for _i, _cb in zip(_its, _cbs):
        _other_by_iter[_i].append(_cb)

# Per-iteration mean across all contributing even participants
_other_iters = np.array(sorted(_other_by_iter.keys()))
_other_means  = np.array([np.mean(_other_by_iter[i]) for i in _other_iters])

iters  = np.array(iters)
cbs    = np.array(cbs, dtype=float)

def running_mean(xs, ys, window=3):
    order = np.argsort(xs)
    xs_s, ys_s = xs[order], ys[order]
    k = min(window, len(ys_s))
    rm = np.convolve(ys_s, np.ones(k) / k, mode="valid")
    xs_rm = xs_s[k // 2: k // 2 + len(rm)]
    return xs_rm, rm

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Participant 20 — Colorblind-Friendly Color Setting Over Time\n"
             "(Arrow & Projected Route only; sound excluded)",
             fontsize=12, fontweight="bold")

rng = np.random.default_rng(0)
jitter = (rng.random(len(cbs)) - 0.5) * 0.06

for m in set(markers):
    idx = [i for i, mk in enumerate(markers) if mk == m]
    label = "Arrow" if m == "o" else "Projected Route"
    color = TYPE_STYLE["arrow"][1] if m == "o" else TYPE_STYLE["route"][1]
    ax.scatter(iters[idx], cbs[idx] + jitter[idx],
               marker=m, color=color, s=100, zorder=3, label=label)

rm_x, rm_y = running_mean(iters, cbs, window=3)
if len(rm_x) >= 2:
    ax.plot(rm_x, rm_y, color="black", linewidth=2, linestyle="--",
            alpha=0.75, label="P20 rolling mean (w=3)")

# Even-participant comparison line (no scatter points)
_n_others = len({k for k in _other_by_iter})
if len(_other_iters) >= 2:
    _orm_x, _orm_y = running_mean(_other_iters, _other_means, window=3)
    if len(_orm_x) >= 2:
        ax.plot(_orm_x, _orm_y, color="#5a9e6f", linewidth=2, linestyle=":",
                alpha=0.85,
                label=f"Even participants avg (excl. P20, n={len(_other_by_iter.keys())} iters)")

n_cb  = int(cbs.sum())
n_tot = len(cbs)
pct   = 100 * n_cb / n_tot if n_tot else 0
ax.set_title(f"{n_cb}/{n_tot} visual-alert episodes used colorblind mode ({pct:.0f}%)",
             fontsize=10, style="italic")
ax.set_yticks([0, 1])
ax.set_yticklabels(["RGB", "Colorblind"], fontsize=11)
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("Iteration", fontsize=11)
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
ax.legend(fontsize=10)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT_PATH), dpi=150)
print(f"Saved -> {OUT_PATH}")
