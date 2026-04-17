import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RUNS_DIR = Path(__file__).parent / "pipeline_runs"
CSV_P8   = RUNS_DIR / "participant_8_log.csv"
OUT_PATH = RUNS_DIR / "participant_8_sound.png"


def load_sound(path):
    """Return (iters, is_sound) for all rows that have a gui_type column."""
    its, vals = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "gui_type" not in row:
                continue
            its.append(int(row["iteration"]))
            vals.append(1 if row["gui_type"].strip().lower() == "sound" else 0)
    return np.array(its), np.array(vals, dtype=float)


def running_mean(xs, ys, window=3):
    if len(xs) < 2:
        return xs, ys
    order = np.argsort(xs)
    xs_s, ys_s = xs[order], ys[order]
    k = min(window, len(ys_s))
    rm = np.convolve(ys_s, np.ones(k) / k, mode="valid")
    xs_rm = xs_s[k // 2: k // 2 + len(rm)]
    return xs_rm, rm


# ── Participant 8 ─────────────────────────────────────────────────────────────
p8_iters, p8_sound = load_sound(CSV_P8)

# ── Even participants comparison (excl. P8) ───────────────────────────────────
other_by_iter = defaultdict(list)
for pnum in range(0, 100, 2):
    if pnum == 8:
        continue
    p = RUNS_DIR / f"participant_{pnum}_log.csv"
    if not p.exists():
        continue
    its, vals = load_sound(p)
    if len(its) == 0:
        continue
    for i, v in zip(its, vals):
        other_by_iter[i].append(v)

other_iters = np.array(sorted(other_by_iter.keys()))
other_means  = np.array([np.mean(other_by_iter[i]) for i in other_iters])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
n_sound = int(p8_sound.sum())
n_tot   = len(p8_sound)
pct     = 100 * n_sound / n_tot if n_tot else 0

fig.suptitle("Participant 8 — Sound Alert Frequency Over Time",
             fontsize=13, fontweight="bold")
ax.set_title(f"{n_sound}/{n_tot} episodes used sound ({pct:.0f}%)",
             fontsize=10, style="italic")

rng    = np.random.default_rng(1)
jitter = (rng.random(len(p8_sound)) - 0.5) * 0.06
ax.scatter(p8_iters, p8_sound + jitter, color="#e06c3b", s=90,
           zorder=3, label="P8 episode (sound=1, other=0)")

rm_x, rm_y = running_mean(p8_iters, p8_sound, window=3)
if len(rm_x) >= 2:
    ax.plot(rm_x, rm_y, color="black", linewidth=2, linestyle="--",
            alpha=0.8, label="P8 rolling mean (w=3)")

n_others = len([k for k in other_by_iter])
orm_x, orm_y = running_mean(other_iters, other_means, window=3)
if len(orm_x) >= 2:
    ax.plot(orm_x, orm_y, color="#4a90d9", linewidth=2, linestyle=":",
            alpha=0.85,
            label=f"Even participants avg (excl. P8, {n_others} iters)")

ax.set_yticks([0, 1])
ax.set_yticklabels(["Other (Arrow/Route)", "Sound"], fontsize=11)
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("Iteration", fontsize=11)
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
ax.legend(fontsize=10)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT_PATH), dpi=150)
print(f"Saved -> {OUT_PATH}")
