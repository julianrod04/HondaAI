"""
plot_report.py — Generate report-ready figures from pipeline_runs session data.

Run:
    python plot_report.py

Outputs (saved to pipeline_runs/report/):
    fig1_participant_grid.png   — Per-participant score & loss curves (grid)
    fig2_cohort_learning.png    — Mean ± 1-SD learning curves, Fixed vs Adaptive
    fig3_alert_types.png        — Alert type usage per participant + overall pie
    fig4_style_profiles.png     — Mean driving-style dimensions per participant
    fig5_scenario_heatmap.png   — Avg driving score: participant × scenario
    fig6_final_performance.png  — Box plots: last-5-iter score & loss per cohort
    fig7_gui_features.png       — Color-mode & vibration preferences per participant
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────
RUNS_DIR  = Path(__file__).parent / "pipeline_runs"
REPORT_DIR = RUNS_DIR / "report"
REPORT_DIR.mkdir(exist_ok=True)

LOSS_CAP  = 60.0   # metres — exclude clear outliers (same threshold as existing scripts)
EXCLUDE   = {0, 42, 64}   # empty logs or different schema

# ── Colour palette ────────────────────────────────────────────────────────────
FIXED_CLR    = "#2B7BB9"   # blue
ADAPTIVE_CLR = "#E05A2B"   # orange
ALPHA_FILL   = 0.18

# ── Load all participant data ─────────────────────────────────────────────────
def load_participant(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                entry = {
                    "iteration":           int(row["iteration"]),
                    "driving_score":       float(row["driving_score"]),
                    "episode_loss":        float(row["episode_loss"]),
                    "scenario":            row.get("scenario", "unknown").strip(),
                    "style_speed":         float(row.get("style_speed", 0) or 0),
                    "style_efficiency":    float(row.get("style_efficiency", 0) or 0),
                    "style_aggressiveness":float(row.get("style_aggressiveness", 0) or 0),
                    "style_comfort":       float(row.get("style_comfort", 0) or 0),
                    "gui_type":            row.get("gui_type", "").strip().lower(),
                    "gui_location":        row.get("gui_location", "").strip().lower(),
                    "gui_color":           row.get("gui_color", "").strip().lower(),
                    "gui_vibration":       row.get("gui_vibration", "").strip().lower() in ("true", "1"),
                    "gui_lag":             float(row.get("gui_lag", 0) or 0),
                }
                rows.append(entry)
            except (KeyError, ValueError):
                continue
    return rows

all_participants = {}   # pnum -> list[dict]
for f in sorted(RUNS_DIR.glob("participant_*_log.csv")):
    pnum = int(f.stem.split("_")[1])
    if pnum in EXCLUDE:
        continue
    rows = load_participant(f)
    # apply loss cap
    rows = [r for r in rows if r["episode_loss"] <= LOSS_CAP]
    if not rows:
        continue
    all_participants[pnum] = rows

fixed_pnums    = sorted(p for p in all_participants if p % 2 == 1)
adaptive_pnums = sorted(p for p in all_participants if p % 2 == 0)

print(f"Fixed participants   ({len(fixed_pnums)}): {fixed_pnums}")
print(f"Adaptive participants ({len(adaptive_pnums)}): {adaptive_pnums}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def iter_arrays(pnum):
    rows = sorted(all_participants[pnum], key=lambda r: r["iteration"])
    iters  = np.array([r["iteration"]    for r in rows])
    scores = np.array([r["driving_score"] for r in rows])
    losses = np.array([r["episode_loss"]  for r in rows])
    return iters, scores, losses

def cohort_mean_std(pnums, key="driving_score"):
    by_iter = defaultdict(list)
    for p in pnums:
        for r in all_participants[p]:
            by_iter[r["iteration"]].append(r[key])
    iters  = np.array(sorted(by_iter))
    means  = np.array([np.mean(by_iter[i])  for i in iters])
    stds   = np.array([np.std(by_iter[i])   for i in iters])
    return iters, means, stds

# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Per-participant score & loss curves (grid)
# ═════════════════════════════════════════════════════════════════════════════
print("Building fig1_participant_grid …")

all_pnums = sorted(all_participants.keys())
NCOLS = 4
NROWS = math.ceil(len(all_pnums) / NCOLS)
fig1, axes1 = plt.subplots(
    NROWS * 2, NCOLS,
    figsize=(NCOLS * 4.2, NROWS * 4.0),
    gridspec_kw={"hspace": 0.55, "wspace": 0.35},
)
fig1.suptitle(
    "Per-Participant Session Summary — Driving Score & Alert Loss per Iteration",
    fontsize=13, fontweight="bold", y=1.01,
)

for col_idx, pnum in enumerate(all_pnums):
    col  = col_idx % NCOLS
    row_block = col_idx // NCOLS
    ax_s = axes1[row_block * 2,     col]
    ax_l = axes1[row_block * 2 + 1, col]

    iters, scores, losses = iter_arrays(pnum)
    clr = FIXED_CLR if pnum % 2 == 1 else ADAPTIVE_CLR
    mode_label = "Fixed" if pnum % 2 == 1 else "Adaptive"

    ax_s.plot(iters, scores, "o-", color=clr, ms=4, lw=1.5)
    ax_s.set_title(f"P{pnum}  ({mode_label})", fontsize=8.5, fontweight="bold", color=clr)
    ax_s.set_ylabel("Score", fontsize=7)
    ax_s.set_ylim(0, 0.65)
    ax_s.tick_params(labelsize=6)
    ax_s.grid(True, alpha=0.25)
    ax_s.set_xticks([])

    ax_l.plot(iters, losses, "s--", color=clr, ms=4, lw=1.2, alpha=0.85)
    ax_l.set_ylabel("Loss (m)", fontsize=7)
    ax_l.set_ylim(0, LOSS_CAP + 2)
    ax_l.set_xlabel("Iteration", fontsize=7)
    ax_l.tick_params(labelsize=6)
    ax_l.grid(True, alpha=0.25)

# hide unused axes
for col_idx in range(len(all_pnums), NROWS * NCOLS):
    col = col_idx % NCOLS
    rb  = col_idx // NCOLS
    axes1[rb * 2,     col].set_visible(False)
    axes1[rb * 2 + 1, col].set_visible(False)

legend_patches = [
    mpatches.Patch(color=FIXED_CLR,    label="Fixed alert mode"),
    mpatches.Patch(color=ADAPTIVE_CLR, label="Adaptive alert mode"),
]
fig1.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9)

fig1.savefig(str(REPORT_DIR / "fig1_participant_grid.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print("  Saved fig1_participant_grid.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Mean ± 1-SD learning curves, Fixed vs Adaptive
# ═════════════════════════════════════════════════════════════════════════════
print("Building fig2_cohort_learning …")

fig2, (ax2s, ax2l) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
fig2.suptitle("Learning Curves — Fixed vs Adaptive Alert Mode (Mean ± 1 SD)",
              fontsize=12, fontweight="bold")

for pnums, clr, lbl in [
    (fixed_pnums,    FIXED_CLR,    "Fixed"),
    (adaptive_pnums, ADAPTIVE_CLR, "Adaptive"),
]:
    for ax, key, ylabel in [
        (ax2s, "driving_score", "Human Driving Score"),
        (ax2l, "episode_loss",  "Alert Loss (m)"),
    ]:
        xi, mu, sd = cohort_mean_std(pnums, key)
        ax.plot(xi, mu, "o-", color=clr, lw=2, ms=5, label=lbl)
        ax.fill_between(xi, mu - sd, mu + sd, color=clr, alpha=ALPHA_FILL)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25)

ax2s.set_ylim(0, 0.65)
ax2l.set_ylim(0, LOSS_CAP + 2)
ax2l.set_xlabel("Iteration", fontsize=10)
ax2s.legend(fontsize=9, loc="lower right")
ax2l.legend(fontsize=9, loc="upper right")

# Mark rough convergence zone (iterations 14+)
for ax in (ax2s, ax2l):
    ax.axvspan(13.5, 19.5, color="grey", alpha=0.07, label="Late phase")
    ax.set_xticks(range(0, 20))

fig2.savefig(str(REPORT_DIR / "fig2_cohort_learning.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print("  Saved fig2_cohort_learning.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Alert type usage per participant + overall pie
# ═════════════════════════════════════════════════════════════════════════════
print("Building fig3_alert_types …")

TYPE_COLORS = {"arrow": "#4472C4", "route": "#ED7D31", "sound": "#70AD47", "": "#BFBFBF"}

fig3, (ax3bar, ax3pie) = plt.subplots(1, 2, figsize=(14, 5.5),
                                       gridspec_kw={"width_ratios": [3, 1]})
fig3.suptitle("Alert Type Usage Across Adaptive Participants", fontsize=12, fontweight="bold")

all_types = ["arrow", "route", "sound"]
type_totals = defaultdict(int)

# Exclude participants who used only arrow — no type variation to show
def _used_types(pnum):
    return {r["gui_type"] for r in all_participants[pnum] if r["gui_type"] in all_types}

fig3_pnums = [p for p in all_pnums if _used_types(p) != {"arrow"}]

bar_x  = []
stacks = {t: np.zeros(len(fig3_pnums)) for t in all_types}

for idx, pnum in enumerate(fig3_pnums):
    bar_x.append(f"P{pnum}")
    type_counts = defaultdict(int)
    for r in all_participants[pnum]:
        t = r["gui_type"] if r["gui_type"] in all_types else "arrow"
        type_counts[t] += 1
        type_totals[t] += 1
    for t in all_types:
        stacks[t][idx] = type_counts.get(t, 0)

x_pos = np.arange(len(fig3_pnums))
bot   = np.zeros(len(fig3_pnums))
for t in all_types:
    bars = ax3bar.bar(x_pos, stacks[t], bottom=bot,
                      color=TYPE_COLORS[t], label=t.capitalize(), width=0.65)
    bot += stacks[t]

ax3bar.set_xticks(x_pos)
ax3bar.set_xticklabels(bar_x, fontsize=7.5, rotation=45, ha="right")
ax3bar.set_ylabel("Iterations", fontsize=10)
ax3bar.set_title("Alert Type per Participant", fontsize=10)
ax3bar.legend(fontsize=9, loc="upper right")
ax3bar.grid(True, axis="y", alpha=0.25)

pie_sizes  = [type_totals.get(t, 0) for t in all_types]
pie_colors = [TYPE_COLORS[t] for t in all_types]
ax3pie.pie(
    pie_sizes,
    labels=[t.capitalize() for t in all_types],
    colors=pie_colors,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 10},
)
ax3pie.set_title("Overall Distribution", fontsize=10)

fig3.savefig(str(REPORT_DIR / "fig3_alert_types.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig3)
print("  Saved fig3_alert_types.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — Mean driving-style dimensions per participant (grouped bars)
# ═════════════════════════════════════════════════════════════════════════════
print("Building fig4_style_profiles …")

style_keys   = ["style_speed", "style_efficiency", "style_aggressiveness", "style_comfort"]
style_labels = ["Speed", "Efficiency", "Aggressiveness", "Comfort"]
style_colors = ["#4472C4", "#ED7D31", "#70AD47", "#FF0000"]

fig4, ax4 = plt.subplots(figsize=(14, 5))
fig4.suptitle("Mean Driving-Style Profile per Participant", fontsize=12, fontweight="bold")

n_styles = len(style_keys)
width = 0.18
x_pos = np.arange(len(all_pnums))

for si, (sk, sl, sc) in enumerate(zip(style_keys, style_labels, style_colors)):
    means = []
    for pnum in all_pnums:
        vals = [r[sk] for r in all_participants[pnum]]
        means.append(np.mean(vals))
    offset = (si - (n_styles - 1) / 2) * width
    ax4.bar(x_pos + offset, means, width=width, label=sl, color=sc, alpha=0.8)

# Cohort separator line between last fixed and first adaptive
mixed_sorted = sorted(all_pnums)
for i in range(len(mixed_sorted) - 1):
    if (mixed_sorted[i] % 2) != (mixed_sorted[i + 1] % 2):
        ax4.axvline(x=i + 0.5, color="black", lw=1.0, ls="--", alpha=0.5)

ax4.set_xticks(x_pos)
xticklabs = []
for p in all_pnums:
    sfx = " (F)" if p % 2 == 1 else " (A)"
    xticklabs.append(f"P{p}{sfx}")
ax4.set_xticklabels(xticklabs, fontsize=7, rotation=45, ha="right")
ax4.set_ylabel("Mean Style Score (0–1)", fontsize=10)
ax4.set_ylim(0, 1.05)
ax4.legend(fontsize=9, loc="upper right")
ax4.grid(True, axis="y", alpha=0.25)
ax4.set_title("Each cluster = one participant   |   F = Fixed, A = Adaptive", fontsize=9)

fig4.savefig(str(REPORT_DIR / "fig4_style_profiles.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig4)
print("  Saved fig4_style_profiles.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — Scenario heatmap: participant × scenario → avg driving score
# ═════════════════════════════════════════════════════════════════════════════
print("Building fig5_scenario_heatmap …")

scenario_scores = defaultdict(lambda: defaultdict(list))
for pnum in all_pnums:
    for r in all_participants[pnum]:
        scenario_scores[r["scenario"]][pnum].append(r["driving_score"])

all_scenarios = sorted(scenario_scores.keys())

heat_data = np.full((len(all_scenarios), len(all_pnums)), np.nan)
for si, sc in enumerate(all_scenarios):
    for pi, pn in enumerate(all_pnums):
        vals = scenario_scores[sc].get(pn, [])
        if vals:
            heat_data[si, pi] = np.mean(vals)

fig5_h = max(6, len(all_scenarios) * 0.4)
fig5, ax5 = plt.subplots(figsize=(14, fig5_h))
fig5.suptitle("Average Driving Score: Participant × Scenario", fontsize=12, fontweight="bold")

cmap5 = LinearSegmentedColormap.from_list("score", ["#d73027", "#fee090", "#1a9850"])
im5   = ax5.imshow(heat_data, aspect="auto", cmap=cmap5, vmin=0, vmax=0.65,
                   interpolation="nearest")

ax5.set_xticks(range(len(all_pnums)))
ax5.set_xticklabels([f"P{p}" for p in all_pnums], fontsize=7.5, rotation=45, ha="right")
ax5.set_yticks(range(len(all_scenarios)))
ax5.set_yticklabels([s.replace("_", " ") for s in all_scenarios], fontsize=7)
ax5.set_xlabel("Participant", fontsize=10)
ax5.set_ylabel("Scenario", fontsize=10)

cbar5 = fig5.colorbar(im5, ax=ax5, fraction=0.025, pad=0.03)
cbar5.set_label("Avg Driving Score", fontsize=9)

# annotate cells
for si in range(len(all_scenarios)):
    for pi in range(len(all_pnums)):
        v = heat_data[si, pi]
        if not np.isnan(v):
            ax5.text(pi, si, f"{v:.2f}", ha="center", va="center",
                     fontsize=5.5, color="black" if v > 0.15 else "white")

fig5.savefig(str(REPORT_DIR / "fig5_scenario_heatmap.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig5)
print("  Saved fig5_scenario_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — Box plots: last-5-iter performance, Fixed vs Adaptive
# ═════════════════════════════════════════════════════════════════════════════
print("Building fig6_final_performance …")

LATE_ITERS = 5   # last N iterations considered "final"

def late_values(pnums, key):
    out = []
    for p in pnums:
        rows = sorted(all_participants[p], key=lambda r: r["iteration"])
        late = rows[-LATE_ITERS:]
        out.extend(r[key] for r in late)
    return out

fig6, (ax6s, ax6l) = plt.subplots(1, 2, figsize=(10, 5.5))
fig6.suptitle(f"Final Performance  (last {LATE_ITERS} iterations)\nFixed vs Adaptive Alert Mode",
              fontsize=12, fontweight="bold")

for ax, key, ylabel, ylim in [
    (ax6s, "driving_score", "Human Driving Score",  (0, 0.75)),
    (ax6l, "episode_loss",  "Alert Loss (m)",        (0, LOSS_CAP + 2)),
]:
    fixed_vals    = late_values(fixed_pnums,    key)
    adaptive_vals = late_values(adaptive_pnums, key)
    bp = ax.boxplot(
        [fixed_vals, adaptive_vals],
        labels=["Fixed", "Adaptive"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    for patch, clr in zip(bp["boxes"], [FIXED_CLR, ADAPTIVE_CLR]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.65)

    # overlay individual points
    for xi, (vals, clr) in enumerate([(fixed_vals, FIXED_CLR), (adaptive_vals, ADAPTIVE_CLR)], 1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, color=clr, alpha=0.55, s=18, zorder=4)

    # annotate medians
    for xi, vals in enumerate([fixed_vals, adaptive_vals], 1):
        med = np.median(vals)
        ax.text(xi, med, f" {med:.3f}", va="center", fontsize=8.5, color="black")

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)

fig6.savefig(str(REPORT_DIR / "fig6_final_performance.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig6)
print("  Saved fig6_final_performance.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — GUI feature preferences per participant over iterations
#          (color mode fraction + vibration fraction)
# ═════════════════════════════════════════════════════════════════════════════
print("Building fig7_gui_features …")

fig7, axes7 = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                            gridspec_kw={"hspace": 0.18})
fig7.suptitle("Alert GUI Feature Preferences per Participant\n"
              "(fraction of iterations selecting each option)",
              fontsize=12, fontweight="bold")

ax7c, ax7v = axes7   # color mode, vibration

x_pos  = np.arange(len(all_pnums))
cb_frac, vib_frac = [], []

for pnum in all_pnums:
    rows      = all_participants[pnum]
    visual    = [r for r in rows if r["gui_type"] != "sound"]
    n         = len(rows)
    n_vis     = len(visual)
    cb_frac.append(
        sum(1 for r in visual if "colorblind" in r["gui_color"]) / max(n_vis, 1)
    )
    vib_frac.append(
        sum(1 for r in rows if r["gui_vibration"]) / max(n, 1)
    )

bar_w = 0.62
bars_cb = ax7c.bar(x_pos, cb_frac, width=bar_w, color=[
    FIXED_CLR if p % 2 == 1 else ADAPTIVE_CLR for p in all_pnums
], alpha=0.75)
ax7c.axhline(0.5, ls="--", lw=1, color="grey", alpha=0.7)
ax7c.set_ylabel("Colorblind mode\nfraction", fontsize=9)
ax7c.set_ylim(0, 1.05)
ax7c.set_title("Colorblind colour mode selection  (visual alerts only)", fontsize=9.5)
ax7c.grid(True, axis="y", alpha=0.2)

bars_vib = ax7v.bar(x_pos, vib_frac, width=bar_w, color=[
    FIXED_CLR if p % 2 == 1 else ADAPTIVE_CLR for p in all_pnums
], alpha=0.75)
ax7v.axhline(0.5, ls="--", lw=1, color="grey", alpha=0.7)
ax7v.set_ylabel("Vibration ON\nfraction", fontsize=9)
ax7v.set_ylim(0, 1.05)
ax7v.set_title("Vibration feature enabled", fontsize=9.5)
ax7v.grid(True, axis="y", alpha=0.2)
ax7v.set_xlabel("Participant", fontsize=10)

for ax in (ax7c, ax7v):
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"P{p}" for p in all_pnums], fontsize=8, rotation=45, ha="right")

legend_patches2 = [
    mpatches.Patch(color=FIXED_CLR,    label="Fixed mode"),
    mpatches.Patch(color=ADAPTIVE_CLR, label="Adaptive mode"),
]
fig7.legend(handles=legend_patches2, loc="upper right", fontsize=9, framealpha=0.9)

fig7.savefig(str(REPORT_DIR / "fig7_gui_features.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig7)
print("  Saved fig7_gui_features.png")

# ═════════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to: {REPORT_DIR}")
