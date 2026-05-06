"""
plot_report2.py — Discovery-driven report figures from pipeline_runs data.

Run:
    python plot_report2.py

Outputs (pipeline_runs/report/):
    fig8_alert_type_perf.png    — Sound alerts underperform visual types (box plots)
    fig9_scenario_difficulty.py — Scenario difficulty ranking + crash event counts
    fig10_style_drift.png       — Style dimension drift over training (both cohorts)
    fig11_score_volatility.png  — Mean score vs variance per participant (bubble chart)
    fig12_lag_sweet_spot.png    — GUI lag non-linear response curve
    fig13_efficiency_predictor  — Style efficiency vs driving score (strongest correlate)
    fig14_crash_map.png         — Zero-score crash events: participant × scenario heatmap
    fig15_adaptive_catchup.png  — Fixed vs adaptive convergence trajectory
"""

import csv, math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats as sp_stats

# ── Paths ─────────────────────────────────────────────────────────────────────
RUNS_DIR   = Path(__file__).parent / "pipeline_runs"
REPORT_DIR = RUNS_DIR / "report"
REPORT_DIR.mkdir(exist_ok=True)

LOSS_CAP = 60.0
EXCLUDE  = {0, 42, 64}

FIXED_CLR    = "#2B7BB9"
ADAPTIVE_CLR = "#E05A2B"
TYPE_COLORS  = {"arrow": "#4472C4", "route": "#ED7D31", "sound": "#C00000"}
ALPHA_FILL   = 0.16

# ── Load data ─────────────────────────────────────────────────────────────────
def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "iteration":            int(row["iteration"]),
                    "driving_score":        float(row["driving_score"]),
                    "episode_loss":         float(row["episode_loss"]),
                    "scenario":             row.get("scenario", "").strip(),
                    "style_speed":          float(row.get("style_speed",  0) or 0),
                    "style_efficiency":     float(row.get("style_efficiency", 0) or 0),
                    "style_aggressiveness": float(row.get("style_aggressiveness", 0) or 0),
                    "style_comfort":        float(row.get("style_comfort", 0) or 0),
                    "gui_type":             row.get("gui_type", "").strip().lower(),
                    "gui_location":         row.get("gui_location", "").strip().lower(),
                    "gui_color":            row.get("gui_color", "").strip().lower(),
                    "gui_vibration":        row.get("gui_vibration", "").strip().lower() in ("true", "1"),
                    "gui_lag":              float(row.get("gui_lag", 0) or 0),
                    "gui_p0_name":          row.get("gui_p0_name", "").strip(),
                    "gui_p0_val":           float(row.get("gui_p0_val", 0) or 0),
                })
            except (KeyError, ValueError):
                continue
    return rows

all_p = {}
for f in sorted(RUNS_DIR.glob("participant_*_log.csv")):
    pnum = int(f.stem.split("_")[1])
    if pnum in EXCLUDE: continue
    rows = [r for r in load(f) if r["episode_loss"] <= LOSS_CAP]
    if rows: all_p[pnum] = rows

fixed_p    = {p: r for p, r in all_p.items() if p % 2 == 1}
adaptive_p = {p: r for p, r in all_p.items() if p % 2 == 0}

all_rows = [r for rows in all_p.values() for r in rows]

# ── Helper: cohort mean/std by iteration ─────────────────────────────────────
def cohort_mean_std(pdict, key):
    by_iter = defaultdict(list)
    for rows in pdict.values():
        for r in rows: by_iter[r["iteration"]].append(r[key])
    xi   = np.array(sorted(by_iter))
    mu   = np.array([np.mean(by_iter[i]) for i in xi])
    sd   = np.array([np.std(by_iter[i])  for i in xi])
    return xi, mu, sd

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Alert type performance: sound vs visual (arrow / route)
# ─────────────────────────────────────────────────────────────────────────────
print("Building fig8_alert_type_perf …")

type_score = defaultdict(list)
type_loss  = defaultdict(list)
for r in all_rows:
    t = r["gui_type"] if r["gui_type"] in ("arrow","route","sound") else "arrow"
    type_score[t].append(r["driving_score"])
    type_loss[t].append(r["episode_loss"])

TYPES_ORD = ["arrow", "route", "sound"]
TYPE_LABELS = ["Arrow\n(visual)", "Route\n(visual)", "Sound\n(audio)"]

fig8, (ax8s, ax8l) = plt.subplots(1, 2, figsize=(11, 5.5))
fig8.suptitle("Alert Type Performance Comparison\nSound Alerts Underperform Visual Types",
              fontsize=12, fontweight="bold")

rng = np.random.default_rng(0)
for ax, data_dict, ylabel, ylim, title in [
    (ax8s, type_score, "Human Driving Score",  (0, 0.72), "Driving Score by Alert Type"),
    (ax8l, type_loss,  "Alert Loss (m)",        (0, 62),   "Alert Loss by Alert Type"),
]:
    vals_list = [data_dict[t] for t in TYPES_ORD]
    bp = ax.boxplot(
        vals_list,
        tick_labels=TYPE_LABELS,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        widths=0.5,
    )
    for patch, t in zip(bp["boxes"], TYPES_ORD):
        patch.set_facecolor(TYPE_COLORS[t])
        patch.set_alpha(0.65)

    for xi, (t, vals) in enumerate(zip(TYPES_ORD, vals_list), 1):
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(xi + jitter, vals, color=TYPE_COLORS[t], alpha=0.35, s=12, zorder=4)
        med = np.median(vals)
        ax.text(xi + 0.30, med, f"{med:.3f}", va="center", fontsize=9, fontweight="bold")

    # Significance bracket: arrow vs sound
    y_top = ylim[1] * 0.88
    ax.annotate("", xy=(3, y_top), xytext=(1, y_top),
                arrowprops=dict(arrowstyle="-", lw=1.5, color="black"))
    t_stat, p_val = sp_stats.ttest_ind(data_dict["arrow"], data_dict["sound"])
    star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    ax.text(2, y_top * 1.008, f"p={p_val:.4f} {star}", ha="center", fontsize=8.5)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)

# Sample sizes
for t, xi in zip(TYPES_ORD, [1, 2, 3]):
    n = len(type_score[t])
    ax8s.text(xi, -0.055, f"n={n}", ha="center", fontsize=8, color=TYPE_COLORS[t], transform=ax8s.get_xaxis_transform())

fig8.savefig(str(REPORT_DIR / "fig8_alert_type_perf.png"), dpi=150, bbox_inches="tight")
plt.close(fig8)
print("  Saved fig8_alert_type_perf.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Scenario difficulty ranking + crash counts
# ─────────────────────────────────────────────────────────────────────────────
print("Building fig9_scenario_difficulty …")

sc_scores  = defaultdict(list)
sc_crashes = defaultdict(int)
for r in all_rows:
    sc_scores[r["scenario"]].append(r["driving_score"])
    if r["driving_score"] < 0.05:
        sc_crashes[r["scenario"]] += 1

scenarios_ranked = sorted(sc_scores.keys(), key=lambda s: np.mean(sc_scores[s]))
sc_means = [np.mean(sc_scores[s]) for s in scenarios_ranked]
sc_sems  = [sp_stats.sem(sc_scores[s]) for s in scenarios_ranked]
sc_crash_counts = [sc_crashes.get(s, 0) for s in scenarios_ranked]

# Color bars by difficulty tier
def tier_color(mean):
    if mean < 0.32: return "#C00000"    # hard
    if mean < 0.42: return "#ED7D31"    # medium
    return "#70AD47"                    # easy

bar_colors = [tier_color(m) for m in sc_means]

fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [3, 1], "wspace": 0.04})
fig9.suptitle("Scenario Difficulty Ranking  (all participants, outliers excluded)",
              fontsize=12, fontweight="bold")

y_pos = np.arange(len(scenarios_ranked))
ax9a.barh(y_pos, sc_means, xerr=sc_sems, color=bar_colors, alpha=0.82,
          height=0.65, error_kw={"elinewidth": 1.2, "capsize": 3, "capthick": 1.2})
ax9a.axvline(np.mean(sc_means), ls="--", lw=1.5, color="black", alpha=0.55, label="Grand mean")
for yi, (m, sem) in enumerate(zip(sc_means, sc_sems)):
    ax9a.text(m + sem + 0.008, yi, f"{m:.3f}", va="center", fontsize=8)
ax9a.set_yticks(y_pos)
sc_labels = [s.replace("_", " ").replace("town0", "T0") for s in scenarios_ranked]
ax9a.set_yticklabels(sc_labels, fontsize=8.5)
ax9a.set_xlabel("Mean Driving Score  (higher = easier)", fontsize=10)
ax9a.set_xlim(0, 0.72)
ax9a.legend(fontsize=9)
ax9a.grid(True, axis="x", alpha=0.25)

legend_patches = [
    mpatches.Patch(color="#C00000", label="Hard  (< 0.32)", alpha=0.82),
    mpatches.Patch(color="#ED7D31", label="Medium (0.32–0.42)", alpha=0.82),
    mpatches.Patch(color="#70AD47", label="Easy  (> 0.42)", alpha=0.82),
]
ax9a.legend(handles=legend_patches, fontsize=8, loc="lower right")

# Crash count column
crash_colors = ["#C00000" if c > 0 else "#DDDDDD" for c in sc_crash_counts]
ax9b.barh(y_pos, sc_crash_counts, color=crash_colors, alpha=0.8, height=0.65)
for yi, cnt in enumerate(sc_crash_counts):
    if cnt > 0:
        ax9b.text(cnt + 0.05, yi, str(cnt), va="center", fontsize=9, color="#C00000", fontweight="bold")
ax9b.set_xlabel("Zero-score crashes", fontsize=10)
ax9b.set_yticks(y_pos)
ax9b.set_yticklabels([])
ax9b.set_xlim(0, max(sc_crash_counts) + 1.5)
ax9b.grid(True, axis="x", alpha=0.25)
ax9b.set_title("Crashes", fontsize=9)

fig9.savefig(str(REPORT_DIR / "fig9_scenario_difficulty.png"), dpi=150, bbox_inches="tight")
plt.close(fig9)
print("  Saved fig9_scenario_difficulty.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Style drift over training (both cohorts)
# ─────────────────────────────────────────────────────────────────────────────
print("Building fig10_style_drift …")

style_meta = [
    ("style_speed",           "Speed",           "#4472C4"),
    ("style_efficiency",      "Efficiency",      "#ED7D31"),
    ("style_aggressiveness",  "Aggressiveness",  "#C00000"),
    ("style_comfort",         "Comfort",         "#70AD47"),
]

fig10, axes10 = plt.subplots(2, 2, figsize=(12, 8), sharex=True,
                              gridspec_kw={"hspace": 0.38, "wspace": 0.30})
fig10.suptitle("Driving Style Drift Over Training\n"
               "Efficiency rises significantly; Comfort stays flat",
               fontsize=12, fontweight="bold")

for ax, (sk, sl, sc) in zip(axes10.flat, style_meta):
    for pdict, clr, lbl in [(fixed_p, FIXED_CLR, "Fixed"), (adaptive_p, ADAPTIVE_CLR, "Adaptive")]:
        xi, mu, sd = cohort_mean_std(pdict, sk)
        ax.plot(xi, mu, "o-", color=clr, lw=2, ms=4, label=lbl)
        ax.fill_between(xi, mu - sd, mu + sd, color=clr, alpha=ALPHA_FILL)

        # annotate delta
        delta = mu[-1] - mu[0]
        ax.annotate(
            f"{lbl[0]}: {delta:+.3f}",
            xy=(xi[-1], mu[-1]),
            xytext=(xi[-1] - 3, mu[-1] + (0.02 if lbl == "Fixed" else -0.03)),
            fontsize=7.5, color=clr,
            arrowprops=dict(arrowstyle="->", color=clr, lw=0.8),
        )

    ax.set_title(sl, fontsize=10, fontweight="bold", color=sc)
    ax.set_ylabel("Style Score (0–1)", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, 20, 2))
    ax.grid(True, alpha=0.22)
    if ax in axes10[1]:
        ax.set_xlabel("Iteration", fontsize=9)

axes10.flat[0].legend(fontsize=8.5, loc="lower right")
fig10.savefig(str(REPORT_DIR / "fig10_style_drift.png"), dpi=150, bbox_inches="tight")
plt.close(fig10)
print("  Saved fig10_style_drift.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Score volatility bubble chart (mean vs std, bubble = zero-crashes)
# ─────────────────────────────────────────────────────────────────────────────
print("Building fig11_score_volatility …")

fig11, ax11 = plt.subplots(figsize=(9, 6))
fig11.suptitle("Participant Score Consistency\n"
               "X = mean score  |  Y = score std dev  |  Bubble size = zero-score crashes",
               fontsize=11, fontweight="bold")

for pnum in sorted(all_p):
    scores     = [r["driving_score"] for r in all_p[pnum]]
    n_crashes  = sum(1 for s in scores if s < 0.05)
    mean_s     = np.mean(scores)
    std_s      = np.std(scores)
    clr        = FIXED_CLR if pnum % 2 == 1 else ADAPTIVE_CLR
    bubble_sz  = 120 + n_crashes * 220    # base size + crash penalty

    ax11.scatter(mean_s, std_s, s=bubble_sz, color=clr, alpha=0.72, edgecolors="white", lw=1.2, zorder=4)
    ax11.text(mean_s + 0.003, std_s + 0.002, f"P{pnum}", fontsize=8, color=clr, fontweight="bold")

# Quadrant lines
ax11.axhline(0.12, ls="--", lw=1.2, color="grey", alpha=0.55)
ax11.axvline(0.38, ls="--", lw=1.2, color="grey", alpha=0.55)
ax11.text(0.29, 0.215, "Low score\nHigh volatility", ha="center", fontsize=8, color="#888888", style="italic")
ax11.text(0.44, 0.215, "High score\nHigh volatility", ha="center", fontsize=8, color="#888888", style="italic")
ax11.text(0.29, 0.045, "Low score\nConsistent", ha="center", fontsize=8, color="#888888", style="italic")
ax11.text(0.44, 0.045, "High score\nConsistent ✓", ha="center", fontsize=8, color="#444444", style="italic", fontweight="bold")

ax11.set_xlabel("Mean Driving Score", fontsize=10)
ax11.set_ylabel("Score Std Dev  (higher = more volatile)", fontsize=10)
ax11.set_xlim(0.22, 0.54)
ax11.set_ylim(0.05, 0.27)
ax11.grid(True, alpha=0.22)

legend_patches = [
    mpatches.Patch(color=FIXED_CLR,    label="Fixed mode",    alpha=0.72),
    mpatches.Patch(color=ADAPTIVE_CLR, label="Adaptive mode", alpha=0.72),
]
# Bubble size legend
for nc, lbl in [(0, "0 crashes"), (1, "1 crash"), (3, "3 crashes")]:
    ax11.scatter([], [], s=120 + nc * 220, color="#888888", alpha=0.65, label=lbl)
all_handles = legend_patches + ax11.get_legend_handles_labels()[0][2:]
all_labels  = [p.get_label() for p in legend_patches] + ["0 crashes", "1 crash", "3 crashes"]
ax11.legend(all_handles, all_labels, fontsize=8.5, loc="upper left", framealpha=0.9)

fig11.savefig(str(REPORT_DIR / "fig11_score_volatility.png"), dpi=150, bbox_inches="tight")
plt.close(fig11)
print("  Saved fig11_score_volatility.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 — GUI lag non-linear response curve
# ─────────────────────────────────────────────────────────────────────────────
print("Building fig12_lag_sweet_spot …")

lag_scores = defaultdict(list)
for r in all_rows:
    bucket = round(r["gui_lag"] * 4) / 4   # 0.25s bins
    lag_scores[bucket].append(r["driving_score"])

buckets = np.array(sorted(lag_scores))
means   = np.array([np.mean(lag_scores[b]) for b in buckets])
sems    = np.array([sp_stats.sem(lag_scores[b])  for b in buckets])
counts  = np.array([len(lag_scores[b])            for b in buckets])

fig12, ax12a = plt.subplots(figsize=(9, 5))
fig12.suptitle("GUI Alert Lag vs Driving Score\n"
               "~0.5 s lag outperforms ~1.0 s — unexpected non-linear response",
               fontsize=12, fontweight="bold")

ax12b = ax12a.twinx()

ax12b.bar(buckets, counts, width=0.22, color="#CCCCCC", alpha=0.5, zorder=1, label="Sample count")
ax12a.errorbar(buckets, means, yerr=sems, fmt="o-", color="#2B7BB9",
               lw=2, ms=7, capsize=5, capthick=1.5, zorder=5)

# Fit & plot a cubic spline for smoothing
from scipy.interpolate import UnivariateSpline
if len(buckets) >= 4:
    spl = UnivariateSpline(buckets, means, s=0.001, k=3)
    xs  = np.linspace(buckets.min(), buckets.max(), 200)
    ax12a.plot(xs, spl(xs), "--", color="#2B7BB9", lw=1.2, alpha=0.5)

# Annotate peak
peak_idx = np.argmax(means)
ax12a.annotate(
    f"Peak: {means[peak_idx]:.3f}\n@ lag={buckets[peak_idx]:.2f}s",
    xy=(buckets[peak_idx], means[peak_idx]),
    xytext=(buckets[peak_idx] + 0.20, means[peak_idx] - 0.018),
    fontsize=9, color="#2B7BB9",
    arrowprops=dict(arrowstyle="->", color="#2B7BB9", lw=1.0),
)

ax12a.set_xlabel("GUI Alert Lag (seconds)", fontsize=10)
ax12a.set_ylabel("Mean Driving Score", fontsize=10, color="#2B7BB9")
ax12a.tick_params(axis="y", labelcolor="#2B7BB9")
ax12a.set_ylim(0.30, 0.46)
ax12b.set_ylabel("Sample Count", fontsize=10, color="#888888")
ax12b.tick_params(axis="y", labelcolor="#888888")
ax12a.grid(True, alpha=0.25)

lines1, labels1 = ax12a.get_legend_handles_labels()
lines2, labels2 = ax12b.get_legend_handles_labels()
ax12a.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

fig12.savefig(str(REPORT_DIR / "fig12_lag_sweet_spot.png"), dpi=150, bbox_inches="tight")
plt.close(fig12)
print("  Saved fig12_lag_sweet_spot.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 — Style efficiency vs driving score (strongest predictor, r=+0.349)
# ─────────────────────────────────────────────────────────────────────────────
print("Building fig13_efficiency_predictor …")

fig13, axes13 = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.30})
fig13.suptitle("Driving Style → Score Correlations\n"
               "Efficiency is the strongest predictor of performance (r = +0.35)",
               fontsize=12, fontweight="bold")

style_meta_corr = [
    ("style_efficiency",     "Efficiency",      "#ED7D31", True),
    ("style_aggressiveness", "Aggressiveness",  "#C00000", False),
]

scores_all = np.array([r["driving_score"] for r in all_rows])

for ax, (sk, sl, sc, _) in zip(axes13, style_meta_corr):
    vals = np.array([r[sk] for r in all_rows])
    pnums_all = []
    for p, rows in all_p.items():
        pnums_all.extend([p] * len(rows))
    pnums_arr = np.array(pnums_all)

    # Per-participant scatter
    for pnum in sorted(all_p):
        mask = pnums_arr == pnum
        clr  = FIXED_CLR if pnum % 2 == 1 else ADAPTIVE_CLR
        ax.scatter(vals[mask], scores_all[mask], color=clr, alpha=0.28, s=18, zorder=3)

    # Overall regression line
    slope, intercept, r, p_val, se = sp_stats.linregress(vals, scores_all)
    xs = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(xs, slope * xs + intercept, "k-", lw=2.2, zorder=6,
            label=f"r = {r:+.3f}  (p={p_val:.2e})")

    # Per-participant means
    p_eff  = np.array([np.mean([r[sk]             for r in all_p[p]]) for p in sorted(all_p)])
    p_sc   = np.array([np.mean([r["driving_score"] for r in all_p[p]]) for p in sorted(all_p)])
    p_clrs = [FIXED_CLR if p % 2 == 1 else ADAPTIVE_CLR for p in sorted(all_p)]
    ax.scatter(p_eff, p_sc, color=p_clrs, edgecolors="black", lw=1.0, s=90, zorder=8)
    for xi, yi, pnum in zip(p_eff, p_sc, sorted(all_p)):
        ax.text(xi + 0.005, yi + 0.004, f"P{pnum}", fontsize=7,
                color=FIXED_CLR if pnum % 2 == 1 else ADAPTIVE_CLR)

    ax.set_xlabel(f"Style: {sl} (0–1)", fontsize=10)
    ax.set_ylabel("Driving Score" if sk == "style_efficiency" else "", fontsize=10)
    ax.set_ylim(0, 0.68)
    ax.legend(fontsize=9.5, loc="lower right" if r > 0 else "upper right")
    ax.grid(True, alpha=0.22)

legend_patches2 = [
    mpatches.Patch(color=FIXED_CLR,    alpha=0.7, label="Fixed mode"),
    mpatches.Patch(color=ADAPTIVE_CLR, alpha=0.7, label="Adaptive mode"),
]
axes13[0].legend(
    handles=axes13[0].get_legend_handles_labels()[0] + legend_patches2,
    labels =axes13[0].get_legend_handles_labels()[1] + [p.get_label() for p in legend_patches2],
    fontsize=8.5, loc="lower right",
)

fig13.savefig(str(REPORT_DIR / "fig13_efficiency_predictor.png"), dpi=150, bbox_inches="tight")
plt.close(fig13)
print("  Saved fig13_efficiency_predictor.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 14 — Zero-score crash map: participant × scenario
# ─────────────────────────────────────────────────────────────────────────────
print("Building fig14_crash_map …")

crash_scenarios = sorted({r["scenario"] for r in all_rows if r["driving_score"] < 0.05})
all_pnums_sorted = sorted(all_p.keys())

crash_matrix = np.zeros((len(crash_scenarios), len(all_pnums_sorted)), dtype=int)
for pi, pnum in enumerate(all_pnums_sorted):
    for r in all_p[pnum]:
        if r["driving_score"] < 0.05 and r["scenario"] in crash_scenarios:
            si = crash_scenarios.index(r["scenario"])
            crash_matrix[si, pi] += 1

# Also show all scores as background color (avg score, faint)
full_sc_sorted = sorted({r["scenario"] for r in all_rows})
sc_avg = {}
for sc in full_sc_sorted:
    vals = [r["driving_score"] for r in all_rows if r["scenario"] == sc]
    sc_avg[sc] = np.mean(vals) if vals else 0

fig14, ax14 = plt.subplots(figsize=(13, 5))
fig14.suptitle("Zero-Score Crash Events: Participant × Scenario\n"
               "Town04 crossing scenarios account for the most crashes",
               fontsize=12, fontweight="bold")

im14 = ax14.imshow(crash_matrix, aspect="auto", cmap="Reds", vmin=0, vmax=3,
                   interpolation="nearest")

for si in range(len(crash_scenarios)):
    for pi in range(len(all_pnums_sorted)):
        v = crash_matrix[si, pi]
        if v > 0:
            ax14.text(pi, si, str(v), ha="center", va="center",
                      fontsize=11, fontweight="bold",
                      color="white" if v >= 2 else "#C00000")

ax14.set_xticks(range(len(all_pnums_sorted)))
ax14.set_xticklabels(
    [f"P{p}\n{'F' if p%2==1 else 'A'}" for p in all_pnums_sorted],
    fontsize=8.5
)
ax14.set_yticks(range(len(crash_scenarios)))
ax14.set_yticklabels([s.replace("_", " ").replace("town0", "T0") for s in crash_scenarios], fontsize=8.5)
ax14.set_xlabel("Participant  (F=Fixed, A=Adaptive)", fontsize=10)
ax14.set_ylabel("Scenario", fontsize=10)

cbar14 = fig14.colorbar(im14, ax=ax14, fraction=0.025, pad=0.02)
cbar14.set_label("Crash count", fontsize=9)
cbar14.set_ticks([0, 1, 2, 3])

# Annotate total crashes per scenario
crash_row_totals = crash_matrix.sum(axis=1)
for si, total in enumerate(crash_row_totals):
    ax14.text(len(all_pnums_sorted) - 0.35, si, f"  Σ={total}",
              va="center", fontsize=8.5, color="#C00000", fontweight="bold")

fig14.savefig(str(REPORT_DIR / "fig14_crash_map.png"), dpi=150, bbox_inches="tight")
plt.close(fig14)
print("  Saved fig14_crash_map.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 15 — Adaptive vs Fixed convergence: same end-state, faster catch-up
# ─────────────────────────────────────────────────────────────────────────────
print("Building fig15_adaptive_catchup …")

fig15, (ax15s, ax15l) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                      gridspec_kw={"hspace": 0.12})
fig15.suptitle("Adaptive Mode Catches Up to Fixed Mode Despite Weaker Early Start\n"
               "Both cohorts converge to similar final performance",
               fontsize=12, fontweight="bold")

for pdict, clr, lbl in [(fixed_p, FIXED_CLR, "Fixed"), (adaptive_p, ADAPTIVE_CLR, "Adaptive")]:
    for ax, key, ylabel in [(ax15s, "driving_score", "Human Driving Score"),
                             (ax15l, "episode_loss",  "Alert Loss (m)")]:
        xi, mu, sd = cohort_mean_std(pdict, key)
        ax.plot(xi, mu, "o-", color=clr, lw=2.5, ms=5.5, label=f"{lbl} (mean)", zorder=5)
        ax.fill_between(xi, mu - sd, mu + sd, color=clr, alpha=ALPHA_FILL, zorder=3)

        # Individual participant thin lines
        for p, rows in pdict.items():
            s = sorted(rows, key=lambda r: r["iteration"])
            pxi = [r["iteration"] for r in s]
            pyi = [r[key]         for r in s]
            ax.plot(pxi, pyi, color=clr, lw=0.5, alpha=0.18, zorder=2)

# Annotate convergence window
for ax in (ax15s, ax15l):
    ax.axvspan(13.5, 19.5, color="grey", alpha=0.08, zorder=1)
    ax.text(16.5, ax.get_ylim()[1] * 0.97, "Late\nphase", ha="center", va="top",
            fontsize=8, color="#666666", style="italic")
    ax.set_xticks(range(0, 20))
    ax.grid(True, alpha=0.22)

ax15s.set_ylabel("Human Driving Score", fontsize=10)
ax15s.set_ylim(0, 0.65)
ax15s.legend(fontsize=9.5, loc="lower right")

ax15l.set_ylabel("Alert Loss (m)", fontsize=10)
ax15l.set_ylim(0, LOSS_CAP + 2)
ax15l.set_xlabel("Iteration", fontsize=10)
ax15l.legend(fontsize=9.5, loc="upper right")

# Delta annotation on score plot
xi_f, mu_f, _ = cohort_mean_std(fixed_p,    "driving_score")
xi_a, mu_a, _ = cohort_mean_std(adaptive_p, "driving_score")
delta_start = mu_f[0] - mu_a[0]
delta_end   = mu_f[-1] - mu_a[-1]
ax15s.annotate(
    f"Gap iter 0: Δ={delta_start:+.3f}",
    xy=(0, (mu_f[0] + mu_a[0]) / 2),
    xytext=(1.5, 0.20),
    fontsize=8.5, color="#555555",
    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
)
ax15s.annotate(
    f"Gap iter 19: Δ={delta_end:+.3f}",
    xy=(19, (mu_f[-1] + mu_a[-1]) / 2),
    xytext=(15.5, 0.22),
    fontsize=8.5, color="#555555",
    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
)

fig15.savefig(str(REPORT_DIR / "fig15_adaptive_catchup.png"), dpi=150, bbox_inches="tight")
plt.close(fig15)
print("  Saved fig15_adaptive_catchup.png")

print(f"\nAll discovery figures saved to: {REPORT_DIR}")
