"""
Render every figure used in the documentation, and stage the tutorial datasets
for download from the site.

Figures are generated from the tutorial datasets, not drawn by hand, so a change
to the data or to `methods.py` shows up here rather than silently contradicting
the text. Styling goes through Chronotopia's own `styles.py` — the docs use the
same "Journal" style and the same palette the app ships with, so a reader
following a tutorial sees figures that look like their own screen.

Run from the repository root:
    python docs/make_figures.py

Writes SVG (for the site) and PNG (for anywhere that can't take SVG) into
docs/assets/figures/, and copies tutorials/*.csv into docs/downloads/ so the
site can serve them. The copies are generated, not committed — `tutorials/` is
the single source of truth for the data, and a second copy in git would drift.
Run this before `mkdocs serve`; CI runs it before `mkdocs build`.

Colour rules, taken from the accessibility work in v0.7.4/v0.7.5 and re-checked
here:
  * categorical hues are assigned per entity in a fixed order and never cycled
  * two of the five palette slots fall below 3:1 contrast on white, so every
    multi-series figure carries a legend AND direct labels — identity never
    rests on colour alone
  * label text is ink, never the series colour: the line beside it carries the
    identity, and #eda100 as text on white is unreadable
  * magnitude uses one hue light-to-dark; period-relative-to-24 h uses a
    diverging scale with a neutral grey midpoint, matching the plate view
"""

from __future__ import annotations

import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import plots
import styles
from methods import methods

TUT = os.path.join(REPO, "tutorials")
OUT = os.path.join(REPO, "docs", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

# ── the palette, by entity ──────────────────────────────────────────────────
P = styles.PALETTES["Chronotopia"]["colors"]        # 5 slots, fixed order
ORANGE, SKY, BLUE, AMBER, VIOLET = P

INK = "#222222"          # primary text
INK_2 = "#555555"        # secondary text
MUTED = "#8A8A8A"        # de-emphasised series
HAIRLINE = "#DCDCDA"
SURFACE = "white"

# Entity -> colour. Set once; every figure reads from here, so a genotype keeps
# its hue across the whole documentation.
GENOTYPE_COLOR = {
    "Wild type": BLUE,
    "Short period": ORANGE,
    "Long period": VIOLET,
    "Arrhythmic": MUTED,
}
CLASS_COLOR = {
    "Core clock": BLUE,
    "Clock-controlled": ORANGE,
    "Non-rhythmic": MUTED,
}
METHOD_COLOR = {
    "Lomb-Scargle Periodogram": BLUE,
    "Wavelet Transform": ORANGE,
    "Autocorrelation": VIOLET,
}

LINE_W = 1.6             # ~2px at the sizes used here
MARKER_S = 5.0           # >= 8px diameter
ZEITGEBER_FILL = "#EBEBEB"


def new_fig(w=7.2, h=3.4, **kw):
    styles.apply("Journal", "notebook", "Chronotopia")
    fig, ax = plt.subplots(figsize=(w, h), **kw)
    return fig, ax


def save(fig, name: str, pad=0.03):
    """SVG for the site, PNG for everything else. Margins are reserved rather
    than trimmed, so direct labels sitting outside the axes survive."""
    for ext, kwargs in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    bbox_inches="tight", pad_inches=pad,
                    facecolor=SURFACE, **kwargs)
    plt.close(fig)
    print(f"  {name}.svg  {name}.png")


def day_ticks(ax, t_max, step=24):
    ax.set_xticks(np.arange(0, t_max + 1, step))
    ax.set_xlabel("Time (h)")


def end_labels(ax, items, x, dx=1.0, size=8, min_gap_frac=0.075):
    """
    Direct labels at the right-hand end of a set of lines.

    Labels are pushed apart when they would collide and joined to their line by
    a leader, rather than stacked next to the wrong trace. Text is ink; the
    leader carries the colour — two of the five palette slots are below 3:1 on
    white and are unreadable as text.

    items: list of (y, text, color)
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac

    items = sorted(items, key=lambda it: it[0])
    placed = [float(it[0]) for it in items]
    for i in range(1, len(placed)):                     # push up
        if placed[i] - placed[i - 1] < gap:
            placed[i] = placed[i - 1] + gap
    for i in range(len(placed) - 2, -1, -1):            # then back down
        if placed[i + 1] - placed[i] < gap:
            placed[i] = placed[i + 1] - gap

    for (y_true, text, color), y_lab in zip(items, placed):
        ax.plot([x, x + dx * 0.55, x + dx * 0.80],
                [y_true, y_lab, y_lab],
                color=color, lw=1.2, solid_capstyle="butt",
                clip_on=False, zorder=2)
        ax.annotate(text, xy=(x + dx * 0.90, y_lab), va="center", ha="left",
                    fontsize=size, color=INK, annotation_clip=False)


def legend_below(ax, handles=None, labels=None, ncol=3, y=-0.30):
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, fontsize=8)
    leg = (ax.legend(handles=handles, labels=labels, **kw) if handles
           else ax.legend(**kw))
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def tidy(ax, legend_title=None, loc=None, ncol=1):
    if loc is not None and ax.get_legend_handles_labels()[0]:
        leg = ax.legend(title=legend_title, loc=loc, ncol=ncol,
                        frameon=False, fontsize=8, title_fontsize=8)
        for t in leg.get_texts():
            t.set_color(INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if ax.get_title():
        ax.title.set_size(9.5)
        ax.title.set_color(INK)


# ═══════════════════════════════════════════════════════════════════════════
#  Load
# ═══════════════════════════════════════════════════════════════════════════

d1 = pd.read_csv(os.path.join(TUT, "tutorial_1_short_series_omics.csv"))
lay1 = pd.read_csv(os.path.join(TUT, "tutorial_1_short_series_layout.csv"))
tr1 = pd.read_csv(os.path.join(TUT, "tutorial_1_short_series_truth.csv")).set_index("Sample")
cols1 = [c for c in d1.columns if c != "Time"]

d2 = pd.read_csv(os.path.join(TUT, "tutorial_2_long_series_luciferase.csv"))
tr2 = pd.read_csv(os.path.join(TUT, "tutorial_2_long_series_truth.csv")).set_index("Sample")
ent2 = pd.read_csv(os.path.join(TUT, "tutorial_2_long_series_entrainment.csv"))
cols2 = [c for c in d2.columns if c != "Time"]

clean1 = d1.dropna()
agg1 = clean1.groupby("Time", as_index=False).mean()
sd1 = clean1.groupby("Time").std()

RELEASE_H = 48.0
free2 = d2[d2["Time"] >= RELEASE_H].reset_index(drop=True)
det2 = free2.copy()
det2[cols2] = methods.detrend(free2, cols2, "Time", "Rolling mean")


def cosinor(t, y, period=24.0):
    """(amplitude, peak phase in hours, fitted curve) for a fixed period."""
    w = 2 * np.pi / period
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return (float(np.hypot(beta[1], beta[2])),
            float((np.arctan2(beta[2], beta[1]) * period / (2 * np.pi)) % period),
            X @ beta)


AMP1 = {}
PHASE1 = {}
for c in cols1:
    a, p, _ = cosinor(agg1["Time"].to_numpy(), agg1[c].to_numpy())
    AMP1[c], PHASE1[c] = a, p
AMP1 = pd.Series(AMP1)
PHASE1 = pd.Series(PHASE1)


# ═══════════════════════════════════════════════════════════════════════════
#  Tutorial 1
# ═══════════════════════════════════════════════════════════════════════════

def fig_t1_traces():
    """Four rhythmic transcripts and one flat one, mean +/- SD."""
    show = [("NR1D1", ORANGE), ("DBP", AMBER), ("PER2", BLUE),
            ("ARNTL", VIOLET), ("ACTB", MUTED)]
    fig, ax = new_fig(7.2, 3.4)
    t = agg1["Time"].to_numpy()
    labels = []
    for name, color in show:
        y = agg1[name].to_numpy()
        e = sd1[name].to_numpy()
        ax.fill_between(t, y - e, y + e, color=color, alpha=0.10, lw=0)
        ax.plot(t, y, color=color, lw=LINE_W, marker="o", ms=MARKER_S,
                mfc=color, mec=SURFACE, mew=1.4, label=name, zorder=3)
        labels.append((y[-1], name, color))
    day_ticks(ax, 48)
    ax.set_ylabel("Expression (log2 CPM)")
    ax.set_title("Four rhythmic transcripts and one flat one — mean ± SD, n = 3")
    ax.set_xlim(-1, 48)
    ax.set_ylim(5.4, 13.9)
    end_labels(ax, labels, x=48, dx=4.0)
    legend_below(ax, ncol=5, y=-0.24)
    tidy(ax)
    save(fig, "t1-traces")


def fig_t1_phase_map():
    """Peak phase per transcript, ordered. The thing a 48 h series does well."""
    order = PHASE1[tr1.Detectable == "yes"].sort_values()
    border = PHASE1[tr1.Detectable == "borderline"].sort_values()
    fig, ax = new_fig(6.4, 4.6)

    rows = list(order.items()) + list(border.items())
    for i, (name, ph) in enumerate(rows):
        detectable = tr1.Detectable[name] == "yes"
        color = BLUE if tr1.Class[name] == "Core clock" else ORANGE
        true_ph = tr1.True_peak_phase_CT_h[name]
        ax.plot([true_ph, ph], [i, i], color=HAIRLINE, lw=1.0, zorder=1)
        ax.plot(true_ph, i, marker="|", ms=11, color=INK_2, mew=1.4, zorder=4)
        ax.plot(ph, i, marker="o", ms=MARKER_S + 1,
                mfc=color if detectable else SURFACE,
                mec=color, mew=1.6, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("Peak phase (CT, h)")
    ax.set_title("Recovered peak phase vs the planted value")
    ax.axhline(len(order) - 0.5, color=HAIRLINE, lw=1.0)
    ax.text(23.6, len(order) - 0.36, "borderline detection", fontsize=7.5,
            color=INK_2, ha="right", va="top")
    handles = [
        Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=MARKER_S + 1,
               label="Core clock"),
        Line2D([], [], marker="o", ls="", mfc=ORANGE, mec=ORANGE, ms=MARKER_S + 1,
               label="Clock-controlled"),
        Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK_2, mew=1.6,
               ms=MARKER_S + 1, label="borderline"),
        Line2D([], [], marker="|", ls="", color=INK_2, ms=9, label="planted phase"),
    ]
    legend_below(ax, handles=handles, labels=[h.get_label() for h in handles],
                 ncol=4, y=-0.13)
    tidy(ax)
    save(fig, "t1-phase-map")


def fig_t1_detection_limit():
    """Recovered amplitude against planted amplitude, with the three bands."""
    fig, ax = new_fig(6.4, 4.0)
    bands = [
        (0.0, 0.20, "#F4F4F3", "below the noise floor"),
        (0.20, 0.46, "#EDEDEB", "borderline"),
    ]
    for lo, hi, fill, _ in bands:
        ax.axhspan(lo, hi, color=fill, lw=0, zorder=0)
    ax.axhline(0.46, color=HAIRLINE, lw=1.0, zorder=1)
    ax.axhline(0.20, color=HAIRLINE, lw=1.0, zorder=1)

    for name in cols1:
        planted = tr1.True_log2_amplitude[name]
        rec = AMP1[name]
        det = tr1.Detectable[name]
        color = {"yes": BLUE, "borderline": ORANGE, "no": MUTED}[det]
        ax.plot(planted, rec, marker="o", ms=MARKER_S + 1, mfc=color,
                mec=SURFACE, mew=1.2, ls="", zorder=3)
    lim = 2.4
    ax.plot([0, lim], [0, lim], color=HAIRLINE, lw=1.0, zorder=1)

    for name, dx, dy, ha in [("NR1D1", -0.06, 0.06, "right"),
                             ("DBP", -0.06, 0.06, "right"),
                             ("CLOCK", 0.07, 0.02, "left"),
                             ("RORA", 0.07, 0.0, "left"),
                             ("ACTB", 0.07, 0.0, "left")]:
        ax.annotate(name, (tr1.True_log2_amplitude[name] + dx, AMP1[name] + dy),
                    fontsize=8, color=INK, ha=ha, va="bottom")

    ax.text(2.32, 0.10, "below the noise floor", fontsize=7.5, color=INK_2,
            ha="right", va="center")
    ax.text(2.32, 0.33, "borderline", fontsize=7.5, color=INK_2,
            ha="right", va="center")
    ax.set_xlabel("Planted amplitude (log2)")
    ax.set_ylabel("Recovered 24 h cosinor amplitude (log2)")
    ax.set_title("What this design can and cannot detect")
    ax.set_xlim(-0.08, 2.4)
    ax.set_ylim(0, 2.4)
    handles = [Line2D([], [], marker="o", ls="", mfc=c, mec=SURFACE, mew=1.2,
                      ms=MARKER_S + 1, label=l)
               for c, l in [(BLUE, "detectable"), (ORANGE, "borderline"),
                            (MUTED, "not detectable")]]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    for t_ in leg.get_texts():
        t_.set_color(INK)
    tidy(ax)
    save(fig, "t1-detection-limit")


def fig_t1_period_fails():
    """48 h does not constrain a period. One panel, one message."""
    per = methods.period_estimation(clean1, cols1, "Time",
                                    method="Lomb-Scargle Periodogram",
                                    min_period=16, max_period=32).astype(float)
    rhy = tr1[tr1.Detectable == "yes"].index
    vals = per[rhy].sort_values()
    fig, ax = new_fig(6.4, 3.2)
    ax.axvspan(vals.min(), vals.max(), color="#F4F4F3", lw=0, zorder=0)
    ax.axvline(24.0, color=INK_2, lw=1.2, zorder=1)
    ax.plot(vals.values, np.arange(len(vals)), marker="o", ms=MARKER_S + 1,
            ls="", mfc=BLUE, mec=SURFACE, mew=1.2, zorder=3)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index, fontsize=7.5)
    ax.set_ylim(len(vals) + 0.4, -1.4)
    ax.set_xlim(21.5, 26.5)
    ax.set_xlabel("Estimated period (h) — Lomb-Scargle")
    ax.annotate("every one of these\nwas planted at 24.0 h",
                xy=(24.0, -1.25), xytext=(24.2, -1.25), fontsize=8,
                color=INK, va="top")
    for x, ha in ((vals.min(), "right"), (vals.max(), "left")):
        ax.annotate(f"{x:.2f} h", xy=(x, -0.75), fontsize=7.5,
                    color=INK_2, ha=ha, va="center")
    ax.set_title("Two cycles is not enough to pin a period")
    tidy(ax)
    save(fig, "t1-period-fails")


def fig_t1_data_shape():
    """The replicate grid, so the missing-value behaviour is visible."""
    present = d1.assign(_rep=d1.groupby("Time").cumcount() + 1)
    times = sorted(d1["Time"].unique())
    fig, ax = new_fig(7.2, 2.0)
    for _, row in present.iterrows():
        x = times.index(row["Time"])
        y = row["_rep"]
        complete = not row[cols1].isna().any()
        ax.add_patch(Rectangle((x - 0.38, y - 0.38), 0.76, 0.76,
                               facecolor=BLUE if complete else "#F4C9AE",
                               edgecolor=SURFACE, lw=1.6))
    ax.set_xlim(-0.7, len(times) - 0.3)
    ax.set_ylim(0.4, 3.6)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f"{t:.0f}" for t in times], fontsize=8)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["rep 1", "rep 2", "rep 3"], fontsize=8)
    ax.set_xlabel("Time (h)")
    ax.set_title("Every row in the file", pad=22)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.annotate("replicate never collected", xy=(5, 3.32),
                fontsize=7.5, color=INK, ha="center", va="bottom")
    handles = [
        Line2D([], [], marker="s", ls="", mfc=BLUE, mec=SURFACE, ms=8,
               label="complete"),
        Line2D([], [], marker="s", ls="", mfc="#F4C9AE", mec=SURFACE, ms=8,
               label="missing value in PPARGC1A — whole row dropped"),
    ]
    legend_below(ax, handles=handles, labels=[h.get_label() for h in handles],
                 ncol=2, y=-0.45)
    tidy(ax)
    save(fig, "t1-data-shape")


# ═══════════════════════════════════════════════════════════════════════════
#  Tutorial 2
# ═══════════════════════════════════════════════════════════════════════════

def fig_t2_raw():
    """One well, whole recording, with the two things you must deal with."""
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    styles.apply("Journal", "notebook", "Chronotopia")

    t = d2["Time"].to_numpy()
    y = d2["A1"].to_numpy()
    ax.axvspan(0, RELEASE_H, color=ZEITGEBER_FILL, lw=0, zorder=0)
    ax.plot(t, y, color=BLUE, lw=1.0, zorder=3)
    ax.set_ylim(0, 6800)
    day_ticks(ax, 144)
    ax.set_ylabel("Bioluminescence (counts/s)")
    ax.set_title("Well A1 — the whole recording")
    ax.annotate("zeitgeber on", xy=(24, 6450), fontsize=8, color=INK_2,
                ha="center", va="center")
    ax.annotate("released", xy=(RELEASE_H + 3, 6450), fontsize=8, color=INK,
                va="center")
    ax.axvline(RELEASE_H, color=INK_2, lw=1.0, zorder=2)
    # Anchor on a point that is actually inside the axes — the trace itself is
    # off the top of this scale for the first hour and a half.
    t_anchor = float(t[np.argmax(y < 6000)])
    ax.annotate("medium change\nruns off this scale (~12 000)",
                xy=(t_anchor, 5950), xytext=(9, 900), fontsize=7.5,
                color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.9,
                                shrinkA=0, shrinkB=2))

    m = t < 8
    ax2.plot(t[m], y[m], color=BLUE, lw=1.4)
    ax2.axvspan(0, 3, color="#F7E2D3", lw=0, zorder=0)
    ax2.set_xlim(0, 8)
    ax2.set_xlabel("Time (h)")
    ax2.set_title("First 8 hours")
    ax2.annotate("discard", xy=(1.5, y[m].max() * 0.55), fontsize=8,
                 color=INK, ha="center")
    for a in (ax, ax2):
        tidy(a)
    fig.tight_layout()
    save(fig, "t2-raw")


def fig_t2_detrend():
    """Why detrending is not optional on a six-day recording."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
    styles.apply("Journal", "notebook", "Chronotopia")
    t = free2["Time"].to_numpy()
    raw = free2["A1"].to_numpy()
    # A centred 24 h window is undefined in the first and last 12 h; drawing it
    # there with min_periods would show a hook that is an artefact of the edge,
    # not of the data.
    base = pd.Series(raw).rolling(144, center=True, min_periods=144).mean()

    axes[0].plot(t, raw, color=BLUE, lw=1.0, label="A1, raw")
    axes[0].plot(t, base, color=INK_2, lw=1.6, label="24 h running mean")
    axes[0].set_ylabel("counts/s")
    axes[0].set_title("Raw free-running trace — the baseline falls throughout")
    tidy(axes[0], loc="upper right")

    axes[1].axhline(0, color=HAIRLINE, lw=1.0)
    axes[1].plot(t, det2["A1"].to_numpy(), color=BLUE, lw=1.0)
    axes[1].set_ylabel("counts/s")
    axes[1].set_title("After Detrending → Rolling mean")
    day_ticks(axes[1], 144)
    axes[1].set_xlim(48, 144)
    axes[1].set_xticks(np.arange(48, 145, 24))
    tidy(axes[1])
    fig.tight_layout()
    save(fig, "t2-detrend")


def fig_t2_genotypes():
    """The four conditions drifting apart after release."""
    fig, ax = new_fig(7.2, 3.6)
    # The first hours after release carry the detrending window's edge, so the
    # plot starts once the rolling mean is fully defined.
    seg = det2[det2["Time"] >= 56].reset_index(drop=True)
    t = seg["Time"].to_numpy()
    labels = []
    for cond in ["Short period", "Wild type", "Long period", "Arrhythmic"]:
        wells = tr2[tr2.Condition == cond].index
        y = seg[wells].mean(axis=1).to_numpy()
        color = GENOTYPE_COLOR[cond]
        ax.plot(t, y, color=color, lw=LINE_W, label=cond, zorder=3)
        labels.append((y[-1], cond, color))
    ax.axhline(0, color=HAIRLINE, lw=1.0, zorder=1)
    ax.set_xlim(56, 144)
    ax.set_ylim(-1150, 1150)
    ax.set_xticks(np.arange(72, 145, 24))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Detrended signal (counts/s)")
    ax.set_title("Free-run, mean of 6 wells per genotype — the rows drift apart")
    end_labels(ax, labels, x=144, dx=8.0)
    legend_below(ax, ncol=4, y=-0.22)
    tidy(ax)
    save(fig, "t2-genotypes")


def fig_t2_period_methods():
    """Every method against the planted period. The figure of tutorial 2."""
    rhythmic = tr2[tr2.Rhythmic_after_release == "yes"]
    conds = ["Short period", "Wild type", "Long period"]
    fig, ax = new_fig(6.8, 3.4)

    # The planted period is a band spanning the full row, so every dot is read
    # against it directly rather than against a bar sitting to one side.
    for i, cond in enumerate(conds):
        wells = rhythmic[rhythmic.Condition == cond].index
        truth = rhythmic.True_intrinsic_period_h[wells]
        ax.add_patch(Rectangle((truth.min(), i - 0.42),
                               truth.max() - truth.min(), 0.84,
                               facecolor="#E4E4E1", edgecolor="none", zorder=0))
        ax.annotate(f"planted {truth.min():.1f}–{truth.max():.1f} h",
                    xy=(truth.max() + 0.12, i - 0.30), fontsize=7.5,
                    color=INK_2, va="center")

    offsets = {"Lomb-Scargle Periodogram": -0.22,
               "Wavelet Transform": 0.0, "Autocorrelation": 0.22}
    for meth, dy in offsets.items():
        est = methods.period_estimation(det2, cols2, "Time", method=meth,
                                        min_period=16, max_period=32).astype(float)
        for i, cond in enumerate(conds):
            wells = rhythmic[rhythmic.Condition == cond].index
            v = est[wells]
            ax.plot(v.values, np.full(len(v), i + dy), marker="o",
                    ms=MARKER_S + 1, ls="", mfc=METHOD_COLOR[meth], mec=SURFACE,
                    mew=1.2, zorder=3, label=meth if i == 0 else None)

    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds, fontsize=8.5)
    ax.set_ylim(len(conds) - 0.45, -0.55)
    ax.set_xlabel("Estimated period (h)")
    ax.set_xlim(20.5, 27.8)
    ax.set_title("Three methods, six wells each, against the planted period")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Rectangle((0, 0), 1, 1, facecolor="#E4E4E1", edgecolor="none"))
    labels.append("planted period")
    legend_below(ax, handles=handles, labels=labels, ncol=4, y=-0.20)
    tidy(ax)
    save(fig, "t2-period-methods")


def fig_t2_plate():
    """A plate map coloured by period — diverging around 24 h, neutral midpoint."""
    # The plate overlay in the app is driven by the cosinor fit, not by the
    # sidebar's period method, so this figure uses the same source: one sine
    # sweep per well, with R^2 deciding which wells carry a period at all.
    sweep, _ = methods.sine_sweep(det2, "Time", cols2,
                                  period_min=16, period_max=32, period_step=0.05)
    sweep = sweep.set_index("sample")
    est = sweep["period"]
    good = sweep["r2"] >= 0.5

    cmap = plt.get_cmap("coolwarm_r")
    norm = plt.Normalize(24 - 3, 24 + 3)

    fig, ax = new_fig(6.2, 2.9)
    for r, row in enumerate("ABCD"):
        for c in range(1, 7):
            well = f"{row}{c}"
            val = est[well]
            face = cmap(norm(val)) if good[well] else "#EFEFED"
            ax.add_patch(Rectangle((c - 0.44, r - 0.44), 0.88, 0.88,
                                   facecolor=face, edgecolor=SURFACE, lw=2.0))
            txt = f"{val:.1f}" if good[well] else "n/a"
            ax.text(c, r, txt, ha="center", va="center", fontsize=7.5,
                    color=INK if good[well] else INK_2)
    ax.set_xlim(0.4, 6.6)
    ax.set_ylim(-0.6, 3.6)
    ax.invert_yaxis()
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(4))
    ax.set_yticklabels(list("ABCD"))
    ax.tick_params(length=0, labelsize=8)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Plate view, overlay = Period (h) — row is genotype, "
                 "grey = no rhythm to fit")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.03)
    cb.set_label("Period (h)", size=8, color=INK, labelpad=8)
    cb.ax.tick_params(labelsize=7.5, colors=INK, pad=2)
    cb.outline.set_visible(False)
    save(fig, "t2-plate")


def fig_t2_driven_vs_clock():
    """An arrhythmic well follows the zeitgeber. That is not a clock."""
    fig, ax = new_fig(7.2, 3.2)
    # Trimmed at 3 h and 12 h respectively: the medium-change transient and the
    # centred window's edge would both set the y-scale for the whole panel.
    seg = d2[(d2["Time"] >= 12) & (d2["Time"] <= 132)].reset_index(drop=True)
    t = seg["Time"].to_numpy()
    for well, cond in [("A1", "Wild type"), ("D1", "Arrhythmic")]:
        s = seg[well]
        y = (s - s.rolling(144, center=True, min_periods=1).mean()).to_numpy()
        color = GENOTYPE_COLOR[cond]
        ax.plot(t, y, color=color, lw=1.2, label=f"{well} — {cond}", zorder=3)
    ax.axvspan(12, RELEASE_H, color=ZEITGEBER_FILL, lw=0, zorder=0)
    ax.axvline(RELEASE_H, color=INK_2, lw=1.0, zorder=2)
    ax.set_xlim(12, 132)
    ax.set_ylim(-1500, 1900)
    ax.set_xticks(np.arange(24, 133, 24))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Detrended signal (counts/s)")
    ax.set_title("Both oscillate while the zeitgeber is on; only one keeps going")
    ax.annotate("driven", xy=(30, 1620), fontsize=8, color=INK_2, ha="center")
    ax.annotate("free-running", xy=(90, 1620), fontsize=8, color=INK_2,
                ha="center")
    legend_below(ax, ncol=2, y=-0.22)
    tidy(ax)
    save(fig, "t2-driven-vs-clock")


# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ═══════════════════════════════════════════════════════════════════════════
#
# These run the real extractor over the long-series tutorial data, so the page
# quotes numbers the reader can reproduce rather than illustrative ones.

_FEATURES = {}


def feature_matrix():
    """The 100-feature matrix for the detrended free-running segment, cached."""
    if "fx" not in _FEATURES:
        import features as F
        from chronotopia_feature_extractor import ChronotopiaFeatureExtractor
        with F.silence_extractor_warnings():
            fx = ChronotopiaFeatureExtractor.extract_batch(
                det2, t_col="Time", data_cols=cols2, verbose=False)
        _FEATURES["fx"] = fx
    return _FEATURES["fx"]


def fig_feat_packages():
    """What each feature package looks at, for one well.

    The overlays are drawn by the extractor's own `plot_*` methods, so this is
    the app's view rather than a reimplementation. The panel layout is ours:
    `plot_summary` builds its own multi-panel figure, but as of v0.8.0 that
    figure overlaps its axes and leaves the spectral row empty, so the panels
    are placed here instead.

    The trace-overlay plotters (cosinor, cycles, baseline) assume the signal is
    already on the axes — they draw only their own annotation — so the raw trace
    goes down first.
    """
    from matplotlib.ticker import FuncFormatter
    from chronotopia_feature_extractor import ChronotopiaFeatureExtractor

    styles.apply("Journal", "notebook", "Chronotopia")
    # signal first, then time — the constructor's order, not (t, y).
    t = det2["Time"].to_numpy()
    y = det2["A1"].to_numpy()
    ext = ChronotopiaFeatureExtractor(y, t)

    fig, axes = plt.subplots(4, 2, figsize=(9.6, 11.0))
    axes = axes.ravel()

    panels = [
        ("cosinor", "Cosinor — one sine, fitted",
         "period, amplitude, MESOR, acrophase, R²"),
        ("damped_cosinor", "Damped cosinor — the decay, fitted",
         "period, amplitude, damping time constant, half-life"),
        ("cycles", "Cycles — peaks and troughs found",
         "peak intervals, prominences, complete cycles"),
        ("waveform", "Waveform — the shape of a cycle",
         "rise and fall time, FWHM, asymmetry"),
        ("baseline", "Baseline — what is not rhythmic",
         "slope, depletion, non-stationarity"),
        ("lomb_scargle", "Lomb-Scargle — the periodogram",
         "peak period and power, bandwidth, FAP"),
        ("wavelet_ridge", "Wavelet ridge — period over time",
         "instantaneous period, drift, damping, half-life"),
    ]

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    for ax, (pkg, title, sub) in zip(axes, panels):
        if pkg in ("cosinor", "damped_cosinor", "cycles", "baseline"):
            ax.plot(t, y, color=MUTED, lw=0.7, zorder=1)
        getattr(ext, f"plot_{pkg}")(ax, color=BLUE)
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=8.5, color=INK,
                     linespacing=1.5)
        ax.title.set_fontsize(8.5)
        # Some plotters set their own limits from an internal, resampled x.
        if pkg in ("cosinor", "damped_cosinor", "cycles", "waveform", "baseline"):
            ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=7, colors=INK)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)
        leg = ax.get_legend()
        if leg is not None:
            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(handles, labels, loc="upper center",
                            bbox_to_anchor=(0.5, -0.30), ncol=3,
                            frameon=False, fontsize=6.5)
            for txt in leg.get_texts():
                txt.set_color(INK)

    # plot_wavelet_ridge plots against sample index rather than hours. Index i
    # is t[0] + i*dt, so relabelling the ticks is exact, not a fudge.
    wav = axes[len(panels) - 1]
    t0, dt = float(t[0]), float(np.median(np.diff(t)))
    wav.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{t0 + v * dt:.0f}"))
    wav.set_xlabel("Time (h)")

    for ax in axes[:len(panels) - 1]:
        ax.set_xlabel("Time (h)")

    fig.suptitle("Well A1, free-running segment — what each package measures",
                 fontsize=10, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "feat-packages", pad=0.05)


def fig_feat_period_estimators():
    """Six independent period estimates, against the planted value.

    Six categories, five hues that pass the all-pairs colour-vision gate — so the
    two cosinor variants share BLUE and are told apart by marker shape, the same
    hue-and-shape scheme `plots.feature_volcano` uses. Sharing the hue between the
    plain and the damped cosinor is deliberate: they are the same model with one
    extra term.
    """
    fx = feature_matrix().set_index("sample_id")
    est = {
        "Cosinor fit": "cosinor_period",
        "Damped cosinor": "damped_cosinor_period",
        "Peak intervals": "cycles_period_event_based",
        "Lomb-Scargle": "lomb_scargle_peak_period_h",
        "Wavelet ridge": "wavelet_ridge_period_mean",
        "FFT fundamental": "harmonic_fundamental_period_h",
    }
    colors = [BLUE, BLUE, SKY, VIOLET, AMBER, ORANGE]
    markers = ["o", "s", "o", "o", "o", "o"]
    conds = ["Short period", "Wild type", "Long period"]
    rhythmic = tr2[tr2.Rhythmic_after_release == "yes"]

    fig, ax = new_fig(7.2, 3.8)
    for i, cond in enumerate(conds):
        wells = rhythmic[rhythmic.Condition == cond].index
        truth = rhythmic.True_intrinsic_period_h[wells]
        ax.add_patch(Rectangle((truth.min(), i - 0.44),
                               truth.max() - truth.min(), 0.88,
                               facecolor="#E4E4E1", edgecolor="none", zorder=0))

    offs = np.linspace(-0.32, 0.32, len(est))
    for (label, col), color, marker, dy in zip(est.items(), colors, markers, offs):
        for i, cond in enumerate(conds):
            wells = rhythmic[rhythmic.Condition == cond].index
            v = fx.loc[wells, col]
            ax.plot(v.values, np.full(len(v), i + dy), marker=marker, ms=MARKER_S,
                    ls="", mfc=color, mec=SURFACE, mew=1.1, zorder=3,
                    label=label if i == 0 else None)

    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds, fontsize=8.5)
    ax.set_ylim(len(conds) - 0.4, -0.6)
    ax.set_xlim(19.5, 28)
    ax.set_xlabel("Estimated period (h)")
    ax.set_title("Twenty of the 108 features are period. They do not agree.")

    # The FFT fundamental is the point of the figure: it returns the same value
    # in all three rows, so a rule through all of them says it better than an
    # arrow into one.
    fft_med = float(fx.loc[rhythmic.index, "harmonic_fundamental_period_h"].median())
    ax.plot([fft_med, fft_med], [-0.45, len(conds) - 0.55], color=ORANGE,
            lw=1.2, zorder=1)
    ax.annotate(f"every FFT estimate lands on {fft_med:.2f} h,\n"
                f"whatever the genotype — the frequency\nbins are too coarse to "
                f"tell them apart",
                xy=(fft_med + 0.15, -0.42), fontsize=7.5, color=INK,
                ha="left", va="top")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Rectangle((0, 0), 1, 1, facecolor="#E4E4E1", edgecolor="none"))
    labels.append("planted period")
    legend_below(ax, handles=handles, labels=labels, ncol=4, y=-0.18)
    tidy(ax)
    save(fig, "feat-period-estimators")


def fig_feat_redundancy():
    """Absolute Spearman correlation among the clustered features."""
    import features as F
    fx = feature_matrix()
    clusters = F.redundancy_clusters(fx)
    order = clusters.sort_values(["cluster", "feature"])
    names = order.feature.tolist()

    corr = fx[names].corr(method="spearman").abs()
    fig, ax = new_fig(6.6, 6.0)
    im = ax.imshow(corr.to_numpy(), cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")

    # Cluster boundaries, drawn in the surface colour so the blocks separate
    # without a stroke around each cell.
    bounds, start = [], 0
    for _, grp in order.groupby("cluster", sort=True):
        bounds.append((start, start + len(grp), grp.concept.iloc[0]))
        start += len(grp)
    gutter = len(names) + 1.5
    for lo, hi, concept in bounds:
        ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo,
                               fill=False, edgecolor=ORANGE, lw=1.4))
        mid = (lo + hi - 1) / 2
        ax.plot([hi - 0.5, gutter - 0.4], [mid, mid], color=HAIRLINE, lw=0.8,
                clip_on=False, zorder=2)
        ax.annotate(concept, xy=(gutter, mid), fontsize=6.5, color=INK,
                    va="center", ha="left", annotation_clip=False)

    ax.set_xticks([])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5.5)
    ax.tick_params(length=0)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{len(names)} of 100 features fall into "
                 f"{len(bounds)} redundancy clusters", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.12)
    cb.set_label("|Spearman rho|", size=8, color=INK, labelpad=8)
    cb.ax.tick_params(labelsize=7.5, colors=INK, pad=2)
    cb.outline.set_visible(False)
    save(fig, "feat-redundancy")


def fig_feat_effects():
    """Which concepts separate which comparison.

    Cliff's delta saturates at 1.0 whenever two groups of six do not overlap,
    which is most of the significant features here — so plotting the effect
    size would put nearly every point on the same tick. What actually varies
    between comparisons is *which concepts* respond, so that is what is drawn.
    """
    import features as F
    fx = feature_matrix().copy()
    fx["Condition"] = fx.sample_id.map(tr2.Condition)

    comparisons = [
        ("Wild type vs arrhythmic", "Wild type", "Arrhythmic", BLUE),
        ("Short vs long period", "Short period", "Long period", ORANGE),
    ]
    frames = {}
    for label, a, b, _ in comparisons:
        res, meta = F.compare_conditions(fx, "Condition", a, b)
        frames[label] = (res, meta)

    concepts = [c for c in F.CONCEPT_ORDER
                if any(c in set(r.concept) for r, _ in frames.values())]
    fig, ax = new_fig(7.2, 4.2)
    h = 0.34
    for j, (label, _, _, color) in enumerate(comparisons):
        res, meta = frames[label]
        offset = (j - 0.5) * h
        for i, concept in enumerate(concepts):
            grp = res[res.concept == concept]
            if not len(grp):
                continue
            frac = grp.significant.mean()
            ax.add_patch(Rectangle((0, i + offset - h / 2 + 0.02), frac, h - 0.04,
                                   facecolor=color, edgecolor="none", zorder=3))
            ax.annotate(f"{int(grp.significant.sum())}/{len(grp)}",
                        xy=(frac + 0.015, i + offset), fontsize=7,
                        color=INK_2, va="center")

    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=8)
    ax.set_ylim(len(concepts) - 0.45, -0.75)
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of that concept's features surviving FDR")
    ax.set_title("Different comparisons light up different concepts")
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
               for *_, c in comparisons]
    labels = [f"{lab} (n={frames[lab][1]['n_tested']} tested)"
              for lab, *_ in comparisons]
    legend_below(ax, handles=handles, labels=labels, ncol=2, y=-0.16)
    tidy(ax)
    save(fig, "feat-effects")


def fig_feat_volcano():
    """The volcano the Compare conditions view draws, straight from plots.py.

    Rendered by the app's own function rather than redrawn here, so the page can
    never show a figure the app does not produce.
    """
    import features as F
    fx = feature_matrix().copy()
    fx["Condition"] = fx.sample_id.map(tr2.Condition)
    res, meta = F.compare_conditions(fx, "Condition", "Wild type", "Arrhythmic")
    styles.apply("Journal", "notebook", "Chronotopia")
    fig = plots.feature_volcano(res, meta, alpha=meta["alpha"])
    save(fig, "feat-volcano")


def fig_pp_baselines():
    """What each detrending estimator thinks the baseline is, on one real well.

    Five estimators as of v0.7.7 — Running median and Sinc low-pass were dropped
    after `detrend_redundancy.py` measured what they were worth.

    Same well and same trimming as tutorial 2, so a reader moving between the two
    pages is looking at the same trace.
    """
    from scipy.optimize import curve_fit

    d = d2[d2["Time"] >= 3.0].reset_index(drop=True)      # drop the medium-change spike
    t = d["Time"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    y = d["A1"].to_numpy(float)
    frame = pd.DataFrame({"Time": t, "A1": y})
    win = int(round(24 / dt))

    panels = [
        ("Rolling mean", methods.rolling_baseline(frame, ["A1"], win)["A1"]),
        ("LOESS", methods.loess_baseline(frame, ["A1"], "Time", 48.0)["A1"]),
        ("Exponential fit", methods.exponential_baseline(frame, ["A1"], "Time")["A1"]),
        ("Cubic", methods.polynomial_baseline(frame, ["A1"], "Time", 3)["A1"]),
        ("Linear", methods.polynomial_baseline(frame, ["A1"], "Time", 1)["A1"]),
    ]

    styles.apply("Journal", "notebook", "Chronotopia")
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 4.6), sharex=True, sharey=True)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, (name, base) in zip(axes.ravel(), panels):
        ax.plot(t, y, color=MUTED, lw=0.7, alpha=0.85, zorder=1)
        ax.plot(t, base, color=BLUE, lw=LINE_W, zorder=3, solid_capstyle="round")
        ax.set_title(name, fontsize=9, color=INK, pad=4, loc="left")
        ax.set_xticks(np.arange(0, 145, 48))
        ax.grid(True, color=HAIRLINE, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    # sharex hides tick labels on all but the last row, and with five panels in a
    # 2x3 grid the last row does not reach the third column — that panel would be
    # left floating over an unlabelled axis. Label the bottom-most VISIBLE panel of
    # each column instead.
    ncol = axes.shape[1]
    for col in range(ncol):
        rows = [r for r in range(axes.shape[0]) if r * ncol + col < len(panels)]
        if rows:
            ax = axes[max(rows)][col]
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Counts/s")
    fig.legend(handles=[Line2D([], [], color=MUTED, lw=1.2, label="Raw trace (well A1)"),
                        Line2D([], [], color=BLUE, lw=LINE_W, label="Estimated baseline")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    save(fig, "pp-baselines")


def fig_pp_subtract_vs_divide():
    """Why the removal mode matters more than the estimator.

    The generator builds these traces as baseline(t) * (1 + envelope(t) * wave),
    so the baseline is a multiplying factor and the planted damping tau is known.
    Subtracting leaves the residual still multiplied by a falling baseline.
    """
    from pyboat import sliding_window_amplitude

    d = d2[d2["Time"] >= 3.0].reset_index(drop=True)
    t = d["Time"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    win = int(round(24 / dt))
    free = (t >= 48) & (t <= 144)
    tt = t[free]

    well = "C6"
    y = d[well].to_numpy(float)
    frame = pd.DataFrame({"Time": t, well: y})
    base = methods.rolling_baseline(frame, [well], win)[well].to_numpy()

    sub = (y - base)[free]
    div = (y / base - 1.0)[free]
    amp0 = float(tr2.loc[well, "Initial_relative_amplitude"])
    planted = amp0 * 1.0263 * np.exp(-(tt - 48) / 142.0)   # 1.0263 = half p-p of the wave

    styles.apply("Journal", "notebook", "Chronotopia")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    for ax, (title, x, ref, col, unit) in zip(axes, [
            ("Subtract the baseline   ( y − baseline )", sub,
             planted * float(base[free][0]), BLUE, "Counts/s"),
            ("Divide by the baseline   ( y ÷ baseline )", div,
             planted, ORANGE, "Relative amplitude")]):
        env = sliding_window_amplitude(x, window_size=24.0, dt=dt)
        ax.axhline(0, color=HAIRLINE, lw=1)
        ax.plot(tt, x, color=col, lw=0.85, alpha=0.9, zorder=2)
        for sign in (1, -1):
            ax.plot(tt, sign * env, color=INK, lw=1.5, zorder=4)
            ax.plot(tt, sign * ref, color=VIOLET, lw=1.7, ls=(0, (4, 2)), zorder=5)
        good = env > 1e-12
        tau = -1 / np.polyfit(tt[good], np.log(env[good]), 1)[0]
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
        ax.text(0.5, 0.97, f"measured τ = {tau:.0f} h", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color=INK_2)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(unit, labelpad=6)
        ax.set_xticks(np.arange(48, 145, 24))
        ax.set_ylim(-1.55 * float(np.max(env)), 1.9 * float(np.max(env)))
        ax.grid(True, color=HAIRLINE, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.5, label="Measured envelope"),
                        Line2D([], [], color=VIOLET, lw=1.7, ls=(0, (4, 2)),
                               label="Damping actually planted in the data (τ = 142 h)")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=[0, 0.06, 1, 0.99], w_pad=3.0)
    save(fig, "pp-subtract-vs-divide")


# ═══════════════════════════════════════════════════════════════════════════

FIGURES = [
    fig_t1_traces, fig_t1_phase_map, fig_t1_detection_limit,
    fig_t1_period_fails, fig_t1_data_shape,
    fig_t2_raw, fig_t2_detrend, fig_t2_genotypes,
    fig_t2_period_methods, fig_t2_plate, fig_t2_driven_vs_clock,
    fig_feat_packages, fig_feat_period_estimators, fig_feat_redundancy,
    fig_feat_effects, fig_feat_volcano,
    fig_pp_baselines, fig_pp_subtract_vs_divide,
]


def stage_downloads() -> None:
    """Copy the tutorial datasets where MkDocs can serve them."""
    import shutil
    dest = os.path.join(REPO, "docs", "downloads")
    os.makedirs(dest, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(TUT)):
        if name.endswith(".csv"):
            shutil.copy2(os.path.join(TUT, name), os.path.join(dest, name))
            n += 1
    print(f"{n} datasets staged into docs/downloads/")


def main() -> None:
    print(f"writing to {OUT}")
    for fn in FIGURES:
        fn()
    print(f"{len(FIGURES)} figures, {len(FIGURES) * 2} files")
    stage_downloads()


if __name__ == "__main__":
    main()
