"""
plates.py
=========
Microplate detection and plate-shaped visualisation for Chronotopia.

What this module decides
------------------------
Given the sample column names of an uploaded dataset, work out whether the
experiment is a microplate and, if so, which well each sample sits in.

Detection is **name-first**. Well positions are read out of the column names
(`A1`, `A01`, `G8`, `sample_H12_ctrl`, ...) because that is what instruments
actually emit and it is the only evidence that survives re-ordering, partial
plates and excluded wells. Falling back to "there are 96 columns, so it must be
a plate" is offered only as a last resort, and is reported as such.

What this module deliberately does NOT do
-----------------------------------------
It does not invent experimental conditions. The previous behaviour assigned
`Condition = plate column` to any 96-column file, which silently produced
grouped statistics nobody asked for. Geometry (Well / Row / Col) is a fact about
the file; grouping is a claim about the experiment. This module supplies the
first and leaves the second to `group_by_geometry`, which the user opts into.

Public API
----------
    parse_well("sample_H12")        -> ("H", 12)   or None
    detect_plate(sample_names)      -> PlateLayout or None
    group_by_geometry(layout, "Row")-> pd.Series of condition labels
    plot_plate(...)                 -> matplotlib Figure
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ── plate geometry ───────────────────────────────────────────────────────────
# size -> (n_rows, n_cols). SBS standard formats, smallest first: detection
# picks the smallest format that can hold every well it parsed.
PLATE_FORMATS: dict[int, tuple[int, int]] = {
    6:   (2, 3),
    12:  (3, 4),
    24:  (4, 6),
    48:  (6, 8),
    96:  (8, 12),
    384: (16, 24),
}

ROW_LETTERS = string.ascii_uppercase  # A..P covers 16 rows (384-well)

# A well ID is a single letter A-P followed by 1-2 digits. The lookbehind stops
# us matching the tail of a word ("Cond" would otherwise yield "d..."), and the
# trailing (?!\d) stops "P53" being read as row P column 5.
_WELL_RE = re.compile(r"(?<![A-Za-z0-9])([A-Pa-p])[\s_\-]?(\d{1,2})(?!\d)")

# Minimum fraction of columns that must yield a well ID before we trust names.
_NAME_CONFIDENCE = 0.60

# Highest column index on any supported plate — anything past this is not a well.
_MAX_COL = max(c for _, c in PLATE_FORMATS.values())

# Formats we are willing to infer from the SAMPLE COUNT ALONE, with no well IDs
# in the names. Restricted on purpose: a file with 12 or 24 columns is far more
# likely to be 12 or 24 ordinary samples than a 12- or 24-well plate, and calling
# that a plate is a false positive the user then has to notice and undo. A file
# with exactly 96 or 384 columns is a much stronger coincidence.
_POSITIONAL_SIZES = {96, 384}


def parse_well(name) -> tuple[str, int] | None:
    """
    Extract (row_letter, column_number) from a sample name.

    Returns None when the name carries no well ID. Case-insensitive; tolerates
    zero padding, separators and surrounding text:

        "A1"                -> ("A", 1)
        "a01"               -> ("A", 1)
        "G8"                -> ("G", 8)
        "Well B02"          -> ("B", 2)
        "sample_H12_ctrl"   -> ("H", 12)
        "Plate1_C7"         -> ("C", 7)
        "Time"              -> None
        "sample_1"          -> None      (no row letter)
        "COL_12"            -> None
        "A0"                -> None      (plates are 1-indexed)
        "P53"               -> None      (no plate has 53 columns)

    When a name contains more than one candidate the LAST is used, so a plate or
    batch prefix ("Plate2_A01") does not win over the real well. Candidates whose
    column number is out of range are skipped rather than failing the whole name,
    so "P53_A07" still resolves to A7.
    """
    if name is None:
        return None
    for letter, number in reversed(_WELL_RE.findall(str(name))):
        col = int(number)
        if 1 <= col <= _MAX_COL:
            return letter.upper(), col
    return None


@dataclass
class PlateLayout:
    """The result of a successful plate detection."""
    n_rows: int
    n_cols: int
    size: int
    wells: pd.DataFrame        # Sample, Well, Row, Col, RowIdx, ColIdx
    source: str                # "names" or "count"
    n_named: int = 0           # how many samples carried a parsable well ID
    n_total: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.size}-well ({self.n_rows}×{self.n_cols})"

    def well_at(self, row_idx: int, col_idx: int):
        """Sample name at a zero-based grid position, or None if the well is empty."""
        hit = self.wells[(self.wells.RowIdx == row_idx) & (self.wells.ColIdx == col_idx)]
        return None if hit.empty else hit.iloc[0]["Sample"]

    def describe(self) -> str:
        if self.source == "names":
            msg = (f"Detected a {self.label} plate from the sample names "
                   f"({self.n_named}/{self.n_total} wells identified).")
        else:
            msg = (f"{self.n_total} samples matches a {self.label} plate, but no well IDs "
                   f"were found in the column names — wells were filled in upload order, "
                   f"left to right then top to bottom. Check the plate view before trusting it.")
        return " ".join([msg] + self.notes)


def _fit_format(rows_needed: int, cols_needed: int, n_samples: int):
    """Smallest standard format that holds the given grid extent and sample count."""
    for size, (r, c) in PLATE_FORMATS.items():
        if r >= rows_needed and c >= cols_needed and size >= n_samples:
            return size, r, c
    return None


def detect_plate(sample_names, allow_positional: bool = True) -> PlateLayout | None:
    """
    Decide whether `sample_names` describes a microplate.

    Strategy
    --------
    1. Parse a well ID out of every name. If at least 60% yield one, and those
       wells are unique, choose the smallest standard format that contains them
       all. Partial plates are fine — 40 samples spanning A1..D10 is reported as
       a 96-well plate with 56 empty wells, which is what it is.
    2. Otherwise, if the sample count is exactly 96 or 384 and `allow_positional`
       is set, fill row-major and flag the result as a guess. Smaller formats are
       deliberately excluded here — see `_POSITIONAL_SIZES`.
    3. Otherwise return None. A dataset is not a plate just because someone
       happened to measure 24 things.
    """
    names = list(sample_names)
    n = len(names)
    if n == 0:
        return None

    parsed = [parse_well(s) for s in names]
    named = [(s, p) for s, p in zip(names, parsed) if p is not None]
    notes: list[str] = []

    if named and len(named) / n >= _NAME_CONFIDENCE:
        wells = [p for _, p in named]
        if len(set(wells)) == len(wells):
            rows_needed = max(ROW_LETTERS.index(r) for r, _ in wells) + 1
            cols_needed = max(c for _, c in wells)
            fit = _fit_format(rows_needed, cols_needed, len(named))
            if fit is not None:
                size, n_rows, n_cols = fit
                if len(named) < n:
                    notes.append(
                        f"{n - len(named)} sample(s) had no recognisable well ID and are "
                        f"not shown on the plate."
                    )
                frame = pd.DataFrame(
                    [
                        {
                            "Sample": s,
                            "Well": f"{r}{c:02d}",
                            "Row": r,
                            "Col": c,
                            "RowIdx": ROW_LETTERS.index(r),
                            "ColIdx": c - 1,
                        }
                        for s, (r, c) in named
                    ]
                ).sort_values(["RowIdx", "ColIdx"]).reset_index(drop=True)
                return PlateLayout(n_rows, n_cols, size, frame, "names",
                                   len(named), n, notes)
        else:
            dupes = pd.Series([f"{r}{c:02d}" for r, c in wells])
            repeated = sorted(dupes[dupes.duplicated()].unique())[:4]
            notes.append(
                f"Well IDs repeat ({', '.join(repeated)}...), so they cannot be "
                f"positions on a single plate."
            )

    if allow_positional and n in _POSITIONAL_SIZES:
        n_rows, n_cols = PLATE_FORMATS[n]
        frame = pd.DataFrame(
            [
                {
                    "Sample": s,
                    "Well": f"{ROW_LETTERS[i // n_cols]}{i % n_cols + 1:02d}",
                    "Row": ROW_LETTERS[i // n_cols],
                    "Col": i % n_cols + 1,
                    "RowIdx": i // n_cols,
                    "ColIdx": i % n_cols,
                }
                for i, s in enumerate(names)
            ]
        )
        return PlateLayout(n_rows, n_cols, n, frame, "count", 0, n, notes)

    return None


def group_by_geometry(layout: PlateLayout, by: str = "None") -> pd.Series | None:
    """
    Turn plate geometry into condition labels — only when the user asks for it.

    by = "Row"     -> "Row A", "Row B", ...
    by = "Column"  -> "Col 01", "Col 02", ...
    by = "None"    -> None (no grouping; the caller leaves Condition unset)
    """
    if by == "Row":
        return layout.wells["Row"].map(lambda r: f"Row {r}")
    if by == "Column":
        return layout.wells["Col"].map(lambda c: f"Col {c:02d}")
    return None


# ── visualisation ────────────────────────────────────────────────────────────

def plot_plate(
    df,
    t_col,
    layout: PlateLayout,
    t0=None,
    t1=None,
    shared_y: bool = True,
    line_color: str = "#1F7A8C",
    bg_color: str = "white",
    empty_color: str = "#F2F2F0",
    show_sample_names: bool = False,
    show_well_ids: bool = False,
    annotations: dict | pd.Series | None = None,
    well_colors: dict | pd.Series | None = None,
    legend: dict | None = None,
    title: str | None = None,
):
    """
    Draw every trace in its physical well position.

    Parameters
    ----------
    shared_y   : one y-scale across the whole plate, so well-to-well amplitude
                 differences are real and not an artefact of per-panel scaling.
                 Turn off to see the shape of low-amplitude wells.
    annotations: {sample_name: str} drawn inside its well. This is the extension
                 point for period / rhythmicity / R^2 overlays — the caller
                 formats the text, so this function needs no knowledge of what
                 is being shown.
    well_colors: {sample_name: colour} for the well background, e.g. a
                 significance or metric ramp. Same rationale as `annotations`.
    legend     : output of `build_overlay` — either a colorbar spec or a set of
                 swatches. Drawn beneath the plate so the colours are readable
                 as values rather than decoration.
    """
    plot = df
    if t0 is not None:
        plot = plot[plot[t_col] >= t0]
    if t1 is not None:
        plot = plot[plot[t_col] <= t1]

    present = [s for s in layout.wells["Sample"] if s in plot.columns]
    if not present:
        raise ValueError("None of the plate's wells are present in the data.")

    if shared_y:
        lo = float(np.nanmin(plot[present].to_numpy()))
        hi = float(np.nanmax(plot[present].to_numpy()))
        pad = (hi - lo) * 0.07 or 1.0
        ylim = (lo - pad, hi + pad)
    else:
        ylim = None

    n_rows, n_cols = layout.n_rows, layout.n_cols
    # Keep wells roughly square and the whole figure a sane size for 384 too
    cell = 1.15 if layout.size <= 96 else 0.62
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell * n_cols + 1.0, cell * 0.78 * n_rows + 1.0),
        squeeze=False, sharex=True, sharey=bool(shared_y),
    )
    fig.patch.set_facecolor(bg_color)

    tvals = plot[t_col].to_numpy()
    lookup = {s: i for i, s in enumerate(layout.wells["Sample"])}
    ann = dict(annotations) if annotations is not None else {}
    wcol = dict(well_colors) if well_colors is not None else {}

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_linewidth(0.5)
                side.set_color("#CFCFCB")

            sample = layout.well_at(r, c)
            has_data = sample is not None and sample in plot.columns

            ax.set_facecolor(wcol.get(sample, bg_color) if has_data else empty_color)

            if has_data:
                ax.plot(tvals, plot[sample].to_numpy(),
                        color=line_color, lw=0.9, solid_capstyle="round")
                if ylim is not None:
                    ax.set_ylim(*ylim)
                label = []
                if show_sample_names:
                    label.append(str(sample))
                if show_well_ids:
                    hit = layout.wells[layout.wells.Sample == sample]
                    if not hit.empty:
                        label.append(str(hit.iloc[0]["Well"]))
                if sample in ann and str(ann[sample]):
                    label.append(str(ann[sample]))
                if label:
                    # Sits on top of the trace, so it needs its own backing or the
                    # signal reads straight through the text.
                    face = wcol.get(sample, bg_color)
                    ax.text(0.5, 0.97, "\n".join(label), transform=ax.transAxes,
                            ha="center", va="top", fontsize=6.0,
                            color=_ink_for(face), fontweight="bold",
                            linespacing=1.25, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.18", ec="none",
                                      fc=face, alpha=0.82))

            # Row letters down the left, column numbers across the top
            if c == 0:
                ax.set_ylabel(ROW_LETTERS[r], rotation=0, ha="right", va="center",
                              fontsize=9, labelpad=8, color="#52514e")
            if r == 0:
                ax.set_title(str(c + 1), fontsize=9, pad=4, color="#52514e")

    # Explicit margins: the default left margin leaves a wide dead band that is very
    # visible on a 12- or 24-column grid. Reserve just enough for the row letters.
    fig_h = cell * 0.78 * n_rows + 1.0
    top = 1 - (0.55 / fig_h)
    bottom = 0.02 if not legend else (0.62 / fig_h)
    fig.suptitle(title or f"{layout.label} plate", fontsize=12, fontweight="bold",
                 x=0.012, y=0.995, ha="left", va="top")
    fig.subplots_adjust(left=0.045, right=0.995, top=top, bottom=bottom,
                        wspace=0.12, hspace=0.12)

    if legend:
        _draw_legend(fig, legend, bottom)
    return fig


def _draw_legend(fig, legend, bottom):
    """Colorbar for a numeric metric, labelled swatches for a categorical one."""
    if legend.get("kind") == "colorbar":
        cax = fig.add_axes([0.30, bottom * 0.34, 0.40, max(0.008, bottom * 0.16)])
        mappable = plt.cm.ScalarMappable(norm=legend["norm"], cmap=legend["cmap"])
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.set_label(legend.get("label", ""), fontsize=9)
        cbar.ax.tick_params(labelsize=8, length=3)
        cbar.outline.set_visible(False)
        if legend.get("note"):
            cax.annotate(legend["note"], xy=(0.5, -2.6), xycoords="axes fraction",
                         ha="center", va="top", fontsize=8, color="#52514e")
    elif legend.get("kind") == "swatches":
        entries = legend.get("entries", [])
        if not entries:
            return
        lax = fig.add_axes([0.045, bottom * 0.16, 0.90, max(0.02, bottom * 0.40)])
        lax.axis("off")
        handles = [mpatches.Patch(facecolor=c, edgecolor="#CFCFCB", label=str(k))
                   for k, c in entries]
        lax.legend(handles=handles, loc="center", ncol=len(handles),
                   frameon=False, fontsize=9, title=legend.get("label"),
                   title_fontsize=9)


# ═══════════════════════════════════════════════════════════════════════════
#  Overlays (v0.7.8)
# ═══════════════════════════════════════════════════════════════════════════
#
# A metric is turned into two independent things: a LABEL drawn inside each well
# and a COLOUR for the well background. They are separate on purpose — colour
# carries the pattern you see across the plate at a glance, the label carries the
# value you need when you have found the well you care about.
#
# Colour follows the job the number does, not personal taste:
#   diverging  — period, where the meaningful question is "longer or shorter than
#                24 h?". Two hues either side of a neutral grey midpoint at 24.
#   sequential — amplitude, R², noise: more is more. One hue, light to dark.
#   cyclic     — acrophase: 0 h and 24 h are the same time of day, so the ramp
#                has to wrap or midnight would look maximally far from midnight.
#   status     — rhythmic / arrhythmic. Reserved colours, always paired with a
#                text label so the verdict never rests on colour alone.

METRICS: dict[str, dict] = {
    "Period (h)": {
        "kind": "diverging", "feature": "cosinor_period", "result_col": "Periods",
        "center": 24.0, "fmt": "{:.1f}", "cmap": "coolwarm_r",
    },
    "Acrophase (h)": {
        "kind": "cyclic", "feature": "cosinor_acrophase_h",
        "vmin": 0.0, "vmax": 24.0, "fmt": "{:.1f}", "cmap": "twilight",
    },
    "Amplitude": {
        "kind": "sequential", "feature": "cosinor_amplitude",
        "fmt": "{:.2f}", "cmap": "Blues",
    },
    "Cosinor R²": {
        "kind": "sequential", "feature": "cosinor_r2",
        "vmin": 0.0, "vmax": 1.0, "fmt": "{:.2f}", "cmap": "Greens",
    },
    "Noise (residual SD)": {
        "kind": "sequential", "feature": "cosinor_residual_std",
        "fmt": "{:.2f}", "cmap": "OrRd",
    },
    "Rhythmicity": {
        "kind": "status", "needs_results": True, "fmt": None,
        "colors": {"rhythmic": "#CDE7DA", "arrhythmic": "#F3D7DE", "n/a": "#EFEFED"},
    },
}

# One extraction covers period, amplitude, R² and residual SD — they all fall out
# of the same cosinor fit, so there is no reason to run the other seven packages.
METRIC_PACKAGES = ["cosinor"]


def metric_names(has_results: bool = False) -> list[str]:
    """Metrics offerable right now. Rhythmicity needs an analysis to have been run."""
    return [n for n, s in METRICS.items()
            if has_results or not s.get("needs_results")]


def compute_metric(name, df, t_col, samples, features=None,
                   result_df=None, q_col=None, thresh=0.05):
    """
    One metric as a Series indexed by sample name.

    `features` is a DataFrame from ChronotopiaFeatureExtractor.extract_batch with
    a `sample_id` column. Pass it in rather than extracting here, so the caller
    can cache the expensive part.
    """
    spec = METRICS.get(name)
    if spec is None:
        return pd.Series(dtype=float)

    if spec["kind"] == "status":
        if result_df is None or q_col is None or q_col not in result_df.columns:
            return pd.Series("n/a", index=samples, dtype=object)
        flags = result_df.set_index("CycID")[q_col] <= thresh
        return pd.Series(
            [("rhythmic" if bool(flags.get(s, False)) else
              "arrhythmic" if s in flags.index else "n/a") for s in samples],
            index=samples, dtype=object,
        )

    # Prefer the analysis's own period column when it exists — otherwise the
    # plate would disagree with the results table for the same samples.
    rc = spec.get("result_col")
    if rc and result_df is not None and rc in result_df.columns:
        series = result_df.set_index("CycID")[rc]
        return pd.Series([series.get(s, np.nan) for s in samples],
                         index=samples, dtype=float)

    if features is None or spec["feature"] not in getattr(features, "columns", []):
        return pd.Series(np.nan, index=samples, dtype=float)

    series = features.set_index("sample_id")[spec["feature"]]
    out = pd.Series([series.get(s, np.nan) for s in samples],
                    index=samples, dtype=float)
    if spec["kind"] == "cyclic":
        # An acrophase is a time of day. The cosinor fit can return values just
        # outside [0, 24) (25.83 h is 1.83 h), and without wrapping those clip to
        # the end of the colour ramp — the well would read as "late evening" when
        # it is actually just after midnight.
        period = float(spec.get("vmax", 24.0)) - float(spec.get("vmin", 0.0))
        out = out % period
    return out


def _ink_for(bg) -> str:
    """Black or white text, whichever survives on this background."""
    r, g, b = mcolors.to_rgb(bg)

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#0b0b0b" if luminance > 0.45 else "#ffffff"


MASK_COLOR = "#EDEDEB"


def build_overlay(name, values, spec=None, mask=None):
    """
    Turn a metric Series into per-well colours plus what the legend needs.

    `mask` is an iterable of sample names to render in flat grey instead of on
    the ramp. It exists for a real hazard: a cosinor fit returns a period for an
    arrhythmic well too, and that number is noise. Colouring it puts a confident
    "27.2 h" on a well that has no rhythm at all, which reads as a finding. Grey
    out the wells that failed the rhythmicity test and the ramp then describes
    only the wells where the metric means something.

    Returns (colors, legend) where colors is {sample: hex} and legend is either
    {"kind": "colorbar", "norm", "cmap", "label"} or
    {"kind": "swatches", "entries": [(label, colour), ...], "label": str}.
    """
    spec = spec or METRICS.get(name, {})
    kind = spec.get("kind", "sequential")
    masked = set(mask or ())

    if kind == "status":
        palette = spec["colors"]
        colors = {s: palette.get(str(v), palette["n/a"]) for s, v in values.items()}
        present = [k for k in ("rhythmic", "arrhythmic", "n/a")
                   if k in set(values.astype(str))]
        return colors, {"kind": "swatches", "label": name,
                        "entries": [(k, palette[k]) for k in present]}

    numeric = pd.to_numeric(values, errors="coerce")
    # Masked wells must not stretch the colour scale either — one arrhythmic
    # well fitting at 30 h would otherwise compress every real period into the
    # middle of the ramp.
    scale_from = numeric[[s not in masked for s in numeric.index]]
    finite = scale_from[np.isfinite(scale_from)]
    if finite.empty:
        finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {}, None

    cmap = plt.get_cmap(spec.get("cmap", "Blues"))

    if kind == "diverging":
        center = float(spec.get("center", 24.0))
        reach = max(abs(finite.max() - center), abs(center - finite.min()), 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=center - reach, vcenter=center,
                                    vmax=center + reach)
    else:
        vmin = spec.get("vmin", float(finite.min()))
        vmax = spec.get("vmax", float(finite.max()))
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    colors = {}
    for sample, v in numeric.items():
        if sample in masked:
            colors[sample] = MASK_COLOR
        elif np.isfinite(v):
            colors[sample] = mcolors.to_hex(cmap(norm(v)))
    legend = {"kind": "colorbar", "norm": norm, "cmap": cmap, "label": name}
    if masked:
        legend["note"] = f"{len(masked)} well(s) not rhythmic — shown grey"
    return colors, legend


def format_labels(values, name, spec=None, mask=None):
    """Metric values as short strings for drawing inside the wells."""
    spec = spec or METRICS.get(name, {})
    fmt = spec.get("fmt")
    masked = set(mask or ())
    out = {}
    for sample, v in values.items():
        if sample in masked:
            out[sample] = ""      # grey well, no number to misread
            continue
        if fmt is None:
            out[sample] = "" if str(v) == "n/a" else str(v)
        elif isinstance(v, (int, float, np.floating)) and np.isfinite(v):
            out[sample] = fmt.format(v)
    return out
