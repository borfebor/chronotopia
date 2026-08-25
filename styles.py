"""
styles.py
=========
One place that decides what every Chronotopia figure looks like.

Why this module exists
----------------------
Style was previously three loose calls in app.py — `sns.set_style`,
`sns.set_context`, `sns.set_palette` — with the palette list built as
`SEABORN_PALETTES + plt.colormaps()`: 204 entries, of which about 180 are
*continuous* ramps being cycled as categorical series colours. Choosing
`viridis` for a set of conditions implies an ordering between them that does not
exist, and gives poor separation between neighbouring series.

This module keeps the arrangement the app already has (a style picker and a
context picker) and adds:

  * a publication baseline applied under every style — the export settings that
    decide whether a figure is usable in a manuscript at all;
  * named styles defined declaratively in rcParams, including ggplot and a
    journal style;
  * a curated palette list with each option's measured colour-vision-deficiency
    separation, so the choice is informed rather than alphabetical.

Public API
----------
    apply(style_name, context_name, palette_name) -> resolved axes facecolor
    STYLES, CONTEXTS, PALETTES, PALETTE_NAMES, palette_label()
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════
#  Publication baseline — applied under every style
# ═══════════════════════════════════════════════════════════════════════════

PUBLICATION_RC = {
    # ── Text stays TEXT in exported vectors ──────────────────────────────────
    # matplotlib's default (svg.fonttype='path') converts every label into
    # outlines, so a downloaded SVG cannot be edited in Illustrator, Inkscape or
    # Affinity — you cannot fix a typo or restyle an axis label without redoing
    # the figure. 'none' keeps real text. fonttype 42 does the same for PDF/PS
    # by embedding TrueType rather than Type-3, which several journals require
    # outright. This single block is most of what "publication-ready" means for
    # a plotting tool.
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "figure.dpi": 110,

    # ── Consistent marks across every plot type ──────────────────────────────
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "lines.markeredgewidth": 0.6,
    "patch.linewidth": 0.8,
    "axes.axisbelow": True,          # data over gridlines, never under
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",    # left-aligned titles read better in figures
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "errorbar.capsize": 3,

    # A stack, not a single font: the first one present on the machine wins, so
    # figures look the same on a laptop and in a Docker container.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "mathtext.default": "regular",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Styles
# ═══════════════════════════════════════════════════════════════════════════
#
# Each style is (seaborn base, rcParam overrides, one-line description).
# Defined declaratively rather than via sns.despine(), because despine() has to
# run after plotting — rcParams apply to every figure the app draws without
# plots.py needing to know anything about styling.

_GGPLOT_PANEL = "#EBEBEB"
_GGPLOT_INK = "#3C3C3C"

STYLES: dict[str, dict] = {
    "Ticks": {
        "base": "ticks",
        "rc": {},
        "help": "Seaborn 'ticks'. White panel, full frame, tick marks, no grid.",
    },
    "White": {
        "base": "white",
        "rc": {},
        "help": "Seaborn 'white'. Clean white panel with a full frame and no grid or ticks.",
    },
    "White grid": {
        "base": "whitegrid",
        "rc": {},
        "help": "Seaborn 'whitegrid'. White panel with a light grid — easy to read values off.",
    },
    "Dark grid": {
        "base": "darkgrid",
        "rc": {},
        "help": "Seaborn 'darkgrid'. Grey panel, white grid. Good on slides.",
    },
    "Dark": {
        "base": "dark",
        "rc": {},
        "help": "Seaborn 'dark'. Grey panel, no grid.",
    },
    # ── new in v0.7.5 ────────────────────────────────────────────────────────
    "Framed grid": {
        "base": "whitegrid",
        "rc": {
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.direction": "out",
            "ytick.direction": "out",     # added for symmetry with xtick
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.color": "black",
            "ytick.color": "black",       # added for symmetry with xtick
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        },
        "help": "White grid inside a black frame, with long outward ticks on both axes.",
    },
    "ggplot": {
        "base": "darkgrid",
        "rc": {
            "figure.facecolor": "white",
            "axes.facecolor": _GGPLOT_PANEL,
            "axes.edgecolor": "none",
            "axes.linewidth": 0.0,
            "axes.grid": True,
            "grid.color": "white",
            "grid.linewidth": 1.1,
            "grid.linestyle": "-",
            "axes.labelcolor": _GGPLOT_INK,
            "text.color": _GGPLOT_INK,
            "xtick.color": _GGPLOT_INK,
            "ytick.color": _GGPLOT_INK,
            "xtick.bottom": False,
            "ytick.left": False,
            "axes.titlecolor": _GGPLOT_INK,
        },
        "help": "ggplot2 look: grey panel, white gridlines, no frame.",
    },
    "Journal": {
        "base": "ticks",
        "rc": {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.color": "black",
            "ytick.color": "black",
        },
        "help": "Two spines, outward ticks, no grid — the usual house style for "
                "Nature, Cell and eLife figures.",
    },
    "Minimal": {
        "base": "white",
        "rc": {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": True,
            # Both axes, not just horizontal rules. The plot functions place x ticks
            # every 24 h, so the vertical gridlines land exactly on day boundaries and
            # act as a free period reference — the reader can see at a glance whether
            # a peak is drifting relative to 24 h. That is worth more here than the
            # tidiness of horizontal-only rules.
            "axes.grid.axis": "both",
            "grid.color": "#DCDCDA",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#B4B4B0",
            "axes.linewidth": 0.9,
            "xtick.bottom": True,
            "xtick.direction": "out",
            "xtick.major.size": 3,
            "ytick.left": False,
        },
        "help": "No frame, faint horizontal rules only. Uncluttered for talks and "
                "for figures with many panels.",
    },
}

STYLE_NAMES = list(STYLES.keys())
DEFAULT_STYLE = "Journal"

# Every rcParam any style touches. rcParams are global and persist across reruns, so
# without resetting these first, switching styles LEAKS: pick Framed grid (tick size 6)
# then White grid, and the long ticks stay, because seaborn's whitegrid never mentions
# tick size. The nastiest case was axes.grid.axis — Minimal used to set it to "y", and
# every grid style chosen afterwards silently lost its vertical gridlines.
_STYLE_KEYS = sorted({k for spec in STYLES.values() for k in spec["rc"]})

# Grid styles want gridlines on BOTH axes. With x ticks placed every 24 h by the plot
# functions, the vertical lines mark day boundaries and give a period reference.
_GRID_AXIS_DEFAULT = "both"

CONTEXTS = ["paper", "notebook", "talk", "poster"]
DEFAULT_CONTEXT = "notebook"


# ═══════════════════════════════════════════════════════════════════════════
#  Palettes
# ═══════════════════════════════════════════════════════════════════════════
#
# `cvd` is the worst-pair colour-vision-deficiency separation over the first
# five colours, in OKLab ΔE x100, measured with the accessibility validator.
# Read it as: how far apart the two most similar series will look to a reader
# with the common form of colour blindness. >= 8 is the target; below ~4 means
# two series are effectively the same colour to that reader.
#
# The numbers are sobering and are the reason this list is curated. matplotlib's
# default tab10 scores 0.7. seaborn's default deep scores 1.9. Only the first
# four entries below clear the target on neighbouring series, and only the first
# clears it for *every* pair.

PALETTES: dict[str, dict] = {
    "Chronotopia":  {"cvd": 13.0, "all_pairs": True,
                     "colors": ["#d55e00", "#56b4e9", "#2a78d6", "#eda100", "#4a3aa7"],
                     "note": "Built for this app; every pair separates."},
    "colorblind":   {"cvd": 8.4, "all_pairs": False, "colors": "colorblind",
                     "note": "Seaborn's CVD-oriented palette."},
    "Set1":         {"cvd": 3.5, "all_pairs": False, "colors": "Set1",
                     "note": "Strong saturated hues."},
    "Dark2":        {"cvd": 1.7, "all_pairs": False, "colors": "Dark2",
                     "note": "Muted and dark; prints well."},
    "muted":        {"cvd": 4.0, "all_pairs": False, "colors": "muted"},
    "bright":       {"cvd": 3.6, "all_pairs": False, "colors": "bright"},
    "Paired":       {"cvd": 7.2, "all_pairs": False, "colors": "Paired",
                     "note": "Light/dark pairs — good for treated vs control."},
    "dark":         {"cvd": 3.1, "all_pairs": False, "colors": "dark"},
    "deep":         {"cvd": 1.9, "all_pairs": False, "colors": "deep",
                     "note": "Seaborn's default."},
    "Set2":         {"cvd": 1.5, "all_pairs": False, "colors": "Set2"},
    "Accent":       {"cvd": 2.3, "all_pairs": False, "colors": "Accent"},
    "tab10":        {"cvd": 0.7, "all_pairs": False, "colors": "tab10",
                     "note": "matplotlib's default — two series are nearly identical "
                             "under deuteranopia."},
    "Set3":         {"cvd": 3.6, "all_pairs": False, "colors": "Set3"},
    "pastel":       {"cvd": 1.5, "all_pairs": False, "colors": "pastel"},
    "Pastel1":      {"cvd": 1.1, "all_pairs": False, "colors": "Pastel1"},
}

PALETTE_NAMES = list(PALETTES.keys())
DEFAULT_PALETTE = "Chronotopia"

# Target from the accessibility validator: pairs at or above this stay
# distinguishable under the common colour-vision deficiencies.
CVD_TARGET = 8.0


def palette_label(name: str) -> str:
    """Menu label carrying the measured separation, e.g. 'Chronotopia  ✓ 13.0'."""
    meta = PALETTES.get(name)
    if meta is None:
        return name
    mark = "✓" if meta["cvd"] >= CVD_TARGET else "·"
    return f"{name}  {mark} {meta['cvd']:.1f}"


def palette_help() -> str:
    safe = [n for n, m in PALETTES.items() if m["cvd"] >= CVD_TARGET]
    return (
        "The number is the worst-pair separation under colour-vision deficiency "
        f"(≥ {CVD_TARGET:.0f} is the target, ✓). Measured over the first five colours. "
        f"Clearing it: {', '.join(safe)}. "
        "Everything else has at least one pair of series that a colour-blind reader "
        "will struggle to tell apart — usable for one or two series, risky for four. "
        "Only Chronotopia clears the target for every pair rather than neighbours."
    )


def resolve_palette(name: str):
    """Palette name -> something seaborn's set_palette accepts."""
    meta = PALETTES.get(name)
    if meta is None:
        return name                      # a raw matplotlib colormap name
    return meta["colors"]


# ═══════════════════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════════════════

def apply(style_name: str = DEFAULT_STYLE,
          context_name: str = DEFAULT_CONTEXT,
          palette_name: str = DEFAULT_PALETTE,
          editable_text: bool = True) -> str:
    """
    Apply style, context and palette, and return the resolved axes facecolor.

    The facecolor is returned because app.py uses it as the default background
    for the entrainment shading, and each style sets a different one.
    """
    spec = STYLES.get(style_name, STYLES[DEFAULT_STYLE])

    # Clean slate for anything a style might set. rcParams are process-global and
    # survive Streamlit reruns, so without this a key set by one style persists into
    # the next one that doesn't mention it. Reset only the keys we own — a blanket
    # mpl.rcdefaults() would also reset the backend, which Streamlit relies on.
    mpl.rcParams.update({
        k: mpl.rcParamsDefault[k] for k in _STYLE_KEYS if k in mpl.rcParamsDefault
    })

    sns.set_style(spec["base"])
    sns.set_context(context_name if context_name in CONTEXTS else DEFAULT_CONTEXT)

    rc = dict(PUBLICATION_RC)
    if mpl.rcParams.get("axes.grid"):
        rc["axes.grid.axis"] = _GRID_AXIS_DEFAULT
    if not editable_text:
        # Outlined text: the SVG no longer depends on the reader having the font,
        # at the cost of no longer being editable as text.
        rc["svg.fonttype"] = "path"
    rc.update(spec["rc"])
    mpl.rcParams.update(rc)

    try:
        sns.set_palette(resolve_palette(palette_name))
    except (ValueError, KeyError):
        sns.set_palette(resolve_palette(DEFAULT_PALETTE))

    face = mpl.rcParams.get("axes.facecolor", "white")
    return mcolors.to_hex(face) if face not in (None, "none") else "white"


def preview_colors(palette_name: str, n: int = 5) -> list:
    """First n colours of a palette as hex, for a swatch."""
    try:
        return sns.color_palette(resolve_palette(palette_name), n).as_hex()
    except Exception:
        return []


def all_colormap_names() -> list:
    """Every matplotlib colormap, for the escape hatch in the UI."""
    return sorted(n for n in plt.colormaps() if not n.endswith("_r"))
