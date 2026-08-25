"""
docs.py
=======
Every tooltip in Chronotopia, in one place — and the mechanism that attaches them.

Why a module rather than 97 inline `help=` strings
--------------------------------------------------
Scattered tooltips drift. They get written for the control someone happened to be
editing, in whatever voice that day called for, and there is no way to ask "which
controls are still undocumented?" or to read the documentation as a whole.

Keeping the text here buys three things:

  * `attach()` injects help automatically, keyed on the widget's label, so no call
    site has to be edited and any NEW widget with a known label is documented the
    moment it appears;
  * `coverage()` makes "is this app documented?" a testable question;
  * `as_markdown()` turns the same text into a reference document the user can
    read or export, so the tooltips and the manual cannot disagree.

Labels repeat across the app on purpose — "Column to preview" appears nine times
and means the same thing every time, so it gets one entry.

Writing guidance for anything added here: say what the control changes and when
it matters, not what it is called. "Which column holds time" is worth writing;
"select the time column" is not.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
#  The text
# ═══════════════════════════════════════════════════════════════════════════
#
# Grouped by where the control lives, so this file reads like a walkthrough.

SECTIONS: dict[str, dict[str, str]] = {

    "Loading data": {
        "Upload your data":
            "A table with one time column and one column per sample. CSV, TSV, TXT or "
            "XLSX. Every column that is not the time column is treated as a signal, so "
            "strip anything that is not measured data before uploading.",
        "Generate example dataset":
            "Builds a synthetic recording with known properties — useful for trying a "
            "view before committing your own data, and for checking that a result you "
            "are unsure about behaves as expected on data where the answer is known.",
        ":material/upload_file: Upload new file":
            "Clears the current dataset and starts again. Cached analysis results are "
            "kept until you re-run, and the app warns if they belong to another file.",
        "Show data preview":
            "First five timepoints of the first five samples, to confirm the file "
            "parsed the way you expected.",
        "Time column":
            "Which column holds time. Everything else becomes a signal, so this choice "
            "also decides what gets analysed.",
        "Time unit":
            "The unit your time column is written in. Guessed from the spacing between "
            "timepoints, but check it: this converts everything to hours, and getting "
            "it wrong scales every period, smoothing window and filter cut-off by the "
            "same factor.",
        "Upload your experimental layout":
            "A two-column table — Sample and Condition — that assigns samples to "
            "experimental groups. Without it the app treats every sample as its own "
            "group, so grouped plots, comparisons and per-condition statistics stay "
            "hidden. Download the template below for a version pre-filled with your "
            "sample names.",
        "Download layout template":
            "Your sample names with an empty Condition column, ready to fill in. On a "
            "detected plate it also carries the well, row and column, so you can group "
            "by plate geometry without retyping anything.",
        "Group wells by":
            "Plate geometry is detected automatically, but it says nothing about your "
            "experimental design. Leave this on None and upload a layout to define real "
            "conditions; pick Row or Column only if that genuinely is how the plate was "
            "laid out.",
    },

    "Example dataset": {
        "Days generated": "Length of the synthetic recording.",
        "Timepoints per day": "Sampling density. 24 gives hourly data, 144 gives "
                              "10-minute data.",
        "Number of samples": "How many synthetic traces to build.",
        "Ratio of rhythmic (%)": "What fraction of the traces carry a real rhythm. The "
                                 "rest are drift plus noise, so you can see how a method "
                                 "behaves on signals that should not pass.",
        "Free running period": "Intrinsic period of the rhythmic samples, in hours. Set "
                               "it away from 24 to see how a view handles a drifting "
                               "rhythm.",
        "Entrainment days": "How many days of driven cycling before release into free "
                            "run. Zero means free-running throughout.",
        "Entrainment period": "Period of the driving cycle (T), in hours.",
        "Waveform": "Shape of the underlying oscillation. Square and saw are useful for "
                    "checking that waveform and harmonic features behave — a pure sine "
                    "has no harmonics to find.",
    },

    "Time range and preprocessing": {
        "Starting Timepoint":
            "Trims the start of the recording for every downstream calculation, not just "
            "the plot. Use it to drop a settling period before the rhythm establishes.",
        "Last Timepoint":
            "Trims the end of the recording for every downstream calculation. Use it to "
            "cut a period where the preparation was failing.",
        "Always start time from 0":
            "Shifts the time axis so the first point is zero. Only affects labelling and "
            "the placement of day boundaries, never the measured period.",
        "Smoothening":
            "Reduces point-to-point noise before analysis. Mean is a plain rolling "
            "average; Savitzky-Golay fits a local polynomial and so preserves peak "
            "height and width where an average flattens them; Resample averages onto a "
            "regular hourly grid; DCT removes components below a cut-off period. "
            "Smoothing changes amplitude and waveform features — if those are what you "
            "are measuring, prefer None and handle noise in the model instead.",
        "Savitzky-Golay window (h)":
            "Width of the local polynomial fit, in hours. 6 h keeps a 24 h rhythm at "
            "99.8% amplitude and its 12 h harmonic at 97%, while cutting white noise to "
            "about 40%. Above roughly 12 h the harmonic starts to go and the waveform "
            "visibly rounds off.",
        "Polynomial degree":
            "1 is a moving average. 2 (the default) preserves peak height and width. "
            "4 follows sharp peaks more closely but smooths less. Odd degrees are not "
            "offered: a symmetric Savitzky-Golay filter of degree 2k+1 gives the same "
            "result as degree 2k everywhere except the first and last half-window.",
        "Normalization":
            "Puts samples on a comparable scale. Z-score centres each sample and divides "
            "by its own spread; Sample-wise Min-Max rescales each to 0-100; Global "
            "Min-Max uses one scale across the whole dataset and so preserves relative "
            "amplitude between samples. Anything other than Global destroys between-"
            "sample amplitude comparisons.",
        "Detrending":
            "How the non-rhythmic baseline is estimated, so the oscillation can be "
            "measured on its own. Linear and Cubic fit a polynomial against the time "
            "column — fast, but a cubic has no asymptote and bends back up at the end "
            "of a recording. Rolling mean is the default and the most accurate in the "
            "interior, though its amplitude cost depends strongly on period. LOESS "
            "fits a local line instead: it costs almost the same amplitude at any "
            "period in your range, and a mid-run artefact moves it about six times "
            "less than it moves the rolling mean. Exponential fit is the physically "
            "right shape for substrate consumption in a bioluminescence run and "
            "behaves best at the two ends of the recording, but it cannot follow a "
            "baseline that turns over. Rolling Hilbert is separate: it removes a "
            "rolling mean and then divides by the signal envelope, flattening a "
            "damping rhythm to constant amplitude — right when you want period or "
            "phase out of a decaying trace, wrong when damping is what you are "
            "measuring. Every method leaves missing timepoints missing rather than "
            "spreading them. Whichever you pick, detrending invalidates the trend and "
            "baseline features and shifts what the QC flags mean, since the mean of a "
            "detrended trace is near zero.",
        "Baseline removal":
            "Whether the baseline is taken out by subtraction or by division. Subtract "
            "assumes the baseline is an added offset and leaves the result in the "
            "original units. Divide assumes it is a multiplying factor and returns the "
            "fractional deviation from it, still centred on zero. For bioluminescence "
            "and fluorescence, Divide is usually the honest choice: substrate "
            "depletion, cell number and bleaching scale the whole signal, oscillation "
            "included, so a subtracted residual is still multiplied by a falling "
            "baseline and the rhythm appears to damp faster than it does. On the "
            "tutorial 2 dataset, where damping is planted at 120-165 h, subtraction "
            "reports 89-94 h and division reports 161 h. Division is refused on data "
            "whose baseline crosses zero — anything already centred, z-scored or "
            "background-subtracted — and falls back to subtraction with a message.",
        "Detrending window (h)":
            "Width of the moving average subtracted from the signal. A centred moving "
            "average cancels exactly at period = window, so setting this to the period "
            "you are measuring removes the baseline and keeps the rhythm at full "
            "amplitude. It defaults to the middle of your Period range for that reason. "
            "Shorter windows eat into the rhythm (a 20 h window keeps only 81% of a 24 h "
            "oscillation), longer ones overshoot and add a phase-inverted copy back in "
            "(a 30 h window returns 118%). The caption underneath reports the actual "
            "cost at both ends of your period range. If your samples genuinely span a "
            "range of periods, no single window is right for all of them — LOESS does "
            "not have this problem.",
        "LOESS span (h)":
            "Width of the neighbourhood each local line is fitted through. It must be "
            "comfortably longer than the period: a line fitted over one cycle follows "
            "the rhythm and removes it, leaving only 57% of a 24 h oscillation. Twice "
            "the period is the default and is where the amplitude cost stops depending "
            "on period — at a 48 h span a 20 h, 24 h and 28 h rhythm all survive at "
            "100-108%, where a 24 h rolling-mean window keeps 116% and 65% of the same "
            "two extremes. Widen it further if the baseline is smooth and you want the "
            "rhythm untouched; narrow it only if the baseline moves fast.",
        "Exclude by":
            "Whether to pick samples to drop individually or by condition. Grouping by "
            "condition requires an uploaded layout.",
        "Select data to exclude":
            "Samples removed from every calculation and plot from here on, not merely "
            "hidden. The QC view on the Feature extraction page can generate this list "
            "for you.",
    },

    "Entrainment": {
        "Mode":
            "How the app learns when the zeitgeber was on. 'manual' if you know the "
            "schedule; 'from data' to detect it from a recorded light or temperature "
            "channel already in the file; 'upload' to supply that channel separately.",
        "Select feature columns":
            "The column holding the recorded zeitgeber — a light or temperature trace. "
            "It is removed from the sample list, since it is a condition, not a "
            "measurement.",
        "Upload your entrainment data":
            "A two-column table of time and zeitgeber state, for when the driving signal "
            "was logged separately from the samples.",
        "Entrainment cycles": "How many complete driven cycles preceded free run.",
        "T cycle": "Period of the driving cycle, in hours. 24 for a normal light-dark "
                   "cycle; other values for T-cycle experiments.",
        "Day length": "Hours of the 'on' phase within each cycle — 12 for 12:12.",
        "Zeitgeber type":
            "Which pair of conditions alternated. Sets the shading colours so the plot "
            "reads the way the experiment ran.",
        "Free running conditions":
            "Which of the two conditions the samples were left in after release. This "
            "sets the plot background, so the shaded bands mark the other phase.",
        "Color order":
            "Whether the first half-cycle is the shaded phase or the unshaded one. Flip "
            "it if the bands are offset from your protocol by half a cycle.",
        "Entrainment band": "Colour of the shaded zeitgeber phase.",
        "Background color": "Colour of the unshaded phase, and the panel background.",
        "Exclude entrainment from period estimation":
            "Estimates period only from the free-running segment. Usually what you want: "
            "during entrainment the rhythm is driven at the zeitgeber period, so "
            "including it pulls the estimate towards T rather than the intrinsic period.",
    },

    "Analysis parameters": {
        "Period Estimation":
            "How the dominant period of each trace is found. FFT is fast and wants "
            "evenly spaced data; Lomb-Scargle handles gaps and irregular sampling; "
            "Wavelet Transform tracks a period that changes over the recording; "
            "Autocorrelation is robust to waveform shape but needs several cycles; "
            "Damped Cosinor fits a decaying sinusoid, so a rhythm that is losing "
            "amplitude does not broaden the estimate the way it broadens a "
            "periodogram peak — its worst-case error on the tutorial data is a "
            "third smaller than the next best method. It assumes a single damped "
            "sinusoid, so check the fit if the waveform is far from sinusoidal or "
            "the period drifts, and it needs at least three cycles to be offered.",
        "Period range":
            "The window searched for a period, in hours. Narrow it to keep an estimator "
            "from locking onto a harmonic; widen it if you expect non-circadian rhythms. "
            "Periods shorter than twice your sampling interval cannot be resolved "
            "whatever the setting.",
        "Testing method":
            "How rhythmicity is decided. meta2d combines JTK, ARS and Lomb-Scargle and "
            "is the usual default; the three can also be run alone. PermCosinor is a "
            "permutation cosinor that makes no distributional assumption but is slower. "
            "Tempo is the machine-learning classifier and returns a probability rather "
            "than a q-value.",
        "Significance threshold":
            "The FDR-corrected q-value below which a sample counts as rhythmic. Applies "
            "to the pie charts, group summaries and comparisons as well as the table.",
        "Minimum rhythmic probability":
            "Tempo scores each trace from 0 to 1. A sample counts as rhythmic at or "
            "above this probability. Stored internally as 1 minus this value so the same "
            "machinery that handles q-values can handle it.",
        "Model": "Which trained classifier Tempo uses.",
        "Minimum time":
            "Start of the window the rhythmicity test runs on. Independent of the plot "
            "range, so you can test the free-running segment while still plotting the "
            "whole recording.",
        "Last time": "End of the window the rhythmicity test runs on.",
        "Run analysis":
            "Runs period estimation and the selected rhythmicity test over every sample. "
            "The slowest step in the app; results are held until you load another file.",
        "Compare groups":
            "Pairwise tests between conditions on rhythmic fraction, period and "
            "amplitude. Needs an uploaded layout and a completed analysis.",
    },

    "Visualisation": {
        "Type of plot to visualize":
            "Which view to draw. Some appear only when their requirements are met — "
            "grouped views need a layout, Phase plot needs entrainment, Plate view needs "
            "a detected plate, Wavelet Ridge needs the wavelet period method.",
        "Starting time to plot": "Start of the plotted range. Affects the figure only, "
                                 "not the analysis.",
        "End time to plot": "End of the plotted range. Affects the figure only, not the "
                            "analysis.",
        "Column to preview": "Which sample this view draws.",
        "Data unit": "Y-axis label. Purely cosmetic, but worth setting before exporting "
                     "a figure for a manuscript.",
        "Show datapoints": "Marks each measured point on the trace, so you can see "
                           "sampling density and spot gaps.",
        "Rhythmicity evaluation":
            "Annotates the plot with a rhythmicity verdict for this sample. The selected "
            "testing method runs on this one trace; Tempo is offered separately when it "
            "is not already the method.",
        "Plot N times": "How many days are shown side by side on each row. 2 gives the "
                        "conventional double-plotted actogram, where a rhythm longer "
                        "than 24 h drifts rightwards down the panel.",
        "Signal color": "Fill colour for the actogram traces.",
        "Trace color": "Line colour for the plate traces.",
        "Re-scale Y axis by subset amplitude":
            "Gives each day its own y-scale. Reveals the shape of quiet days, at the "
            "cost of making them look as strong as loud ones.",
        "Plots per row": "How many samples sit side by side in the multi-actogram grid.",
        "Adjust height": "Vertical size of the figure. Increase it if days are cramped.",
        "Choose the group to inspect": "Which condition's samples to draw.",
        "Choose condition": "Restricts the view to one condition, or All.",
        "Colormap plette": "Colour scale for the correlation matrix. Diverging scales "
                           "(vlag, coolwarm) are the right choice here because the "
                           "midpoint — zero correlation — is meaningful.",
        "Show annotation": "Prints the correlation value inside each cell. Turn it off "
                           "above roughly 20 samples, where the labels stop fitting.",
        "Show period estimation": "Adds a panel showing each sample's estimated period "
                                  "beside the classification.",
    },

    "Comparison views": {
        "Samples to compare (up to 5)":
            "Traces drawn on one axis. Capped at five: beyond that no colour set keeps "
            "every pair distinguishable, and the plot stops being a comparison.",
        "Conditions to compare (2–4)":
            "Groups drawn on one axis, as mean with spread. Capped at four for the same "
            "reason as samples.",
        "Style": "Mean ± SD draws a ribbon around the group mean; Mean + Replicates "
                 "draws the mean over its faded individual traces, so you can see "
                 "whether the spread comes from a real difference or one outlier.",
        "Accessible colours":
            "Uses a fixed palette checked so that every pair of series stays "
            "distinguishable, including for the common forms of colour-vision "
            "deficiency. Uncheck to use the palette selected above — note that "
            "seaborn's 'colorblind' palette puts two oranges in slots 2 and 4, which are "
            "hard to tell apart when compared directly.",
    },

    "Plate view": {
        "Y scaling":
            "Shared keeps one y-axis across the whole plate, so differences in amplitude "
            "between wells are real. Per well rescales each panel to its own range — "
            "useful for seeing the shape of quiet wells, misleading about their size.",
        "Overlay":
            "Colours each well by a metric and prints its value. Period, amplitude, R² "
            "and noise all come from one cosinor fit per well; Rhythmicity comes from "
            "the analysis run and needs one to exist.",
        "Label with sample name": "Prints the sample name inside each well.",
        "Label with well ID": "Prints the well position (A01, B07) inside each well.",
        "Grey out non-rhythmic wells":
            "A cosinor fit returns a period for an arrhythmic well too, and that number "
            "is noise. Leaving it coloured puts a confident value on a well with no "
            "rhythm, which reads as a finding. Greyed wells are also left out of the "
            "colour scale, so they cannot compress the range for the wells that matter.",
    },

    "Period sweep": {
        "Period range to sweep (h)":
            "The window of trial periods. Widen it well beyond the circadian range — the "
            "point of this view is finding components you were not looking for, such as "
            "12 h and 8 h harmonics.",
        "Resolution (h)": "Spacing of the trial periods. Finer resolution sharpens the "
                          "peaks and costs proportionally more time.",
        "Minimum R²":
            "Signals fitting worse than this are left out of the histogram. They still "
            "count towards the mean R² curve above it, which is a property of the whole "
            "set rather than of any one signal.",
        "Peaks to label": "How many dominant periods to mark, ranked by prominence.",
        "Split by condition":
            "Sweeps each condition separately. Useful when the question is whether a "
            "component appears or disappears between groups. Peaks are found on the "
            "pooled landscape so the same reference lines sit under every group.",
    },

    "Features": {
        "View":
            "Single feature is the per-feature view. Compare tests every feature at once "
            "with the multiplicity correction made explicit. Feature quality reports "
            "missingness and redundancy — the check worth running before training "
            "anything on this table. QC flags samples that look unreliable.",
        "Feature group": "Features grouped by what they measure rather than by which "
                         "package computed them. Several of these groups hold more than "
                         "one estimate of the same quantity.",
        "Feature": "Which feature to plot. Its meaning and source are described below "
                   "the selector.",
        "Group A": "The reference group. Effect sizes are signed relative to it, so a "
                   "negative value means lower in this group.",
        "Group B": "The group compared against the reference.",
        "Test":
            "auto uses a rank-based test when the smaller group has fewer than 8 "
            "samples, because a t-test at that size rests on a normality assumption "
            "nobody can check. Whichever test runs is stated on the figure.",
        "Features package": "Which family of features to compute and overlay on the "
                            "trace. Each has a paired diagnostic drawing so you can see "
                            "what the numbers were derived from.",
        "Rhythmicity evaluation (ML)": "Adds Tempo's verdict for this sample to the "
                                       "figure.",
        "Show": "Package overlay draws the selected features on the trace. Cohort "
                "context places this sample's features on a percentile scale against "
                "every other sample, so a value can be judged rather than merely read.",
        "Features to show": "How many features to display, most unusual first.",
    },

    "Style and export": {
        "Select style": "Panel, grid and tick styling for every figure. Journal is the "
                        "usual house style for Nature, Cell and eLife.",
        "Select context": "Scales fonts and line widths for the medium — 'paper' is "
                          "smallest, 'poster' largest. Geometry is unchanged.",
        "Select palette":
            "The number beside each name is the worst-pair separation under colour-"
            "vision deficiency, measured over the first five colours; 8 or above is the "
            "target and is marked with a tick. Most standard palettes do not reach it — "
            "matplotlib's tab10 scores 0.7, meaning two of its hues are effectively "
            "identical to a colour-blind reader.",
        "All colormaps":
            "Adds every matplotlib colormap. Most are continuous ramps meant for "
            "heatmaps: cycling one across separate series gives poor separation and "
            "implies an ordering between conditions that does not exist.",
        "Editable text in exports":
            "Keeps axis labels and titles as real text in the exported SVG and PDF, so "
            "they can be edited in Illustrator or Inkscape, and embeds TrueType rather "
            "than Type-3 fonts, which several journals require. Uncheck to convert text "
            "to outlines instead — the file then renders identically anywhere but can no "
            "longer be edited as text.",
        "Download Plot as SVG": "Vector export of the current figure, at the styling and "
                                "font settings selected above.",
        "Download clean data": "The dataset after trimming, smoothing, normalisation, "
                               "detrending and sample exclusion — what the analysis "
                               "actually ran on, not the raw upload.",
        "Export features": "The complete feature table, every column. Pair it with the "
                           "data dictionary so the columns stay interpretable.",
        "Export comparison": "Effect size, test statistic, raw p and FDR q for every "
                             "feature tested.",
        "Export data dictionary": "What each feature means, which package produced it, "
                                  "and whether it describes the biology or the "
                                  "recording. Ships alongside the feature table.",
        "Export sweep results": "Best-fit period, amplitude, phase and R² for every "
                                "signal in the sweep.",
        "Export exclusion list": "Samples that failed QC, ready to paste into the "
                                 "exclusion control in the sidebar.",
        "Export analysis results": "The full rhythmicity table — q-values, periods and "
                                   "whatever else the selected method reported.",
        "Download MetaCycle results": "MetaCycle's own output table, as R wrote it — the "
                                      "per-method p-values (JTK, ARS, LS) alongside the "
                                      "integrated meta2d period, phase and amplitude.",
        ":material/docs: Prepare report":
            "Builds a multi-page PDF covering phases, periods, group results and "
            "individual traces. Works before an analysis has been run; the statistical "
            "sections are simply omitted.",
        ":material/download: Download report": "The assembled PDF report.",
    },

    "Help": {
        "Download the control reference":
            "Every tooltip in the app as one Markdown document, generated from the same "
            "text the tooltips use — useful for writing a methods section, or for "
            "handing to someone learning the app.",
    },
}

# Flattened for lookup. Later sections win on a clash, which never happens today —
# `coverage()` would catch it.
HELP: dict[str, str] = {}
for _section, _entries in SECTIONS.items():
    HELP.update(_entries)


import re as _re

# A few labels are built with an f-string that interpolates a limit — "Conditions to
# compare (2–4)" reads better than a bare noun, but the number moves if the cap in
# plots.py changes. Indexing the labels with the trailing parenthetical stripped means
# the tooltip survives that, instead of silently vanishing.
_PAREN = _re.compile(r"\s*\([^()]*\)\s*$")


def _norm(label: str) -> str:
    return _PAREN.sub("", label).strip()


_NORM: dict[str, str] = {}
for _label, _text in HELP.items():
    _NORM.setdefault(_norm(_label), _text)


def h(label: str, default: str | None = None) -> str | None:
    """Tooltip for a label, or `default` when undocumented."""
    if label in HELP:
        return HELP[label]
    if isinstance(label, str):
        hit = _NORM.get(_norm(label))
        if hit is not None:
            return hit
    return default


# ═══════════════════════════════════════════════════════════════════════════
#  Attachment
# ═══════════════════════════════════════════════════════════════════════════

_WIDGETS = (
    "selectbox", "slider", "select_slider", "checkbox", "radio", "multiselect",
    "number_input", "text_input", "text_area", "toggle", "pills", "color_picker",
    "file_uploader", "button", "download_button", "date_input", "time_input",
    "segmented_control",
)

_attached = False


def attach(st_module=None, dg=None, _force=False) -> int:
    """
    Make every widget consult HELP automatically.

    Wraps the widget constructors so that a call with no explicit `help=` picks up
    the entry for its label, if there is one. An explicit `help=` always wins, so
    a control can still say something specific to its context.

    Both bindings are patched. `st.selectbox` and `DeltaGenerator.selectbox` are
    separate objects, and roughly half of Chronotopia's widgets are created on a
    column or the sidebar rather than on `st` directly — patching only one would
    document only half the app.

    `st_module` and `dg` exist so the test harness can hand in stand-ins and check
    the wrapping without a running Streamlit server; leave them unset in the app.

    Returns the number of constructors wrapped. Idempotent.
    """
    global _attached
    if _attached and not _force:
        return 0

    if st_module is None or dg is None:
        import streamlit as _st
        from streamlit.delta_generator import DeltaGenerator as _DG
        st_module = st_module or _st
        dg = dg or _DG

    target, DeltaGenerator = st_module, dg
    wrapped = 0

    def make(func, is_method):
        def wrapper(*args, **kwargs):
            if kwargs.get("help") is None:
                label = None
                idx = 1 if is_method else 0
                if len(args) > idx and isinstance(args[idx], str):
                    label = args[idx]
                elif isinstance(kwargs.get("label"), str):
                    label = kwargs["label"]
                if label is not None:
                    text = h(label)
                    if text:
                        kwargs["help"] = text
            return func(*args, **kwargs)
        wrapper.__name__ = getattr(func, "__name__", "widget")
        wrapper.__doc__ = getattr(func, "__doc__", None)
        wrapper.__chronotopia_documented__ = True
        return wrapper

    for name in _WIDGETS:
        method = getattr(DeltaGenerator, name, None)
        if method is not None and not getattr(method, "__chronotopia_documented__", False):
            setattr(DeltaGenerator, name, make(method, is_method=True))
            wrapped += 1
        fn = getattr(target, name, None)
        if fn is not None and not getattr(fn, "__chronotopia_documented__", False):
            setattr(target, name, make(fn, is_method=False))
            wrapped += 1

    _attached = True
    return wrapped


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════

def coverage(labels) -> dict:
    """
    Which of the given labels are documented.

    Turns "is this app documented?" into something a test can answer, so a new
    control without a tooltip fails the suite instead of shipping.
    """
    labels = [l for l in labels if isinstance(l, str) and l]
    documented = [l for l in labels if h(l) is not None]
    missing = sorted(set(l for l in labels if h(l) is None))
    seen = set(_norm(l) for l in labels)
    return {
        "n_labels": len(set(labels)),
        "n_documented": len(set(documented)),
        "missing": missing,
        "pct": 100.0 * len(set(documented)) / len(set(labels)) if labels else 100.0,
        "unused": sorted(k for k in HELP if _norm(k) not in seen),
    }


def as_markdown(title: str = "Chronotopia — control reference") -> str:
    """
    The same text as a reference document.

    Written from the identical dictionary the tooltips use, so the manual cannot
    drift out of step with the app.
    """
    lines = [f"# {title}", "",
             "Every control in the app, grouped by where it appears. This is generated "
             "from the same text shown in the tooltips, so the two cannot disagree.", ""]
    for section, entries in SECTIONS.items():
        lines.append(f"## {section}")
        lines.append("")
        for label, text in entries.items():
            clean = label.replace(":material/upload_file: ", "").replace(
                ":material/docs: ", "").replace(":material/download: ", "")
            lines.append(f"**{clean}**")
            lines.append("")
            lines.append(text)
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"{len(HELP)} controls documented across {len(SECTIONS)} sections.")
    return "\n".join(lines)
