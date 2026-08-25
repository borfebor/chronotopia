# Chronotopia 0.8.0

**Released 2026-08-25.** This is the first tagged release. Everything below is
new since the code currently on the `main` branch at
[github.com/borfebor/chronotopia](https://github.com/borfebor/chronotopia),
which carries no version tag and is referred to here as **0.7**.

Several fixes change the numbers Chronotopia reports for the same input file.
**If you have results from the 0.7 branch, re-run them.** The section below says
which numbers moved and why.

If you use Chronotopia in published work, record the version — the numerical
defaults change between releases. See [Citing Chronotopia](https://borfebor.github.io/chronotopia/reference/citing/).

---

## Fixes that change your results

Read this section before anything else.

**`delta_t` was wrong for any file not already in hours.** The sampling interval
was computed before the time-unit conversion, so the smoothing window, filter
cut-offs, entrainment square wave and the header text were all off by the
conversion factor. It is now recomputed after conversion.

**Autocorrelation returned lags, not hours.** Period estimates from that method
were in samples.

**The wavelet sampling frequency was inverted.** pyBOAT's `fs` is samples per
hour, not the interval between them.

**The Hilbert transform ran along the wrong axis.**

**Detrending failed on missing data, three different ways.** *Linear* raised and
stopped the app on a single missing point. *Cubic* and *Rolling Hilbert* silently
returned an entirely empty column, so one gap in one well removed that well from
the analysis without any message. Every method now leaves gaps as gaps.

**Linear and Cubic detrending fitted against the sample index, not time.** On an
irregularly sampled or gap-filled recording the trend removed was not the trend
in the data.

**The rolling-mean detrending window was fixed at about 20 h** regardless of the
biology. A centred moving average cancels exactly at window = period, so that
default kept 100% of a 20 h rhythm but only 65% of a 28 h one — a period-dependent
amplitude bias sitting upstream of every amplitude feature. The window now
defaults to the middle of your **Period range**, and the app reports the actual
cost at both ends of that range.

**"Rolling Hilbert" returned the same amplitude for every trace.** It divided by
the raw analytic modulus, which is mathematically unit amplitude by construction,
so a wild-type well, an arrhythmic well and a dead well came out identical. The
envelope is smoothed over one cycle before division now.

**Singular cosinor design matrices no longer raise.** Least-squares goes through
a pseudo-inverse, which matters at short recordings and coarse sampling.

**Benjamini–Hochberg is NaN-safe.** A single failed fit no longer poisons the
q-values for the whole plate.

**pyBOAT figure leak — the default period method is about 9× faster.**
`WAnalyzer.compute_spectrum` defaults to `do_plot=True`, so the wavelet method
drew and retained one figure per sample per rerun.

---

## Detrending is now two choices

The single **Detrending** dropdown has become an estimator plus a **Baseline
removal** mode. This is the largest behavioural change in the release.

Subtracting assumes the baseline is an added offset. Dividing assumes it is a
multiplying factor and returns the fractional deviation from it, still centred on
zero. For bioluminescence and fluorescence the multiplicative reading is usually
the correct one: substrate depletion, cell number and bleaching scale the whole
signal, oscillation included.

On the long-series tutorial data, where damping is planted with τ in 120–165 h:

| Estimator | Subtract | Divide |
|---|---|---|
| Rolling mean | 94 h | **161 h** |
| LOESS | 89 h | **144 h** |
| Exponential fit | 92 h | **162 h** |
| Cubic | 93 h | **158 h** |

Every subtractive method reports damping roughly 1.7× too fast. Division also
recovers relative amplitude to within 5% of the planted value and preserves
between-sample amplitude differences far better (correlation with the planted
amplitude 0.93–0.97, against 0.78–0.80).

Division is refused, with a message, on data whose baseline crosses zero —
anything already centred, z-scored or background-subtracted — and falls back to
subtraction.

**Estimators.** Linear, Cubic, Rolling mean, **LOESS** (new), **Exponential fit**
(new), plus Rolling Hilbert for envelope removal. LOESS is the safe general
choice: its amplitude cost barely depends on period, where a 24 h moving-average
window keeps 116% of a 20 h rhythm and 84% of a 28 h one.

Butterworth band-pass and the duplicate Hilbert detrend were removed during the
0.7 line — the Butterworth band was hardcoded to 18–30 h, ignored the period
slider, and frequently landed above Nyquist on coarsely sampled recordings.

---

## New analysis

**Damped cosinor**, as both a period-estimation method and a feature package. It
fits `A·exp(-t/τ)·cos(2πt/T + φ) + C`, so damping is measured rather than removed,
and every parameter carries a standard error — no periodogram or envelope method
provides one. On the tutorial data its median period error ties the best existing
method; its **worst-well** error is 0.47 h against 0.63 h. A decaying rhythm
broadens a periodogram peak, and that is where the other estimators lose wells.

**Feature extraction.** Around 108 features per sample on a long recording across
nine packages, each mapped to one of ten concepts, with a data dictionary, a
`biology` / `recording` role flag, quality and redundancy reporting, per-sample
cohort context, and a differential comparison across conditions that corrects
across the whole feature matrix rather than the one feature you happened to look
at.

**Period sweep** — the whole period landscape rather than one number.

**Plate view** for 6- to 384-well layouts, with automatic plate detection.

**Comparison views** — samples and conditions side by side.

**Rhythmicity testing** — MetaCycle (`meta2d`, JTK, ARS, LS) where R is
available, plus a permutation cosinor and a random-forest classifier that need
only Python. All FDR-corrected.

---

## Figures and readability

**The feature volcano was unreadable and is fixed.** It asked for ten concept
colours from a five-colour palette and got the palette back twice, so *Period*
and *Waveform shape* were the same orange and *Phase* and *Harmonics* the same
blue, with nothing but legend order to tell them apart. Concept is now carried by
hue **and** marker shape. Tied points — Cliff's delta saturates at ±1 whenever two
small groups do not overlap — are spread sideways by a bounded amount so a pile of
thirty features no longer renders as six visible marks.

**A colour-vision-checked palette** across every view, gated on all-pairs
separation rather than adjacent pairs, since these views exist to compare any
series to any other.

**Publication-ready export** — SVG throughout, with editable text.

---

## Documentation

A full documentation site at
[borfebor.github.io/chronotopia](https://borfebor.github.io/chronotopia/):
installation, data format, two worked tutorials with ground-truth datasets, the
feature-extraction reference, a **Preprocessing** page covering every detrending
method and what each one costs, reproducibility notes and citation guidance.

**A control reference generated from the app's own tooltips**, so the manual
cannot say one thing while the app says another.

Every figure on the site is rendered from the tutorial datasets by the app's own
plotting code, and CI fails if a committed figure differs from a freshly rendered
one.

**Two tutorial datasets with ground truth** — a 48 h transcriptomics timecourse
and a 144 h 24-well luciferase recording — both carrying deliberate mess:
replicate rows, a lost replicate, missing values, a medium-change transient,
baseline drift, damping.

**A verification harness** of 434 checks across `verify.py`, `verify_stage1.py`
and `tutorials/verify_tutorial_data.py`, run in CI. Claims made in the
documentation are asserted against the code, so the two cannot drift apart.

---

## Structure

The 0.7 branch was four Python files. This release separates concerns:
`methods.py` (analysis), `plots.py` (all figure drawing), `plates.py` (plate
detection and layout), `features.py` (the feature dictionary and comparisons),
`styles.py` (styling and palettes), `docs.py` (every tooltip in one place).

Added `CITATION.cff`, a `mkdocs.yml` site build, and a CI workflow.

---

## Known issues

**Replicate rows break three of four period methods.** `app.py` passes data to
`methods.period_estimation` without averaging repeated timepoints. FFT raises
`AssertionError: Time array must be uniformly sampled`, Autocorrelation returns
NaN, and Wavelet reads the sampling interval off the row spacing rather than the
timepoints and comes back about 1.6 h wrong. Lomb-Scargle is correct. Average
your replicates before upload if you need another method.

**Time unit auto-guess is wrong for hourly data.** Any file whose sampling
interval is above 1 is assumed to be in minutes, so a 4-hourly timecourse already
in hours opens with 48 h read as 48 minutes. Check the header line on every
upload.

**QC flags do not survive detrending.** *Flat trace* uses `(max − min) / |mean|`,
and the mean of a detrended trace is near zero, so the ratio is meaningless. On
raw luciferase traces the opposite happens: every well trips *Strong drift*,
because a decaying baseline is normal there rather than a defect.

**No LICENSE file yet.** One is needed before the repository can be archived for
a DOI. Note that the dependency chain includes rpy2 (GPL-2.0), which constrains
the choice for the distributed whole.
