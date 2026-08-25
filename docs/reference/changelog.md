# Changelog

Versioning follows a patch digit within the minor series. `0.8.0` is the first
tagged public release; everything before it was development numbering, and the
`0.7.x` entries below are kept so the changes remain traceable.

If you use Chronotopia in published work, [record the version](citing.md) — the
numerical defaults below change between releases.

## 0.8.0 — first tagged release

The first version published as a release rather than a moving branch. It gathers
everything from the 0.7 development line and adds a preprocessing overhaul.

**Detrending is now two choices, not one.** The `Detrending` dropdown picks how
the baseline is *estimated*; a new `Baseline removal` control picks whether it is
*subtracted* or *divided out*. For bioluminescence the baseline is a multiplying
factor, so subtracting it leaves the rhythm looking like it damps far faster than
it does — on the long-series tutorial, where damping is planted at 120–165 h,
subtraction reports 89–94 h and division 161 h. See
[Preprocessing](preprocessing.md).

**Detrending estimators: LOESS added, the menu kept short.** Six estimators —
Linear, Cubic, Rolling mean, LOESS, Exponential fit, and Rolling Hilbert for
envelope removal. LOESS costs almost the same amplitude at any period in range,
where a moving average at a 24 h window keeps 116% of a 20 h rhythm and 84% of a
28 h one.

**The rolling window now tracks the period you are measuring** instead of a fixed
~20 h, and the app reports the amplitude cost at both ends of your period range.
The old default quietly scaled amplitudes by anything between 0.65 and 1.2
depending on a sample's period.

**Detrending no longer breaks on missing data.** Linear used to raise and stop
the app on a single missing point; Cubic and Rolling Hilbert silently returned an
empty column, removing that sample from the analysis without saying so. Every
method now leaves gaps as gaps.

**Damped cosinor**, as both a period-estimation method and a ninth feature
package. It fits `A·exp(-t/τ)·cos(2πt/T + φ) + C`, so damping is measured rather
than removed, and every parameter carries a standard error. Its worst-well period
error on the tutorial data is 0.47 h against 0.63 h for the next best method.
Feature count goes from 100 to 108 on a long recording.

**The feature volcano is readable again.** It asked for ten concept colours from
a five-colour palette and got the palette back twice, so two pairs of concepts
were indistinguishable. Concept is now carried by hue *and* marker shape.

**New documentation**: a [Preprocessing](preprocessing.md) reference page, and a
[Control reference](controls.md) generated from the same text as the in-app
tooltips, so the manual cannot drift from the app.

## 0.7.14 — feature extraction page

A page for the part of Chronotopia that has no equivalent elsewhere: the ~100
features per sample, and the layer that makes them usable.

Documents the eight packages, the ten concepts, the `biology` / `recording` role
flag and why it exists, the five independent period estimators and how far apart
they land, feature quality and redundancy clustering, the differential comparison
with its automatic test selection, and cohort context.

Every count on the page is measured from the long-series tutorial data and
asserted in CI — the verification harness went from 40 to 62 checks.

## 0.7.13 — documentation site

This site. MkDocs Material, deployed to GitHub Pages by CI.

Eleven figures rendered from the tutorial datasets by `docs/make_figures.py`,
using the app's own `styles.py`, so what you see here matches what you see on
your screen. Figures are committed, and the build **fails if a committed figure
differs from a freshly rendered one** — a figure cannot silently disagree with
the analysis that produced it.

Added `CITATION.cff`, and `docs/check_docs.py`, which guards the latent
collision between the `docs.py` module and the `docs/` directory.

## 0.7.12 — tutorial datasets

Two worked examples with generated data, walkthroughs, and a 40-check
verification harness.

**Short series** — 48 h at 4-hourly sampling, 24 transcripts, 3 replicates per
timepoint. Demonstrates that 48 h cannot support a period estimate (Lomb-Scargle
returns 22.9–25.3 h for transcripts all planted at 24.0) while recovering phase
within half an hour. Ships transcripts that are rhythmic but below the detection
limit of the design, so the ground truth distinguishes *planted* from
*detectable*.

**Long series** — 144 h at 10-minute sampling, 24-well plate, four genotypes,
two days of entrainment then four of free-run. Demonstrates trimming, entrainment
exclusion, detrending, and that three period methods agree on the genotype
*comparison* to within 0.35 h while disagreeing on the *absolute* value by up to
0.7 h.

Both datasets carry deliberate mess: replicate rows, a lost replicate, missing
values, a medium-change transient, baseline drift, damping.

## 0.7.11 — tooltips

`docs.py`: every tooltip in one place, attached automatically, plus an in-app
control reference. `verify.py` section 23 fails if a label in `app.py` has no
entry, so a control cannot ship undocumented.

## 0.7.10 — feature analytics

`features.py`: a data dictionary mapping all ~100 features to concepts, a
differential comparison across conditions with effect sizes and FDR correction,
quality and redundancy analysis, and cohort context. NaN-safe Benjamini-Hochberg.

## 0.7.9 — period sweep

A cosinor sweep across a period grid, showing the whole landscape rather than one
number. Pseudo-inverse fix for singular cosinor design matrices.

## 0.7.8 — plate overlays

Well-level overlays for period, acrophase, amplitude, R², noise and the
rhythmicity verdict. ML model picker gated on the Tempo method; lazy classifier
loading.

## 0.7.7 — report fixes

Fixed a crash when generating a report before any analysis had run. Entrainment
overlays and readable annotation in the PDF. Tempo promoted to a testing method
rather than an extra verdict stapled to every run.

## 0.7.6 — grid and style leak

x gridlines on all grid styles. Fixed rcParams leaking between styles —
matplotlib's rcParams are process-global and survive Streamlit reruns, so a key
set by one style persisted into the next style that did not mention it.

## 0.7.5 — plot styling

`styles.py`: nine styles, four contexts, curated palettes with measured
colour-vision-deficiency separation, and a publication export baseline.

## 0.7.4 — comparison views

**Compare samples** (up to 5 traces) and **Compare conditions** (2–4 groups).

The palette needed real work: seaborn's `colorblind` fails all-pairs separation
at 5 series. 420 five-colour combinations were searched against the accessibility
gates; eight passed. The winner became the app default in 0.7.5.

Also fixed a **pyBOAT figure leak**: `WAnalyzer.compute_spectrum` defaults to
`do_plot=True`, so wavelet — the default period method — drew and never closed
one matplotlib figure *per sample, per rerun*. A 96-well plate leaked 96 figures
on every widget interaction. Passing `do_plot=False` made the default period
method **9× faster** with periods unchanged.

## 0.7.3 — plates and plots

Plate detection for 6, 12, 24, 48, 96 and 384-well formats from sample names.
New `plates.py` and `plots.py`; all figure drawing moved out of `methods.py`.

## 0.7.2 — smoothing changes

Savitzky-Golay in. Butterworth band-pass out — its 18–30 h band was hardcoded,
ignored the period slider, and frequently landed above Nyquist for coarsely
sampled recordings. The duplicate Hilbert detrend out, having become
bit-for-bit identical to "Rolling Hilbert" once the 0.7.1 fixes landed.

## 0.7.1 — correctness fixes

`delta_t` recomputed after unit conversion (everything downstream treats it as
hours, so any file not already in hours had smoothing windows, filter cutoffs
and header text wrong by the conversion factor). Autocorrelation returned in
hours. Corrected the Hilbert transform axis. Corrected the wavelet sampling
frequency — `fs` is samples per hour, not the interval. Added guards throughout.

## 0.7 — baseline

The version the review started from.

---

## Known issues

**Replicate rows break three of four period methods.** `app.py` passes the data
to `methods.period_estimation` without averaging repeated timepoints. FFT raises
`AssertionError: Time array must be uniformly sampled`, Autocorrelation returns
NaN, and Wavelet reads the sampling interval off the row spacing rather than the
timepoints and comes back ~1.6 h wrong. Lomb-Scargle is correct. Average your
replicates before upload if you need another method.

**Time unit auto-guess is wrong for hourly data.** Any file whose sampling
interval is above 1 is assumed to be in minutes, so a 4-hourly timecourse
already in hours opens with 48 h read as 48 minutes.
[Check the header line](../getting-started/data-format.md#time-units) on every
upload.

**Lomb-Scargle's frequency grid is coarse at long periods.** `autopower()` runs
at astropy's default resolution, putting consecutive candidate periods ~1.4 h
apart near 26 h over a 96 h window. Long periods come back quantised, and
repeated identical values across wells look like agreement between them.

**`Period = nan h`.** On the built-in example dataset with the default period
method, the Lineplot title can print a raw NaN — the wavelet ridge comes back
empty under the current power threshold.

**`qc_flags` does not identify arrhythmic wells, on either input.** The verdict
flips depending on whether you detrended, and neither answer is useful. On raw
traces every well trips *Strong drift*, because a decaying baseline is normal
for a luciferase recording rather than a defect. On detrended traces *Flat
trace* uses `(max − min) / |mean|`, and the mean of a detrended trace is near
zero, so the ratio is dominated by that denominator and ranks arrhythmic wells
as having *more* dynamic range than wild type. Separately, `cosinor_r2 < 0.05`
is too lax to catch wells sitting at 0.12–0.17, and `cycles_n_complete_cycles`
counts peaks found in noise. Use the Rhythm strength concept directly for now.

**`plot_summary` renders overlapping axes and an empty panel.** The
per-package `plot_*` methods are fine individually; it is the multi-panel
layout that is broken. Separately, `plot_wavelet_ridge` plots against sample
index rather than hours.
