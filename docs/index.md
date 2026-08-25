---
hide:
  - navigation
---

# Chronotopia

**Analysis of time-course data for circadian biology.** Upload a recording, and
Chronotopia will preprocess it, estimate periods, test for rhythmicity, extract
around a hundred features per sample, and produce publication-ready figures and
a multi-page PDF report — in a browser, with no scripting.

It handles both ends of the experimental range: a transcriptomics timecourse of
thirteen timepoints with replicates, and a six-day bioluminescence recording of
a 96-well plate at ten-minute resolution.

---

## Start here

<div class="grid cards" markdown>

- :material-download: **[Install it](getting-started/installation.md)**

    Docker in three commands, or a local Python environment.

- :material-file-table: **[Get your data in](getting-started/data-format.md)**

    One time column, one column per sample. What the layout file needs, and how
    plates are detected.

- :material-school: **[Tutorial 1 — short series](tutorials/short-series.md)**

    A 48 h transcriptomics timecourse. Rhythmicity, phase, and why you should
    not ask this experiment for a period.

- :material-school-outline: **[Tutorial 2 — long series](tutorials/long-series.md)**

    Six days of luciferase across a 24-well plate. Trimming, entrainment,
    detrending, and period estimation done properly.

- :material-tag-multiple: **[Feature extraction](features.md)**

    ~100 numbers per sample across nine packages, each one named, grouped by
    concept, and checked for redundancy before you use it.

- :material-tune: **[Preprocessing](preprocessing.md)**

    Every detrending method, what each one costs, and why subtracting a
    multiplicative baseline invents damping that is not there.

</div>

---

## What it does

**[Preprocessing](preprocessing.md)** — smoothing (rolling mean,
Savitzky-Golay, DCT, resampling), normalisation (z-score, sample-wise and global
min-max), and detrending: six baseline estimators (linear, cubic, rolling mean,
running median, sinc low-pass, exponential fit) that you can either subtract or
divide by, plus rolling Hilbert for envelope removal. Entrainment windows can be
excluded from everything downstream, so a driven rhythm is never mistaken for a
clock.

**Period estimation** — Lomb-Scargle, wavelet transform (pyBOAT), FFT,
autocorrelation, a damped cosinor that fits the decay instead of ignoring it, and
a cosinor sweep that shows the whole period landscape rather than one number.

**Rhythmicity testing** — MetaCycle (`meta2d`, JTK, ARS, LS) where R is
available, plus a permutation cosinor and a random-forest classifier that need
only Python. All FDR-corrected.

**Features** — around a hundred numbers per sample across nine packages, each
mapped to a concept (period, phase, amplitude, rhythm strength, waveform shape,
harmonics, damping, trend, noise, recording) with a data dictionary, redundancy
clustering, and a differential comparison across conditions.

**Plates** — 6, 12, 24, 48, 96 and 384-well formats, detected from the sample
names. Well-level overlays for period, acrophase, amplitude, R², noise and the
rhythmicity verdict.

---

## The figures in this documentation are generated, not drawn

Every plot on this site is rendered from the tutorial datasets by
[`docs/make_figures.py`](https://github.com/borfebor/chronotopia/blob/main/docs/make_figures.py),
using the app's own styles and palette. The datasets themselves are generated
from a stated model with fixed seeds, and every claim the tutorials make about
what you will see is asserted by a test that runs in CI.

If the analysis code changes and a documented result moves, the build fails.
See [Reproducibility](reference/reproducibility.md).

---

!!! info "Chronotopia is research software"

    It is under active development and the interface changes between versions.
    If you use it in published work, please [cite the version you
    used](reference/citing.md) — the changelog records what changed and when.
