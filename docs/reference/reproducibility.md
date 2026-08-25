# Reproducibility

Everything on this site — every dataset, every figure, every number quoted in a
tutorial — is produced by a committed script from a fixed seed, and checked
against the application's own code in CI. This page says how, so a reviewer can
re-run it.

## Three commands

```bash
python tutorials/make_tutorial_data.py     # regenerate both datasets
python tutorials/verify_tutorial_data.py   # 40 checks against methods.py / plates.py
python docs/make_figures.py                # re-render every figure on this site
```

The datasets come back **byte-identical**. `make_tutorial_data.py` imports only
NumPy and pandas — no Chronotopia code — so it runs without a full app install,
and the seeds (`20260804`, `20260805`) are constants in the file.

## What the datasets are

Neither tutorial dataset is real. Both are generated from a model written out in
full in the generator, with every planted parameter recorded in a `_truth.csv`
next to the data.

**Short series** — 24 transcripts, log2 CPM, rhythms modelled as a cosine on the
log scale with a small second harmonic for the *Per* genes. Peak phases and
relative amplitudes follow published mouse liver behaviour. Three transcripts sit
deliberately at the edge of what 13 timepoints at n=3 can resolve.

**Long series** — 24 wells of counts per second, built as
`baseline(t) × (1 + envelope(t) × waveform(phase(t)))` with Poisson read noise on
top. Values are positive counts rather than a centred sine; the baseline decays
as the medium is consumed; the oscillation damps only after release; a
medium-change transient dominates the first two hours.

Synthetic data is honest about what it contains and silent about everything it
does not. These datasets are for learning the tool and for checking that an
analysis pipeline recovers a known answer — not a substitute for testing on real
recordings.

## What the harness checks

`tutorials/verify_tutorial_data.py` runs both datasets through the app's own
`methods.py` and `plates.py`, not a reimplementation. Forty checks, covering:

| Area | Checks |
|---|---|
| Format | import through `methods.importer`, layout schema, sampling interval, replicate structure, missing-value behaviour |
| Plates | 24-well detection from names, survival of the layout rename, `group_by_geometry('Row')` reproducing the genotypes |
| Entrainment | the schedule file resolving to 2 cycles of 24 h ending at hour 48 |
| Short series | recovered phases within 1.5 h of planted; the three amplitude bands separating; period *not* being recoverable |
| Long series | three period methods within 1.5 h of planted; genotypes ordered and separated by more than method error; detrending measurably helping; arrhythmic wells carrying no rhythm after release |

Every claim the tutorials make about what you will see is asserted here. If
`methods.py` changes and a documented result moves, this fails.

## What CI does

The [`docs` workflow](https://github.com/borfebor/chronotopia/blob/main/.github/workflows/docs.yml)
runs on every push that touches the docs, the tutorials, or the three modules
the figures depend on (`methods.py`, `plates.py`, `styles.py`). In order:

1. `docs/check_docs.py` — confirms the `docs/` directory has not shadowed
   `docs.py` (see below)
2. `tutorials/verify_tutorial_data.py` — the 40 checks
3. `docs/make_figures.py` — re-renders every figure
4. **a staleness check** — if a freshly rendered figure differs from the
   committed one, the build fails

That last step is the one that matters. Figures are committed so they are
visible in a diff, and the build refuses to pass if a committed figure no longer
matches what the code produces. A figure on this site cannot silently disagree
with the analysis it came from.

Finally `mkdocs build --strict`, which turns broken internal links and missing
files into errors rather than warnings.

## Figures

`docs/make_figures.py` renders all eleven figures to SVG and PNG. It imports
Chronotopia's `styles.py` and applies the same "Journal" style, "notebook"
context and "Chronotopia" palette the app ships with, so a figure here looks
like what you see on your own screen.

The palette is the one built in v0.7.4 and validated for colour-vision
deficiency: worst-pair separation ΔE 13.0 (deutan), 16.3 for normal vision,
across **all** pairs rather than only neighbouring ones. Two of its five slots
fall below 3:1 contrast on white, so every multi-series figure here carries both
a legend and direct labels — identity never rests on colour alone. Label text is
ink rather than the series colour, for the same reason.

## A structural note: `docs.py` and `docs/`

The repository contains both a module `docs.py` (every tooltip in the app) and a
directory `docs/` (this site). Python resolves a regular module ahead of a
namespace package, so `import docs` in `app.py` finds `docs.py` and everything
works.

That is a property of import precedence, not of intent. Adding
`docs/__init__.py` would make `docs/` a regular package, flip the resolution,
and break the app at startup with an error pointing nowhere near the cause.
`docs/check_docs.py` asserts this in CI.

If it ever fails, the durable fix is to rename `docs.py` to `tooltips.py` and
update the import in `app.py` and the section-23 check in `verify.py`.

## Environment

CI runs Python 3.11, matching the Docker image. `rpy2` is not installed for the
docs build — R and MetaCycle are needed only for the `meta2d`, `JTK`, `ARS` and
`LS` testing methods, not for period estimation, feature extraction or any
figure on this site.

`scikit-learn` is pinned to 1.8.0 because the bundled random-forest classifier
was trained under that version and the version string is embedded in the pickle.
Retrain the model before changing that pin.
