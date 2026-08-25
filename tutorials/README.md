# Tutorial datasets

Two synthetic datasets used by the Chronotopia tutorials. The walkthroughs live
on the documentation site:

- **[Tutorial 1 — a short time series](https://borfebor.github.io/chronotopia/tutorials/short-series/)**
  · 48 h at 4-hourly sampling, 24 transcripts, 3 replicates per timepoint
- **[Tutorial 2 — a long time series](https://borfebor.github.io/chronotopia/tutorials/long-series/)**
  · 144 h at 10-minute sampling, 24-well plate, 4 genotypes

The pages are written from `docs/tutorials/*.md`; this directory holds only the
data and the scripts that produce and check it.

## Files

**Tutorial 1**

| File | Role |
|---|---|
| `tutorial_1_short_series_omics.csv` | the data — upload this |
| `tutorial_1_short_series_layout.csv` | `Sample, Condition` — upload as layout |
| `tutorial_1_short_series_truth.csv` | the answers — read at the end |

**Tutorial 2**

| File | Role |
|---|---|
| `tutorial_2_long_series_luciferase.csv` | the data — upload this |
| `tutorial_2_long_series_layout.csv` | `Sample, Condition` — upload as layout |
| `tutorial_2_long_series_entrainment.csv` | `Time`, 0/1 zeitgeber — for "upload" entrainment mode |
| `tutorial_2_long_series_truth.csv` | the answers — read at the end |

## Regenerating

```bash
python tutorials/make_tutorial_data.py
```

Fixed seeds, so the files come back byte-identical. numpy and pandas only — no
Chronotopia import, so this works without a full app install.

## Verifying

```bash
python tutorials/verify_tutorial_data.py
```

Runs both datasets through Chronotopia's own `methods.py` and `plates.py` and
asserts every claim the walkthroughs make: the file formats, plate detection,
the layout merge, the entrainment cycle count, the recovered periods and phases
against the planted ones, and the separation between the rhythmic, borderline
and flat groups. 40 checks. Run it after any change to `methods.py` — if a
walkthrough goes stale, this is what tells you.

It also prints, separately, the app behaviours the tutorials work around. Those
do not fail the run:

- **Time unit auto-guess.** Any file whose sampling interval is above 1 is
  assumed to be in minutes. Tutorial 1's data is in hours at 4 h intervals, so
  it opens with the wrong unit selected.
- **Replicate rows and period estimation.** `app.py` passes `fr_data` to
  `methods.period_estimation` without averaging repeated timepoints. FFT raises
  `AssertionError: Time array must be uniformly sampled`, Autocorrelation
  returns NaN, and Wavelet reads the sampling interval off the row spacing and
  comes back ~1.6 h wrong. Only Lomb-Scargle handles replicate rows correctly.
- **Lomb-Scargle frequency grid.** `autopower()` is called with astropy's
  default resolution, which over a 96 h window puts consecutive candidate
  periods ~1.4 h apart near 26 h. Long periods come back quantised.

CI runs this on every push that touches `methods.py`, `plates.py` or `styles.py`.
