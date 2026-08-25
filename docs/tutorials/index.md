# Tutorials

Two worked examples. Each comes with a dataset you upload straight into the app
and a walkthrough that names the settings to pick, why, and what you should see.
Do them in order — the second assumes you know where the controls are.

| | [1 · Short series](short-series.md) | [2 · Long series](long-series.md) |
|---|---|---|
| Experiment | transcriptomics / proteomics | bioluminescence reporter |
| Duration | 48 h | 144 h |
| Sampling | every 4 h — 13 timepoints | every 10 min — 865 timepoints |
| Samples | 24 transcripts, 3 replicates each | 24 wells, 4 genotypes |
| Teaches | rhythmicity, phase, missing data, detection limits | trimming, entrainment, detrending, period, plates |
| Does **not** teach | period estimation — 48 h cannot support it | replicate handling |

## The data is synthetic, and that is the point

Neither dataset is taken from a repository. Both are generated from a stated
model with fixed seeds, which buys three things a real dataset cannot:

**Known answers.** Every period, phase, amplitude and rhythmicity call was
planted deliberately and is written down in a `_truth.csv` shipped alongside the
data. You can check your analysis instead of trusting it.

**Known detection limits.** Tutorial 1 contains transcripts whose rhythms are
real but too small for the design to find. No public dataset lets you
demonstrate that distinction, because nobody knows the ground truth there
either.

**Deliberate mess.** Replicate rows, a lost replicate, missing values, a
medium-change artefact, baseline drift, damping. Each is present because it
changes what you should do, and each is called out where it matters.

They are not a substitute for testing on real data before trusting a result.
Synthetic data is honest about what it contains and silent about everything it
does not.

## Regenerating and checking

```bash
python tutorials/make_tutorial_data.py     # rebuilds both datasets, byte-identically
python tutorials/verify_tutorial_data.py   # 40 checks against the app's own code
```

The verification harness asserts every claim these tutorials make — the file
formats, plate detection, the layout merge, the entrainment cycle count, the
recovered periods and phases against the planted ones, and the separation
between the rhythmic, borderline and flat groups. It runs in CI, so a tutorial
cannot quietly go stale.
