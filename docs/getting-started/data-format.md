# Your data

Chronotopia reads `.csv`, `.tsv`, `.txt` (tab-separated) and `.xlsx`. Everything
below applies to all four.

## The data file

One row per measurement, one column per sample, and a column holding time.

| Time | Sample_A | Sample_B | Sample_C |
|---|---|---|---|
| 0 | 4.12 | 3.88 | 4.51 |
| 4 | 5.02 | 4.71 | 4.49 |
| 8 | 4.66 | 4.30 | 4.55 |

That is the whole requirement. A few details that matter in practice:

**The time column can be called anything.** You pick it from the **Time column**
selector after upload; the first column is not assumed.

**Column names become sample names.** They are stripped of surrounding
whitespace on load and otherwise used as-is, so they end up on every figure and
in every results table. Name them how you want them cited.

**Replicates are repeated timepoints.** Three biological replicates at 0 h means
three rows with `Time = 0`. Chronotopia detects this and offers the mean ± SD
and mean + replicates views.

!!! warning "Replicates and period estimation, v0.8.0"

    Repeated timepoints are handled correctly by the plots and the rhythmicity
    tests, but the period estimators are passed the rows as they stand. On
    replicate data, **Fast Fourier Transform** raises an error about non-uniform
    sampling, **Autocorrelation** returns nothing, and **Wavelet Transform**
    reads the sampling interval off the row spacing and comes back roughly 1.6 h
    wrong. **Lomb-Scargle Periodogram** is correct.

    If you need another method, average your replicates before upload.

## Time units

After upload, set **Time unit** to whatever your file is in — minutes, hours,
days or seconds. Everything downstream works in hours.

!!! danger "Check this selector on every upload"

    The guess is made from the size of the sampling interval: anything above 1
    is assumed to be minutes, because most raw instrument exports are. **Data
    already in hours at intervals coarser than one hour is therefore guessed
    wrong** — a 4-hourly timecourse in hours opens as though it were 4-minute
    sampling, and 48 h of data is read as 48 minutes.

    The header line under *Data Preview* is the fastest check:

    > Experiment with 24 sample recorded for 48.0 hours (recorded every = 4.0 h)

    If the duration is not what you expect, the unit is wrong.

## Missing values

Rows containing a missing value are dropped — **the whole row, for every
sample**, not just the sample with the gap. One unquantified protein at one
timepoint therefore costs that timepoint for everything else in the file.

If one sample is much gappier than the rest, exclude it first with **Exclude
samples from data** in the sidebar. You keep the timepoints; you lose only the
sample you chose to lose.

## The layout file

Optional, and worth doing. Two columns:

```csv
Sample,Condition
A1,Wild type
A2,Wild type
B1,Mutant
B2,Mutant
```

`Sample` must match the data file's column names exactly. `Condition` is
whatever grouping you want to compare — genotype, treatment, tissue, gene class.

Upload it under **Upload experimental layout**. A pre-filled template with your
sample names already in it is available from the same panel, which saves
retyping 96 well IDs.

With a layout attached you gain:

- **Lineplot [Mean ± SD]** and **Lineplot [Mean + Replicates]**
- **Compare conditions** (2 to 4 groups)
- condition-aware feature comparison, with effect sizes and FDR correction
- grouped sections in the PDF report
- **Exclude by** condition rather than sample by sample

Extra columns are allowed and become additional options in the exclusion menu.

## Plates

If your sample names carry well IDs, the plate format is detected automatically
— 6, 12, 24, 48, 96 and 384 are supported, and the smallest format that holds
every well is chosen.

The parser is tolerant. All of these resolve:

| Name | Well |
|---|---|
| `A1` | A1 |
| `a01` | A1 |
| `Well B02` | B2 |
| `sample_H12_ctrl` | H12 |
| `Plate1_C7` | C7 |
| `P53_A07` | A7 |

And these correctly do not:

| Name | Why |
|---|---|
| `sample_1` | no row letter |
| `A0` | plates are 1-indexed |
| `P53` | no plate has 53 columns |

At least 60% of columns must yield a well ID before the names are trusted. A
bare sample count is only accepted for 96 and 384 — a file with 24 columns is
far more likely to be 24 samples than a 24-well plate, and guessing wrong there
produces grouped statistics you never asked for.

Detection gets you the **Plate view** and the option to **Group wells by** row
or column. Note that geometric grouping is opt-in and never invented for you: an
uploaded layout always wins.

## The entrainment file

Only needed if you use **Entrainment parameters → Mode → upload**. Two columns:
time, and the zeitgeber state.

```csv
Time,Zeitgeber
0.0,0
0.1667,0
...
12.0,1
```

The **last** column is read as the signal and the first as time. A 0/1 signal is
counted by rising edges, so **start your schedule in the off half** — a file
beginning with the lights already on loses its first edge and reports one cycle
fewer than it has. Continuous signals (a recorded temperature trace, for
instance) are handled through the Hilbert envelope instead.

## A worked example of each

The two tutorial datasets are complete, well-formed examples you can open next
to your own file:

<div class="ct-downloads" markdown>
[:material-file-delimited: Short series data](../downloads/tutorial_1_short_series_omics.csv)
[:material-file-delimited: Short series layout](../downloads/tutorial_1_short_series_layout.csv)
[:material-file-delimited: Long series data](../downloads/tutorial_2_long_series_luciferase.csv)
[:material-file-delimited: Long series layout](../downloads/tutorial_2_long_series_layout.csv)
[:material-file-delimited: Entrainment schedule](../downloads/tutorial_2_long_series_entrainment.csv)
</div>
