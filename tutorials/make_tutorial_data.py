"""
Generate the two Chronotopia tutorial datasets.

Both are synthetic, but built to behave like the real thing: the numbers come
from a stated model, the ground truth is written out alongside them, and the
awkward parts of real files (replicate rows, a lost replicate, missing values,
baseline drift, damping, a medium-change artefact) are present on purpose.

Run:
    python tutorials/make_tutorial_data.py

Writes into the directory this file lives in:

    tutorial_1_short_series_omics.csv          data
    tutorial_1_short_series_layout.csv         layout (Sample, Condition)
    tutorial_1_short_series_truth.csv          ground truth
    tutorial_2_long_series_luciferase.csv      data
    tutorial_2_long_series_layout.csv          layout (Sample, Condition)
    tutorial_2_long_series_entrainment.csv     light schedule for "upload" mode
    tutorial_2_long_series_truth.csv           ground truth

Nothing here imports Chronotopia — numpy and pandas only — so the files can be
regenerated without a working app install.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

SEED_OMICS = 20260804
SEED_LUC = 20260805


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1 — short series: a circadian transcriptomics / proteomics timecourse
# ─────────────────────────────────────────────────────────────────────────────
#
# Design: mouse liver, released into constant darkness, sampled every 4 h for
# 48 h, three biological replicates per timepoint. 13 timepoints, 39 rows.
#
# Values are log2 CPM. Rhythms are modelled as a cosine on the log scale with a
# small second harmonic for the Per genes (their profiles are visibly peaked
# rather than sinusoidal). Peak phases and relative amplitudes follow published
# mouse liver behaviour closely enough to be recognisable.
#
# Three transcripts sit near the limit of what 13 timepoints at n=3 can resolve
# — CLOCK, CRY2, SLC2A1. That is deliberate: the tutorial uses them to show what
# an FDR-corrected result looks like when a study is at the edge of its power.

#            name         baseline  amp   peak(CT)  harm2  noise  class
TRANSCRIPTS = [
    ("ARNTL",      7.4,  1.10, 23.0, 0.10, 0.22, "Core clock"),
    ("CLOCK",      8.1,  0.15, 21.0, 0.00, 0.20, "Core clock"),
    ("NPAS2",      6.2,  0.90, 23.5, 0.05, 0.26, "Core clock"),
    ("PER1",       7.9,  1.00, 10.0, 0.22, 0.22, "Core clock"),
    ("PER2",       7.2,  1.20, 13.0, 0.25, 0.22, "Core clock"),
    ("PER3",       6.0,  1.00, 12.0, 0.20, 0.28, "Core clock"),
    ("CRY1",       7.0,  0.80, 17.0, 0.08, 0.24, "Core clock"),
    ("CRY2",       7.6,  0.35, 10.0, 0.00, 0.24, "Core clock"),
    ("NR1D1",      8.6,  2.20,  5.0, 0.30, 0.22, "Core clock"),
    ("NR1D2",      7.7,  0.90,  7.0, 0.12, 0.23, "Core clock"),
    ("DBP",        8.9,  1.80, 10.0, 0.25, 0.22, "Clock-controlled"),
    ("TEF",        7.8,  0.70, 10.5, 0.08, 0.23, "Clock-controlled"),
    ("HLF",        7.1,  0.80, 11.0, 0.10, 0.25, "Clock-controlled"),
    ("CIART",      6.8,  1.50,  8.0, 0.20, 0.25, "Clock-controlled"),
    ("RORA",       6.5,  0.40, 18.0, 0.00, 0.26, "Clock-controlled"),
    ("WEE1",       6.9,  0.50, 12.5, 0.06, 0.25, "Clock-controlled"),
    ("SLC2A1",     7.3,  0.30,  2.0, 0.00, 0.24, "Clock-controlled"),
    ("PPARGC1A",   5.9,  0.90, 22.0, 0.10, 0.30, "Clock-controlled"),
    ("ACTB",      13.1,  0.00,  0.0, 0.00, 0.14, "Non-rhythmic"),
    ("GAPDH",     12.4,  0.00,  0.0, 0.00, 0.15, "Non-rhythmic"),
    ("TUBB",      10.8,  0.00,  0.0, 0.00, 0.17, "Non-rhythmic"),
    ("RPL13A",    12.9,  0.00,  0.0, 0.00, 0.14, "Non-rhythmic"),
    ("B2M",       11.2,  0.00,  0.0, 0.00, 0.18, "Non-rhythmic"),
    ("HPRT1",      9.3,  0.00,  0.0, 0.00, 0.20, "Non-rhythmic"),
]

# What 13 timepoints at n=3 with this much noise can actually support. Measured,
# not assumed: verify_tutorial_data.py fits a 24 h cosinor to each transcript and
# checks that these three bands really do separate. Anything in the middle band
# is labelled "borderline" rather than claimed as a rhythm — CLOCK sits below
# even that, which is faithful to liver, where CLOCK mRNA is barely rhythmic.
DETECTABLE_AMP = 0.45
BORDERLINE_AMP = 0.20

PERIOD_TRUE = 24.0
N_REPLICATES = 3
LOST_REPLICATE_AT = 20.0   # one timepoint yields only 2 replicates
MISSING_IN = "PPARGC1A"    # a low-abundance transcript with unquantified values
MISSING_AT = [(8.0, 2), (36.0, 1)]   # (time, replicate index) left blank


def make_short_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED_OMICS)

    times = np.arange(0.0, 48.0 + 1e-9, 4.0)

    rows = []
    for t in times:
        n_rep = N_REPLICATES - 1 if np.isclose(t, LOST_REPLICATE_AT) else N_REPLICATES
        for rep in range(1, n_rep + 1):
            rows.append((t, rep))

    frame = pd.DataFrame(rows, columns=["Time", "_rep"])

    for name, base, amp, peak, harm2, noise, _cls in TRANSCRIPTS:
        t = frame["Time"].to_numpy()
        theta = 2 * np.pi * (t - peak) / PERIOD_TRUE
        signal = base + amp * np.cos(theta) + amp * harm2 * np.cos(2 * theta)

        # Per-replicate offset: biological replicates differ in overall level,
        # which is what makes a paired look at the data worthwhile.
        rep_offset = {1: 0.0, 2: 0.12, 3: -0.09}
        signal = signal + frame["_rep"].map(rep_offset).to_numpy()

        signal = signal + rng.normal(0.0, noise, size=len(frame))
        frame[name] = np.round(signal, 4)

    for t_missing, rep_missing in MISSING_AT:
        mask = np.isclose(frame["Time"], t_missing) & (frame["_rep"] == rep_missing)
        frame.loc[mask, MISSING_IN] = np.nan

    data = frame.drop(columns="_rep")

    layout = pd.DataFrame(
        {
            "Sample": [t[0] for t in TRANSCRIPTS],
            "Condition": [t[6] for t in TRANSCRIPTS],
        }
    )

    truth = pd.DataFrame(
        {
            "Sample": [t[0] for t in TRANSCRIPTS],
            "Class": [t[6] for t in TRANSCRIPTS],
            # What was put in ...
            "Planted_rhythm": ["yes" if t[2] > 0 else "no" for t in TRANSCRIPTS],
            # ... and what this experimental design can actually get back out.
            "Detectable": [
                "yes" if t[2] >= DETECTABLE_AMP
                else ("borderline" if t[2] >= BORDERLINE_AMP else "no")
                for t in TRANSCRIPTS
            ],
            "True_period_h": [PERIOD_TRUE if t[2] > 0 else np.nan for t in TRANSCRIPTS],
            "True_peak_phase_CT_h": [t[3] if t[2] > 0 else np.nan for t in TRANSCRIPTS],
            "True_log2_amplitude": [t[2] for t in TRANSCRIPTS],
            "Noise_sd_log2": [t[5] for t in TRANSCRIPTS],
        }
    )

    return data, layout, truth


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2 — long series: a bioluminescence reporter recording
# ─────────────────────────────────────────────────────────────────────────────
#
# Design: PER2::LUC fibroblasts in a 24-well plate, read every 10 minutes for
# 6 days. The first 48 h are a 12:12 temperature cycle (the zeitgeber); at 48 h
# the cycle is released and the cells free-run for 96 h.
#
# Row of the plate = genotype:
#   A  wild type            intrinsic period 24.3 h
#   B  short-period mutant  intrinsic period 22.1 h   (a CK1e tau-like allele)
#   C  long-period mutant   intrinsic period 26.4 h
#   D  arrhythmic mutant    no self-sustained rhythm after release
#
# Counts are built as   baseline(t) * (1 + envelope(t) * waveform(phase(t)))
# with Poisson read noise on top, so:
#   - values are positive counts/s, not a centred sine
#   - the baseline decays as the medium is consumed  -> detrending is required
#   - the oscillation damps over days                -> amplitude is not constant
#   - a medium-change transient dominates the first ~2.5 h -> trim the start

LUC_DURATION_H = 144.0
LUC_DT_H = 10.0 / 60.0          # 10-minute sampling
LUC_ENTRAIN_H = 48.0            # two 12:12 cycles, then release
LUC_T_CYCLE = 24.0

GENOTYPES = {
    "A": ("Wild type", 24.3, 1.0),
    "B": ("Short period", 22.1, 0.95),
    "C": ("Long period", 26.4, 0.90),
    "D": ("Arrhythmic", np.nan, 0.0),
}
LUC_COLS = 6                    # 24-well plate is 4 rows x 6 columns


def _skewed_wave(phase: np.ndarray) -> np.ndarray:
    """A cosine with a sharpened peak and a broadened trough.

    Bioluminescence traces are not sinusoidal: the rise is faster than the
    decay. A small second harmonic reproduces that asymmetry without inventing
    structure the period estimators would then have to fight.
    """
    return np.cos(phase) + 0.18 * np.cos(2 * phase - 0.6)


def make_long_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED_LUC)

    t = np.arange(0.0, LUC_DURATION_H + 1e-9, LUC_DT_H)
    n = len(t)

    data = pd.DataFrame({"Time": np.round(t, 4)})
    truth_rows = []

    for row_letter, (condition, intrinsic_period, rhythm_strength) in GENOTYPES.items():
        for col in range(1, LUC_COLS + 1):
            well = f"{row_letter}{col}"

            # ── phase ────────────────────────────────────────────────────────
            # Entrained: locked to the 24 h zeitgeber. After release: advances
            # at the well's own intrinsic frequency, continuing from whatever
            # phase it held at the moment of release (so the trace is smooth).
            well_period = (
                intrinsic_period + rng.normal(0.0, 0.18)
                if np.isfinite(intrinsic_period) else np.nan
            )
            phase0 = rng.normal(0.0, 0.15)          # tight during entrainment

            omega_ent = 2 * np.pi / LUC_T_CYCLE
            entrained = t < LUC_ENTRAIN_H
            phase = np.empty(n)
            phase[entrained] = omega_ent * t[entrained] + phase0

            if np.isfinite(well_period):
                omega_free = 2 * np.pi / well_period
            else:
                omega_free = omega_ent  # unused; damped to nothing below
            phase_at_release = omega_ent * LUC_ENTRAIN_H + phase0
            phase[~entrained] = (
                phase_at_release + omega_free * (t[~entrained] - LUC_ENTRAIN_H)
            )

            # ── envelope ─────────────────────────────────────────────────────
            # Relative amplitude, as a fraction of the running baseline.
            amp0 = rng.uniform(0.32, 0.42) * rhythm_strength
            if rhythm_strength > 0:
                # Damping only starts at release: the zeitgeber holds the
                # population together, and desynchrony sets in once it stops.
                # tau is chosen so the rhythm has visibly damped by day 6 but is
                # still unmistakably there — a reporter that flatlines by day 4
                # would make half the tutorial's analyses pointless.
                damp_tau = rng.uniform(120.0, 165.0)   # hours
                envelope = np.where(
                    entrained,
                    amp0,
                    amp0 * np.exp(-(t - LUC_ENTRAIN_H) / damp_tau),
                )
            else:
                # Arrhythmic wells still follow the zeitgeber while it is on —
                # a driven response, not a clock — and lose it within a cycle.
                envelope = np.where(
                    entrained,
                    amp0 + 0.16,
                    0.16 * np.exp(-(t - LUC_ENTRAIN_H) / 9.0),
                )
                phase[~entrained] = phase_at_release + omega_ent * (
                    t[~entrained] - LUC_ENTRAIN_H
                )

            # ── baseline ─────────────────────────────────────────────────────
            # Substrate consumption: a slow exponential run-down onto a floor,
            # plus a large transient from the medium change at t = 0.
            level = rng.uniform(3600.0, 5000.0)
            floor = rng.uniform(1100.0, 1500.0)
            decay_tau = rng.uniform(100.0, 130.0)
            baseline = floor + (level - floor) * np.exp(-t / decay_tau)

            transient = rng.uniform(4500.0, 7000.0) * np.exp(-t / 0.75)

            clean = baseline * (1.0 + envelope * _skewed_wave(phase)) + transient

            # ── read noise ───────────────────────────────────────────────────
            # Photon counting, so noise scales with the square root of signal.
            counts = rng.poisson(np.clip(clean, 1.0, None)).astype(float)
            data[well] = np.round(counts, 1)

            truth_rows.append(
                {
                    "Sample": well,
                    "Condition": condition,
                    "True_intrinsic_period_h": (
                        round(float(well_period), 3) if np.isfinite(well_period) else np.nan
                    ),
                    "Entrained_period_h": LUC_T_CYCLE,
                    "Rhythmic_after_release": "yes" if rhythm_strength > 0 else "no",
                    "Initial_relative_amplitude": round(float(amp0), 3),
                    "Baseline_decay_tau_h": round(float(decay_tau), 1),
                }
            )

    truth = pd.DataFrame(truth_rows)
    layout = truth[["Sample", "Condition"]].copy()

    # ── entrainment schedule ────────────────────────────────────────────────
    # Chronotopia's "upload" entrainment mode reads the first column as time and
    # the LAST column as the signal, and counts rising edges. Starting in the
    # dark half means both cycles produce a rising edge, so it reports 2 cycles
    # of 24 h and places the release boundary at 48 h.
    warm = np.where(
        (t < LUC_ENTRAIN_H) & (np.floor(t / 12.0).astype(int) % 2 == 1), 1, 0
    )
    entrainment = pd.DataFrame({"Time": np.round(t, 4), "Zeitgeber": warm})

    return data, layout, truth, entrainment


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    short_data, short_layout, short_truth = make_short_series()
    short_data.to_csv(os.path.join(OUT, "tutorial_1_short_series_omics.csv"), index=False)
    short_layout.to_csv(os.path.join(OUT, "tutorial_1_short_series_layout.csv"), index=False)
    short_truth.to_csv(os.path.join(OUT, "tutorial_1_short_series_truth.csv"), index=False)

    long_data, long_layout, long_truth, long_ent = make_long_series()
    long_data.to_csv(os.path.join(OUT, "tutorial_2_long_series_luciferase.csv"), index=False)
    long_layout.to_csv(os.path.join(OUT, "tutorial_2_long_series_layout.csv"), index=False)
    long_truth.to_csv(os.path.join(OUT, "tutorial_2_long_series_truth.csv"), index=False)
    long_ent.to_csv(os.path.join(OUT, "tutorial_2_long_series_entrainment.csv"), index=False)

    print(f"short series : {short_data.shape[0]} rows x {short_data.shape[1] - 1} samples")
    print(f"long series  : {long_data.shape[0]} rows x {long_data.shape[1] - 1} wells")


if __name__ == "__main__":
    main()
