"""
features.py
===========
Makes the feature matrix self-describing, and turns it into answers.

`chronotopia_feature_extractor` computes ~100 numbers per sample. That is a
strength, but a flat table of 100 opaque column names serves nobody: the UI can
only browse it one column at a time, a reader cannot tell `noise_band_snr_ultradian`
from `cosinor_amplitude`, and anyone training a model on it has no way to know
that four of the columns describe the *recording* rather than the biology.

This module adds the missing layer:

  describe_features()   name -> concept, package, role, description
  quality_report()      missingness, variance, usability per feature
  redundancy_clusters() which features are measuring the same thing
  compare_conditions()  every feature at once: effect size, test, FDR
  cohort_percentiles()  where one sample sits in the population
  qc_flags()            which samples look unreliable, and why

Nothing here drops or rewrites data. `role` is a label, not a filter — the export
stays complete and the caller decides what to use.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  Concepts
# ═══════════════════════════════════════════════════════════════════════════
#
# Features are grouped by WHAT THEY MEASURE, not by which package produced them.
# The packages are an implementation detail; "period" is the question. Five of
# the columns are independent estimates of period from four different methods —
# grouping by concept makes that visible instead of implying five separate
# findings.

CONCEPTS: dict[str, str] = {
    "Period": "How long one cycle lasts, and how steady that is. Several independent "
              "estimates live here — cosinor, peak intervals, Lomb-Scargle and the "
              "wavelet ridge — so treat them as one measurement made four ways.",
    "Phase": "When in the cycle the peak falls, and how consistently.",
    "Amplitude": "How large the oscillation is, and how stable that size is.",
    "Rhythm strength": "How convincingly the trace oscillates at all — fit quality, "
                       "spectral power, significance, number of complete cycles.",
    "Waveform shape": "The shape of a cycle rather than its size: rise and fall times, "
                      "width, asymmetry, departure from a pure sinusoid.",
    "Harmonics": "Energy at fractions of the fundamental — 12 h and 8 h components, "
                 "spectral complexity, secondary peaks.",
    "Damping & drift": "How the rhythm changes across the recording: period drifting, "
                       "amplitude damping, half-life.",
    "Trend & baseline": "Non-rhythmic behaviour — slope, depletion, non-stationarity.",
    "Noise & quality": "Residual noise, signal-to-noise, coverage and usable fraction.",
    "Recording": "Properties of the RECORDING, not the biology: duration, number of "
                 "points, sampling interval. Useful for QC. Dangerous as model inputs — "
                 "see the role flag.",
}

# role values
BIOLOGY = "biology"
RECORDING = "recording"

# First match wins. Ordered so specific patterns precede general ones.
_RULES: list[tuple[str, str, str, str]] = [
    # ── recording metadata ──────────────────────────────────────────────────
    (r"^meta_", "Recording", RECORDING,
     "Property of the recording itself, not of the biology."),

    # ── damped cosinor (before the plain cosinor rules — those are anchored at
    #    ^cosinor_ and would not match anyway, but keeping the family together
    #    makes the ordering intent obvious to the next reader) ────────────────
    (r"^damped_cosinor_period(_se)?$", "Period", BIOLOGY,
     "Period of the fitted damped cosine, and its standard error."),
    (r"^damped_cosinor_(damping_tau|damping_tau_se|half_life)$",
     "Damping & drift", BIOLOGY,
     "How fast the rhythm loses amplitude, fitted rather than measured off an "
     "envelope, and how well that decay is pinned down."),
    (r"^damped_cosinor_acrophase", "Phase", BIOLOGY,
     "Time of the first peak of the fitted damped cosine."),
    (r"^damped_cosinor_amplitude$", "Amplitude", BIOLOGY,
     "Amplitude at the start of the window, before any decay."),
    (r"^damped_cosinor_r2$", "Rhythm strength", BIOLOGY,
     "How well one damped sinusoid explains the trace."),

    # ── period ──────────────────────────────────────────────────────────────
    (r"^cosinor_period$", "Period", BIOLOGY, "Period of the best-fitting cosinor."),
    (r"^cycles_period_event_based$", "Period", BIOLOGY,
     "Period from the median interval between detected peaks."),
    (r"^cycles_(ipi|iti)_", "Period", BIOLOGY,
     "Interval between successive peaks (ipi) or troughs (iti)."),
    (r"^harmonic_fundamental_period_h$", "Period", BIOLOGY,
     "Period of the strongest FFT component."),
    (r"^lomb_scargle_peak_period_h$", "Period", BIOLOGY,
     "Period of the tallest Lomb-Scargle peak."),
    (r"^lomb_scargle_bandwidth_h$", "Period", BIOLOGY,
     "Width of the Lomb-Scargle peak — how sharply the period is defined."),
    (r"^waveform_cycle_period_cv$", "Period", BIOLOGY,
     "Cycle-to-cycle variability of the period."),
    (r"^wavelet_ridge_period_(mean|std|cv|iqr|early|late)$", "Period", BIOLOGY,
     "Instantaneous period along the wavelet ridge."),

    # ── damping & drift (must precede the general wavelet/amplitude rules) ──
    (r"^wavelet_ridge_(period|amplitude)_trend_", "Damping & drift", BIOLOGY,
     "Whether the period or amplitude moves systematically across the recording."),
    (r"^wavelet_ridge_period_(is_drifting|change)$", "Damping & drift", BIOLOGY,
     "Whether and how far the period shifts over the recording."),
    (r"^wavelet_ridge_(is_damping|damping_rate|half_life)$", "Damping & drift", BIOLOGY,
     "Loss of amplitude over time, and how fast."),

    # ── phase ───────────────────────────────────────────────────────────────
    (r"^cosinor_acrophase", "Phase", BIOLOGY, "Time of the cosinor peak."),
    (r"^wavelet_ridge_phase_", "Phase", BIOLOGY,
     "Phase consistency along the ridge."),

    # ── amplitude ───────────────────────────────────────────────────────────
    (r"^cosinor_(amplitude|mesor)$", "Amplitude", BIOLOGY,
     "Half-range of the fitted cosine (amplitude) and its midline (MESOR)."),
    (r"^wavelet_ridge_amplitude_(mean|std|cv)$", "Amplitude", BIOLOGY,
     "Instantaneous amplitude along the ridge."),
    (r"^waveform_cycle_amp_cv$", "Amplitude", BIOLOGY,
     "Cycle-to-cycle variability of amplitude."),
    (r"^cycles_prominence_", "Amplitude", BIOLOGY,
     "How far detected peaks stand above their surroundings."),
    (r"^noise_signal_dynamic_range$", "Amplitude", BIOLOGY,
     "Span between the lowest and highest signal values."),

    # ── rhythm strength ─────────────────────────────────────────────────────
    (r"^cosinor_(r2|p_value|fit_snr)$", "Rhythm strength", BIOLOGY,
     "How well a single cosine explains the trace."),
    (r"^lomb_scargle_(peak_power|power_ratio|fap|mean_power|n_significant_peaks)$",
     "Rhythm strength", BIOLOGY, "Strength and significance of the periodogram peak."),
    (r"^harmonic_(fundamental_power_fraction|circadian_band_fraction)$",
     "Rhythm strength", BIOLOGY, "Share of spectral power in the main component."),
    (r"^cycles_(n_peaks|n_troughs|n_complete_cycles|peaks_per_day)$",
     "Rhythm strength", BIOLOGY, "How many cycles are actually present."),
    (r"^wavelet_ridge_(power_mean_normalized|overall_stability_score)$",
     "Rhythm strength", BIOLOGY, "Ridge strength and overall stability."),

    # ── waveform shape ──────────────────────────────────────────────────────
    (r"^waveform_", "Waveform shape", BIOLOGY,
     "Geometry of a cycle: rise, fall, width, asymmetry."),

    # ── harmonics ───────────────────────────────────────────────────────────
    (r"^harmonic_", "Harmonics", BIOLOGY,
     "Energy at fractions of the fundamental, and spectral complexity."),
    (r"^lomb_scargle_second_peak_ratio$", "Harmonics", BIOLOGY,
     "Strength of the second periodogram peak relative to the first."),
    (r"^noise_band_snr_", "Harmonics", BIOLOGY,
     "Signal-to-noise in the ultradian and infradian bands."),

    # ── trend & baseline ────────────────────────────────────────────────────
    (r"^baseline_", "Trend & baseline", BIOLOGY,
     "Non-rhythmic drift, depletion and stationarity of the baseline."),

    # ── noise & quality ─────────────────────────────────────────────────────
    (r"^noise_", "Noise & quality", BIOLOGY,
     "Residual noise and signal-to-noise after the rhythm is accounted for."),
    (r"^cosinor_residual_std$", "Noise & quality", BIOLOGY,
     "Spread of the residuals around the cosinor fit."),
    (r"^wavelet_ridge_ridge_", "Noise & quality", BIOLOGY,
     "How much of the recording carries a trackable ridge."),
    (r"_error$", "Noise & quality", RECORDING,
     "A package failed for this sample. Text, not a number — exclude before fitting."),
]

_COMPILED = [(re.compile(p), c, r, d) for p, c, r, d in _RULES]

CONCEPT_ORDER = list(CONCEPTS.keys())


def classify_feature(name: str) -> tuple[str, str, str]:
    """(concept, role, description) for one feature name."""
    for rx, concept, role, desc in _COMPILED:
        if rx.search(name):
            return concept, role, desc
    return "Other", BIOLOGY, "Unclassified — added to the extractor since the dictionary was written."


def describe_features(columns) -> pd.DataFrame:
    """
    The data dictionary: one row per feature.

    Ships with the export so a colleague — or a model six months from now — can
    tell what a column means and whether it describes biology or the instrument.
    """
    rows = []
    for name in columns:
        if name in ("sample_id", "Condition"):
            continue
        concept, role, desc = classify_feature(name)
        pkg = name.split("_")[0] if "_" in name else name
        for known in ("wavelet_ridge", "lomb_scargle"):
            if name.startswith(known):
                pkg = known
        rows.append({"feature": name, "concept": concept, "package": pkg,
                     "role": role, "description": desc})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_order"] = out["concept"].map(
        {c: i for i, c in enumerate(CONCEPT_ORDER)}).fillna(len(CONCEPT_ORDER))
    return out.sort_values(["_order", "feature"]).drop(columns="_order").reset_index(drop=True)


def features_in_concept(columns, concept: str) -> list[str]:
    return [c for c in columns if c not in ("sample_id", "Condition")
            and classify_feature(c)[0] == concept]


# ═══════════════════════════════════════════════════════════════════════════
#  Quality
# ═══════════════════════════════════════════════════════════════════════════

def numeric_features(features: pd.DataFrame) -> list[str]:
    """Feature columns that are actually numeric — the candidate model inputs."""
    skip = {"sample_id", "Condition"}
    return [c for c in features.columns
            if c not in skip and pd.api.types.is_numeric_dtype(features[c])]


def silence_extractor_warnings():
    """
    Context manager for calls into the feature extractor.

    Short traces legitimately produce empty peak lists, so `np.nanmean` on them
    emits "Mean of empty slice". That is expected behaviour reporting itself as a
    problem, and on a 96-well plate it buries the console. Suppressed at OUR call
    sites rather than by editing the extractor, which is the user's file.
    """
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("ignore", category=RuntimeWarning)
    return _Suppressed(ctx)


class _Suppressed:
    def __init__(self, ctx):
        self._ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False


def quality_report(features: pd.DataFrame) -> pd.DataFrame:
    """
    Per-feature usability. This is the check you would otherwise run by hand
    before training anything.

    `missing_pct` matters more than it looks: the extractor routes short
    recordings away from some packages, so in a cohort of mixed length whole
    blocks of columns are absent for a subset of samples — and that absence
    correlates with recording length, which is exactly the pattern that makes
    naive imputation leak information.
    """
    skip = {"sample_id", "Condition"}
    rows = []
    n = len(features)
    for col in features.columns:
        if col in skip:
            continue
        s = features[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        vals = pd.to_numeric(s, errors="coerce") if numeric else s
        n_missing = int(vals.isna().sum())
        n_unique = int(vals.nunique(dropna=True))
        concept, role, _ = classify_feature(col)
        rows.append({
            "feature": col,
            "concept": concept,
            "role": role,
            "numeric": numeric,
            "missing_pct": 100.0 * n_missing / n if n else 0.0,
            "n_unique": n_unique,
            "std": float(np.nanstd(vals.astype(float))) if numeric and n_unique else np.nan,
            "constant": n_unique <= 1,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["usable"] = (out["numeric"] & ~out["constant"] & (out["missing_pct"] < 100))
    return out.sort_values(["usable", "missing_pct", "feature"],
                           ascending=[True, False, True]).reset_index(drop=True)


def redundancy_clusters(features: pd.DataFrame, threshold: float = 0.95,
                        method: str = "spearman") -> pd.DataFrame:
    """
    Groups of features that move together above |r| >= threshold.

    Spearman by default: several of these features are bounded or heavily skewed
    (power fractions, coefficients of variation), and rank correlation does not
    care. Reported as clusters rather than a full matrix because the useful
    question is "how many independent things am I actually measuring".
    """
    cols = numeric_features(features)
    frame = features[cols].astype(float)
    frame = frame.loc[:, frame.nunique(dropna=True) > 1]
    if frame.shape[1] < 2 or len(frame) < 3:
        return pd.DataFrame(columns=["cluster", "feature", "concept", "n_in_cluster"])

    corr = frame.corr(method=method).abs()
    remaining = list(corr.columns)
    clusters, cid = [], 0
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        for other in list(remaining):
            if corr.loc[seed, other] >= threshold:
                group.append(other)
                remaining.remove(other)
        if len(group) > 1:
            cid += 1
            for f in group:
                clusters.append({"cluster": cid, "feature": f,
                                 "concept": classify_feature(f)[0],
                                 "n_in_cluster": len(group)})
    return pd.DataFrame(clusters)


# ═══════════════════════════════════════════════════════════════════════════
#  Differential comparison
# ═══════════════════════════════════════════════════════════════════════════

def _hedges_g(a, b):
    """Standardised mean difference, bias-corrected for small samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    d = (np.mean(a) - np.mean(b)) / pooled
    # Hedges' correction — matters at the group sizes typical here
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * j)


def _cliffs_delta(a, b):
    """Rank-based effect size: P(a > b) - P(a < b). Bounded to [-1, 1]."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    n_a, n_b = len(a), len(b)
    A = np.asarray(a, dtype=float)[:, None]
    B = np.asarray(b, dtype=float)[None, :]
    return float((np.sum(A > B) - np.sum(A < B)) / (n_a * n_b))


PARAMETRIC_MIN_N = 8


def compare_conditions(features: pd.DataFrame, group_col: str, group_a: str,
                       group_b: str, test: str = "auto", alpha: float = 0.05):
    """
    Every feature at once: effect size, test, and FDR across the feature set.

    Returns (results, meta). `results` has one row per feature with n per group,
    the effect size, the raw p-value and the BH q-value.

    Two deliberate choices:

    * The correction runs over ALL features tested, not per feature. Browsing 100
      boxplots and reporting the one that looks different is 100 comparisons
      whether or not anyone counted them; doing it here makes that explicit.
    * `test="auto"` picks rank-based methods when the smaller group has fewer
      than 8 members, because a t-test on n=3 rests on a normality assumption
      nobody can check at that size. The choice is reported, not hidden.
    """
    from scipy import stats

    if group_col not in features.columns:
        raise ValueError(f"No grouping column '{group_col}' in the feature table.")

    a_mask = features[group_col].astype(str) == str(group_a)
    b_mask = features[group_col].astype(str) == str(group_b)
    n_a, n_b = int(a_mask.sum()), int(b_mask.sum())
    if n_a < 2 or n_b < 2:
        raise ValueError(
            f"Need at least 2 samples per group — got {n_a} for '{group_a}' "
            f"and {n_b} for '{group_b}'."
        )

    if test == "auto":
        chosen = "rank" if min(n_a, n_b) < PARAMETRIC_MIN_N else "parametric"
        reason = (f"smaller group has n={min(n_a, n_b)} (< {PARAMETRIC_MIN_N}), "
                  f"so rank-based") if chosen == "rank" else \
                 (f"both groups have n>={PARAMETRIC_MIN_N}, so Welch's t-test")
    else:
        chosen, reason = test, "chosen manually"

    rows = []
    for col in numeric_features(features):
        a = pd.to_numeric(features.loc[a_mask, col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(features.loc[b_mask, col], errors="coerce").dropna().to_numpy()
        if len(a) < 2 or len(b) < 2 or (np.ptp(np.concatenate([a, b])) == 0):
            continue
        try:
            if chosen == "rank":
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                effect = _cliffs_delta(a, b)
            else:
                stat, p = stats.ttest_ind(a, b, equal_var=False)
                effect = _hedges_g(a, b)
                if not np.isfinite(effect):
                    # Zero pooled variance — e.g. a 0/1 flag feature that is
                    # constant within each group but differs between them. Hedges'
                    # g is undefined there; Cliff's delta is not, and a perfect
                    # separation is exactly what we want to report.
                    effect = _cliffs_delta(a, b)
        except Exception:
            continue
        concept, role, _ = classify_feature(col)
        rows.append({
            "feature": col, "concept": concept, "role": role,
            "n_a": len(a), "n_b": len(b),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "effect": effect, "p": float(p),
        })

    results = pd.DataFrame(rows)
    meta = {"test": chosen, "reason": reason, "n_a": n_a, "n_b": n_b,
            "group_a": group_a, "group_b": group_b,
            "effect_name": "Cliff's delta" if chosen == "rank" else "Hedges' g",
            "n_tested": len(results), "alpha": alpha}
    if results.empty:
        return results, meta

    results["q"] = _bh(results["p"].to_numpy())
    results["significant"] = results["q"].le(alpha).fillna(False)
    meta["n_significant"] = int(results["significant"].sum())
    return (results.sort_values("q").reset_index(drop=True), meta)


def _bh(pvalues):
    """
    Benjamini-Hochberg q-values, NaN-safe.

    A single NaN p-value used to make EVERY q NaN: np.argsort sorts NaN last, so
    after the reversal it comes first and np.minimum.accumulate carries it across
    the whole array. Silent, and it turns a table of real findings into a table
    of blanks. NaNs are now held out of the correction and returned as NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out

    p_ok = p[ok]
    order = np.argsort(p_ok)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    q = np.minimum(1.0, p_ok * m / ranks)
    q = np.minimum.accumulate(q[order][::-1])[::-1]
    q_ok = np.empty(m)
    q_ok[order] = q
    out[ok] = q_ok
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Cohort context  &  QC
# ═══════════════════════════════════════════════════════════════════════════

def cohort_percentiles(features: pd.DataFrame, sample_id: str,
                       columns=None) -> pd.DataFrame:
    """
    Where one sample sits in the cohort, per feature.

    A raw feature value is not interpretable on its own — nobody knows whether
    `baseline_depletion_index = 0.42` is normal. A percentile against the other
    samples in the same experiment is interpretable immediately.
    """
    if "sample_id" not in features.columns:
        raise ValueError("Feature table has no sample_id column.")
    row = features[features["sample_id"].astype(str) == str(sample_id)]
    if row.empty:
        raise ValueError(f"Sample '{sample_id}' is not in the feature table.")

    cols = columns or numeric_features(features)
    out = []
    for col in cols:
        values = pd.to_numeric(features[col], errors="coerce")
        v = pd.to_numeric(row[col], errors="coerce").iloc[0]
        finite = values.dropna()
        if not np.isfinite(v) or len(finite) < 3:
            continue
        # Midrank percentile: ties count half. With a strict `<`, a feature that
        # is identical across the cohort (recording duration, say) scored 0 and
        # then dominated the "most extreme" ranking — the opposite of the truth.
        n_below = int((finite < v).sum())
        n_equal = int((finite == v).sum())
        pct = 100.0 * (n_below + 0.5 * n_equal) / len(finite)
        concept, role, _ = classify_feature(col)
        out.append({"feature": col, "concept": concept, "role": role,
                    "value": float(v), "percentile": float(pct),
                    "cohort_median": float(finite.median()),
                    "n_cohort": int(len(finite))})
    frame = pd.DataFrame(out)
    if frame.empty:
        return frame
    # Most extreme first — that is what someone opening this view wants to see.
    frame["extremity"] = (frame["percentile"] - 50).abs()
    return frame.sort_values("extremity", ascending=False).reset_index(drop=True)


# Cohort-relative wherever possible: an absolute noise threshold that suits a
# luminometer is meaningless for a temperature logger, but "this well is noisier
# than 95% of the plate" travels.
QC_RULES = {
    "Too few cycles": {
        "feature": "cycles_n_complete_cycles", "mode": "below", "value": 2.0,
        "why": "Fewer than 2 complete cycles — period and amplitude are not estimable.",
    },
    "Poor rhythm fit": {
        "feature": "cosinor_r2", "mode": "below", "value": 0.05,
        "why": "A cosine explains almost none of the variance.",
    },
    "Flat trace": {
        "feature": "noise_signal_dynamic_range", "mode": "percentile_below", "value": 2.0,
        "why": "Dynamic range in the bottom 2% of the cohort — possibly a dead well.",
    },
    "High noise": {
        "feature": "noise_residual_std", "mode": "percentile_above", "value": 95.0,
        "why": "Residual noise above the 95th percentile of the cohort.",
    },
    "Strong drift": {
        "feature": "baseline_trend_r2", "mode": "above", "value": 0.5,
        "why": "More than half the variance is a monotonic trend, not a rhythm.",
    },
}


def qc_flags(features: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    """
    Per-sample QC verdict with reasons.

    Returns one row per sample: the flags it triggered and a short reason string.
    Rules whose feature is missing from the table are skipped rather than
    silently passing everything.
    """
    rules = rules or QC_RULES
    if "sample_id" not in features.columns:
        raise ValueError("Feature table has no sample_id column.")

    flags = {s: [] for s in features["sample_id"]}
    reasons = {s: [] for s in features["sample_id"]}
    applied = []

    for label, rule in rules.items():
        col = rule["feature"]
        if col not in features.columns:
            continue
        values = pd.to_numeric(features[col], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            continue
        applied.append(label)

        mode, v = rule["mode"], rule["value"]
        if mode == "below":
            hit = values < v
        elif mode == "above":
            hit = values > v
        elif mode == "percentile_below":
            hit = values < np.nanpercentile(finite, v)
        elif mode == "percentile_above":
            hit = values > np.nanpercentile(finite, v)
        else:
            continue

        for sample, is_hit in zip(features["sample_id"], hit.fillna(False)):
            if is_hit:
                flags[sample].append(label)
                reasons[sample].append(rule["why"])

    out = pd.DataFrame({
        "sample_id": features["sample_id"],
        "n_flags": [len(flags[s]) for s in features["sample_id"]],
        "flags": ["; ".join(flags[s]) if flags[s] else "" for s in features["sample_id"]],
        "reasons": [" ".join(reasons[s]) for s in features["sample_id"]],
    })
    out["verdict"] = np.where(out["n_flags"] == 0, "pass",
                              np.where(out["n_flags"] == 1, "warn", "fail"))
    out.attrs["rules_applied"] = applied
    return out.sort_values(["n_flags", "sample_id"], ascending=[False, True]).reset_index(drop=True)
