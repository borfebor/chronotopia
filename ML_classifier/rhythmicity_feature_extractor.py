"""
rhythmicity_feature_extractor.py
=================================
Fast, wavelet-free feature extractor for circadian rhythmicity classification.

Design principles:
  - No PyBoat / wavelet ridge dependency: features must be computable even
    when no ridge exists (i.e. for arrhythmic signals).
  - All features are continuous scores, never pass/fail booleans from a
    hand-tuned detector — the model learns its own thresholds.
  - Works for any recording duration (2 days to 2+ weeks) and any sampling
    rate that gives at least 6 points per circadian window.
  - Target runtime: < 50 ms per signal on a modern laptop.

Feature groups (18 features total):
  1. Spectral  (4)  — Lomb-Scargle + FFT power in the circadian band
  2. ACF       (4)  — autocorrelation at biologically meaningful lags
  3. Sinfit    (3)  — best sinusoidal fit quality over a period sweep
  4. Morphology(4)  — waveform geometry, no spectral decomposition
  5. Regularity(3)  — permutation entropy + zero-crossing rate + CV

Dependencies: numpy, scipy, pandas  (all already in Chronotopia's stack)

Usage
-----
    from rhythmicity_feature_extractor import RhythmicityFeatureExtractor

    extractor = RhythmicityFeatureExtractor()

    # Single signal → dict
    features = extractor.extract(signal_array, time_array)

    # Batch → DataFrame (one row per signal)
    feat_df = extractor.extract_batch(df, t_col="time_hours", data_cols=cols)
"""

import warnings
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# ACF backend (statsmodels preferred, pure-numpy fallback)
# ---------------------------------------------------------------------------

try:
    from statsmodels.tsa.stattools import acf as _sm_acf

    def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
        return _sm_acf(x, nlags=nlags, fft=True, missing="drop")

except ImportError:
    def _acf(x: np.ndarray, nlags: int) -> np.ndarray:           # noqa: E302
        x = x - np.mean(x)
        N = len(x)
        nlags = min(nlags, N - 1)
        full = np.correlate(x, x, mode="full")
        acf_v = full[N - 1: N + nlags] / (np.var(x) * N + 1e-12)
        return acf_v


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, fill: float = 0.0) -> float:
    return float(a / b) if abs(b) > 1e-12 else fill


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    slope, intercept, *_ = linregress(t, x)
    return x - (slope * t + intercept)


def _sinusoid(t, A, T, phi, C):
    return A * np.sin(2.0 * np.pi / T * t + phi) + C


# ---------------------------------------------------------------------------
# Feature group 1: Spectral
# ---------------------------------------------------------------------------

def _spectral_features(
    x: np.ndarray,
    t: np.ndarray,
    period_min: float,
    period_max: float,
) -> dict:
    """
    Lomb-Scargle + FFT features in the circadian window.

    Lomb-Scargle is preferred over plain FFT for unevenly sampled data,
    but we include both because they capture slightly different aspects
    of spectral concentration.
    """
    freq_min = 1.0 / period_max
    freq_max = 1.0 / period_min
    n_freqs = 300

    freqs_ls = np.linspace(freq_min, freq_max, n_freqs)
    angular  = 2.0 * np.pi * freqs_ls
    x_c      = x - np.mean(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pgram = scipy_signal.lombscargle(t, x_c, angular, normalize=True)

    peak_idx    = int(np.argmax(pgram))
    ls_peak     = float(pgram[peak_idx])
    ls_mean     = float(np.mean(pgram))
    ls_ratio    = _safe_div(ls_peak, ls_mean, fill=1.0)
    ls_period   = float(1.0 / freqs_ls[peak_idx])

    # FFT (on uniformly spaced grid — use median dt)
    dt = float(np.median(np.diff(t)))
    fft_power = np.abs(np.fft.rfft(x_c)) ** 2
    fft_freqs = np.fft.rfftfreq(len(x), d=dt)

    pos = fft_freqs > 0
    fft_power = fft_power[pos]
    fft_freqs = fft_freqs[pos]

    circ_mask = (fft_freqs >= freq_min) & (fft_freqs <= freq_max)
    total_pow = float(np.sum(fft_power)) + 1e-12

    if circ_mask.any():
        circ_peak     = float(np.max(fft_power[circ_mask]))
        circ_fraction = float(np.sum(fft_power[circ_mask])) / total_pow
        # Is the circadian peak the global dominant peak?
        global_peak   = float(np.max(fft_power))
        is_dominant   = float((circ_peak / (global_peak + 1e-12)) > 0.85)
    else:
        circ_fraction = 0.0
        is_dominant   = 0.0

    return {
        "ls_peak_power":      ls_peak,       # Lomb-Scargle peak power (normalised)
        "ls_power_ratio":     ls_ratio,       # peak / mean background (SNR proxy)
        "fft_circadian_frac": circ_fraction,  # fraction of FFT power in circ. band
        "fft_is_dominant":    is_dominant,    # 1 if circadian = global peak
    }


# ---------------------------------------------------------------------------
# Feature group 2: Autocorrelation
# ---------------------------------------------------------------------------

def _acf_features(
    x: np.ndarray,
    t: np.ndarray,
    period_min: float,
    period_max: float,
) -> dict:
    """
    ACF at biologically fixed lags + peak in the circadian window.

    acf_at_24h is the single most informative feature for circadian data:
    a genuine 24 h rhythm will have high positive autocorrelation at lag 24 h,
    while noise or non-circadian trends will not.
    """
    dt   = float(np.median(np.diff(t)))
    N    = len(x)
    nlags = min(N - 2, int(np.ceil(period_max / dt)) + 2)

    try:
        acf_vals = _acf(x, nlags)
    except Exception:
        acf_vals = np.full(nlags + 1, np.nan)

    def _at(period_h: float) -> float:
        lag = int(round(period_h / dt))
        return float(acf_vals[lag]) if 0 < lag < len(acf_vals) else np.nan

    acf_24 = _at(24.0)
    acf_12 = _at(12.0)   # sub-harmonic check

    # Peak ACF anywhere in the circadian window
    lag_lo = max(1, int(round(period_min / dt)))
    lag_hi = min(len(acf_vals) - 1, int(round(period_max / dt)))
    window = acf_vals[lag_lo: lag_hi + 1]
    acf_peak = float(np.nanmax(window)) if len(window) > 0 else np.nan

    # ACF at half-period lag (detects whether 12 h harmonic dominates)
    # If acf_12 > acf_24 the rhythm is more likely ultradian
    ultradian_bias = float(acf_12 - acf_24) if not (np.isnan(acf_12) or np.isnan(acf_24)) else 0.0

    return {
        "acf_at_24h":      acf_24,         # core circadian autocorrelation
        "acf_peak_circ":   acf_peak,       # best ACF in 18–30 h window
        "acf_at_12h":      acf_12,         # sub-harmonic evidence
        "acf_ultradian":   ultradian_bias, # positive = 12 h dominates over 24 h
    }


# ---------------------------------------------------------------------------
# Feature group 3: Sinusoidal fit quality
# ---------------------------------------------------------------------------

def _sinfit_features(
    x: np.ndarray,
    t: np.ndarray,
    period_min: float,
    period_max: float,
    n_periods: int = 40,
) -> dict:
    """
    Sweep candidate periods and return the best sinusoidal fit quality.

    Uses a closed-form linear projection (A·sin + B·cos + C) instead of
    iterative curve_fit — ~100× faster while giving identical R² values.
    For a fixed period T the model is linear in its parameters:
        y = A·sin(2π/T · t) + B·cos(2π/T · t) + C
    This is solved exactly via least squares in one step.
    """
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))

    # Flat signal guard
    if ss_tot < 1e-12:
        return {"sinfit_r2": 0.0, "sinfit_snr": 0.0, "sinfit_period": np.nan}

    best_r2     = -np.inf
    best_period = np.nan

    for T in np.linspace(period_min, period_max, n_periods):
        omega = 2.0 * np.pi / T
        # Design matrix: [sin, cos, 1]
        A = np.column_stack([
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones(len(t)),
        ])
        # Solve in one shot — O(N) effectively
        coeffs, res, *_ = np.linalg.lstsq(A, x, rcond=None)
        y_hat  = A @ coeffs
        ss_res = float(np.sum((x - y_hat) ** 2))
        r2     = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2     = r2
            best_period = float(T)

    best_r2    = float(np.clip(best_r2, -1.0, 1.0))
    sinfit_snr = _safe_div(max(best_r2, 0.0), max(1.0 - best_r2, 1e-6))

    return {
        "sinfit_r2":     best_r2,
        "sinfit_snr":    sinfit_snr,
        "sinfit_period": best_period,
    }


# ---------------------------------------------------------------------------
# Feature group 4: Waveform morphology
# ---------------------------------------------------------------------------

def _morphology_features(x: np.ndarray, t: np.ndarray) -> dict:
    """
    Peak/trough geometry — entirely in the time domain, no spectral step.

    peaks_per_day is the most interpretable single feature: a circadian signal
    should have ~1 peak per 24 h. Too many peaks = noise; too few = flat/trending.
    """
    x_dt = _detrend_linear(x)
    dt   = float(np.median(np.diff(t)))

    peak_idx, _   = scipy_signal.find_peaks(x_dt)
    trough_idx, _ = scipy_signal.find_peaks(-x_dt)

    duration_days = (t[-1] - t[0]) / 24.0
    peaks_per_day = _safe_div(len(peak_idx), duration_days, fill=0.0)

    # Relative amplitude: how large is the oscillation relative to its mean?
    x_mean = float(np.mean(np.abs(x)))
    x_range = float(np.max(x) - np.min(x))
    relative_amplitude = _safe_div(x_range, x_mean + 1e-9)

    # Coefficient of variation
    cv = _safe_div(float(np.std(x, ddof=1)), float(np.abs(np.mean(x))) + 1e-9)

    # Peak-to-peak interval regularity (std of inter-peak intervals)
    if len(peak_idx) >= 3:
        ipi      = np.diff(peak_idx) * dt          # inter-peak intervals in hours
        ipi_cv   = _safe_div(float(np.std(ipi)), float(np.mean(ipi)) + 1e-9)
    else:
        ipi_cv   = np.nan

    return {
        "morph_peaks_per_day":  peaks_per_day,      # expected ≈ 1 for circadian
        "morph_rel_amplitude":  relative_amplitude, # oscillation depth
        "morph_cv":             cv,                 # signal coefficient of variation
        "morph_ipi_cv":         ipi_cv,             # inter-peak interval regularity
    }


# ---------------------------------------------------------------------------
# Feature group 5: Regularity / complexity
# ---------------------------------------------------------------------------

def _permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalised permutation entropy in [0, 1]. Low = periodic, high = noisy."""
    import math
    N = len(x)
    n_pat = N - delay * (order - 1)
    if n_pat < 1:
        return np.nan
    patterns = np.array(
        [np.argsort(x[i: i + order * delay: delay]) for i in range(n_pat)]
    )
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / len(patterns)
    pe    = -np.sum(probs * np.log2(probs + 1e-12))
    max_pe = np.log2(math.factorial(order))
    return float(pe / max_pe) if max_pe > 0 else np.nan


def _regularity_features(x: np.ndarray, t: np.ndarray) -> dict:
    """
    Three complementary regularity indices, all O(N) or O(N log N).
    """
    # Zero-crossing rate on mean-centred signal
    x_c = x - np.mean(x)
    zcr = float(np.sum(np.diff(np.sign(x_c)) != 0)) / max(len(x_c) - 1, 1)

    # Permutation entropy
    pe = _permutation_entropy(x, order=3, delay=1)

    # Trend-to-noise ratio: fraction of variance explained by a linear trend
    # High values indicate a drifting baseline rather than an oscillation
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-12)
    slope, intercept, r, *_ = linregress(t_norm, x)
    trend_r2 = float(r ** 2)

    return {
        "reg_perm_entropy": pe,        # low for periodic, high for random
        "reg_zcr":          zcr,       # zero-crossing rate
        "reg_trend_r2":     trend_r2,  # high if signal is mostly a linear trend
    }


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

# Canonical ordered list of the 18 features — used for model alignment
FEATURE_NAMES = [
    # Spectral
    "ls_peak_power",
    "ls_power_ratio",
    "fft_circadian_frac",
    "fft_is_dominant",
    # ACF
    "acf_at_24h",
    "acf_peak_circ",
    "acf_at_12h",
    "acf_ultradian",
    # Sinfit
    "sinfit_r2",
    "sinfit_snr",
    "sinfit_period",
    # Morphology
    "morph_peaks_per_day",
    "morph_rel_amplitude",
    "morph_cv",
    "morph_ipi_cv",
    # Regularity
    "reg_perm_entropy",
    "reg_zcr",
    "reg_trend_r2",
]


class RhythmicityFeatureExtractor:
    """
    Extracts 18 fast, wavelet-free features for rhythmicity classification.

    Parameters
    ----------
    period_min : float  Minimum circadian period to consider (hours). Default 18.
    period_max : float  Maximum circadian period to consider (hours). Default 30.
    """

    def __init__(self, period_min: float = 18.0, period_max: float = 30.0):
        self.period_min = period_min
        self.period_max = period_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, x: np.ndarray, t: np.ndarray) -> dict:
        """
        Extract all features from a single signal.

        Parameters
        ----------
        x : array-like  Signal values.
        t : array-like  Time axis in hours.

        Returns
        -------
        dict  {feature_name: float}  — always exactly FEATURE_NAMES keys.
        """
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)

        valid = np.isfinite(x) & np.isfinite(t)
        x, t  = x[valid], t[valid]

        if len(x) < 6:
            raise ValueError(f"Signal too short: {len(x)} valid points (need ≥ 6).")

        feats = {}
        feats.update(_spectral_features(x, t, self.period_min, self.period_max))
        feats.update(_acf_features(x, t, self.period_min, self.period_max))
        feats.update(_sinfit_features(x, t, self.period_min, self.period_max))
        feats.update(_morphology_features(x, t))
        feats.update(_regularity_features(x, t))

        # Guarantee canonical ordering and fill any unexpected NaNs
        return {k: feats.get(k, np.nan) for k in FEATURE_NAMES}

    def extract_batch(
        self,
        df: pd.DataFrame,
        t_col: str,
        data_cols: list,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Extract features for all signals in a DataFrame.

        Returns
        -------
        pd.DataFrame  One row per signal, columns = ['sample_id'] + FEATURE_NAMES.
        """
        t       = df[t_col].values
        records = []
        errors  = []

        for col in data_cols:
            try:
                feats = self.extract(df[col].values, t)
                feats["sample_id"] = col
                records.append(feats)
            except Exception as e:
                errors.append((col, str(e)))
                if verbose:
                    print(f"  [WARN] {col}: {e}")

        if verbose and errors:
            print(f"\n{len(errors)}/{len(data_cols)} signals failed extraction.")

        out = pd.DataFrame(records)
        cols = ["sample_id"] + FEATURE_NAMES
        return out[[c for c in cols if c in out.columns]]

    @property
    def feature_names(self) -> list:
        return list(FEATURE_NAMES)
