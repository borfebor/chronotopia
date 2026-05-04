"""
chronotopia_feature_extractor.py
=================================
Unified feature extractor for Chronotopia.

Replaces and consolidates:
  - WaveletRidgeFeatureExtractor  (Wavelet_Features.py)
  - HierarchicalRhythmicityDetector.extract_all_metrics()  (hierarchical_detector.py)
  - ShortSeriesFeatureExtractor   (short_series_feature_extractor.py)

Architecture
------------
Features are organised into eight coherent *packages*. Each package:
  - extracts a logically related set of features together
  - exposes a paired `plot_<package>(ax, ...)` method that overlays the
    computed results on the raw signal for visual validation
  - is callable independently or as part of the full extraction

Routing is automatic based on recording properties:
  - short  : duration ≤ 48 h  OR  n_points < 20
             → cosinor, waveform, cycles, baseline, harmonic, noise, lomb_scargle
  - long   : duration > 48 h  AND  n_points ≥ 20
             → all of the above  +  wavelet_ridge

Packages
--------
  1. cosinor        Classical least-squares cosinor fit. MESOR, acrophase,
                    amplitude, R², p-value, residuals.
  2. waveform       Per-peak rise/fall time, FWHM, asymmetry index,
                    cycle-to-cycle amplitude and period variance.
  3. cycles         Event-based: peak/trough detection, inter-cycle intervals,
                    complete cycle count, peak prominence statistics.
  4. baseline       Rolling mean drift, substrate-depletion proxy, ADF-style
                    non-stationarity score, linear/quadratic trend metrics.
  5. harmonic       FFT-based: fundamental power, harmonic power ratios
                    (12 h, 8 h), secondary peak ratio, spectral complexity.
  6. noise          Residual noise after cosinor fit, per-band SNR profile,
                    noise floor, residual autocorrelation structure.
  7. wavelet_ridge  Instantaneous period/amplitude/phase from PyBoat ridge.
                    Consolidated from WaveletRidgeFeatureExtractor +
                    HierarchicalRhythmicityDetector. Long series only.
  8. lomb_scargle   Lomb-Scargle periodogram metrics. Works on all lengths
                    and handles irregular sampling.

Usage
-----
    from chronotopia_feature_extractor import ChronotopiaFeatureExtractor

    # Full extraction (all applicable packages)
    ext = ChronotopiaFeatureExtractor(signal, time, period_range=(18, 30))
    features = ext.extract()           # → flat dict
    df_row   = ext.to_series()         # → pandas Series

    # Selective extraction
    features = ext.extract(packages=["cosinor", "cycles", "harmonic"])

    # Batch from a DataFrame
    feat_df = ChronotopiaFeatureExtractor.extract_batch(
        df, t_col="time_hours", data_cols=["s1", "s2", "s3"]
    )

    # Visualisation — overlay a package on an existing matplotlib axis
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time, signal)
    ext.plot_cosinor(ax)
    ext.plot_cycles(ax)
    ext.plot_waveform(ax)

Dependencies
------------
    numpy, scipy, pandas, matplotlib
    pyboat          (wavelet_ridge package; gracefully skipped if absent)
    statsmodels     (ADF test in baseline package; gracefully skipped)

Author: Chronotopia / borfebor
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit
from scipy.stats import linregress, circmean, circstd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    from pyboat import WAnalyzer
    _HAS_PYBOAT = True
except ImportError:
    _HAS_PYBOAT = False

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from statsmodels.tsa.stattools import acf as _sm_acf
    def _acf(x, nlags):
        return _sm_acf(x, nlags=nlags, fft=False, missing="drop")
except ImportError:
    def _acf(x, nlags):
        x = np.asarray(x, dtype=float) - np.mean(x)
        N = len(x)
        nlags = min(nlags, N - 1)
        full = np.correlate(x, x, mode="full")
        acov = full[N - 1: N + nlags]
        v = np.var(x) * N
        return acov / v if v > 0 else acov * 0


# ── small helpers ────────────────────────────────────────────────────────────

def _safe(a, b, fill=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) > 1e-12, a / b, fill)
    return float(out) if out.ndim == 0 else out


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    s, i, *_ = linregress(t, x)
    return x - (s * t + i)


def _cosinor_model(t, A, phi, mesor, T):
    return A * np.cos(2 * np.pi / T * t + phi) + mesor


# ── package colour palette (for plots) ───────────────────────────────────────
_PKG_COLORS = {
    "cosinor":       "#1D9E75",
    "waveform":      "#E85D24",
    "cycles":        "#185FA5",
    "baseline":      "#BA7517",
    "harmonic":      "#993556",
    "noise":         "#5F5E5A",
    "wavelet_ridge": "#534AB7",
    "lomb_scargle":  "#0F6E56",
}

ALL_PACKAGES = list(_PKG_COLORS.keys())


# ════════════════════════════════════════════════════════════════════════════
#  Main class
# ════════════════════════════════════════════════════════════════════════════

class ChronotopiaFeatureExtractor:
    """
    Unified feature extractor with eight coherent, plotable packages.

    Parameters
    ----------
    signal : array-like
        Time series values.
    time : array-like
        Time axis in hours (need not start at 0, may be irregular).
    period_range : tuple (min_h, max_h)
        Expected circadian period window. Default (18, 30).
    cosinor_period : float or None
        Fixed period for cosinor fit. If None, the best-fit period from the
        Lomb-Scargle periodogram is used automatically.
    """

    def __init__(
        self,
        signal,
        time,
        period_range: tuple = (18, 30),
        cosinor_period: float | None = None,
    ):
        self.signal_raw = np.asarray(signal, dtype=float)
        self.time = np.asarray(time, dtype=float)
        self.period_range = period_range
        self._cosinor_period_override = cosinor_period

        # Drop NaN
        valid = np.isfinite(self.signal_raw) & np.isfinite(self.time)
        self.signal = self.signal_raw[valid]
        self.time   = self.time[valid]

        if len(self.signal) < 4:
            raise ValueError(f"Signal too short after NaN removal: {len(self.signal)} points.")

        self.dt       = float(np.median(np.diff(self.time)))
        self.duration = float(self.time[-1] - self.time[0])
        self.n        = len(self.signal)

        # Routing flag
        self.is_short = (self.duration <= 48.0) or (self.n < 20)

        # Cache for computed results (populated lazily by each package)
        self._cache: dict = {}

    # ── routing ─────────────────────────────────────────────────────────────

    def available_packages(self) -> list[str]:
        """Return the packages applicable to this recording."""
        pkgs = ["cosinor", "waveform", "cycles", "baseline",
                "harmonic", "noise", "lomb_scargle"]
        if not self.is_short and _HAS_PYBOAT:
            pkgs.append("wavelet_ridge")
        return pkgs

    # ── main extraction API ──────────────────────────────────────────────────

    def extract(self, packages: list[str] | None = None) -> dict:
        """
        Extract features from all (or selected) packages.

        Returns a flat dict of feature_name → value.
        Package prefix is used as namespace, e.g. "cosinor_r2".
        """
        if packages is None:
            packages = self.available_packages()

        result = {
            # Recording metadata (always present)
            "meta_duration_h":   self.duration,
            "meta_n_points":     self.n,
            "meta_dt_h":         self.dt,
            "meta_is_short":     int(self.is_short),
        }

        dispatch = {
            "cosinor":       self._pkg_cosinor,
            "waveform":      self._pkg_waveform,
            "cycles":        self._pkg_cycles,
            "baseline":      self._pkg_baseline,
            "harmonic":      self._pkg_harmonic,
            "noise":         self._pkg_noise,
            "lomb_scargle":  self._pkg_lomb_scargle,
            "wavelet_ridge": self._pkg_wavelet_ridge,
        }

        for pkg in packages:
            if pkg not in dispatch:
                warnings.warn(f"Unknown package '{pkg}' — skipped.")
                continue
            try:
                feats = dispatch[pkg]()
                result.update({f"{pkg}_{k}": v for k, v in feats.items()})
            except Exception as exc:
                warnings.warn(f"Package '{pkg}' failed: {exc}")
                result[f"{pkg}_error"] = str(exc)

        return result

    def to_series(self, packages: list[str] | None = None) -> pd.Series:
        """Return features as a pandas Series."""
        return pd.Series(self.extract(packages))

    # ── batch helper ─────────────────────────────────────────────────────────

    @staticmethod
    def extract_batch(
        df: pd.DataFrame,
        t_col: str,
        data_cols: list[str],
        packages: list[str] | None = None,
        period_range: tuple = (18, 30),
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Extract features for every column in data_cols.

        Returns a DataFrame with one row per signal.
        """
        t = df[t_col].values
        records, errors = [], []

        for col in data_cols:
            try:
                ext = ChronotopiaFeatureExtractor(
                    df[col].values, t, period_range=period_range
                )
                row = ext.extract(packages)
                row["sample_id"] = col
                records.append(row)
            except Exception as e:
                errors.append((col, str(e)))
                if verbose:
                    print(f"  [WARN] {col}: {e}")

        if verbose and errors:
            print(f"\n{len(errors)}/{len(data_cols)} signals failed.")

        out = pd.DataFrame(records)
        cols = ["sample_id"] + [c for c in out.columns if c != "sample_id"]
        return out[cols]

    # ════════════════════════════════════════════════════════════════════════
    #  PACKAGE 1 — COSINOR
    # ════════════════════════════════════════════════════════════════════════

    def _pkg_cosinor(self) -> dict:
        """
        Classical cosinor analysis.

        Features
        --------
        period          Period used for the fit (hours)
        amplitude       Half the peak-to-trough oscillation (A in the model)
        mesor           Midline estimating statistic of rhythm
        acrophase_rad   Phase at peak (radians, in [0, 2π])
        acrophase_h     Acrophase expressed as clock time (hours into period)
        r2              Proportion of variance explained by the cosinor model
        p_value         F-test zero-amplitude test p-value
        residual_std    Std dev of fit residuals
        fit_snr         Amplitude / residual_std
        """
        if "cosinor" in self._cache:
            return self._cache["cosinor"]

        T = self._cosinor_period_override
        if T is None:
            ls = self._pkg_lomb_scargle()
            T  = ls.get("peak_period_h", 24.0)
            if np.isnan(T):
                T = 24.0

        t, x = self.time, self.signal
        A0    = (np.max(x) - np.min(x)) / 2
        C0    = np.mean(x)

        try:
            popt, pcov = curve_fit(
                lambda t, A, phi, C: _cosinor_model(t, A, phi, C, T),
                t, x,
                p0=[A0, 0.0, C0],
                bounds=([0, -np.pi, -np.inf], [np.inf, np.pi, np.inf]),
                maxfev=4000,
            )
            A, phi, mesor = popt
            A = abs(A)

            fitted  = _cosinor_model(t, A, phi, mesor, T)
            resid   = x - fitted
            ss_res  = np.sum(resid ** 2)
            ss_tot  = np.sum((x - np.mean(x)) ** 2)
            r2      = float(np.clip(1 - _safe(ss_res, ss_tot), -1, 1))

            # Zero-amplitude F-test
            n, k = len(x), 2
            f_stat = ((ss_tot - ss_res) / k) / (ss_res / max(n - k - 1, 1))
            from scipy.stats import f as f_dist
            p_value = float(1 - f_dist.cdf(f_stat, k, max(n - k - 1, 1)))

            # Acrophase: map phi to [0, T) clock hours
            acro_rad = float(phi % (2 * np.pi))
            acro_h   = float(acro_rad / (2 * np.pi) * T)

            resid_std = float(np.std(resid))

            out = {
                "period":        float(T),
                "amplitude":     float(A),
                "mesor":         float(mesor),
                "acrophase_rad": acro_rad,
                "acrophase_h":   acro_h,
                "r2":            r2,
                "p_value":       p_value,
                "residual_std":  resid_std,
                "fit_snr":       _safe(A, resid_std),
            }
        except Exception as e:
            out = {
                "period": float(T), "amplitude": np.nan, "mesor": float(np.mean(x)),
                "acrophase_rad": np.nan, "acrophase_h": np.nan,
                "r2": np.nan, "p_value": np.nan,
                "residual_std": np.nan, "fit_snr": np.nan,
            }

        self._cache["cosinor"] = out
        return out

    def plot_cosinor(self, ax: plt.Axes, color: str | None = None, label: bool = True):
        """
        Overlay the cosinor fit on ax.

        Draws:
          - dashed fitted curve
          - horizontal MESOR line
          - vertical acrophase marker
          - shaded amplitude envelope
        """
        c   = color or _PKG_COLORS["cosinor"]
        f   = self._pkg_cosinor()
        if np.isnan(f["amplitude"]):
            return

        T    = f["period"]
        t    = self.time
        fitted = _cosinor_model(t, f["amplitude"], f["acrophase_rad"], f["mesor"], T)

        ax.plot(t, fitted, "--", color=c, lw=1.8, alpha=0.85,
                label=f"Cosinor fit  T={T:.1f} h  R²={f['r2']:.2f}" if label else None)
        ax.axhline(f["mesor"], color=c, lw=0.8, ls=":", alpha=0.6,
                   label=f"MESOR={f['mesor']:.2f}" if label else None)
        ax.fill_between(t, fitted - f["residual_std"],
                        fitted + f["residual_std"], color=c, alpha=0.10)
        title = f"Cosinor fit  T={T:.1f} h  R²={f['r2']:.2f}"
        ax.set_title(title, loc='left')
        # Acrophase tick
        acro_t = self.time[0] + f["acrophase_h"]
        while acro_t < t[0]:
            acro_t += T
        while acro_t <= t[-1]:
            ax.axvline(acro_t, color=c, lw=0.8, alpha=0.45, ls=(0, (2, 4)))
            acro_t += T

    # ════════════════════════════════════════════════════════════════════════
    #  PACKAGE 2 — WAVEFORM
    # ════════════════════════════════════════════════════════════════════════

    def _pkg_waveform(self) -> dict:
        """
        Per-cycle waveform shape features.

        Features
        --------
        rise_time_mean / _std      Mean/std of trough→peak duration (h)
        fall_time_mean / _std      Mean/std of peak→trough duration (h)
        asymmetry_index            (fall − rise) / (fall + rise); >0 = slow decay
        fwhm_mean / _std           Full-width at half-prominence, mean/std (h)
        fwhm_relative              fwhm_mean / period (dimensionless)
        cycle_amp_cv               CV of per-cycle peak-to-trough amplitudes
        cycle_period_cv            CV of consecutive peak-to-peak intervals
        waveform_r2_vs_sinusoid    R² of best sinusoidal fit (from cosinor pkg)
        """
        if "waveform" in self._cache:
            return self._cache["waveform"]

        x   = _detrend_linear(self.signal)
        t   = self.time
        dt  = self.dt

        # Minimum distance between peaks: half the shortest expected period
        min_dist = max(1, int(self.period_range[0] / dt / 2))
        #prom_value = (np.max(x) - np.min(x)) / 10
        prom_value = np.mean(np.abs(scipy_signal.hilbert(x))) * 1.5
 
        peaks,   pp = scipy_signal.find_peaks(x,  distance=min_dist, prominence=prom_value, width=min_dist/10)
        troughs, tp = scipy_signal.find_peaks(-x, distance=min_dist, prominence=prom_value, width=min_dist/10)
        
        rise_times, fall_times, fwhms, cycle_amps, cycle_periods = [], [], [], [], []

        # Match each peak to its nearest preceding trough and following trough
        for pk in peaks:
            # --- preceding trough ---
            pre = troughs[troughs < pk]
            suf = troughs[troughs > pk]
            if len(pre) == 0 or len(suf) == 0:
                continue
            tr_pre = pre[-1]
            tr_suf = suf[0]

            rise = float((t[pk] - t[tr_pre]))
            fall = float((t[tr_suf] - t[pk]))
            rise_times.append(rise)
            fall_times.append(fall)
            
            amp = float(x[pk] - 0.5 * (x[tr_pre] + x[tr_suf]))
            cycle_amps.append(amp)

            # FWHM: half-prominence width from scipy
            widths = scipy_signal.peak_widths(x, [pk], rel_height=0.5)
            fwhm_pts = float(widths[0][0])
            fwhms.append(fwhm_pts * dt)

        # Peak-to-peak intervals → cycle period CV
        if len(peaks) >= 2:
            intervals = np.diff(t[peaks])
            cycle_periods = list(intervals)

        def _stats(arr):
            if len(arr) < 1:
                return np.nan, np.nan
            a = np.array(arr)
            return float(np.mean(a)), float(np.std(a))

        r_m, r_s = _stats(rise_times)
        f_m, f_s = _stats(fall_times)
        fw_m, fw_s = _stats(fwhms)
        ca_m, ca_s = _stats(cycle_amps)
        cp_m, cp_s = _stats(cycle_periods)

        if r_m + f_m > 0:
            asym = _safe(f_m - r_m, f_m + r_m)
        else:
            asym = np.nan

        cosinor_r2 = self._pkg_cosinor().get("r2", np.nan)
        period_est = self._pkg_cosinor().get("period", 24.0)

        out = {
            "rise_time_mean":       r_m,
            "rise_time_std":        r_s,
            "fall_time_mean":       f_m,
            "fall_time_std":        f_s,
            "asymmetry_index":      asym,
            "fwhm_mean":            fw_m,
            "fwhm_std":             fw_s,
            "fwhm_relative":        _safe(fw_m, period_est),
            "cycle_amp_cv":         _safe(ca_s, ca_m) if ca_m > 0 else np.nan,
            "cycle_period_cv":      _safe(cp_s, cp_m) if cp_m > 0 else np.nan,
            "waveform_r2_vs_sinusoid": cosinor_r2,
        }

        # Cache intermediate peak/trough indices for plotting
        self._cache["_waveform_peaks"]   = peaks
        self._cache["_waveform_pp"]   = pp
        self._cache["_waveform_troughs"] = troughs
        self._cache["waveform"] = out
        return out

    def plot_waveform(self, ax: plt.Axes, color: str | None = None, label: bool = True):
        """
        Overlay waveform shape markers on ax.

        Draws:
          - peak markers (triangles up)
          - trough markers (triangles down)
          - rise/fall shaded intervals for first few cycles
          - FWHM annotation bracket on first peak
        """
        c = color or _PKG_COLORS["waveform"]
        self._pkg_waveform()  # ensure cache populated

        peaks   = self._cache.get("_waveform_peaks", np.array([]))
        troughs = self._cache.get("_waveform_troughs", np.array([]))
        pp = self._cache.get("_waveform_pp", np.array([]))
        waveform = self._cache.get("waveform", np.array([]))
        x, t    = self.signal, self.time
        dt  = self.dt

        import streamlit as st

        if len(peaks):
            ax.plot(t[peaks], x[peaks], "^", color=c, ms=7, zorder=5,
                    label="Peaks" if label else None)
        if len(troughs):
            ax.plot(t[troughs], x[troughs], "v", color=c, ms=7,
                    alpha=0.6, zorder=5,
                    label="Troughs" if label else None)
        
        if len(pp):
            ax.vlines(x=t[peaks], ymin=x[peaks] - pp["prominences"],
                    ymax = x[peaks], color = "C1")
            ax.hlines(y=pp["width_heights"], xmin=pp["left_ips"]/(1/dt),
                    xmax=pp["right_ips"]/(1/dt), color = "C1")

        from scipy.signal import savgol_filter

        # window_length must be odd and > polyorder
        # For dt=1h, window=5 is usually safe
        win_s = int(1 / dt * 10) * 1
        dx_smooth = savgol_filter(x, window_length=win_s, polyorder=3, deriv=1, delta=dt)
        d2x_smooth = savgol_filter(x, window_length=win_s, polyorder=3, deriv=2, delta=dt)

        ax2 = ax.twinx()  

        time_over = len(dx_smooth[dx_smooth >  0]) / len(dx_smooth)

        th =  dx_smooth.mean() * 2.5
        ax2.hlines(th, t.min(), t.max(), ls='--')
        time_over_th = len(dx_smooth[dx_smooth >  th]) / len(dx_smooth)
        explosivity = time_over_th / time_over
        st.markdown(f"""Time over 0: {time_over:.2f}
Explosivity: {1-explosivity:.2f}
        """)
        ax2.plot(t, dx_smooth, color='blue')  
        # Shade rise / fall for up to 3 cycles
        x_dt = _detrend_linear(x)
        for i, pk in enumerate(peaks[:3]):
            pre = troughs[troughs < pk]
            suf = troughs[troughs > pk]
            if not len(pre) or not len(suf):
                continue
            tr_pre, tr_suf = pre[-1], suf[0]
            ax.axvspan(t[tr_pre], t[pk],   alpha=0.08, color=c)
            ax.axvspan(t[pk],    t[tr_suf], alpha=0.05, color="#E85D24")
        
        info = f"""FWMH: {pp['widths'].mean()/(1/dt):.2f} h 
R2 to sin: {waveform['waveform_r2_vs_sinusoid']:.2f}
        """
        ax.set_title(info, loc='left')
        #ax.annotate(info, xy=(0, 0), xycoords='axes fraction', 
        #            ha='left', va='bottom', 
        #            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8), fontsize=10)
        # FWHM bracket on first peak
        f = self._cache.get("waveform", {})
        if len(peaks) and not np.isnan(f.get("fwhm_mean", np.nan)):
            pk   = peaks[0]
            hw   = f["fwhm_mean"] / 2
            ph   = x[pk]
            # half-prominence height
            pre  = troughs[troughs < pk]
            base = x[pre[-1]] if len(pre) else np.min(x)
            half_h = base + (ph - base) * 0.5
            ax.annotate(
                "", xy=(t[pk] + hw, half_h), xytext=(t[pk] - hw, half_h),
                arrowprops=dict(arrowstyle="<->", color=c, lw=1.2),
            )
            ax.text(t[pk],  base + (ph - base) * 0.4, f"FWHM\n{f['fwhm_mean']:.1f}h",
                    fontsize=10, color=c, ha="center", va="top")

    # ════════════════════════════════════════════════════════════════════════
    #  PACKAGE 3 — CYCLES
    # ════════════════════════════════════════════════════════════════════════

    def _pkg_cycles(self) -> dict:
        """
        Event-based cycle statistics.

        Features
        --------
        n_peaks / n_troughs        Detected peaks and troughs
        n_complete_cycles          Min(n_peaks, n_troughs) − 1 (conservative)
        ipi_mean / _std / _cv      Inter-peak interval: mean, std, CV (h)
        iti_mean / _std / _cv      Inter-trough interval: mean, std, cv (h)
        prominence_mean / _max     Peak prominence statistics
        prominence_cv              CV of peak prominences
        peaks_per_day              Duration-normalised peak rate
        period_event_based         Median IPI (independent period estimate)
        """
        if "cycles" in self._cache:
            return self._cache["cycles"]

        x  = _detrend_linear(self.signal)
        t  = self.time
        dt = self.dt

        min_dist = max(1, int(self.period_range[0] / dt / 2))
        #prom_value = (np.max(x) - np.min(x)) / 10
        prom_value = np.mean(np.abs(scipy_signal.hilbert(x))) * 1.5

        peaks,   pp = scipy_signal.find_peaks(x,  distance=min_dist, prominence=prom_value)
        troughs, tp = scipy_signal.find_peaks(-x, distance=min_dist, prominence=prom_value)

        proms = pp["prominences"] if len(peaks) else np.array([np.nan])
        ipis  = np.diff(t[peaks])   if len(peaks)   >= 2 else np.array([np.nan])
        itis  = np.diff(t[troughs]) if len(troughs)  >= 2 else np.array([np.nan])

        n_complete = max(0, min(len(peaks), len(troughs)) - 1)

        out = {
            "n_peaks":          int(len(peaks)),
            "n_troughs":        int(len(troughs)),
            "n_complete_cycles": int(n_complete),
            "ipi_mean":         float(np.nanmean(ipis)),
            "ipi_std":          float(np.nanstd(ipis)),
            "ipi_cv":           float(_safe(np.nanstd(ipis), np.nanmean(ipis))),
            "iti_mean":         float(np.nanmean(itis)),
            "iti_std":          float(np.nanstd(itis)),
            "iti_cv":           float(_safe(np.nanstd(itis), np.nanmean(itis))),
            "prominence_mean":  float(np.nanmean(proms)),
            "prominence_max":   float(np.nanmax(proms)) if len(proms) else np.nan,
            "prominence_cv":    float(_safe(np.nanstd(proms), np.nanmean(proms))),
            "peaks_per_day":    float(len(peaks) / (self.duration / 24)) if self.duration > 0 else np.nan,
            "period_event_based": float(np.nanmedian(ipis)),
        }

        self._cache["_cycles_peaks"]   = peaks
        self._cache["_cycles_troughs"] = troughs
        self._cache["_cycles_proms"]   = proms
        self._cache["cycles"] = out
        return out

    def plot_cycles(self, ax: plt.Axes, color: str | None = None, label: bool = True):
        """
        Overlay cycle event markers and IPI annotations on ax.

        Draws:
          - numbered peak labels
          - IPI brackets between consecutive peaks
          - prominence whiskers at each peak
        """
        c = color or _PKG_COLORS["cycles"]
        self._pkg_cycles()

        peaks  = self._cache.get("_cycles_peaks",   np.array([]))
        proms  = self._cache.get("_cycles_proms",   np.array([]))
        x, t   = self.signal, self.time

        for i, pk in enumerate(peaks):
            ax.text(t[pk], x[pk], str(i + 1),
                    fontsize=7, ha="center", va="bottom", color=c, fontweight="bold")

        # IPI bracket between consecutive peaks
        for i in range(min(len(peaks) - 1, 4)):
            y_br = x[peaks[i]] + 0.05 * (x.max() - x.min())
            mid  = 0.5 * (t[peaks[i]] + t[peaks[i + 1]])
            ipi  = t[peaks[i + 1]] - t[peaks[i]]
            ax.annotate("", xy=(t[peaks[i + 1]], y_br), xytext=(t[peaks[i]], y_br),
                        arrowprops=dict(arrowstyle="<->", color=c, lw=0.9, alpha=0.7))
            ax.text(mid, x[peaks[i]] + 0.02 * (x.max() - x.min()), f"{ipi:.1f} h", fontsize=10, ha="center",
                    va="top", color=c, alpha=0.9)

        # Prominence whiskers
        if len(proms):
            for pk, pr in zip(peaks, proms):
                ax.vlines(t[pk], x[pk] - pr, x[pk],
                          color=c, lw=0.8, alpha=0.4,
                          label="Prominence" if pk == peaks[0] and label else None)

    # ════════════════════════════════════════════════════════════════════════
    #  PACKAGE 4 — BASELINE
    # ════════════════════════════════════════════════════════════════════════

    def _pkg_baseline(self) -> dict:
        """
        Slow baseline / trend characterisation.

        Features
        --------
        trend_slope_per_day     Linear slope of rolling mean (units/day)
        trend_r2                R² of linear fit to rolling mean
        trend_is_significant    Whether slope p-value < 0.05
        depletion_index         Relative change in rolling mean (end/start − 1)
        mean_early / _late      Signal mean in first vs last third of recording
        nonstationarity_score   ADF test statistic normalised to [0, 1];
                                higher = more likely non-stationary / trending
                                (uses statsmodels if available, else variance-ratio)
        variance_ratio          Variance in last half / first half (>1 = growing)
        rolling_mean_cv         CV of rolling mean (how much baseline wanders)
        """
        if "baseline" in self._cache:
            return self._cache["baseline"]

        x, t = self.signal, self.time
        win  = max(3, int(self.period_range[1] / self.dt))   # ~1 period window

        roll = pd.Series(x).rolling(win, center=True, min_periods=2).mean().values
        valid = np.isfinite(roll)
        roll_v = roll[valid]
        t_v    = t[valid]

        if len(roll_v) > 2:
            sl, ic, rv, pv, _ = linregress(t_v, roll_v)
            slope_per_day = float(sl * 24)
            trend_r2      = float(rv ** 2)
            trend_sig     = bool(pv < 0.05)
        else:
            slope_per_day = np.nan
            trend_r2      = np.nan
            trend_sig     = False

        n3 = len(x) // 3
        mean_early = float(np.mean(x[:n3])) if n3 > 0 else np.nan
        mean_late  = float(np.mean(x[-n3:])) if n3 > 0 else np.nan
        depletion  = _safe(mean_late - mean_early, abs(mean_early) + 1e-9)

        # Non-stationarity
        if _HAS_STATSMODELS and len(x) >= 10:
            try:
                adf_res = _adfuller(x, autolag="AIC")
                # Map ADF stat to [0,1]: more positive = more non-stationary
                adf_stat = float(adf_res[0])
                crit_1   = adf_res[4].get("1%", -3.5)
                ns_score = float(np.clip(1 - (adf_stat - crit_1) / abs(crit_1), 0, 1))
            except Exception:
                ns_score = np.nan
        else:
            # Variance ratio fallback
            h = len(x) // 2
            v1 = np.var(x[:h])
            v2 = np.var(x[h:])
            ns_score = float(np.clip(_safe(v2, v1 + 1e-9) - 1, 0, None))

        h = len(x) // 2
        var_ratio = _safe(np.var(x[h:]), np.var(x[:h]) + 1e-9)

        out = {
            "trend_slope_per_day":   slope_per_day,
            "trend_r2":              trend_r2,
            "trend_is_significant":  int(trend_sig),
            "depletion_index":       depletion,
            "mean_early":            mean_early,
            "mean_late":             mean_late,
            "nonstationarity_score": ns_score,
            "variance_ratio":        float(var_ratio),
            "rolling_mean_cv":       float(_safe(np.nanstd(roll_v), np.nanmean(np.abs(roll_v)) + 1e-9)),
        }

        self._cache["_baseline_roll"] = (t_v, roll_v)
        self._cache["baseline"] = out
        return out

    def plot_baseline(self, ax: plt.Axes, color: str | None = None, label: bool = True):
        """
        Overlay rolling baseline and linear trend on ax.

        Draws:
          - rolling mean curve
          - linear trend line
          - early/late mean markers
        """
        c = color or _PKG_COLORS["baseline"]
        self._pkg_baseline()

        t_v, roll_v = self._cache.get("_baseline_roll", (np.array([]), np.array([])))
        if len(t_v) < 2:
            return

        f = self._cache["baseline"]
        ax.plot(t_v, roll_v, "-", color=c, lw=2.0, alpha=0.7,
                label="Rolling mean" if label else None)

        # Linear trend overlay
        sl  = f["trend_slope_per_day"] / 24  # per hour
        ic  = np.mean(roll_v) - sl * np.mean(t_v)
        t_l = np.array([t_v[0], t_v[-1]])
        ax.plot(t_l, sl * t_l + ic, "--", color=c, lw=1.2, alpha=0.55,
                label=f"Trend {f['trend_slope_per_day']:+.3f}/day" if label else None)
        ax.legend(loc='lower right', ncol=2, frameon=True)
        # Early / late mean horizontal segments
        n3 = len(self.time) // 3
        for val, seg, lbl in [
            (f["mean_early"], self.time[:n3],  "Early mean"),
            (f["mean_late"],  self.time[-n3:], "Late mean"),
        ]:
            if not np.isnan(val):
                ax.hlines(val, seg[0], seg[-1], color=c,
                          lw=1.5, ls=(0, (4, 2)), alpha=0.5)

    # ════════════════════════════════════════════════════════════════════════
    #  PACKAGE 5 — HARMONIC
    # ════════════════════════════════════════════════════════════════════════

    def _pkg_harmonic(self) -> dict:
        """
        FFT harmonic structure analysis.

        Features
        --------
        fundamental_period_h        Period at the dominant FFT peak (h)
        fundamental_power_fraction  Fraction of total power at fundamental
        harmonic2_power_fraction    Power fraction at 1st harmonic (T/2, ~12 h)
        harmonic3_power_fraction    Power fraction at 2nd harmonic (T/3, ~8 h)
        harmonic_ratio_2            harmonic2 / fundamental (0 = pure sinusoid)
        harmonic_ratio_3            harmonic3 / fundamental
        harmonic_distortion_index   Total harmonic distortion (THD)
        secondary_peak_ratio        Power at 2nd-strongest peak / fundamental
        circadian_band_fraction     Fraction of total power in period_range
        spectral_entropy            Shannon entropy of normalised PSD
        """
        if "harmonic" in self._cache:
            return self._cache["harmonic"]

        x    = self.signal - np.mean(self.signal)
        N    = len(x)
        fv   = np.abs(fft(x * np.hanning(N))) ** 2
        freq = fftfreq(N, self.dt)

        pos       = freq > 0
        pw        = fv[pos]
        fr        = freq[pos]
        total_pw  = np.sum(pw) + 1e-12

        # Fundamental: peak in circadian band
        fmin, fmax = 1 / self.period_range[1], 1 / self.period_range[0]
        circ_mask  = (fr >= fmin) & (fr <= fmax)

        if np.any(circ_mask):
            fund_idx   = np.argmax(pw[circ_mask])
            fund_freq  = fr[circ_mask][fund_idx]
            fund_power = pw[circ_mask][fund_idx]
            fund_T     = 1.0 / fund_freq
        else:
            fund_idx   = np.argmax(pw)
            fund_freq  = fr[fund_idx]
            fund_power = pw[fund_idx]
            fund_T     = 1.0 / fund_freq if fund_freq > 0 else 24.0

        fund_frac = _safe(fund_power, total_pw)

        # Harmonics: search in a ±20% window around T/2 and T/3
        def _harmonic_power(target_T, width=0.20):
            lo, hi = 1 / (target_T * (1 + width)), 1 / (target_T * (1 - width))
            mask   = (fr >= lo) & (fr <= hi)
            return float(np.max(pw[mask])) if np.any(mask) else 0.0

        h2_power = _harmonic_power(fund_T / 2)
        h3_power = _harmonic_power(fund_T / 3)
        h2_frac  = _safe(h2_power, total_pw)
        h3_frac  = _safe(h3_power, total_pw)

        thd = float(np.sqrt(h2_power ** 2 + h3_power ** 2) / (fund_power + 1e-12))

        # Second-strongest circadian-band peak
        pw_circ = pw[circ_mask].copy() if np.any(circ_mask) else np.array([])
        if len(pw_circ) > 1:
            pw_circ[np.argmax(pw_circ)] = 0
            sec_power = np.max(pw_circ)
            sec_ratio = _safe(sec_power, fund_power)
        else:
            sec_ratio = 0.0

        # Spectral entropy
        pn = pw / total_pw
        entropy = float(-np.sum(pn * np.log(pn + 1e-12)) / np.log(len(pn)))

        # Circadian band fraction
        circ_frac = _safe(np.sum(pw[circ_mask]), total_pw) if np.any(circ_mask) else 0.0

        out = {
            "fundamental_period_h":       float(fund_T),
            "fundamental_power_fraction": float(fund_frac),
            "harmonic2_power_fraction":   float(h2_frac),
            "harmonic3_power_fraction":   float(h3_frac),
            "harmonic_ratio_2":           float(_safe(h2_power, fund_power)),
            "harmonic_ratio_3":           float(_safe(h3_power, fund_power)),
            "harmonic_distortion_index":  float(thd),
            "secondary_peak_ratio":       float(sec_ratio),
            "circadian_band_fraction":    float(circ_frac),
            "spectral_entropy":           entropy,
        }

        self._cache["_harmonic_freq"] = fr
        self._cache["_harmonic_power"] = pw
        self._cache["_harmonic_total"] = total_pw
        self._cache["harmonic"] = out
        return out

    def plot_harmonic(self, ax: plt.Axes, color: str | None = None, label: bool = True):
        """
        Draw the power spectrum with annotated fundamental and harmonics.

        Note: this replaces the raw signal on ax with the spectrum.
              Recommended to use on a dedicated axis, not the trace axis.
        """
        c = color or _PKG_COLORS["harmonic"]
        self._pkg_harmonic()

        fr = self._cache.get("_harmonic_freq")
        pw = self._cache.get("_harmonic_power")
        if fr is None:
            return

        f    = self._cache["harmonic"]
        # Convert to period axis (clip to reasonable range)
        with np.errstate(divide="ignore"):
            per = np.where(fr > 0, 1 / fr, np.inf)
        mask = (per >= 4) & (per <= 100)

        ax.fill_between(per[mask], pw[mask], alpha=0.25, color=c)
        ax.plot(per[mask], pw[mask], color=c, lw=1.2,
                label="Power spectrum" if label else None)

        # Fundamental
        ax.axvline(f["fundamental_period_h"], color=c, lw=1.5,
                   label=f"Fundamental {f['fundamental_period_h']:.1f} h" if label else None)
        # Harmonics
        for n, key, ls in [(2, "harmonic2_power_fraction", "--"),
                           (3, "harmonic3_power_fraction", ":")]:
            hT = f["fundamental_period_h"] / n
            ax.axvline(hT, color=c, lw=1.0, ls=ls, alpha=0.6,
                       label=f"H{n} ({hT:.1f} h)" if label else None)

        # Circadian band shading
        ax.axvspan(*self.period_range, color=c, alpha=0.07,
                   label="Circadian window" if label else None)

        ax.set_xlabel("Period (h)")
        ax.set_ylabel("Power")
        ax.set_xlim(4, 60)

    # ════════════════════════════════════════════════════════════════════════
    #  PACKAGE 6 — NOISE
    # ════════════════════════════════════════════════════════════════════════

    def _pkg_noise(self) -> dict:
        """
        Noise characterisation and signal quality.

        Features
        --------
        residual_std            Std dev of cosinor-fit residuals
        residual_acf_lag1       ACF of residuals at lag 1 (structured noise?)
        residual_is_white       Whether residual ACF[1] is below threshold
        noise_floor_estimate    Mean power in non-circadian frequency bands
        snr_spectral            Circadian peak power / noise floor power
        band_snr_ultradian      Power ratio 8–18 h band vs noise floor
        band_snr_infradian      Power ratio 30–60 h band vs noise floor
        usable_fraction         Fraction of timepoints within 3σ of signal mean
        signal_dynamic_range    (max − min) / mean; proxy for detector linearity
        """
        if "noise" in self._cache:
            return self._cache["noise"]

        # Residuals from cosinor fit
        cos_f = self._pkg_cosinor()
        if not np.isnan(cos_f.get("amplitude", np.nan)):
            fitted = _cosinor_model(
                self.time, cos_f["amplitude"], cos_f["acrophase_rad"],
                cos_f["mesor"], cos_f["period"]
            )
            resid     = self.signal - fitted
            r_std     = float(np.std(resid))
            acf_v     = _acf(resid, nlags=min(5, len(resid) - 2))
            r_acf1    = float(acf_v[1]) if len(acf_v) > 1 else np.nan
            is_white  = int(abs(r_acf1) < 0.3) if not np.isnan(r_acf1) else 0
        else:
            r_std, r_acf1, is_white = np.nan, np.nan, 0

        # Spectral noise floor (median power outside circadian band)
        fr  = self._cache.get("_harmonic_freq")
        pw  = self._cache.get("_harmonic_power")
        if fr is None:
            self._pkg_harmonic()
            fr  = self._cache["_harmonic_freq"]
            pw  = self._cache["_harmonic_power"]

        fmin, fmax = 1 / self.period_range[1], 1 / self.period_range[0]
        circ_mask  = (fr >= fmin) & (fr <= fmax)
        noise_mask = ~circ_mask & (fr > 0)
        noise_floor = float(np.median(pw[noise_mask])) if np.any(noise_mask) else np.nan

        fund_power = pw[circ_mask].max() if np.any(circ_mask) else np.nan
        snr_spec   = _safe(fund_power, noise_floor) if not np.isnan(noise_floor) else np.nan

        def _band_snr(lo_h, hi_h):
            lo_f, hi_f = 1 / hi_h, 1 / lo_h
            m = (fr >= lo_f) & (fr <= hi_f)
            if not np.any(m) or np.isnan(noise_floor) or noise_floor < 1e-12:
                return np.nan
            return float(_safe(np.max(pw[m]), noise_floor))

        x   = self.signal
        mu  = np.mean(x)
        sig = np.std(x)
        usable = float(np.mean(np.abs(x - mu) < 3 * sig))
        dyn_range = _safe(np.max(x) - np.min(x), abs(mu) + 1e-9)

        out = {
            "residual_std":          r_std,
            "residual_acf_lag1":     r_acf1,
            "residual_is_white":     is_white,
            "noise_floor_estimate":  noise_floor,
            "snr_spectral":          snr_spec,
            "band_snr_ultradian":    _band_snr(8, 18),
            "band_snr_infradian":    _band_snr(30, 60),
            "usable_fraction":       usable,
            "signal_dynamic_range":  float(dyn_range),
        }

        self._cache["_noise_resid"] = resid if not np.isnan(r_std) else None
        self._cache["noise"] = out
        return out

    def plot_noise(self, ax: plt.Axes, color: str | None = None, label: bool = True):
        """
        Overlay cosinor residuals as a shaded error band on ax.
        """
        c     = color or _PKG_COLORS["noise"]
        self._pkg_noise()
        resid = self._cache.get("_noise_resid")
        if resid is None:
            return

        cos_f   = self._cache["cosinor"]
        fitted  = _cosinor_model(
            self.time, cos_f["amplitude"], cos_f["acrophase_rad"],
            cos_f["mesor"], cos_f["period"]
        )
        ax.fill_between(self.time, fitted, self.signal,
                        color=c, alpha=0.20,
                        label="Residuals" if label else None)
        ax.plot(self.time, resid + fitted - np.mean(resid), ":",
                color=c, lw=0.8, alpha=0.5)

    # ════════════════════════════════════════════════════════════════════════
    #  PACKAGE 7 — LOMB-SCARGLE
    # ════════════════════════════════════════════════════════════════════════

    def _pkg_lomb_scargle(self) -> dict:
        """
        Lomb-Scargle periodogram (works for all lengths and irregular sampling).

        Features
        --------
        peak_period_h       Period at LS peak in circadian window (h)
        peak_power          Normalised LS power at peak
        power_ratio         Peak power / mean power in window
        fap                 Baluev false-alarm probability at peak
        mean_power          Mean LS power in circadian window
        bandwidth_h         Width of the LS peak at half-maximum (h)
        n_significant_peaks Number of peaks with power > 2× mean
        second_peak_ratio   Second peak power / first peak power
        """
        if "lomb_scargle" in self._cache:
            return self._cache["lomb_scargle"]

        t, x = self.time, self.signal
        x_c  = x - np.mean(x)
        n_f  = 500
        freq = np.linspace(1 / self.period_range[1], 1 / self.period_range[0], n_f)
        ang  = 2 * np.pi * freq

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pgram = scipy_signal.lombscargle(t, x_c, ang, normalize=True)

        pk_idx   = int(np.argmax(pgram))
        pk_power = float(pgram[pk_idx])
        pk_T     = float(1.0 / freq[pk_idx])
        mean_pw  = float(np.mean(pgram))
        pw_ratio = _safe(pk_power, mean_pw)

        N   = len(x)
        fap = float(np.clip(1 - (1 - np.exp(-pk_power)) ** N, 0, 1))

        # Peak width at half-max
        half    = pk_power / 2
        above   = pgram >= half
        if np.any(above):
            segs = np.diff(above.astype(int))
            starts = np.where(segs == 1)[0]
            ends   = np.where(segs == -1)[0]
            if len(starts) and len(ends):
                bw = float(abs(1 / freq[ends[0]] - 1 / freq[starts[0]]))
            else:
                bw = np.nan
        else:
            bw = np.nan

        # Secondary peaks
        thresh = 2 * mean_pw
        n_sig  = int(np.sum(pgram > thresh))
        pg2    = pgram.copy()
        pg2[pk_idx] = 0
        sec_ratio = _safe(float(np.max(pg2)), pk_power)

        out = {
            "peak_period_h":       pk_T,
            "peak_power":          pk_power,
            "power_ratio":         float(pw_ratio),
            "fap":                 fap,
            "mean_power":          mean_pw,
            "bandwidth_h":         float(bw) if not np.isnan(bw) else np.nan,
            "n_significant_peaks": n_sig,
            "second_peak_ratio":   float(sec_ratio),
        }

        self._cache["_ls_freq"]  = freq
        self._cache["_ls_pgram"] = pgram
        self._cache["lomb_scargle"] = out
        return out

    def plot_lomb_scargle(self, ax: plt.Axes, color: str | None = None, label: bool = True):
        """
        Draw the Lomb-Scargle periodogram on ax (dedicated axis recommended).
        """
        c = color or _PKG_COLORS["lomb_scargle"]
        self._pkg_lomb_scargle()

        freq  = self._cache.get("_ls_freq")
        pgram = self._cache.get("_ls_pgram")
        if freq is None:
            return

        f   = self._cache["lomb_scargle"]
        per = 1 / freq

        ax.fill_between(per, pgram, alpha=0.25, color=c)
        ax.plot(per, pgram, color=c, lw=1.5,
                label="Lomb-Scargle" if label else None)
        ax.axvline(f["peak_period_h"], color=c, lw=1.5, ls="--",
                   label=f"Peak {f['peak_period_h']:.1f} h" if label else None)
        ax.axhline(2 * f["mean_power"], color=c, lw=0.8, ls=":",
                   alpha=0.6, label="2× mean power" if label else None)
        ax.legend()
        ax.set_xlabel("Period (h)")
        ax.set_ylabel("Normalised power")
        ax.set_xlim(self.period_range[0] - 2, self.period_range[1] + 2)

    # ════════════════════════════════════════════════════════════════════════
    #  PACKAGE 8 — WAVELET RIDGE  (long series only)
    # ════════════════════════════════════════════════════════════════════════

    def _pkg_wavelet_ridge(self) -> dict:
        """
        Instantaneous period / amplitude / phase from PyBoat wavelet ridge.
        Only computed for long series (duration > 48 h, n ≥ 20).

        Features — period
        -----------------
        period_mean / _std / _cv / _iqr
        period_trend_slope / _pvalue
        period_is_drifting
        period_early / _late / _change

        Features — amplitude
        --------------------
        amplitude_mean / _std / _cv
        amplitude_trend_slope / _pvalue
        is_damping / damping_rate / half_life

        Features — phase
        ----------------
        phase_coherence
        phase_circular_variance
        phase_velocity_std

        Features — ridge quality
        ------------------------
        ridge_coverage / _continuity / _n_gaps
        power_mean_normalized
        overall_stability_score
        """
        if "wavelet_ridge" in self._cache:
            return self._cache["wavelet_ridge"]

        if self.is_short:
            return {}

        if not _HAS_PYBOAT:
            warnings.warn("pyboat not installed — wavelet_ridge package unavailable.")
            return {}

        periods_ax = np.linspace(self.period_range[0], self.period_range[1], 100)
        wAn = WAnalyzer(periods_ax, self.dt, p_max=20)
        wAn.compute_spectrum(self.signal, do_plot=False)

        smooth = min(20, max(3, self.n // 10))
        try:
            wAn.get_maxRidge(power_thresh=0.0, smoothing_wsize=smooth)
            rd = wAn.ridge_data
            has_ridge = len(rd) > 0
        except Exception:
            has_ridge = False
            rd = None

        if not has_ridge:
            self._cache["wavelet_ridge"] = {}
            return {}

        per  = rd["periods"].values
        pw   = rd["power"].values
        amp  = np.sqrt(np.maximum(pw, 0))
        phase= rd["phase"].values
        t_rd = rd.index.values
        N    = self.n

        # Period stats
        def _cv(a):
            m = np.mean(a)
            return _safe(np.std(a), m) if m > 0 else np.nan

        sl_p, _, _, pv_p, _ = linregress(t_rd, per) if len(per) > 2 else (0, 0, 0, 1, 0)
        mid = len(per) // 2
        per_e = float(np.mean(per[:mid]))
        per_l = float(np.mean(per[mid:]))

        # Amplitude
        sl_a, _, _, pv_a, _ = linregress(t_rd, amp) if len(amp) > 2 else (0, 0, 0, 1, 0)
        is_damp = bool((sl_a < -1e-4) and (pv_a < 0.05))
        if is_damp and np.mean(amp) > 0:
            try:
                def _exp(t, A0, lam): return A0 * np.exp(-lam * t)
                popt, _ = curve_fit(_exp, t_rd - t_rd[0], amp,
                                    p0=[amp[0], 0.01], maxfev=1000)
                damp_rate = float(popt[1])
                half_life = float(np.log(2) / damp_rate) if damp_rate > 0 else np.inf
            except Exception:
                damp_rate = np.nan; half_life = np.nan
        else:
            damp_rate = 0.0; half_life = np.inf

        # Phase coherence (circular statistics)
        R = float(np.sqrt(np.mean(np.cos(phase))**2 + np.mean(np.sin(phase))**2))
        circ_var = 1 - R
        ph_unwrap = np.unwrap(phase)
        ph_vel    = np.gradient(ph_unwrap, t_rd)
        ph_vel_std = float(np.std(ph_vel))

        # Ridge quality
        coverage   = len(rd) / N
        dt_rd      = np.diff(t_rd)
        n_gaps     = int(np.sum(dt_rd > 2 * self.dt))
        continuity = 1 - _safe(n_gaps, max(len(dt_rd), 1))
        total_pw   = np.mean(wAn.modulus) if hasattr(wAn, "modulus") else 1.0
        pw_norm    = _safe(np.mean(pw), total_pw + 1e-12)
        stab       = float(np.sqrt(
            (1 / (1 + _cv(per))) * (1 / (1 + _cv(amp)))
        )) if not np.isnan(_cv(per)) and not np.isnan(_cv(amp)) else np.nan

        out = {
            # Period
            "period_mean":          float(np.mean(per)),
            "period_std":           float(np.std(per)),
            "period_cv":            float(_cv(per)),
            "period_iqr":           float(np.percentile(per, 75) - np.percentile(per, 25)),
            "period_trend_slope":   float(sl_p),
            "period_trend_pvalue":  float(pv_p),
            "period_is_drifting":   int(abs(sl_p) > 0.01 and pv_p < 0.05),
            "period_early":         per_e,
            "period_late":          per_l,
            "period_change":        float(per_l - per_e),
            # Amplitude
            "amplitude_mean":       float(np.mean(amp)),
            "amplitude_std":        float(np.std(amp)),
            "amplitude_cv":         float(_cv(amp)),
            "amplitude_trend_slope":float(sl_a),
            "amplitude_trend_pvalue":float(pv_a),
            "is_damping":           int(is_damp),
            "damping_rate":         damp_rate,
            "half_life":            half_life,
            # Phase
            "phase_coherence":         R,
            "phase_circular_variance": float(circ_var),
            "phase_velocity_std":      ph_vel_std,
            # Ridge quality
            "ridge_coverage":          float(coverage),
            "ridge_continuity":        float(continuity),
            "ridge_n_gaps":            n_gaps,
            "power_mean_normalized":   float(pw_norm),
            "overall_stability_score": stab,
        }

        self._cache["_wavelet_rd"]  = rd
        self._cache["_wavelet_wAn"] = wAn
        self._cache["wavelet_ridge"] = out
        return out

    def plot_wavelet_ridge(self, ax: plt.Axes, color: str | None = None, label: bool = True):
        """
        Overlay instantaneous period and amplitude envelope from wavelet ridge.

        Draws on a twin axis:
          - amplitude envelope (shaded)
          - instantaneous period as a colour annotation strip below x-axis
        """
        c = color or _PKG_COLORS["wavelet_ridge"]
        self._pkg_wavelet_ridge()

        rd = self._cache.get("_wavelet_rd")
        if rd is None:
            return

        t_rd = rd.index.values
        amp  = np.sqrt(np.maximum(rd["power"].values, 0))
        per  = rd["periods"].values

        # Amplitude envelope on signal axis
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.1))
        ax3.fill_between(t_rd, 0, amp, color=c, alpha=0.15,
                        label="Wavelet amplitude" if label else None)
        ax3.plot(t_rd, amp, color=c, lw=1.2, alpha=0.6)

        # Instantaneous period as colour strip (twin axis)
        ax2 = ax.twinx()
        ax2.plot(t_rd, per, color=c, lw=1.5, ls="--", alpha=0.7,
                 label=f"Inst. period (mean={np.mean(per):.1f} h)" if label else None)
        ax2.set_ylabel("Period (h)", color=c, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=c, labelsize=10)
        ax3.set_ylabel("Amplitude", fontsize=10)
        ax.set_ylabel('Signal', fontsize=10)
        pmin, pmax = self.period_range
        ax2.set_ylim(pmin - 2, pmax + 2)
        #ax.set_xticks([t for t in range(0, t_rd, 6)])
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        #plt.legend()

    # ════════════════════════════════════════════════════════════════════════
    #  CONVENIENCE — full visual summary
    # ════════════════════════════════════════════════════════════════════════

    def plot_summary(
        self,
        packages: list[str] | None = None,
        title: str = "",
        figsize: tuple | None = None,
    ) -> plt.Figure:
        """
        Generate a multi-panel visual summary for a single signal.

        Layout
        ------
        Row 0 (tall): raw signal + cosinor + waveform + cycles + noise overlays
        Row 1 (medium): baseline overlay on signal
        Row 2 (shorter): Lomb-Scargle periodogram | harmonic power spectrum
        Row 3 (shorter, long series only): wavelet ridge instantaneous period

        Parameters
        ----------
        packages : list or None
            Which packages to include. Defaults to all available.
        title : str
            Figure suptitle (e.g. sample name).
        figsize : tuple or None
            Override figure size.

        Returns
        -------
        matplotlib Figure
        """
        if packages is None:
            packages = self.available_packages()

        has_wavelet = "wavelet_ridge" in packages and not self.is_short and _HAS_PYBOAT

        n_rows = 3 + int(has_wavelet)
        h_ratios = [3, 2, 1.5] + ([1.5] if has_wavelet else [])
        fs = figsize or (12, 2.8 * n_rows)

        fig, axes = plt.subplots(
            n_rows, 2 if not has_wavelet else 1,
            figsize=fs,
            gridspec_kw={"height_ratios": h_ratios},
            squeeze=False,
            layout="tight",
        )

        # ── Row 0: signal + overlays ────────────────────────────────────────
        ax0 = fig.add_subplot(fig.add_gridspec(n_rows, 1, height_ratios=h_ratios)[0]) \
              if has_wavelet else axes[0, 0]
        if has_wavelet:
            axes[0, 0].remove()

        ax0.plot(self.time, self.signal, color="#888780", lw=1.2, alpha=0.8,
                 label="Signal")
        ax0.set_title("Signal overview", fontsize=10, loc="left")
        ax0.set_xlabel("Time (h)")
        ax0.set_ylabel("Signal")

        for pkg, meth in [
            ("cosinor",  self.plot_cosinor),
            ("waveform", self.plot_waveform),
            ("cycles",   self.plot_cycles),
            ("noise",    self.plot_noise),
        ]:
            if pkg in packages:
                try:
                    meth(ax0)
                except Exception:
                    pass
        ax0.legend(fontsize=7, ncol=2, framealpha=0.5)

        # ── Row 1: baseline ────────────────────────────────────────────────
        if not has_wavelet:
            ax1 = axes[1, 0]
        else:
            ax1 = fig.add_subplot(fig.add_gridspec(n_rows, 1, height_ratios=h_ratios)[1])

        ax1.plot(self.time, self.signal, color="#888780", lw=1.0, alpha=0.5)
        ax1.set_title("Baseline & trend", fontsize=10, loc="left")
        ax1.set_xlabel("Time (h)")
        ax1.set_ylabel("Signal")
        if "baseline" in packages:
            try:
                self.plot_baseline(ax1)
            except Exception:
                pass
        ax1.legend(fontsize=7, framealpha=0.5)

        # ── Row 2: periodograms ────────────────────────────────────────────
        if not has_wavelet:
            ax2a, ax2b = axes[2, 0], axes[2, 1]
        else:
            gs2 = fig.add_gridspec(n_rows, 2, height_ratios=h_ratios)[2, :]
            ax2a = fig.add_subplot(gs2[0] if hasattr(gs2, "__len__") else gs2)
            ax2b = ax2a   # fallback; split below

        if not has_wavelet:
            if "lomb_scargle" in packages:
                try:
                    self.plot_lomb_scargle(ax2a)
                    ax2a.legend(fontsize=7, framealpha=0.5)
                except Exception:
                    pass
            if "harmonic" in packages:
                try:
                    self.plot_harmonic(ax2b)
                    ax2b.legend(fontsize=7, framealpha=0.5)
                except Exception:
                    pass

        # ── Row 3 (wavelet, long only) ─────────────────────────────────────
        if has_wavelet and n_rows >= 4:
            ax3 = axes[3, 0] if not has_wavelet else \
                  fig.add_subplot(fig.add_gridspec(n_rows, 1, height_ratios=h_ratios,
                  )[3])
            ax3.spines['left'].set_position(('outward', 40))
            ax3.plot(self.time, self.signal, color="#888780", lw=1.0, alpha=0.5)
            ax3.set_title("Wavelet ridge (instantaneous)", fontsize=10, loc="left")
            ax3.set_xlabel("Time (h)")
            ax3.set_ylabel("Amplitude")
            try:
                self.plot_wavelet_ridge(ax3)
                ax3.legend(fontsize=7, framealpha=0.5)
            except Exception:
                pass

        if title:
            fig.suptitle(title, fontsize=13, fontweight="500")

        return fig


# ════════════════════════════════════════════════════════════════════════════
#  Quick smoke test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    t   = np.arange(0, 240, 2.0)
    sig = (2.0 * np.sin(2 * np.pi * t / 24.5 - 0.3)
           + 0.4 * np.sin(2 * np.pi * t / 12)      # harmonic
           + np.random.normal(0, 0.4, len(t))
           + 0.003 * t)                              # slow drift

    ext  = ChronotopiaFeatureExtractor(sig, t, period_range=(18, 30))
    feat = ext.extract()

    print(f"Available packages : {ext.available_packages()}")
    print(f"Total features     : {len(feat)}")
    print()
    for k, v in sorted(feat.items()):
        print(f"  {k:<45} {v}")

    fig = ext.plot_summary(title="Smoke test — synthetic 24.5 h rhythm")
    plt.savefig("/mnt/user-data/outputs/chronotopia_feature_extractor_demo.png",
                dpi=120, bbox_inches="tight")
    print("\nDemo figure saved.")
