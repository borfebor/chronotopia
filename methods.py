#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:39:58 2025

@author: borfebor
"""

import warnings

import pandas as pd
import numpy as np
import streamlit as st
from scipy import signal
from scipy.optimize import curve_fit
from scipy.fft import dct, idct
# NOTE: all figure drawing moved to plots.py in v0.7.3 — seaborn, pyplot,
# LinearSegmentedColormap, PdfPages and BytesIO are imported there now, not here.
from statsmodels.tsa.tsatools import detrend
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import fisher_exact, ttest_ind
from itertools import combinations
from astropy.timeseries import LombScargle
from pyboat import WAnalyzer


class methods:
    
    """
    A collection of static methods for time-series processing, detrending,
    normalization, visualization, and statistical analysis.
    """

    @staticmethod
    def example_data():
        time = np.arange(0, 10 * 24 * 60, 10)  # 10 days, 10-minute intervals
        df = pd.DataFrame({'Time': time})
        np.random.seed(42)
        for i in range(10):
            if i < 8:
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.8, 1.2)
                noise = np.random.normal(0, 0.2, len(time))
                signal_data = amplitude * np.sin(2 * np.pi * time / (24 * 60) + phase) + noise + 1
            else:
                signal_data = np.random.normal(1, 0.2, len(time))
            df[f'Sample_{i+1}'] = signal_data
        return df
    
    def generate_rhythm_dataset(
        num_days=10,
        points_per_day=24,
        n_samples=50,
        percent_rhythmic=0.6,
        period=24.0,               # intrinsic period in hours (can be scalar or (min,max))
        entrain=False,
        entrain_start_day=0,
        entrain_end_day=4,
        entrain_period=24.0,       # period of the entraining cycle in hours
        noise_sd=0.5,
        amp_range=(0.8, 1.2),
        phase_jitter_sd=0.2,       # radians jitter when entrained
        intrinsic_period_jitter=0.2, # hours sd to jitter each sample's intrinsic period
        nonrhythm_drift=True,
        random_seed=None,
        waveform='sin'             # 'sin' or 'square' or 'saw'
    ):
        """
        Returns (data_df, meta_df, time_hours)
        data_df: pandas DataFrame with columns ['time_hours', 'sample_0', 'sample_1', ...]
        meta_df: pandas DataFrame with per-sample metadata (is_rhythmic, amplitude, intrinsic_period, phase0)
        time_hours: numpy array of times in hours
        """
        rng = np.random.default_rng(random_seed)
    
        total_points = int(num_days * points_per_day) +1
        dt = 24 / points_per_day
        time_hours = np.arange(0, total_points) * dt
    
        # Determine rhythmic samples
        n_rhyth = int(round(n_samples * percent_rhythmic))
        is_rhythmic = np.array([True]*n_rhyth + [False]*(n_samples-n_rhyth))
        rng.shuffle(is_rhythmic)
    
        # Allow period argument to be scalar or (min,max) for sampling
        if np.isscalar(period):
            period_arr = rng.normal(loc=period, scale=intrinsic_period_jitter, size=n_samples)
        else:
            # period given as (min,max)
            period_arr = rng.uniform(low=period[0], high=period[1], size=n_samples)
    
        # amplitude & initial phase
        amp_arr = rng.uniform(amp_range[0], amp_range[1], size=n_samples)
        phase0 = rng.uniform(-np.pi, np.pi, size=n_samples)
    
        # entrainment times in hours
        t_entrain_start = entrain_start_day * 24.0
        t_entrain_end = entrain_end_day * 24.0
    
        # precompute driving phase if entrain
        if entrain:
            driving_phase = 2 * np.pi * (time_hours / entrain_period)
        else:
            driving_phase = None
    
        # prepare output array
        data = np.zeros((total_points, n_samples))
    
        for i in range(n_samples):
            A = amp_arr[i]
            intrinsic_T = max(0.1, period_arr[i])
            # track instantaneous phase
            phase = np.zeros(total_points)
    
            if is_rhythmic[i]:
                # before entrainment window (or if no entrainment)
                for t_idx, t in enumerate(time_hours):
                    if entrain and (t_entrain_start <= t < t_entrain_end):
                        # follow driving phase + small per-sample phase offset and jitter
                        sample_phase = driving_phase[t_idx] + phase0[i] + rng.normal(0, phase_jitter_sd)
                        phase[t_idx] = sample_phase
                    else:
                        if entrain:
                            # if we are at the first point after release, find phase at release and continue with intrinsic freq
                            if t == 0:
                                # no prior value
                                phase[t_idx] = phase0[i]
                            else:
                                # If previous time was entrained, continue from that phase; otherwise continue advancing
                                prev_phase = phase[t_idx-1]
                                # advance by intrinsic angular velocity
                                omega = 2*np.pi / intrinsic_T
                                phase[t_idx] = prev_phase + omega * (dt)
                        else:
                            # never entrained: simple intrinsic evolution from phase0
                            if t_idx == 0:
                                phase[t_idx] = phase0[i]
                            else:
                                omega = 2*np.pi / intrinsic_T
                                phase[t_idx] = phase[t_idx-1] + omega * (dt)
    
                # compute waveform
                if waveform == 'sin':
                    signal = A * np.sin(phase)
                elif waveform == 'square':
                    signal = A * np.sign(np.sin(phase))
                elif waveform == 'saw':
                    # sawtooth from -1 to 1
                    signal = A * (2*(phase/(2*np.pi) - np.floor(phase/(2*np.pi)+0.5)))
                else:
                    raise ValueError("unsupported waveform")
    
                # add noise
                signal = signal + rng.normal(0, noise_sd, size=signal.shape)
    
            else:
                # non-rhythmic: low freq drift + white noise
                drift = np.zeros_like(time_hours)
                if nonrhythm_drift:
                    n_trend_components = rng.integers(1,4)
                    for _ in range(n_trend_components):
                        freq = rng.uniform(0.01, 0.2)  # cycles per hour ~ very slow
                        amp = rng.uniform(0.1, 1.0) * A
                        phase_tr = rng.uniform(0, 2*np.pi)
                        drift += amp * np.sin(2*np.pi*freq*time_hours + phase_tr)
                signal = drift + rng.normal(0, noise_sd*1.5, size=time_hours.shape)
    
            data[:, i] = signal
    
        # Build DataFrames
        cols = [f"sample_{i+1}" for i in range(n_samples)]
        data_df = pd.DataFrame(data, columns=cols)
        data_df.insert(0, "time_hours", time_hours)
    
        meta_df = pd.DataFrame({
            "sample": cols,
            "is_rhythmic": is_rhythmic,
            "amplitude": amp_arr,
            "intrinsic_period_hours": period_arr,
            "phase0_radians": phase0
        })
    
        return data_df, meta_df, time_hours

    @staticmethod
    def importer(file):
        name = file if isinstance(file, str) else file.name

        try:
            if name.upper().endswith(('TXT', 'TSV')):
                return pd.read_csv(file, sep='\t')
            elif name.upper().endswith('CSV'):
                return pd.read_csv(file)
            elif name.upper().endswith('XLSX'):
                return pd.read_excel(file)
            else:
                st.warning("Unsupported file format. Use XLSX, CSV, or TXT.")
                return None
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
        
    @staticmethod
    def time_changer(x, unit='Minutes'):
        conversions = {'Minutes': x / 60, 'Hours': x, 'Days': x * 24, 'Seconds': x / 3600}
        return conversions.get(unit, x)

    @staticmethod
    def hourly(df, t_col):
        return df[df[t_col] % 1 == 0]
    
    @staticmethod
    def resampling(df, t_col):
        
        dt = np.mean(df[t_col].diff())
        if dt > 1:
            st.error('Sample frequency is more than 1 hour per sample, interpolation is needed in this case')
            st.stop()
        # Convert numeric seconds to Timedelta and set as index
        df['duration_sec'] = pd.to_timedelta(df[t_col], unit='h')
        df = df.set_index('duration_sec')

        # Resample to 1-second intervals and take the mean
        resampled_df = df.resample('1h').mean()
        resampled_df[t_col] = resampled_df[t_col].astype(int)
        return resampled_df.reset_index(drop=True)

    # ── NaN handling for detrending ─────────────────────────────────────────────
    #
    # Every detrending method except the rolling mean used to fail on missing data,
    # and each failed differently: `scipy.signal.detrend` RAISES
    # ("array must not contain infs or NaNs"), while `statsmodels.detrend` and
    # `signal.hilbert` silently return an all-NaN column — so a single gap in one
    # well quietly wiped that well from the analysis. `savitzky_golay` already
    # handled this properly; the helpers below give the detrending path the same
    # contract: interpolate for the fit, then put the gaps back where they were.

    @staticmethod
    def _fill_for_fit(frame):
        """
        Return (filled_frame, nan_mask). The filled frame is safe to hand to a
        least-squares fit or an FFT-based filter; re-apply the mask afterwards
        with `.mask(nan_mask)` so a gap stays a gap in the output.
        """
        frame = pd.DataFrame(frame).astype(float)
        nan_mask = frame.isna()
        if nan_mask.to_numpy().any():
            # Columns that are entirely NaN interpolate to nothing — 0.0 keeps the
            # fit well-posed and the mask blanks the column again on the way out.
            frame = frame.interpolate(limit_direction="both").fillna(0.0)
        return frame, nan_mask

    # ── baseline estimators ─────────────────────────────────────────────────────
    #
    # Detrending is two independent decisions that used to be welded into one name:
    # HOW the baseline is estimated, and HOW it is removed. Splitting them means
    # every estimator gains a divisive variant for free, and the subtract-vs-divide
    # choice becomes visible instead of being buried inside a label.
    #
    # Each estimator returns the BASELINE itself, aligned to df and with NaNs put
    # back where the input had them. `detrend()` then subtracts it or divides by it.

    #: Estimators the UI offers. "Rolling Hilbert" is deliberately absent — it is a
    #: compound (rolling-mean baseline, then envelope division), not a baseline.
    BASELINE_METHODS = ("Linear", "Cubic", "Rolling mean", "LOESS",
                        "Exponential fit")

    #: Estimators that need the window slider, in hours.
    WINDOWED_METHODS = ("Rolling mean", "Rolling Hilbert")

    # NOTE: "Running median" and "Sinc low-pass" were offered briefly in v0.7.6 and
    # removed in v0.7.7 after `detrend_redundancy.py` measured what each was worth.
    #
    # Sinc was the most redundant control in the app: the residual it produced
    # correlated with the rolling mean's at r = 0.994 over 24 real wells, it was
    # never better than the rolling mean on a stepped or artefact-hit baseline, and
    # it cost 40x the time for that answer.
    #
    # The running median was the worst estimator in all five stress cases — 3.5-5.6%
    # interior error against a known baseline where the rolling mean gives 0.3-4.7%
    # — because a median only ever returns a value the data actually took, so on an
    # asymmetric waveform it is biased and it steps. Its one real advantage was
    # resistance to a mid-run artefact (a 6 h excursion moves its baseline by 0.3%
    # against the rolling mean's 16.8%), and LOESS gets most of that (2.8%) without
    # paying the bias. The median was spending 5% everywhere to save 3% occasionally.
    #
    # STL was measured at the same time and deliberately NOT added: it is the most
    # accurate estimator tested, but it costs 9 s per 96-well plate (60 s for the
    # robust variant) against 0.02 s for the rolling mean, and it needs the period
    # as an integer number of samples — the quantity the user is trying to measure.

    @staticmethod
    def polynomial_baseline(df, cols, t_col, order=1):
        """
        Least-squares polynomial of the given order, evaluated at every timepoint.

        Two differences from the previous `signal.detrend` / `statsmodels.detrend`
        implementations, both of which mattered on real recordings:

        * The fit uses the TIME column, not the sample index. The old versions
          assumed evenly spaced samples, so on an irregularly sampled or gap-filled
          recording the trend they removed was not the trend in the data.
        * NaNs are dropped from the fit rather than poisoning it. The polynomial is
          estimated from the finite points and evaluated everywhere.

        `order=1` is the "Linear" option, `order=3` the "Cubic" one.
        """
        frame = pd.DataFrame(df[cols]).astype(float)
        t = np.asarray(df[t_col], dtype=float)

        # Centre and scale time before fitting. Raw hours over a 6-day recording
        # give a badly conditioned Vandermonde matrix at order 3 and numpy warns.
        spread = np.nanstd(t)
        tn = (t - np.nanmean(t)) / (spread if spread > 0 else 1.0)

        out = {}
        for col in frame.columns:
            y = frame[col].to_numpy(dtype=float)
            good = np.isfinite(y) & np.isfinite(tn)
            if good.sum() <= order + 1:
                # Not enough points to identify the polynomial — a flat baseline at
                # the column mean leaves the trace essentially untouched.
                out[col] = np.full_like(y, np.nanmean(y) if good.any() else 0.0)
                continue
            with warnings.catch_warnings():
                # RankWarning moved to np.exceptions in numpy 2.0
                warnings.simplefilter("ignore", getattr(
                    getattr(np, "exceptions", np), "RankWarning", UserWarning))
                coef = np.polyfit(tn[good], y[good], order)
            out[col] = np.polyval(coef, tn)

        return pd.DataFrame(out, index=frame.index, columns=frame.columns)

    @staticmethod
    def rolling_baseline(df, cols, window=10):
        """Centred moving mean. Fast, and the most accurate estimator here in the
        interior of a recording — but see `moving_average_gain` for what the window
        does to amplitude, and `loess_baseline` for the artefact-resistant option."""
        return df[cols].rolling(window=window, center=True, min_periods=1).mean()

    @staticmethod
    def loess_baseline(df, cols, t_col, span_h=48.0):
        """
        Locally weighted regression (LOESS/LOWESS): a straight line fitted through
        the points near each timepoint, weighted by distance and re-weighted to
        discount outliers.

        Two things it does better than the moving average, both measured:

        * **Its amplitude cost barely depends on period.** A centred moving average
          is a perfect notch only at window = period, so at a 24 h window it keeps
          1.156 of a 20 h rhythm and 0.652 of a 28 h one — a 1.8x spread that
          biases every amplitude comparison between samples of different period.
          At a span of twice the period LOESS keeps 1.076 / 1.072 / 1.005 across
          20 / 24 / 28 h. Nearly flat.
        * **An artefact does not spread.** A 6 h excursion moves the rolling-mean
          baseline by 16.8% of the baseline OUTSIDE the artefact; LOESS moves 2.8%.
          That resistance comes from the robustifying iterations, so `it=3` stays.

        `span_h` MUST be comfortably longer than the period. A local line fitted
        over one period follows the oscillation and removes it: at span = period
        only 0.571 of a 24 h rhythm survives. Twice the period is the default.

        `delta` is the speed knob: within that distance lowess interpolates instead
        of fitting. At 1% of the recording it is 6x faster — 0.96 s for a 96-well
        plate against 5.97 s — and the baseline moves by 0.06%.
        """
        filled, nan_mask = methods._fill_for_fit(df[cols])
        t = np.asarray(df[t_col], dtype=float)
        span = float(np.nanmax(t) - np.nanmin(t)) or 1.0
        frac = float(np.clip(float(span_h) / span, 0.05, 1.0))

        out = {}
        for col in filled.columns:
            out[col] = lowess(filled[col].to_numpy(dtype=float), t,
                              frac=frac, it=3, delta=0.01 * span,
                              return_sorted=False)
        return pd.DataFrame(out, index=filled.index,
                            columns=filled.columns).mask(nan_mask)

    @staticmethod
    def loess_gain(span_h, period_h, delta_t, n_points):
        """
        Fraction of a `period_h` oscillation that survives LOESS detrending at this
        span.

        There is no closed form the way there is for a moving average, so this
        measures it: run the filter over a synthetic sine of that period and read
        off what is left. One lowess call on a short series, a few milliseconds —
        cheap enough to show the user the real number instead of a rule of thumb.
        """
        if not all(np.isfinite([span_h, period_h, delta_t])) or min(
                span_h, period_h, delta_t) <= 0:
            return np.nan
        n = int(min(max(n_points, 64), 2000))
        t = np.arange(n) * delta_t
        span = float(t[-1]) or 1.0
        if span <= period_h:
            return np.nan
        y = np.sin(2 * np.pi * t / period_h)
        base = lowess(y, t, frac=float(np.clip(span_h / span, 0.05, 1.0)),
                      it=3, delta=0.01 * span, return_sorted=False)
        mid = slice(n // 4, 3 * n // 4)
        ref = np.ptp(y[mid])
        return float(np.ptp((y - base)[mid]) / ref) if ref else np.nan

    @staticmethod
    def exponential_baseline(df, cols, t_col):
        """
        A + B*exp(-t/tau), the shape substrate consumption actually has.

        A cubic is a fudge for this: it has no asymptote, so it bends back up at
        the end of the recording. The exponential has the lowest edge error of any
        estimator here (5% against the rolling mean's 12%) precisely because its
        shape is right rather than merely flexible.

        Falls back to a linear baseline for any column the fit cannot converge on —
        a flat or noise-only well has no decay to find.
        """
        frame = pd.DataFrame(df[cols]).astype(float)
        t = np.asarray(df[t_col], dtype=float)
        t0 = t - np.nanmin(t)

        def model(x, a, tau, c):
            return a * np.exp(-x / max(tau, 1e-6)) + c

        span = float(np.nanmax(t0)) if np.isfinite(np.nanmax(t0)) else 1.0
        failed, out = [], {}
        for col in frame.columns:
            y = frame[col].to_numpy(dtype=float)
            good = np.isfinite(y) & np.isfinite(t0)
            if good.sum() < 4:
                failed.append(col)
                out[col] = np.full_like(y, np.nanmean(y) if good.any() else 0.0)
                continue
            first, last = float(y[good][0]), float(y[good][-1])
            try:
                with warnings.catch_warnings():
                    # A trace with no decay to find gives an unidentifiable tau and
                    # curve_fit warns about the covariance. The fit still returns a
                    # usable (nearly flat) baseline, which is the right answer here.
                    warnings.simplefilter("ignore")
                    p, _ = curve_fit(
                        model, t0[good], y[good],
                        p0=[first - last, max(span / 2, 1.0), last], maxfev=20000)
                out[col] = model(t0, *p)
            except Exception:
                failed.append(col)
                out[col] = np.polyval(np.polyfit(t0[good], y[good], 1), t0)

        if failed:
            methods._note(
                f"No exponential decay could be fitted to {len(failed)} sample(s) "
                f"({', '.join(map(str, failed[:4]))}"
                f"{'…' if len(failed) > 4 else ''}) — a linear baseline was used "
                "for those instead.")
        return pd.DataFrame(out, index=frame.index, columns=frame.columns)

    @staticmethod
    def estimate_baseline(df, cols, t_col, method, window=10, delta_t=None,
                          span_h=48.0):
        """Dispatch to the requested estimator. Returns None for unknown names."""
        builders = {
            "Linear": lambda: methods.polynomial_baseline(df, cols, t_col, 1),
            "Cubic": lambda: methods.polynomial_baseline(df, cols, t_col, 3),
            "Rolling mean": lambda: methods.rolling_baseline(df, cols, window),
            "LOESS": lambda: methods.loess_baseline(df, cols, t_col, span_h),
            "Exponential fit": lambda: methods.exponential_baseline(df, cols, t_col),
        }
        builder = builders.get(method)
        return builder() if builder else None

    @staticmethod
    def _note(message, level="info"):
        """Report to the UI when there is one, and stay silent in headless use
        (docs/make_figures.py and the verify scripts call detrend outside Streamlit)."""
        try:
            getattr(st, level)(message)
        except Exception:
            pass

    @staticmethod
    def polynomial_detrend(df, cols, t_col, order=1):
        """Subtract a least-squares polynomial. Kept as the name the older call
        sites use; the fit itself now lives in `polynomial_baseline`."""
        return pd.DataFrame(df[cols]).astype(float) - methods.polynomial_baseline(
            df, cols, t_col, order)

    @staticmethod
    def linear_detrend(df, cols, t_col=None):
        if t_col is None:                       # kept for older call sites
            filled, nan_mask = methods._fill_for_fit(df[cols])
            out = pd.DataFrame(signal.detrend(filled.to_numpy(), type='linear', axis=0),
                               index=filled.index, columns=filled.columns)
            return out.mask(nan_mask)
        return methods.polynomial_detrend(df, cols, t_col, order=1)

    @staticmethod
    def cubic_detrend(df, cols, t_col=None):
        if t_col is None:                       # kept for older call sites
            filled, nan_mask = methods._fill_for_fit(df[cols])
            out = pd.DataFrame(np.asarray(detrend(filled, order=3)),
                               index=filled.index, columns=filled.columns)
            return out.mask(nan_mask)
        return methods.polynomial_detrend(df, cols, t_col, order=3)

    @staticmethod
    def rolling_mean(df, cols, window=10):
        return df[cols] - df[cols].rolling(window=window, center=True, min_periods=1).mean()

    # NOTE: `hilbert_rolling_mean` ("Hilbert + Rolling mean") was removed in v0.7.2.
    # Once the Stage-1 fixes centred rolling() and corrected the Hilbert axis, it
    # became bit-for-bit identical to envelope_rolling() ("Rolling Hilbert"):
    # centred rolling-mean baseline subtraction, then division by the Hilbert
    # envelope along time. Two names for one algorithm. "Rolling Hilbert" is kept.

    @staticmethod
    def moving_average_gain(window_pts, period_h, delta_t):
        """
        Fraction of a `period_h` oscillation that SURVIVES rolling-mean detrending
        with a centred window of `window_pts` samples.

        A centred moving average of N samples has frequency response
        H = sin(pi*N*dt/T) / (N*sin(pi*dt/T)), and subtracting it leaves 1 - H.
        H is exactly zero when the window equals the period, so the ideal window
        for a 24 h rhythm is 24 h: gain 1.000, nothing lost.

        Away from that, the cost is real and asymmetric. At the app's old fixed
        20 h default: T=20 h keeps 1.000, T=24 h keeps 0.809, T=28 h keeps 0.652 —
        so a short-period genotype came out of preprocessing with half again as
        much apparent amplitude as a long-period one. Windows LONGER than the
        period overshoot instead (30 h window on a 24 h rhythm: 1.180), because H
        goes negative and the subtraction adds a phase-inverted copy back in.

        Returned so the UI can show the number rather than leave it implicit.
        """
        if not np.isfinite(delta_t) or delta_t <= 0 or not np.isfinite(period_h) or period_h <= 0:
            return np.nan
        n = int(round(window_pts))
        if n < 1:
            return np.nan
        denom = n * np.sin(np.pi * delta_t / period_h)
        if abs(denom) < 1e-12:
            return np.nan
        h = np.sin(np.pi * n * delta_t / period_h) / denom
        return float(1.0 - h)

    @staticmethod
    def estimate_envelope(y, win=None):
        """
        Divide by the amplitude envelope, flattening a damped rhythm to constant
        amplitude.

        `signal.hilbert` returns the ANALYTIC MODULUS, which is only a smooth
        envelope for a narrowband signal. On a real noisy trace it wobbles
        cycle-to-cycle, and dividing by it inflates exactly the low-amplitude
        stretches — the troughs and the tail of a damping run — where the noise
        lives. Smoothing the modulus over roughly one cycle before dividing gives
        the envelope the shape it is supposed to have.

        `win` is the smoothing window in samples; pass the same window used for the
        rolling-mean step. None keeps the old raw-modulus behaviour.
        """
        frame = pd.DataFrame(y)
        filled, nan_mask = methods._fill_for_fit(frame)

        env = pd.DataFrame(np.abs(signal.hilbert(filled.to_numpy(dtype=float), axis=0)),
                           index=filled.index, columns=filled.columns)
        if win is not None and int(win) > 1:
            env = env.rolling(int(win), center=True, min_periods=1).mean()

        # A near-zero envelope means there is nothing to normalise there; NaN is
        # honest, a huge quotient is not.
        env = env.where(env > 1e-12)
        out = filled / env
        return out.mask(nan_mask)

    @staticmethod
    def rolling(y, win=20):
        # centred, to match rolling_mean() — an uncentred window shifts the phase
        # of the detrended trace by half the window.
        rol = y.rolling(win, center=True, min_periods=1).mean()
        return y - rol

    @staticmethod
    def envelope_rolling(y, win=20):
        y = methods.rolling(y, win=win)
        y = methods.estimate_envelope(y, win=win)
        return y

    # NOTE: Butterworth band-pass smoothing (`butter_bandpass_filter` / `apply_butter`)
    # was removed in v0.7.2. Its 18-30 h band was hardcoded and ignored the period
    # slider, and the band frequently landed above Nyquist for coarsely sampled
    # recordings, where scipy raises "critical frequencies must be 0 < Wn < 1".
    # Savitzky-Golay below covers the same need without the failure mode.

    # ── Savitzky-Golay smoothing ────────────────────────────────────────────────

    @staticmethod
    def savgol_window(window_h, delta_t, polyorder, n_points):
        """
        Translate a smoothing window expressed in HOURS into a valid
        `scipy.signal.savgol_filter` window_length (in samples).

        savgol_filter requires window_length to be odd, strictly greater than
        polyorder, and (with mode='interp') no longer than the series itself.
        Rather than letting any of those raise, we clamp and report what we did.

        Returns
        -------
        (n, polyorder, note) : the window length actually usable in samples, the
        polyorder actually usable, and a human-readable note (empty string when
        the requested window was used unchanged). `n` is None when the series is
        too short to smooth at all.
        """
        if not np.isfinite(delta_t) or delta_t <= 0:
            return None, polyorder, "Sampling interval is unknown — smoothing skipped."

        requested = int(round(window_h / delta_t))
        n = requested if requested % 2 == 1 else requested + 1

        # Largest usable odd window: cannot exceed the number of timepoints
        max_n = n_points if n_points % 2 == 1 else n_points - 1
        # Smallest odd window strictly greater than polyorder
        min_n = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2

        if max_n < 3:
            return None, polyorder, (
                f"Only {n_points} timepoints — too few to smooth. Smoothing skipped."
            )

        # If polyorder cannot fit in the longest available window, lower it
        if min_n > max_n:
            polyorder = max_n - 1
            min_n = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2

        note = ""
        if n < min_n:
            note = (
                f"A {window_h:g} h window is only {max(requested, 1)} sample(s) at "
                f"{delta_t:.2f} h sampling — too short for a degree-{polyorder} fit. "
                f"Using {min_n} points ({min_n * delta_t:.1f} h) instead."
            )
            n = min_n
        elif n > max_n:
            note = (
                f"A {window_h:g} h window is longer than the recording. "
                f"Using {max_n} points ({max_n * delta_t:.1f} h) instead."
            )
            n = max_n

        return n, polyorder, note

    @staticmethod
    def savitzky_golay(df, cols, delta_t, window_h=6.0, polyorder=2):
        """
        Savitzky-Golay smoothing with the window specified in HOURS.

        A local least-squares polynomial fit. Unlike a moving average it preserves
        peak height and width, which matters here because amplitude and waveform
        shape are downstream features.

        Why hours and not samples: the same number of samples means very different
        things at 1-min and 1-h sampling. Expressed in hours the filter has the same
        effect on the biology regardless of how densely the run was recorded.

        Why 6 h by default: measured gain on a 24 h rhythm is 0.998 (0.2% amplitude
        loss) and 0.975 at the 12 h harmonic that carries waveform asymmetry, while
        white noise drops to ~42% RMS at 30-min sampling and everything at 4 h and
        below is suppressed below 0.15. A 12 h window would cut the 12 h harmonic to
        0.73 and visibly round the waveform.

        Returns (smoothed_df, note).
        """
        values = df[cols].to_numpy(dtype=float)
        n, polyorder, note = methods.savgol_window(
            window_h, delta_t, polyorder, values.shape[0]
        )
        if n is None:
            return df[cols], note

        # savgol_filter has no NaN handling — a single gap would otherwise poison a
        # whole window. Interpolate for the fit, then put the gaps back.
        frame = pd.DataFrame(values, index=df.index, columns=cols)
        nan_mask = frame.isna()
        if nan_mask.to_numpy().any():
            frame = frame.interpolate(limit_direction="both")
            frame = frame.fillna(0.0)   # columns that are entirely NaN

        smoothed = signal.savgol_filter(
            frame.to_numpy(dtype=float),
            window_length=n,
            polyorder=polyorder,
            axis=0,
            mode="interp",
        )
        out = pd.DataFrame(smoothed, index=df.index, columns=cols)
        return out.mask(nan_mask), note

    @staticmethod
    def dct_period_filter(signal, dt, min_period=6):

        N = len(signal)
        coeff = dct(signal, norm='ortho')

        freqs = np.arange(N) / (2*N*dt)
        periods = 1 / (freqs + 1e-10)

        coeff[periods < min_period] = 0

        filtered = idct(coeff, norm='ortho')
        
        return filtered

    @staticmethod
    def remove_baseline(df, cols, baseline, removal="Subtract"):
        """
        Take the baseline out of the signal, either additively or multiplicatively.

        Subtract  ->  y - baseline          (residual, in the original units)
        Divide    ->  y / baseline - 1      (relative deviation, i.e. dF/F)

        Why divide at all: for bioluminescence and fluorescence the baseline is a
        MULTIPLYING factor, not an added offset — substrate depletion, cell number
        and bleaching scale the whole signal, oscillation included. Subtracting
        leaves the residual still multiplied by a decaying baseline, so the rhythm
        looks like it is damping faster than it is. On tutorial 2, where the
        generator plants damping with tau in 120-165 h, every subtractive method
        reports 89-94 h; dividing gives 161 h and recovers the relative amplitude
        to within 5% of the planted value.

        The `- 1` matters: it puts the output back on a zero centre, so everything
        downstream (plots, normalisation, the amplitude features, the QC flags)
        sees the same shape of trace it sees after subtraction, in units of
        fractional deviation from baseline.

        Division is refused, with a message, when the baseline is not safely
        positive — on centred, z-scored or background-subtracted data the baseline
        passes through zero and the quotient explodes. In that case the subtractive
        result is returned instead, so the app keeps working.
        """
        signal_df = pd.DataFrame(df[cols]).astype(float)
        if removal != "Divide":
            return signal_df - baseline

        b = pd.DataFrame(baseline).astype(float)
        scale = float(np.nanmedian(np.abs(b.to_numpy())))
        floor = max(scale * 1e-3, 1e-12)
        unsafe = b.columns[(b.min(skipna=True) <= floor).to_numpy()].tolist()
        if unsafe:
            methods._note(
                f"Divide is only meaningful where the baseline stays positive, and "
                f"it reaches zero or below in {len(unsafe)} sample(s) "
                f"({', '.join(map(str, unsafe[:4]))}{'…' if len(unsafe) > 4 else ''}). "
                "That normally means the data has already been centred or "
                "background-subtracted. Subtracting the baseline instead.",
                level="warning")
            return signal_df - baseline

        return signal_df / b - 1.0

    @staticmethod
    def detrend(df, cols, t_col, method='None', period_range=None,
                removal='Subtract'):
        """
        `period_range` is the (min, max) of the sidebar Period range slider. The
        rolling window defaults to the middle of that band, because a centred
        moving average is a perfect notch only when the window equals the period —
        see `moving_average_gain` for the numbers. The window used to default to a
        flat ~20 h regardless of the biology, which quietly scaled amplitudes by
        anything between 0.65 and 1.2 depending on a sample's period.

        `removal` picks how the baseline comes out — see `remove_baseline`. It
        defaults to 'Subtract', so every existing call site (the verify scripts,
        docs/make_figures.py, tutorials/verify_tutorial_data.py) keeps the exact
        behaviour it had.
        """
        if method == 'None':
            return df[cols]

        delta_t = float(df[t_col].diff().mean())
        if not np.isfinite(delta_t) or delta_t <= 0:
            delta_t = 1.0

        # Target window: the middle of the period band the user is searching in.
        target_h = float(np.mean(period_range)) if period_range else 24.0
        span_h = float(df[t_col].max() - df[t_col].min())

        win = 10
        if method in methods.WINDOWED_METHODS:
            lo_h = max(2 * delta_t, round(0.5 * target_h, 1))
            hi_h = min(max(3 * target_h, lo_h + delta_t), max(span_h, lo_h + delta_t))
            default_h = float(np.clip(target_h, lo_h, hi_h))
            # Static label so docs.attach() can key a tooltip to it — the old label
            # interpolated the suggested window and so could never be documented.
            win_h = st.slider(
                "Detrending window (h)",
                float(lo_h), float(hi_h), default_h, step=float(max(delta_t, 0.1)),
            )
            win = max(1, int(round(win_h / delta_t)))

            # Say out loud what this window does to the amplitude at both ends of
            # the period band, so a biased comparison cannot happen silently.
            lo_p, hi_p = (period_range if period_range else (target_h, target_h))
            g_lo = methods.moving_average_gain(win, lo_p, delta_t)
            g_hi = methods.moving_average_gain(win, hi_p, delta_t)
            if np.isfinite(g_lo) and np.isfinite(g_hi):
                methods._note(
                    f"Window {win_h:g} h ({win} points) keeps "
                    f"{g_lo:.0%} of a {lo_p:g} h rhythm and {g_hi:.0%} of a {hi_p:g} h one.",
                    level="caption",
                )
                if max(abs(g_lo - 1), abs(g_hi - 1)) > 0.15:
                    methods._note(
                        "Across your period range this window scales amplitude by "
                        f"{min(g_lo, g_hi):.0%}-{max(g_lo, g_hi):.0%}. That is fine if "
                        "the range is just a generous search band and your samples all "
                        "sit near {0:g} h — but if their periods genuinely differ, "
                        "amplitude comparisons between them carry this bias. Period and "
                        "phase are unaffected.".format(target_h),
                        level="warning",
                    )

        # LOESS takes a SPAN, not a window, and the two are not interchangeable: a
        # local line fitted over one period follows the oscillation and removes it
        # (span = period leaves only 0.571 of a 24 h rhythm). Twice the period is
        # the default, where the amplitude cost is both small and — unlike the
        # moving average — nearly independent of the period.
        loess_span = 2 * target_h
        if method == 'LOESS':
            lo_s = max(2 * delta_t, round(1.5 * target_h, 1))
            hi_s = max(min(5 * target_h, max(span_h, lo_s + 1)), lo_s + 1)
            loess_span = st.slider(
                "LOESS span (h)",
                float(lo_s), float(hi_s), float(np.clip(2 * target_h, lo_s, hi_s)),
                step=1.0,
            )
            lo_p, hi_p = (period_range if period_range else (target_h, target_h))
            g_lo = methods.loess_gain(loess_span, lo_p, delta_t, len(df))
            g_hi = methods.loess_gain(loess_span, hi_p, delta_t, len(df))
            if np.isfinite(g_lo) and np.isfinite(g_hi):
                methods._note(
                    f"Span {loess_span:g} h keeps {g_lo:.0%} of a {lo_p:g} h rhythm "
                    f"and {g_hi:.0%} of a {hi_p:g} h one.",
                    level="caption",
                )
                if min(g_lo, g_hi) < 0.9:
                    methods._note(
                        f"A {loess_span:g} h span is close to the period you are "
                        "measuring, so the local fit is following the rhythm and "
                        f"removing it — only {min(g_lo, g_hi):.0%} survives. Widen the "
                        f"span to about {2 * target_h:g} h.",
                        level="warning",
                    )

        # "Rolling Hilbert" stays a compound of its own: rolling-mean baseline plus
        # envelope division. `removal` does not apply — it already divides, by the
        # envelope rather than by the baseline.
        if method == 'Rolling Hilbert':
            return methods.envelope_rolling(df[cols], win)

        baseline = methods.estimate_baseline(
            df, cols, t_col, method, window=win, delta_t=delta_t, span_h=loess_span)
        if baseline is None:
            return df[cols]

        return methods.remove_baseline(df, cols, baseline, removal)

    @staticmethod
    def min_max(df, cols, mode='all'):
        if mode == 'all':
            global_min = df[cols].min().min()
            global_max = df[cols].max().max()
            return (df[cols] - global_min) / (global_max - global_min) * 100
        return (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min()) * 100

    @staticmethod
    def z_score(df, cols):
        std = df[cols].std()
        # Replace 0 standard deviations with 1 to avoid NaN (the numerator will be 0 anyway)
        std = std.replace(0, 1)
        return (df[cols] - df[cols].mean()) / std

    @staticmethod
    def normalize(df, cols, method='None'):
        if method == 'None':
            return df[cols]
        
        methods_map = {
            'Sample-wise Min-Max': lambda: methods.min_max(df, cols, mode='sample'),
            'Global Min-Max': lambda: methods.min_max(df, cols, mode='all'),
            'Z-Score': lambda: methods.z_score(df, cols)
        }
        
        # Retrieve the lambda function from the dictionary
        selected_method = methods_map.get(method)
        
        if selected_method:
            return selected_method() # Execute it only here
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def autocovariance(x):
        x = x - np.mean(x)
        n = len(x)
        if n == 0:
            raise ValueError("Signal is empty")
        fft_size = 2 * n
        f = np.fft.rfft(x, n=fft_size)
        acov = np.fft.irfft(f * np.conj(f))[:n]
        return acov / n

    def autocorrelation(x):
        acov = methods.autocovariance(x)
        if acov[0] == 0:
            raise ValueError("Signal has zero variance")
        return acov / acov[0]

    def period_correlation(data, delta_t, min_period=None, max_period=None, threshold=0.2):
        """
        Dominant period from the autocorrelation function, **in hours**.

        Parameters
        ----------
        data       : array-like signal
        delta_t    : sampling interval in hours (required — the ACF peak is a lag
                     index, and without delta_t the result is in samples, not hours)
        min_period : lower bound of the search window in hours (optional)
        max_period : upper bound of the search window in hours (optional)
        threshold  : minimum ACF height for a peak to count
        """
        if not np.isfinite(delta_t) or delta_t <= 0:
            return np.nan

        ac = methods.autocorrelation(data)
        n = len(data)

        # Translate the period window (hours) into a lag window (samples)
        min_lag = 2 if min_period is None else max(2, int(np.floor(min_period / delta_t)))
        hard_max = n // 2
        max_lag = hard_max if max_period is None else min(hard_max, int(np.ceil(max_period / delta_t)))

        if max_lag <= min_lag:
            return np.nan

        peaks, props = signal.find_peaks(ac[min_lag:max_lag + 1], height=threshold)
        if len(peaks) == 0:
            return np.nan

        # Adjust indices back to original ac array, then convert lag -> hours
        peaks = peaks + min_lag
        best_lag = peaks[np.argmax(props['peak_heights'])]
        return float(best_lag * delta_t)
    
    def fft_period_old(signal, t):
        # Perform FFT
        freq_vals = np.fft.fftfreq(len(t))
        power = np.abs(np.fft.fft(signal))**2
    
        # Only view the positive frequencies
        positive_indices = freq_vals > 0
        freq_vals = freq_vals[positive_indices]
        power = power[positive_indices]
    
        # Find peak in power spectrum
        peak_indices = np.argsort(power)[::-1]
        peak_frequency = freq_vals[peak_indices[0]]
        peak_period = 1/peak_frequency
        #print(f"Peak period (FFT): {peak_period}")
        return peak_period

    def fft_period(signal, t, snr_threshold=5.0):
        """
        Estimate the dominant period of a signal using FFT.
        
        Parameters
        ----------
        signal : array-like, real-valued
        t : array-like, uniformly spaced time points
        snr_threshold : minimum peak-to-mean power ratio to accept result
        
        Returns
        -------
        float : estimated period in same units as t, or None if no clear peak
        """
        dt_vals = np.diff(t)
        assert np.allclose(dt_vals, dt_vals[0], rtol=1e-3), "Time array must be uniformly sampled"
        dt = dt_vals[0]

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(signal))
        windowed = signal * window

        # FFT on real signal — only positive frequencies returned
        power = np.abs(np.fft.rfft(windowed))**2
        freq_vals = np.fft.rfftfreq(len(signal), d=dt)

        # Exclude DC component
        freq_vals = freq_vals[1:]
        power = power[1:]

        peak_idx = np.argmax(power)
        snr = power[peak_idx] / np.mean(power)
        if snr < snr_threshold:
            return np.nan  # No significant periodic component found

        return 1.0 / freq_vals[peak_idx]
    
    @staticmethod
    def damped_cosinor_fit(y, t, min_period=18, max_period=36):
        """
        Fit  A * exp(-t/tau) * cos(2*pi*t/T + phi) + C  and return the parameters
        with their standard errors.

        Every other period method here is non-parametric: they look for where the
        energy sits and read a period off the peak. That works without assuming a
        shape, and it pays for it — over a 96 h window an FFT's bins are ~1.4 h
        apart near 24 h, and Lomb-Scargle's peak is broadened by the very damping
        that a decaying rhythm has. Writing the damping into the model instead of
        fighting it recovers the planted period on the tutorial dataset to 0.08 h
        against 0.28 h for every method above, and returns a standard error, which
        none of them do.

        The cost is the assumption: this is a single damped sinusoid, so a strongly
        non-sinusoidal waveform or a rhythm whose period drifts mid-recording will
        fit badly. `r2` is returned so that can be checked rather than assumed.

        Returns a dict; every value is NaN if the fit does not converge.
        """
        fail = {"period": np.nan, "period_se": np.nan, "amplitude": np.nan,
                "amplitude_se": np.nan, "damping_tau": np.nan,
                "damping_tau_se": np.nan, "acrophase_h": np.nan,
                "mesor": np.nan, "r2": np.nan}

        y = np.asarray(y, dtype=float)
        t = np.asarray(t, dtype=float)
        good = np.isfinite(y) & np.isfinite(t)
        if good.sum() < 8:
            return fail
        y, t = y[good], t[good]
        t0 = t - t.min()
        span = float(t0.max())
        if span <= 0:
            return fail

        def model(x, A, tau, T, phi, C):
            return A * np.exp(-x / max(tau, 1e-6)) * np.cos(
                2 * np.pi * x / max(T, 1e-6) + phi) + C

        # Start from the middle of the period band, an undamped guess, and the
        # observed half-range. tau starts at the recording length: "not obviously
        # damping" is the neutral prior, and the fit moves off it easily.
        T0 = float(np.clip(0.5 * (min_period + max_period), min_period, max_period))
        p0 = [(np.nanmax(y) - np.nanmin(y)) / 2 or 1.0, span, T0, 0.0, float(np.nanmean(y))]
        bounds = ([0, 1e-3, float(min_period), -2 * np.pi, -np.inf],
                  [np.inf, 1e6, float(max_period), 2 * np.pi, np.inf])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(model, t0, y, p0=p0, bounds=bounds,
                                       maxfev=40000)
        except Exception:
            return fail

        A, tau, T, phi, C = popt
        se = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.full(5, np.nan)
        resid = y - model(t0, *popt)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan

        # Peak time within the first cycle, to match cosinor_acrophase_h.
        acro_h = float(((-phi) % (2 * np.pi)) / (2 * np.pi) * T)

        return {"period": float(T), "period_se": float(se[2]),
                "amplitude": float(abs(A)), "amplitude_se": float(se[0]),
                "damping_tau": float(tau), "damping_tau_se": float(se[1]),
                "acrophase_h": acro_h, "mesor": float(C), "r2": r2}

    @staticmethod
    def damped_cosinor_period(y, t, min_period=18, max_period=36):
        """Period only, for `period_estimation`."""
        return methods.damped_cosinor_fit(y, t, min_period, max_period)["period"]

    def Lomb_Scargle(signal, t, min_period, max_period):
        
        frequency, power = LombScargle(t, signal).autopower(minimum_frequency=1/max_period,
                                                                maximum_frequency=1/min_period)
        peak = np.argmax(power)
        peak_frequency = frequency[peak]
        peak_period = 1/peak_frequency
            
        return peak_period
    
    def wavelet(data, t_col, min_period, max_period):
        periods = np.linspace(min_period, max_period, 100)
        dt = np.mean(np.diff(t_col))  # assumes sorted time
        
        wAn = WAnalyzer(periods, dt, p_max=20)
    
        # do_plot defaults to True in pyboat, so this drew — and leaked — one
        # matplotlib figure per sample per Streamlit rerun. Wavelet Transform is the
        # default period method, so a 96-well plate leaked 96 figures every time any
        # widget was touched. We only want the numbers here; the Wavelet Ridge view
        # draws its own figure separately.
        wAn.compute_spectrum(data, do_plot=False)

        # fs is a sampling FREQUENCY (samples per hour), i.e. 1/dt — not dt.
        f, Pxx = signal.periodogram((data - data.mean()) / data.std(),
                                    fs=1 / dt)
        max_power = Pxx.max()
        dominant_freq = f[Pxx.argmax()]

        suggested = int(1 / t_col.diff().mean() * 10) * 4
        #thresh = max_power * dominant_freq
        #st.write(thresh, suggested)
        wAn.get_maxRidge(power_thresh = max_power * dominant_freq, 
                         smoothing_wsize=suggested)
        
        if wAn.ridge_data['power'].sum() == 0:
            return np.nan  # or some default
        return np.average(
            wAn.ridge_data['periods'],
            weights=wAn.ridge_data['power']
        )
        #wAn.get_maxRidge(power_thresh = 10, smoothing_wsize=20)
        #return np.average(wAn.ridge_data['periods'], weights=wAn.ridge_data['power'])  # this is a pandas DataFrame holding the ridge results
    
    @staticmethod
    def sampling_interval(t):
        """
        Mean sampling interval, in whatever unit `t` is in.

        Derived from the SORTED UNIQUE timepoints. Do not use
        `df[t_col].value_counts().index` for this — value_counts sorts by
        frequency, not by time, so with uneven replicate counts (or simply
        unsorted rows) np.diff over that index is meaningless and can even
        come out negative.
        """
        u = np.sort(pd.unique(np.asarray(t, dtype=float)))
        if len(u) < 2:
            return np.nan
        return float(np.mean(np.diff(u)))

    @staticmethod
    def period_estimation(df, cols, t_col, method='None', min_period=18, max_period=36):
        if method == 'None':
            return 'No period estimation'

        delta_t = methods.sampling_interval(df[t_col].values)

        methods_map = {
            'Fast Fourier Transform (FFT)': lambda: df[cols].apply(lambda x: methods.fft_period(x, df[t_col].values)),
            'Lomb-Scargle Periodogram':    lambda: df[cols].apply(lambda x: methods.Lomb_Scargle(x, df[t_col].values, min_period, max_period)),
            'Autocorrelation':             lambda: df[cols].apply(lambda x: methods.period_correlation(x, delta_t, min_period, max_period)),
            'Wavelet Transform':           lambda: df[cols].apply(lambda x: methods.wavelet(x, df[t_col], min_period, max_period)),
            'Damped Cosinor':              lambda: df[cols].apply(lambda x: methods.damped_cosinor_period(x, df[t_col].values, min_period, max_period)),
        }

        # Get the selected method and execute it if exists, otherwise return the original df[cols]
        return methods_map.get(method, lambda: df[cols])()

    
    def phase_time_arranger(df, t_col, T=24):
        df['norm_time'] = df[t_col].astype(int)
        df['norm_time'] = df['norm_time'] - df['norm_time'].min()
    
        df['norm_day'] = df.norm_time / T - np.trunc(df.norm_time / T)
        df['norm_day'] = df.norm_day * T
        df['norm_day'] = df['norm_day'].astype(int)
        return df
    
    def find_phase(df, p_col, T, delta_t):
        
        d = T/delta_t * 0.5
        peaks = signal.find_peaks(df[p_col], distance=d)[0]
        #peak_hours = df.iloc[peaks]['norm_day'].values
        return peaks
    
    def sine_model(t, A, phi, C, T=24):
        return A * np.sin(2 * np.pi * t / T + phi) + C
    
    def sine_phase(t, data, T=24):
        # Fit the model
        p0 = [np.std(data), 0, np.mean(data)]
    
        # Fit sine curve
        params, _ = curve_fit(lambda t, A, phi, C: methods.sine_model(t, A, phi, C, T),
                          t, data, p0=p0)
        
        # Convert fitted phase to hours
        fitted_signal = methods.sine_model(t, *params)
        
        peaks = signal.find_peaks(fitted_signal)[0]

        peak_hours = np.mean(t[peaks] % 24)
        
        return peak_hours
    
    def phase_calculation(df, t_col, p_col, T=24, delta_t=1):
    
        df = methods.phase_time_arranger(df, t_col, T)
        delta_t = np.mean(np.diff(df[t_col].values))  #
        
        peak_hours = methods.find_phase(df, p_col, T, delta_t)
        
        return peak_hours
    
    def multicomparison(result_df, layout_df, conditions, method, thresh):
        
        sig_comparison = []
        per_comparison = []
        amp_comparison = []
        
        for x, y in combinations(conditions, 2):
            
            compar = f"{x}\n{y}"
            
            on_x = layout_df[layout_df.Condition == x]['name'].unique()
            on_y = layout_df[layout_df.Condition == y]['name'].unique()
            
            sorted_x = result_df[result_df['CycID'].isin(on_x)]
            sorted_y = result_df[result_df['CycID'].isin(on_y)]
            
            cols = [col for col in result_df.columns if method in col]

            per_col = 'Periods'
            q_cols = [col for col in cols if 'BH.Q' in col.upper()]
            if not q_cols:
                continue
            q_col = q_cols[0]
            # Not every method reports an amplitude — Tempo returns a probability,
            # not a cosinor fit — so the amplitude comparison is skipped rather
            # than raising IndexError on a missing column.
            amp_cols = [col for col in cols if 'AMP' in col.upper()]
            amp_col = amp_cols[0] if amp_cols else None
            
            table = []
            for i in [sorted_x, sorted_y]:
                significant = i[i[q_col] <= thresh]
                non_sig = i.shape[0] - significant.shape[0]
                sig = significant.shape[0]
                table.append([sig, non_sig])
            
            odds, p = fisher_exact(table, alternative='two-sided')
            t_stat_per, p_per = ttest_ind(sorted_x[per_col].values,
                                        sorted_y[per_col].values, equal_var=False)  
            sig_comparison.append([x, y, compar, p, p < thresh, ])
            per_comparison.append([x, y, compar, p_per, p_per < thresh])
            if amp_col is not None:
                t_stat_amp, p_amp = ttest_ind(sorted_x[amp_col].values,
                                              sorted_y[amp_col].values, equal_var=False)
                amp_comparison.append([x, y, compar, p_amp, p_amp < thresh])
        
        summary = pd.DataFrame()
        
        for n, d in enumerate([sig_comparison, per_comparison, amp_comparison]):
            if not d:                      # e.g. no amplitude column for this method
                continue
            temp = pd.DataFrame(d, columns=['group1', 'group2', 'comparison', 'p-val', 'reject'])
            temp['tested'] = ['Rhythmicity', 'Period', 'Amplitude'][n]
            summary = pd.concat([summary, temp]).reset_index(drop=True)
            
        return summary

    def make_square_signal(
            delta_time=1.0,
            n_days=5,
            period=24,
            on_ratio=0.5,
            starting_time=0,
            order=0
        ):
            total_hours = n_days * period + starting_time
        
            n_samples = int(total_hours / delta_time) + 1
            time_h = np.linspace(starting_time, total_hours, n_samples)

            on_duration = period * on_ratio
            if order == 0:
                ent_signal = pd.Series(((time_h % period) < on_duration).astype(int))
            else: 
                ent_signal = pd.Series(((time_h % period) > on_duration).astype(int))
            
            return ent_signal[:-1]

    def add_entrainment(df, t_col, n_days=5, period=24, on_ratio=0.5, release=0):
        
        data = df.copy()

        delta_t = methods.sampling_interval(data[t_col].values)

        data['entrainment'] = methods.make_square_signal(delta_t, n_days, period, on_ratio)
        #data['entrainment'] = data['entrainment'].fillna(release)
        
        return data[[t_col, 'entrainment']].dropna()

    def count_entrainment_days(ent_data, delta_t, thresh=0.1):
        """
        Counts entrainment cycles and estimates period T.
        Handles both sinusoidal signals (via Hilbert envelope + zero crossings)
        and binary/step signals (via edge detection).
        
        Parameters
        ----------
        ent_data : array-like
            Entrainment signal (raw values, not time column)
        delta_t : float
            Sampling interval in hours
        thresh : float
            Minimum envelope amplitude to count a cycle (for sinusoidal signals)
        
        Returns
        -------
        cycles : int
            Number of complete entrainment cycles detected
        T : float
            Estimated period in hours
        cutoff_time : float
            Duration of the entrainment segment in hours (cycles × T)
        """
        ent_data = np.asarray(ent_data, dtype=float)
        
        # --- Detect signal type ---
        unique_vals = np.unique(ent_data)
        is_binary = len(unique_vals) <= 2 or (
            np.all((ent_data == 0) | (ent_data == 1)) 
        )
        
        if is_binary:
            # Edge-based detection: count rising edges = number of cycles
            edges = np.where(np.diff(ent_data.astype(int)) > 0)[0]
            
            if len(edges) < 1:
                return 0, 0.0, 0.0
            
            cycles = len(edges)
            
            # Period = average distance between rising edges
            if len(edges) >= 2:
                T = float(np.mean(np.diff(edges)) * delta_t)
            else:
                # Only one cycle: estimate from total signal duration
                T = float(np.sum(ent_data > 0) * delta_t)  # fallback: active duration
            
            cutoff_time = cycles * T
            return cycles, T, cutoff_time

        else:
            # Sinusoidal / continuous signal
            # Mean-center before zero-crossing detection
            ent_centered = ent_data - np.mean(ent_data)
            
            analytic_signal = signal.hilbert(ent_centered)
            envelope = np.abs(analytic_signal)
            
            # Rising zero crossings only (every crossing = half cycle → every 2 = full cycle)
            sign_changes = np.diff(np.sign(ent_centered))
            rising_crosses = np.where(sign_changes > 0)[0]   # neg→pos
            
            if len(rising_crosses) < 2:
                return 0, 0.0, 0.0
            
            lengths = []
            for i in range(len(rising_crosses) - 1):
                idx_start = rising_crosses[i]
                idx_end = rising_crosses[i + 1]
                
                # Only count cycle if envelope is above threshold throughout
                if np.mean(envelope[idx_start:idx_end]) > thresh:
                    lengths.append((idx_end - idx_start) * delta_t)
                else:
                    break  # signal amplitude collapsed, entrainment ended
            
            cycles = len(lengths)
            T = float(np.mean(lengths)) if lengths else 0.0
            cutoff_time = float(np.sum(lengths))  # actual cumulative duration, not cycles×T
            
            return cycles, T, cutoff_time

    @staticmethod
    def run_metacycle(df, t_col, data_cols,
                    cyc_methods=("JTK", "LS", "ARS"),
                    min_per=22, max_per=32,
                    parallelize=True,
                    n_replicates=1):
        """
        Run MetaCycle meta2d on a subset of signals.

        Parameters
        ----------
        df          : DataFrame containing t_col and data_cols
        t_col       : Name of the time column (hourly, integer-rounded)
        data_cols   : List of signal column names to test
        cyc_methods : Tuple of MetaCycle methods to use
        min_per     : Minimum period passed to meta2d
        max_per     : Maximum period passed to meta2d
        parallelize : Whether to use MetaCycle's parallel backend

        Returns
        -------
        pd.DataFrame with meta2d results indexed by CycID,
        or None if MetaCycle fails.
        """
        import tempfile
        import shutil
        import os

        from rpy2.robjects.packages import importr
        from rpy2.robjects.vectors import StrVector
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()

        MetaCycle = importr("MetaCycle")

        # Prepare input: round time, filter to integer hours, transpose
        work = df[[t_col] + list(data_cols)].copy()
        work[t_col] = work[t_col].apply(lambda x: round(x, 1))
        work = work[np.isclose(work[t_col] % 1, 0)]
        #rdf  = work.set_index(t_col).transpose().reset_index()

        if np.mean(n_replicates) > 1:
                rdf = work.groupby(t_col).agg({col:('mean') for col in data_cols}).transpose()
        else:
                rdf = work.set_index(t_col).transpose().reset_index()
            

        input_path = None
        output_dir = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="w"
            ) as tmp:
                rdf.to_csv(tmp.name, sep="\t", index=False)
                input_path = tmp.name

            output_dir = tempfile.mkdtemp()

            meta2dout = MetaCycle.meta2d(
                infile=input_path,
                filestyle="txt",
                timepoints="line1",
                cycMethod=StrVector(list(cyc_methods)),
                minper=min_per,
                maxper=max_per,
                outputFile=False,
                outdir=output_dir,
                parallelize=parallelize,
                outIntegration="onlyIntegration",
            )

            result_df = pandas2ri.rpy2py(meta2dout.rx2("meta"))
            return result_df

        except Exception as e:
            import streamlit as st
            st.error(f"MetaCycle failed: {e}")
            return None

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    #from joblib import Parallel, delayed
    
    # ─────────────────────────────────────────────
    # 0. DataFrame interface
    # ─────────────────────────────────────────────
    
    def from_dataframe(df, time_col="time"):
        """
        Extract time vector, data matrix, and signal names from a DataFrame.
    
        Parameters
        ----------
        df       : pd.DataFrame  — one column is time, the rest are signals
        time_col : str           — name of the time column
    
        Returns
        -------
        t     : np.ndarray, shape (n_timepoints,)
        data  : np.ndarray, shape (n_signals, n_timepoints)
        names : list of str
        """
        t    = df[time_col].values.astype(float)
        cols = [c for c in df.columns if c != time_col]
        data = df[cols].values.T          # (n_signals, n_timepoints)
        return t, data, cols
 
 
    # ─────────────────────────────────────────────
    # 0. DataFrame interface
    # ─────────────────────────────────────────────
    
    def from_dataframe(df, time_col="time"):
        """
        Extract time vector, data matrix, and signal names from a DataFrame.
    
        Parameters
        ----------
        df       : pd.DataFrame  — one column is time, the rest are signals
        time_col : str           — name of the time column
    
        Returns
        -------
        t     : np.ndarray, shape (n_timepoints,)
        data  : np.ndarray, shape (n_signals, n_timepoints)
        names : list of str
        """
        t    = df[time_col].values.astype(float)
        cols = [c for c in df.columns if c != time_col]
        data = df[cols].values.T          # (n_signals, n_timepoints)
        return t, data, cols
    
    
    # ─────────────────────────────────────────────
    # 1. Precompute projection matrices for all periods
    # ─────────────────────────────────────────────
    
    def build_projection_matrices(t, periods):
        """
        Precompute OLS projection matrices for all candidate periods.
    
        For each period T, the cosinor model is:
            y = X @ b + e,  X = [cos(2πt/T), sin(2πt/T), 1]
    
        The OLS solution is b = (XᵀX)⁻¹Xᵀ y.
        We precompute P = (XᵀX)⁻¹Xᵀ  (shape 3 × n_t) once per period so that
        any signal fit reduces to a single matrix-vector multiply: b = P @ y.
    
        Returns
        -------
        Xs : np.ndarray, shape (n_periods, n_t, 3)   — design matrices
        Ps : np.ndarray, shape (n_periods, 3, n_t)   — projection matrices
        """
        periods = np.asarray(periods)
        n_p, n_t = len(periods), len(t)
        omegas   = 2 * np.pi / periods
        phases   = np.outer(omegas, t)                        # (n_p, n_t)
        ones     = np.ones((n_p, n_t))
        Xs       = np.stack([np.cos(phases), np.sin(phases), ones], axis=2)
                                                            # (n_p, n_t, 3)
        Ps = np.empty((n_p, 3, n_t))
        for i in range(n_p):
            X     = Xs[i]                                     # (n_t, 3)
            # pinv, not solve. At a trial period of exactly 2*dt (and its
            # submultiples) the sine regressor is sampled at its zero crossings,
            # so it vanishes, X.T @ X is singular and np.linalg.solve raises
            # LinAlgError. The pseudoinverse equals (X'X)^-1 X' whenever X has
            # full rank and degrades to a least-norm solution when it does not,
            # so the sweep survives periods it cannot resolve instead of dying on
            # them. Cost is negligible: 281 SVDs of a (n_t x 3) matrix.
            Ps[i] = np.linalg.pinv(X)                        # (3, n_t)

        return Xs, Ps
    
    
    # ─────────────────────────────────────────────
    # 2. Grid search for one signal (projection matrices)
    # ─────────────────────────────────────────────
    
    def best_period_vectorized(y, Xs, Ps, periods):
        """
        Find best-fit period for signal y using precomputed projection matrices.
    
        b = P @ y  is a dot product — no solve in the hot path.
    
        Parameters
        ----------
        y       : np.ndarray (n_t,)
        Xs      : np.ndarray (n_periods, n_t, 3)
        Ps      : np.ndarray (n_periods, 3, n_t)
        periods : np.ndarray (n_periods,)
    
        Returns
        -------
        dict with period, amplitude, phase_rad, phase_h, intercept, rss, r2
        """
        ss_tot = np.sum((y - y.mean()) ** 2)
    
        # Coefficients for all periods: B[p] = P[p] @ y  →  (n_periods, 3)
        B = np.einsum("pjt,t->pj", Ps, y)
    
        # Fitted values: y_fit[p,t] = sum_j X[p,t,j] * B[p,j]  →  (n_periods, n_t)
        Y_fit = np.einsum("ptj,pj->pt", Xs, B)
    
        # Residual sum of squares for each period
        resid   = y[np.newaxis, :] - Y_fit            # (n_periods, n_t)
        rss_all = np.einsum("pt,pt->p", resid, resid)  # (n_periods,)
    
        best_i    = int(np.argmin(rss_all))
        b1, b2, c = B[best_i]
        T         = periods[best_i]
        rss       = float(rss_all[best_i])
        r2        = 1 - rss / ss_tot if ss_tot > 0 else 0.0
        amplitude = float(np.sqrt(b1**2 + b2**2))
        phase_rad = float(np.arctan2(-b2, b1))
        phase_h   = (phase_rad % (2 * np.pi)) / (2 * np.pi) * T
    
        return dict(period=T, amplitude=amplitude, phase_rad=phase_rad,
                    phase_h=phase_h, intercept=float(c), rss=rss, r2=r2,
                    beta1=float(b1), beta2=float(b2))
    
    
    # ─────────────────────────────────────────────
    # 2b. Sine sweep — all signals, all periods, one pass
    # ─────────────────────────────────────────────

    @staticmethod
    def sine_sweep(df, t_col, cols, period_min=2.0, period_max=30.0,
                   period_step=0.1):
        """
        Fit a sinusoid at every trial period to every signal, in one batched pass.

        This is the "what periods are in this dataset?" question rather than
        "is this one trace rhythmic?". It reuses the same projection matrices as
        `detect_rhythmicity`, but loops over periods instead of over signals, so
        each iteration is two matrix products covering the whole dataset. A
        transcriptome's worth of columns costs barely more than a plate's.

        Returns
        -------
        results : DataFrame, one row per signal
            sample, period, amplitude, phase_h, r2, rss
            `period` is the best-fitting trial period — the value your histogram
            was built on.
        landscape : DataFrame, one row per trial period
            period, mean_r2, median_r2, frac_best, n_best

            `mean_r2` is the aggregate the per-signal argmax cannot show. A gene
            whose best fit is 24 h may still carry a strong 12 h component; that
            shows up as a secondary bump here but is invisible in a histogram of
            best periods, because each signal contributes exactly one count at
            its winning period.
        """
        cols = list(cols)
        if not cols:
            raise ValueError("No signals to sweep.")

        t = np.asarray(df[t_col].to_numpy(), dtype=float)
        Y = df[cols].to_numpy(dtype=float)                 # (n_t, n_signals)

        finite_t = np.isfinite(t)
        if not finite_t.all():
            t, Y = t[finite_t], Y[finite_t]

        # Gaps are filled with the signal's own mean. A mean-valued point sits on
        # the fitted MESOR, so it neither pulls the fit nor inflates R² the way a
        # zero-fill would; dropping whole rows instead would discard every other
        # signal's data for one missing well.
        n_missing = int((~np.isfinite(Y)).sum())
        if n_missing:
            col_means = np.nanmean(Y, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            Y = np.where(np.isfinite(Y), Y, col_means)

        if len(t) < 4:
            raise ValueError(f"Need at least 4 timepoints to sweep, got {len(t)}.")

        # Nyquist: a period shorter than twice the sampling interval cannot be
        # resolved, whatever the fit reports. Clamp rather than let the sweep
        # produce confident nonsense at the short end.
        dt = float(np.median(np.diff(np.sort(np.unique(t)))))
        nyquist = 2.0 * dt
        clamped = None
        if np.isfinite(nyquist) and period_min < nyquist:
            clamped = (period_min, nyquist)
            period_min = nyquist

        periods = np.arange(period_min, period_max + period_step / 2, period_step)
        if len(periods) < 2:
            raise ValueError("Period range is too narrow for a sweep.")

        Xs, Ps = methods.build_projection_matrices(t, periods)

        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)    # (n_signals,)
        safe_tot = np.where(ss_tot > 0, ss_tot, np.nan)

        n_p, n_s = len(periods), Y.shape[1]
        rss = np.empty((n_p, n_s))
        b1 = np.empty((n_p, n_s))
        b2 = np.empty((n_p, n_s))

        for i in range(n_p):
            B = Ps[i] @ Y                                   # (3, n_signals)
            resid = Y - Xs[i] @ B                            # (n_t, n_signals)
            rss[i] = np.einsum("ts,ts->s", resid, resid)
            b1[i], b2[i] = B[0], B[1]

        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = 1.0 - rss / safe_tot                        # (n_p, n_signals)

        best = np.argmin(rss, axis=0)                        # (n_signals,)
        take = (best, np.arange(n_s))
        best_b1, best_b2 = b1[take], b2[take]
        best_period = periods[best]
        phase_rad = np.arctan2(-best_b2, best_b1)
        phase_h = (phase_rad % (2 * np.pi)) / (2 * np.pi) * best_period

        results = pd.DataFrame({
            "sample": cols,
            "period": best_period,
            "amplitude": np.sqrt(best_b1 ** 2 + best_b2 ** 2),
            "phase_h": phase_h,
            "r2": r2[take],
            "rss": rss[take],
        })
        if n_missing:
            results.attrs["n_missing_filled"] = n_missing
        if clamped:
            results.attrs["nyquist_clamped"] = clamped

        counts = np.bincount(best, minlength=n_p).astype(float)
        # A flat signal has zero total variance, so its R² is NaN by construction
        # at every period. That is the honest answer, and it is expected here —
        # no need to warn about the all-NaN slice it produces.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_r2 = np.nanmean(r2, axis=1)
            median_r2 = np.nanmedian(r2, axis=1)
        landscape = pd.DataFrame({
            "period": periods,
            "mean_r2": mean_r2,
            "median_r2": median_r2,
            "n_best": counts.astype(int),
            "frac_best": counts / max(n_s, 1),
        })
        return results, landscape

    @staticmethod
    def sweep_peaks(landscape, column="mean_r2", n_peaks=5, prominence_frac=0.02):
        """
        Dominant periods in a sweep landscape, strongest first.

        Prominence is scaled to the landscape's own range rather than fixed, so
        the same setting works for a dataset where everything fits well and one
        where nothing does.
        """
        y = np.asarray(landscape[column], dtype=float)
        x = np.asarray(landscape["period"], dtype=float)
        if len(y) < 3 or not np.isfinite(y).any():
            return pd.DataFrame(columns=["period", column, "prominence"])

        span = np.nanmax(y) - np.nanmin(y)
        prom = max(span * prominence_frac, 1e-9)
        idx, props = signal.find_peaks(np.nan_to_num(y, nan=np.nanmin(y)),
                                       prominence=prom)
        if len(idx) == 0:
            return pd.DataFrame(columns=["period", column, "prominence"])

        out = pd.DataFrame({
            "period": x[idx],
            column: y[idx],
            "prominence": props["prominences"],
        }).sort_values("prominence", ascending=False).head(n_peaks)
        return out.reset_index(drop=True)

    # ─────────────────────────────────────────────
    # 3. F-test
    # ─────────────────────────────────────────────
    
    def ftest_pvalue(y, fit):
        """
        Compare rhythmic model (3 params) against intercept-only null.
    
        F = [(RSS_null - RSS_model) / Δdf] / [RSS_model / (n - p)]
        """
        from scipy.stats import f as f_dist
        n         = len(y)
        p         = 3
        rss_null  = np.sum((y - y.mean()) ** 2)
        rss_model = fit["rss"]
        if rss_model == 0:
            return 0.0
        F     = ((rss_null - rss_model) / (p - 1)) / (rss_model / (n - p))
        p_val = 1 - f_dist.cdf(F, dfn=p - 1, dfd=n - p)
        return float(p_val)
    
    
    # ─────────────────────────────────────────────
    # 4. Permutation test — fully batched
    # ─────────────────────────────────────────────
    
    def permutation_pvalue(y, fit, Xs, Ps, periods, n_permutations=1000, seed=42):
        """
        Permutation test with full grid search, fully vectorized across permutations.
    
        Instead of looping over permutations in Python, we:
        1. Generate all shuffled signals at once: Y_perm (n_perm × n_t)
        2. For each period, fit all permutations in one batched matrix multiply:
                B_perm = P @ Y_perm.T  →  (3, n_perm)
            then compute amplitudes for all permutations simultaneously.
        3. Track the best amplitude across periods for each permutation.
    
        This replaces n_permutations × n_periods Python function calls with
        n_periods numpy matrix multiplications.
    
        p-value = fraction of permutations with best amplitude >= observed.
        """
        rng     = np.random.default_rng(seed)
        obs_amp = fit["amplitude"]
        n_t     = len(y)
    
        # Generate all permuted signals at once — (n_perm, n_t)
        Y_perm = np.array([rng.permutation(y) for _ in range(n_permutations)])
    
        # best_amp_perm[k] = best amplitude across all periods for permutation k
        best_amp_perm = np.zeros(n_permutations)
    
        for i in range(len(periods)):
            P  = Ps[i]                              # (3, n_t)
            X  = Xs[i]                              # (n_t, 3)
    
            # Fit all permutations at this period in one call
            B  = P @ Y_perm.T                       # (3, n_perm)
            b1, b2 = B[0], B[1]
            amps = np.sqrt(b1**2 + b2**2)           # (n_perm,)
    
            np.maximum(best_amp_perm, amps, out=best_amp_perm)
    
        return float(np.sum(best_amp_perm >= obs_amp) / n_permutations)
    
    
    # ─────────────────────────────────────────────
    # 5. BH-FDR correction
    # ─────────────────────────────────────────────
    
    def bh_fdr(pvalues):
        """
        Benjamini-Hochberg FDR correction. Returns q-values.
        """
        # NaN-safe. A single NaN p-value otherwise makes EVERY q NaN: argsort puts
        # NaN last, the reversal puts it first, and minimum.accumulate carries it
        # across the whole array. NaNs are held out and returned as NaN.
        pvalues = np.asarray(pvalues, dtype=float)
        out     = np.full(pvalues.shape, np.nan)
        ok      = np.isfinite(pvalues)
        m       = int(ok.sum())
        if m == 0:
            return out

        p_ok    = pvalues[ok]
        order   = np.argsort(p_ok)
        ranks   = np.empty(m, dtype=int)
        ranks[order] = np.arange(1, m + 1)
        qvalues = np.minimum(1.0, p_ok * m / ranks)
        qvalues = np.minimum.accumulate(qvalues[order][::-1])[::-1]
        q_ok    = np.empty(m)
        q_ok[order] = qvalues
        out[ok] = q_ok
        return out
    
    
    # ─────────────────────────────────────────────
    # 6. Per-signal worker (runs in parallel)
    # ─────────────────────────────────────────────
    
    def _process_signal(name, y, Xs, Ps, periods, n_permutations, seed):
        fit     = methods.best_period_vectorized(y, Xs, Ps, periods)
        p_ftest = methods.ftest_pvalue(y, fit)
        p_perm  = methods.permutation_pvalue(y, fit, Xs, Ps, periods, n_permutations, seed)
        return dict(
            CycID    = name,
            period_h  = round(fit["period"], 2),
            amplitude = round(fit["amplitude"], 6),
            phase_h   = round(fit["phase_h"], 3),
            phase_rad = round(fit["phase_rad"], 4),
            r2        = round(fit["r2"], 4),
            p_ftest   = p_ftest,
            p_perm    = p_perm,
        )
    
    
    # ─────────────────────────────────────────────
    # 7. Main entry point
    # ─────────────────────────────────────────────
    
    def detect_rhythmicity(
        t,
        data,
        signal_names=None,
        period_min=20.0,
        period_max=28.0,
        period_step=0.1,
        n_permutations=1000,
        fdr_alpha=0.05,
        n_jobs=-1,
        seed=42,
    ):
        """
        Detect circadian rhythmicity across multiple signals.
    
        Parameters
        ----------
        t              : array-like (n_timepoints,)        time in hours
        data           : array-like (n_signals, n_timepoints)
        signal_names   : list of str, optional
        period_min/max : float     grid-search bounds (hours)
        period_step    : float     grid resolution (hours)
        n_permutations : int       permutations per signal — increase for finer
                                p-value resolution, decrease to save time
        fdr_alpha      : float     BH-FDR threshold for the 'rhythmic' flag
        n_jobs         : int       parallel workers (-1 = all cores)
        seed           : int       base random seed (each signal gets seed+i)
    
        Returns
        -------
        pd.DataFrame — one row per signal, sorted by F-test p-value
        """
        from joblib import Parallel, delayed

        t    = np.asarray(t, dtype=float)
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[np.newaxis, :]
    
        n_signals = data.shape[0]
        if signal_names is None:
            signal_names = [f"signal_{i}" for i in range(n_signals)]
    
        periods = np.arange(period_min, period_max + period_step, period_step)
    
        # Precompute design and projection matrices once — shared across all signals
        # and all permutations
        Xs, Ps = methods.build_projection_matrices(t, periods)
    
        print(f"NakedClock: {n_signals} signals × {len(periods)} periods "
            f"× {n_permutations} permutations  (n_jobs={n_jobs})")
    
        rows = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(methods._process_signal)(
                name, y, Xs, Ps, periods, n_permutations, seed + i
            )
            for i, (name, y) in enumerate(zip(signal_names, data))
        )
    
        df = pd.DataFrame(rows)
        df["q_ftest"]  = methods.bh_fdr(df["p_ftest"].values)
        df["q_perm"]   = methods.bh_fdr(df["p_perm"].values)
        df["reject"] = (df["q_ftest"] < fdr_alpha) & (df["q_perm"] < fdr_alpha)
    
        return df.sort_values("p_ftest").reset_index(drop=True)