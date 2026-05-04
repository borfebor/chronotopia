#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:39:58 2025

@author: borfebor
"""

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from scipy import signal
from scipy.optimize import curve_fit
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from statsmodels.tsa.tsatools import detrend
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

    @staticmethod
    def linear_detrend(df, cols):
        return signal.detrend(df[cols], type='linear')

    @staticmethod
    def cubic_detrend(df, cols):
        return detrend(df[cols], order=3)

    @staticmethod
    def rolling_mean(df, cols, window=10):
        return df[cols] - df[cols].rolling(window=window, center=True, min_periods=1).mean()

    @staticmethod
    def hilbert_rolling_mean(df, cols, window=10):
        baseline = df[cols].rolling(window=window, center=True, min_periods=1).mean()
        detrended = df[cols] - baseline
        N_samples = len(cols)
        envelope = np.abs(signal.hilbert(detrended, N=N_samples))
        return detrended / envelope
    
    @staticmethod
    def estimate_envelope(y):
        # Prefer Hilbert envelope (works even if peaks messy)
        analytic = np.abs(signal.hilbert(y, axis=0))
        return y / analytic

    @staticmethod
    def rolling(y, win=20):
        rol = y.rolling(win, min_periods=1).mean()
        return y - rol
    
    @staticmethod
    def envelope_rolling(y, win=20):
        y = methods.rolling(y, win=20)
        y = methods.estimate_envelope(y)
        return y

    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        """
        Band-pass Butterworth filter:
        - data : array of signal values
        - lowcut / highcut : frequencies (cycles per hour)
        - fs : sampling frequency (samples per hour)
        """
        nyq = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.filtfilt(b, a, data)  # zero-phase filtering
        return y

    def apply_butter(data, data_cols, fs, lowcut=30, highcut=18):

        filtered = np.apply_along_axis(
                    methods.butter_bandpass_filter,
                    axis=0,
                    arr=data[data_cols],
                    lowcut=1/lowcut,
                    highcut=1/highcut,
                    fs=fs,
                    order=4
                    )
        return pd.DataFrame(filtered, columns=data_cols)

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
    def detrend(df, cols, t_col, method='None'):
        if method == 'None':
            return df[cols]

        suggested = int(10 / df[t_col].diff().mean())
        
        mean_interval = df[t_col].diff().mean()
        window_hours = int(mean_interval*suggested*2)
        win = st.slider(f"Window size (suggested = {window_hours} h | {suggested*2} points)", int(suggested), 
                        int(suggested * 4), int(suggested*2)) if 'Rolling' in method else 10

        methods_map = {
            'Linear': methods.linear_detrend(df, cols),
            'Rolling mean': methods.rolling_mean(df, cols, win),
            'Hilbert + Rolling mean': methods.hilbert_rolling_mean(df, cols, win),
            'Rolling Hilbert': methods.envelope_rolling(df[cols], win),
            'Cubic': methods.cubic_detrend(df, cols),
        }
        return methods_map.get(method, df[cols])

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

    def period_correlation(data, min_lag=2, max_lag=None, threshold=0.2):
        ac = methods.autocorrelation(data)
        if max_lag is None:
            max_lag = len(data) // 2

        peaks, props = signal.find_peaks(ac[min_lag:max_lag+1], height=threshold)
        if len(peaks) == 0:
            return np.nan  # or raise, depending on your use case

        # Adjust indices back to original ac array
        peaks += min_lag
        return peaks[np.argmax(props['peak_heights'])]
    
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
    
        wAn.compute_spectrum(data)
        
        f, Pxx = signal.periodogram((data - data.mean()) / data.std(), 
                                    fs=dt)
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
    def period_estimation(df, cols, t_col, method='None', min_period=18, max_period=36):
        if method == 'None':
            return 'No period estimation'
    
        methods_map = {
            'Fast Fourier Transform (FFT)': lambda: df[cols].apply(lambda x: methods.fft_period(x, df[t_col].values)),
            'Lomb-Scargle Periodogram':    lambda: df[cols].apply(lambda x: methods.Lomb_Scargle(x, df[t_col].values, min_period, max_period)),
            'Autocorrelation':             lambda: df[cols].apply(lambda x: methods.period_correlation(x)),
            'Wavelet Transform':           lambda: df[cols].apply(lambda x: methods.wavelet(x, df[t_col], min_period, max_period))
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
    
    def phase_plot(ent, ax, peaks, group='norm_day', pal=['#EBEBEB', '#FFFFFF'], order=0):

        if order == 0:
            pal = pal[::-1]
        # Simulated data
        peak_hours = peaks#ent.iloc[peaks][group].values  # 100 genes
        angles = 2 * np.pi * peak_hours / 24
    
        #pal = sns.color_palette('vlag', 5).as_hex()
    
        # Histogram
        num_bins = 24
        bins = np.linspace(0, 2 * np.pi, num_bins + 1)
        counts, _ = np.histogram(angles, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
    
        # === Step 1: Draw background half-sectors ===
        # Fill from 0 to π (e.g., "night")
        ax.bar(x=np.linspace(0, np.pi, 100), height=[max(counts)*1.2]*100,
               width=np.pi/100, bottom=0, color=pal[0], alpha=1, edgecolor='none', zorder=-10)
    
        # Fill from π to 2π (e.g., "day")
        ax.bar(x=np.linspace(np.pi, 2*np.pi, 100), height=[max(counts)*1.2]*100,
               width=np.pi/100, bottom=0, color=pal[1], alpha=1, edgecolor='none', zorder=-10)
    
        # === Step 2: Draw actual data bars ===
        bars = ax.bar(bin_centers, counts, width=2*np.pi/num_bins, bottom=0.0,
                      align='center', alpha=1, color='#022F40', edgecolor='k')
    
        # === Step 3: Styling ===
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        xtick_hours = [0, 6, 12, 18]
        xtick_angles = [2 * np.pi * h / 24 for h in xtick_hours]
    
        ax.set_xticks(xtick_angles)
        ax.set_xticklabels([str(h) for h in xtick_hours])
        #ax.set_yticklabels([i for i in range(, 10, 2)])
        ax.set_ylim(0, max(counts)*1.2)
        plt.locator_params(axis='y', nbins=2)
        
    def plot_entrainment(fig, plot, t_col, xtick_start, xtick_end, ent_days, order=0, T=24, color='#EBEBEB'):
        
            start_time = xtick_start
            end_time = (start_time + T * ent_days) 
            
            # If Time is datetime, convert to numeric hours for easier spacing
            if np.issubdtype(plot[t_col].dtype, np.datetime64):
                time_unit = 'datetime'
                total_seconds = (end_time - start_time).total_seconds()
                num_bands = int(total_seconds // (12 * 3600)) 
                delta = pd.Timedelta(hours=12)
            else:
                time_unit = 'numeric'
                num_bands = int((end_time - start_time) // (T/2)) 
                delta = (T/2)
                
            for i in range(num_bands):
                band_start = start_time + i * delta + T/2 * order
                band_end = band_start + delta 
                if i % 2 == 0:  # Every other band
                    plt.axvspan(band_start, band_end, color=color, alpha=1, zorder=-10)
            return fig
        
    def plot_entrainment_ax(ax, plot, t_col, xtick_start, xtick_end, ent_days, order=0, T=24, color='#EBEBEB'):
            
                start_time = xtick_start
                end_time = (start_time + T * ent_days) 
                
                # If Time is datetime, convert to numeric hours for easier spacing
                if np.issubdtype(plot[t_col].dtype, np.datetime64):
                    time_unit = 'datetime'
                    total_seconds = (end_time - start_time).total_seconds()
                    num_bands = int(total_seconds // (12 * 3600)) 
                    delta = pd.Timedelta(hours=12)
                else:
                    time_unit = 'numeric'
                    num_bands = int((end_time - start_time) // (T/2)) 
                    delta = (T/2)
                    
                for i in range(num_bands):
                    band_start = start_time + i * delta + T/2 * order
                    band_end = band_start + delta 
                    if i % 2 == 0:  # Every other band
                        ax.axvspan(band_start, band_end, color=color, alpha=1, zorder=-10)
        
    def plot(df, t_col, p_col, t0, t1, bg_color='white', ent=False, ent_days=0, features=None, 
             order=0, T=24, color='white', unit='Measured unit'):
        
        fig, ax = plt.subplots(1, figsize=(10, 4))
        ax.set_facecolor(bg_color)
        
        plot = df[(df[t_col] >= t0) & (df[t_col] <= t1) ]
        #plt.plot(plot[t_col], plot[p_col])
        sns.lineplot(plot, x=t_col, y=p_col)
        
        scat = st.toggle('Show datapoints', True)
        if scat:
            sns.scatterplot(plot, x=t_col, y=p_col, edgecolor='k', zorder=10)
        
        # Get actual min and max from your data
        xmin = plot[t_col].min()
        xmax = plot[t_col].max()
        
        ymin = plot[p_col].min()
        ymax = plot[p_col].max()
        
        # Añadir margen manual (5%)
        x_margin = (xmax - xmin) * 0.07
        y_margin = (ymax - ymin) * 0.07

        # Calculate start and end of xticks, rounded to nearest multiples of 24
        xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
        xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
        
        if features is not None:
            #st.write(features)
            vmin, vmax = plot[p_col].min(), plot[p_col].max()
            methods.feature_entrainment(ax, features, bg_color, color, ymin - y_margin, ymax + y_margin,
                order=order)
        
        ax.set_xlim(xmin - x_margin, xmax + x_margin)
        ax.set_ylim(ymin - y_margin, ymax + y_margin)
        #if ent_days > 0:
            # Example for creating banded background every 12 hours
            
        #    fig = methods.plot_entrainment(fig, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
        
        # Generate ticks at every 24 units
        xticks = np.arange(xtick_start, xtick_end + 1, 24)
        plt.xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
        plt.xlabel('Time (h)')
        plt.ylabel(unit)
        return fig
    
    def multiplot(ax, df, t_col, p_col, t0, t1, bg_color='white', ent=False, ent_days=0, 
             order=0, T=24, color='white', unit='Measured unit'):
        
       # fig, ax = plt.subplots(1, figsize=(10, 4))
        ax.set_facecolor(bg_color)
        
        plot = df[(df[t_col] >= t0) & (df[t_col] <= t1) ]
        plt.plot(plot[t_col], plot[p_col])
        
        # Get actual min and max from your data
        xmin = plot[t_col].min()
        xmax = plot[t_col].max()
        
        # Calculate start and end of xticks, rounded to nearest multiples of 24
        xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
        xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
        
        if ent_days > 0:
            # Example for creating banded background every 12 hours
    
            methods.plot_entrainment(ax, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
        
        # Generate ticks at every 24 units
        xticks = np.arange(xtick_start, xtick_end + 1, 24)
        plt.xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
        plt.xlabel('Time (h)')
        plt.ylabel(unit)
        return ax
    
    def grouped_plot(df, t_col, t0, t1, group, layout,  bg_color='white', ent=False, ent_days=0, 
             order=0, T=24, color='white', unit='Measured unit'):
        
        cols = layout[layout.Condition == group]['name'].to_list()     
        
        fig, ax = plt.subplots(1, figsize=(10, 4))
        ax.set_facecolor(bg_color)
        
        plot = df[(df[t_col] >= t0) & (df[t_col] <= t1) ]
                
        mu1 = plot[cols].mean(axis=1)
        sigma1 = plot[cols].std(axis=1)

        #ax.plot(t, mu1, lw=2, label='mean population 1', color='blue')
        ax.plot(plot[t_col], mu1, lw=2, )
        ax.fill_between(plot[t_col], mu1+sigma1, mu1-sigma1, facecolor='grey', alpha=0.3, zorder=10)
        
        # Get actual min and max from your data
        xmin = plot[t_col].min()
        xmax = plot[t_col].max()
        
        # Calculate start and end of xticks, rounded to nearest multiples of 24
        xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
        xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
        
        if ent_days > 0:
            # Example for creating banded background every 12 hours
    
            fig = methods.plot_entrainment(fig, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
        
        # Generate ticks at every 24 units
        xticks = np.arange(xtick_start, xtick_end + 1, 24)
        plt.xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
        plt.xlabel('Time (h)')
        plt.ylabel(unit)
        plt.title(f"{group} (N={len(cols)})", fontsize=15)
        return fig
    
    def grouped_plot_traces(df, t_col, t0, t1, group, layout,  bg_color='white', ent=False, ent_days=0, 
             order=0, T=24, color='white', unit='Measured unit'):
        
        cols = layout[layout.Condition == group]['name'].to_list()     
        
        fig, ax = plt.subplots(1, figsize=(10, 4))
        ax.set_facecolor(bg_color)
        
        plot = df[(df[t_col] >= t0) & (df[t_col] <= t1) ]
                
        mu1 = plot[cols].mean(axis=1)
        sigma1 = plot[cols].std(axis=1)

        #ax.plot(t, mu1, lw=2, label='mean population 1', color='blue')
        ax.plot(plot[t_col], mu1, lw=2, )
        
        for col in cols:
            ax.plot(plot[t_col], plot[col], lw=2, alpha=0.2)
        #ax.fill_between(plot[t_col], mu1+sigma1, mu1-sigma1, facecolor='grey', alpha=0.3, zorder=10)
        
        # Get actual min and max from your data
        xmin = plot[t_col].min()
        xmax = plot[t_col].max()
        
        # Calculate start and end of xticks, rounded to nearest multiples of 24
        xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
        xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
        
        if ent_days > 0:
            # Example for creating banded background every 12 hours
            fig = methods.plot_entrainment(fig, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
        
        # Generate ticks at every 24 units
        xticks = np.arange(xtick_start, xtick_end + 1, 24)
        plt.xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
        plt.xlabel('Time (h)')
        plt.ylabel(unit)
        plt.title(f"{group} (N={len(cols)})", fontsize=15)
        return fig
    
    def double_plot(df, t_col, p_col, ent_days, T, order, t0, t1, times = 2, entrainment_data=None,
                    bg_color='white', band_color='white', signal_color='#1F7A8C'):
        
        df = df[(df[t_col] >= t0) & (df[t_col] <= t1)]
        df[t_col] = df[t_col] - t0
        df['d'] = df[t_col].apply(lambda x: int(x/24))
        df[p_col] = (df[p_col] - df[p_col].min()) / (df[p_col].max() - df[p_col].min())

        if entrainment_data is not None:

            if 'entrainment' not in entrainment_data.columns:
                entrainment_data['entrainment'] = entrainment_data.iloc[:,-1]
            
            entrainment_data['entrainment'] = (entrainment_data['entrainment'] - entrainment_data['entrainment'].min()) / (entrainment_data['entrainment'].max() - entrainment_data['entrainment'].min())

        days = int(np.round(df[t_col].max() / 24))

        yscaling = st.checkbox('Re-scale Y axis by subset amplitude', False)
        global_vmean = df[p_col].mean()   # robust to outliers
        global_vmax = np.nanpercentile(df[p_col], 99)
        
        if days < 2:
            st.error('This function needs more than 2 days to plot')
            st.stop()

        fig, ax = plt.subplots(days, 1, figsize=(10, days*0.5))
        
        for i in range(1, days+1):
            
            bot = (i * 24 - 24 * times) 
            #top = (i * 24 * times)
            #bot = (i - 1) * 24
            top = bot + 24 * times
            plot = df[(df[t_col].between(bot, top, inclusive='both'))]
 
            plot['time_col'] = plot[t_col]
            plot['time_col'] = plot['time_col'].apply(lambda x: x- bot)
            
            ax[i -1 ].set_facecolor(bg_color)
            
            ax2 = ax[i -1 ].twinx()
            ax2.fill_between(plot.time_col, plot[p_col], color=signal_color, zorder=10)
            ax2.set_xlim(0, 24*times)    
            
            # Get the actual ylim being used for this subplot
            if yscaling:
                y_lo = plot[p_col].mean()
                y_hi = np.nanpercentile(plot[p_col], 99) + np.nanpercentile(plot[p_col], 99) * 0.07#plot[p_col].mean() + plot[p_col].std() * 2
            else:
                y_lo = global_vmean
                y_hi = global_vmax

            ax2.set_ylim(y_lo, y_hi)
            ax[i -1 ].set_yticks([])
            ax2.set_yticks([])

            # x-tick spacing
            dist = {1: 6}.get(times) or (12 if times == 2 else 24)
            
            if i == days:
                ax[i -1 ].set_xticks([i for i in range(0, 24*times+1, dist)])
            else:
                ax[i -1 ].set_xticks([])
            
            ax[i - 1 ].set_ylabel(f"Day {i}", rotation=0, ha='right', va='center')
            
            if entrainment_data is not None:

                ax[i-1].set_ylim(y_lo, y_hi)

                plot_entrainment = entrainment_data[(entrainment_data[t_col].between(bot, top, inclusive='both'))].copy()
                
                # Guard against empty slice
                if plot_entrainment.empty:
                    continue
                    
                plot_entrainment['time_col'] = plot_entrainment[t_col].apply(lambda x: x - bot)

                xmin = plot['time_col'].min()
                xmax = plot['time_col'].max()
                ymin = entrainment_data['entrainment'].min()
                ymax = entrainment_data['entrainment'].max()

                # Guard against NaN in plot data itself
                if any(np.isnan(v) or np.isinf(v) for v in [xmin, xmax, ymin, ymax]):
                    continue

                y_margin = (ymax - ymin) * 0.07
                # Calculate start and end of xticks, rounded to nearest multiples of 24
                xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
                xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
                colors = [band_color, bg_color] if order == 0 else [bg_color, band_color]
                cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
                ax[i -1 ].imshow(np.vstack((plot_entrainment['entrainment'],)), 
                    extent=(plot_entrainment['time_col'].min(),
                            plot_entrainment['time_col'].max(), 
                            ymin, ymax), aspect='auto' ,cmap=cmap1, zorder=-2,
                            vmin=global_vmean, vmax=global_vmax)
                ax[i -1 ].set_ylim(ymin, ymax)

        fig.subplots_adjust(hspace=0)
        return fig
    
    def easy_pdf_report(figures):
        
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
    
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Rhythmicity Report'
            d['Author'] = 'Your Name'
    
        buffer.seek(0)
        return buffer
    
    def easy_pdf_report_new(figures, page_size=(24, 8.5)):  # landscape
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            for fig in figures:
                fig.set_size_inches(*page_size)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            d = pdf.infodict()
            d['Title'] = 'Rhythmicity Report'
            d['Author'] = 'CycleAnalysis'
            #d['CreationDate'] = datetime.now()
        buffer.seek(0)
        return buffer
    
    def simple_plot(df, t_col, col, 
                    unit='Measured unit', 
                    bg_color='white', title=None):
                    
            fig, ax = plt.subplots(1, figsize=(12, 7))
            ax.set_facecolor(bg_color)
            #ax.plot(df[t_col], df[col])
            sns.lineplot(df, x=t_col, y=col, ax=ax)
            if title == None:
                ax.set_title(col)
            else:
                plt.suptitle(title)
            
            xmin = df[t_col].min()
            xmax = df[t_col].max()
            
            # callate start and end of xticks, rounded to nearest multiples of 24
            xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
            xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
            
            ax.set_xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
            ax.set_xlabel('Time (h)')
            ax.set_ylabel(unit)  
            
            return fig
    
    def split_plot(df, t_col, col, 
                            ent=False, ent_days = 0, unit='Measured unit', 
                            bg_color='white', band_color='lightblue',
                            order=0, T=24, title=None):
        
            fig, ax = plt.subplots(1, 2, figsize=(20, 7))
            for i in range(2):
                ax[i].set_facecolor(bg_color)
            ent_data = df[df[t_col] <= ent_days * T]
            fr_data = df[df[t_col] >= ent_days * T]
            ax[0].plot(ent_data[t_col], ent_data[col])
            ax[0].set_title(f"Entrainment")
            
            # Get actual min and max from your data
            xmin = ent_data[t_col].min()
            xmax = ent_data[t_col].max()
            
            # Calculate start and end of xticks, rounded to nearest multiples of 24
            xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
            xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
            
            if ent_days > 0:
                
                start_time = xtick_start
                end_time = (start_time + T * ent_days) 

                num_bands = int((end_time - start_time) // (T/2)) 
                delta = (T/2)
                    
                for i in range(num_bands):
                    band_start = start_time + i * delta + T/2 * order
                    band_end = band_start + delta 
                    if i % 2 == 0:  # Every other band
                        ax[0].axvspan(band_start, band_end, color=band_color, alpha=1)
                        
            xticks = np.arange(xtick_start, xtick_end + 1, 24)
            ax[0].set_xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
            ax[0].set_xlabel('Time (h)')
            ax[0].set_ylabel(unit)
            
            # Get actual min and max from your data
            xmin = fr_data[t_col].min()
            xmax = fr_data[t_col].max()
            
            # Calculate start and end of xticks, rounded to nearest multiples of 24
            xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
            xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
            
            ax[1].plot(fr_data[t_col], fr_data[col])
            ax[1].set_title(f"Free Running")
            ax[1].set_xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
            ax[1].set_xlabel('Time (h)')
            ax[1].set_ylabel(unit)
            
            if title == None:
                plt.suptitle(col)
            else:
                plt.suptitle(title)
            
            return fig
        
    def grouped_plot_traces_export(ax, df, t_col, t0, t1, group, layout,  bg_color='white', ent=False, ent_days=0, 
             order=0, T=24, color='white', unit='Measured unit'):
        
        cols = layout[layout.Condition == group]['name'].to_list()     
        
        #fig, ax = plt.subplots(1, figsize=(10, 4))
        ax.set_facecolor(bg_color)
        
        plot = df[(df[t_col] >= t0) & (df[t_col] <= t1) ]
                
        mu1 = plot[cols].mean(axis=1)
        sigma1 = plot[cols].std(axis=1)

        #ax.plot(t, mu1, lw=2, label='mean population 1', color='blue')
        ax.plot(plot[t_col], mu1, lw=2, )
        
        for col in cols:
            ax.plot(plot[t_col], plot[col], lw=2, alpha=0.2)
        #ax.fill_between(plot[t_col], mu1+sigma1, mu1-sigma1, facecolor='grey', alpha=0.3, zorder=10)
        
        # Get actual min and max from your data
        xmin = plot[t_col].min()
        xmax = plot[t_col].max()
        
        # Calculate start and end of xticks, rounded to nearest multiples of 24
        xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
        xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
        
        if ent_days > 0:
            # Example for creating banded background every 12 hours
            ax = methods.plot_entrainment(ax, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
        
        # Generate ticks at every 24 units
        xticks = np.arange(xtick_start, xtick_end + 1, 24)
        ax.set_xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
        ax.set_xlabel('Time (h)')
        ax.set_ylabel(unit)
        ax.set_title(f"{group} (N={len(cols)})", fontsize=15, loc='left')
        return ax
    
    def plot_table_on_ax(ax, df):
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
    def pie_chart(ax, df, method='meta2d', group='', thresh=0.05):
            
            group = group.replace('_', '-')
            cols = [col for col in df.columns if method in col]
            q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
            
            significant = df[df[q_col] <= thresh]
            
            replicates = df.shape[0]
            sig_replicates = significant.shape[0]
            percent = np.round(sig_replicates / replicates * 100, 1)
            not_sig = 100 - percent

            vlag_pal = sns.color_palette('vlag', 6)
            pal = [vlag_pal[0], vlag_pal[-1]]#['#57C4E5','#F97068']

            ax.pie([percent, not_sig], labels=['Significant', 'Not significant'], 
            autopct='%1.1f%%', colors=pal, startangle=90,
            wedgeprops=dict(width=0.6))
        
    def text(ax, df, method='meta2d', group='', thresh=0.05):
        
        group = group.replace('_', '-')
        cols = [col for col in df.columns if method in col]
        per_col = 'Periods'#[col for col in cols if 'PERIOD' in col.upper()][0]
        q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
        
        period = f"{np.round(df[per_col].mean(),1)} ± {np.round(df[per_col].std(),1)}"
        significant = df[df[q_col] <= thresh]
        
        replicates = df.shape[0]
        sig_replicates = significant.shape[0]
        percent = np.round(sig_replicates / replicates * 100, 1)
        
        ax.axis("off")
        formatted_text = (
    f"$\\bf{{{group} \\,  summary:}}$\n\n"
    f"$\\bf{{N}}$: {df.shape[0]} replicates\n"
    f"$\\bf{{Rhythmic\\ replicates}}$: {sig_replicates}/{replicates} ({percent}%)\n"
    f"$\\bf{{Method}}$: {method} - Significance threshold = {thresh}\n"
    f"$\\bf{{Detected\\ period}}$: {period} h"
)

        ax.text(0, 1, formatted_text, fontsize=15, va='top', ha='left', transform=ax.transAxes)
        
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

            per_col = 'Periods'#[col for col in cols if 'PERIOD' in col.upper()][0]
            q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
            amp_col = [col for col in cols if 'AMP' in col.upper()][0]
            
            table = []
            for i in [sorted_x, sorted_y]:
                significant = i[i[q_col] <= thresh]
                non_sig = i.shape[0] - significant.shape[0]
                sig = significant.shape[0]
                table.append([sig, non_sig])
            
            odds, p = fisher_exact(table, alternative='two-sided')
            t_stat_per, p_per = ttest_ind(sorted_x[per_col].values,
                                        sorted_y[per_col].values, equal_var=False)  
            t_stat_amp, p_amp = ttest_ind(sorted_x[amp_col].values,
                                        sorted_y[amp_col].values, equal_var=False)  
            
            sig_comparison.append([x, y, compar, p, p < thresh, ])
            per_comparison.append([x, y, compar, p_per, p_per < thresh])
            amp_comparison.append([x, y, compar, p_amp, p_amp < thresh])  
        
        summary = pd.DataFrame()
        
        for n, d in enumerate([sig_comparison, per_comparison, amp_comparison]):
            temp = pd.DataFrame(d, columns=['group1', 'group2', 'comparison', 'p-val', 'reject'])
            temp['tested'] = ['Rhythmicity', 'Period', 'Amplitude'][n]
            summary = pd.concat([summary, temp]).reset_index(drop=True)
            
        return summary

    def multi_subplot():

        #fig, ax = plt.subplots(days, 1, figsize=(10, days*0.5))
        fig = plt.figure(figsize=(10, days*0.5))
        gs = fig.add_gridspec(2, 4)  

    def multi_acto(ax, df, t_col, p_col, ent_days, T, order, t0, t1, times = 2, entrainment_data=None,
                        yscaling=False, bg_color='white', band_color='white', signal_color='#1F7A8C', title='None'):
            
            df = df[(df[t_col] >= t0) & (df[t_col] <= t1)]
            df[t_col] = df[t_col] - t0
            df['d'] = df[t_col].apply(lambda x: int(x/24))
            df[p_col] = (df[p_col] - df[p_col].min()) / (df[p_col].max() - df[p_col].min())

            days = int(np.round(df[t_col].max() / 24))

            global_vmean = df[p_col].mean()   # robust to outliers
            global_vmax = np.nanpercentile(df[p_col], 99)
            
            if entrainment_data is not None:

                if 'entrainment' not in entrainment_data.columns:
                    entrainment_data['entrainment'] = entrainment_data.iloc[:,-1]
                
                entrainment_data['entrainment'] = (entrainment_data['entrainment'] - entrainment_data['entrainment'].min()) / (entrainment_data['entrainment'].max() - entrainment_data['entrainment'].min())
            
            if days < 2:
                st.error('This function needs more than 2 days to plot')
                st.stop()

            #fig, ax = plt.subplots(days, 1, figsize=(10, days*0.5))
            if title != 'None':
                ax[0].set_title(title)

            for i in range(1, days+1):
                
                bot = (i * 24 - 24 * times) 
                top = bot + 24 * times
                plot = df[(df[t_col].between(bot, top, inclusive='both'))]

                plot['time_col'] = plot[t_col]
                plot['time_col'] = plot['time_col'].apply(lambda x: x- bot)
                
                ax[i -1 ].set_facecolor(bg_color)
                ax2 = ax[i -1 ].twinx()

                ax2.fill_between(plot.time_col, plot[p_col], color=signal_color)
                ax2.set_xlim(0, 24*times)    
                
                # Get the actual ylim being used for this subplot
                if yscaling:
                    y_lo = plot[p_col].mean()
                    y_hi = np.nanpercentile(plot[p_col], 99) + np.nanpercentile(plot[p_col], 99) * 0.07#plot[p_col].mean() + plot[p_col].std() * 2
                else:
                    y_lo = global_vmean
                    y_hi = global_vmax

                ax2.set_ylim(y_lo, y_hi)
                ax[i -1 ].set_yticks([])
                ax2.set_yticks([])
                
                if times == 2:
                    dist = 12
                elif times > 2:
                    dist = 24
                else:
                    dist = 6
                
                if i == days:
                    ax[i -1 ].set_xticks([i for i in range(0, 24*times+1, dist)])
                else:
                    ax[i -1 ].set_xticks([])
                
                ax[i - 1 ].set_ylabel(f"Day {i}", rotation=0, ha='right', va='center')
                
                days_of_entrainment = [i for i in range(1, ent_days+1)]
                belong_to_entrainment = [i * 24 - 24 for i in plot.d.unique() if i in days_of_entrainment]

                if entrainment_data is not None:

                    plot_entrainment = entrainment_data[(entrainment_data[t_col].between(bot, top, inclusive='both'))].copy()
                    
                    # Guard against empty slice
                    if plot_entrainment.empty:
                        continue
                        
                    plot_entrainment['time_col'] = plot_entrainment[t_col].apply(lambda x: x - bot)

                    xmin = plot['time_col'].min()
                    xmax = plot['time_col'].max()
                    ymin = entrainment_data['entrainment'].min()
                    ymax = entrainment_data['entrainment'].max()

                    # Guard against NaN in plot data itself
                    if any(np.isnan(v) or np.isinf(v) for v in [xmin, xmax, ymin, ymax]):
                        continue

                    y_margin = (ymax - ymin) * 0.07

                    # Calculate start and end of xticks, rounded to nearest multiples of 24
                    xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
                    xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24

                    colors = [band_color, bg_color] if order == 0 else [bg_color, band_color]
                    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
                    ax[i -1 ].imshow(np.vstack((plot_entrainment['entrainment'],)), 
                            extent=(plot_entrainment['time_col'].min(),
                            plot_entrainment['time_col'].max(), 
                            y_lo, y_hi), aspect='auto' ,cmap=cmap1, zorder=-2,
                            vmin=global_vmean, vmax=global_vmax)
                
            return
    
    def feature_entrainment(ax, feat_data, ent_color, bg_color, lower, upper, order=0):

        colors = [ent_color, bg_color] if order == 0 else [bg_color, ent_color]
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
        #plt.plot(feat_data.iloc[:, 0], feat_data.iloc[:, 1])
        ax.imshow(np.vstack((feat_data.iloc[:, 1],)), 
            extent=(feat_data.iloc[:, 0].min() ,feat_data.iloc[:, 0].max(), 
                        lower, upper), aspect='auto'
                        , cmap=cmap1, vmin=feat_data.iloc[:, 1].min(), vmax=feat_data.iloc[:, 1].max())

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

        times = data[t_col].value_counts()
        n_replicates = times.unique()
        delta_t = np.mean(np.diff(times.index))  # assumes sorted time
        
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
            Ps[i] = np.linalg.solve(X.T @ X, X.T)            # (3, n_t)
    
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
        pvalues = np.asarray(pvalues, dtype=float)
        m       = len(pvalues)
        order   = np.argsort(pvalues)
        ranks   = np.empty(m, dtype=int)
        ranks[order] = np.arange(1, m + 1)
        qvalues = np.minimum(1.0, pvalues * m / ranks)
        qvalues = np.minimum.accumulate(
            qvalues[order][::-1]
        )[::-1][np.argsort(order)]
        return qvalues
    
    
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