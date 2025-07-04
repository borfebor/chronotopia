#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:39:58 2025

@author: borfebor
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy import signal
import matplotlib.pyplot as plt
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
        envelope = np.abs(signal.hilbert(detrended))
        return detrended / envelope

    @staticmethod
    def detrend(df, cols, t_col, method='None'):
        if method == 'None':
            return df[cols]

        suggested = int(1 / df[t_col].diff().mean() * 10)
        win = st.slider(f"Window size (suggested = {suggested*2})", int(suggested), int(suggested * 4), int(suggested*2)) if 'Rolling' in method else 10

        methods_map = {
            'Linear': methods.linear_detrend(df, cols),
            'Rolling mean': methods.rolling_mean(df, cols, win),
            'Hilbert + Rolling mean': methods.hilbert_rolling_mean(df, cols, win),
            'Cubic': methods.cubic_detrend(df, cols),
        }
        return methods_map.get(method, df[cols])

    @staticmethod
    def min_max(df, cols, mode='all'):
        if mode == 'all':
            top = df[cols].max().max()
            return (df[cols] - df[cols].min()) / (top - df[cols].min()) * 100
        return (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min()) * 100

    @staticmethod
    def z_score(df, cols):
        return (df[cols] - df[cols].mean()) / df[cols].std()

    @staticmethod
    def normalize(df, cols, method='None'):
        methods_map = {
            'Sample-wise Min-Max': methods.min_max(df, cols, mode='sample'),
            'Global Min-Max': methods.min_max(df, cols, mode='all'),
            'Z-Score': methods.z_score(df, cols)
        }
        return methods_map.get(method, df[cols])

    def autocovariance(x):
        """Compute the autocovariance of a 1D array."""
        x = x - np.mean(x)
        result = np.correlate(x, x, mode='full')
        return result[result.size//2:]

    def autocorrelation(x):
        """Normalize autocovariance to produce autocorrelation."""
        acov = methods.autocovariance(x)
        return acov / acov[0]
    
    def period_correlation(signal):
        ac = methods.autocorrelation(signal)
    
        min_lag = 20
        max_lag = 28
        # Find maximum in this range
        lag_range = range(min_lag, max_lag+1)
        peak_lag = lag_range[np.argmax([ac[lag] for lag in lag_range])]
        return peak_lag
    
    def fft_period(signal, t):
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
    
    def Lomb_Scargle(signal, t):
        
        frequency, power = LombScargle(t, signal).autopower(minimum_frequency=1/100,
                                                                maximum_frequency=1/10)
        peak = np.argmax(power)
        peak_frequency = frequency[peak]
        peak_period = 1/peak_frequency
            
        return peak_period
    
    def wavelet(signal, t_col):
        periods = np.linspace(18, 36, 100)
        dt = np.mean(np.diff(t_col))  # assumes sorted time
        
        wAn = WAnalyzer(periods, dt, p_max=20)
    
        wAn.compute_spectrum(signal)
    
        wAn.get_maxRidge(power_thresh = 10, smoothing_wsize=20)
        return wAn.ridge_data  # this is a pandas DataFrame holding the ridge results
    
    @staticmethod
    def period_estimation(df, cols, t_col, method='None'):
        if method == 'None':
            return 'No period estimation'
        
        methods_map = {
            'Fast Fourier Transform (FFT)': df[cols].apply(lambda x: methods.fft_period(x, df[t_col].values)),
            'Lomb-Scargle Periodogram': df[cols].apply(lambda x: methods.Lomb_Scargle(x, df[t_col].values)),
            'Autocorrelation':  df[cols].apply(lambda x: methods.period_correlation(x)),
            'Wavelet Transform': df[cols].apply(lambda x: methods.wavelet(x, df[t_col])['periods'].mean())
        }
        return methods_map.get(method, df[cols])

        
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
                    plt.axvspan(band_start, band_end, color=color, alpha=1)
            return fig
        
    def plot(df, t_col, p_col, t0, t1, bg_color='white', ent=False, ent_days=0, 
             order=0, T=24, color='white', unit='Measured unit'):
        
        fig, ax = plt.subplots(1, figsize=(10, 4))
        ax.set_facecolor(bg_color)
        
        plot = df[(df[t_col] >= t0) & (df[t_col] <= t1) ]
        plt.plot(plot[t_col], plot[p_col])
        
        # Get actual min and max from your data
        xmin = plot[t_col].min()
        xmax = plot[t_col].max()
        
        # Calculate start and end of xticks, rounded to nearest multiples of 24
        xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
        xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
        
        if ent == True:
            # Example for creating banded background every 12 hours
    
            fig = methods.plot_entrainment(fig, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
        
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
        
        if ent == True:
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
        
        if ent == True:
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
        
        if ent == True:
            # Example for creating banded background every 12 hours
            fig = methods.plot_entrainment(fig, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
        
        # Generate ticks at every 24 units
        xticks = np.arange(xtick_start, xtick_end + 1, 24)
        plt.xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
        plt.xlabel('Time (h)')
        plt.ylabel(unit)
        plt.title(f"{group} (N={len(cols)})", fontsize=15)
        return fig
    
    def double_plot(df, t_col, p_col, ent_days, T, order, t0, t1, times = 2, 
                    bg_color='white', band_color='white'):
        
        df['d'] = df[t_col].apply(lambda x: int(x/24))
        df = df[(df[t_col] >= t0) & (df[t_col] <= t1)]
        df[t_col] = df[t_col] - t0
        
        days = int(np.round(df[t_col].max() / 24))
        
        if days < 2:
            st.error('This function needs more than 2 days to plot')
            st.stop()

        fig, ax = plt.subplots(days, 1)
        
        for i in range(1, days+1):
            
            bot = (i * 24 - 24 * times) 
            top = (i * 24 * times)
            plot = df[(df[t_col] <= top) & (df[t_col]  >= bot)]
            #print(plot.Minutes.min(), plot.Minutes.max(),)
            plot['time_col'] = plot[t_col]
            plot['time_col'] = plot['time_col'].apply(lambda x: x- bot)
            
            ax[i -1 ].set_facecolor(bg_color)
            
            ax[i -1 ].fill_between(plot.time_col, plot[p_col], color='#1F7A8C')
            ax[i -1 ].set_xlim(0, 24*times)    
            ax[i -1 ].set_ylim(plot[p_col].mean(), plot[p_col].mean()+plot[p_col].std()*2)
            ax[i -1 ].set_yticks([])
            
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
            
            for t in belong_to_entrainment:
                
                start_time = t - np.min(belong_to_entrainment)
                end_time = (start_time + T) 
                
                num_bands = int(len(belong_to_entrainment))#int((end_time - start_time) // (T/2)) 
                delta = (T/2)
                
                for n in range(num_bands):
                     band_start = start_time + n * delta + T/2 * order
                     band_end = band_start + delta 
                     if n % 2 == 0:  # Every other band
                         ax[i-1].axvspan(band_start, band_end, color=band_color, alpha=1, zorder=-10)
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
    
    def simple_plot(df, t_col, col, 
                    unit='Measured unit', 
                    bg_color='white', title=None):
                    
            fig, ax = plt.subplots(1, figsize=(12, 7))
            ax.set_facecolor(bg_color)
            ax.plot(df[t_col], df[col])
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
            
            if ent == True:
                
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
        
        if ent == True:
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
            
            pal = ['#F97068', '#57C4E5']

            ax.pie([percent, not_sig], labels=['Significant', 'Not significant'], autopct='%1.1f%%', colors=pal)
        
    def text(ax, df, method='meta2d', group='', thresh=0.05):
        
        group = group.replace('_', '-')
        cols = [col for col in df.columns if method in col]
        per_col = [col for col in cols if 'PERIOD' in col.upper()][0]
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

            per_col = [col for col in cols if 'PERIOD' in col.upper()][0]
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
