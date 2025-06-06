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

class methods:
    
    def example_data():
    
        # Settings
        minutes_per_day = 24 * 60
        total_minutes = minutes_per_day * 10
        interval = 10
        n_timepoints = total_minutes // interval
        time = np.arange(0, total_minutes, interval)
        
        # Initialize dataframe
        df = pd.DataFrame({'Time': time})
        
        # Oscillating samples
        np.random.seed(42)
        n_samples = 10
        n_oscillating = 8
        
        for i in range(n_samples):
            if i < n_oscillating:
                # Oscillating: sine wave with random phase and amplitude
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.8, 1.2)
                noise = np.random.normal(0, 0.2, size=n_timepoints)
                signal = amplitude * np.sin(2 * np.pi * time / (24 * 60) + phase) + noise + 1
            else:
                # Non-oscillating: flat with some noise
                signal = np.random.normal(1, 0.2, size=n_timepoints)
            
            df[f'Sample_{i+1}'] = signal
        
        return df
    
    def importer(file):
        """
        Loads a file (either in CSV, TSV, or XLSX format) into a Pandas DataFrame.
        
        Parameters:
        file : str or file-like object
            The file can be provided as a string representing the file path (in which case it's assumed to be on disk)
            or as a file-like object (e.g., uploaded file in Streamlit).
        
        Returns:
        df : pandas.DataFrame or None
            The file content as a DataFrame if it is of a supported type (CSV, TSV, XLSX), otherwise returns None.
        """
        
        if isinstance(file, str):
            file_name = file
        else:
            file_name = file.name

        if 'TXT' in file_name.upper() or 'TSV' in file_name.upper():
            df = pd.read_csv(file, sep='\t')  # Tab-separated values
        elif 'CSV' in file_name.upper():
            df = pd.read_csv(file, sep=',')  # Comma-separated values
        elif 'XLSX' in file_name.upper():
            df = pd.read_excel(file)  # Excel file
        else:
            st.warning("""Not compatible format. Make sure that your data is either in XLSX, CSV or TXT""")
            return None  # Return None explicitly if format is unsupported
    
        return df
    
    def time_changer(x, unit='Minutes'):
    
        unit_dict = {'Minutes': x / 60,
                    'Hours': x ,
                    'Days': x * 24,
                    'Seconds': x / 60**2 ,  }
        
        return unit_dict[unit]
    
    def hourly(df, t_col):
        return df[df[t_col] % 1 == 0]
    
    def linear_detrend(df, data_cols):
        return signal.detrend(df[data_cols], type='linear')
    
    def cubic_detrend(df, data_cols):
        return detrend(df[data_cols], order=3)
        
    def rolling_mean(df, data_cols, window_size=10):
        
        rolling_mean = df[data_cols].rolling(window=window_size, center=True, min_periods=1).mean()
        return df[data_cols] - rolling_mean
    
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        nyq = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data)    
    
    def hilbert_rolling_mean(df, data_cols, window_size=10):
        
        rolling_mean = df[data_cols].rolling(window=window_size, center=True, min_periods=1).mean()
        
        detrended_signal = df[data_cols] - rolling_mean
        # Step 2: Calculate the amplitude envelope using the Hilbert transform
        analytic_signal = signal.hilbert(detrended_signal)
        amplitude_envelope = np.abs(analytic_signal)
        # Step 3: Normalize the amplitude
        normalized_signal = detrended_signal / amplitude_envelope
        return normalized_signal
    
    def detrend(df, data_cols, t_col, method='None'):
        if method == 'None':
            return df[data_cols]
        else:
            suggested = int(1/(df[t_col].diff().mean()) * 10)
            bot, top = int(suggested), int(suggested * 4)
            win_size = st.slider('Window size', bot, top, int(suggested*2))
            meth = {'Linear': methods.linear_detrend(df, data_cols),
                   'Rolling mean': methods.rolling_mean(df, data_cols, win_size),
                   'Hilbert + Rolling mean': methods.hilbert_rolling_mean(df, data_cols, win_size),
                   'Cubic': methods.cubic_detrend(df, data_cols),}
            return meth[method]
        
    def min_max(df, data_cols):
        top = df[data_cols].max().max()
        return (df[data_cols] - df[data_cols].min()) / (top - df[data_cols].min()) * 100
    
    def z_score(df, data_cols):
        mean = df[data_cols].mean()
        std = df[data_cols].std()
        return (df[data_cols] - mean) / (std)

    def normalize(df, data_cols, method='None'):
        if method == 'None':
            return df[data_cols]
        else:
            meth = {'Min-Max': methods.min_max(df, data_cols),
                   'Z-Score': methods.z_score(df, data_cols)}
            return meth[method]
        
    
    def grouped_report(buffer, df, t_col, t0, t1, conditions, layout,
                                      bg_color='white', ent=False, ent_days=0,
                                      order=0, T=24, band_color='white', unit='Measurement'):
    
        with PdfPages(buffer) as pdf:
            for group in conditions:
                fig = methods.grouped_plot_traces(df, t_col, t0, t1, group=group, layout=layout,
                                                  bg_color=bg_color, ent=ent, 
                         ent_days=ent_days, order=order, T=T, color=band_color, unit=unit)
                pdf.savefig(fig)
                plt.close(fig)
                
        buffer.seek(0)
        return buffer
        
    
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
            
            # Calculate start and end of xticks, rounded to nearest multiples of 24
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
    #def normalization(df, cols, method):
        
    #    for col in ps.columns: