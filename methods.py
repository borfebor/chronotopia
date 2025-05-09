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
            st.warning('Not compatible format')
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
    
    def rolling_mean(df, data_cols, window_size=10):
        
        rolling_mean = df[data_cols].rolling(window=window_size, center=True).mean()
        return df[data_cols] - rolling_mean
    
    def detrend(df, data_cols, t_col, method='None'):
        if method == 'None':
            return df[data_cols]
        else:
            win_size = int(1/(df[t_col].diff().mean()) * 10)
            st.write(f"Rolling mean window size: {win_size}")
            meth = {'Linear': methods.linear_detrend(df, data_cols),
                   'Rolling mean': methods.rolling_mean(df, data_cols, win_size)}
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
        
    def generate_pdf_report(df, t_col, data_cols, ent=False, ent_days = 0, unit='Measured unit', bg_color='white', band_color='lightblue'):
        buffer = BytesIO()
            
        with PdfPages(buffer) as pdf:
            if ent == False:
                for col in data_cols:  # Replace with your loop over data
                    fig, ax = plt.subplots(1, figsize=(10, 5))
                    ax.set_facecolor(bg_color)
                    ax.plot(df[t_col], df[col])
                    ax.set_title(col)
                    pdf.savefig(fig)
                    plt.close(fig)
            else: 
                
                for col in data_cols:  # Replace with your loop over data
                    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                    for i in range(2):
                        ax[i].set_facecolor(bg_color)
                    ent_data = df[df[t_col] <= ent_days * 24]
                    fr_data = df[df[t_col] >= ent_days * 24]
                    ax[0].plot(ent_data[t_col], ent_data[col])
                    ax[0].set_title(f"Entrainment")
                    
                    # Get actual min and max from your data
                    xmin = ent_data[t_col].min()
                    xmax = ent_data[t_col].max()
                    
                    # Calculate start and end of xticks, rounded to nearest multiples of 24
                    xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
                    xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
                    
                    if ent == True:
                        # Example for creating banded background every 12 hours
                        start_time = xtick_start
                        end_time = (start_time + 24 * ent_days) 
                        
                        # If Time is datetime, convert to numeric hours for easier spacing
                        if np.issubdtype(ent_data[t_col].dtype, np.datetime64):
                            time_unit = 'datetime'
                            total_seconds = (end_time - start_time).total_seconds()
                            num_bands = int(total_seconds // (12 * 3600)) 
                            delta = pd.Timedelta(hours=12)
                        else:
                            time_unit = 'numeric'
                            num_bands = int((end_time - start_time) // 12) 
                            delta = 12
                            
                        for i in range(num_bands):
                            band_start = start_time + i * delta
                            band_end = band_start + delta
                            if i % 2 == 0:  # Every other band
                                ax[0].axvspan(band_start, band_end, color=band_color, alpha=0.8)
                                
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
                    ax[1].set_xticks([i for i in range(int(xtick_end), int(xtick_end), 24)])
                    ax[1].set_xlabel('Time (h)')
                    ax[1].set_ylabel(unit)
                    plt.suptitle(col)
                    
                    pdf.savefig(fig)
                    plt.close(fig)
                    
                # Add metadata (optional)
                d = pdf.infodict()
                d['Title'] = 'Rhythmicity Report'
                d['Author'] = 'Your Name'
                # Add metadata (optional)
                d = pdf.infodict()
                d['Title'] = 'Rhythmicity Report'
                d['Author'] = 'Your Name'
    
        buffer.seek(0)
        return buffer
        
    def figure_w_entrainment(df, t_col, data_col, t_plot, p_col, ent, ent_days, unit):
        fig = plt.figure(figsize=(10, 4))
        
        plot = df[(df[t_col] >= t_plot[0]) & (df[t_col] <= t_plot[1]) ]
        plt.plot(plot[t_col], plot[p_col])
        
        # Get actual min and max from your data
        xmin = plot[t_col].min()
        xmax = plot[t_col].max()
        
        # Calculate start and end of xticks, rounded to nearest multiples of 24
        xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
        xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
        
        if ent == True:
            # Example for creating banded background every 12 hours
            start_time = xtick_start
            end_time = (start_time + 24 * ent_days) 
            
            # If Time is datetime, convert to numeric hours for easier spacing
            if np.issubdtype(plot['Time'].dtype, np.datetime64):
                time_unit = 'datetime'
                total_seconds = (end_time - start_time).total_seconds()
                num_bands = int(total_seconds // (12 * 3600)) 
                delta = pd.Timedelta(hours=12)
            else:
                time_unit = 'numeric'
                num_bands = int((end_time - start_time) // 12) 
                delta = 12
                
            for i in range(num_bands):
                band_start = start_time + i * delta
                band_end = band_start + delta
                if i % 2 == 0:  # Every other band
                    plt.axvspan(band_start, band_end, color='lightblue', alpha=0.5)
        
        # Generate ticks at every 24 units
        xticks = np.arange(xtick_start, xtick_end + 1, 24)
        plt.xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
        plt.xlabel('Time (h)')
        plt.ylabel(unit)
        return fig
    
    def plot_entrainment(fig, plot, xtick_start, xtick_end, ent_days, T=24, color='#EBEBEB'):
        
            start_time = xtick_start
            end_time = (start_time + T * ent_days) 
            
            # If Time is datetime, convert to numeric hours for easier spacing
            if np.issubdtype(plot['Time'].dtype, np.datetime64):
                time_unit = 'datetime'
                total_seconds = (end_time - start_time).total_seconds()
                num_bands = int(total_seconds // (12 * 3600)) 
                delta = pd.Timedelta(hours=12)
            else:
                time_unit = 'numeric'
                num_bands = int((end_time - start_time) // (T/2)) 
                delta = (T/2)
                
            for i in range(num_bands):
                band_start = start_time + i * delta
                band_end = band_start + delta
                if i % 2 == 0:  # Every other band
                    plt.axvspan(band_start, band_end, color=color, alpha=1)
            return fig
    #def normalization(df, cols, method):
        
    #    for col in ps.columns: