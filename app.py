#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:55:41 2025

@author: borfebor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from methods import methods
import scipy.spatial.distance as ssd
from scipy.optimize import curve_fit
from sklearn.manifold import MDS

from io import BytesIO

from PIL import Image
import subprocess
import tempfile
import os
import math
from pyboat import WAnalyzer

tab_logo = Image.open('tab_logo.png')

st.set_page_config(
     page_title="Chronotopia",
     page_icon=tab_logo,
     layout="centered",
     initial_sidebar_state="expanded"
)

image = Image.open('logo.png')
intro_image = Image.open('chrono_intro.png')
st.sidebar.image(image)

def convert_for_download(df):
        return df.to_csv(sep='\t').encode("utf-8")
    
    
version = "0.6.2"
st.sidebar.write(f"Version {version}")    
st.sidebar.header('Data uploading')

pop_upload = st.sidebar.popover('Upload your data', width='stretch')

with pop_upload:
    uploaded_file = st.file_uploader('Upload your data',
                        type=['csv','txt','xlsx', 'tsv'])

ex_place = st.sidebar.empty()

with ex_place:
    #eg = st.expander('Example datasets', True)
    example = st.toggle('Generate example dataset')

if example:
    uploaded_file = 'hola'

if uploaded_file is None:
    st.image(intro_image) 

    st.stop()

if uploaded_file is not None:
    messages = st.empty()
    messages2 = st.empty()
    
    st.header('Data Preview')
    sum_pre = st.empty()
    on = st.toggle("Show data preview")
    preview = st.empty()

    st.header('Data Analysis')
    
    settings = st.expander('Data analysis settings (Filtering, Normalization, Detrending...)')  
    with settings:

        c1, c2 = st.columns(2)
    
    if uploaded_file == 'hola':
        
        with st.sidebar.popover('Example dataset parameters', width='stretch'):
            eg1, eg2 = st.columns(2)
            ex_days = eg1.number_input('Days generated', 1, 20, 7,  step=1)    
            ex_datapoints = eg2.number_input('Timepoints per day', 4, 144, 12, step=1) 
            ex_samples = eg1.number_input('Number of samples', 1, 96, 5,  step=1)   
            ex_percent = eg2.number_input('Ratio of rhythmic (%)', 0, 100, 80,  step=1)/ 100
            ex_period = eg1.number_input('Free running period', 2, 48, 24,  step=1)   
            ex_ent_days = eg2.number_input('Entrainment days', 0, ex_days, 0,  step=1)  
            ex_ent_period = eg1.number_input('Entrainment period', 2, 48, 24,  step=1) 
            ex_waveform = eg2.selectbox('Waveform', ['sin', 'square', 'saw'], 0) 
            if ex_ent_days > 0:
                ex_entrain = True
            else: 
                ex_entrain = False
        np.random.seed(42)
        df, meta, time_hours = methods.generate_rhythm_dataset(
            num_days=ex_days,
            points_per_day=ex_datapoints,
            n_samples=ex_samples,
            percent_rhythmic=ex_percent,
            period=ex_period,               # intrinsic period in hours (can be scalar or (min,max))
            entrain=ex_entrain,
            entrain_start_day=0,
            entrain_end_day=ex_ent_days,
            entrain_period=ex_ent_period,       # period of the entraining cycle in hours
            noise_sd=np.random.randint(0,10)/10,
            amp_range=(0.8, 1.2),
            phase_jitter_sd=0.2,       # radians jitter when entrained
            intrinsic_period_jitter=0.2, # hours sd to jitter each sample's intrinsic period
            nonrhythm_drift=True,
            random_seed=42,
            waveform=ex_waveform             # 'sin' or 'square' or 'saw'
        )
    else:
        df = methods.importer(uploaded_file)
        ex_place.empty()
        
    layout = st.sidebar.popover('Upload experimental layout', width='stretch')
    
    df.columns = [col.strip() for col in df.columns]
    
    col_t, col_unit = st.sidebar.columns(2)
    t_col = col_t.selectbox('Time column', [col for col in df.columns] )
    
    times = df[t_col].value_counts()
    n_replicates = times.unique()
    delta_t = np.mean(np.diff(times.index))  # assumes sorted time
    
    t_options = ['Minutes', 'Hours', 'Days', 'Seconds']
    
    if uploaded_file == 'hola':
        default = t_options.index('Hours')
    else:
        if delta_t > 1:
            default = t_options.index('Minutes')
        else:
            default = t_options.index('Hours')
        
    t_unit = col_unit.selectbox('Time unit', t_options, default)

    data_cols = [col for col in df.columns if col != t_col]
    
    if len(data_cols) == 96:
        
        template = pd.DataFrame(data_cols, columns=['Sample'])
        template['Condition'] = [f"COL_{int(i[-2:])}" for i in data_cols]
        layout_df = template.copy()
        layout_df['name'] = layout_df.Sample
            
    layout_file = None
    
    with layout:
        st.header('Experimental groups')
        template = pd.DataFrame(data_cols, columns=['Sample'])
        template['Condition'] = 'YOUR_CONDITION'
        
        csv = convert_for_download(template)
        
        st.download_button(label="Download layout template",
                        data=csv,
                        file_name='sample_layout_template.txt',
                        mime='text/csv',
                        type='primary',
                        help='Here you can download your data',
                        width='stretch',)
        layout_file = st.file_uploader('Upload your experimental layout',
                                type=['csv','txt','xlsx', 'tsv'])
        
        if layout_file is not None:
            
            layout_df = methods.importer(layout_file)
            
            layout_df['name'] = layout_df.Condition + " - [" + layout_df.Sample + "]"
            
            name_dict = dict(zip(layout_df.Sample, layout_df.name))
            df = df.rename(columns=name_dict)
            data_cols = [col for col in df.columns if col != t_col]
    
    df[t_col] = df[t_col].apply(lambda x: methods.time_changer(x, t_unit))
    
    t_start = c1.number_input('Starting Timepoint', df[t_col].min(), df[t_col].max(), df[t_col].min())
    t_end =  c2.number_input('Last Timepoint', t_start, df[t_col].max(),df[t_col].max() )
    
    df = df[(df[t_col] >= t_start)  & (df[t_col] <= t_end)]
    
    st.sidebar.header('Analysis paramenters')
    
    hourly = c1.toggle('Smoothen the data', False)
    normalize_time = c2.toggle('Always start time from 0', True)

    ent = st.sidebar.popover('Entrainment parameters', width='stretch')

    ent_exclude = ent.toggle('Exclude entrainment from period estimation', True)
    exclusion = st.sidebar.popover('Exclude samples from data', width='stretch')

    period_methods = ['Fast Fourier Transform (FFT)', 'Lomb-Scargle Periodogram', 'Wavelet Transform']
    
    if df[t_col].size >= 30:
        period_methods = period_methods + ['Autocorrelation']
    
    period_estimation = st.sidebar.selectbox('Period Estimation', period_methods, 1)
    period_len_min, period_len_max = st.sidebar.slider("Period range", 1, 100, (24-8, 24+8), step=1)

    test_a_bit = st.sidebar.popover('Rhythmicity Analysis Parameters',  width='stretch')#st.sidebar.toggle('Rhythmicity analysis parameters', False)
    
    if normalize_time == True:
        
        df[t_col] = df[t_col] - df[t_col].min()
    
    with settings:
    
        exclusion_place = st.empty()  
        
    max_days = int(df[t_col].max() / 24) + 1
    
    backgroud = {'None': 'white',
                 'Darkness': '#EBEBEB' ,
                 'Light':'#FFD685' ,
                 'Warm': '#fbe3d4',
                 'Cold': '#dbeaf2'}
    
    bg_color = backgroud['None']
    ent_color = backgroud['None']
    
    if hourly == True:
        # Smooth with a 1-hour window
        samples_per_hour = int(round(1 / delta_t))
        if samples_per_hour < 1:
            samples_per_hour = 1
        df[data_cols] = df[data_cols].rolling(window=samples_per_hour, center=True, min_periods=1).mean()
        df = df.dropna()
        #st.stop()
        
    norm_meth = c1.selectbox('Normalization', ['None', 'Z-Score', 'Sample-wise Min-Max', 'Global Min-Max'])
    detrend_meth = c2.selectbox('Detrending', ['None', 'Linear', 'Rolling mean', 'Hilbert + Rolling mean', 'Cubic'])

    df[data_cols] = methods.detrend(df, data_cols, t_col, detrend_meth)
    df[data_cols] = methods.normalize(df, data_cols, norm_meth)
        
    with ent:
                
        #with settings:
        col1, col2 = st.columns(2)
        ent_days = col1.number_input('Entrainment cycles', 0, max_days, 0,  step=1) 
        T = col2.number_input('T cycle', 6, 48, 24,  step=1) 
        cycle_type = col1.selectbox('Zeitgeber type', ['Darkness - Light', 'Light - Darkness', 'Cold - Warm', 'Warm - Cold', 'Custom'], 0) 
        ord_place = col2.empty()
        
    if ent_days > 0:
        
        if cycle_type == 'Custom':
                color1, color2 = st.columns(2)
                ent_color = col1.color_picker('Entrainment band', '#9BD1E5')
                bg_color = col2.color_picker('Background color', '#ffffff')
                order = ord_place.selectbox('Color order', [0, 1], 0)
        else:
                parts = [part.strip() for part in cycle_type.split("-")]
                fr_options = [i for i in ['Light', 'Darkness', 'Cold', 'Warm'] if i in parts]
        
                freerun_type = ord_place.selectbox('Free running conditions', fr_options, 1) 
                bg_color = backgroud[freerun_type]
                #band_color = parts.remove(freerun_type)
                band_type = [i for i in parts if i != freerun_type][0]
                ent_color = backgroud[band_type]
                
                order = parts.index(band_type)
        
        entrain_data = df[df[t_col] <= df[t_col].min() + T * ent_days].reset_index(drop=True)
        
        if np.mean(n_replicates) > 1:
            entrain_data = entrain_data.groupby(t_col).agg({col:('mean') for col in data_cols}).reset_index()
        phases = entrain_data[data_cols].apply(lambda x: methods.sine_phase(entrain_data[t_col], x))

    else:
        T, order, ent_days = 0, 0, 0
            
    with exclusion:
        
        ex_type, ex_cols = st.columns([1,2])
        if 'layout_df' in globals():
            ex_options = layout_df.columns
        else:
            ex_options = ['Sample']
            
        ex_col = ex_type.selectbox('Exclude by', ex_options)
        
        if 'layout_df' in globals():
            ex_values = layout_df[ex_col].unique()
            exclusion_list = ex_cols.multiselect("Select data to exclude", ex_values)
            exclusion_list = layout_df[layout_df[ex_col].isin(exclusion_list)]['name'].to_list()

        else:
            ex_values = data_cols
            exclusion_list = ex_cols.multiselect("Select data to exclude", ex_values)
            
        df = df.drop(columns=exclusion_list)
        if (len(exclusion_list) > 0) & (len(exclusion_list) <= 5):
            st.write(f"{', '.join(exclusion_list)} excluded from data")
        elif (len(exclusion_list) > 5):
            arg = f"{', '.join(exclusion_list[:5])}"
            st.write(f"{arg} and {len(exclusion_list[5:])} other samples were excluded from the data")

        data_cols =  [col for col in df.columns if col != t_col]
        
        if 'layout_df' in globals():
            layout_df = layout_df[~layout_df['name'].isin(exclusion_list)]

    df = df.dropna()
    fr_data = df[df[t_col] >= df[t_col].min() + T * ent_days].reset_index(drop=True) if ent and ent_exclude else df.copy()
             
    method = 'meta2d'
    thresh = 0.05
        
    with test_a_bit:
            t1, t2 = st.columns(2)
            t_start_test = t1.number_input('Minimum time', int(fr_data[t_col].min()), int(fr_data[t_col].max()), int(fr_data[t_col].min()),  step=1)    
            t_end_test = t2.number_input('Last time', int(df[df[t_col] > t_start_test][t_col].min()), int(df[df[t_col] > t_start_test][t_col].max()), int(df[df[t_col] > t_start_test][t_col].max()), step=1) 
            method = t1.selectbox('Testing method', ['meta2d', 'JTK', 'ARS', 'LS', ], 0)   
            thresh = t2.selectbox('Significance threshold', [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], 0) 
        
    duration = np.round(df[t_col].max(),1)
    sum_pre.write(f"Experiment with {len(data_cols)} sample recorded for {duration} hours (recorded every = {delta_t:.1f} h)")
    
    conditions = []
    visu = ['Lineplot', 'Actogram', 'Correlation', 'PCA']
    
    if 'layout_df' in globals():
        
        conditions = list(layout_df.Condition.unique())
        
        visu = visu + ['Lineplot [Mean ± SD]', 'Lineplot [Mean + Replicates]']
        
    if period_estimation == 'Wavelet Transform':
        
        visu = visu + ['Wavelet Ridge']
    
    if ent_days > 0:
        
        visu = visu + ['Phase plot']
    
    viz_settings = st.expander('Visualization settings (Plot type, sample selection, data unit...)')  

    with viz_settings:
    
        c, c1, c2 = st.columns([2, 1, 1])
        t0 = c1.number_input('Starting time to plot', int(df[t_col].min()), int(df[t_col].max()), int(df[t_col].min()),  step=1)    
        t1 = c2.number_input('End time to plot', int(df[df[t_col] > t0][t_col].min()), int(df[df[t_col] > t0][t_col].max()), int(df[df[t_col] > t0][t_col].max()), step=1) 
        
        plot_type = c.selectbox("Type of plot to visualize", visu)

    short = df[[t_col] + data_cols[:5]].iloc[:5]
    
    if on:
        preview.dataframe(short.set_index(t_col))
    
    pre_plot = st.empty()

    with viz_settings:
        
        cus1, cus2, cus3 = st.columns(3)
        style = cus1.selectbox('Select style', ['white', 'ticks', 'whitegrid', 'darkgrid', 'dark'])
        context = cus2.selectbox('Select context', ['talk', 'paper', 'notebook', 'poster'], 2)        
        palettes = list(sns.palettes.SEABORN_PALETTES.keys()) + [name for name in plt.colormaps()]
        
        palette = cus3.selectbox('Select context', palettes, palettes.index('colorblind'))

        sns.set_style(style)
        sns.set_context(context)
        sns.set_palette(palette)
        
        if bg_color == 'white':
            # Get the current style dictionary
            style_dict = sns.axes_style()
            
            # Extract the background color of the axes
            bg_color = style_dict.get('axes.facecolor')

        if plot_type == 'Lineplot':
            
            p_col = st.selectbox('Column to preview', data_cols)
            unit = st.text_input('Data unit', 'Measured unit')
            per = methods.period_estimation(fr_data, [p_col], t_col, method=period_estimation, 
                                            min_period=period_len_min, max_period=period_len_max)
            per = np.round(per, 2)

            fig = methods.plot(df, t_col, p_col, t0, t1, bg_color=bg_color, ent=ent, 
                         ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)
        
            plt.title(f"{p_col}. Period = {per.loc[p_col]} h ({period_estimation}-calculated).")

            pre_plot.pyplot(fig)
                
        elif plot_type == 'Lineplot [Mean ± SD]':
                
            p_col = st.selectbox('Column to preview', conditions)
            unit = st.text_input('Data unit', 'Measured unit')
            
            fig = methods.grouped_plot(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                     ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)

            pre_plot.pyplot(fig)
            
        elif plot_type == 'Lineplot [Mean + Replicates]':
                
            p_col = st.selectbox('Column to preview', conditions)
            unit = st.text_input('Data unit', 'Measured unit')
            #pre_plot = st.empty()
            
            fig = methods.grouped_plot_traces(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                     ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)    
            
        elif plot_type == 'Actogram':
            p_col = st.selectbox('Column to preview', data_cols)
            times = st.number_input("Plot N times", 1, int(np.round(df[t_col].max() / 24)), 1)
            #pre_plot = st.empty()
            if np.mean(n_replicates) > 1:
                df_plot = df.groupby(t_col).agg({col:('mean') for col in data_cols}).reset_index()
            else:
                df_plot = df.copy()

            fig = methods.double_plot(df_plot, t_col, p_col, ent_days, T, order, t0=t0, t1=t1, times=times, 
                                      bg_color=bg_color, band_color=ent_color)
            
        elif plot_type == 'Phase plot':
            
            p_col = st.selectbox('Column to preview', data_cols)
            unit = st.text_input('Data unit', 'Measured unit')

            peaks = phases.loc[p_col]
            
            fig = plt.figure(figsize=(7, 3), layout='tight')
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

            ax = fig.add_subplot(gs[0, 0])              # normal Cartesian plot
            ax2 = fig.add_subplot(gs[0, 1], polar=True) # polar plot

            sns.lineplot(entrain_data, x=t_col, y=p_col, ax=ax)
            
            t_mod = entrain_data[t_col] % 24
            signal = entrain_data[p_col]
            popt, _ = curve_fit(methods.sine_model, entrain_data[t_col], signal, p0=[1, 0, np.mean(signal)])

            fitted_signal = methods.sine_model(entrain_data[t_col], *popt)
            ax.plot(entrain_data[t_col], fitted_signal, linestyle='--', color='k', alpha=0.8)
            xtick_start = (entrain_data[t_col].min() // 24) * 24          # floor to nearest lower multiple of 24
            xtick_end = ((entrain_data[t_col].max()  // 24) + 1) * 24      # ceil to next multiple of 24
            
            methods.plot_entrainment_ax(ax, entrain_data, t_col, xtick_start, xtick_end,
                                           ent_days, order=order, T=T, color=ent_color)

            ax.set_xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
            ax.set_ylabel(unit)
            ax.set_xlabel('Time (h)')
                
            methods.phase_plot(entrain_data, ax2, peaks, pal=[bg_color, ent_color], order=order)

        elif plot_type == 'Wavelet Ridge':
            
            p_col = st.selectbox('Column to preview', data_cols)
            unit = st.text_input('Data unit', 'Measured unit')
            
            signal = df[p_col]
            periods = np.linspace(18, 36, 100)
            dt = np.mean(np.diff(df[t_col]))  # assumes sorted time
            
            wAn = WAnalyzer(periods, dt, p_max=20)
        
            wAn.compute_spectrum(signal)
        
            wAn.get_maxRidge(power_thresh = 10, smoothing_wsize=20)

            rd = wAn.ridge_data # this is a pandas DataFrame holding the ridge results
            
            #fig, ax = plt.subplots(1, 2, layout='constrained', gridspec_kw={'width_ratios': [4, 1]})
            fig, axes = plt.subplot_mosaic("AAAAC;DDDDD", layout='constrained')

            wAn.draw_Ridge()
            sns.kdeplot(
                    rd, y='periods', x='time',
                    fill=True, thresh=0, levels=100, cmap="viridis",
                        bw_adjust=0.5, # smoother KDE
                    clip=((rd['time'].min(), rd['time'].max()), (18, 36)), ax=axes['A'] 
                )
                #plt.ylim(18, 36)
            sns.lineplot(rd, x='time', y='periods', color='w', ax=axes['A'])

            sns.kdeplot(rd, y='periods', fill=True, ax=axes['C'])
            axes['D'].plot(df[t_col], df[p_col])
            plt.suptitle(f"{p_col} Estimated period: {np.average(rd.periods, weights=rd.power):.2f} h")

        elif plot_type == 'Correlation':
            
            p_col = st.selectbox('Colormap plette', ['viridis', 'vlag', 'coolwarm'], 1)
            annot = st.selectbox('Show annotation', [True, False], 1)
            ## Transpose to make each row a separate observation
            list_of_series = [df[col].tolist() for col in data_cols]
            
            # Now stack into a 2D array
            array = np.stack(list_of_series)
            
            # Pair-wise distance matrix (for instance, euclidean)
            dist_matrix = 1-ssd.pdist(np.stack(list_of_series), metric='correlation')
            dist_matrix = ssd.squareform(dist_matrix)
            
            fig, ax = plt.subplots(figsize=(0.5*len(array), 0.5*len(array)))
            sns.heatmap(dist_matrix, cmap=p_col, square=True, annot=annot,
                        yticklabels=data_cols, xticklabels=data_cols, vmax=1, vmin=-1, center=0)
            plt.title("Correlation Between Time Series")
            plt.xlabel("Samples")
            plt.ylabel("Samples")
            
        elif plot_type == 'PCA':
            
            list_of_series = [df[col].tolist() for col in data_cols]
            annot = st.selectbox('Show annotation', [True, False], 1)
            array = np.stack(list_of_series)
            
            # Pair-wise distance matrix (for instance, euclidean)
            dist_matrix = 1-ssd.pdist(np.stack(list_of_series), metric='euclidean')
            dist_matrix = ssd.squareform(dist_matrix)

            mds = MDS(n_components=2, dissimilarity='precomputed')
            embedding = mds.fit_transform(dist_matrix)
            
            if 'layout_df' in globals():
            # Map group names to colors
                groups = dict(zip(layout_df.name, layout_df.Condition))
                unique_groups =  list(set(groups.values()))
                color_map = {group: color for group, color in zip(unique_groups, plt.cm.tab20.colors)}

            fig, ax = plt.subplots(figsize=(7, 7))

            plotted_groups = set()

            for i, label in enumerate(data_cols):
                if 'layout_df' in globals():
                    group = groups[label]
                    color = color_map[group]
                else:
                    color = '#F97068'
                    group = 'default'
                    
                show_label = group if group not in plotted_groups else "_nolegend_"  # avoid duplicates
                plotted_groups.add(group)
                ax.scatter(embedding[i, 0], embedding[i, 1], color=color, label=show_label, 
                           linewidth=1, edgecolor='k', alpha=0.7, s=50)
                if annot == True:
                    ax.text(embedding[i, 0], embedding[i, 1], label.replace(group, ''), ha='left', va='bottom')

            ax.legend(title='Groups')

            plt.title("Multidimensional Scaling of Time Series")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            
    if 'unit' not in globals():
        unit = 'signal'   
     
    pre_plot.pyplot(fig)
    # Convert to BytesIO for download
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    
    # Add download button
    st.download_button(
        label="Download Plot as PNG",
        data=buf,
        file_name="my_plot.png",
        mime="image/png",
        width='stretch',
    )
    
    csv = convert_for_download(df)
    
    st.sidebar.header('Final steps')

    analysis_button = st.sidebar.button('Run analysis', type='primary', width='stretch')
    st.sidebar.download_button(label="Download clean data",
                    data=csv,
                    file_name='clean_data.txt',
                    mime='text/csv',
                    help='Here you can download your data',
                    width='stretch',)
    
    if analysis_button:
        
        with st.spinner("Running R script..."):
            st.toast('Calculating periods...!')
            
            periods = methods.period_estimation(df, data_cols, t_col, method=period_estimation,
                                                min_period=period_len_min, max_period=period_len_max).rename('Period')
            periods = np.round(periods, 2)
            
            # Transpose and set index
            df[t_col] = df[t_col].apply(lambda x: np.round(x,1))
            test_df = df[np.isclose(df[t_col] % 1, 0)]
            
            if np.mean(n_replicates) > 1:
                rdf = test_df.groupby(t_col).agg({col:('mean') for col in data_cols}).transpose()
            else:
                rdf = test_df[(test_df[t_col] >= t_start_test) & (test_df[t_col] <= t_end_test)].set_index(t_col).transpose().reset_index()
            
            #st.stop()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w') as temp_file:
                rdf.to_csv(temp_file.name, sep="\t", index=False)
                input_path = temp_file.name
                
            output_dir = tempfile.mkdtemp()
            
            st.toast('Testing rhythmicity...!')

            result = subprocess.run(
                ["Rscript", "run_meta2d.R", input_path, output_dir],
                capture_output=True,
                text=True
            )
            
            st.text("STDOUT:\n" + result.stdout)
            #st.text("STDERR:\n" + result.stderr)
            
            if result.returncode != 0:
                messages.error("R script failed.")
                csv = convert_for_download(rdf)
                
                messages2.download_button(label="Download data for Metacycle testing",
                                data=csv,
                                file_name='data_for_metacycle.txt',
                                mime='text/csv',
                                type='primary',
                                help='Here you can download your data',
                                width='stretch',)
            else:
                st.toast('Report ready to download!', icon='🎉')

                result_df = pd.read_csv(os.path.join(output_dir, "meta2d_result.csv"))
                col_sorter = [i for i in result_df.columns if 'meta.' in i]
                result_df = result_df[col_sorter]
                result_df.columns = [i.replace('meta.', '') for i in result_df.columns]
                result_df = result_df.set_index('CycID')
                result_df['Periods'] = periods
                result_df = result_df.reset_index()
                
                cols = [col for col in result_df.columns if method in col]
                q_col = [col for col in cols if 'BH.Q' in col.upper()][0]  
                
                result_df['reject'] = np.where(result_df[q_col] <= thresh, True, False)
                messages.dataframe(result_df)
                st.session_state["result_df"] = result_df  # Save in session state

                csv = convert_for_download(result_df)
                
                messages2.download_button(label="Download MetaCycle results",
                                data=csv,
                                file_name='meta2d_results.txt',
                                mime='text/csv',
                                type='primary',
                                help='Here you can download your data',
                                width='stretch',)
                            
    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
        
        if "layout_df" in globals():

            baloon = st.sidebar.button('Compare groups', 
                                       width='stretch',)
            if baloon:
                #st.balloons()
                sum_stats = methods.multicomparison(result_df, layout_df, conditions, method, thresh)
                st.write(sum_stats)
                st.session_state["sum_stats"] = sum_stats  # Save in session state
    
    if "sum_stats" in st.session_state:
        sum_stats = st.session_state["sum_stats"]
            
    report_spot = st.sidebar.empty()
    report_button = report_spot.button(
                            label="📄 Prepare report",
                            width='stretch'
                        )

    if report_button:
        
        with st.spinner("Preparing report..."):
                st.toast('Preparing report...!')

                figures = []
                
                if ent_days > 0:      
                        
                        if 'layout_df' in globals():                             
                                n_conditions = layout_df.Condition.unique()                              
                        else:
                                n_conditions = data_cols
                                
                        N = len(n_conditions)  # number of experiments
                        cols = math.ceil(math.sqrt(N))
                        rows = math.ceil(N / cols)
                        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3),
                                                         layout='tight', subplot_kw={'polar': True}  )
                            
                        if np.size(axes) > 1:
                                axes = axes.flatten()  # Flatten to simplify indexing
                                
                        for n, condition in enumerate(n_conditions):
                                if np.size(axes) > 1:
                                        ax = axes[n]
                                else:
                                        ax = axes
                                if 'layout_df' in globals():
                                    group = layout_df[layout_df.Condition == condition]['name'].to_list()   
                                else:
                                    group = condition
                                
                                methods.phase_plot(phases, ax, phases.loc[group],
                                                       pal=[bg_color, ent_color], order=order)
                                ax.set_title(condition)
                                
                            # Hide unused axes
                        for j in range(N, np.size(axes)):
                                fig.delaxes(axes[j])
                                
                        plt.suptitle('Phase calculation', fontsize=20, weight='bold')
                        figures.append(fig)
                        
                if "result_df" in globals():
                                ##st.write(result_df)
                                res = result_df.set_index('CycID')
                                mix = res.copy()
                                hue_unit = 'reject'
                                rows = result_df.shape[0]
        
                                if 'layout_df' in globals():
                                    trans = layout_df.set_index('name')
                                    mix = pd.concat([res, trans], axis=1)
                                    rows = mix.Condition.nunique()
                                    hue_unit = 'Condition'
                                
                                per_col = 'Periods'
                                                
                                if 'layout_df' in globals():
                                    fig, axes = plt.subplots(1, 2, figsize=(8, rows), layout='constrained')
                                    
                                    for n, ax in enumerate(axes):
                                        plot_data = mix[mix.reject == True] if n == 1 else mix
                                        title = 'Only rhythmic' if n == 1 else 'All samples'
                                        
                                        sns.pointplot(plot_data, y='Condition', x=per_col, hue=hue_unit, #linecolor='k',
                                                  ax=ax, capsize=0.2).set(xlim=(period_len_min,
                                                                        period_len_max))
                                        sns.stripplot(plot_data, y='Condition', x=per_col,  hue=hue_unit, edgecolor='k', 
                                                      linewidth=1, alpha=0.7, legend=False, ax=ax)

                                        ax.set_ylabel('')
                                        ax.set_title(title)
                                    #ax2.set_ylabel('')
                                else:
                                    fig, axes = plt.subplots(1, 1, figsize=(4, rows / 2), layout='tight')
        
                                    sns.pointplot(mix, y=mix.index, x=per_col, join=False, hue=hue_unit,
                                              markeredgecolor='k', markeredgewidth=1, alpha=0.7).set(xlim=(period_len_min,
                                                                                                period_len_max),
                                                                                                ylabel='')
                                plt.suptitle(f'Period estimation ({period_estimation}-calculated)', fontsize=20, weight='bold')
                                                                                            
                                figures.append(fig)
                
                if conditions != []:
                            
                    N = len(conditions)  # number of experiments
                    cols = math.ceil(math.sqrt(N))
                    rows = math.ceil(N / cols)
                    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), layout='tight')
                    if np.size(axes) > 1:
                                axes = axes.flatten()  # Flatten to simplify indexing
                            
                    for n, group in enumerate(conditions):
                                if np.size(axes) > 1:
                                    ax = axes[n]
                                else:
                                    ax = axes
                                sorter = layout_df[layout_df.Condition == group]['name'].unique()
                                sorted_result = result_df[result_df['CycID'].isin(sorter)]
                                
                                methods.pie_chart(ax, sorted_result, method=method, group=group, thresh=thresh)
                                ax.set_title(group)
                                
                            # Hide unused axes
                    for j in range(N, np.size(axes)):
                                fig.delaxes(axes[j])
                              
                    plt.legend(ncol=2)
                    figures.append(fig)
                            
                            
                    if "sum_stats" in globals():
                        
                        columns = [col for col in result_df.columns if method in col]
                        
                        look_for = dict(zip(['Rhythmicity', 'Period', 'Amplitude'], ['BH.Q', 'PERIOD', 'AMP']))
            
                        for cat in sum_stats.tested.unique():
                            
                            sorted_stats = sum_stats[(sum_stats.tested == cat) & (sum_stats.reject == True)]

                            look_col = [col for col in columns if look_for[cat] in col.upper()][0]
                            
                            if sorted_stats.shape[0] > 0:
                                
                                N = sorted_stats.shape[0] # number of experiments
                                cols = math.ceil(math.sqrt(N))
                                rows = math.ceil(N / cols)
                                fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 5), layout='tight')
                                
                                if rows > 1:
                                    axes = axes.flatten()  # Flatten to simplify indexing
                                
                                for n, d in sorted_stats.reset_index().iterrows():
                                
                                    # Get names for each group
                                    names_group1 = layout_df.loc[layout_df.Condition == d.group1, 'name']
                                    names_group2 = layout_df.loc[layout_df.Condition == d.group2, 'name']
                                    
                                    # Get values to plot
                                    if cat == 'Rhythmicity':
                                        
                                        # Get values for each group
                                        values_group1 = result_df.loc[result_df['CycID'].isin(names_group1), look_col].values
                                        values_group2 = result_df.loc[result_df['CycID'].isin(names_group2), look_col].values
                                        
                                        # Count how many values are below the threshold
                                        count_below_group1 = (values_group1 < thresh).sum() / len(values_group1)
                                        count_below_group2 = (values_group2 < thresh).sum() / len(values_group2)
                                        
                                        # Data for bar plot
                                        counts = [count_below_group1, count_below_group2]
                                        labels = [d.group1, d.group2]
                                        colors = ['#F97068', '#57C4E5']
                                        
                                        # Select correct axis
                                        ax = axes[n] if np.size(axes) > 1 else axes
                                        
                                        # Create bar plot
                                        bars = ax.bar(labels, counts, color=colors, width=0.8)

                                    else:
                                    
                                        values_group1 = result_df.loc[result_df['CycID'].isin(names_group1), look_col].values
                                        values_group2 = result_df.loc[result_df['CycID'].isin(names_group2), look_col].values
                                        
                                        # Prepare colors
                                        colors = ['#F97068', '#57C4E5']
                                        data_to_plot = [values_group1, values_group2]
                                        
                                        # Select correct axis
                                        ax = axes[n] if np.size(axes) > 1 else axes
                                        
                                        # Create boxplot
                                        bplot = ax.boxplot(data_to_plot, widths=0.8, patch_artist=True, showmeans=True)
                                        
                                        # Set colors and title
                                        for patch, color in zip(bplot['boxes'], colors):
                                            patch.set_facecolor(color)
                                            
                                        for mean in bplot['means']:
                                            mean.set_color('k')
                                            mean.set_linewidth(2)
                                            
                                        for median in bplot['medians']:
                                            median.set_color('black')  # or any other color
                                            median.set_linewidth(2)
                                            
                                    ax.set_ylabel(cat)
   
                                    ax.set_xticklabels([d.group1, d.group2])
                                    title= f"p-val: {d['p-val']:.4f}"
                                    ax.set_title(title)
                                    
                                plt.suptitle(f"{cat} differences", fontsize=20, weight='bold')
                                figures.append(fig)

                    #st.stop()
                    for group in conditions:
                        
                        if "result_df" in globals():

                            fig, ax = plt.subplots(2, figsize=(10, 7), height_ratios=(1, 2))
                            methods.grouped_plot_traces_export(ax[1], df, t_col, t0, t1, group=group, layout=layout_df, bg_color=bg_color, ent=ent, 
                                 ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)
                            
                            sorter = layout_df[layout_df.Condition == group]['name'].unique()

                            sorted_result = result_df[result_df['CycID'].isin(sorter)]
                            if sorted_result.shape[0] == 0:
                                st.error('Oops, it might be that the IDs from the groups and the analysis do not match. Just re-run the analysis making sure that the layout have the correct names.')
                                st.stop()
                            methods.text(ax[0], sorted_result, method=method, group=group, thresh=thresh)
                            figures.append(fig)

                        else:
                            fig, ax = plt.subplots(1, figsize=(10, 4))
                            methods.grouped_plot_traces_export(ax, df, t_col, t0, t1, group=group, layout=layout_df, bg_color=bg_color, ent=ent, 
                                 ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)
                        #methods.plot_table_on_ax(ax)
                            figures.append(fig)
                        
                        sorter = layout_df[layout_df['Condition'] == group]
                        names = sorter['name'].to_list()
                        
                        N = len(sorter)  # number of experiments
                        cols = math.ceil(math.sqrt(N))
                        rows = math.ceil(N / cols)
                        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), layout='tight')
                        if np.size(axes) > 1:
                                axes = axes.flatten()  # Flatten to simplify indexing
                                
                        for n, subgroup in enumerate(names):
                                 if np.size(axes) > 1:
                                     ax = axes[n]
                                 else:
                                     ax = axes   
                                 
                                 ax.plot(df[t_col], df[subgroup])
                            
                                 if "result_df" in globals():
                                    focus = result_df[result_df['CycID'] == subgroup]
    
                                    cols = [col for col in focus.columns if method in col]
        
                                    per_col = 'Periods'
                                    q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
                                    q = np.round(focus[q_col].mean(), 5)
                                    reject = q <= thresh
                                    period = f"{focus[per_col].mean():.1f}"
                                    title = f"{subgroup}.\nPeriod: {period} h. q-value: {q} ({method} tested).\nReject: {reject} (Sig. thresh = {thresh})"
                                 else:
                                    title = subgroup
                                    
                                 ax.set_title(title)
                                 ax.set_xlabel('Time (h)')
                                 ax.set_ylabel(unit)
                                # Get actual min and max from your data
                                 xmin = df[t_col].min()
                                 xmax = df[t_col].max()
                            
                            # Calculate start and end of xticks, rounded to nearest multiples of 24
                                 xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
                                 xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
                                 ax.set_xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
                                    
                        # Hide unused axes
                        for j in range(N, np.size(axes)):
                            fig.delaxes(axes[j])
                                  
                        plt.suptitle(group, weight='bold', fontsize=20)
                        figures.append(fig)
                
                if ent_days > 0:
                    
                    for col in data_cols:
                        
                        if "result_df" in globals():
                            focus = result_df[result_df['CycID'] == col]
                            cols = [col for col in focus.columns if method in col]

                            per_col = 'Periods'
                            q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
                            q = np.round(focus[q_col].mean(), 5)
                            reject = q <= thresh
                            period = f"{focus[per_col].mean():.1f}"
                            title = f"{col}.\nPeriod: {period} h. q-value: {q} ({method} tested).\nReject: {reject} (Sig. thresh = {thresh})"
                        else:
                            title=None

                        fig = methods.split_plot(df, t_col, col,
                                                 ent=ent, ent_days = ent_days, unit=unit, 
                                                 bg_color=bg_color, band_color=ent_color,
                                                 order=order, T=T, title=title)
                        figures.append(fig)

                else:
                    for col in data_cols:
                        if "result_df" in globals():
                            focus = result_df[result_df['CycID'] == col]
                            cols = [col for col in focus.columns if method in col]

                            per_col = 'Periods'
                            q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
                            q = np.round(focus[q_col].mean(), 5)
                            reject = q <= thresh
                            period = f"{focus[per_col].mean():.1f}"
                            title = f"{col}.\nPeriod: {period} h. q-value: {q} ({method} tested).\nReject: {reject} (Sig. thresh = {thresh})"
                        else:
                            title=None
                            
                        fig = methods.simple_plot( df, t_col, col, title=title)
                        figures.append(fig)
                    
                pdf_buffer = methods.easy_pdf_report(figures)
                
                st.toast('Report ready to download!', icon='🎉')
                report_spot.download_button(
                        label="↓ Download report",
                        data=pdf_buffer,
                        file_name="rhythmicity_report.pdf",
                        mime="application/pdf",
                        width='stretch'
                    )


