#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:55:41 2025

@author: borfebor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import streamlit as st
from methods import methods
import scipy.spatial.distance as ssd
from scipy.optimize import curve_fit
from sklearn.manifold import MDS
from datetime import datetime

from io import BytesIO
import base64

from PIL import Image
import subprocess
import tempfile
import os
import math
from pyboat import WAnalyzer

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri

from ML_classifier.rhythmicity_feature_extractor import RhythmicityFeatureExtractor

tab_logo = Image.open('tab_logo.png')
today = datetime.today().strftime('%Y%m%d')

st.set_page_config(
     page_title="Chronotopia",
     page_icon=tab_logo,
     layout="centered",
     initial_sidebar_state="expanded"
)

image = Image.open('logo.png')
with st.sidebar.container(border=True):
    st.image(image)

def convert_for_download(df):
        return df.to_csv(sep='\t').encode("utf-8")

version = "0.7"
st.sidebar.write(f"Version {version}")    
st.sidebar.header('Data uploading')

up_site = st.sidebar.empty()

uploaded_file = up_site.file_uploader('Upload your data', width='stretch', 
type=['csv','txt','xlsx', 'tsv'], key='my_file_uploader')

#st.write(st.session_state)

ex_place = st.sidebar.empty()

with ex_place:
    example = st.toggle('Generate example dataset')

if example:
    uploaded_file = 'hola'

if uploaded_file is None:
    import landing

if uploaded_file is not None:
    messages = st.empty()
    messages2 = st.empty()
    
    reset = up_site.button(':material/upload_file: Upload new file', type='primary', width='stretch')

    if reset:
        del st.session_state["my_file_uploader"]
        st.rerun()
        
    st.header('Data Preview')
    sum_pre = st.empty()
    on = st.pills("Show data preview", ['Show data preview'], width='stretch', label_visibility='collapsed')
    preview = st.empty()

    st.header('Data Analysis')
    
    settings = st.expander('Data analysis settings (Filtering, Normalization, Detrending...)')  
    with settings:

        c1, c2 = st.columns(2, vertical_alignment="top")
    
    if uploaded_file == 'hola':
        file_name = 'Example data'
        up_site.write(f"Dataset: {file_name}")
        with st.sidebar.popover('Example dataset parameters', width='stretch'):
            eg1, eg2 = st.columns(2, vertical_alignment="top")
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
        file_name = uploaded_file.name.split('.')[0]
        ex_place.write(f"Dataset: {file_name}")
        
    layout = st.sidebar.popover('Upload experimental layout', width='stretch')
    
    df.columns = [col.strip() for col in df.columns]
    
    col_t, col_unit = st.sidebar.columns(2, vertical_alignment="top")
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

    df[t_col] = df[t_col].apply(lambda x: methods.time_changer(x, t_unit))

    if len(data_cols) == 96:
        import re
    
        def extract_col_id(name):
            # Try last 2 chars as number first
            try:
                return f"COL_{int(name[-2:])}"
            except ValueError:
                pass
            # Fall back to any trailing digits
            m = re.search(r'(\d+)$', name)
            if m:
                return f"COL_{int(m.group(1))}"
            # Last resort: use position-based grouping (8 rows × 12 cols)
            return None

        conditions = [extract_col_id(c) for c in data_cols]
    
        # If extraction failed for any column, fall back to position-based grouping
        if None in conditions:
            conditions = [f"COL_{(i % 12) + 1:02d}" for i in range(96)]
        
        template = pd.DataFrame(data_cols, columns=['Sample'])
        template['Condition'] = conditions#[f"COL_{int(i[-2:])}" for i in data_cols]
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
    
    
    t_start = c1.number_input('Starting Timepoint', df[t_col].min(), df[t_col].max(), df[t_col].min())
    t_end =  c2.number_input('Last Timepoint', t_start, df[t_col].max(),df[t_col].max() )
    
    df = df[(df[t_col] >= t_start)  & (df[t_col] <= t_end)]
    
    st.sidebar.header('Analysis paramenters')
    
    #hourly = c1.toggle('Smoothen the data', False)
    normalize_time = c1.toggle('Always start time from 0', True)

    with settings:
        col0, col1, col2 = st.columns(3, vertical_alignment="top")

    ent = st.sidebar.popover('Entrainment parameters', width='stretch')

    ent_exclude = ent.toggle('Exclude entrainment from period estimation', True)
    exclusion = st.sidebar.popover('Exclude samples from data', width='stretch')

    period_methods = ['Fast Fourier Transform (FFT)', 'Lomb-Scargle Periodogram', 'Wavelet Transform']
    
    if df[t_col].size >= 30:
        period_methods = period_methods + ['Autocorrelation']
    
    period_estimation = st.sidebar.selectbox('Period Estimation', period_methods, 2)
    period_len_min, period_len_max = st.sidebar.slider("Period range", 1, 100, (24-8, 24+8), step=1)

    test_a_bit = st.sidebar.popover('Rhythmicity Analysis Parameters',  width='stretch')

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
    
    hourly = col0.selectbox('Smoothening', ['None', 'Mean', 'Resample', 'Butterworth filtering', 'DCT'])

    if hourly != 'None':
        # Smooth with a 1-hour window
        samples_per_hour = int(round(1 / delta_t))
        if samples_per_hour < 1:
            samples_per_hour = 1
        if hourly == 'Mean':
            df[data_cols] = df[data_cols].rolling(window=samples_per_hour, center=True, min_periods=1).mean()
        df = df.dropna()
        #st.stop()
        
    norm_meth = col1.selectbox('Normalization', ['None', 'Z-Score', 'Sample-wise Min-Max', 'Global Min-Max'])
    detrend_meth = col2.selectbox('Detrending', ['None', 'Linear', 'Rolling mean', 'Hilbert + Rolling mean', 'Rolling Hilbert','Cubic'])

    #df[data_cols] = methods.detrend(df, data_cols, t_col, detrend_meth)
    #df[data_cols] = methods.normalize(df, data_cols, norm_meth)

    if hourly == 'Butterworth filtering':
            fs = 1 / delta_t
            df[data_cols] = methods.apply_butter(df, data_cols, fs)

    if hourly == 'DCT':
        df[data_cols] = df[data_cols].apply(methods.dct_period_filter, dt=delta_t)

    if hourly == 'Resample':
        df = methods.resampling(df, t_col)

    with ent:
        ent_mode = st.radio('Mode', ['manual', 'from data', 'upload'], width='stretch', horizontal=True)
        ft_place = st.empty()
        col1, col2 = st.columns(2, vertical_alignment="top")
        ent_cutoff_time = None  # absolute time boundary — used for all modes

        if ent_mode == 'from data':
            entrainment_features = ft_place.selectbox('Select feature columns', data_cols)
            entrainment_feat_data = df[[t_col]+[entrainment_features]]
            df = df.drop(columns=entrainment_features)
            ent_days, T, ent_cutoff_duration = methods.count_entrainment_days(entrainment_feat_data[entrainment_features], delta_t)
            ent_cutoff_time = df[t_col].min() + ent_cutoff_duration

            data_cols =  [col for col in df.columns if col != t_col]
            st.write(f"Detected {ent_days} of {T} h")

        elif ent_mode == 'upload':
            entrainment_features = ft_place.file_uploader('Upload your entrainment data',
                                type=['csv','txt','xlsx', 'tsv'])
            if entrainment_features is not None:
                entrainment_feat_data = methods.importer(entrainment_features)
                ent_days, T, ent_cutoff_duration = methods.count_entrainment_days(entrainment_feat_data.iloc[:, -1], delta_t)
                ent_cutoff_time = entrainment_feat_data.iloc[:, 0].min() + ent_cutoff_duration
            else:
                ent_days, T = 0, 24
                entrainment_feat_data = None

        else:
            ent_days = 0
            ent_days = col1.number_input('Entrainment cycles', 0, max_days, 0,  step=1) 
            T = col2.number_input('T cycle', 6, 48, 24,  step=1) 
            ratio = st.slider('Day length', 1, T, int(T/2), step=1) / T
            entrainment_feat_data = methods.add_entrainment(df, t_col,
                                                    n_days=ent_days,
                                                    period=T,
                                                    on_ratio=ratio,
                                                    release=0)

        cycle_type = col1.selectbox('Zeitgeber type', ['Darkness - Light', 'Light - Darkness', 'Cold - Warm', 'Warm - Cold', 'Custom'], 0) 
        ord_place = col2.empty()
 
    if ent_days > 0:
        
        if cycle_type == 'Custom':
                color1, color2 = st.columns(2, vertical_alignment="top")
                ent_color = col1.color_picker('Entrainment band', '#9BD1E5')
                bg_color = col2.color_picker('Background color', '#ffffff')
                order = ord_place.selectbox('Color order', [0, 1], 0)
        else:
                parts = [part.strip() for part in cycle_type.split("-")]
                fr_options = [i for i in ['Light', 'Darkness', 'Cold', 'Warm'] if i in parts]
        
                freerun_type = ord_place.selectbox('Free running conditions', fr_options, 1) 
                bg_color = backgroud[freerun_type]
                band_type = [i for i in parts if i != freerun_type][0]
                ent_color = backgroud[band_type]
                
                order = parts.index(band_type)
        
        entrain_data = df[df[t_col] <= df[t_col].min() + T * ent_days].reset_index(drop=True)
        
        if np.mean(n_replicates) > 1:
            entrain_data = entrain_data.groupby(t_col).agg({col:('mean') for col in data_cols}).reset_index()
        phases = entrain_data[data_cols].apply(
            lambda x: methods.sine_phase(entrain_data[t_col], x))

    else:
            T, order, ent_days, entrainment_feat_data, ent_cutoff_time = 0, 0, 0, None, None

    df[data_cols] = methods.detrend(df, data_cols, t_col, detrend_meth)
    df[data_cols] = methods.normalize(df, data_cols, norm_meth) 
         
    with exclusion:
        
        ex_type, ex_cols = st.columns([1,2], vertical_alignment="top")
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

    model = {
                'Tempo': {
                    'model_path': 'ML_classifier/rhythmicity_classifier.pkl',
                    'feature_names_path': 'ML_classifier/feature_names.pkl' , # Assumes default location,
                    'metadata_path':'ML_classifier/model_metadata.pkl'
                }
            }

    with test_a_bit:
            load_model = st.selectbox('Rhythmicity evaluation model', model.keys(), 0)
            t1, t2 = st.columns(2, vertical_alignment="top")
            t_start_test = t1.number_input('Minimum time', int(fr_data[t_col].min()), int(fr_data[t_col].max()), int(fr_data[t_col].min()),  step=1)    
            t_end_test = t2.number_input('Last time', int(df[df[t_col] > t_start_test][t_col].min()), int(df[df[t_col] > t_start_test][t_col].max()), int(df[df[t_col] > t_start_test][t_col].max()), step=1) 
            method = t1.selectbox('Testing method', ['meta2d', 'JTK', 'ARS', 'LS', 'PermCosinor'], 0)   
            thresh = t2.selectbox('Significance threshold', [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], 0) 

    from ML_classifier.ml_rhythmicity_classifier import MLRhythmicityClassifier

    # Now always works the same way
    selected = model[load_model]
    classifier = MLRhythmicityClassifier(
                model_path=selected['model_path'],
                feature_names_path=selected['feature_names_path']
            )
        
    duration = np.round(df[t_col].max(),1)
    sum_pre.write(f"Experiment with {len(data_cols)} sample recorded for {duration} hours (recorded every = {delta_t:.1f} h)")
    
    conditions = []
    visu = ['Lineplot', 'Actogram', 'Multi-actogram', 'Feature extraction', 'Sample Insights', 'Rhythmicity Model Evaluation', 'PCA',  'Correlation', ]
    
    if 'layout_df' in globals():
        
        conditions = list(layout_df.Condition.unique())
        
        visu = visu + ['Lineplot [Mean ± SD]', 'Lineplot [Mean + Replicates]']
        
    if period_estimation == 'Wavelet Transform':
        
        visu = visu + ['Wavelet Ridge']
    
    if ent_days > 0:
        
        visu = visu + ['Phase plot']
    
    viz_settings = st.expander('Visualization settings (Plot type, sample selection, data unit...)')  

    with viz_settings:
    
        c, c1, c2 = st.columns([2, 1, 1], vertical_alignment="top")
        t0 = c1.number_input('Starting time to plot', int(df[t_col].min()), int(df[t_col].max()), int(df[t_col].min()),  step=1)    
        t1 = c2.number_input('End time to plot', int(df[df[t_col] > t0][t_col].min()), int(df[df[t_col] > t0][t_col].max()), int(df[df[t_col] > t0][t_col].max()), step=1) 
        
        # Initialize plot selection only on first run
        if 'active_plot' not in st.session_state:
            st.session_state['active_plot'] = 0
        
        plot_type = c.selectbox("Type of plot to visualize", visu, st.session_state['active_plot'])

        # Persist the selection explicitly
        st.session_state['active_plot'] = visu.index(plot_type)

    short = df[[t_col] + data_cols[:5]].iloc[:5]
    
    if on:
        preview.dataframe(short.set_index(t_col))
    
    pre_plot = st.empty()

    with viz_settings:
        
        cus1, cus2, cus3 = st.columns(3, vertical_alignment="top")
        style = cus1.selectbox('Select style', ['white', 'ticks', 'whitegrid', 'darkgrid', 'dark'], 1)
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
            ft_col, clf_col = st.columns([2, 1])
            ml_included = st.pills('Rhythmicity evaluation', ['None', f'{method}', 'ML'], default='None', width='stretch')

            fig = methods.plot(df, t_col, p_col, t0, t1, bg_color=bg_color, ent=ent, features=entrainment_feat_data,
                         ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)

            if ml_included == f"{method}":
                if ml_included == 'PermCosinor':
                    result = methods.detect_rhythmicity(
                        df[t_col], df[p_col], signal_names=[p_col],
                        n_permutations=500,   # increase to 5000+ for publication
                        fdr_alpha=thresh,
                    )
                    st.write(result)
                    #q_col = [c for c in result_cols if "BH.Q" in c.upper()]
                    q_val = result[['q_ftest', 'q_perm']].mean().max()
                    result['evaluation'] = np.where(result['reject'] == True,
                    "Rhythmic",  "Arrhythmic")
                else:
                    result = methods.run_metacycle(df, t_col, [p_col], cyc_methods=[method] if method != 'meta2d' else ['JTK', 'ARS', 'LS'])
                    result_cols = [c for c in result.columns if method in c]
                    q_col = [c for c in result_cols if "BH.Q" in c.upper()]
                    q_val = result[q_col[0]].squeeze()
                    result['evaluation'] = np.where(result[q_col[0]] <= thresh,
                    "Rhythmic",  "Arrhythmic")
                
                q_val = f"{q_val:.4f}" if q_val >= 0.0001 else "< 0.0001"
                info_text = f"Classification: {result['evaluation'].values[0]}\nq-value: {q_val} (Threshold: {thresh})"
                plt.annotate(info_text, xy=(0.98, 0.95), xycoords='axes fraction', 
                    ha='right', va='top', 
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8), fontsize=10) 

            if ml_included == 'ML':
                classification = classifier.predict(df[p_col], df[t_col])
                info_text = (f"Classification: {'Rhythmic' if classification['is_rhythmic'] else 'Arrhythmic'}\n"
                            f"Rhythmic Probability: {classification['probability_rhythmic']:.0%}"
                            f" ({classification['confidence'].capitalize()} confidence)"
                            )
                            #f"Arrhythmic Probability: {classification['probability_arrhythmic']:.0%}")

                plt.annotate(info_text, xy=(0.98, 0.95), xycoords='axes fraction', 
                    ha='right', va='top', 
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8), fontsize=10)
        
            plt.title(f"{p_col}. Period = {per.loc[p_col]} h ({period_estimation}-calculated).", loc='left')

            pre_plot.pyplot(fig)
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"

        elif plot_type == 'Lineplot [Mean ± SD]':
                
            p_col = st.selectbox('Column to preview', conditions)
            unit = st.text_input('Data unit', 'Measured unit')
            
            fig = methods.grouped_plot(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                     ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)

            pre_plot.pyplot(fig)
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"


        elif plot_type == 'Lineplot [Mean + Replicates]':
                
            p_col = st.selectbox('Column to preview', conditions)
            unit = st.text_input('Data unit', 'Measured unit')
            #pre_plot = st.empty()
            
            fig = methods.grouped_plot_traces(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                     ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)    
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"

        elif plot_type == 'Actogram':
            p_col = st.selectbox('Column to preview', data_cols)

            cols = st.columns([4,1], vertical_alignment="top")
            times = cols[0].number_input("Plot N times", 1, int(np.round(df[t_col].max() / 24)), 1)
            color = cols[1].color_picker('Signal color', '#1F7A8C')
            #pre_plot = st.empty()
            if np.mean(n_replicates) > 1:
                df_plot = df.groupby(t_col).agg({col:('mean') for col in data_cols}).reset_index()
            else:
                df_plot = df.copy()

            fig = methods.double_plot(df_plot, t_col, p_col, ent_days, T, order, t0=t0, t1=t1, times=times,
                                     entrainment_data=entrainment_feat_data,
                                      bg_color=bg_color, band_color=ent_color, signal_color=color)
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"

        elif plot_type == 'Multi-actogram':

            if 'layout_df' in globals():
                conditions = list(layout_df.Condition.unique())
                c_cols = st.selectbox('Choose the group to inspect',conditions)
                d_cols = layout_df[layout_df.Condition == c_cols]['name'].to_list()
                plot_cols = st.multiselect('Column to preview', data_cols, d_cols)
                fig_name = f"{today}_{c_cols}_{plot_type.replace(' ','_')}_{file_name}.svg"

            else:
                plot_cols = st.multiselect('Column to preview', data_cols, data_cols[:2])
                fig_name = f"{today}_{plot_type.replace(' ','_')}_{file_name}.svg"

            cols = st.columns([2,2, 1,1], vertical_alignment="top")
            times = cols[0].number_input("Plot N times", 1, int(np.round(df[t_col].max() / 24)), 1)
            color = cols[3].color_picker('Signal color', '#1F7A8C')
            per = methods.period_estimation(fr_data, plot_cols, t_col, method=period_estimation, 
                                            min_period=period_len_min, max_period=period_len_max)
            per = np.round(per, 2)
            if np.mean(n_replicates) > 1:
                df_plot = df.groupby(t_col).agg({col:('mean') for col in data_cols}).reset_index()
            else:
                df_plot = df.copy()

            yscaling = st.checkbox('Re-scale Y axis by subset amplitude', False)

            n_plots = len(plot_cols)
            days = int(np.round(df[t_col].max() / 24))
            plots_per_row = cols[1].number_input('Plots per row', 1, 6, 2)
            hgt = int(np.round(n_plots)/plots_per_row)+1
            suggested_h = int(np.round(days * 0.2 * n_plots/plots_per_row))+1
            h = cols[2].number_input('Adjust height', suggested_h, suggested_h*4, suggested_h)
            fig = plt.figure(figsize=(3*plots_per_row, h*2), layout='tight')
            gs = fig.add_gridspec(hgt, plots_per_row)

            for i, col in enumerate(plot_cols):
            # Left panel → 10 stacked subplots
                left_gs = gs[i].subgridspec(days, 1, hspace=0.05)
                ax = [fig.add_subplot(left_gs[i]) for i in range(days)]
                title = f"{col}\n(τ = {per.loc[col]} h)"
                methods.multi_acto(ax, df_plot, t_col, plot_cols[i], ent_days, T, order, t0=t0, t1=t1, times=times, 
                entrainment_data=entrainment_feat_data, yscaling=yscaling,
                                      bg_color=bg_color, band_color=ent_color, signal_color=color, title=title)
            fig.subplots_adjust(hspace=0)

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
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"


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
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"

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
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"

        elif plot_type == 'PCA':

            if "layout_df" in globals():
                
                options = ['All'] + [c for c in layout_df.Condition.unique()]
                subsetting = st.selectbox('Choose condition', options)
                if subsetting == 'All':
                    selected_ids = data_cols
                else:
                    selected_ids = layout_df[layout_df.Condition == subsetting]['name'].to_list()

            else:
                selected_ids = data_cols

            classification = df[selected_ids].apply(lambda x: classifier.predict(x, df[t_col]))
            classification = classification.apply(pd.Series)

            vlag_pal = sns.color_palette('vlag', 6)

            pal = {"True low": "#95C6BA",
            "True medium": "#6DB0A0",
            "True high": "#539987",
            "False low": "#D3889D",
            "False medium": "#C25B78",
            "False high": "#993955"}

            rhythm_confidence = ["True high","True medium", "True low",
            "False low", "False medium", "False high"]

            validation = (
                classification
                .groupby(['is_rhythmic', 'confidence'])
                .size()
            )

            validation.index = [f"{ir} {conf}" for ir, conf in validation.index]
            values = {k: validation[k] for k in pal if k in validation}
            pal = {k:vlag_pal[n] for n, k in enumerate(rhythm_confidence)}

            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            # ── PCA on the raw traces ─────────────────────────────────────────────────────
            X = df[selected_ids].T                        # samples × timepoints
            X_scaled = StandardScaler().fit_transform(X)

            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)

            pca_df = pd.DataFrame(coords, index=selected_ids, columns=['PC1', 'PC2'])
            pca_df = pca_df.join(classification)          # attach is_rhythmic, confidence
            pca_df['label'] = pca_df['is_rhythmic'].astype(str) + ' ' + pca_df['confidence']

            # Attach condition from layout_df if available
            if 'layout_df' in globals():
                condition_map = layout_df.set_index('name')['Condition']
                pca_df = pca_df.join(condition_map)
                color_by = 'Condition'
                conditions = pca_df['Condition'].unique()
                cond_pal = dict(zip(conditions, sns.color_palette('tab10', len(conditions))))
            else:
                pca_df['Condition'] = 'All samples'
                color_by = 'Condition'
                cond_pal = {'All samples': 'steelblue'}

            var1, var2 = pca.explained_variance_ratio_ * 100

            # ── plot ──────────────────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(7, 6))

            # Overlay marker shape to encode rhythmicity on top of condition colour
            for (cond, rhythmic), sub in pca_df.groupby([color_by, 'is_rhythmic']):
                marker = 'o' if str(rhythmic) == 'True' else 'X'
                ax.scatter(
                    sub['PC1'], sub['PC2'],
                    color=cond_pal[cond],
                    marker=marker,
                    s=70, edgecolors='k', linewidths=0.4, alpha=0.85, zorder=3
                )

            # Sample ID annotations (optional — can be slow with many samples)
            if len(selected_ids) <= 30:
                for idx, row in pca_df.iterrows():
                    ax.annotate(idx, (row['PC1'], row['PC2']),
                                fontsize=7, alpha=0.6,
                                xytext=(4, 4), textcoords='offset points')

            ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', zorder=0)
            ax.axvline(0, color='gray', linewidth=0.6, linestyle='--', zorder=0)
            ax.set_xlabel(f'PC1  ({var1:.1f}% variance)', fontsize=11)
            ax.set_ylabel(f'PC2  ({var2:.1f}% variance)', fontsize=11)
            ax.set_title('PCA — ML Classification', weight='bold', fontsize=13)
            # Two-part legend: conditions (color) + rhythmicity (shape)
            condition_handles = [
                plt.scatter([], [], color=c, s=60, edgecolors='k', linewidths=0.4, label=g)
                for g, c in cond_pal.items()
            ]
            shape_handles = [
                plt.scatter([], [], marker='o', color='gray', s=60, edgecolors='k', linewidths=0.4, label='Rhythmic'),
                plt.scatter([], [], marker='X', color='gray', s=60, edgecolors='k', linewidths=0.4, label='Arrhythmic'),
            ]
            leg1 = ax.legend(handles=condition_handles, title='Condition', title_fontsize=10,
                            bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, frameon=False)
            ax.add_artist(leg1)
            ax.legend(handles=shape_handles, title='Rhythmicity', title_fontsize=10,
                    bbox_to_anchor=(1.02, 0.4), loc='upper left', fontsize=9, frameon=False)
            plt.tight_layout()
            fig_name = f"{today}_{plot_type.replace(' ','_')}_{file_name}.svg"

        elif plot_type == 'Rhythmicity Model Evaluation':

            if "layout_df" in globals():
                
                options = ['All'] + [c for c in layout_df.Condition.unique()]
                subsetting = st.selectbox('Choose condition', options)
                if subsetting == 'All':
                    selected_ids = data_cols
                else:
                    selected_ids = layout_df[layout_df.Condition == subsetting]['name'].to_list()

            else:
                selected_ids = data_cols

            classification = df[selected_ids].apply(lambda x: classifier.predict(x, df[t_col]))
            classification = classification.apply(pd.Series)

            vlag_pal = sns.color_palette('vlag', 6)

            pal = {"True low": "#95C6BA",
            "True medium": "#6DB0A0",
            "True high": "#539987",
            "False low": "#D3889D",
            "False medium": "#C25B78",
            "False high": "#993955"}

            rhythm_confidence = ["True high","True medium", "True low",
            "False low", "False medium", "False high"]

            validation = (
                classification
                .groupby(['is_rhythmic', 'confidence'])
                .size()
            )

            validation.index = [f"{ir} {conf}" for ir, conf in validation.index]
            values = {k: validation[k] for k in pal if k in validation}
            pal = {k:vlag_pal[n] for n, k in enumerate(rhythm_confidence)}

            show_periods = st.checkbox('Show period estimation', value=False)

            # ── period estimation ─────────────────────────────────────────────────────────
            if show_periods:
                periods = methods.period_estimation(
                    fr_data, selected_ids, t_col,
                    method=period_estimation,
                    min_period=period_len_min,
                    max_period=period_len_max
                ).rename('Period')
                
                # Attach to classification so we can colour by rhythmicity
                classification = classification.join(periods)

            # ── confidence buckets ──────────────────────────────────────────────────────
            high   = classification[classification.probability_rhythmic > 0.70].index.tolist()
            medium = classification[(classification.probability_rhythmic.between(0.3, 0.70))].index.tolist()
            low    = classification[classification.probability_rhythmic < 0.3].index.tolist()

            # split by rhythmic/arrhythmic for colour
            is_rhythmic = classification.is_rhythmic.astype(str)

            n_trace_cols = 2 if show_periods else 1
            fig = plt.figure(figsize=(14 + (5 if show_periods else 0), 8))
            gs  = fig.add_gridspec(
                3, 2 + (1 if show_periods else 0),
                width_ratios=(1, 2, 1.2) if show_periods else (1, 2),
                hspace=0.5, wspace=0.35
            )

            ax_pie    = fig.add_subplot(gs[:, 0])
            ax_high   = fig.add_subplot(gs[0, 1])
            ax_medium = fig.add_subplot(gs[1, 1])
            ax_low    = fig.add_subplot(gs[2, 1])

            # ── pie chart ────────────────────────────────────────────────────────────────
            ax_pie.pie(
                values.values(),
                labels=[f"{k} (n={v})" for k, v in values.items()],
                colors=[pal[k] for k in values],
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.6),                # donut style, easier to read
            )
            ax_pie.set_title('Rhythmicity Summary', weight='bold')

            # ── trace panels ─────────────────────────────────────────────────────────────
            trace_panels = [
                (ax_high,   high,   'High probability  (≥ 0.70)'),
                (ax_medium, medium, 'Medium probability  (0.35 – 0.70)'),
                (ax_low,    low,    'Low probability  (< 0.35)'),
            ]

            for ax, ids, panel_title in trace_panels:
                if ids:
                    for col in ids:
                        color = pal[f"{is_rhythmic[col]} {classification.loc[col, 'confidence']}"]
                        ax.plot(df[t_col], df[col], color=color, alpha=0.6, linewidth=0.9)
                else:
                    ax.text(0.5, 0.5, 'No samples', transform=ax.transAxes,
                            ha='center', va='center', color='gray')

                ax.set_title(panel_title, fontsize=10, weight='bold')
                ax.set_xlabel('Time (h)')
                #ax.set_ylabel(unit)
                ax.set_xticks([i for i in range(
                    int((df[t_col].min() // 24) * 24),
                    int(((df[t_col].max() // 24) + 1) * 24),
                    24
                )])

            if show_periods:

                ax_per = fig.add_subplot(gs[:, 2])

                period_data = classification.dropna(subset=['Period'])
                
                # Sort for a cleaner strip plot feel
                for i, (idx, row) in enumerate(period_data.sort_values('Period').iterrows()):
                    color = pal[f"{row.is_rhythmic} {row.confidence}"]
                    ax_per.scatter(row['Period'], i, color=color, s=60, zorder=3)

                # Median line per rhythmicity group
                for label, grp in period_data.groupby('is_rhythmic'):
                    ax_per.axvline(
                        grp['Period'].median(),
                        linestyle='--', linewidth=1.2,
                        color='#539987' if str(label) == 'True' else '#993955',
                        label=f"Median ({'rhythmic' if str(label) == 'True' else 'arrhythmic'})"
                    )

                ax_per.set_xlim(period_len_min, period_len_max)
                ax_per.set_xlabel('Period (h)')
                ax_per.set_yticks([])
                ax_per.set_title('Period Estimation', weight='bold', fontsize=10)
                ax_per.legend(fontsize=8, loc='lower right')
                ax_per.axvline(24, color='gray', linewidth=0.8, linestyle=':', zorder=0)  # 24h reference

            plt.suptitle(f'ML Classification — {subsetting if "layout_df" in globals() else "All samples"}',
                        weight='bold', fontsize=13)

            fig_name = f"{today}_{plot_type.replace(' ','_')}_{file_name}.svg"
    
        elif plot_type == 'Feature extraction':

            from chronotopia_feature_extractor import ChronotopiaFeatureExtractor

            features_df = ChronotopiaFeatureExtractor.extract_batch(
                    df, t_col=t_col, data_cols=data_cols
            )

            dispatch = { 
                            "Cosinor Analysis": {'feature': "cosinor", 'info': "Classical least-squares cosinor fit. MESOR, acrophase amplitude, R², p-value, residuals."},  
                            "Waveform": {'feature': "waveform", 'info': "Per-peak rise/fall time, FWHM, asymmetry index, cycle-to-cycle amplitude and period variance."}, 
                            "Cycles": {'feature': "cycles", 'info': "Event-based: peak/trough detection, inter-cycle intervals, complete cycle count, peak prominence statistics."}, 
                            "Baseline": {'feature': "baseline", 'info': "Rolling mean drift, substrate-depletion proxy, ADF-style non-stationarity score, linear/quadratic trend metrics."}, 
                            "Harmonic": {'feature': "harmonic", 'info': "FFT-based: fundamental power, harmonic power ratios (12 h, 8 h), secondary peak ratio, spectral complexity."}, 
                            "Noise": {'feature': "noise", 'info':"Residual noise after cosinor fit, per-band SNR profile, noise floor, residual autocorrelation structure."}, 
                            "Lomb-Scargle": {'feature': "lomb_scargle", 'info': "Lomb-Scargle periodogram metrics. Works on all lengths and handles irregular sampling."},
                            "Wavelet Ridge": {'feature': "wavelet_ridge", 'info': "Instantaneous period/amplitude/phase from Wavelet Transform ridge. Long series only."}
                        }

            group_col, ft_col = st.columns([1, 1.5])
            ft_group = group_col.selectbox('Choose feature to preview', dispatch.keys())
            
            ft_options = [col for col in features_df.columns[1:] if dispatch[ft_group]['feature'] in col]
            feature_col = ft_col.selectbox('Choose feature to preview', ft_options)
            
            st.info(dispatch[ft_group]['info'])
            if 'layout_df' in globals():
                lay_dict = dict(zip(layout_df['name'], layout_df['Condition']))
                features_df['Condition'] = features_df['sample_id'].replace(lay_dict)

            l = layout_df['Condition'].nunique()*0.4 if 'layout_df' in globals() else len(data_cols)*0.2
            fig, ax = plt.subplots(figsize=(5, l))
            sns.boxplot(features_df, 
                        y='Condition' if 'layout_df' in globals() else "sample_id",
                        x=feature_col,
                        hue='Condition' if 'layout_df' in globals() else "sample_id", legend=False)
            sns.stripplot(features_df, 
                        y='Condition' if 'layout_df' in globals() else "sample_id",
                        x=feature_col, alpha=0.8, edgecolor='k', linewidth=1, s=8,
                        hue='Condition' if 'layout_df' in globals() else "sample_id", legend=False)

            fig_name = f"{today}_{plot_type.replace(' ','_')}_{file_name}.svg"

            stat_csv = convert_for_download(features_df)
            st.download_button(label="Export features",
                            data=stat_csv,
                            file_name=f"{today}_{file_name}_features.txt",
                            mime='text/csv',
                            help='Here you can download the extracted features',
                            width='stretch', type='primary')

        elif plot_type == 'Sample Insights':

            p_col = st.selectbox('Column to preview', data_cols)
            unit = st.text_input('Data unit', 'Measured unit')
            per = methods.period_estimation(fr_data, [p_col], t_col, method=period_estimation, 
                                            min_period=period_len_min, max_period=period_len_max)
            per = np.round(per, 2)

            from chronotopia_feature_extractor import ChronotopiaFeatureExtractor

            # Full extraction (all applicable packages)
            ext = ChronotopiaFeatureExtractor(df[p_col], df[t_col], period_range=(18, 30))

            # Selective extraction
            dispatch = { 
                "Cosinor Analysis": {'feature': "cosinor", 'plot': ext.plot_cosinor},  
                "Waveform": {'feature': "waveform", 'plot': ext.plot_waveform}, 
                "Cycles": {'feature': "cycles", 'plot':ext.plot_cycles}, 
                "Baseline": {'feature': "baseline", 'plot':ext.plot_baseline}, 
                "Harmonic": {'feature': "harmonic", 'plot':ext.plot_harmonic}, 
                "Noise": {'feature': "noise", 'plot':ext.plot_noise}, 
                "Lomb-Scargle": {'feature': "lomb_scargle", 'plot':ext.plot_lomb_scargle},
                "Wavelet Ridge": {'feature': "wavelet_ridge", 'plot':ext.plot_wavelet_ridge}
            }

            ft_col, clf_col = st.columns([2, 1])

            extracted_ft = ft_col.selectbox("Features package", dispatch.keys())
            features = ext.extract(packages=[dispatch[extracted_ft]['feature']])
            ml_included = clf_col.pills('Rhythmicity evaluation (ML)', ['Include', 'Exclude'], default='Exclude', width='stretch')
            with st.expander(f'Show {extracted_ft} features'):
                st.write(features)
            # Visualisation — overlay a package on an existing matplotlib axis
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df[t_col], df[p_col])
            dispatch[extracted_ft]['plot'](ax)
            if ml_included == 'Include':
                classification = classifier.predict(df[p_col], df[t_col])
                info_text = (f"Classification: {'Rhythmic' if classification['is_rhythmic'] else 'Arrhythmic'}\n"
                            f"Rhythmic Probability: {classification['probability_rhythmic']:.0%}"
                            f" ({classification['confidence'].capitalize()} confidence)"
                            )
                            #f"Arrhythmic Probability: {classification['probability_arrhythmic']:.0%}")

                plt.annotate(info_text, xy=(0.98, 0.95), xycoords='axes fraction', 
                    ha='right', va='top', 
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8), fontsize=10)

            fig_name = f"{today}_{plot_type.replace(' ','_')}_{file_name}.svg"


    if 'unit' not in globals():
        unit = 'signal'   

    #unit = unit if 'unit' not in globals() else "signal"
    pre_plot.pyplot(fig)
    # Convert to BytesIO for download
    buf = BytesIO()
    fig.savefig(buf, format="svg")
    buf.seek(0)
    # Add download button
    st.download_button(
        label="Download Plot as SVG",
        data=buf,
        file_name=fig_name,
        mime="image/svg",
        width='stretch',
    )
    
    csv = convert_for_download(df)
    
    st.sidebar.header('Final steps')

    analysis_button = st.sidebar.button('Run analysis', type='primary', width='stretch')
    st.sidebar.download_button(label="Download clean data",
                    data=csv,
                    file_name=f"{today}_{file_name}_clean_data.txt",
                    mime='text/csv',
                    help='Here you can download your data',
                    width='stretch',)
    
    if analysis_button:
        step = 0
        bar_text = "Working on the analysis"
        my_bar = messages.progress(step, text=bar_text)
        with st.spinner("Running R script..."):
            import time
            start = time.time()

            st.toast('Calculating periods...!')
            
            periods = methods.period_estimation(df, data_cols, t_col, method=period_estimation,
                                                min_period=period_len_min, max_period=period_len_max).rename('Period')
            
            my_bar.progress(step + 25, text=bar_text)

            classification = df[data_cols].apply(lambda x: classifier.predict(x, df[t_col]))
            classification = classification.apply(pd.Series)
            my_bar.progress(step + 50, text=bar_text)
            # Transpose and set index
            test_df = df[df[t_col].between(t_start_test, t_end_test)]
   
            if method == 'PermCosinor':

                def from_dataframe(df, time_col="time"):
                    t = df[time_col].values
                    signal_cols = [c for c in df.columns if c != time_col]
                    data = df[signal_cols].values.T   # shape: (n_signals, n_timepoints)
                    return t, data, signal_cols

                t, data, signal_cols = from_dataframe(test_df, t_col)
                result_df = methods.detect_rhythmicity(
                    t       = t,          # 1D array in hours
                    data    = data,        # shape: (n_genes, n_timepoints)
                    signal_names = signal_cols,
                    n_permutations = 5000,              # increase for publication
                    fdr_alpha = thresh,
                )
                result_df['PermCosinorBH.Q'] = result_df[['q_ftest', 'q_perm']].mean(axis=1)
            else:
                result_df = methods.run_metacycle(
                    test_df, t_col, data_cols,
                    cyc_methods=["JTK", "LS", "ARS"],
                    min_per=period_len_min,
                    max_per=period_len_max,
                    n_replicates=n_replicates
                )
                cols = [col for col in result_df.columns if method in col]

                q_col = [col for col in cols if 'BH.Q' in col.upper()][0]  
                
                result_df['reject'] = np.where(result_df[q_col] <= thresh, True, False)

            my_bar.progress(step + 75, text=bar_text)
          
            result_df = result_df.set_index('CycID')
            result_df['Periods'] = periods
            #result_df = result_df.reset_index()    
            #result_df = result_df.set_index('CycID')
            result_df[classification.columns] = classification
            
            result_df = result_df.reset_index()

            st.session_state["result_df"] = result_df  # Save in session state
            st.session_state["tested_file"] = file_name

            csv = convert_for_download(result_df.set_index('CycID'))
                
            messages2.download_button(label="Download MetaCycle results",
                                data=csv,
                                file_name=f"{today}_{file_name}_stats.txt",
                                mime='text/csv',
                                type='primary',
                                help='Here you can download your data',
                                width='stretch',)
            my_bar.progress(step + 100, text=bar_text)
            messages.dataframe(result_df)
            end = time.time()
            elapsed = end-start
            elapsed = f"{elapsed/60:.2f} min" if elapsed > 60 else f"{elapsed:.2f} s"
            messages2.success(f"""Took {elapsed} to process {result_df.shape[0]} signals with {test_df.shape[0]} timepoints (deltatime = {delta_t} h ) """)

    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
        if st.session_state["tested_file"] != file_name:
            messages.warning(
    "⚠️ Cached results were found that do not match the currently loaded file. "
    "This may cause errors or inconsistent analyses. "
    "Please re-run the analysis for the current file to ensure compatibility."
)
        #export_stats = st.sidebar.button('Export analysis', type='primary', width='stretch')
        stat_csv = convert_for_download(result_df)
        st.sidebar.download_button(label="Export analysis results",
                        data=stat_csv,
                        file_name=f"{today}_{file_name}_stats.txt",
                        mime='text/csv',
                        help='Here you can download your rhythmicity analysis',
                        width='stretch',)
        
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
                            label=":material/docs: Prepare report",
                            width='stretch'
                        )

    if report_button:
        from rhythmicity_report import RhythmicityReport
        report = RhythmicityReport(
            df=df,
            t_col=t_col,
            result_df=result_df if 'result_df' in globals() else None,
            layout_df=layout_df if 'layout_df' in globals() else None,
            phases=phases if 'phases' in globals() else None,
            sum_stats=sum_stats if 'sum_stats' in globals() else None,
            methods=methods,
            method=method,
            thresh=thresh,
            ent=ent,
            ent_days=ent_days,
            ent_color=ent_color,
            bg_color=bg_color,
            unit=unit,
            T=T,
            order=order,
            t0=t0,
            t1=t1,
            conditions=conditions,
            data_cols=data_cols,
            period_len_min=period_len_min,
            period_len_max=period_len_max,
            period_estimation=period_estimation,
            file_name=file_name,
        )
        pdf_buffer = report.build().to_pdf()
        st.toast('Report ready to download!', icon=':material/celebration:')
        report_spot.download_button(
                        label=":material/download: Download report",
                        data=pdf_buffer,
                        file_name=f"{today}_{file_name}_rhythmicity_report.pdf",
                        mime="application/pdf",
                        width='stretch'
                    )
        


