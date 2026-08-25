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
import plots
import plates
import styles
import features as ftx
import docs
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

# Give every widget its tooltip. Keyed on the label, so no call site needs editing
# and any control added later is documented the moment its label is in docs.py.
# Must run before the first widget is created.
docs.attach()

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

version = "0.8.0"

ver_col, doc_col = st.sidebar.columns([3, 2], vertical_alignment="center")
ver_col.write(f"Version {version}")
with doc_col.popover(':material/help: Help', width='stretch'):
    st.markdown(
        "Every control in Chronotopia carries a tooltip — hover the **?** beside it. "
        "The full reference is below, and downloadable."
    )
    st.download_button(
        label="Download the control reference",
        data=docs.as_markdown().encode("utf-8"),
        file_name=f"{today}_chronotopia_v{version}_reference.md",
        mime="text/markdown",
        width='stretch',
    )
    with st.container(height=420):
        st.markdown(docs.as_markdown(title=f"Control reference — v{version}"))

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
    # Call it, don't `import landing` — a module body executes only once per
    # process, so on the second rerun nothing rendered and the trailing st.stop()
    # never fired, leaving a blank page.
    from landing import render_landing
    render_landing()
    st.stop()

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

    # Sampling interval in the file's OWN units — used only to guess the unit below.
    raw_delta_t = methods.sampling_interval(df[t_col].values)

    t_options = ['Minutes', 'Hours', 'Days', 'Seconds']

    if uploaded_file == 'hola':
        default = t_options.index('Hours')
    else:
        if raw_delta_t > 1:
            default = t_options.index('Minutes')
        else:
            default = t_options.index('Hours')

    t_unit = col_unit.selectbox('Time unit', t_options, default)

    data_cols = [col for col in df.columns if col != t_col]

    df[t_col] = df[t_col].apply(lambda x: methods.time_changer(x, t_unit))

    # Recompute AFTER the unit conversion: everything downstream (smoothing window,
    # Butterworth fs, DCT cut-off, entrainment square wave, the header text) treats
    # delta_t as hours. Computing it before the conversion made all of those wrong
    # by the conversion factor for any file not already in hours.
    delta_t = methods.sampling_interval(df[t_col].values)
    if not np.isfinite(delta_t) or delta_t <= 0:
        st.error(
            f"Could not determine a valid sampling interval from column '{t_col}'. "
            "Check that it contains at least two distinct, increasing timepoints."
        )
        st.stop()

    # ── plate detection ──────────────────────────────────────────────────────
    # Supports 6/12/24/48/96/384. Well positions are read from the sample names
    # (A1 / A01 / G8 / sample_H12_ctrl); a bare sample count is only trusted for
    # 96 and 384. Geometry is recorded, but Condition is NOT invented — the old
    # code silently set Condition = plate column for any 96-column file, which
    # produced grouped statistics the user never asked for.
    plate = plates.detect_plate(data_cols)
    plate_group_by = 'None'

    layout_file = None
    
    with layout:
        st.header('Experimental groups')

        if plate is not None:
            st.success(f":material/grid_on: {plate.describe()}")
            plate_group_by = st.radio(
                'Group wells by', ['None', 'Row', 'Column'],
                horizontal=True, width='stretch')

        template = pd.DataFrame(data_cols, columns=['Sample'])
        template['Condition'] = 'YOUR_CONDITION'
        if plate is not None:
            # Pre-fill the template with the wells we found, so the user edits
            # conditions rather than retyping 96 sample names.
            template = template.merge(
                plate.wells[['Sample', 'Well', 'Row', 'Col']], on='Sample', how='left'
            )

        csv = convert_for_download(template)

        st.download_button(label="Download layout template",
                        data=csv,
                        file_name='sample_layout_template.txt',
                        mime='text/csv',
                        type='primary',
                        width='stretch',)
        layout_file = st.file_uploader('Upload your experimental layout',
                                type=['csv','txt','xlsx', 'tsv'])

        if layout_file is not None:

            layout_df = methods.importer(layout_file)

            layout_df['name'] = layout_df.Condition + " - [" + layout_df.Sample + "]"

            name_dict = dict(zip(layout_df.Sample, layout_df.name))
            df = df.rename(columns=name_dict)
            data_cols = [col for col in df.columns if col != t_col]

        elif plate is not None and plate_group_by != 'None':
            # Opt-in geometric grouping. Only reached when no real layout was
            # uploaded — an uploaded layout always wins.
            layout_df = plate.wells[['Sample']].copy()
            layout_df['Condition'] = plates.group_by_geometry(plate, plate_group_by).values
            layout_df['name'] = layout_df.Sample

    # Wells whose sample was renamed by an uploaded layout must follow, or the
    # plate view would look for columns that no longer exist.
    if plate is not None and layout_file is not None:
        plate.wells['Sample'] = plate.wells['Sample'].replace(name_dict)


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

    # A damped sinusoid has five free parameters, so it needs several cycles to be
    # identifiable — on a two-cycle recording the damping and the amplitude trade
    # off against each other and the fit is meaningless. Three cycles at the
    # longest period in range is the floor.
    if float(df[t_col].max() - df[t_col].min()) >= 72:
        period_methods = period_methods + ['Damped Cosinor']
    
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
    
    hourly = col0.selectbox('Smoothening', ['None', 'Mean', 'Savitzky-Golay', 'Resample', 'DCT'])

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
    # Estimating the baseline and removing it are separate choices; the dropdown
    # picks the estimator, the radio underneath picks how it comes out. The five
    # original names are kept in their original positions, so a saved settings
    # string, a report, or one of the verify scripts still resolves.
    detrend_meth = col2.selectbox(
        'Detrending',
        ['None', 'Linear', 'Rolling mean', 'Rolling Hilbert', 'Cubic',
         'LOESS', 'Exponential fit'])

    # "Rolling Hilbert" already divides — by the envelope, not the baseline — so it
    # has no removal choice to make.
    if detrend_meth in methods.BASELINE_METHODS:
        removal_meth = col2.radio('Baseline removal', ['Subtract', 'Divide'],
                                  horizontal=True)
    else:
        removal_meth = 'Subtract'

    #df[data_cols] = methods.detrend(df, data_cols, t_col, detrend_meth)
    #df[data_cols] = methods.normalize(df, data_cols, norm_meth)

    if hourly == 'Savitzky-Golay':
        with settings:
            sg1, sg2 = st.columns([2, 1], vertical_alignment="top")
            sg_window = sg1.slider(
                'Savitzky-Golay window (h)', 1.0, 24.0, 6.0, step=0.5)
            sg_order = sg2.selectbox(
                'Polynomial degree', [1, 2, 4], 1)
        smoothed, sg_note = methods.savitzky_golay(
            df, data_cols, delta_t, window_h=sg_window, polyorder=sg_order
        )
        df[data_cols] = smoothed
        if sg_note:
            st.info(sg_note)

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

    # The period range is passed in so the rolling window can default to the period
    # being measured. A moving average is a perfect notch only at window == period;
    # the old fixed ~20 h window scaled amplitudes by 0.65-1.2 depending on period.
    df[data_cols] = methods.detrend(df, data_cols, t_col, detrend_meth,
                                    period_range=(period_len_min, period_len_max),
                                    removal=removal_meth)
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
            t1, t2 = st.columns(2, vertical_alignment="top")
            t_start_test = t1.number_input('Minimum time', int(fr_data[t_col].min()), int(fr_data[t_col].max()), int(fr_data[t_col].min()),  step=1)    
            t_end_test = t2.number_input('Last time', int(df[df[t_col] > t_start_test][t_col].min()), int(df[df[t_col] > t_start_test][t_col].max()), int(df[df[t_col] > t_start_test][t_col].max()), step=1) 
            # Tempo is one of the methods now, not an extra verdict stapled onto
            # whichever method you picked. It used to run on every analysis, so
            # every result table and every report carried a second classification
            # nobody asked for — and cost ~1.2 s per 48 samples to produce it.
            method = t1.selectbox(
                'Testing method', ['meta2d', 'JTK', 'ARS', 'LS', 'PermCosinor', 'Tempo'], 0)
            if method == 'Tempo':
                p_min = t2.selectbox(
                    'Minimum rhythmic probability', [0.5, 0.7, 0.8, 0.9, 0.95], 0)
                # Stored as 1 - p so the q-value machinery downstream (pie charts,
                # group summaries, comparisons) keeps working unchanged.
                thresh = round(1 - p_min, 4)
                # The model picker only makes sense once Tempo is the method. It used
                # to sit at the top of these parameters for every method, implying a
                # model was involved in a JTK or meta2d run.
                load_model = t2.selectbox('Model', list(model.keys()), 0)
            else:
                load_model = next(iter(model))
                thresh = t2.selectbox('Significance threshold',
                                      [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], 0)


    @st.cache_resource(show_spinner="Loading rhythmicity model…")
    def load_classifier(model_path, feature_names_path):
        """
        Unpickle the classifier once per session instead of once per rerun.

        This used to run unconditionally at the top of every script execution —
        a 1.3 MB Random Forest read from disk on every slider drag, even for a
        meta2d run that never touches it. It is now built on demand by the four
        call sites that actually need it (the Tempo analysis, the Lineplot Tempo
        pill, PCA and the Rhythmicity Model Evaluation view).
        """
        from ML_classifier.ml_rhythmicity_classifier import MLRhythmicityClassifier
        return MLRhythmicityClassifier(
            model_path=model_path, feature_names_path=feature_names_path
        )


    def get_classifier():
        selected = model[load_model]
        return load_classifier(selected['model_path'], selected['feature_names_path'])


    @st.cache_data(show_spinner=False)
    def cached_features(frame, time_col, sample_names, packages):
        """The full feature matrix, cached. The page used to re-extract all 8
        packages on every widget interaction — 3x the work it displayed."""
        from chronotopia_feature_extractor import ChronotopiaFeatureExtractor
        # Short traces produce empty peak lists, so the extractor emits a stream of
        # "Mean of empty slice" RuntimeWarnings for behaviour that is expected.
        # Silenced here rather than by editing the extractor.
        with ftx.silence_extractor_warnings():
            return ChronotopiaFeatureExtractor.extract_batch(
                frame, t_col=time_col, data_cols=list(sample_names),
                packages=list(packages) if packages else None, verbose=False,
            )


    @st.cache_data(show_spinner=False)
    def cached_sweep(frame, time_col, sample_names, p_min, p_max, p_step):
        """One sweep per (data, range) combination — changing the R² slider or the
        peak count then costs nothing."""
        return methods.sine_sweep(frame, time_col, list(sample_names),
                                  period_min=p_min, period_max=p_max,
                                  period_step=p_step)


    @st.cache_data(show_spinner=False)
    def cached_plate_features(frame, time_col, sample_names):
        """
        Cosinor fit per well, for the plate overlays.

        Only the `cosinor` package runs: period, amplitude, R² and residual SD all
        come out of the same fit, so the other seven packages would be pure cost.
        Cached because otherwise every change of overlay, label or colour re-fits
        all 96 wells.
        """
        from chronotopia_feature_extractor import ChronotopiaFeatureExtractor
        return ChronotopiaFeatureExtractor.extract_batch(
            frame, t_col=time_col, data_cols=list(sample_names),
            packages=plates.METRIC_PACKAGES, verbose=False,
        )


    duration = np.round(df[t_col].max(),1)
    sum_pre.write(f"Experiment with {len(data_cols)} sample recorded for {duration} hours (recorded every = {delta_t:.1f} h)")
    
    conditions = []
    visu = ['Lineplot', 'Actogram', 'Multi-actogram', 'Feature extraction', 'Sample Insights', 'Rhythmicity Model Evaluation', 'PCA',  'Correlation', ]
    
    if len(data_cols) >= 2:

        visu = visu + ['Compare samples', 'Period sweep']

    if 'layout_df' in globals():

        conditions = list(layout_df.Condition.unique())

        visu = visu + ['Lineplot [Mean ± SD]', 'Lineplot [Mean + Replicates]']

        if len(conditions) >= 2:

            visu = visu + ['Compare conditions']

    if period_estimation == 'Wavelet Transform':
        
        visu = visu + ['Wavelet Ridge']
    
    if ent_days > 0:

        visu = visu + ['Phase plot']

    if plate is not None:

        visu = visu + ['Plate view']

    viz_settings = st.expander('Visualization settings (Plot type, sample selection, data unit...)')

    with viz_settings:
    
        c, c1, c2 = st.columns([3, 2, 2], vertical_alignment="top")
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
        
        cus_ticks1, cus_ticks2 = st.columns([2,1], vertical_alignment="top")
        cus1, cus2, cus3 = st.columns(3, vertical_alignment="top")

        style = cus1.selectbox(
            'Select style', styles.STYLE_NAMES,
            styles.STYLE_NAMES.index(styles.DEFAULT_STYLE),
            help="\n\n".join(f"**{k}** — {v['help']}" for k, v in styles.STYLES.items()),
        )
        context = cus2.selectbox(
            'Select context', styles.CONTEXTS,
            styles.CONTEXTS.index(styles.DEFAULT_CONTEXT))

        # The palette list used to be SEABORN_PALETTES + plt.colormaps() — 204 entries,
        # ~180 of them continuous ramps. Cycling a sequential ramp across categorical
        # series gives poor separation and implies an ordering that isn't there.
        all_maps = cus_ticks2.checkbox(
            'All colormaps', False)
        if all_maps:
            pal_options = styles.PALETTE_NAMES + [
                m for m in styles.all_colormap_names() if m not in styles.PALETTE_NAMES
            ]
        else:
            pal_options = styles.PALETTE_NAMES

        palette = cus3.selectbox(
            'Select palette', pal_options,
            pal_options.index(styles.DEFAULT_PALETTE),
            format_func=styles.palette_label,
            help=styles.palette_help(),
        )

        editable_svg = cus_ticks1.checkbox(
            'Editable text in exports', True)

        style_facecolor = styles.apply(style, context, palette, editable_text=editable_svg)

        if bg_color == 'white':
            # Each style sets its own panel colour; the entrainment shading has to
            # sit on the same background or the band edges show as seams.
            bg_color = style_facecolor


        if plot_type == 'Lineplot':
            
            p_col = st.selectbox('Column to preview', data_cols)
            unit = st.text_input('Data unit', 'Measured unit')
            per = methods.period_estimation(fr_data, [p_col], t_col, method=period_estimation, 
                                            min_period=period_len_min, max_period=period_len_max)
            per = np.round(per, 2)
            ft_col, clf_col = st.columns([2, 1])
            # 'Tempo' only appears as a separate preview when it is NOT already the
            # selected method — otherwise the pill row offered the same check twice.
            eval_options = ['None', method] if method == 'Tempo' else ['None', method, 'Tempo']
            ml_included = st.pills('Rhythmicity evaluation', eval_options,
                                   default='None', width='stretch')

            fig = plots.plot(df, t_col, p_col, t0, t1, bg_color=bg_color, ent=ent, features=entrainment_feat_data,
                         ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)

            if ml_included == method and method != 'Tempo':
                # Was `if ml_included == 'PermCosinor'` — true only because
                # ml_included equals method inside this branch. It read as a bug
                # and became one as soon as a method without a q-value existed.
                if method == 'PermCosinor':
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
                    # Use the same window and period range as the batch run below,
                    # otherwise the q-value shown here can disagree with the exported
                    # table for the very same sample.
                    result = methods.run_metacycle(
                        df[df[t_col].between(t_start_test, t_end_test)], t_col, [p_col],
                        cyc_methods=[method] if method != 'meta2d' else ['JTK', 'ARS', 'LS'],
                        min_per=period_len_min,
                        max_per=period_len_max,
                        n_replicates=n_replicates,
                    )
                    if result is None:
                        st.stop()
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

            if ml_included == 'Tempo':
                classification = get_classifier().predict(df[p_col], df[t_col])
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
            
            fig = plots.grouped_plot(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                     ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)

            pre_plot.pyplot(fig)
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"


        elif plot_type == 'Lineplot [Mean + Replicates]':
                
            p_col = st.selectbox('Column to preview', conditions)
            unit = st.text_input('Data unit', 'Measured unit')
            #pre_plot = st.empty()
            
            fig = plots.grouped_plot_traces(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
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

            fig = plots.double_plot(df_plot, t_col, p_col, ent_days, T, order, t0=t0, t1=t1, times=times,
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
                plots.multi_acto(ax, df_plot, t_col, plot_cols[i], ent_days, T, order, t0=t0, t1=t1, times=times, 
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
            
            plots.plot_entrainment_ax(ax, entrain_data, t_col, xtick_start, xtick_end,
                                           ent_days, order=order, T=T, color=ent_color)

            ax.set_xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
            ax.set_ylabel(unit)
            ax.set_xlabel('Time (h)')
                
            plots.phase_plot(entrain_data, ax2, peaks, pal=[bg_color, ent_color], order=order)
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"


        elif plot_type == 'Wavelet Ridge':
            
            p_col = st.selectbox('Column to preview', data_cols)
            unit = st.text_input('Data unit', 'Measured unit')
            
            signal = df[p_col]
            # Honour the sidebar "Period range" slider instead of a hardcoded 18-36 h.
            periods = np.linspace(period_len_min, period_len_max, 100)
            dt = delta_t

            wAn = WAnalyzer(periods, dt, p_max=20)

            wAn.compute_spectrum(signal)

            # power_thresh was hardcoded to 10. On normalised or detrended traces the
            # wavelet power never reaches it, the ridge comes back empty, and the
            # np.average() below dies with "Weights sum to zero, can't be normalized".
            # Scale the threshold to the spectrum actually in front of us.
            spectrum_max = float(np.nanmax(wAn.modulus)) if wAn.modulus is not None else 0.0
            power_thresh = max(0.0, 0.1 * spectrum_max)
            # pyboat wants an odd smoothing window, shorter than the series
            smoothing = min(len(signal) // 4 * 2 + 1, 21)
            smoothing = smoothing if smoothing >= 5 else None

            wAn.get_maxRidge(power_thresh=power_thresh, smoothing_wsize=smoothing)

            rd = wAn.ridge_data # this is a pandas DataFrame holding the ridge results

            if rd is None or len(rd) == 0 or float(np.nansum(rd['power'])) <= 0:
                st.warning(
                    f"No wavelet ridge could be extracted for **{p_col}** in the "
                    f"{period_len_min}-{period_len_max} h band. The signal has no sustained "
                    "oscillation strong enough to track — widen the period range, reduce "
                    "detrending, or pick another sample."
                )
                st.stop()

            #fig, ax = plt.subplots(1, 2, layout='constrained', gridspec_kw={'width_ratios': [4, 1]})
            fig, axes = plt.subplot_mosaic("AAAAC;DDDDD", layout='constrained')

            wAn.draw_Ridge()
            sns.kdeplot(
                    rd, y='periods', x='time',
                    fill=True, thresh=0, levels=100, cmap="viridis",
                        bw_adjust=0.5, # smoother KDE
                    clip=((rd['time'].min(), rd['time'].max()),
                          (period_len_min, period_len_max)), ax=axes['A']
                )
                #plt.ylim(18, 36)
            sns.lineplot(rd, x='time', y='periods', color='w', ax=axes['A'])

            sns.kdeplot(rd, y='periods', fill=True, ax=axes['C'])
            axes['D'].plot(df[t_col], df[p_col])
            plt.suptitle(f"{p_col} Estimated period: {np.average(rd.periods, weights=rd.power):.2f} h")
            fig_name = f"{today}_{p_col}_{plot_type.replace(' ','_')}_{file_name}.svg"

        elif plot_type in ('Compare samples', 'Compare conditions'):

            is_groups = plot_type == 'Compare conditions'
            cap = plots.MAX_COMPARE_GROUPS if is_groups else plots.MAX_COMPARE_SAMPLES

            if is_groups:
                picked = st.multiselect(
                    f'Conditions to compare (2–{cap})', conditions,
                    conditions[:min(2, len(conditions))], max_selections=cap,
                )
            else:
                picked = st.multiselect(
                    f'Samples to compare (up to {cap})', data_cols,
                    data_cols[:min(2, len(data_cols))], max_selections=cap,
                )

            cc1, cc2, cc3 = st.columns([1.3, 1, 1], vertical_alignment="top")
            unit = cc1.text_input('Data unit', 'Measured unit')
            if is_groups:
                cmp_style = cc2.selectbox('Style', ['Mean ± SD', 'Mean + Replicates'], 0)
            else:
                cmp_points = cc2.checkbox('Show datapoints', False)
            cmp_safe = cc3.checkbox(
                'Accessible colours', True)

            cmp_palette = None if cmp_safe else sns.color_palette(palette).as_hex()

            if len(picked) < (2 if is_groups else 1):
                st.info(
                    f"Select at least {'two conditions' if is_groups else 'one sample'} "
                    f"to draw the comparison."
                )
                st.stop()

            if is_groups:
                fig = plots.compare_groups(
                    df, t_col, picked, layout_df, t0, t1, style=cmp_style,
                    unit=unit, bg_color=bg_color, features=entrainment_feat_data,
                    ent_days=ent_days, order=order, T=T, color=ent_color,
                    palette=cmp_palette,
                )
                fig_name = f"{today}_{'_vs_'.join(str(g) for g in picked)}_{plot_type.replace(' ','_')}_{file_name}.svg"
            else:
                fig = plots.compare_samples(
                    df, t_col, picked, t0, t1,
                    unit=unit, bg_color=bg_color, features=entrainment_feat_data,
                    ent_days=ent_days, order=order, T=T, color=ent_color,
                    palette=cmp_palette, show_points=cmp_points,
                    n_replicates=n_replicates,
                )
                fig_name = f"{today}_{plot_type.replace(' ','_')}_{file_name}.svg"

            with st.expander('Period of the compared traces'):
                cmp_cols = picked if not is_groups else [
                    s for g in picked
                    for s in layout_df[layout_df.Condition == g]['name'].to_list()
                ]
                cmp_cols = [c for c in cmp_cols if c in fr_data.columns]
                if cmp_cols:
                    per_cmp = methods.period_estimation(
                        fr_data, cmp_cols, t_col, method=period_estimation,
                        min_period=period_len_min, max_period=period_len_max,
                    )
                    st.dataframe(np.round(per_cmp, 2).rename(f'Period (h) — {period_estimation}'))

        elif plot_type == 'Period sweep':

            st.caption(
                "Fits a sinusoid at every trial period to every signal and asks which "
                "periods the dataset actually contains — not whether one trace is "
                "rhythmic. Built for wide datasets: a transcriptome sweeps in about a "
                "second."
            )

            sw1, sw2, sw3 = st.columns([2, 1, 1], vertical_alignment="top")
            sw_range = sw1.slider('Period range to sweep (h)', 1, 72, (2, 30), step=1)
            sw_step = sw2.selectbox('Resolution (h)', [0.5, 0.25, 0.1, 0.05], 2)
            sw_r2 = sw3.slider('Minimum R²', 0.0, 0.95, 0.30, step=0.05)

            sw4, sw5 = st.columns([1, 1], vertical_alignment="top")
            sw_peaks = sw4.number_input('Peaks to label', 0, 10, 3, step=1)
            sw_group = False
            if 'layout_df' in globals() and len(conditions) >= 2:
                sw_group = sw5.checkbox(
                    'Split by condition', False)

            sweep_sets = {}
            with st.spinner(f"Sweeping {len(data_cols)} signals…"):
                if sw_group:
                    for cond in conditions:
                        g_cols = [c for c in layout_df[layout_df.Condition == cond]['name']
                                  if c in df.columns]
                        if len(g_cols) >= 1:
                            sweep_sets[cond] = cached_sweep(
                                df, t_col, tuple(g_cols), float(sw_range[0]),
                                float(sw_range[1]), float(sw_step))
                else:
                    sweep_sets['All signals'] = cached_sweep(
                        df, t_col, tuple(data_cols), float(sw_range[0]),
                        float(sw_range[1]), float(sw_step))

            sweep_results = {k: v[0] for k, v in sweep_sets.items()}
            sweep_lands = {k: v[1] for k, v in sweep_sets.items()}

            # Peaks are found on the pooled landscape so the same reference lines
            # sit under every group, making the comparison a comparison.
            pooled = (pd.concat(sweep_lands.values())
                        .groupby('period', as_index=False).mean())
            sweep_pk = (methods.sweep_peaks(pooled, n_peaks=int(sw_peaks))
                        if sw_peaks else None)

            first = next(iter(sweep_results.values()))
            if first.attrs.get('nyquist_clamped'):
                asked, used = first.attrs['nyquist_clamped']
                st.info(
                    f"Sampling is {delta_t:.2f} h, so periods below {used:.1f} h "
                    f"(twice the interval) cannot be resolved. Swept from {used:.1f} h "
                    f"instead of {asked:g} h."
                )
            if first.attrs.get('n_missing_filled'):
                st.caption(f"{first.attrs['n_missing_filled']} missing value(s) filled "
                           f"with each signal's mean before fitting.")

            fig = plots.period_sweep(
                sweep_lands, sweep_results, r2_thresh=sw_r2, peaks=sweep_pk,
                period_min=sw_range[0], period_max=sw_range[1],
                title=f"Period sweep · {file_name}",
            )
            fig_name = f"{today}_Period_sweep_{file_name}.svg"

            if sweep_pk is not None and len(sweep_pk):
                st.dataframe(
                    sweep_pk.rename(columns={'period': 'Period (h)',
                                             'mean_r2': 'Mean R²',
                                             'prominence': 'Prominence'}),
                    hide_index=True, width='stretch',
                )

            sweep_table = pd.concat(
                [r.assign(group=k) for k, r in sweep_results.items()], ignore_index=True)
            st.download_button(
                label="Export sweep results",
                data=convert_for_download(sweep_table.set_index('sample')),
                file_name=f"{today}_{file_name}_period_sweep.txt",
                mime='text/csv', width='stretch', type='primary'
            )

        elif plot_type == 'Plate view':

            st.caption(plate.describe())

            pv1, pv2, pv3 = st.columns([1, 1, 1], vertical_alignment="top")
            pv_scale = pv1.selectbox(
                'Y scaling', ['Shared across plate', 'Per well'], 0)
            pv_color = pv2.color_picker('Trace color', '#1F7A8C')

            plate_results = st.session_state.get("result_df") \
                if st.session_state.get("tested_file") == file_name else None
            plate_qcol = None
            if plate_results is not None:
                _qc = [c for c in plate_results.columns
                       if method in c and 'BH.Q' in c.upper()]
                plate_qcol = _qc[0] if _qc else None

            pv_metric = pv3.selectbox(
                'Overlay', ['None'] + plates.metric_names(has_results=plate_qcol is not None),
                0)

            lbl1, lbl2, lbl3 = st.columns([1, 1, 1.4], vertical_alignment="top")
            pv_names = lbl1.checkbox('Label with sample name', False)
            pv_wells = lbl2.checkbox('Label with well ID', False)
            pv_mask = False
            if pv_metric != 'None' and plate_qcol is not None \
                    and plates.METRICS[pv_metric]['kind'] != 'status':
                pv_mask = lbl3.checkbox(
                    'Grey out non-rhythmic wells', True)

            missing = [s for s in plate.wells['Sample'] if s not in df.columns]
            if missing:
                st.caption(
                    f"{len(missing)} well(s) are not plotted — excluded from the data "
                    f"or dropped during preprocessing."
                )

            # Replicate timepoints would draw as a zig-zag in a panel this small;
            # average them, as the actogram views already do.
            df_plate = (df.groupby(t_col).agg({c: 'mean' for c in data_cols}).reset_index()
                        if np.mean(n_replicates) > 1 else df)

            well_colors, well_labels, overlay_legend = None, None, None
            if pv_metric != 'None':
                plate_samples = [s for s in plate.wells['Sample'] if s in df_plate.columns]
                spec = plates.METRICS[pv_metric]

                feats = None
                if spec.get('feature'):
                    with st.spinner(f"Fitting {len(plate_samples)} wells…"):
                        feats = cached_plate_features(df_plate, t_col, tuple(plate_samples))

                metric_values = plates.compute_metric(
                    pv_metric, df_plate, t_col, plate_samples, features=feats,
                    result_df=plate_results, q_col=plate_qcol, thresh=thresh,
                )

                mask = ()
                if pv_mask and plate_qcol is not None:
                    flags = plate_results.set_index('CycID')[plate_qcol] <= thresh
                    mask = [s for s in plate_samples if not bool(flags.get(s, False))]

                well_colors, overlay_legend = plates.build_overlay(
                    pv_metric, metric_values, mask=mask)
                well_labels = plates.format_labels(metric_values, pv_metric, mask=mask)

            fig = plates.plot_plate(
                df_plate, t_col, plate,
                t0=t0, t1=t1,
                shared_y=(pv_scale == 'Shared across plate'),
                line_color=pv_color,
                bg_color=bg_color,
                show_sample_names=pv_names,
                show_well_ids=pv_wells,
                well_colors=well_colors,
                annotations=well_labels,
                legend=overlay_legend,
                title=f"{plate.label} · {file_name}"
                      + (f" · {pv_metric}" if pv_metric != 'None' else ""),
            )
            suffix = f"_{pv_metric.split()[0]}" if pv_metric != 'None' else ""
            fig_name = f"{today}_{plot_type.replace(' ','_')}{suffix}_{file_name}.svg"

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

            clf = get_classifier()
            classification = df[selected_ids].apply(lambda x: clf.predict(x, df[t_col]))
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

            clf = get_classifier()
            classification = df[selected_ids].apply(lambda x: clf.predict(x, df[t_col]))
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
            # Defined once and reused in the panel titles below, so the boundaries and
            # the labels the user reads can never drift apart again.
            P_HIGH, P_LOW = 0.70, 0.30
            p_rhythmic = classification.probability_rhythmic
            high   = classification[p_rhythmic >= P_HIGH].index.tolist()
            medium = classification[(p_rhythmic >= P_LOW) & (p_rhythmic < P_HIGH)].index.tolist()
            low    = classification[p_rhythmic < P_LOW].index.tolist()

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
                (ax_high,   high,   f'High probability  (≥ {P_HIGH:.2f})'),
                (ax_medium, medium, f'Medium probability  ({P_LOW:.2f} – {P_HIGH:.2f})'),
                (ax_low,    low,    f'Low probability  (< {P_LOW:.2f})'),
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

            with st.spinner(f"Extracting features for {len(data_cols)} samples…"):
                features_df = cached_features(df, t_col, tuple(data_cols), None)

            if 'layout_df' in globals():
                lay_dict = dict(zip(layout_df['name'], layout_df['Condition']))
                features_df = features_df.copy()
                features_df['Condition'] = features_df['sample_id'].replace(lay_dict)

            fx_mode = st.radio(
                'View', ['Single feature', 'Compare conditions', 'Feature quality', 'QC samples'],
                horizontal=True, width='stretch')

            dictionary = ftx.describe_features(features_df.columns)

            # ── the original per-feature view, now grouped by concept ────────
            if fx_mode == 'Single feature':
                c_col, f_col = st.columns([1, 1.5])
                concepts = [c for c in ftx.CONCEPT_ORDER
                            if any(dictionary.concept == c)]
                concept = c_col.selectbox('Feature group', concepts)
                opts = list(dictionary[dictionary.concept == concept]['feature'])
                feature_col = f_col.selectbox('Feature', opts)

                st.info(ftx.CONCEPTS.get(concept, ''))
                row = dictionary[dictionary.feature == feature_col].iloc[0]
                st.caption(f"**{feature_col}** — {row.description} "
                           f"(from the `{row['package']}` package"
                           + (", describes the RECORDING, not the biology"
                              if row.role == ftx.RECORDING else "") + ")")

                grouped = 'Condition' in features_df.columns
                l = (features_df['Condition'].nunique() * 0.5 + 1.5) if grouped \
                    else (len(data_cols) * 0.2 + 1.0)
                fig, ax = plt.subplots(figsize=(6, max(2.0, l)))
                ykey = 'Condition' if grouped else 'sample_id'
                sns.boxplot(features_df, y=ykey, x=feature_col, hue=ykey,
                            legend=False, ax=ax)
                sns.stripplot(features_df, y=ykey, x=feature_col, alpha=0.8,
                              edgecolor='k', linewidth=1, s=8, hue=ykey,
                              legend=False, ax=ax)
                ax.set_ylabel('')
                fig_name = f"{today}_{feature_col}_{file_name}.svg"

            # ── every feature at once ────────────────────────────────────────
            elif fx_mode == 'Compare conditions':
                if 'Condition' not in features_df.columns or len(conditions) < 2:
                    st.info("Upload an experimental layout with at least two "
                            "conditions to compare features between groups.")
                    st.stop()

                g1, g2, g3 = st.columns([1, 1, 1], vertical_alignment="top")
                cond_a = g1.selectbox('Group A', conditions, 0)
                cond_b = g2.selectbox('Group B', conditions,
                                      1 if len(conditions) > 1 else 0)
                fx_test = g3.selectbox(
                    'Test', ['auto', 'parametric', 'rank'], 0,
                    help="auto uses a rank-based test when the smaller group has "
                         f"fewer than {ftx.PARAMETRIC_MIN_N} samples, because a "
                         "t-test at that size rests on a normality assumption "
                         "nobody can check. Whichever runs is stated on the figure.",
                )
                if cond_a == cond_b:
                    st.info("Pick two different conditions.")
                    st.stop()

                try:
                    fx_res, fx_meta = ftx.compare_conditions(
                        features_df, 'Condition', cond_a, cond_b,
                        test=fx_test, alpha=thresh if thresh < 0.5 else 0.05)
                except ValueError as exc:
                    st.warning(str(exc))
                    st.stop()

                fig = plots.feature_volcano(fx_res, fx_meta,
                                            alpha=fx_meta['alpha'])
                fig_name = f"{today}_{cond_a}_vs_{cond_b}_features_{file_name}.svg"

                st.caption(
                    f"Corrected across all {fx_meta['n_tested']} features tested. "
                    "Browsing feature-by-feature and reporting the one that looks "
                    "different is the same number of comparisons, just uncounted."
                )
                show = fx_res.copy()
                show['|effect|'] = show['effect'].abs()
                st.dataframe(
                    show.sort_values('|effect|', ascending=False)
                        [['feature', 'concept', 'median_a', 'median_b', 'effect',
                          'p', 'q', 'significant']]
                        .rename(columns={'median_a': f'median {cond_a}',
                                         'median_b': f'median {cond_b}',
                                         'effect': fx_meta['effect_name']}),
                    hide_index=True, width='stretch', height=280,
                )
                st.download_button(
                    "Export comparison", convert_for_download(fx_res.set_index('feature')),
                    file_name=f"{today}_{file_name}_{cond_a}_vs_{cond_b}_features.txt",
                    mime='text/csv', width='stretch')

            # ── is this table fit to train on? ───────────────────────────────
            elif fx_mode == 'Feature quality':
                q = ftx.quality_report(features_df)
                n_bad = int((~q.usable).sum())
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Features", len(q))
                m2.metric("Unusable", n_bad,
                          help="Non-numeric, constant, or entirely missing.")
                m3.metric("Any missing", int((q.missing_pct > 0).sum()))
                m4.metric("Recording, not biology",
                          int((q.role == ftx.RECORDING).sum()))

                if int((q.missing_pct > 0).sum()):
                    st.warning(
                        "Missing values here are structural, not random: the extractor "
                        "routes short recordings away from some packages, so whole "
                        "blocks of columns are absent for the shorter samples. That "
                        "absence correlates with recording length — impute with care."
                    )
                if int((q.role == ftx.RECORDING).sum()):
                    st.warning(
                        "Columns flagged `recording` describe the measurement, not the "
                        "biology — duration, number of points, sampling interval. They "
                        "are useful for QC and risky as model inputs: if recording "
                        "length happens to track your conditions, a model can score "
                        "well without looking at the rhythm. They stay in the export; "
                        "the flag is there so the choice is yours."
                    )

                st.dataframe(q, hide_index=True, width='stretch', height=260)

                cl = ftx.redundancy_clusters(features_df, threshold=0.95)
                if len(cl):
                    st.caption(
                        f"{cl.cluster.nunique()} groups of features move together at "
                        f"|rho| >= 0.95, covering {len(cl)} of {len(q)} columns — so the "
                        "table holds fewer independent measurements than it has columns."
                    )
                    st.dataframe(cl, hide_index=True, width='stretch', height=200)

                fig, ax = plt.subplots(figsize=(7, 3.2))
                counts = dictionary.concept.value_counts().reindex(
                    [c for c in ftx.CONCEPT_ORDER if c in set(dictionary.concept)])
                ax.barh(counts.index[::-1], counts.values[::-1], color='#2a78d6')
                ax.set_xlabel('Features'); ax.set_title('What the table measures',
                                                        loc='left', fontsize=11.5)
                fig_name = f"{today}_feature_quality_{file_name}.svg"

                st.download_button(
                    "Export data dictionary", convert_for_download(dictionary.set_index('feature')),
                    file_name=f"{today}_{file_name}_feature_dictionary.txt",
                    mime='text/csv', width='stretch')

            # ── which samples look unreliable ────────────────────────────────
            else:
                qc = ftx.qc_flags(features_df)
                v = qc.verdict.value_counts()
                m1, m2, m3 = st.columns(3)
                m1.metric("Pass", int(v.get('pass', 0)))
                m2.metric("Warn", int(v.get('warn', 0)))
                m3.metric("Fail", int(v.get('fail', 0)))
                st.caption("Thresholds are cohort-relative where possible — "
                           "\"noisier than 95% of this plate\" travels between "
                           "instruments in a way that an absolute cut-off does not. "
                           "Rules applied: " + ", ".join(qc.attrs.get('rules_applied', [])))
                st.dataframe(qc[qc.n_flags > 0][['sample_id', 'verdict', 'flags', 'reasons']],
                             hide_index=True, width='stretch', height=260)

                flagged = qc[qc.verdict == 'fail']['sample_id'].tolist()
                if flagged:
                    st.download_button(
                        "Export exclusion list", "\n".join(flagged).encode(),
                        file_name=f"{today}_{file_name}_qc_exclude.txt",
                        mime='text/plain', width='stretch', type='primary')

                fig, ax = plt.subplots(figsize=(6, 3))
                order = ['pass', 'warn', 'fail']
                ax.bar(order, [int(v.get(k, 0)) for k in order],
                       color=['#1baf7a', '#eda100', '#d55e00'])
                ax.set_ylabel('Samples'); ax.set_title('QC verdict', loc='left',
                                                       fontsize=11.5)
                fig_name = f"{today}_qc_{file_name}.svg"

            stat_csv = convert_for_download(features_df)
            st.download_button(label="Export features",
                            data=stat_csv,
                            file_name=f"{today}_{file_name}_features.txt",
                            mime='text/csv',
                            width='stretch', type='primary')

        elif plot_type == 'Sample Insights':

            p_col = st.selectbox('Column to preview', data_cols)
            unit = st.text_input('Data unit', 'Measured unit')
            per = methods.period_estimation(fr_data, [p_col], t_col, method=period_estimation, 
                                            min_period=period_len_min, max_period=period_len_max)
            per = np.round(per, 2)

            from chronotopia_feature_extractor import ChronotopiaFeatureExtractor

            # Honour the sidebar period range. This was hardcoded to (18, 30), so
            # Sample Insights could disagree with every other view in the app.
            ext = ChronotopiaFeatureExtractor(
                df[p_col], df[t_col],
                period_range=(float(period_len_min), float(period_len_max)),
            )

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

            ft_col, clf_col = st.columns([3, 2])

            extracted_ft = ft_col.selectbox("Features package", dispatch.keys())
            features = ext.extract(packages=[dispatch[extracted_ft]['feature']])
            ml_included = clf_col.pills('Rhythmicity evaluation (ML)', ['Include', 'Exclude'], default='Exclude', width='stretch')
            with st.expander(f'Show {extracted_ft} features'):
                st.write(features)
            # Visualisation — overlay a package on an existing matplotlib axis
            si_view = st.radio(
                'Show', ['Package overlay', 'Cohort context'], horizontal=True,
                width='stretch')

            if si_view == 'Cohort context':
                if len(data_cols) < 4:
                    st.info("Cohort context needs at least 4 samples to compare against.")
                    st.stop()
                with st.spinner(f"Extracting features for {len(data_cols)} samples…"):
                    cohort = cached_features(df, t_col, tuple(data_cols), None)
                si_pct = ftx.cohort_percentiles(cohort, p_col)
                if si_pct.empty:
                    st.info("No feature could be compared across the cohort.")
                    st.stop()
                si_n = st.slider('Features to show', 6, 40, 18, step=2)
                fig = plots.cohort_context(si_pct, p_col, top_n=si_n)
                extreme = si_pct.head(3)
                st.caption(
                    "Most unusual: " + ", ".join(
                        f"**{r.feature}** ({r.percentile:.0f}th pct)"
                        for _, r in extreme.iterrows())
                    + f" — against {int(si_pct['n_cohort'].max())} other samples."
                )
                with st.expander('Full percentile table'):
                    st.dataframe(si_pct, hide_index=True, width='stretch')
                fig_name = f"{today}_{p_col}_cohort_context_{file_name}.svg"
                pre_plot.pyplot(fig)
                st.stop()

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df[t_col], df[p_col])
            dispatch[extracted_ft]['plot'](ax)
            if ml_included == 'Include':
                classification = get_classifier().predict(df[p_col], df[t_col])
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

            my_bar.progress(step + 50, text=bar_text)
            # Transpose and set index
            test_df = df[df[t_col].between(t_start_test, t_end_test)]

            if method == 'Tempo':
                # The ML classifier as a first-class method. Produces the same
                # column shape as the statistical methods so everything
                # downstream — pie charts, group summaries, the report — works
                # without special-casing: q = 1 - P(rhythmic), and `thresh` was
                # set to 1 - the probability cutoff chosen in the sidebar.
                clf = get_classifier()
                classification = test_df[data_cols].apply(
                    lambda x: clf.predict(x, test_df[t_col])
                ).apply(pd.Series)
                # `features` is a dict of the 18 model inputs per sample — useful for
                # debugging, useless in a results table and it bloats the CSV export.
                classification = classification.drop(columns=['features'], errors='ignore')
                result_df = classification.reset_index(names='CycID')
                result_df['Tempo_BH.Q'] = 1 - result_df['probability_rhythmic']
                result_df['reject'] = result_df['Tempo_BH.Q'] <= thresh

            elif method == 'PermCosinor':

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
                # run_metacycle reports its own st.error and returns None on failure —
                # without this guard the next line raises AttributeError on top of it.
                if result_df is None:
                    my_bar.empty()
                    st.stop()
                cols = [col for col in result_df.columns if method in col]

                q_col = [col for col in cols if 'BH.Q' in col.upper()][0]  
                
                result_df['reject'] = np.where(result_df[q_col] <= thresh, True, False)

            my_bar.progress(step + 75, text=bar_text)
          
            result_df = result_df.set_index('CycID')
            result_df['Periods'] = periods
            # The classifier's columns are only merged in when Tempo IS the method
            # (they are already present in that branch). Previously they were
            # appended to every result regardless of what was selected.
            result_df = result_df.reset_index()

            st.session_state["result_df"] = result_df  # Save in session state
            st.session_state["tested_file"] = file_name

            csv = convert_for_download(result_df.set_index('CycID'))
                
            messages2.download_button(label="Download MetaCycle results",
                                data=csv,
                                file_name=f"{today}_{file_name}_stats.txt",
                                mime='text/csv',
                                type='primary',
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
        


