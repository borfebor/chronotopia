#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:55:41 2025

@author: borfebor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from methods import methods

def convert_for_download(df):
        return df.to_csv(sep='\t').encode("utf-8")
    
st.sidebar.header('Data uploading')

uploaded_file = st.sidebar.file_uploader('Upload your data',
                        type=['csv','txt','xlsx', 'tsv'])
example = st.sidebar.toggle('Check example dataset')

if example:
    uploaded_file = 'hola'

if uploaded_file is None:
    #st.image(instructions)
    st.markdown("""
                ### Example Time Series Dataset

| Time (min) |   A   |   B   |   C   |   D   |   E   |
|------------|-------|-------|-------|-------|-------|
|     0      | 1.23  | 4.56  | 3.21  | 2.34  | 5.67  |
|    10      | 1.45  | 4.80  | 3.00  | 2.50  | 5.50  |
|    20      | 1.60  | 4.90  | 3.10  | 2.70  | 5.30  |
|    30      | 1.75  | 5.10  | 3.30  | 2.80  | 5.20  |
|    40      | 1.90  | 5.20  | 3.50  | 3.00  | 5.00  |
|    50      | 2.00  | 5.30  | 3.60  | 3.10  | 4.90  |
                """)
    st.stop()

if uploaded_file is not None:
    messages = st.empty()
    messages2 = st.empty()

    st.header('Data Analysis')
    c1, c2 = st.columns(2)
    
    if uploaded_file == 'hola':
        df = methods.example_data()
    else:
        df = methods.importer(uploaded_file)
        
    layout = st.sidebar.checkbox('Include experimental layout', False)

    t_col = st.sidebar.selectbox('Time column', [col for col in df.columns] )
    t_unit = st.sidebar.selectbox('Time unit', ['Minutes', 'Hours', 'Days', 'Seconds'])
    data_cols = [col for col in df.columns if col != t_col]
    
    if layout == True:

        st.sidebar.header('Experimental groups')
        template = pd.DataFrame(data_cols, columns=['Sample'])
        template['Condition'] = 'YOUR_CONDITION'
        
        csv = convert_for_download(template)
        
        st.sidebar.download_button(label="Download layout template",
                        data=csv,
                        file_name='sample_layout_template.txt',
                        mime='text/csv',
                        type='primary',
                        help='Here you can download your data',
                        use_container_width=True,)
        layout_file = st.sidebar.file_uploader('Upload your experimental layout',
                                type=['csv','txt','xlsx', 'tsv'])
        
        if layout_file is not None:
            layout_df = methods.importer(layout_file)
            
            layout_df['name'] = layout_df.Condition + " - [" + layout_df.Sample + "]"
            
            name_dict = dict(zip(layout_df.Sample, layout_df.name))
            df = df.rename(columns=name_dict)
            data_cols = [col for col in df.columns if col != t_col]

    
    df[t_col] = df[t_col].apply(lambda x: methods.time_changer(x, t_unit))
    
    st.sidebar.header('Analysis paramenters')
    
    hourly = st.sidebar.checkbox('Smoothen the data hourly', False)
    ent = st.sidebar.checkbox('Include entrainment data', False)
    
    if hourly == True:
        df = methods.hourly(df, t_col)
    
    if ent == True:
        ent_days = st.select_slider('Days of entrainment', [i for i in range(1, 6)])
    
    norm_meth = c1.selectbox('Normalization', ['None', 'Z-Score', 'Min-Max'])
    detrend_meth = c2.selectbox('Detrending', ['None', 'Linear', 'Rolling mean'])
    
    df[data_cols] = methods.detrend(df, data_cols, t_col, detrend_meth)
    df[data_cols] = methods.normalize(df, data_cols, norm_meth)
    
    df = df.dropna()
    
    st.header('Data Preview')
        
    duration = np.round(df[t_col].max(),1)
    st.write(f"Experiment with data for {duration} hours")
    
    preview = st.empty()
    p_col = st.selectbox('Column to preview', data_cols)
    unit = st.text_input('Data unit', 'Measured unit')
    t_plot = st.slider('Time period to plot', df[t_col].min(), df[t_col].max(), (df[t_col].min(), df[t_col].max()))

    pre_plot = st.empty()
    short = df[[t_col] + data_cols[:5]].iloc[:5]
    preview.table(short)
    
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
    pre_plot.pyplot(fig)
    
    csv = convert_for_download(df)
    
    st.sidebar.header('Final steps')

    analysis_button = st.sidebar.button('Run analysis', type='primary', use_container_width=True)
    st.sidebar.download_button(label="Download clean data",
                    data=csv,
                    file_name='clean_data.txt',
                    mime='text/csv',
                    help='Here you can download your data',
                    use_container_width=True,)
    
    pdf_buffer = methods.generate_pdf_report(df, t_col, data_cols, ent, ent_days, unit)
    st.sidebar.download_button(
            label="📄 Download report",
            data=pdf_buffer,
            file_name="rhythmicity_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    if analysis_button:
        
        with st.spinner("Running R script..."):

            # Transpose and set index
            rdf = df.transpose()
            
            messages.warning("""
            Hi,
            
            I am still working on the implementation of this feature. In the meantime, you can download the formated data to run it in Metacycle.
            
            Best,
            
            Borja""")
            csv = convert_for_download(rdf)
            
            messages2.download_button(label="Download data for Metacycle testing",
                            data=csv,
                            file_name='data_for_metacycle.txt',
                            mime='text/csv',
                            type='primary',
                            help='Here you can download your data',
                            use_container_width=True,)
            
                #st.dataframe(result_df)
                



