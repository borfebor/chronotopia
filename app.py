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
from io import BytesIO

from PIL import Image
import subprocess
import tempfile
import os

image = Image.open('logo.png')
st.sidebar.image(image)

def convert_for_download(df):
        return df.to_csv(sep='\t').encode("utf-8")
    
    
version = "0.3.1"
st.sidebar.write(f"Version {version}")    
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
        
    layout = st.sidebar.toggle('Include experimental layout', False)
    
    df.columns = [col.strip() for col in df.columns]

    t_col = st.sidebar.selectbox('Time column', [col for col in df.columns] )
    t_unit = st.sidebar.selectbox('Time unit', ['Minutes', 'Hours', 'Days', 'Seconds'])
    
    data_cols = [col for col in df.columns if col != t_col]
    #df.columns = t_col + data_cols
    
    layout_file = None
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
    
    hourly = st.sidebar.toggle('Smoothen the data hourly', False)
    ent = st.sidebar.toggle('Include entrainment data', False)
    exclusion = st.sidebar.toggle('Select samples to exclude', False)
    test_a_bit = st.sidebar.toggle('Rhythmicity analysis parameters', False)
    
    if exclusion:
        exclusion_list = st.multiselect("Select samples to exclude", data_cols)
        df = df.drop(columns=exclusion_list)
        data_cols =  [col for col in df.columns if col != t_col]
    
    method = 'meta2d'
    thresh = 0.05
    if test_a_bit:
        t1, t2 = st.sidebar.columns(2)
        t_start_test = t1.number_input('Minimum time', int(df[t_col].min()), int(df[t_col].max()), int(df[t_col].min()),  step=1)    
        t_end_test = t2.number_input('Last time', int(df[df[t_col] > t_start_test][t_col].min()), int(df[df[t_col] > t_start_test][t_col].max()), int(df[df[t_col] > t_start_test][t_col].max()), step=1) 
        method = t1.selectbox('Preferred testing method', ['meta2d', 'JTK', 'ARS', 'LS', ], 0)   
        thresh = t2.selectbox('Significance threshold', [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], 0)   
        
    else:
        t_start_test, t_end_test = df[t_col].min(), df[t_col].max()  
        
    max_days = int(df[t_col].max() / 24) + 1
    
    backgroud = {'None': 'white',
                 'Darkness': '#EBEBEB' ,
                 'Light':'#FFD685' ,
                 'Warm': '#fbe3d4',
                 'Cold': '#dbeaf2'}
    
    bg_color = backgroud['None']
    ent_color = backgroud['None']
    
    if hourly == True:
        df = methods.hourly(df, t_col)
    
    if ent == True:
        col1, col2, col3, col4 = st.columns(4)
        ent_days = col1.number_input('Entrainment cycles', 1, max_days, 1,  step=1) 
        T = col2.number_input('T cycle', 6, 48, 24,  step=1) 
        cycle_type = col3.selectbox('Zeitgeber type', ['Darkness - Light', 'Light - Darkness', 'Cold - Warm', 'Warm - Cold'], 0) 
        
        parts = [part.strip() for part in cycle_type.split("-")]
        fr_options = [i for i in ['Light', 'Darkness', 'Cold', 'Warm'] if i in parts]

        freerun_type = col4.selectbox('Free running conditions', fr_options, 1) 
        bg_color = backgroud[freerun_type]
        #band_color = parts.remove(freerun_type)
        band_type = [i for i in parts if i != freerun_type][0]
        ent_color = backgroud[band_type]
        
        order = parts.index(band_type)
        #st.write(ent_color)
    else:
        T = 0
        order = 0
        ent_days = 0
    
    norm_meth = c1.selectbox('Normalization', ['None', 'Z-Score', 'Min-Max'])
    detrend_meth = c2.selectbox('Detrending', ['None', 'Linear', 'Rolling mean'])
    
    df[data_cols] = methods.detrend(df, data_cols, t_col, detrend_meth)
    df[data_cols] = methods.normalize(df, data_cols, norm_meth)
    
    df = df.dropna()
    
    st.header('Data Preview')
        
    duration = np.round(df[t_col].max(),1)
    st.write(f"Experiment with data for {duration} hours")
    
    preview = st.empty()
    
    conditions = []
    visu = ['Lineplot', 'Actogram']

    if layout_file is not None:
        
        conditions = list(layout_df.Condition.unique())
        #p_data_cols = data_cols + conditions
        
        visu = visu + ['Lineplot [Mean ± SD]', 'Lineplot [Mean + Replicates]']
        
    #else:
     #   p_data_cols = data_cols
    
    c, c1, c2 = st.columns([2, 1, 1])
    t0 = c1.number_input('Starting time to plot', int(df[t_col].min()), int(df[t_col].max()), int(df[t_col].min()),  step=1)    
    t1 = c2.number_input('End time to plot', int(df[df[t_col] > t0][t_col].min()), int(df[df[t_col] > t0][t_col].max()), int(df[df[t_col] > t0][t_col].max()), step=1) 
    
    plot_type = c.selectbox("Type of plot to visualize", visu)

    short = df[[t_col] + data_cols[:5]].iloc[:5]
    preview.table(short)
        
    if plot_type == 'Lineplot':
        
        p_col = st.selectbox('Column to preview', data_cols)
        unit = st.text_input('Data unit', 'Measured unit')
        pre_plot = st.empty()
        fig = methods.plot(df, t_col, p_col, t0, t1, bg_color=bg_color, ent=ent, 
                     ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)
        
        pre_plot.pyplot(fig)
            
    elif plot_type == 'Lineplot [Mean ± SD]':
            
        p_col = st.selectbox('Column to preview', conditions)
        unit = st.text_input('Data unit', 'Measured unit')
        pre_plot = st.empty()
        
        fig = methods.grouped_plot(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                 ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)
        pre_plot.pyplot(fig)
        
    elif plot_type == 'Lineplot [Mean + Replicates]':
            
        p_col = st.selectbox('Column to preview', conditions)
        unit = st.text_input('Data unit', 'Measured unit')
        pre_plot = st.empty()
        
        fig = methods.grouped_plot_traces(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                 ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)
        pre_plot.pyplot(fig)

        
    elif plot_type == 'Actogram':
        p_col = st.selectbox('Column to preview', data_cols)
        times = st.number_input("Plot N times", 1, int(np.round(df[t_col].max() / 24)), 1)
        pre_plot = st.empty()
        fig = methods.double_plot(df, t_col, p_col, ent_days, T, order, t0=t0, t1=t1, times=times, 
                                  bg_color=bg_color, band_color=ent_color)
        
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
        use_container_width=True,
    )
    
    csv = convert_for_download(df)
    
    st.sidebar.header('Final steps')

    analysis_button = st.sidebar.button('Run analysis', type='primary', use_container_width=True)
    st.sidebar.download_button(label="Download clean data",
                    data=csv,
                    file_name='clean_data.txt',
                    mime='text/csv',
                    help='Here you can download your data',
                    use_container_width=True,)
    
    report_spot = st.sidebar.empty()
    report_button = report_spot.button(
                        label="📄 Prepare report",
                        use_container_width=True
                    )

    if analysis_button:
        
        with st.spinner("Running R script..."):
            st.toast('Running MetaCycle...!')

            # Transpose and set index
            rdf = df[(df[t_col] >= t_start_test) & (df[t_col] <= t_end_test)].set_index(t_col).transpose().reset_index()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w') as temp_file:
                rdf.to_csv(temp_file.name, sep="\t", index=False)
                input_path = temp_file.name
                
            output_dir = tempfile.mkdtemp()

            result = subprocess.run(
                ["Rscript", "run_meta2d.R", input_path, output_dir],
                capture_output=True,
                text=True
            )
            
            st.text("STDOUT:\n" + result.stdout)
            #mest.text("STDERR:\n" + result.stderr)
            
            if result.returncode != 0:
                messages.error("R script failed.")
                csv = convert_for_download(rdf)
                
                messages2.download_button(label="Download data for Metacycle testing",
                                data=csv,
                                file_name='data_for_metacycle.txt',
                                mime='text/csv',
                                type='primary',
                                help='Here you can download your data',
                                use_container_width=True,)
            else:
                st.toast('Report ready to download!', icon='🎉')

                result_df = pd.read_csv(os.path.join(output_dir, "meta2d_result.csv"))
                col_sorter = [i for i in result_df.columns if 'meta.' in i]
                result_df = result_df[col_sorter]
                result_df.columns = [i.replace('meta.', '') for i in result_df.columns]
                
                cols = [col for col in result_df.columns if method in col]
                per_col = [col for col in cols if 'PERIOD' in col.upper()][0]
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
                                use_container_width=True,)
                            
    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
            
    if report_button:
        
        with st.spinner("Preparing report..."):
                st.toast('Preparing report...!')

                figures = []
                
                if conditions != []:
                    
                    #for group in conditions:
                        
                    if "result_df" in globals():
                        if "layout_df" in globals():
                            
                            res = result_df.set_index('CycID')
                            trans = layout_df.set_index('name')
                            mix = pd.concat([res, trans], axis=1)
                            
                            cols = [col for col in mix.columns if method in col]

                            per_col = [col for col in cols if 'PERIOD' in col.upper()][0]
                            
                            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 7), layout='tight')
                            
                            sns.boxplot(mix, x='Condition', y=per_col, hue='Condition', ax=ax1)
                            sns.swarmplot(mix, x='Condition', y=per_col, hue='Condition',
                                          legend=False, edgecolor='k', size=8, linewidth=1, ax=ax1)
                            
                            ax1.tick_params(axis='x', rotation=90)
                            ax1.set_ylabel('Period (h)')
                            ax1.set_title('Tested periodicity (all samples)')
                                                                                    
                            mix2 = mix[mix.reject == True]
                                                        
                            sns.boxplot(mix2, x='Condition', y=per_col, hue='Condition', ax=ax2)
                            sns.swarmplot(mix2, x='Condition', y=per_col, hue='Condition', 
                                          legend=False, edgecolor='k', size=8,  linewidth=1, ax=ax2)
                            
                            ax2.tick_params(axis='x', rotation=90)
                            ax2.set_ylabel('Period (h)')
                            ax2.set_title('Tested periodicity (only rhythmic samples)')
                            
                            st.pyplot(fig)
                            
                            figures.append(fig)
                                                
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
                
                if ent == True:
                    for col in data_cols:
                        
                        if "result_df" in globals():
                            focus = result_df[result_df['CycID'] == col]
                            cols = [col for col in focus.columns if method in col]

                            per_col = [col for col in cols if 'PERIOD' in col.upper()][0]
                            q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
                            q = np.round(focus[q_col].mean(), 5)
                            reject = q <= thresh
                            period = f"{np.round(focus[per_col].mean(),1)}"
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

                            per_col = [col for col in cols if 'PERIOD' in col.upper()][0]
                            q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
                            q = np.round(focus[q_col].mean(), 5)
                            reject = q <= thresh
                            period = f"{np.round(focus[per_col].mean(),1)}"
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
                        use_container_width=True
                    )


