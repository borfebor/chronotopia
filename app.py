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
from sklearn.manifold import MDS

from io import BytesIO

from PIL import Image
import subprocess
import tempfile
import os
import math

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
    
    
version = "0.5.4"
st.sidebar.write(f"Version {version}")    
st.sidebar.header('Data uploading')

uploaded_file = st.sidebar.file_uploader('Upload your data',
                        type=['csv','txt','xlsx', 'tsv'])
example = st.sidebar.toggle('Check example dataset')

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
        df = methods.example_data()
    else:
        df = methods.importer(uploaded_file)
        
    layout = st.sidebar.toggle('Include experimental layout', False)
    
    df.columns = [col.strip() for col in df.columns]
    
    t_col = st.sidebar.selectbox('Time column', [col for col in df.columns] )
    
    delta_t = np.mean(np.diff(df[t_col].values))  # assumes sorted time
    
    t_options = ['Minutes', 'Hours', 'Days', 'Seconds']
    if delta_t > 1:
        default = t_options.index('Minutes')
    else:
        default = t_options.index('Hours')
        
    t_unit = st.sidebar.selectbox('Time unit', t_options, default)

    data_cols = [col for col in df.columns if col != t_col]
    
    if len(data_cols) == 96:
        
        template = pd.DataFrame(data_cols, columns=['Sample'])
        template['Condition'] = [f"COL_{int(i[-2:])}" for i in data_cols]
        layout_df = template.copy()
        layout_df['name'] = layout_df.Sample
            
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
    
    t_start = c1.number_input('Starting Timepoint', df[t_col].min(), df[t_col].max(), df[t_col].min())
    t_end =  c2.number_input('Last Timepoint', t_start, df[t_col].max(),df[t_col].max() )
    
    df = df[(df[t_col] >= t_start)  & (df[t_col] <= t_end)]
    
    st.sidebar.header('Analysis paramenters')
    
    hourly = st.sidebar.toggle('Smoothen the data', False)
    #outfilter = st.sidebar.toggle('Filter outliers', False)
    ent = st.sidebar.toggle('Include entrainment data', False)
    exclusion = st.sidebar.toggle('Select samples to exclude', False)
    test_a_bit = st.sidebar.expander('Analysis Parameters')#st.sidebar.toggle('Rhythmicity analysis parameters', False)
    
    with settings:
    
        exclusion_place = st.empty()
    
    method = 'meta2d'
    thresh = 0.05
    with test_a_bit:
        t1, t2 = st.columns(2)
        t_start_test = t1.number_input('Minimum time', int(df[t_col].min()), int(df[t_col].max()), int(df[t_col].min()),  step=1)    
        t_end_test = t2.number_input('Last time', int(df[df[t_col] > t_start_test][t_col].min()), int(df[df[t_col] > t_start_test][t_col].max()), int(df[df[t_col] > t_start_test][t_col].max()), step=1) 
        method = t1.selectbox('Testing method', ['meta2d', 'JTK', 'ARS', 'LS', ], 0)   
        thresh = t2.selectbox('Significance threshold', [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], 0)   
        period_estimation = st.selectbox('Period Estimation', ['Fast Fourier Transform (FFT)', 'Autocorrelation', 'Lomb-Scargle Periodogram', 'Wavelet Transform'], 2)

        
    #else:
    #    t_start_test, t_end_test = df[t_col].min(), df[t_col].max()  
        
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
        delta_t = np.mean(np.diff(df[t_col].values))  # assumes sorted time
        samples_per_hour = int(round(1 / delta_t))
        df[data_cols] = df[data_cols].rolling(window=samples_per_hour, center=True, min_periods=1).mean()
        df = df.dropna()
        #st.stop()
        
    if ent == True:
        with settings:
            col1, col2, col3, col4 = st.columns(4)
            ent_days = col1.number_input('Entrainment cycles', 1, max_days, 1,  step=1) 
            T = col2.number_input('T cycle', 6, 48, 24,  step=1) 
            cycle_type = col3.selectbox('Zeitgeber type', ['Darkness - Light', 'Light - Darkness', 'Cold - Warm', 'Warm - Cold', 'Custom'], 0) 
            
            if cycle_type == 'Custom':
                color1, color2 = st.columns(2)
                ent_color = color1.color_picker('Entrainment band', '#9BD1E5')
                bg_color = color2.color_picker('Background color', '#ffffff')
                order = col4.selectbox('Color order', [0, 1], 0)
            else:
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
            
    norm_meth = c1.selectbox('Normalization', ['None', 'Z-Score', 'Sample-wise Min-Max', 'Global Min-Max'])
    detrend_meth = c2.selectbox('Detrending', ['None', 'Linear', 'Rolling mean', 'Hilbert + Rolling mean', 'Cubic'])
    

    df[data_cols] = methods.detrend(df, data_cols, t_col, detrend_meth)
    df[data_cols] = methods.normalize(df, data_cols, norm_meth)
    
    if exclusion:
        
        ex_type, ex_cols = exclusion_place.columns([1,2])
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
            #st.write(f"Excluded {exclusion_list} from {ex_col}")
            layout_df = layout_df[~layout_df['name'].isin(exclusion_list)]
    
    df = df.dropna()
        
    duration = np.round(df[t_col].max(),1)
    sum_pre.write(f"Experiment with {len(data_cols)} sample recorded for {duration} hours (recorded every = {np.round(delta_t,2)} h)")
    
    conditions = []
    visu = ['Lineplot', 'Actogram', 'Correlation', 'PCA']
    
    if 'layout_df' in globals():
        
        conditions = list(layout_df.Condition.unique())
        
        visu = visu + ['Lineplot [Mean ± SD]', 'Lineplot [Mean + Replicates]']
    
    viz_settings = st.expander('Visualization settings (Plot type, sample selection, data unit...)')  

    with viz_settings:
    
        c, c1, c2 = st.columns([2, 1, 1])
        t0 = c1.number_input('Starting time to plot', int(df[t_col].min()), int(df[t_col].max()), int(df[t_col].min()),  step=1)    
        t1 = c2.number_input('End time to plot', int(df[df[t_col] > t0][t_col].min()), int(df[df[t_col] > t0][t_col].max()), int(df[df[t_col] > t0][t_col].max()), step=1) 
        
        plot_type = c.selectbox("Type of plot to visualize", visu)

    short = df[[t_col] + data_cols[:5]].iloc[:5]
    
    if on:
        preview.table(short)
    
    pre_plot = st.empty()

    with viz_settings:
 
        if plot_type == 'Lineplot':
            
            p_col = st.selectbox('Column to preview', data_cols)
            unit = st.text_input('Data unit', 'Measured unit')
            per = methods.period_estimation(df, [p_col], t_col, method=period_estimation)
            per = np.round(per, 2)

            fig = methods.plot(df, t_col, p_col, t0, t1, bg_color=bg_color, ent=ent, 
                         ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)
            
            plt.title(f"{p_col}. Period = {per.loc[p_col]} h ({period_estimation}-calculated)")

            pre_plot.pyplot(fig)
                
        elif plot_type == 'Lineplot [Mean ± SD]':
                
            p_col = st.selectbox('Column to preview', conditions)
            unit = st.text_input('Data unit', 'Measured unit')
            #pre_plot = st.empty()
            
            fig = methods.grouped_plot(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                     ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)

            pre_plot.pyplot(fig)
            
        elif plot_type == 'Lineplot [Mean + Replicates]':
                
            p_col = st.selectbox('Column to preview', conditions)
            unit = st.text_input('Data unit', 'Measured unit')
            #pre_plot = st.empty()
            
            fig = methods.grouped_plot_traces(df, t_col, t0, t1, group=p_col, layout=layout_df, bg_color=bg_color, ent=ent, 
                     ent_days=ent_days, order=order, T=T, color=ent_color, unit=unit)
            #pre_plot.pyplot(fig)
    
            
        elif plot_type == 'Actogram':
            p_col = st.selectbox('Column to preview', data_cols)
            times = st.number_input("Plot N times", 1, int(np.round(df[t_col].max() / 24)), 1)
            #pre_plot = st.empty()

            fig = methods.double_plot(df, t_col, p_col, ent_days, T, order, t0=t0, t1=t1, times=times, 
                                      bg_color=bg_color, band_color=ent_color)
            #plt.suptitle(f"{p_col}. Period = {periods.loc[p_col]} h ({period_estimation}-calculated)")


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
            
            # Plot a heatmap
            
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
            #plt.scatter(embedding[:, 0], embedding[:, 1])
            #for i, label in enumerate(data_cols):
            #    plt.text(embedding[i, 0], embedding[i, 1], label, ha='left', va='bottom')
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
    

    if analysis_button:
        
        with st.spinner("Running R script..."):
            st.toast('Calculating periods...!')
            
            periods = methods.period_estimation(df, data_cols, t_col, method=period_estimation).rename('Period')
            periods = np.round(periods, 2)
            
            # Transpose and set index
            df[t_col] = df[t_col].apply(lambda x: np.round(x,1))
            test_df = df[np.isclose(df[t_col] % 1, 0)]
            
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
                                use_container_width=True,)
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
                                use_container_width=True,)
                            
    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
        
        if "layout_df" in globals():

            baloon = st.sidebar.button('Compare groups', 
                                       use_container_width=True,)
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
                            use_container_width=True
                        )
    
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
                            
                            per_col = 'Periods'
                            
                            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 7), layout='tight')
                            
                            for ax in (ax1, ax2):
                                
                                if ax == ax2:
                                    plot = mix[mix.reject == True]
                                    ax.set_title('Tested periodicity (only rhythmic samples)')

                                else:
                                    plot = mix
                                    ax.set_title('Tested periodicity (all samples)')
                            
                                sns.pointplot(plot, x='Condition', y=per_col, hue='Condition', ax=ax, capsize=0.2)
                                sns.swarmplot(plot, x='Condition', y=per_col, hue='Condition',
                                              legend=False, edgecolor='k', size=8, linewidth=1, ax=ax)
                                
                                ax.tick_params(axis='x', rotation=90)
                                ax.set_ylabel('Period (h)')
                                ax.set_ylim(18, 36)
                                                                                        
                            figures.append(fig)
                            
                            N = len(conditions)  # number of experiments
                            cols = math.ceil(math.sqrt(N))
                            rows = math.ceil(N / cols)
                            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), layout='tight')
                            axes = axes.flatten()  # Flatten to simplify indexing
                            
                            for n, group in enumerate(conditions):
                                ax = axes[n]
                                sorter = layout_df[layout_df.Condition == group]['name'].unique()
                                sorted_result = result_df[result_df['CycID'].isin(sorter)]
                                
                                methods.pie_chart(ax, sorted_result, method=method, group=group, thresh=thresh)
                                ax.set_title(group)
                                
                            # Hide unused axes
                            for j in range(N, len(axes)):
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
                                    title= f"p-val: {np.round(d['p-val'], 4)}"
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
                        axes = axes.flatten()  # Flatten to simplify indexing
                                
                        for n, subgroup in enumerate(names):
                            ax = axes[n]
                            ax.plot(df[t_col], df[subgroup])
                            #ax = methods.multiplot(ax, df, t_col, group, t0, t1)
                            
                            if "result_df" in globals():
                                focus = result_df[result_df['CycID'] == subgroup]

                                cols = [col for col in focus.columns if method in col]
    
                                per_col = 'Periods'
                                q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
                                q = np.round(focus[q_col].mean(), 5)
                                reject = q <= thresh
                                period = f"{np.round(focus[per_col].mean(),1)}"
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
                        for j in range(N, len(axes)):
                            fig.delaxes(axes[j])
                                  
                        plt.suptitle(group, weight='bold', fontsize=20)
                        figures.append(fig)
                
                if ent == True:
                    for col in data_cols:
                        
                        if "result_df" in globals():
                            focus = result_df[result_df['CycID'] == col]
                            cols = [col for col in focus.columns if method in col]

                            per_col = 'Periods'
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

                            per_col = 'Periods'
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


