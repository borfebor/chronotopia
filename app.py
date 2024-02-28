#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:51:30 2023

@author: borfebor
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
from PIL import Image

from methods import methods

image = Image.open('CHRONO.png')
st.image(image)

file = st.file_uploader(label='Add your timeseries')



if file != None:
    
    df = methods.importer(file)
    
    st.sidebar.header('Settings')
    
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    
    time_col = c1.selectbox('Select time column', df.columns)
    
    time_format = c2.selectbox('Select time format', ['hh:mm:ss', 'mm:ss', 'ss', 'mm'], 0)
    
    last_unit = c3.selectbox('Select input last time unit', ['Sec', 'Min', 'Hour'], 0)
    
    translate = {'Sec': {'Sec':1, 'Min':1/60, 'Hour':1/3600, 'Day': 1/(24*60*60)},
                'Min': {'Sec':60, 'Min':1, 'Hour':1/60, 'Day': 1/(24*60)},
                'Hour': {'Sec':60*60, 'Min':60, 'Hour':1, 'Day': 1/24}, 
                'Day': {'Sec':60*60*24, 'Min':60*24, 'Hour':24, 'Day': 1}}

    translate_options = list(translate[last_unit].keys())
    
    viz_uni = c4.selectbox('Select desired time unit', translate_options, 3)

    df = methods.find_the_start(df, time_col=time_col)
    
    df = methods.time_qc(df, time_col=time_col)
    
    df, time_col = methods.time_formater(df, time_col=time_col, last_unit=last_unit)
    
    df, time_col = methods.time_translator(df, time_col=time_col, last_unit=last_unit, viz_unit=viz_uni)
    
    items = df.columns.to_list()
    items.remove(time_col)

    columns = pd.DataFrame(items, columns = ['Column'])
    columns['Group'] = pd.Series(items)
    columns['Replicate'] = pd.Series(items)
    columns = columns.set_index('Column')
    
    apply = st.sidebar.checkbox('Apply groups')

    params_holder = st.sidebar.empty()
    
    export_params = columns.to_csv(sep='\t').encode('utf-8')

    st.sidebar.download_button(
          label="Get groups template",
          data=export_params,
          file_name='params.txt',
          mime='text/csv',
          help='I will help you',
          use_container_width=True,
      )
    
    groups_file = st.sidebar.file_uploader(label='Or upload your groups here')
    
    if groups_file != None:
    
        columns = methods.importer(groups_file)
        columns = columns.set_index(columns.columns[0])
        
    experiments = params_holder.data_editor(columns)
    experiments['col_name'] = experiments.Group + '_' + experiments.Replicate
    
    params = experiments.to_dict()
    
    df = df.rename(columns=params['col_name'])
    
    col1, col2 = st.columns([1,1])
    
    table = st.expander(label='Your data')
    
    table.dataframe(df.set_index(time_col))
    
    normalization = col1.selectbox('Select how to normalize the data', ['None', 'max_min', 'z_score'], 0)
    
    detrending = col2.selectbox('Select how to detrend the data', ['None', 'linear', 'rolling'], 0)
    
    
    if apply == True:
    
        data = df.melt(id_vars=time_col, 
                       value_vars=list(params['col_name'].values()), var_name='variable')
        
        data[['Group', 'Replicate']] = data['variable'].str.rsplit('_', n=1, expand=True)
        
        
        data = methods.detrending(data,  how=detrending, rolling_window=10)
            
        data = methods.normalization(data,  how=normalization)
    
        pio.templates.default = "simple_white"
        
             
        group_to_display = st.multiselect('Select group(s) to display', 
                                           list(data.Group.unique()), 
                                           list(data.Group.unique())[:3])
        
        entrainment_box = st.expander('Entrainment settings')
        
        ent1, ent2, ent3, ent4 = entrainment_box.columns(4)
        
        entrainment_true = ent1.checkbox('Add entrainment data', True)
        cycles = ent2.slider('Number of cycles', int(data[time_col].min()) ,
                                 int(data[time_col].max()), 3)
        periods = ent3.slider(f'Period length ({viz_uni})', int(data[time_col].min()) ,
                                 int(data[time_col].max()), 1)
        exposure = ent4.slider('Cycles ratio (1 : x)', 2,
                                 int(data[time_col].max()), 2)
            
        color1, colors2 = entrainment_box.columns(2)
            
        bg_colour = color1.color_picker('Pick A Background Color', '#FFFFFF')
        zeit_colour = colors2.color_picker('Pick A Zeitgeber Color', '#FFD6C2')
        
        period_box = st.expander('Period calculation settings')
        
        peak_detection = period_box.checkbox('Find peaks and calculate period', True)
        
        if peak_detection == True:
        
            start_time, end_time = period_box.select_slider(
                        'Select the time frame for period calculation',
                        options=np.round(data[time_col].unique()),
                        value=(np.round(data[time_col].min()), np.round(data[time_col].max())))
        
        if len(group_to_display) < 2:
            
            data_col = 'variable'
            group_to_display = data[data['Group'].isin(group_to_display)][data_col].unique()
            
        else:
            data_col = 'Group'
            
            
        plot, maximum, minimum, fig, periods = methods.lineplot(data, data_col, time_col, 
                                                                group_to_display, entrainment_true, 
                                                                periods, cycles, exposure, start_time, end_time,
                                                                bg_colour, zeit_colour, peak_detection)
        
        
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        if len(periods) > 0:
            
            period_table, period_box = st.columns(2)
            
            period_table.header('Calculated periods')
            
            box, periods = methods.period_calculation(data, group_to_display, viz_uni)
            
            period_table.dataframe(periods
                                   , use_container_width=True)
    
            period_box.plotly_chart(box, theme="streamlit", use_container_width=True)
            
        if viz_uni == 'Day':
                
                st.header('Double plot actogram')
                displayed_group = st.selectbox('Select group to display', 
                                                   group_to_display, 
                                                   0)
        
                acto = methods.actogram(plot, displayed_group, time_col, data_col)
        
                st.plotly_chart(acto, theme="streamlit", use_container_width=False)      
else: 
                  
    st.header('While you format your data, please enjoy the best songs about clocks')
    playlist = ['https://open.spotify.com/track/6LBmaJYwbLHfQwIreMCLlw?si=97523fb6e9664c49', 
                'https://www.youtube.com/watch?v=ZgdufzXvjqw&pp=ygUVcm9jayBhcm91bmQgdGhlIGNsb2Nr',
                'https://www.youtube.com/watch?v=7bLgGYFLhgQ&pp=ygUPc3RvcCB0aGUgY2xvY2tz',
                'https://www.youtube.com/watch?v=Qr0-7Ds79zo&pp=ygUFdGltZSA%3D',
                'https://www.youtube.com/watch?v=VrDfSZ_6f4U',
                'https://www.youtube.com/watch?v=iP6XpLQM2Cs',
                ]
    song = np.randint(0, len(playlist))
    st.video(playlist[song])
    
    
