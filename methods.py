#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:56:54 2023

@author: borfebor
"""

import numpy as np
import pandas as pd
import streamlit as st

from scipy import signal

import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import seaborn as sns

class methods:
    
    def importer(file):
        
        file_type = file.name.split('.')[1].upper()
        
        if file_type == 'TXT':
            
            df = pd.read_csv(file, sep='\t')
            df = methods.true_columns(df)
            return df.reset_index(drop=True).dropna(axis=1, how='all')
            
        if file_type == 'CSV':
            
            df = pd.read_csv(file, sep=',')
            df = methods.true_columns(df)
            return df.reset_index(drop=True).dropna(axis=1, how='all')
            
        if file_type == 'XLSX':
                
            df = pd.read_excel(file)
            df = methods.true_columns(df)
            return df.reset_index(drop=True).dropna(axis=1, how='all')
        
        else:
            st.warning('Not compatible format')
    
    def true_columns(df):
        
        candidates = [col for col in df.columns if 'UNNAMED' not in col.upper()]
        
        if len(candidates) != len(df.columns):
            
            i = 0
            
            while len(candidates) != len(df.columns):
                
                df.columns = df.iloc[i]
                candidates = [col for col in df.columns if 'UNNAMED' not in col.upper()]
                print(candidates)
                i += 1
                
            else:
                
                df = df.iloc[(i+1):]    
                
        return df
    
    def find_the_start(df, time_col='Time'):
    
        cols = [col for col in df.columns if col != time_col]
    
        starting_row = 0
        success = False
    
        while success == False:
    
            try:
                df[cols].iloc[starting_row].astype(float)
                success = True
    
            except:
                starting_row += 1
    
        df = df.iloc[starting_row:]
        df[cols] = df[cols].astype(float)
        
        return df.reset_index(drop=True)
    
    def time_qc(df, formatting='%H:%M:%S', time_col='Time', separator=':'):

        length = len(formatting.split(separator))
    
        df[time_col] = np.where(df[time_col].str.split(separator).apply(lambda x: len(x)) != length,
                 df[time_col] + ':00', df[time_col])
        
        return df
    
    def time_formater(df, formatting='%H:%M:%S', time_col='Time', separator=':', last_unit='Sec'):
        
        units = {'Sec': {'%d': 24*3600, '%H': 3600, '%M' : 60, '%S': 1},
                'Min': {'%d': 24*60, '%H': 60, '%M' : 1, '%S': 1/60},
                'Hour': {'%d': 24, '%H': 1, '%M' : 1/60, '%S': 1/3600}}
        
        multiplier = [units[last_unit][item] for item in formatting.split(separator)]
        
        new_time_col = f'Time ({last_unit})'
    
        df[new_time_col] = df[time_col].str.split(':').apply(lambda x: 
                                                                   np.sum(np.array(np.float_(x)) * np.array(multiplier)))
        
        df = df.drop(columns=time_col)
        
        return df, new_time_col
    
    
    def time_translator(df, time_col, last_unit='Sec', viz_unit='Hour'):
    
        translate = {'Sec': {'Sec':1, 'Min':1/60, 'Hour':1/3600, 'Day': 1/(24*60*60)},
                    'Min': {'Sec':60, 'Min':1, 'Hour':1/60, 'Day': 1/(24*60)},
                    'Hour': {'Sec':60*60, 'Min':60, 'Hour':1, 'Day': 1/24}, 
                    'Day': {'Sec':60*60*24, 'Min':60*24, 'Hour':24, 'Day': 1}}
        
        df[time_col] = df[time_col] * translate[last_unit][viz_unit]
        
        new_time_col = f'Time ({viz_unit})'
        
        df = df.rename(columns={time_col:new_time_col})
        
        return df, new_time_col
    
    
    def detrending(data, how='rolling', rolling_window=10):
    
        results = pd.DataFrame()
        
        for item in data.variable.unique():
            
            testing_data = data[data.variable == item].reset_index(drop=True)
    
            testing_data['detrending'] = how
    
            treated = testing_data.copy()
            
            if how == 'linear':
    
                treated['value'] = signal.detrend(treated['value'], type='linear')
                
            elif how == 'rolling':
                
                rolling_mean = treated.value.rolling(window=rolling_window, center=True).mean()
                treated['value'] = treated.value - rolling_mean
    
            results = pd.concat([results, treated]).reset_index(drop=True)
    
        return results
    
    def normalization(data, how='z_score'):
        
        norm_test = data.copy()
            
        if how == 'None':
                
                pass
        
        else:
            
            for item in norm_test.variable.unique():
            
                if how == 'max_min':
        
                    norm_test['value'] = np.where(norm_test.variable == item,
                                                 (norm_test.value - norm_test.value.mean()) / norm_test.value.std(),
                                                  norm_test.value)
        
                elif how == 'z_score':
        
                    norm_test['value'] = np.where(norm_test.variable == item,
                                                 (norm_test.value - norm_test.value.mean()) / norm_test.value.std(),
                                                  norm_test.value)
        
            norm_test['detrending'] = norm_test.detrending + '_normalized'
    
        return norm_test
    
    def entrainment_cycles(period, cycles, exposure):

        duration = period / exposure
    
        day_cycles = [duration,duration,period,period,None]
    
        i = 0
        x_cycles = []
    
        while i < cycles:
    
            for x in day_cycles:
                try:
                    x_cycles.append(x + i)
                except:
                    x_cycles.append(x)
            i += 1
    
        return x_cycles
    
    def lineplot(data, data_col, time_col, group_to_display, entrainment_true,
                 periods, cycles, exposure, start_time, end_time, bg_colour, zeit_colour,
                 peak_detection):
    
        plot = data.groupby([data_col, time_col]).apply(lambda x:
                                                                                                  pd.Series({
                                                         'value' : x.value.mean(),
                                                         'upper' : x.value.mean() + x.value.std(),
                                                         'lower' : x.value.mean() - x.value.std(),
                                                                                                      
                                                                                                  })).reset_index()
            
        maximum = plot[plot[data_col].isin(group_to_display)].upper.max()
        minimum = plot[plot[data_col].isin(group_to_display)].lower.min()
            
        fig = go.Figure()
          
        colors = sns.color_palette('cubehelix', len(group_to_display))
        
        if entrainment_true == True:
            
            y_cycles = [minimum,maximum,maximum,minimum,None] * cycles
            x_cycles = methods.entrainment_cycles(periods, cycles, exposure)
                
            fig.add_trace(
                go.Scatter(
                    name='Entrainment',
                    y=y_cycles, 
                    x=x_cycles,
                    line=dict(color=bg_colour,width=0),
                    fill="toself", fillcolor=zeit_colour,))
        else:
            bg_colour = '#FFFFFF'
            
        periods = []
        
        for num, item in enumerate(group_to_display):
            
           
            sorted_plot = plot[plot[data_col] == item]
            
            if peak_detection == True:
                
                period_data = sorted_plot[(sorted_plot[time_col] > start_time) &
                                          (sorted_plot[time_col] < end_time)]
                
                
                peaks, heigth = signal.find_peaks(period_data.value, height=period_data.value.mean())
                
                fig.add_trace(go.Scatter(
                            name='Peak',
                            x=period_data.reset_index().iloc[peaks][time_col],
                            y=heigth['peak_heights'],
                            mode='markers',
                            showlegend=False
                        ))
                
                period = np.diff(period_data.reset_index().iloc[peaks][time_col].values).mean()
                periods.append([item, period])
            
            fig.add_trace(
                        go.Scatter(
                            name=f'{item}',
                            x=sorted_plot[time_col],
                            y=sorted_plot['value'],
                            mode='lines',
                            line=dict(color=f'rgb{colors[num]}'),
                        ))
            fig.add_trace(
                        go.Scatter(
                            name='Upper Bound',
                            x=sorted_plot[time_col],
                            y=sorted_plot['upper'],
                            mode='lines',
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            showlegend=False
                        ))
            fig.add_trace(
                        go.Scatter(
                            name='Lower Bound',
                            x=sorted_plot[time_col],
                            y=sorted_plot['lower'],
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            mode='lines',
                            fillcolor=f'rgba{colors[num]+tuple([0.2])}',
                            fill='tonexty',
                            showlegend=False
                        )
                    )
              
        fig.update_layout(
                yaxis_title='Luminescense (UA) Detrended, normalized',
                xaxis_title=f'{time_col}',
                hovermode="x",
                plot_bgcolor=bg_colour,
                yaxis_range=[minimum,maximum]
            )
        
        return plot, maximum, minimum, fig, periods
    
    def period_calculation(data, group_to_display, viz_uni):
        
        colors = sns.color_palette('cubehelix', len(group_to_display)).as_hex()
    
        per_col = f'Period ({viz_uni})'
        
        mean_periods = data[data.Group.isin(group_to_display)].groupby(['Group', 'Replicate']).apply(lambda x: pd.Series({
                'Period': np.mean(np.diff(signal.find_peaks(x['value'], height=x.value.mean())[0]))})).reset_index()
        
        period_values = mean_periods.groupby('Group').apply(lambda x: pd.Series({
            'Periods' : x.Period,
               per_col: f'{np.round(x.Period.mean(), 2)} ± {np.round(x.Period.std(), 2)}'}))
        
        box = go.Figure()
        
        for num, group in enumerate(group_to_display):
            
            box.add_trace(go.Box(
                        y=period_values.loc[group]['Periods'],
                        x=[group] * len(period_values.loc[group]['Periods']),
                        name=group,
                        fillcolor=colors[num],
                        line_color='#000000'
                    ))
        
        
        box.update_layout(
                yaxis_title=per_col,
                hovermode="x",
            )
        
        return box, period_values[per_col]
    
    def actogram(plot, displayed_group, time_col, data_col):
    
        
        example = plot[plot[data_col] == displayed_group].copy()
        
        example['day_y'] = example[time_col].apply(lambda x: np.ceil(x))
    
        example['hour_x'] = example[time_col].apply(lambda x: (x + 1 - np.ceil(x)) * 24)
        
            
        pio.templates.default = "simple_white"
        
        acto = make_subplots(rows=example.day_y.max(), cols=2, 
                        shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0, horizontal_spacing=0)
    
        for num, item in enumerate(example.dropna(subset='value')['day_y'].unique()):
            
            plot_left = example[(example.day_y == item) & (example.value > example.value.mean())]
            
            acto.add_trace(
                go.Scatter(x=plot_left.hour_x, y=plot_left.value, name="yaxis data", mode='lines',
                            line=dict(color='#287B8F'),
                            fill='tozeroy', showlegend=False),
                
                row=num+1, col=2)
            
            plot_right = example[(example.day_y == item-1) & (example.value > 0)]
            
            acto.add_trace(
                    go.Scatter(x=plot_right.hour_x, y=plot_right.value, name="yaxis data", mode='lines',
                                line=dict(color='#287B8F'), showlegend=False,
                                fill='tozeroy'),
                    row=num+1, col=1)
            
            acto.update_yaxes(title=f'{item}',
                             range=[0, (example.value.max()+example.value.std())], 
                             row=num+1, col=1,
                            showticklabels=False)
            acto.update_xaxes(range=[0, 24], 
                              tickvals = [0, 6, 12, 18, 24],
                             row=num+1, col=1)
            acto.update_xaxes(range=[0, 24], 
                              tickvals = [6, 12, 18, 24],
                             row=num+1, col=2)
            
        acto.update_layout(
            width = 500,
            height = 600
            )  
        
        return acto
