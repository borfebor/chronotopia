"""
plots.py
========
All figure-drawing for Chronotopia.

Moved out of `methods.py` in v0.7.3 without behavioural change: these functions are
byte-identical to their previous versions apart from dedenting them out of the
`methods` class and rebinding internal `methods.<plot fn>` calls to plain module
functions. Numeric helpers (detrending, period estimation, statistics) stay in
methods.py; this module is only about turning data into figures.

Call them as `plots.plot(...)`, `plots.double_plot(...)` and so on.

Note: several of these still call `st.toggle` / `st.checkbox` / `st.error`
internally. That is inherited, not new, and is the reason they cannot yet be
wrapped in `@st.cache_data` — see the code assessment, section 5.2.
"""

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def phase_plot(ent, ax, peaks, group='norm_day', pal=['#EBEBEB', '#FFFFFF'], order=0):

    if order == 0:
        pal = pal[::-1]
    # Simulated data
    peak_hours = peaks#ent.iloc[peaks][group].values  # 100 genes
    angles = 2 * np.pi * peak_hours / 24

    #pal = sns.color_palette('vlag', 5).as_hex()

    # Histogram
    num_bins = 24
    bins = np.linspace(0, 2 * np.pi, num_bins + 1)
    counts, _ = np.histogram(angles, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2


    # === Step 1: Draw background half-sectors ===
    # Fill from 0 to π (e.g., "night")
    ax.bar(x=np.linspace(0, np.pi, 100), height=[max(counts)*1.2]*100,
           width=np.pi/100, bottom=0, color=pal[0], alpha=1, edgecolor='none', zorder=-10)

    # Fill from π to 2π (e.g., "day")
    ax.bar(x=np.linspace(np.pi, 2*np.pi, 100), height=[max(counts)*1.2]*100,
           width=np.pi/100, bottom=0, color=pal[1], alpha=1, edgecolor='none', zorder=-10)

    # === Step 2: Draw actual data bars ===
    bars = ax.bar(bin_centers, counts, width=2*np.pi/num_bins, bottom=0.0,
                  align='center', alpha=1, color='#022F40', edgecolor='k')

    # === Step 3: Styling ===
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    xtick_hours = [0, 6, 12, 18]
    xtick_angles = [2 * np.pi * h / 24 for h in xtick_hours]

    ax.set_xticks(xtick_angles)
    ax.set_xticklabels([str(h) for h in xtick_hours])
    #ax.set_yticklabels([i for i in range(, 10, 2)])
    ax.set_ylim(0, max(counts)*1.2)
    plt.locator_params(axis='y', nbins=2)

# NOTE: `plot_entrainment(fig, ...)` was removed in v0.7.7. It drew its bands with
# `plt.axvspan`, i.e. onto matplotlib's *current* axes rather than the one it was
# handed. In a single-axes figure that happened to be right; in a grid it put every
# band on the last-created panel, which is why a multi-condition report page showed
# the zeitgeber shading only under the final condition. `plot_entrainment_ax` below
# was always the correct one — everything uses it now.

def plot_entrainment_ax(ax, plot, t_col, xtick_start, xtick_end, ent_days, order=0, T=24, color='#EBEBEB'):
        
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
                    ax.axvspan(band_start, band_end, color=color, alpha=1, zorder=-10)

def plot(df, t_col, p_col, t0, t1, bg_color='white', ent=False, ent_days=0, features=None, 
         order=0, T=24, color='white', unit='Measured unit'):
    
    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.set_facecolor(bg_color)
    
    plot = df[(df[t_col] >= t0) & (df[t_col] <= t1) ]
    #plt.plot(plot[t_col], plot[p_col])
    sns.lineplot(plot, x=t_col, y=p_col)
    
    scat = st.toggle('Show datapoints', True)
    if scat:
        sns.scatterplot(plot, x=t_col, y=p_col, edgecolor='k', zorder=10)
    
    # Get actual min and max from your data
    xmin = plot[t_col].min()
    xmax = plot[t_col].max()
    
    ymin = plot[p_col].min()
    ymax = plot[p_col].max()
    
    # Añadir margen manual (5%)
    x_margin = (xmax - xmin) * 0.07
    y_margin = (ymax - ymin) * 0.07

    # Calculate start and end of xticks, rounded to nearest multiples of 24
    xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
    xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
    
    if features is not None:
        #st.write(features)
        vmin, vmax = plot[p_col].min(), plot[p_col].max()
        feature_entrainment(ax, features, bg_color, color, ymin - y_margin, ymax + y_margin,
            order=order)
    
    ax.set_xlim(xmin - x_margin, xmax + x_margin)
    ax.set_ylim(ymin - y_margin, ymax + y_margin)
    #if ent_days > 0:
        # Example for creating banded background every 12 hours
        
    #    fig = plot_entrainment(fig, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
    
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
    
    if ent_days > 0:
        # Example for creating banded background every 12 hours

        plot_entrainment_ax(ax, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
    
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
    
    if ent_days > 0:
        # Example for creating banded background every 12 hours
        plot_entrainment_ax(ax, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
    
    # Generate ticks at every 24 units
    xticks = np.arange(xtick_start, xtick_end + 1, 24)
    plt.xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
    plt.xlabel('Time (h)')
    plt.ylabel(unit)
    plt.title(f"{group} (N={len(cols)})", fontsize=15)
    return fig

def double_plot(df, t_col, p_col, ent_days, T, order, t0, t1, times = 2, entrainment_data=None,
                bg_color='white', band_color='white', signal_color='#1F7A8C'):
    
    df = df[(df[t_col] >= t0) & (df[t_col] <= t1)]
    df[t_col] = df[t_col] - t0
    df['d'] = df[t_col].apply(lambda x: int(x/24))
    df[p_col] = (df[p_col] - df[p_col].min()) / (df[p_col].max() - df[p_col].min())

    if entrainment_data is not None:

        if 'entrainment' not in entrainment_data.columns:
            entrainment_data['entrainment'] = entrainment_data.iloc[:,-1]
        
        entrainment_data['entrainment'] = (entrainment_data['entrainment'] - entrainment_data['entrainment'].min()) / (entrainment_data['entrainment'].max() - entrainment_data['entrainment'].min())

    days = int(np.round(df[t_col].max() / 24))

    yscaling = st.checkbox('Re-scale Y axis by subset amplitude', False)
    global_vmean = df[p_col].mean()   # robust to outliers
    global_vmax = np.nanpercentile(df[p_col], 99)
    
    if days < 2:
        st.error('This function needs more than 2 days to plot')
        st.stop()

    fig, ax = plt.subplots(days, 1, figsize=(10, days*0.5))
    
    for i in range(1, days+1):
        
        bot = (i * 24 - 24 * times) 
        #top = (i * 24 * times)
        #bot = (i - 1) * 24
        top = bot + 24 * times
        plot = df[(df[t_col].between(bot, top, inclusive='both'))]
 
        plot['time_col'] = plot[t_col]
        plot['time_col'] = plot['time_col'].apply(lambda x: x- bot)
        
        ax[i -1 ].set_facecolor(bg_color)
        
        ax2 = ax[i -1 ].twinx()
        ax2.fill_between(plot.time_col, plot[p_col], color=signal_color, zorder=10)
        ax2.set_xlim(0, 24*times)    
        
        # Get the actual ylim being used for this subplot
        if yscaling:
            y_lo = plot[p_col].mean()
            y_hi = np.nanpercentile(plot[p_col], 99) + np.nanpercentile(plot[p_col], 99) * 0.07#plot[p_col].mean() + plot[p_col].std() * 2
        else:
            y_lo = global_vmean
            y_hi = global_vmax

        ax2.set_ylim(y_lo, y_hi)
        ax[i -1 ].set_yticks([])
        ax2.set_yticks([])

        # x-tick spacing
        dist = {1: 6}.get(times) or (12 if times == 2 else 24)
        
        if i == days:
            ax[i -1 ].set_xticks([i for i in range(0, 24*times+1, dist)])
        else:
            ax[i -1 ].set_xticks([])
        
        ax[i - 1 ].set_ylabel(f"Day {i}", rotation=0, ha='right', va='center')
        
        if entrainment_data is not None:

            ax[i-1].set_ylim(y_lo, y_hi)

            plot_entrainment = entrainment_data[(entrainment_data[t_col].between(bot, top, inclusive='both'))].copy()
            
            # Guard against empty slice
            if plot_entrainment.empty:
                continue
                
            plot_entrainment['time_col'] = plot_entrainment[t_col].apply(lambda x: x - bot)

            xmin = plot['time_col'].min()
            xmax = plot['time_col'].max()
            ymin = entrainment_data['entrainment'].min()
            ymax = entrainment_data['entrainment'].max()

            # Guard against NaN in plot data itself
            if any(np.isnan(v) or np.isinf(v) for v in [xmin, xmax, ymin, ymax]):
                continue

            y_margin = (ymax - ymin) * 0.07
            # Calculate start and end of xticks, rounded to nearest multiples of 24
            xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
            xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24
            colors = [band_color, bg_color] if order == 0 else [bg_color, band_color]
            cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
            ax[i -1 ].imshow(np.vstack((plot_entrainment['entrainment'],)), 
                extent=(plot_entrainment['time_col'].min(),
                        plot_entrainment['time_col'].max(), 
                        ymin, ymax), aspect='auto' ,cmap=cmap1, zorder=-2,
                        vmin=global_vmean, vmax=global_vmax)
            ax[i -1 ].set_ylim(ymin, ymax)

    fig.subplots_adjust(hspace=0)
    return fig

def simple_plot(df, t_col, col, 
                unit='Measured unit', 
                bg_color='white', title=None):
                
        fig, ax = plt.subplots(1, figsize=(12, 7))
        ax.set_facecolor(bg_color)
        #ax.plot(df[t_col], df[col])
        sns.lineplot(df, x=t_col, y=col, ax=ax)
        if title == None:
            ax.set_title(col)
        else:
            plt.suptitle(title)
        
        xmin = df[t_col].min()
        xmax = df[t_col].max()
        
        # callate start and end of xticks, rounded to nearest multiples of 24
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
        
        if ent_days > 0:
            
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
    
    if ent_days > 0:
        # Example for creating banded background every 12 hours
        plot_entrainment_ax(ax, plot, t_col, xtick_start, xtick_end, ent_days, order=order, T=T, color=color)
    
    # Generate ticks at every 24 units
    xticks = np.arange(xtick_start, xtick_end + 1, 24)
    ax.set_xticks([i for i in range(int(xtick_start), int(xtick_end), 24)])
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(unit)
    ax.set_title(f"{group} (N={len(cols)})", fontsize=15, loc='left')
    return ax

def pie_chart(ax, df, method='meta2d', group='', thresh=0.05):
        
        group = group.replace('_', '-')
        cols = [col for col in df.columns if method in col]
        q_col = [col for col in cols if 'BH.Q' in col.upper()][0]
        
        significant = df[df[q_col] <= thresh]
        
        replicates = df.shape[0]
        sig_replicates = significant.shape[0]
        percent = np.round(sig_replicates / replicates * 100, 1)
        not_sig = 100 - percent

        vlag_pal = sns.color_palette('vlag', 6)
        pal = [vlag_pal[0], vlag_pal[-1]]#['#57C4E5','#F97068']

        ax.pie([percent, not_sig], labels=['Significant', 'Not significant'], 
        autopct='%1.1f%%', colors=pal, startangle=90,
        wedgeprops=dict(width=0.6))

def text(ax, df, method='meta2d', group='', thresh=0.05):
    """
    Compact header block for a condition page.

    Rewritten in v0.7.7: it was five stacked lines at fontsize 15 filling a panel
    that took a third of the page. Now one heading and a single run-on line of
    facts, so the traces below get the space. Also tolerates a method with no
    q-value column (Tempo), where the cutoff is a probability, not significance.
    """
    label = str(group).replace('_', '-')
    cols = [col for col in df.columns if method in col]
    q_cols = [col for col in cols if 'BH.Q' in col.upper()]

    n = df.shape[0]
    facts = [f"n = {n}"]

    if q_cols:
        sig = int((df[q_cols[0]] <= thresh).sum())
        if str(method).lower().startswith('tempo'):
            # thresh is stored as 1 - p_min so the q machinery works; show the
            # probability the user actually chose.
            rule = f"P ≥ {1 - thresh:.2f}"
        else:
            rule = f"q ≤ {thresh:g}"
        facts.append(f"rhythmic {sig}/{n} ({100 * sig / n:.0f}%)")
        facts.append(f"{method}, {rule}")
    else:
        facts.append(f"{method}")

    if 'Periods' in df.columns and np.isfinite(df['Periods'].mean()):
        facts.append(f"τ = {df['Periods'].mean():.1f} ± {df['Periods'].std():.1f} h")

    ax.axis("off")
    ax.text(0, 1.0, label, fontsize=15, fontweight='bold',
            va='top', ha='left', transform=ax.transAxes)
    ax.text(0, 0.52, "   ·   ".join(facts), fontsize=11,
            va='top', ha='left', transform=ax.transAxes, color='#333333')

def multi_acto(ax, df, t_col, p_col, ent_days, T, order, t0, t1, times = 2, entrainment_data=None,
                    yscaling=False, bg_color='white', band_color='white', signal_color='#1F7A8C', title='None'):
        
        df = df[(df[t_col] >= t0) & (df[t_col] <= t1)]
        df[t_col] = df[t_col] - t0
        df['d'] = df[t_col].apply(lambda x: int(x/24))
        df[p_col] = (df[p_col] - df[p_col].min()) / (df[p_col].max() - df[p_col].min())

        days = int(np.round(df[t_col].max() / 24))

        global_vmean = df[p_col].mean()   # robust to outliers
        global_vmax = np.nanpercentile(df[p_col], 99)
        
        if entrainment_data is not None:

            if 'entrainment' not in entrainment_data.columns:
                entrainment_data['entrainment'] = entrainment_data.iloc[:,-1]
            
            entrainment_data['entrainment'] = (entrainment_data['entrainment'] - entrainment_data['entrainment'].min()) / (entrainment_data['entrainment'].max() - entrainment_data['entrainment'].min())
        
        if days < 2:
            st.error('This function needs more than 2 days to plot')
            st.stop()

        #fig, ax = plt.subplots(days, 1, figsize=(10, days*0.5))
        if title != 'None':
            ax[0].set_title(title)

        for i in range(1, days+1):
            
            bot = (i * 24 - 24 * times) 
            top = bot + 24 * times
            plot = df[(df[t_col].between(bot, top, inclusive='both'))]

            plot['time_col'] = plot[t_col]
            plot['time_col'] = plot['time_col'].apply(lambda x: x- bot)
            
            ax[i -1 ].set_facecolor(bg_color)
            ax2 = ax[i -1 ].twinx()

            ax2.fill_between(plot.time_col, plot[p_col], color=signal_color)
            ax2.set_xlim(0, 24*times)    
            
            # Get the actual ylim being used for this subplot
            if yscaling:
                y_lo = plot[p_col].mean()
                y_hi = np.nanpercentile(plot[p_col], 99) + np.nanpercentile(plot[p_col], 99) * 0.07#plot[p_col].mean() + plot[p_col].std() * 2
            else:
                y_lo = global_vmean
                y_hi = global_vmax

            ax2.set_ylim(y_lo, y_hi)
            ax[i -1 ].set_yticks([])
            ax2.set_yticks([])
            
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

            if entrainment_data is not None:

                plot_entrainment = entrainment_data[(entrainment_data[t_col].between(bot, top, inclusive='both'))].copy()
                
                # Guard against empty slice
                if plot_entrainment.empty:
                    continue
                    
                plot_entrainment['time_col'] = plot_entrainment[t_col].apply(lambda x: x - bot)

                xmin = plot['time_col'].min()
                xmax = plot['time_col'].max()
                ymin = entrainment_data['entrainment'].min()
                ymax = entrainment_data['entrainment'].max()

                # Guard against NaN in plot data itself
                if any(np.isnan(v) or np.isinf(v) for v in [xmin, xmax, ymin, ymax]):
                    continue

                y_margin = (ymax - ymin) * 0.07

                # Calculate start and end of xticks, rounded to nearest multiples of 24
                xtick_start = (xmin // 24) * 24          # floor to nearest lower multiple of 24
                xtick_end = ((xmax // 24) + 1) * 24      # ceil to next multiple of 24

                colors = [band_color, bg_color] if order == 0 else [bg_color, band_color]
                cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
                ax[i -1 ].imshow(np.vstack((plot_entrainment['entrainment'],)), 
                        extent=(plot_entrainment['time_col'].min(),
                        plot_entrainment['time_col'].max(), 
                        y_lo, y_hi), aspect='auto' ,cmap=cmap1, zorder=-2,
                        vmin=global_vmean, vmax=global_vmax)
            
        return

def feature_entrainment(ax, feat_data, ent_color, bg_color, lower, upper, order=0):

    colors = [ent_color, bg_color] if order == 0 else [bg_color, ent_color]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    #plt.plot(feat_data.iloc[:, 0], feat_data.iloc[:, 1])
    ax.imshow(np.vstack((feat_data.iloc[:, 1],)), 
        extent=(feat_data.iloc[:, 0].min() ,feat_data.iloc[:, 0].max(), 
                    lower, upper), aspect='auto'
                    , cmap=cmap1, vmin=feat_data.iloc[:, 1].min(), vmax=feat_data.iloc[:, 1].max())



# ═══════════════════════════════════════════════════════════════════════════
#  Comparison views (v0.7.4)
# ═══════════════════════════════════════════════════════════════════════════

# Fixed categorical order for comparison plots, assigned by position and never
# cycled. Chosen by searching 420 five-colour combinations of the seaborn
# `colorblind` and Chronotopia reference palettes against the all-pairs gates:
# worst pair is ΔE 13.0 (deutan) / 14.9 (tritan) for colour-vision deficiency and
# 16.3 for normal vision, on a light surface.
#
# All-pairs — not adjacent-pairs — is the right gate here: the entire point of
# these views is comparing any series to any other, so every pair must be
# separable, not just neighbours in the legend.
#
# For the record, seaborn's `colorblind` slots 1-4 FAIL this test: #de8f05 and
# #d55e00 sit at ΔE 11.4 for normal vision, below the floor of 15. That palette
# is fine for the app's other plots (where series are read one at a time) but not
# for side-by-side comparison.
COMPARE_PALETTE = ["#d55e00", "#56b4e9", "#2a78d6", "#eda100", "#4a3aa7"]

MAX_COMPARE_SAMPLES = 5
MAX_COMPARE_GROUPS = 4


def _compare_colors(n, palette=None):
    """
    n colours in fixed order. A custom palette is used as given.

    Asking for more colours than the palette holds is an ERROR, not something to
    paper over. This used to wrap — `source * (n // len(source) + 1)` — on the
    reasoning that a secondary style encoding would carry the difference. That
    holds for the trace views, which cap at MAX_COMPARE_SAMPLES / _GROUPS and
    direct-label every series. It did not hold for `feature_volcano`, the one
    caller that can exceed the palette: it passes one colour per feature CONCEPT,
    of which there are ten, so Period and Waveform shape came out the same orange
    and Phase and Harmonics the same sky blue, with nothing but the legend order
    to tell them apart and no secondary encoding at all. Wrapping there did not
    degrade the figure gracefully, it made it wrong.

    Callers that genuinely have more categories than hues must add a second
    channel, fold the tail into one "Other", or facet — anything but reuse a hue.
    `feature_volcano` takes the first route: it builds its own concept keys from
    this palette crossed with two marker shapes, so it never calls this with more
    than five.
    """
    source = list(palette) if palette else COMPARE_PALETTE
    if n > len(source):
        raise ValueError(
            f"{n} categories were asked of a {len(source)}-colour palette. "
            "Reusing a hue makes two categories indistinguishable — facet the "
            "figure or fold the smallest categories into one 'Other' instead."
        )
    return source[:n]


def _place_end_labels(ax, entries, x_end, fontsize=9):
    """
    Direct-label each series at its right-hand end, nudging labels apart when
    they would collide.

    Two reasons this exists rather than relying on the legend alone: several of
    the palette steps sit below 3:1 contrast against a white surface, and the
    validator's relief rule requires visible labels in that case; and a reader
    tracing one trace among five should not have to keep looking away to a
    legend box.

    `entries` is [(y_value, text, colour), ...].
    """
    if not entries:
        return
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    min_gap = span * 0.055

    ordered = sorted(entries, key=lambda e: e[0])
    ys = [e[0] for e in ordered]
    for i in range(1, len(ys)):                       # push overlapping up
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    excess = ys[-1] - (hi - span * 0.02)
    if excess > 0:                                    # then slide the stack back
        ys = [y - excess for y in ys]

    for y, (_, text, color) in zip(ys, ordered):
        ax.annotate(text, xy=(x_end, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=fontsize, fontweight="bold", color=color,
                    va="center", ha="left", annotation_clip=False)


def _time_axis(ax, xmin, xmax, unit):
    xtick_start = (xmin // 24) * 24
    xtick_end = ((xmax // 24) + 1) * 24
    ax.set_xticks([i for i in range(int(xtick_start), int(xtick_end) + 1, 24)])
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(unit)
    return xtick_start, xtick_end


def _entrainment_backdrop(ax, plot, t_col, features, ent_days, order, T, color,
                          xtick_start, xtick_end):
    """Shared zeitgeber shading, matching the single-trace Lineplot."""
    if features is not None:
        ymin, ymax = ax.get_ylim()
        feature_entrainment(ax, features, color, 'white', ymin, ymax, order=order)
    elif ent_days > 0:
        plot_entrainment_ax(ax, plot, t_col, xtick_start, xtick_end,
                            ent_days, order=order, T=T, color=color)


def _reserve_margins(fig, direct_labels):
    """
    Reserve figure margins explicitly instead of calling tight_layout().

    tight_layout() shrinks the axes to fit the artists it knows about, but the
    end-of-line labels are annotations with annotation_clip=False living outside
    the axes — it does not reserve space for them. app.py exports with a plain
    fig.savefig(buf, format="svg"), with no bbox_inches="tight" to rescue them,
    so without this the downloaded SVG would have the labels cut off at the edge.
    """
    fig.subplots_adjust(left=0.085, right=0.845 if direct_labels else 0.98,
                        top=0.845, bottom=0.145)


def compare_samples(df, t_col, cols, t0, t1, unit='Measured unit', bg_color='white',
                    features=None, ent_days=0, order=0, T=24, color='white',
                    palette=None, show_points=False, direct_labels=True,
                    show_sd=True, n_replicates=1, figsize=(10, 4.5)):
    """
    Overlay up to five individual traces for direct comparison.

    Replicate timepoints are averaged per sample; with `show_sd` the spread
    across replicates is drawn as a band, so a difference between two samples can
    be read against the noise within each.
    """
    cols = list(cols)[:MAX_COMPARE_SAMPLES]
    if not cols:
        raise ValueError("Select at least one sample to compare.")

    plot = df[(df[t_col] >= t0) & (df[t_col] <= t1)]
    colors = _compare_colors(len(cols), palette)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_facecolor(bg_color)

    has_reps = np.mean(n_replicates) > 1
    ends = []
    for col, c in zip(cols, colors):
        if has_reps:
            grouped = plot.groupby(t_col)[col]
            x, mu, sd = grouped.mean().index.values, grouped.mean().values, grouped.std().values
            if show_sd:
                ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.16,
                                linewidth=0, zorder=1)
        else:
            x, mu = plot[t_col].values, plot[col].values

        ax.plot(x, mu, color=c, lw=2.0, label=str(col), zorder=3,
                solid_capstyle="round")
        if show_points:
            ax.scatter(x, mu, color=c, s=14, edgecolor='k', linewidth=0.4, zorder=4)
        if len(mu):
            ends.append((float(mu[-1]), str(col), c))

    xmin, xmax = plot[t_col].min(), plot[t_col].max()
    xtick_start, xtick_end = _time_axis(ax, xmin, xmax, unit)
    ax.set_xlim(xmin - (xmax - xmin) * 0.02, xmax + (xmax - xmin) * 0.02)

    _entrainment_backdrop(ax, plot, t_col, features, ent_days, order, T, color,
                          xtick_start, xtick_end)

    # A legend is always present for >= 2 series; direct labels are the relief
    # for the palette steps that fall below 3:1 contrast on a light surface.
    if direct_labels:
        _place_end_labels(ax, ends, xmax)
    if len(cols) >= 2:
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.005), ncol=min(len(cols), 5),
                  frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.4)
    ax.set_title(f"Comparison of {len(cols)} samples", loc='left', fontsize=12,
                 fontweight='bold', pad=26 if len(cols) >= 2 else 8)
    _reserve_margins(fig, direct_labels)
    return fig


def compare_groups(df, t_col, groups, layout, t0, t1, style='Mean ± SD',
                   unit='Measured unit', bg_color='white', features=None,
                   ent_days=0, order=0, T=24, color='white', palette=None,
                   direct_labels=True, figsize=(10, 4.5)):
    """
    Compare two to four experimental conditions on one axis.

    style = "Mean ± SD"          mean per condition with an SD ribbon
    style = "Mean + Replicates"  mean per condition over its faded individual traces

    These are the same two summaries as the existing grouped Lineplots, drawn
    together so conditions can be read against each other rather than by flicking
    between figures.
    """
    groups = list(groups)[:MAX_COMPARE_GROUPS]
    if len(groups) < 2:
        raise ValueError("Select at least two conditions to compare.")

    plot = df[(df[t_col] >= t0) & (df[t_col] <= t1)]
    colors = _compare_colors(len(groups), palette)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_facecolor(bg_color)

    x = plot[t_col].values
    ends = []
    for group, c in zip(groups, colors):
        cols = [s for s in layout[layout.Condition == group]['name'].to_list()
                if s in plot.columns]
        if not cols:
            continue
        mu = plot[cols].mean(axis=1).values
        sd = plot[cols].std(axis=1).values

        if style == 'Mean + Replicates':
            for col in cols:
                ax.plot(x, plot[col].values, color=c, lw=1.0, alpha=0.22, zorder=2)
        else:
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18,
                            linewidth=0, zorder=1)

        label = f"{group} (N={len(cols)})"
        ax.plot(x, mu, color=c, lw=2.2, label=label, zorder=3, solid_capstyle="round")
        if len(mu):
            ends.append((float(mu[-1]), str(group), c))

    xmin, xmax = plot[t_col].min(), plot[t_col].max()
    xtick_start, xtick_end = _time_axis(ax, xmin, xmax, unit)
    ax.set_xlim(xmin - (xmax - xmin) * 0.02, xmax + (xmax - xmin) * 0.02)

    _entrainment_backdrop(ax, plot, t_col, features, ent_days, order, T, color,
                          xtick_start, xtick_end)

    if direct_labels:
        _place_end_labels(ax, ends, xmax)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.005), ncol=min(len(groups), 4),
              frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.4)
    ax.set_title(f"{len(groups)} conditions · {style}", loc='left', fontsize=12,
                 fontweight='bold', pad=26)
    _reserve_margins(fig, direct_labels)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Period sweep (v0.7.9)
# ═══════════════════════════════════════════════════════════════════════════

def period_sweep(landscapes, results, r2_thresh=0.3, peaks=None,
                 period_min=None, period_max=None, palette=None,
                 bins=60, figsize=(11, 6.5), title=None):
    """
    Two views of the same sweep, stacked on a shared period axis.

    Top — aggregate fit quality against trial period, averaged over signals.
    Bottom — how many signals fit *best* at each period.

    Both are needed, and they answer different questions. The histogram is what
    you had: each signal contributes one count, at its winning period. That finds
    the dominant components but hides everything else, because a gene whose best
    fit is 24 h contributes nothing at 12 h even when it carries a strong 12 h
    harmonic. The aggregate on top keeps that: a secondary bump there with no
    matching bar below means a component that is widespread but rarely dominant.

    `landscapes` and `results` are dicts keyed by group name, so one call covers
    both the ungrouped case ({"All samples": ...}) and a per-condition split.
    """
    names = list(landscapes.keys())
    colors = _compare_colors(len(names), palette)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw=dict(height_ratios=(1.15, 1), hspace=0.12),
    )

    lo = period_min if period_min is not None else min(l["period"].min() for l in landscapes.values())
    hi = period_max if period_max is not None else max(l["period"].max() for l in landscapes.values())

    # ── aggregate landscape ─────────────────────────────────────────────────
    for name, c in zip(names, colors):
        land = landscapes[name]
        ax_top.plot(land["period"], land["mean_r2"], color=c, lw=2.0,
                    label=str(name), solid_capstyle="round")
    ax_top.set_ylabel("Mean R²  (all signals)")
    ax_top.set_xlim(lo, hi)

    if peaks is not None and len(peaks):
        span = ax_top.get_ylim()[1] - ax_top.get_ylim()[0]
        for _, pk in peaks.iterrows():
            ax_top.axvline(pk["period"], color="#52514e", lw=0.9, ls=":", zorder=0)
            ax_top.annotate(
                f"{pk['period']:.1f} h", xy=(pk["period"], pk["mean_r2"]),
                xytext=(0, 7), textcoords="offset points", ha="center",
                fontsize=9, fontweight="bold", color="#0b0b0b",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
            )
        ax_top.set_ylim(top=ax_top.get_ylim()[1] + span * 0.12)

    # ── best-period histogram ───────────────────────────────────────────────
    edges = np.linspace(lo, hi, bins + 1)
    kept, total = 0, 0
    for name, c in zip(names, colors):
        res = results[name]
        total += len(res)
        sel = res[res["r2"] >= r2_thresh]
        kept += len(sel)
        ax_bot.hist(sel["period"], bins=edges, color=c, alpha=0.65,
                    edgecolor="white", linewidth=0.4, label=str(name))
    ax_bot.set_xlabel("Period (h)")
    ax_bot.set_ylabel(f"Signals best fit here\n(R² ≥ {r2_thresh:g})")

    if peaks is not None and len(peaks):
        for _, pk in peaks.iterrows():
            ax_bot.axvline(pk["period"], color="#52514e", lw=0.9, ls=":", zorder=0)

    if len(names) > 1:
        ax_top.legend(loc="lower left", bbox_to_anchor=(0, 1.005),
                      ncol=min(len(names), 4), frameon=False, fontsize=9)

    head = title or "Period sweep"
    ax_top.set_title(f"{head}   ·   {kept} of {total} signals above R² {r2_thresh:g}",
                     loc="left", fontsize=12, fontweight="bold",
                     pad=24 if len(names) > 1 else 8)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.10)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Feature analytics (v0.7.10)
# ═══════════════════════════════════════════════════════════════════════════

# ── concept keys for the volcano ────────────────────────────────────────────
#
# Ten concepts on one set of axes, and hue alone cannot carry ten categories.
# That is measured, not assumed: run any published ten-colour categorical palette
# through the all-pairs colour-vision gate and it fails. Tol's muted set collapses
# to dE 5.2 between #44AA99 and #CC6677 under deuteranopia; tab10 collapses to
# dE 0.7 between its green and its orange under protanopia, and its brown and grey
# are dE 11.3 apart even with full colour vision. The five hues in COMPARE_PALETTE
# pass all-pairs with room to spare (worst dE 13.0 deutan, 16.3 normal), and that
# is about the ceiling for hue on its own.
#
# So concept is carried by TWO channels at once: hue for position within the half,
# marker shape for which half. Ten concepts, ten combinations, no two alike, and
# every pair of same-hue concepts is separated by a shape rather than by a shade
# a reader has to squint at.
#
# The split follows CONCEPT_ORDER and is not arbitrary — the first five describe
# the rhythm itself, the last five describe spectral content, artefacts and
# metadata, which is roughly the "is this biology or is this my pipeline?" line
# the features page asks readers to hold.
VOLCANO_SHAPES = ("o", "s")          # rhythm concepts, then quality concepts
VOLCANO_QUIET_EDGE = "#8A8A8A"


def _volcano_style(concepts):
    """{concept: (colour, marker)} — hue cycles within a half, shape marks the half."""
    n_hues = len(COMPARE_PALETTE)
    style = {}
    for i, concept in enumerate(concepts):
        style[concept] = (COMPARE_PALETTE[i % n_hues],
                          VOLCANO_SHAPES[min(i // n_hues, len(VOLCANO_SHAPES) - 1)])
    return style


def feature_volcano(results, meta, top_n=6, alpha=0.05, figsize=(11, 6.5)):
    """
    Every feature at once: effect size against FDR-corrected significance.

    Replaces clicking through 100 boxplots. The x-axis is what matters
    scientifically — a q-value says a difference is detectable, an effect size
    says whether it is worth anything — so this is read left-to-right.

    Points are marked by concept, so a column of hits that all belong to
    "Period" reads as one finding rather than fourteen.

    Concept is carried by hue AND marker shape together — see `_volcano_style`
    for why hue alone will not do it at ten categories. Until v0.7.6 this asked
    `_compare_colors` for ten colours from a five-colour palette and got the
    palette back twice, so Period and Harmonics were the same orange with nothing
    to tell them apart. Now every concept has its own hue-and-shape pair.

    Both axes are clipped to a robust range, with out-of-range points marked by a
    caret at the boundary. Without that, one feature at q = 0 (perfect separation,
    which a 0/1 flag feature can produce) sets the y-axis to 300 and flattens
    everything else into the baseline. The caret is drawn beside the point rather
    than replacing its marker, so a clipped point keeps its concept.
    """
    import features as ft

    def _empty(msg):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", color="gray")
        return fig

    if results is None or results.empty:
        return _empty("No feature had enough data in both groups to test.")

    res = results.dropna(subset=["effect", "q"]).copy()
    if res.empty:
        return _empty("No feature produced a usable effect size.")

    res["neglog_q"] = -np.log10(res["q"].clip(lower=1e-300))

    # Robust display limits — the data are not clipped, only the view.
    y_cap = max(float(np.nanpercentile(res["neglog_q"], 95)) * 1.25,
                -np.log10(alpha) * 2.0)
    x_ref = float(np.nanpercentile(res["effect"].abs(), 95))
    x_cap = max(x_ref * 1.35, 0.5)

    res["y_plot"] = res["neglog_q"].clip(upper=y_cap)
    res["x_plot"] = res["effect"].clip(-x_cap, x_cap)
    res["clipped"] = (res["neglog_q"] > y_cap) | (res["effect"].abs() > x_cap)

    # Cliff's delta saturates at ±1 the moment two groups of six stop overlapping,
    # which on a typical comparison is most of the significant features. Drawn at
    # their true coordinates they land on top of each other: 61 hits render as
    # about six visible marks, and the figure silently under-reports itself.
    # Points sharing a coordinate are spread horizontally, deterministically and
    # by a bounded amount, so the size of the pile is visible. The axis label says
    # this is happening — the spread is a drawing device, not a measurement.
    # The spread is deliberately TIGHT. A whole pile never occupies more than
    # SPREAD_SPAN of the x-range, so a reader can see that a mark is thirty
    # features deep without ever reading it as thirty different effect sizes. An
    # unbounded spread (constant step per point) turned a stack of 30 into a bar
    # running off the axis, which is a worse lie than the overplotting it fixed.
    # The exact counts live in the legend, which reads "Period (11/17)".
    SPREAD_SPAN = 0.09
    x_edge = float(res["x_plot"].abs().max())
    # Group on the (x, y) PAIR: points only overplot when both coordinates match.
    # `pd.Index` of 2-tuples groups on the x alone and hands back a bare float, so
    # the unpacking below fails — it has to be a MultiIndex grouped on both levels.
    key = pd.MultiIndex.from_tuples(
        list(zip(res["x_plot"].round(3), res["y_plot"].round(2))))
    offsets = np.zeros(len(res))
    for (kx, _), idx in pd.Series(range(len(res)), index=key).groupby(level=[0, 1]):
        k = len(idx)
        if k <= 1:
            continue
        step = min(0.018 * x_cap, SPREAD_SPAN * x_cap / (k - 1))
        if abs(kx) >= 0.995 * x_edge and x_edge > 0:
            # A pile sitting on the maximum effect spreads INWARD only. Cliff's
            # delta cannot exceed 1, and a symmetric spread there would draw
            # points past a bound the statistic has — a reader who knows the
            # measure would rightly distrust the figure.
            offsets[idx.to_numpy()] = -np.sign(kx) * np.arange(k) * step
        else:
            offsets[idx.to_numpy()] = (np.arange(k) - (k - 1) / 2) * step
    res["x_draw"] = np.clip(res["x_plot"].to_numpy() + offsets, -x_edge, x_edge)
    res["spread"] = np.abs(offsets) > 0

    concepts = [c for c in ft.CONCEPT_ORDER if c in set(res["concept"])]
    concepts += [c for c in sorted(set(res["concept"])) if c not in concepts]
    style = _volcano_style(concepts)

    fig, ax = plt.subplots(figsize=figsize)
    handles = []
    for concept in concepts:
        color, marker = style[concept]
        sub = res[res["concept"] == concept]

        # Two significance steps. A hit is bigger, fully opaque and carries a dark
        # edge; a tested-but-quiet feature is small, translucent and edged in grey.
        # Significance is therefore readable without colour at all, which matters
        # because two of the five hues sit below 3:1 against white.
        for is_sig, size, alpha_pt, edge, lw in (
                (False, 26, 0.45, VOLCANO_QUIET_EDGE, 0.4),
                (True, 66, 0.95, "#20201e", 0.6)):
            part = sub[sub["significant"] == is_sig]
            if part.empty:
                continue
            ax.scatter(part["x_draw"], part["y_plot"], s=size, color=color,
                       marker=marker, alpha=alpha_pt, edgecolors=edge,
                       linewidths=lw, zorder=3)

        n_sig = int(sub["significant"].sum())
        handles.append(Line2D([], [], marker=marker, linestyle="none", color=color,
                              markersize=7, markeredgecolor="#20201e",
                              markeredgewidth=0.6,
                              label=f"{concept}  ({n_sig}/{len(sub)})"))

    # Clipped points keep their concept marker and gain a caret pointing off-axis.
    out = res[res["clipped"]]
    if not out.empty:
        ax.scatter(out["x_draw"], out["y_plot"], s=90, facecolors="none",
                   edgecolors="#20201e", linewidths=0.8, marker="o", zorder=4)

    ax.axhline(-np.log10(alpha), color="#52514e", lw=1.0, ls="--", zorder=1)
    ax.axvline(0, color="#CFCFCB", lw=1.0, zorder=0)
    ax.set_xlim(-x_cap * 1.05, x_cap * 1.05)
    ax.set_ylim(-y_cap * 0.04, y_cap * 1.30)
    ax.annotate(f"q = {alpha:g}", xy=(-x_cap * 1.03, -np.log10(alpha)),
                xytext=(0, 3), textcoords="offset points", fontsize=8.5,
                color="#52514e", va="bottom")

    # Label the strongest hits by EFFECT, not by p — a tiny difference measured
    # precisely is not the headline. Stagger to reduce collisions.
    hits = res[res["significant"]]
    hits = hits.reindex(hits["effect"].abs().sort_values(ascending=False).index)
    # One label per arm at a time, alternating, and stacked upwards in even steps.
    # The top hits nearly all sit at the same saturated coordinate, so a small
    # three-way stagger was not enough — they wrote over each other.
    left_i = right_i = 0
    for _, r in hits.head(top_n).iterrows():
        to_left = r["x_draw"] > 0
        if to_left:
            dy = 14 + 13 * right_i
            right_i += 1
        else:
            dy = 14 + 13 * left_i
            left_i += 1
        ax.annotate(r["feature"], xy=(r["x_draw"], r["y_plot"]),
                    xytext=(-9 if to_left else 9, dy),
                    textcoords="offset points", fontsize=7.5, color="#0b0b0b",
                    ha="right" if to_left else "left", zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#8A8A8A",
                                    shrinkA=0, shrinkB=2),
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    n_clipped = int(res["clipped"].sum())
    n_spread = int(res["spread"].sum())
    # The notes go on a second line. Run end-to-end they overflow the figure on a
    # comparison with many ties, which is exactly the comparison that needs them.
    notes = []
    if n_spread:
        notes.append(f"{n_spread} tied points spread sideways to show the pile")
    if n_clipped:
        notes.append(f"{n_clipped} beyond the axes, ringed")
    ax.set_xlabel(
        f"{meta['effect_name']}   ·   negative = lower in {meta['group_a']}"
        + ("\n" + "   ·   ".join(notes) if notes else ""))
    ax.set_ylabel("-log10  FDR q-value")
    ax.set_title(f"{meta['group_a']} vs {meta['group_b']}   ·   "
                 f"{meta['n_significant']} of {meta['n_tested']} features differ",
                 loc="left", fontsize=12.5, fontweight="bold", pad=22)
    ax.annotate(f"n = {meta['n_a']} vs {meta['n_b']}   ·   {meta['reason']}   ·   "
                f"filled = q < {alpha:g}",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color="#52514e",
                va="bottom")

    if handles:
        leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        frameon=False, fontsize=9, title="Concept  (hits/tested)",
                        title_fontsize=9, handletextpad=0.6, labelspacing=0.7)
        for text in leg.get_texts():           # ink, never the series colour
            text.set_color("#222222")
    fig.subplots_adjust(left=0.085, right=0.745, top=0.88, bottom=0.12)
    return fig


def cohort_context(pct_table, sample_id, top_n=18, figsize=(10, 7.5)):
    """
    Where one sample sits in the cohort, feature by feature.

    A raw feature value cannot be judged alone — nobody knows whether
    `baseline_depletion_index = 0.42` is normal. Drawn as a deviation from the
    cohort median, most extreme first, so "is this sample odd, and how?" is
    answered by the shape.
    """
    if pct_table is None or pct_table.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough samples for cohort context.",
                ha="center", va="center", color="gray")
        return fig

    # Features identical across the whole cohort sit at the midrank by
    # construction and say nothing about this sample; drop them if there is
    # anything more informative to show.
    informative = pct_table[pct_table["extremity"] > 1e-9]
    table = informative if len(informative) >= 3 else pct_table

    sub = table.head(top_n).iloc[::-1]
    y = np.arange(len(sub))
    dev = sub["percentile"].to_numpy() - 50.0

    fig, ax = plt.subplots(figsize=figsize)
    for band, shade in ((25, "#F4F4F2"), (10, "#E8E8E5")):
        ax.axvspan(-band, band, color=shade, zorder=0)
    ax.barh(y, dev, color=np.where(dev >= 0, "#2a78d6", "#d55e00"),
            height=0.7, zorder=3)
    ax.axvline(0, color="#52514e", lw=1.0, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=8)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    ax.set_xlim(-50, 50)
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.set_xticklabels(["0th", "25th", "median", "75th", "100th"])
    ax.set_xlabel("Percentile within the cohort")

    # Values live in their own column on the right, not at the bar tip — at the
    # tip they collided with the feature names whenever a bar ran left.
    for yi, (_, r) in zip(y, sub.iterrows()):
        ax.annotate(f"{r['value']:.3g}", xy=(1.015, yi), xycoords=("axes fraction", "data"),
                    fontsize=7.5, va="center", ha="left", color="#333333",
                    annotation_clip=False)
    ax.annotate("value", xy=(1.015, 1.01), xycoords="axes fraction", fontsize=8,
                color="#52514e", ha="left", va="bottom")

    ax.set_title(f"{sample_id} vs {int(sub['n_cohort'].max())} other samples",
                 loc="left", fontsize=12.5, fontweight="bold", pad=20)
    ax.annotate("shaded bands mark the middle 20% and 50% of the cohort",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9,
                color="#52514e", va="bottom")
    fig.subplots_adjust(left=0.31, right=0.87, top=0.90, bottom=0.09)
    return fig
