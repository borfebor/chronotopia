import streamlit as st
import pandas as pd
from PIL import Image

image = Image.open('logo.png')

with st.container(border=True, horizontal_alignment='center'):
        st.image(image)
st.markdown("""
<style>
.hero {
    max-width: 900px;
    margin: auto;
}

.step {
    display: flex;
    align-items: flex-start;
    gap: 25px;
    margin: 35px 0;
}

.step svg {
    display: block; /* remove inline whitespace */
    margin-top: 0;  /* remove any default top margin */
}

.step-text h3 {
    margin: 0;
}

.step-text p {
    margin-top: 6px;
    opacity: 0.9;
}
</style>
<div class="hero">
    <p style="text-align:center;">
        A flexible toolkit for time-series and rhythmicity analysis.
    </p>
    <div class="step">
        <div class="step-icon">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="size-6" width="100" height="100">
        <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 7.5h-.75A2.25 2.25 0 0 0 4.5 9.75v7.5a2.25 2.25 0 0 0 2.25 2.25h7.5a2.25 2.25 0 0 0 2.25-2.25v-7.5a2.25 2.25 0 0 0-2.25-2.25h-.75m0-3-3-3m0 0-3 3m3-3v11.25m6-2.25h.75a2.25 2.25 0 0 1 2.25 2.25v7.5a2.25 2.25 0 0 1-2.25 2.25h-7.5a2.25 2.25 0 0 1-2.25-2.25v-.75" />
        </svg>
        </div>
        <div class="step-text">
            <h3 style="text-align:left;">1. Upload your time series</h3>
            <p style="text-align:left;">
                Upload your data in tabular format, including a time column expressed in seconds,
                minutes, hours, or days.
            </p>
        </div>
    </div>
    <div class="step">
        <div class="step-icon">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="size-6" width="100" height="100">
        <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0 0 20.25 18V6A2.25 2.25 0 0 0 18 3.75H6A2.25 2.25 0 0 0 3.75 6v12A2.25 2.25 0 0 0 6 20.25Z" />
        </svg>
        </div>
        <div class="step-text">
            <h3 style="text-align:left;">2. Explore and analyse the data</h3>
            <p style="text-align:left;">
                Perform quick and intuitive time-series analysis.
            </p>
        </div>
    </div>
    <div class="step">
        <div class="step-icon">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="size-6" width="100" height="100">
        <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
        </svg>
        </div>
        <div class="step-text">
            <h3 style="text-align:left;">3. Save the formatted report</h3>
            <p style="text-align:left;">
                Generate and export a structured report to support data interpretation.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    #st.space('small')
with st.container(border=True):
        st.markdown("""
    #### Data grouping
    Group samples by experimental conditions to facilitate interpretation and visualization.

    #### Data filtering
    Filter data by time range, samples, or experimental conditions.

    #### Data normalization & trend correction
    Apply linear, cubic, rolling-mean, or amplitude-adjusted mean detrending methods.

    #### Data visualization
    Create customizable actograms and line plots with T-cycle visualization and SVG export.

    #### Period analysis
    Estimate periods using multiple methods, including periodogram, wavelet transform, Fourier transform, and cosinor analysis.

    #### Rhythmicity analysis
    Assess rhythmicity using MetaCycle (meta2d), JTK, ARS, and Lomb-Scargle (LS).

    """)
st.stop()