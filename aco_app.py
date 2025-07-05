import streamlit as st
import pydeck as pdk
import pandas as pd

data = pd.DataFrame({
    "name": ["UiTM"],
    "lat": [3.0730],
    "lon": [101.5055]
})

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position='[lon, lat]',
    get_radius=20,
    get_fill_color=[255, 0, 0]
)

text_layer = pdk.Layer(
    "TextLayer",
    data=data,
    get_position='[lon, lat]',
    get_text='name',
    get_size=18,
    get_color=[0, 0, 0]
)

st.pydeck_chart(pdk.Deck(
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    initial_view_state=pdk.ViewState(
        latitude=3.0730,
        longitude=101.5055,
        zoom=16
    ),
    layers=[scatter_layer, text_layer]
))
