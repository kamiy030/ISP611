import streamlit as st
import pydeck as pdk
import pandas as pd

pdk.settings.mapbox_api_key = st.secrets["mapbox"]["token"]

st.success(f"🔑 Using Mapbox token: {st.secrets['mapbox']['token'][:15]}...")

data = pd.DataFrame({
    "name": ["UiTM"],
    "lat": [3.0730],
    "lon": [101.5055]
})

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/streets-v12",
    initial_view_state=pdk.ViewState(
        latitude=3.0730,
        longitude=101.5055,
        zoom=16
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position='[lon, lat]',
            get_radius=20,
            get_fill_color=[255, 0, 0]
        ),
        pdk.Layer(
            "TextLayer",
            data=data,
            get_position='[lon, lat]',
            get_text='name',
            get_size=18,
            get_color=[0, 0, 0]
        )
    ]
))
