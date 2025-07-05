import pandas as pd
import pydeck as pdk
import streamlit as st

# Already verified this loads
pdk.settings.mapbox_api_key = st.secrets["mapbox"]["token"]

test_data = pd.DataFrame({
    "name": ["Test Point"],
    "lat": [3.0730],
    "lon": [101.5055]
})

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/streets-v12",
    initial_view_state=pdk.ViewState(
        latitude=3.0730,
        longitude=101.5055,
        zoom=17
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=test_data,
            get_position='[lon, lat]',
            get_radius=10,
            get_fill_color=[255, 0, 0]
        ),
        pdk.Layer(
            "TextLayer",
            data=test_data,
            get_position='[lon, lat]',
            get_text='name',
            get_size=18,
            get_color=[0, 0, 0]
        )
    ]
))
