import streamlit as st
import pandas as pd

st.title("Test: Dual File Uploaders")

uploaded_file = st.file_uploader("Upload Distance Matrix CSV (in km)", type=["csv"])
coord_file = st.file_uploader("Upload Coordinates CSV (Building, Latitude, Longitude)", type=["csv"])

if uploaded_file:
    st.success("Distance matrix file uploaded.")

if coord_file:
    st.success("Coordinates file uploaded.")

if uploaded_file and coord_file:
    st.success("Both files are uploaded. Ready to process!")
    distance_matrix = pd.read_csv(uploaded_file, index_col=0)
    coordinates = pd.read_csv(coord_file)
    st.write(distance_matrix.head())
    st.write(coordinates.head())
