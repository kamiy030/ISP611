import streamlit as st
import pandas as pd
import numpy as np
import random
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("📍 Campus Navigation Optimizer using ACO and Real Map")

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader("Upload Distance Matrix CSV (in km)", type=["csv"])
coord_file = st.file_uploader("Upload Coordinates CSV (Building, Latitude, Longitude)", type=["csv"])

if uploaded_file and coord_file:
    distance_matrix = load_csv(uploaded_file)
    coordinates = load_csv(coord_file)

    # Clean whitespace in headers and index
    distance_matrix.columns = distance_matrix.columns.map(str).str.strip()
    distance_matrix.index = distance_matrix.index.map(str).str.strip()
    coordinates.columns = coordinates.columns.map(str).str.strip()
    coordinates["name"] = coordinates["name"].astype(str).str.strip()


    # Prepare node info
    nodes = list(distance_matrix.index)
    n_nodes = len(nodes)

    st.success(f"✅ Loaded {n_nodes} locations")

    # User inputs for ACO
    start_node = st.selectbox("Start Location", nodes, index=0)
    end_node = st.selectbox("End Location", nodes, index=1)
    n_ants = st.slider("Number of Ants", 5, 50, 10)
    n_iterations = st.slider("Number of Iterations", 10, 100, 50)
    alpha = st.slider("Alpha (pheromone importance)", 0.1, 5.0, 1.0)
    beta = st.slider("Beta (heuristic importance)", 0.1, 5.0, 2.0)
    evaporation = st.slider("Pheromone evaporation rate", 0.0, 1.0, 0.5)

    if st.button("🚀 Run ACO Optimization"):
        dist = distance_matrix.values
        pheromone = np.ones((n_nodes, n_nodes))
        best_cost = float("inf")
        best_path = []

        coordinates["name"] = coordinates["name"].astype(str).str.strip()
        coords_dict = dict(zip(coordinates["name"], zip(coordinates["lat"], coordinates["lon"])))


        def select_next_node(visited, current):
            probabilities = []
            for j in range(n_nodes):
                if j not in visited:
                    tau = pheromone[current][j] ** alpha
                    eta = (1.0 / dist[current][j]) ** beta if dist[current][j] > 0 else 0
                    probabilities.append(tau * eta)
                else:
                    probabilities.append(0)
            total = sum(probabilities)
            probabilities = [p / total if total > 0 else 0 for p in probabilities]
            return np.random.choice(range(n_nodes), p=probabilities)

        for _ in range(n_iterations):
            all_paths = []
            all_costs = []
            for _ in range(n_ants):
                start_index = nodes.index(start_node)
                end_index = nodes.index(end_node)
                path = [start_index]
                while path[-1] != end_index:
                    next_node = select_next_node(path, path[-1])
                    if next_node in path:
                        break
                    path.append(next_node)
                if path[-1] != end_index:
                    continue
                cost = sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))
                all_paths.append(path)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            # Update pheromones
            pheromone *= (1 - evaporation)
            for path, cost in zip(all_paths, all_costs):
                for i in range(len(path) - 1):
                    pheromone[path[i]][path[i + 1]] += 1.0 / cost

        best_named_path = [nodes[i] for i in best_path]
        st.success("✅ Best Route Found:")
        st.markdown(" → ".join(best_named_path))
        st.markdown(f"**Total Distance:** `{round(best_cost, 3)} km`")

        # Show error if no path is found
        if not best_named_path:
            st.error("❌ No valid path found from start to end. Try different nodes or increase iterations/ants.")
            st.stop()
        
       # --- Visualize path on map ---
        try:
            path_coords = [coords_dict[name] for name in best_named_path]
            start_lat, start_lon = path_coords[0]
        
            m = folium.Map(location=[start_lat, start_lon], zoom_start=17)
        
            # Add route markers
            folium.Marker(location=path_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(location=path_coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)
        
            # Draw polyline
            folium.PolyLine(path_coords, color="red", weight=5, tooltip="Optimized Path").add_to(m)
        
            # Add all building markers
            for name, (lat, lon) in coords_dict.items():
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=4,
                    color="blue",
                    fill=True,
                    fill_opacity=0.6,
                    popup=name
                ).add_to(m)
        
            # Display map
            st_data = st_folium(m, width=900, height=550)
        
        except Exception as e:
            st.error(f"⚠️ Error displaying map: {e}")
            st.stop()
