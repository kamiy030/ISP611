import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("📍 Campus Navigation Optimizer using ACO")

# --- Upload CSV Files ---
uploaded_file = st.file_uploader("Upload Distance Matrix CSV (in km)", type=["csv"])
coord_file = st.file_uploader("Upload Coordinates CSV (Building, Latitude, Longitude)", type=["csv"])

if uploaded_file and coord_file:
    distance_matrix = pd.read_csv(uploaded_file, index_col=0, dtype=str)
    distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')  # Convert values to numeric, keep headers

    coordinates = pd.read_csv(coord_file)
    coordinates.columns = coordinates.columns.str.strip()
    coordinates["name"] = coordinates["name"].astype(str).str.strip()

    # Ensure names are strings and trimmed
    distance_matrix.index = distance_matrix.index.astype(str).str.strip()
    distance_matrix.columns = distance_matrix.columns.astype(str).str.strip()

    nodes = list(distance_matrix.index)
    n_nodes = len(nodes)

    st.success(f"✅ Loaded {n_nodes} locations.")

    start_node = st.selectbox("Start from", nodes, index=0)
    end_node = st.selectbox("End at", nodes, index=1)
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
                cost = sum(dist[path[i]][path[i+1]] for i in range(len(path)-1))
                all_paths.append(path)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            pheromone *= (1 - evaporation)
            for path, cost in zip(all_paths, all_costs):
                for i in range(len(path) - 1):
                    pheromone[path[i]][path[i + 1]] += 1.0 / cost

        best_named_path = [nodes[i] for i in best_path]
        st.success("✅ Best Route Found:")
        st.markdown(" → ".join(best_named_path))
        st.markdown(f"**Total Distance:** `{round(best_cost, 3)} km`")

        # --- MAP VISUALIZATION ---
        path_coords = [coords_dict[name] for name in best_named_path]

        line_data = pd.DataFrame({
            "from_lat": [path_coords[i][0] for i in range(len(path_coords)-1)],
            "from_lon": [path_coords[i][1] for i in range(len(path_coords)-1)],
            "to_lat": [path_coords[i+1][0] for i in range(len(path_coords)-1)],
            "to_lon": [path_coords[i+1][1] for i in range(len(path_coords)-1)],
        })

        marker_data = pd.DataFrame([
            {"lat": lat, "lon": lon, "name": name}
            for name, (lat, lon) in coords_dict.items()
        ])

        line_layer = pdk.Layer(
            "LineLayer",
            data=line_data,
            get_source_position='[from_lon, from_lat]',
            get_target_position='[to_lon, to_lat]',
            get_width=4,
            get_color=[255, 0, 0],
        )

        dot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=marker_data,
            get_position='[lon, lat]',
            get_radius=6,
            get_fill_color=[0, 0, 255],
        )

        text_layer = pdk.Layer(
            "TextLayer",
            data=marker_data,
            get_position='[lon, lat]',
            get_text='name',
            get_color=[0, 0, 0],
            get_size=16,
            get_alignment_baseline="'bottom'"
        )

        mid_lat, mid_lon = path_coords[0]
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            initial_view_state=pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=17,
                pitch=0
            ),
            layers=[line_layer, dot_layer, text_layer],
            tooltip={"text": "{name}"}
        ))
