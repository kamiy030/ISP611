import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# --- ACO PARAMETERS ---
st.title("📍 Campus Navigation Optimizer using ACO")

# Upload CSV files
uploaded_file = st.file_uploader("Upload Distance Matrix CSV (in km)", type=["csv"])
coord_file = st.file_uploader("Upload Coordinates CSV (Building, Latitude, Longitude)", type=["csv"])

if uploaded_file and coord_file:
    distance_matrix = pd.read_csv(uploaded_file, index_col=0)
    coordinates = pd.read_csv(coord_file)

    nodes = list(distance_matrix.index)
    n_nodes = len(nodes)

    st.success(f"Loaded distance matrix with {n_nodes} buildings.")

    # User input for ACO
    start_node = st.selectbox("Start from", nodes, index=0)
    end_node = st.selectbox("End at", nodes, index=1)
    n_ants = st.slider("Number of Ants", 5, 50, 10)
    n_iterations = st.slider("Number of Iterations", 10, 100, 50)
    alpha = st.slider("Alpha (pheromone importance)", 0.1, 5.0, 1.0)
    beta = st.slider("Beta (heuristic importance)", 0.1, 5.0, 2.0)
    evaporation = st.slider("Pheromone evaporation rate", 0.0, 1.0, 0.5)

    if st.button("Run ACO Optimization"):
        dist = distance_matrix.values
        pheromone = np.ones((n_nodes, n_nodes))
        best_cost = float("inf")
        best_path = []

        # Coordinate lookup dictionary
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

        for iteration in range(n_iterations):
            all_paths = []
            all_costs = []
            for ant in range(n_ants):
                start_index = nodes.index(start_node)
                end_index = nodes.index(end_node)
                path = [start_index]
                while path[-1] != end_index:
                    next_node = select_next_node(path, path[-1])
                    if next_node in path:  # avoid loops
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

            # Update pheromone
            pheromone *= (1 - evaporation)
            for path, cost in zip(all_paths, all_costs):
                for i in range(len(path) - 1):
                    pheromone[path[i]][path[i + 1]] += 1.0 / cost

        best_named_path = [nodes[i] for i in best_path]
        st.success("✅ Best Path Found:")
        st.write(" → ".join(best_named_path))
        st.write(f"Total Distance: {round(best_cost, 3)} km")

        # Extract coordinates for best path
        path_coords = [coords_dict[name] for name in best_named_path]

        # Calculate mid-point for view
        mid_lat = np.mean([lat for lat, lon in path_coords])
        mid_lon = np.mean([lon for lat, lon in path_coords])
        
        # Route Line Layer (with arrows)
        line_layer = pdk.Layer(
            "PathLayer",
            data=[{
                "path": [(lon, lat) for lat, lon in path_coords],
                "name": "ACO Path"
            }],
            get_path="path",
            get_width=5,
            get_color=[255, 0, 0],
            width_min_pixels=4,
            width_scale=1,
            pickable=True
        )
        
        # Marker Layer for Start and End
        marker_data = pd.DataFrame([
            {"lat": path_coords[0][0], "lon": path_coords[0][1], "label": "Start"},
            {"lat": path_coords[-1][0], "lon": path_coords[-1][1], "label": "End"},
        ])
        
        marker_layer = pdk.Layer(
            "TextLayer",
            data=marker_data,
            pickable=True,
            get_position='[lon, lat]',
            get_text='label',
            get_color=[0, 255, 0],
            get_size=16,
            get_alignment_baseline='"bottom"'
        )
        
        # Display updated map
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/streets-v12',  # 🎯 More map-like style
            initial_view_state=pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=17,
                pitch=45,
                bearing=0
            ),
            layers=[line_layer, marker_layer],
            tooltip={"text": "{label}"}
        ))
