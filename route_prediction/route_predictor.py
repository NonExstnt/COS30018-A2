import random
random.seed(0)
import numpy as np
np.random.seed(0)
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import gru_model as gru
import itertools
from geopy import distance
from scipy import stats

class SCATSAnalyzer:
    def __init__(self, scats_data_file : str, k=7):
        scats_locations = scats_data_file.drop_duplicates("SCATS Number")[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]]
        self.locations_data = scats_locations[["NB_LONGITUDE", "NB_LATITUDE"]].values
        self.scats_numbers = scats_locations["SCATS Number"].values
        self.k = k
        self.nearest_neighbours = pd.DataFrame()

    def find_nearest_neighbours(self):
        knn = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree')
        knn.fit(self.locations_data)
        distances, indices = knn.kneighbors(self.locations_data)

        # Exclude the site itself
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        self.nearest_neighbours = pd.DataFrame({
            "SCATS Number": self.scats_numbers,
            "Longitude" : self.locations_data[:,0],
            "Latitude" : self.locations_data[:,1],
            "Nearest Neighbors": [self.scats_numbers[neighbor_indices] for neighbor_indices in indices],
            "Distances (km)": distances.tolist()
        })

        return self.nearest_neighbours

    def display_neighbors(self):
        for index, row in self.nearest_neighbours.iterrows():
            print(f"SCATS {row['SCATS Number']} has nearest neighbors:")
            for neighbor, dist in zip(row['Nearest Neighbors'], row['Distances (km)']):
                print(f"  - SCATS {neighbor}: {dist:.2f} km")

    def create_graph(self):
        G = nx.DiGraph()
        for scats_num in self.scats_numbers:
            G.add_node(scats_num)

        # Add edges based on nearest neighbors
        for i, scats_num in enumerate(self.scats_numbers):
            for neighbor_index, dist in zip(self.nearest_neighbours['Nearest Neighbors'][i], self.nearest_neighbours['Distances (km)'][i]):
                G.add_edge(scats_num, neighbor_index, weight=dist)

        # Visualize the graph
        self.visualize_graph(G)

        return G

    def visualize_graph(self, G):
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrowstyle='-|>', arrowsize=20)
        # nx.draw_networkx_labels(G, pos, labels={scats_num: str(scats_num) for scats_num in self.scats_numbers}, font_size=10)
        #
        # edge_labels = {(scats_num, neighbor): f"{G[scats_num][neighbor]['weight']:.2f} km"
        #                for scats_num in G.nodes for neighbor in G[scats_num]}
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.title("SCATS Nearest Neighbors Directed Graph")
        plt.axis('off')
        plt.show()

class RoutePredictor():
    def __init__(self, scats_data_file : str, scats_site_listing_file : str):
        # scats_locations = scats_data_file.drop_duplicates("SCATS Number")[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]]
        # self.locations_data = scats_locations[["NB_LONGITUDE", "NB_LATITUDE"]].values
        # self.scats_numbers = scats_locations["SCATS Number"].values
        self.scats_data = pd.read_csv(scats_data_file)
        self.scats_site_listing = pd.read_csv(scats_site_listing_file)
        self.scats_analyser = SCATSAnalyzer(self.scats_data)
        self.nearest_neighbours = self.scats_analyser.find_nearest_neighbours()

    # Source: https://blog.finxter.com/5-best-ways-to-round-time-to-the-next-15-minutes-in-python/
    def round_to_next_quarter_hour(self, dt):
        mins_to_add = (-(dt.minute % 15) + 15) % 15
        dt += timedelta(minutes=mins_to_add)
        dt = dt.replace(second=0, microsecond=0)
        return dt

    def get_time_distance(self,
                          distance_km : float,
                          origin_site_number : int,
                          destination_site_number : int,
                          time : datetime,
                          verbose : bool = False):
        a = gru.get_time_series_for_site(origin_site_number, self.scats_data)
        b = gru.get_time_series_for_site(destination_site_number, self.scats_data)

        # Get number of vehicles on site A and site B
        a_volume = a.loc[time].iloc[0]
        b_volume = b.loc[time].iloc[0]

        if verbose: print(f"Site A volume at time {time.strftime('%m/%d/%Y, %H:%M:%S')}: {a_volume}")
        if verbose: print(f"Site B volume at time {time.strftime('%m/%d/%Y, %H:%M:%S')}: {b_volume}")

        # Flow = Vehicles per hour for segment
        flow = (a_volume + b_volume) * 4

        if verbose: print(f"Flow: {flow}")

        density = (a_volume + b_volume) / distance_km

        if verbose: print(f"Density: {density}")

        speed = density / flow

        if verbose: print(f"Base speed: {speed}")

        # Impose speed limit of 60 km/h
        speed = min(speed, 60)

        if verbose: print(f"Capped speed: {speed}")

        time_distance = distance_km / speed

        if verbose: print(f"Base time distance: {time_distance}")

        site_type = self.scats_site_listing.loc[self.scats_site_listing["Site Number"] == origin_site_number].iloc[0]["Site Type"]
        is_intersection = site_type == "INT"

        if is_intersection: time_distance += 30

        if verbose: print(f"Final time distance: {time_distance}")

        return time_distance

    def get_all_time_distances(self, time : datetime):

        if time < datetime(2006, 10, 1) or time > datetime(2006, 10, 31):
            raise ValueError("Provided time must be within October, 2006")

        scats_numbers = list(self.scats_data["SCATS Number"].unique())

        scats_number_combinations = list(itertools.combinations(scats_numbers, 2))
        sites1 = []
        sites2 = []
        distances = []
        time_distances = []

        locations = self.scats_data.drop_duplicates("SCATS Number")
        locations = locations[(np.abs(stats.zscore(locations[["NB_LATITUDE", "NB_LONGITUDE"]])) < 3).all(axis=1)]

        # For each combination in the list:
        for combination in scats_number_combinations:
            site1, site2 = combination  # Unpack the tuple, assign each value to its own variable.

            try:
                site1_loc = locations[locations[
                                          "SCATS Number"] == site1]  # Get the row in the locations DataFrame where the site number is equal to the first site (970)
                site1_loc = (site1_loc["NB_LATITUDE"].to_numpy()[0], site1_loc["NB_LONGITUDE"].to_numpy()[
                    0])  # Extracted the longitude and latitude of this first site to two variables.

                site2_loc = locations[locations["SCATS Number"] == site2]  # Does the same thing with second site (2000)
                site2_loc = (site2_loc["NB_LATITUDE"].to_numpy()[0], site2_loc["NB_LONGITUDE"].to_numpy()[0])

                site_distance = distance.distance(site1_loc, site2_loc).km

                time_distance = self.get_time_distance(site_distance, origin_site_number=site1, destination_site_number=site2, time=time)

                sites1.append(site1); sites2.append(site2); distances.append(site_distance); time_distances.append(time_distance)

            except Exception as e:
                print(f"Error getting time distance for site combination {site1} {site2}\nError msg: {str(e)}")

        return pd.DataFrame({
            "Site1": sites1,
            "Site2": sites2,
            "Distance (km)": distances,
            "Time distance" : time_distances
        })

    def get_neighbours_with_time_distances(self, site_number, time: datetime):

        if time < datetime(2006, 10, 1) or time > datetime(2006, 10, 31):
            raise ValueError("Provided time must be within October, 2006")

        # Round the time to the closest 15-minute block.
        time = self.round_to_next_quarter_hour(time)

        site = self.nearest_neighbours.loc[self.nearest_neighbours['SCATS Number'] == site_number].iloc[0]

        neighbours, distances = site['Nearest Neighbors'], site['Distances (km)']

        site_volume = gru.get_time_series_for_site(site_number, self.scats_data).loc[time].iloc[0]

        print(f"Site {site_number}")
        print(f"Time {time.strftime('%m/%d/%Y, %H:%M:%S')}")
        print(f"Number of cars {site_volume}")
        print("Neighbours:")
        print()

        time_distances = []
        for neighbour, distance in zip(neighbours, distances):
            time_distance = self.get_time_distance(distance, site_number, neighbour, time)
            time_distances.append(time_distance)

        neighbours = pd.DataFrame({
            "Neighbour" : neighbours,
            "Distance" : distances,
            "Time" : time_distances
        })

        return neighbours