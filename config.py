"""
Central configuration for the Astana AI traffic-distribution prototype.

Everything that other modules need to agree on (road-type encodings, capacity
weights, district coordinates, congestion thresholds, file paths) lives here so
there is a single source of truth.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, "data")
GRAPH_CACHE_PATH = os.path.join(DATA_DIR, "astana_graph.graphml")
MODEL_CACHE_PATH = os.path.join(DATA_DIR, "congestion_model.joblib")

os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Road network download
# --------------------------------------------------------------------------- #
PLACE_NAME = "Astana, Kazakhstan"
NETWORK_TYPE = "drive"

# Network build mode:
#   "place" -> ox.graph_from_place(PLACE_NAME)            (full city, slower)
#   "point" -> ox.graph_from_point(ASTANA_CENTER, dist)  (central area, faster)
# The full city graph is large; "point" gives a snappier demo. Default is
# "place" to match the spec, switch to "point" if the dashboard feels heavy.
NETWORK_MODE = "place"
POINT_RADIUS_M = 9000  # used only when NETWORK_MODE == "point"

# Geographic centre of Astana (lat, lon).
ASTANA_CENTER = (51.1605, 71.4704)
DEFAULT_ZOOM = 12

# --------------------------------------------------------------------------- #
# Road-type model
# --------------------------------------------------------------------------- #
# Base capacity contribution of each OSM "highway" class (0..1). Higher means a
# road that can absorb more traffic before congesting.
ROAD_TYPE_CAPACITY = {
    "motorway": 1.00,
    "motorway_link": 0.80,
    "trunk": 0.90,
    "trunk_link": 0.70,
    "primary": 0.80,
    "primary_link": 0.62,
    "secondary": 0.62,
    "secondary_link": 0.52,
    "tertiary": 0.48,
    "tertiary_link": 0.42,
    "unclassified": 0.34,
    "road": 0.34,
    "residential": 0.26,
    "living_street": 0.16,
    "service": 0.16,
}
DEFAULT_ROAD_CAPACITY = 0.30

# Ordinal encoding for the ML model, from least to most important road.
# Higher index == bigger / faster road. Kept stable so a saved model and a
# freshly-built graph always agree on the encoding.
ROAD_TYPES_ORDERED = [
    "living_street",
    "service",
    "residential",
    "unclassified",
    "road",
    "tertiary_link",
    "tertiary",
    "secondary_link",
    "secondary",
    "primary_link",
    "primary",
    "trunk_link",
    "trunk",
    "motorway_link",
    "motorway",
]
ROAD_TYPE_ENCODING = {name: i for i, name in enumerate(ROAD_TYPES_ORDERED)}
MAX_ROAD_TYPE_CODE = len(ROAD_TYPES_ORDERED) - 1
# Fallback code for unknown road types (~ "unclassified").
DEFAULT_ROAD_TYPE_CODE = ROAD_TYPE_ENCODING["unclassified"]

# Typical number of lanes per road type when OSM does not specify it.
DEFAULT_LANES = {
    "motorway": 3,
    "motorway_link": 1,
    "trunk": 3,
    "trunk_link": 1,
    "primary": 2,
    "primary_link": 1,
    "secondary": 2,
    "secondary_link": 1,
    "tertiary": 2,
    "tertiary_link": 1,
    "unclassified": 1,
    "road": 1,
    "residential": 1,
    "living_street": 1,
    "service": 1,
}
DEFAULT_LANE_COUNT = 1

# Road types a heavy vehicle (truck) is allowed to use. Residential / service /
# living streets are excluded so trucks stay on the arterial network.
HEAVY_ALLOWED_ROAD_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
}
HEAVY_MIN_LANES = 2

# --------------------------------------------------------------------------- #
# Capacity score formula weights (must sum to ~1.0)
# --------------------------------------------------------------------------- #
W_ROAD_TYPE = 0.45
W_LANES = 0.35
W_SPEED = 0.20
LANES_NORM_CAP = 4.0      # lanes are normalised against this many
SPEED_NORM_CAP = 110.0    # km/h, speeds normalised against this
CAPACITY_MIN = 0.05
CAPACITY_MAX = 1.00

# Per-edge vehicle capacity (vehicles before the edge is "full"). Used to turn
# an assigned vehicle volume into a 0..1 load fraction.
EDGE_BASE_VEHICLE_CAPACITY = 60.0

# --------------------------------------------------------------------------- #
# Congestion levels (load fraction thresholds)
# --------------------------------------------------------------------------- #
LOAD_FREE_MAX = 0.40        # < 0.40  -> green
LOAD_MODERATE_MAX = 0.70    # 0.40-0.70 -> yellow, > 0.70 -> red

COLOR_FREE = "#2ecc71"
COLOR_MODERATE = "#f1c40f"
COLOR_CONGESTED = "#e74c3c"
COLOR_ROUTE = "#1f4fff"
COLOR_INCIDENT = "#8e44ad"

# --------------------------------------------------------------------------- #
# ML / simulation parameters
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "edge_capacity",
    "road_type_encoded",
    "time_of_day",
    "day_of_week",
    "current_load",
]

# Alert anomaly threshold: actual_load > predicted_load * ANOMALY_RATIO.
ANOMALY_RATIO = 1.40
# Ignore anomalies on edges whose predicted load is below this floor (avoids
# meaningless alerts on near-empty roads).
ANOMALY_MIN_PREDICTED = 0.12

# Cap how many *new* alerts a single detection cycle raises (keeps the operator
# panel focused on the most severe anomalies rather than every busy corridor).
MAX_NEW_ALERTS_PER_STEP = 8

# Default number of synthetic vehicle trips routed per simulation step.
DEFAULT_TRIPS_PER_STEP = 160
# How many edges receive an injected anomaly spike per step (demo-friendly so
# alerts actually appear).
ANOMALY_EDGES_PER_STEP = 4

# Map rendering: cap how many edges are drawn for responsiveness. The full
# graph is still used for the simulation; only the drawing is capped.
MAX_RENDER_EDGES = 1800

# --------------------------------------------------------------------------- #
# Vehicle categories
# --------------------------------------------------------------------------- #
VEHICLE_EMERGENCY = "Emergency"
VEHICLE_HEAVY = "Heavy"
VEHICLE_REGULAR = "Regular"
VEHICLE_TYPES = [VEHICLE_EMERGENCY, VEHICLE_HEAVY, VEHICLE_REGULAR]

# --------------------------------------------------------------------------- #
# Major Astana districts / landmarks (name -> (lat, lon)).
# Coordinates are approximate; they are snapped to the nearest graph node, so
# they only need to be in the right neighbourhood.
# --------------------------------------------------------------------------- #
DISTRICTS = {
    "Bayterek Tower (centre)": (51.1283, 71.4304),
    "Khan Shatyr": (51.1325, 71.4045),
    "Nur-Astana / Triumph": (51.1565, 71.4456),
    "Hazret Sultan Mosque": (51.1248, 71.4717),
    "Railway Station (Nurly Zhol)": (51.1893, 71.4093),
    "Astana Airport (NQZ)": (51.0222, 71.4669),
    "EXPO / Nur Alem": (51.0894, 71.4170),
    "Astana Arena": (51.1083, 71.4147),
    "Saryarka (right bank)": (51.1820, 71.4290),
    "Almaty district (east)": (51.1700, 71.5050),
}

# Days of the week for the UI (index 0 == Monday, matching datetime.weekday()).
DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
