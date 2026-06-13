"""
Central configuration for the Astana AI traffic-distribution prototype.

Everything that other modules need to agree on (road-type encodings, capacity
weights, district coordinates, congestion thresholds, file paths) lives here so
there is a single source of truth.
"""

from __future__ import annotations

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, "data")
GRAPH_CACHE_PATH = os.path.join(DATA_DIR, "astana_graph.graphml")
MODEL_CACHE_PATH = os.path.join(DATA_DIR, "congestion_model.joblib")

os.makedirs(DATA_DIR, exist_ok=True)

PLACE_NAME = "Astana, Kazakhstan"
NETWORK_TYPE = "drive"

NETWORK_MODE = "place"
POINT_RADIUS_M = 9000

ASTANA_CENTER = (51.1605, 71.4704)
DEFAULT_ZOOM = 12

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
DEFAULT_ROAD_TYPE_CODE = ROAD_TYPE_ENCODING["unclassified"]

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

W_ROAD_TYPE = 0.45
W_LANES = 0.35
W_SPEED = 0.20
LANES_NORM_CAP = 4.0
SPEED_NORM_CAP = 110.0
CAPACITY_MIN = 0.05
CAPACITY_MAX = 1.00

EDGE_BASE_VEHICLE_CAPACITY = 60.0

LOAD_FREE_MAX = 0.40
LOAD_MODERATE_MAX = 0.70

COLOR_FREE = "#2ecc71"
COLOR_MODERATE = "#f1c40f"
COLOR_CONGESTED = "#e74c3c"
COLOR_ROUTE = "#1f4fff"
COLOR_INCIDENT = "#8e44ad"

FEATURE_COLUMNS = [
    "edge_capacity",
    "road_type_encoded",
    "time_of_day",
    "day_of_week",
    "current_load",
]

ANOMALY_RATIO = 1.40
ANOMALY_MIN_PREDICTED = 0.12

MAX_NEW_ALERTS_PER_STEP = 8

DEFAULT_TRIPS_PER_STEP = 160
ANOMALY_EDGES_PER_STEP = 4

MAX_RENDER_EDGES = 1800

VEHICLE_EMERGENCY = "Emergency"
VEHICLE_HEAVY = "Heavy"
VEHICLE_REGULAR = "Regular"
VEHICLE_TYPES = [VEHICLE_EMERGENCY, VEHICLE_HEAVY, VEHICLE_REGULAR]

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

DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
