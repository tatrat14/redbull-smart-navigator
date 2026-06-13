"""
Road-graph construction for Astana.

Responsibilities:
  * Download the real Astana road network with OSMnx (cached to GraphML).
  * Parse messy OSM attributes (lanes / maxspeed / highway) robustly.
  * Attach clean numeric attributes to every edge:
        - capacity            (0..1 capacity score)
        - road_type           (string, primary OSM highway class)
        - road_type_encoded   (int ordinal, see config.ROAD_TYPE_ENCODING)
        - lanes_num           (float)
        - speed_kph           (float)
        - length_m            (float)
        - free_flow_time      (float, seconds at free-flow speed)
        - load / predicted_load / congestion_prob  (runtime state, init 0)
  * Provide a synthetic fallback grid so the rest of the pipeline can be
    demonstrated even with no internet / no OSMnx.

The module is import-safe: OSMnx is only imported when actually building from
OSM, so the synthetic path works in minimal environments.
"""

from __future__ import annotations

import ast
import difflib
import math
import os
import re
from typing import Optional

import networkx as nx

import config


def _first(value):
    """OSM attributes are often lists; collapse to a single representative."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and parsed:
                return parsed[0]
        except (ValueError, SyntaxError):
            return value
    return value


def parse_lanes(value, road_type: str) -> float:
    """Return a lane count, falling back to a per-road-type default."""
    raw = _first(value)
    if raw is not None:
        try:
            lanes = float(str(raw).strip())
            if lanes >= 1:
                return lanes
        except (ValueError, TypeError):
            pass
    return float(config.DEFAULT_LANES.get(road_type, config.DEFAULT_LANE_COUNT))


def parse_maxspeed(value) -> Optional[float]:
    """Parse an OSM maxspeed tag into km/h. Returns None if unknown."""
    raw = _first(value)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    try:
        if "mph" in text:
            num = float(text.replace("mph", "").strip())
            return num * 1.60934
        num = float(text.replace("km/h", "").replace("kph", "").strip())
        return num
    except (ValueError, TypeError):
        return None


def normalise_road_type(value) -> str:
    """Collapse the OSM highway tag to one of our known road-type keys."""
    raw = _first(value)
    if raw is None:
        return "unclassified"
    text = str(raw).strip().lower()
    if text in config.ROAD_TYPE_CAPACITY:
        return text
    return "unclassified"


def compute_capacity(road_type: str, lanes: float, speed_kph: float) -> float:
    """Weighted capacity score in [CAPACITY_MIN, CAPACITY_MAX]."""
    base = config.ROAD_TYPE_CAPACITY.get(road_type, config.DEFAULT_ROAD_CAPACITY)
    lanes_norm = min(lanes, config.LANES_NORM_CAP) / config.LANES_NORM_CAP
    speed_norm = min(speed_kph, config.SPEED_NORM_CAP) / config.SPEED_NORM_CAP
    score = (
        config.W_ROAD_TYPE * base
        + config.W_LANES * lanes_norm
        + config.W_SPEED * speed_norm
    )
    return float(min(config.CAPACITY_MAX, max(config.CAPACITY_MIN, score)))


def annotate_edges(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Attach clean numeric attributes to every edge. Idempotent: safe to call on
    a freshly downloaded graph or one reloaded from GraphML.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        road_type = normalise_road_type(data.get("highway"))
        lanes = parse_lanes(data.get("lanes"), road_type)

        speed = data.get("speed_kph")
        try:
            speed = float(speed) if speed is not None else None
        except (ValueError, TypeError):
            speed = None
        if speed is None or speed <= 0:
            speed = parse_maxspeed(data.get("maxspeed"))
        if speed is None or speed <= 0:
            speed = {
                "motorway": 100,
                "motorway_link": 60,
                "trunk": 90,
                "trunk_link": 50,
                "primary": 70,
                "primary_link": 45,
                "secondary": 60,
                "secondary_link": 40,
                "tertiary": 50,
                "tertiary_link": 35,
                "residential": 30,
                "living_street": 15,
                "service": 20,
            }.get(road_type, 40)

        length = data.get("length")
        try:
            length = float(length)
        except (ValueError, TypeError):
            length = _haversine_m(
                G.nodes[u]["y"], G.nodes[u]["x"],
                G.nodes[v]["y"], G.nodes[v]["x"],
            )

        capacity = compute_capacity(road_type, lanes, speed)
        free_flow_time = length / max(speed * 1000.0 / 3600.0, 1.0)

        data["road_type"] = road_type
        data["road_type_encoded"] = config.ROAD_TYPE_ENCODING.get(
            road_type, config.DEFAULT_ROAD_TYPE_CODE
        )
        data["lanes_num"] = float(lanes)
        data["speed_kph"] = float(speed)
        data["length_m"] = float(length)
        data["capacity"] = float(capacity)
        data["base_capacity"] = float(capacity)
        data["free_flow_time"] = float(free_flow_time)
        data["veh_capacity"] = float(
            config.EDGE_BASE_VEHICLE_CAPACITY * lanes * (0.4 + 0.6 * capacity)
        )
        data.setdefault("load", 0.0)
        data.setdefault("predicted_load", 0.0)
        data.setdefault("congestion_prob", 0.0)
        data["incident"] = None
        data["incident_factor"] = 1.0
    return G


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two (lat, lon) points, in kilometres."""
    return _haversine_m(lat1, lon1, lat2, lon2) / 1000.0


_EDGE_DTYPES = {
    "capacity": float,
    "base_capacity": float,
    "road_type_encoded": int,
    "lanes_num": float,
    "speed_kph": float,
    "length_m": float,
    "free_flow_time": float,
    "veh_capacity": float,
    "load": float,
    "predicted_load": float,
    "congestion_prob": float,
    "length": float,
}


def _download_graph():
    import osmnx as ox

    if config.NETWORK_MODE == "point":
        G = ox.graph_from_point(
            config.ASTANA_CENTER,
            dist=config.POINT_RADIUS_M,
            network_type=config.NETWORK_TYPE,
        )
    else:
        G = ox.graph_from_place(config.PLACE_NAME, network_type=config.NETWORK_TYPE)

    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception:
        pass

    try:
        G = ox.truncate.largest_component(G, strongly=True)
    except Exception:
        largest = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    return G


def load_or_build_graph(
    cache_path: str = config.GRAPH_CACHE_PATH,
    allow_synthetic_fallback: bool = True,
    force_rebuild: bool = False,
):
    """
    Return an annotated MultiDiGraph for Astana.

    Order of preference:
      1. Cached GraphML on disk (fast, offline).
      2. Fresh OSMnx download (needs internet; cached for next time).
      3. Synthetic grid fallback (only if 1 & 2 fail and allowed).

    Returns (graph, source) where source is one of
    {"cache", "osm", "synthetic"}.
    """
    if not force_rebuild and os.path.exists(cache_path):
        try:
            import osmnx as ox

            G = ox.load_graphml(cache_path, edge_dtypes=_EDGE_DTYPES)
            G = annotate_edges(G)
            return G, "cache"
        except Exception as exc:
            print(f"[graph_builder] Could not load cache ({exc}); rebuilding.")

    try:
        import osmnx as ox

        G = _download_graph()
        G = annotate_edges(G)
        try:
            ox.save_graphml(G.copy(), cache_path)
        except Exception as exc:
            print(f"[graph_builder] Warning: failed to cache graph ({exc}).")
        return G, "osm"
    except Exception as exc:
        print(f"[graph_builder] OSM download unavailable ({exc}).")
        if not allow_synthetic_fallback:
            raise

    print("[graph_builder] Falling back to synthetic grid graph.")
    G = build_synthetic_graph()
    return G, "synthetic"


def build_synthetic_graph(rows: int = 22, cols: int = 22) -> nx.MultiDiGraph:
    """
    Build a grid road network roughly covering Astana so the full pipeline can
    run with no internet. Nodes get real-ish lat/lon around ASTANA_CENTER.
    """
    G = nx.MultiDiGraph()
    G.graph["crs"] = "epsg:4326"

    lat0, lon0 = config.ASTANA_CENTER
    lat_span, lon_span = 0.13, 0.20
    lat_step = lat_span / (rows - 1)
    lon_step = lon_span / (cols - 1)

    def node_id(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            lat = lat0 - lat_span / 2 + r * lat_step
            lon = lon0 - lon_span / 2 + c * lon_step
            G.add_node(node_id(r, c), x=lon, y=lat)

    def classify(r, c, horizontal):
        idx = r if horizontal else c
        if idx % 7 == 0:
            return "primary", 90
        if idx % 3 == 0:
            return "secondary", 60
        return "residential", 30

    def add_edge(a, b, road_type, speed):
        lat_a, lon_a = G.nodes[a]["y"], G.nodes[a]["x"]
        lat_b, lon_b = G.nodes[b]["y"], G.nodes[b]["x"]
        length = _haversine_m(lat_a, lon_a, lat_b, lon_b)
        lanes = config.DEFAULT_LANES.get(road_type, 1)
        for s, t in ((a, b), (b, a)):
            G.add_edge(
                s, t, 0,
                highway=road_type,
                lanes=lanes,
                maxspeed=speed,
                length=length,
                speed_kph=speed,
            )

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                rt, sp = classify(r, c, horizontal=True)
                add_edge(node_id(r, c), node_id(r, c + 1), rt, sp)
            if r + 1 < rows:
                rt, sp = classify(r, c, horizontal=False)
                add_edge(node_id(r, c), node_id(r + 1, c), rt, sp)

    return annotate_edges(G)


def nearest_node(G, lat: float, lon: float):
    """Nearest graph node to a (lat, lon) point. Uses OSMnx if available."""
    try:
        import osmnx as ox

        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except Exception:
        best_node, best_d = None, float("inf")
        for n, data in G.nodes(data=True):
            d = (data["y"] - lat) ** 2 + (data["x"] - lon) ** 2
            if d < best_d:
                best_d, best_node = d, n
        return best_node


def district_nodes(G):
    """Map each configured district name to its nearest graph node id."""
    return {
        name: nearest_node(G, lat, lon)
        for name, (lat, lon) in config.DISTRICTS.items()
    }


def graph_summary(G) -> dict:
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }


_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ә": "a", "ғ": "g", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
    "һ": "h", "і": "i",
}
_ROAD_WORDS = [
    "көшесі", "даңғылы", "даңғыл", "проспект", "переулок", "микрорайон",
    "шоссе", "проезд", "тұйығы", "алаңы", "улица", "мкр", "пр-т", "пр.",
    "ул.", "street", "avenue", "road", " ave", " st",
]


def _translit(s: str) -> str:
    return "".join(_CYR2LAT.get(ch, ch) for ch in s)


def normalize_street(s: str) -> str:
    """Lowercase, strip road-type words, transliterate to Latin, drop
    punctuation — so spelling/transliteration/script differences collapse."""
    s = str(s).lower().strip()
    for w in _ROAD_WORDS:
        s = s.replace(w, " ")
    s = _translit(s)
    s = re.sub(r"[^0-9a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_street_index(G) -> dict:
    """Index every named street to a representative graph node + centroid, so we
    can route to a (possibly mistyped) street name fully offline."""
    name_to_nodes: dict = {}
    name_to_sum: dict = {}
    for u, v, _, data in G.edges(keys=True, data=True):
        nm = data.get("name")
        if not nm:
            continue
        names = nm if isinstance(nm, (list, tuple)) else [nm]
        for name in names:
            name = str(name).strip()
            if not name:
                continue
            name_to_nodes.setdefault(name, set()).add(u)
            acc = name_to_sum.setdefault(name, [0.0, 0.0, 0])
            acc[0] += G.nodes[u]["y"]
            acc[1] += G.nodes[u]["x"]
            acc[2] += 1

    name_to_node: dict = {}
    name_to_centroid: dict = {}
    name_to_ordered: dict = {}
    for name, nodes in name_to_nodes.items():
        acc = name_to_sum[name]
        clat, clon = acc[0] / acc[2], acc[1] / acc[2]
        name_to_centroid[name] = (clat, clon)
        pts = [(n, G.nodes[n]["x"], G.nodes[n]["y"]) for n in nodes]
        best, best_d = None, float("inf")
        for n, x, y in pts:
            d = (y - clat) ** 2 + (x - clon) ** 2
            if d < best_d:
                best_d, best = d, n
        name_to_node[name] = best
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        if (max(xs) - min(xs)) >= (max(ys) - min(ys)):
            pts.sort(key=lambda p: p[1])
        else:
            pts.sort(key=lambda p: p[2])
        name_to_ordered[name] = [p[0] for p in pts]

    norm_to_names: dict = {}
    for name in name_to_nodes:
        norm_to_names.setdefault(normalize_street(name), []).append(name)

    return {
        "names": sorted(name_to_nodes.keys()),
        "name_to_node": name_to_node,
        "name_to_centroid": name_to_centroid,
        "name_to_ordered": name_to_ordered,
        "norm_to_names": norm_to_names,
    }


HOUSE_NUMBER_CAP = 150.0


def parse_address(text: str):
    """Split a typed address into (street_text, house_number_or_None).

    "Туран 26" -> ("Туран", 26);  "дом 5 Абая" -> ("Абая", 5);  "Туран" -> (..,
    None). A number that is part of the street name (e.g. "150 лет Абая") is
    only stripped when it is clearly a trailing house number.
    """
    text = str(text).strip()
    num = None
    street = text
    m = re.search(r"(?:дом|д\.?|house|#|кв|оф)\s*(\d{1,4})", text, re.I)
    if not m:
        m = re.search(r"(\d{1,4})\s*$", text)
    if m:
        num = int(m.group(1))
        street = (text[: m.start()] + text[m.end():]).strip(" ,.-")
    return street, num


def street_point(index: dict, name: str, house_number: Optional[int] = None):
    """Return a node id on ``name``. With a house number, interpolate a distinct
    point along the street instead of always returning its centroid."""
    if house_number is None:
        return index["name_to_node"][name]
    ordered = index.get("name_to_ordered", {}).get(name)
    if not ordered:
        return index["name_to_node"][name]
    frac = max(0.0, min(1.0, (house_number - 1) / HOUSE_NUMBER_CAP))
    i = int(round(frac * (len(ordered) - 1)))
    return ordered[i]


def match_street(query: str, index: dict, limit: int = 5):
    """Return up to ``limit`` (street_name, score 0..1) best matches for a
    possibly-misspelled query, best first."""
    qn = normalize_street(query)
    if not qn:
        return []
    norm_to_names = index["norm_to_names"]
    norm_keys = list(norm_to_names.keys())

    scored: dict = {}

    def _consider(key, base=0.0):
        score = difflib.SequenceMatcher(None, qn, key).ratio()
        qt, kt = set(qn.split()), set(key.split())
        if qt and kt:
            score = 0.7 * score + 0.3 * (len(qt & kt) / len(qt | kt))
        score = max(score, base)
        for nm in norm_to_names[key]:
            if score > scored.get(nm, 0.0):
                scored[nm] = score

    if qn in norm_to_names:
        _consider(qn, base=1.0)
    for key in difflib.get_close_matches(qn, norm_keys, n=limit * 3, cutoff=0.5):
        _consider(key)
    for key in norm_keys:
        if qn in key or key in qn:
            _consider(key, base=0.82)

    return sorted(scored.items(), key=lambda x: x[1], reverse=True)[:limit]
