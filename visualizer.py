"""
Folium map rendering for the Astana traffic graph.

Draws the road network colour-coded by congestion (green / yellow / red),
overlays active-alert markers and (optionally) a highlighted vehicle route, and
marks the major districts. To stay responsive on the full city graph, only the
most important roads are drawn (capped by ``max_edges``); the simulation always
uses the complete graph.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import folium

import config

EdgeKey = Tuple[int, int, int]


def load_color(load: float) -> str:
    if load > config.LOAD_MODERATE_MAX:
        return config.COLOR_CONGESTED
    if load > config.LOAD_FREE_MAX:
        return config.COLOR_MODERATE
    return config.COLOR_FREE


def _edge_coords(G, u, v, data) -> List[Tuple[float, float]]:
    """Return [(lat, lon), ...] for an edge, using OSM geometry if present."""
    geom = data.get("geometry")
    if geom is not None and hasattr(geom, "coords"):
        try:
            return [(lat, lon) for lon, lat in geom.coords]
        except Exception:
            pass
    return [
        (G.nodes[u]["y"], G.nodes[u]["x"]),
        (G.nodes[v]["y"], G.nodes[v]["x"]),
    ]


def _line_weight(data: dict) -> float:
    """Thicker lines for more important roads."""
    code = data.get("road_type_encoded", config.DEFAULT_ROAD_TYPE_CODE)
    return 1.5 + 4.0 * (code / float(config.MAX_ROAD_TYPE_CODE))


def _select_render_edges(G, max_edges: Optional[int], must_include: set):
    """
    Choose which edges to draw. Always include alert / route edges, then fill up
    to ``max_edges`` with the most important (highest road-type) edges.
    """
    edges = list(G.edges(keys=True, data=True))
    if max_edges is None or len(edges) <= max_edges:
        return edges

    forced = [e for e in edges if (e[0], e[1], e[2]) in must_include]
    rest = [e for e in edges if (e[0], e[1], e[2]) not in must_include]
    rest.sort(
        key=lambda e: e[3].get("road_type_encoded", 0)
        + e[3].get("load", 0.0),  # nudge busy roads into view
        reverse=True,
    )
    budget = max(0, max_edges - len(forced))
    return forced + rest[:budget]


def build_map(
    G,
    alerts: Optional[List] = None,
    route: Optional[object] = None,
    district_node_ids: Optional[Dict[str, int]] = None,
    max_edges: Optional[int] = config.MAX_RENDER_EDGES,
    center: Tuple[float, float] = config.ASTANA_CENTER,
    zoom: int = config.DEFAULT_ZOOM,
    user_location: Optional[Tuple[float, float]] = None,
    user_radius_km: Optional[float] = None,
) -> folium.Map:
    alerts = alerts or []
    route_edges = set(route.edges) if (route and route.found) else set()
    alert_edges = {a.edge for a in alerts if a.status == "active"}
    must_include = route_edges | alert_edges

    fmap = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles="cartodbpositron",
        control_scale=True,
    )

    roads_layer = folium.FeatureGroup(name="Roads (congestion)", show=True)
    edges = _select_render_edges(G, max_edges, must_include)
    for u, v, k, data in edges:
        coords = _edge_coords(G, u, v, data)
        load = data.get("load", 0.0)
        if data.get("incident"):
            color = config.COLOR_INCIDENT
        else:
            color = load_color(load)
        tooltip = (
            f"{data.get('road_type', 'road')} | "
            f"load {load * 100:.0f}% | "
            f"cap {data.get('capacity', 0):.2f} | "
            f"P(cong) {data.get('congestion_prob', 0):.2f}"
        )
        folium.PolyLine(
            coords,
            color=color,
            weight=_line_weight(data),
            opacity=0.75,
            tooltip=tooltip,
        ).add_to(roads_layer)
    roads_layer.add_to(fmap)

    # --- Highlighted route ------------------------------------------------ #
    if route and getattr(route, "found", False):
        route_layer = folium.FeatureGroup(name="Vehicle route", show=True)
        for u, v, k in route.edges:
            data = G.edges[u, v, k]
            coords = _edge_coords(G, u, v, data)
            folium.PolyLine(
                coords,
                color=config.COLOR_ROUTE,
                weight=7,
                opacity=0.9,
            ).add_to(route_layer)
        if route.nodes:
            o, d = route.nodes[0], route.nodes[-1]
            folium.Marker(
                [G.nodes[o]["y"], G.nodes[o]["x"]],
                tooltip="Origin",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(route_layer)
            folium.Marker(
                [G.nodes[d]["y"], G.nodes[d]["x"]],
                tooltip="Destination",
                icon=folium.Icon(color="blue", icon="flag-checkered", prefix="fa"),
            ).add_to(route_layer)
        route_layer.add_to(fmap)

    # --- District markers ------------------------------------------------- #
    if district_node_ids:
        dlayer = folium.FeatureGroup(name="Districts", show=True)
        for name, node_id in district_node_ids.items():
            if node_id in G.nodes:
                folium.CircleMarker(
                    [G.nodes[node_id]["y"], G.nodes[node_id]["x"]],
                    radius=4,
                    color="#34495e",
                    fill=True,
                    fill_opacity=0.9,
                    tooltip=name,
                ).add_to(dlayer)
        dlayer.add_to(fmap)

    # --- Active alert markers --------------------------------------------- #
    active = [a for a in alerts if a.status == "active"]
    if active:
        alayer = folium.FeatureGroup(name="Active alerts", show=True)
        for a in active:
            popup = folium.Popup(
                html=(
                    f"<b>Alert #{a.id}</b><br>"
                    f"{a.road_name} ({a.road_type})<br>"
                    f"actual load: {a.actual_load * 100:.0f}%<br>"
                    f"predicted: {a.predicted_load * 100:.0f}%<br>"
                    f"ratio: {a.ratio:.2f}x<br>"
                    f"<i>Respond in the dashboard panel.</i>"
                ),
                max_width=250,
            )
            folium.Marker(
                [a.lat, a.lon],
                popup=popup,
                tooltip=f"Alert #{a.id}: {a.road_name}",
                icon=folium.Icon(
                    color="red", icon="exclamation-triangle", prefix="fa"
                ),
            ).add_to(alayer)
        alayer.add_to(fmap)

    # --- "You are here" + visibility radius -------------------------------- #
    if user_location is not None:
        ulat, ulon = user_location
        ulayer = folium.FeatureGroup(name="You", show=True)
        if user_radius_km:
            folium.Circle(
                location=[ulat, ulon],
                radius=user_radius_km * 1000.0,
                color="#2c7fb8",
                weight=2,
                fill=True,
                fill_color="#2c7fb8",
                fill_opacity=0.08,
                tooltip=f"Your visibility radius: {user_radius_km:.1f} km",
            ).add_to(ulayer)
        folium.Marker(
            [ulat, ulon],
            tooltip="You are here",
            icon=folium.Icon(color="cadetblue", icon="user", prefix="fa"),
        ).add_to(ulayer)
        ulayer.add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    _add_legend(fmap)
    return fmap


def _add_legend(fmap: folium.Map):
    legend_html = f"""
    <div style="
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: white; padding: 10px 12px; border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 12px; line-height: 1.6;">
      <b>Congestion</b><br>
      <span style="color:{config.COLOR_FREE}">&#9632;</span> Free (&lt;40%)<br>
      <span style="color:{config.COLOR_MODERATE}">&#9632;</span> Moderate (40-70%)<br>
      <span style="color:{config.COLOR_CONGESTED}">&#9632;</span> Congested (&gt;70%)<br>
      <span style="color:{config.COLOR_INCIDENT}">&#9632;</span> Incident<br>
      <span style="color:{config.COLOR_ROUTE}">&#9632;</span> Vehicle route
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
