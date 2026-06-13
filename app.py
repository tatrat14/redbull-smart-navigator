"""
Astana AI Traffic Distribution — Streamlit dashboard.

Run with:
    streamlit run app.py

The complete prototype flow:
    graph construction -> ML prediction -> traffic distribution
    -> anomaly alert -> map visualization
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st

import config
import demand
import graph_builder
import ml_model
from alert_system import AlertSystem, RESPONSE_OPTIONS
from traffic_simulator import TrafficSimulator

EVENT_DISTRICTS = [
    "Bayterek Tower (centre)",
    "Khan Shatyr",
    "EXPO / Nur Alem",
    "Nur-Astana / Triumph",
]


def astana_now() -> datetime:
    """Current local time in Astana (UTC+5). Falls back to a fixed offset if
    the IANA tz database is unavailable."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Almaty"))
    except Exception:
        return datetime.now(timezone.utc) + timedelta(hours=5)


@st.cache_data(show_spinner=False)
def geocode_astana(address: str):
    """Geocode a free-text Astana address to (lat, lon). Returns None on
    failure (e.g. offline or address not found)."""
    try:
        import osmnx as ox

        return tuple(ox.geocode(f"{address}, Astana, Kazakhstan"))
    except Exception:
        return None


def route_streets(G, route):
    """Ordered list of distinct street names along a route (pseudo turn-by-turn)."""
    seq = []
    for u, v, k in getattr(route, "edges", []):
        nm = G.edges[u, v, k].get("name")
        if isinstance(nm, (list, tuple)):
            nm = nm[0] if nm else None
        nm = str(nm) if nm else None
        if nm and (not seq or seq[-1] != nm):
            seq.append(nm)
    return seq

try:
    from streamlit_folium import st_folium
except Exception as exc:
    st.error(
        "`streamlit-folium` is required. Install dependencies with "
        "`pip install -r requirements.txt`.\n\n"
        f"Import error: {exc}"
    )
    st.stop()

import plotly.graph_objects as go

import visualizer

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


st.set_page_config(
    page_title="Astana AI Traffic Distribution",
    page_icon="🚦",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_graph():
    return graph_builder.load_or_build_graph()


@st.cache_resource(show_spinner=False)
def get_model():
    return ml_model.load_or_train_model()


def init_state():
    if "initialized" in st.session_state:
        return

    with st.spinner(
        "Building the Astana road graph (first run downloads from "
        "OpenStreetMap — this can take a minute)..."
    ):
        G, source = get_graph()
    with st.spinner("Training the congestion-prediction model..."):
        model = get_model()

    st.session_state.graph = G
    st.session_state.graph_source = source
    st.session_state.model = model
    st.session_state.district_nodes = graph_builder.district_nodes(G)
    with st.spinner("Indexing Astana streets for typo-tolerant search..."):
        st.session_state.street_index = graph_builder.build_street_index(G)
    st.session_state.simulator = TrafficSimulator(G, model)
    st.session_state.alerts = AlertSystem()
    st.session_state.route = None
    st.session_state.route_info = ""
    st.session_state.has_simulated = False
    first_district = list(config.DISTRICTS.keys())[0]
    st.session_state.user_location = config.DISTRICTS[first_district]
    st.session_state.loc_mode = "District"
    st.session_state.map_center = list(config.ASTANA_CENTER)
    st.session_state.map_zoom = config.DEFAULT_ZOOM
    st.session_state.last_tick = -1
    st.session_state.initialized = True


init_state()

G = st.session_state.graph
model = st.session_state.model
sim: TrafficSimulator = st.session_state.simulator
alerts: AlertSystem = st.session_state.alerts
district_nodes = st.session_state.district_nodes
street_index = st.session_state.street_index


st.sidebar.title("🚦 Controls")

if st.session_state.graph_source == "synthetic":
    st.sidebar.warning(
        "Running on a **synthetic grid** fallback (no OSM download was "
        "possible). Delete `data/astana_graph.graphml` and rerun with internet "
        "to use the real Astana network."
    )
else:
    label = "cached graph" if st.session_state.graph_source == "cache" else "OSM download"
    st.sidebar.caption(f"Road network: real Astana ({label}).")

st.sidebar.subheader("Time")
_now = astana_now()
auto_time = st.sidebar.checkbox(
    "🕒 Use current Astana time", value=True,
    help="Auto-set date & time from the live clock in Astana (UTC+5).",
)
if auto_time:
    time_of_day = _now.hour
    sim_date = _now.date()
    st.sidebar.caption(
        f"Astana now: **{config.DAYS_OF_WEEK[sim_date.weekday()]} {_now:%H:%M}** "
        "(UTC+5)"
    )
else:
    time_of_day = st.sidebar.slider("Time of day (hour)", 0, 23, 8, help="00:00–23:00")
    sim_date = st.sidebar.date_input("Date", value=_now.date())
day_of_week = sim_date.weekday()

prediction = demand.predict_demand(sim_date, time_of_day)

st.sidebar.subheader("Simulation")
st.sidebar.markdown(
    f"**Predicted demand: {prediction.level}** — ~{prediction.trips} vehicles/step"
)
st.sidebar.caption("Why: " + "; ".join(prediction.reasons) + ".")
if prediction.holiday:
    st.sidebar.caption(f"🎉 {prediction.holiday}")

with st.sidebar.expander("Advanced: override demand"):
    override = st.checkbox("Set vehicles manually", value=False)
    manual_trips = st.slider("Vehicles per step", 30, 420, prediction.trips, step=10)
n_trips = manual_trips if override else prediction.trips

live_mode = st.sidebar.toggle(
    "🔴 Live traffic", value=False,
    help="Auto-advance the simulation continuously, like a real navigator.",
)
def _fmt_interval(s):
    return f"{s // 60}m" if s >= 60 else f"{s}s"

if live_mode:
    live_interval = st.sidebar.select_slider(
        "Update every", options=[5, 15, 30, 60], value=60, format_func=_fmt_interval
    )
    if st_autorefresh is None:
        st.sidebar.warning("Install `streamlit-autorefresh` for live mode.")
else:
    live_interval = 60
simulate_clicked = st.sidebar.button(
    "▶ Simulate one step", use_container_width=True, disabled=live_mode
)

st.sidebar.subheader("Route a vehicle")
vehicle_type = st.sidebar.selectbox("Vehicle type", config.VEHICLE_TYPES)
district_names = list(config.DISTRICTS.keys())

MY_LOCATION = "📍 My current location"
from_choice = st.sidebar.selectbox("From", [MY_LOCATION] + district_names, index=0)
dest_address = st.sidebar.text_input(
    "To — any Astana street (typos OK)",
    placeholder="e.g. манггылык ел  /  mangilik el  /  abay",
    help="Typo-tolerant offline street search. Leave empty to use the district.",
)
chosen_street = None
house_number = None
if dest_address.strip():
    street_q, house_number = graph_builder.parse_address(dest_address.strip())
    street_matches = graph_builder.match_street(
        street_q or dest_address.strip(), street_index, limit=5
    )
    if street_matches:
        labels = [f"{nm}  ·  {sc * 100:.0f}%" for nm, sc in street_matches]
        pick = st.sidebar.selectbox(
            "🔎 Did you mean…", labels, index=0,
            help="The AI's best guesses for the street you typed — pick one.",
        )
        chosen_street = street_matches[labels.index(pick)][0]
        if house_number is not None:
            st.sidebar.caption(f"🏠 House number **{house_number}**")
    else:
        st.sidebar.caption("No local street match — will try online geocoding.")
dest_district = st.sidebar.selectbox("…or pick a district", district_names, index=9)
route_clicked = st.sidebar.button("🧭 Compute route", use_container_width=True)

st.sidebar.subheader("👤 Your location")
st.sidebar.caption(
    "You only see — and can report the cause of — alerts within your radius."
)
loc_mode = st.sidebar.radio(
    "Set my location by",
    ["District", "Clicking the map"],
    index=0 if st.session_state.loc_mode == "District" else 1,
    horizontal=True,
)
st.session_state.loc_mode = loc_mode
if loc_mode == "District":
    my_district = st.sidebar.selectbox("I am near", list(config.DISTRICTS.keys()), index=0)
    st.session_state.user_location = config.DISTRICTS[my_district]
else:
    st.sidebar.caption("🗺️ Click anywhere on the map to drop your position.")
visibility_radius = st.sidebar.slider(
    "Visibility radius (km)", 0.5, 10.0, 0.5, step=0.5
)
operator_view = st.sidebar.checkbox(
    "🛰️ Operator view (see all alerts)", value=False,
    help="Bypass proximity and show every alert in the city.",
)

st.sidebar.subheader("Display")
show_all_roads = st.sidebar.checkbox(
    "Show all roads (slower)", value=False,
    help="Draw every edge instead of only the major roads.",
)

st.sidebar.divider()
reset_clicked = st.sidebar.button("♻ Reset simulation", use_container_width=True)


live_advanced = False
if live_mode and st_autorefresh is not None:
    tick = st_autorefresh(interval=live_interval * 1000, key="live_tick")
    if tick != st.session_state.last_tick:
        st.session_state.last_tick = tick
        live_advanced = True


if reset_clicked:
    sim.reset()
    alerts.clear()
    st.session_state.route = None
    st.session_state.route_info = ""
    st.session_state.has_simulated = False
    st.rerun()

if simulate_clicked or live_advanced:
    event_node_ids = [
        district_nodes[name] for name in EVENT_DISTRICTS if name in district_nodes
    ]
    step_trips = n_trips
    if live_advanced and not simulate_clicked and live_interval <= 8:
        step_trips = min(n_trips, 90)
    with st.spinner("Distributing traffic across the city..."):
        stats = sim.simulate_step(
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            n_trips=step_trips,
            district_node_ids=list(district_nodes.values()),
            event_focus=prediction.event_focus,
            event_node_ids=event_node_ids,
        )
        new_alerts = alerts.detect(G, time_of_day, day_of_week)
    st.session_state.has_simulated = True
    if st.session_state.route is not None:
        r = st.session_state.route
        st.session_state.route = sim.route(r.origin, r.destination, r.vehicle_type)
    if simulate_clicked:
        msg = f"Step {sim.step_index}: {len(new_alerts)} new alert(s)."
        if hasattr(st, "toast"):
            st.toast(msg)
        else:
            st.sidebar.success(msg)

if route_clicked:
    if from_choice == MY_LOCATION:
        ulat, ulon = st.session_state.user_location
        origin_node = graph_builder.nearest_node(G, ulat, ulon)
        origin_label = "your location"
    else:
        origin_node = district_nodes[from_choice]
        origin_label = from_choice

    dest_node = None
    dest_label = ""
    geocode_failed = False
    typed = dest_address.strip()
    interpreted_note = ""
    if typed:
        if chosen_street:
            if house_number is not None:
                with st.spinner(f"Locating {chosen_street}, {house_number}…"):
                    coords = geocode_astana(f"{chosen_street} {house_number}")
                if coords:
                    dest_node = graph_builder.nearest_node(G, coords[0], coords[1])
                else:
                    dest_node = graph_builder.street_point(
                        street_index, chosen_street, house_number
                    )
                dest_label = f"{chosen_street}, {house_number}"
            else:
                dest_node = graph_builder.street_point(street_index, chosen_street)
                dest_label = chosen_street
            street_norm = graph_builder.normalize_street(chosen_street)
            if graph_builder.normalize_street(typed).replace(
                    str(house_number or ""), "").strip() not in (street_norm, ""):
                interpreted_note = f"  \n🔎 Interpreted “{typed}” as **{dest_label}**."
        else:
            with st.spinner(f"Finding “{typed}” in Astana…"):
                coords = geocode_astana(typed)
            if coords:
                dest_node = graph_builder.nearest_node(G, coords[0], coords[1])
                dest_label = typed
            else:
                geocode_failed = True
    if dest_node is None and not typed:
        dest_node = district_nodes[dest_district]
        dest_label = dest_district

    if geocode_failed:
        st.session_state.route = None
        st.session_state.route_info = (
            f"⚠ Couldn't find “{typed}”. Check the spelling or pick a district."
        )
    else:
        route = sim.route(origin_node, dest_node, vehicle_type)
        st.session_state.route = route
        if route.found:
            info = (
                f"**{vehicle_type}** — {origin_label} → **{dest_label}**: "
                f"{len(route.nodes)} nodes, ~{route.total_length/1000:.1f} km, "
                f"~{route.total_time/60:.1f} min." + interpreted_note
            )
            if route.restricted_fallback:
                info += f" ⚠ {route.message}"
            st.session_state.route_info = info
        else:
            st.session_state.route_info = f"⚠ {route.message or 'No route found.'}"


title_col, live_col = st.columns([4, 1])
with title_col:
    st.title("🧭 Astana Smart Navigator")
with live_col:
    if live_mode:
        st.markdown(
            f"<div style='text-align:right;padding-top:18px;color:#e74c3c;"
            f"font-weight:600'>🔴 LIVE · {live_interval}s</div>",
            unsafe_allow_html=True,
        )

stats = sim.current_stats(time_of_day, day_of_week)
summary = graph_builder.graph_summary(G)

_route = st.session_state.route
if _route is not None and getattr(_route, "found", False):
    eta_min = _route.total_time / 60
    km = _route.total_length / 1000
    badge = {"Emergency": "🚑", "Heavy": "🚚", "Regular": "🚗"}.get(
        _route.vehicle_type, "🚗"
    )
    e1, e2, e3 = st.columns([1.2, 1, 1])
    e1.metric(f"{badge} {_route.vehicle_type}", f"{eta_min:.0f} min")
    e2.metric("Distance", f"{km:.1f} km")
    avg_speed = (km / (eta_min / 60)) if eta_min > 0 else 0
    e3.metric("Avg speed", f"{avg_speed:.0f} km/h")
    streets = route_streets(G, _route)
    if streets:
        with st.expander(f"🛣️ Directions — {len(streets)} streets"):
            st.markdown(
                "  \n".join(f"**{i+1}.** {nm}" for i, nm in enumerate(streets[:25]))
            )

user_lat, user_lon = st.session_state.user_location
active_alerts = alerts.active_alerts()
nearby_alerts = [
    a for a in active_alerts
    if graph_builder.haversine_km(user_lat, user_lon, a.lat, a.lon) <= visibility_radius
]
visible_alerts = active_alerts if operator_view else nearby_alerts
hidden_count = 0 if operator_view else (len(active_alerts) - len(nearby_alerts))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Edges (roads)", f"{summary['edges']:,}")
c2.metric("Avg load", f"{stats['avg_load']*100:.0f}%")
c3.metric("Congested", f"{stats['pct_congested']:.1f}%",
          help="Edges with load > 70%")
c4.metric("Alerts near you", f"{len(nearby_alerts)}",
          help=f"{len(active_alerts)} active alerts in the whole city")
c5.metric("City alerts", f"{len(active_alerts)}")

if st.session_state.route_info:
    st.info(st.session_state.route_info)

if not st.session_state.has_simulated:
    st.info(
        "👈 Click **Simulate one step** to distribute traffic, then watch the "
        "map colour by congestion and alerts appear. Use **Compute route** to "
        "route an Emergency / Heavy / Regular vehicle between districts."
    )


map_col, panel_col = st.columns([3, 1.3], gap="medium")

with map_col:
    st.subheader("Live congestion map")
    if show_all_roads:
        max_edges = None
    elif live_mode:
        max_edges = 900
    else:
        max_edges = config.MAX_RENDER_EDGES
    _r = st.session_state.route
    if _r is not None and getattr(_r, "nodes", None):
        _o = _r.nodes[0]
        view_center = [G.nodes[_o]["y"], G.nodes[_o]["x"]]
    else:
        view_center = list(st.session_state.user_location)
    fmap = visualizer.build_map(
        G,
        alerts=visible_alerts,
        route=st.session_state.route,
        district_node_ids=district_nodes,
        max_edges=max_edges,
        user_location=st.session_state.user_location,
        user_radius_km=visibility_radius,
        center=view_center,
        zoom=st.session_state.map_zoom,
    )
    try:
        map_data = st_folium(
            fmap, use_container_width=True, height=600,
            returned_objects=["last_clicked"], key="astana_map",
        )
    except TypeError:
        map_data = st_folium(
            fmap, width=900, height=600,
            returned_objects=["last_clicked"], key="astana_map",
        )

    if st.session_state.loc_mode == "Clicking the map" and isinstance(map_data, dict):
        clicked = map_data.get("last_clicked")
        if clicked and "lat" in clicked and "lng" in clicked:
            new_loc = (float(clicked["lat"]), float(clicked["lng"]))
            if new_loc != tuple(st.session_state.user_location):
                st.session_state.user_location = new_loc
                st.rerun()

with panel_col:
    if operator_view:
        st.subheader("⚠ Alerts (operator view)")
    else:
        st.subheader("⚠ Alerts near you")

    if not active_alerts:
        st.caption("No active alerts. Run a simulation step to generate some.")
    elif not visible_alerts:
        st.caption(
            f"None of the {len(active_alerts)} active alert(s) are within your "
            f"{visibility_radius:.1f} km radius. Move closer (change district or "
            "click the map) to report a cause."
        )
    else:
        if operator_view:
            st.caption("Showing **all** city alerts. Classify each to report a cause.")
        else:
            st.caption(
                f"You're close enough to report on **{len(visible_alerts)}** "
                "alert(s). Pick the real cause — it updates the road and trains "
                "the model."
            )

    for a in visible_alerts:
        dist = graph_builder.haversine_km(user_lat, user_lon, a.lat, a.lon)
        with st.container(border=True):
            st.markdown(
                f"**#{a.id} · {a.road_name}**  \n"
                f"{a.road_type} · load **{a.actual_load*100:.0f}%** "
                f"vs predicted {a.predicted_load*100:.0f}% "
                f"(**{a.ratio:.1f}×**)  \n"
                f"📍 {dist:.1f} km from you"
            )
            b1, b2 = st.columns(2)
            b3, b4 = st.columns(2)
            cols = [b1, b2, b3, b4]
            for col, label in zip(cols, RESPONSE_OPTIONS):
                if col.button(label, key=f"alert_{a.id}_{label}",
                              use_container_width=True):
                    alerts.respond(a.id, label, G, model)
                    st.rerun()

    if hidden_count > 0:
        st.caption(
            f"🔕 {hidden_count} more alert(s) elsewhere in the city — not visible "
            "to you. Enable **Operator view** to see them all."
        )

    resolved = [a for a in alerts.alerts if a.status == "resolved"]
    if resolved:
        with st.expander(f"Resolved alerts ({len(resolved)})"):
            for a in resolved[-12:]:
                st.markdown(
                    f"#{a.id} · {a.road_name} → **{a.label}**"
                )
        st.caption(f"Feedback points collected: {alerts.feedback_count()}")


st.subheader("📈 Traffic flow over time")
if sim.history:
    steps = [h["step"] for h in sim.history]
    avg_load = [h["avg_load"] * 100 for h in sim.history]
    pct_cong = [h["pct_congested"] for h in sim.history]
    hover = [f"t={h['time_of_day']:02d}:00 {config.DAYS_OF_WEEK[h['day_of_week']]}"
             for h in sim.history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=avg_load, name="Avg load %", mode="lines+markers",
        line=dict(color="#2980b9", width=3), text=hover,
        hovertemplate="step %{x}<br>%{text}<br>avg load %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=pct_cong, name="Congested edges %", mode="lines+markers",
        line=dict(color="#e74c3c", width=3), text=hover,
        hovertemplate="step %{x}<br>%{text}<br>congested %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Simulation step",
        yaxis_title="Percent",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Run at least one simulation step to populate the chart.")


with st.expander("ℹ️ Model & system details"):
    m = model.train_metrics or {}
    st.markdown(
        f"""
- **Road graph:** {summary['nodes']:,} intersections · {summary['edges']:,} road segments
- **Network source:** {st.session_state.graph_source}
- **Congestion model:** RandomForestClassifier
  ({m.get('n_samples', '—')} synthetic samples,
  test accuracy {m.get('test_accuracy', float('nan')):.3f},
  positive rate {m.get('positive_rate', float('nan')):.3f})
- **Model features:** `{', '.join(config.FEATURE_COLUMNS)}`
- **Anomaly rule:** actual_load > predicted_load × {config.ANOMALY_RATIO}
  (predicted ≥ {config.ANOMALY_MIN_PREDICTED})
- **Feedback points collected:** {alerts.feedback_count()}
        """
    )
    if alerts.feedback_count() > 0 and st.button("Retrain model with feedback"):
        with st.spinner("Retraining with operator feedback..."):
            model.retrain_with_feedback()
            model.save()
        st.success("Model retrained with collected feedback points.")
