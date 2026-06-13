# 🚦 Astana AI Traffic Distribution System

A working prototype of an **AI-powered urban traffic-distribution system** for
**Astana, Kazakhstan**. It models the real road network as a graph, uses machine
learning to predict congestion, and **distributes vehicles evenly across the
city** instead of funnelling everything onto one corridor. It detects traffic
anomalies in real time and lets an operator classify them — feeding the response
straight back into the model.

The full demo flow:

```
graph construction → ML prediction → traffic distribution → anomaly alert → map visualization
```

---

## ✨ Features

### 1. Road graph construction (`graph_builder.py`)
- Downloads the **real Astana road network** with **OSMnx** (`graph_from_place`).
- Builds a **NetworkX** `MultiDiGraph`: nodes = intersections, edges = roads.
- Each edge gets a **capacity score (0–1)** computed from:
  - number of lanes (from OSM, with sensible per-road-type defaults),
  - road type (motorway / primary / residential / …),
  - speed limit.
- The graph is **cached to `data/astana_graph.graphml`** so it never
  re-downloads.
- If OSM is unreachable, it falls back to a **synthetic grid** so the rest of
  the pipeline still runs (a banner warns you when this happens).

### 2. Traffic flow simulation (`traffic_simulator.py` + `demand.py`)
- **Predicted demand, not a manual number.** Each step's traffic volume is
  estimated from **time of day, day of week and Kazakhstan public holidays**
  (Nauryz, Capital/Astana Day, etc.), including pre-holiday "getaway" evenings.
  On big-event holidays trips concentrate on the city centre. The sidebar shows
  the predicted level (Low → Very high) **and why** (e.g. *"morning rush hour"*,
  *"public holiday: Capital Day — major city-centre celebrations"*). A manual
  override is tucked in an expander.
- Distributes synthetic vehicles across the graph with **congestion-aware,
  incremental assignment** — each routed trip raises edge load and reroutes
  later trips, so traffic **spreads out** instead of piling up.
- Three vehicle categories that take **genuinely different roads** (their edge
  costs weight the network structurally differently):
  - **Emergency** (ambulance/fire) — absolute-priority "green wave": fastest
    free-flow path, **ignores congestion *and* incidents** (drives straight
    through jams and accidents).
  - **Heavy** (trucks) — strongly prefers the **widest, highest-capacity
    arterials**; only dips onto a small street for an unavoidable last-mile
    connector. Reliably routes differently from cars.
  - **Regular** — congestion-aware and incident-aware: **detours around** busy
    corridors and reported accidents to balance load.
- Routing endpoints: **from your current location** (or a district) **to any
  Astana street + house number** — with **typo-tolerant, offline street
  search**: type *"манггылык ел"*, *"mangilik el"*, *"abay"* or *"туран 26"* and
  the AI matches the real street (*Мәңгілік Ел даңғылы*, *Абай даңғылы*,
  *Тұран даңғылы* …) across Cyrillic / Latin / Kazakh spelling and small typos,
  shows *"Did you mean…"* suggestions, and routes to the **specific house
  number** (exact building via geocoding, or interpolated along the street
  offline). Online geocoding (OSMnx/Nominatim) is only a fallback.

> **Seeing the difference:** on an evenly-balanced network the car and the
> ambulance may share the direct road (only their *time* differs — that's the
> whole point of even distribution). Report an **Accident** on the route, or
> route in the dense centre, and **Regular visibly detours while Emergency
> drives through** and **Heavy hugs the avenues**.

### 3. ML congestion prediction (`ml_model.py`)
- A **scikit-learn `RandomForestClassifier`** trained on **synthetic traffic
  patterns** (morning/evening peaks, lighter weekends).
- Input features: `[edge_capacity, road_type_encoded, time_of_day,
  day_of_week, current_load]` → output **congestion probability (0–1)**.
- The trained model is cached to `data/congestion_model.joblib`.

### 4. Anomaly detection & alerts (`alert_system.py`)
- Flags an edge when **`actual_load > predicted_load × 1.4`**.
- Each alert appears in the UI with quick-response buttons:
  **Accident · Road works · Public event · Unknown**.
- A response (a) **updates edge weights/capacity in real time** (routes change
  immediately) and (b) **feeds a labelled data point back into the model**.

### 5. Interactive map (`visualizer.py`)
- **Folium** map of Astana, edges **colour-coded by congestion**:
  - 🟢 green `< 40%` · 🟡 yellow `40–70%` · 🔴 red `> 70%` · 🟣 incident.
- Active alerts shown as map markers; the selected vehicle route is highlighted.

### 6. Navigator-style Streamlit dashboard (`app.py`)
- **🔴 Live traffic** toggle — auto-advances the simulation every 2–10 s so the
  roads "breathe" like a real navigator; the route's ETA updates live and the
  map keeps your pan/zoom (it doesn't jump back).
- **Route ETA card** — big **time / distance / average speed** readout per
  vehicle, plus a **🛣️ street-by-street directions** list.
- **Sidebar:**
  - **🕒 Auto Astana time** — date & time default to the live Astana clock
    (UTC+5); untick to set them manually (with a date picker).
  - **Simulation** — *predicted demand* (see §2) + simulate button + Live.
  - **Route** — vehicle type, **From** (📍 your location / district),
    **To** (any typed street + house number, or a district).
  - **👤 Your location** — set by district *or by clicking the map*, plus a
    **visibility radius**; you only see/report alerts within it.
  - Operator view, show-all-roads, reset.
- **Main area:** ETA card, embedded Folium map (with your position + radius),
  congestion metrics, **proximity-filtered** active-alerts panel with response
  buttons, and a **Plotly** traffic-flow-over-time chart.

### 7. Location-based alerts
- Alerts are shown **only to users near them** (within the visibility radius),
  so the person who reports the cause is actually on the spot. Each card shows
  the distance to you; alerts elsewhere are hidden (with a count). Toggle
  **Operator view** to see the whole city.

---

## 🗂 Project structure

```
astana_traffic/
├── app.py                 # Streamlit dashboard (entry point)
├── config.py              # Central config: encodings, weights, districts
├── graph_builder.py       # OSMnx download + graph + edge weights + street search
├── ml_model.py            # Congestion model: synthetic data, train, predict
├── demand.py              # Predicted traffic demand (time + holidays)
├── traffic_simulator.py   # Traffic distribution + vehicle routing
├── alert_system.py        # Anomaly detection + alert/feedback loop
├── visualizer.py          # Folium map rendering
├── data/
│   ├── astana_graph.graphml     # (created on first run)
│   └── congestion_model.joblib  # (created on first run)
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & run

> Requires **Python 3.10+**. The first run needs **internet** to download the
> road network; every run afterwards works **fully offline**.

### 0. Get the code
```bash
git clone https://github.com/tatrat14/redbull-smart-navigator.git
cd redbull-smart-navigator
```

### 1. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> On Windows the plain `python` command can be a broken Microsoft Store stub. If
> `python --version` doesn't print a version, use the launcher: `py -m venv .venv`.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```
Your browser opens at <http://localhost:8501>.

### 4. Try the demo flow
1. Time defaults to **live Astana time** (UTC+5). Click **▶ Simulate one step**.
2. Watch the map colour by congestion. Set **👤 Your location** (district or
   click the map) — only **alerts near you** appear, with their distance.
3. **Classify a nearby alert** (Accident / Road works / …) — the road weight and
   capacity update live, and the data point is fed back to the model.
4. Route a vehicle: **From 📍 your location → To** a street typed *with typos
   and a house number* (e.g. *туран 26*) — pick the AI's **"Did you mean…"**
   guess — or a district. The **ETA card** shows time / distance / speed and the
   street-by-street directions. Switch the **vehicle type** and recompute:
   - **Heavy** hugs the wide avenues — a clearly different path.
   - Put an **Accident** on the route, then recompute **Regular** vs
     **Emergency**: Regular detours, Emergency drives straight through.
5. Flip on **🔴 Live traffic** to watch the city update continuously and the
   **traffic-flow chart** fill in — or click **Simulate one step** manually.

---

## ⚙️ Configuration

Everything tunable lives in [`config.py`](config.py):

| Setting | Meaning |
|---|---|
| `NETWORK_MODE` | `"place"` (full city, default) or `"point"` (central radius, faster) |
| `POINT_RADIUS_M` | radius used when `NETWORK_MODE="point"` |
| `ANOMALY_RATIO` | anomaly threshold (default `1.4`) |
| `LOAD_FREE_MAX` / `LOAD_MODERATE_MAX` | congestion colour thresholds |
| `MAX_RENDER_EDGES` | how many edges to draw (map responsiveness) |
| `DEFAULT_TRIPS_PER_STEP` | vehicles routed per simulation step |
| `DISTRICTS` | the selectable origin/destination points |

**Tip:** the full Astana graph is large. If the map or a simulation step feels
slow, set `NETWORK_MODE = "point"` in `config.py` (delete
`data/astana_graph.graphml` first to force a rebuild), or lower **Vehicles per
step** in the sidebar.

---

## 🧯 Troubleshooting

- **First run is slow / "Building the Astana road graph…"** — it's downloading
  the city from OpenStreetMap. This happens **once**; the graph is then cached.
- **No internet on first run** — the app falls back to a synthetic grid and
  shows a warning. Once you have internet, delete
  `data/astana_graph.graphml` and rerun to fetch the real network.
- **`geopandas` / `osmnx` install fails on Windows** — the geospatial stack
  (GDAL/Fiona/Shapely) can be fiddly via `pip`. The easiest fix is conda:
  ```bash
  conda create -n astana python=3.11
  conda activate astana
  conda install -c conda-forge osmnx geopandas
  pip install streamlit streamlit-folium plotly scikit-learn
  ```
- **Map not showing** — ensure `streamlit-folium` installed correctly
  (`pip install -r requirements.txt`).
- **Rebuild from scratch** — delete the files in `data/` and restart.

---

## 🧠 How the "even distribution" works

Classic routing sends every vehicle down the single shortest path, which is
exactly what *creates* congestion. Here, each routed trip **increases the cost**
of the edges it uses (a BPR-style load penalty plus the ML congestion
probability). The next trip therefore sees those roads as more expensive and
naturally chooses an alternative — an **incremental traffic-assignment** that
balances load across the whole network. Emergency vehicles are exempt (they get
priority), and heavy vehicles are constrained to the arterial network.

---

## 📌 Notes & limitations

- Traffic volumes are **synthetic** (no live feed) but follow realistic
  time-of-day patterns — this is a prototype to demonstrate the architecture.
- Anomalies are injected each step so alerts are easy to see in a demo.
- The model is intentionally lightweight (RandomForest) for fast, offline retrain.
