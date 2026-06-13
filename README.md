# Astana Smart Navigator

A prototype smart traffic system for Astana. It builds the real road map of the
city as a graph, predicts congestion with a small machine-learning model, and
spreads cars across the city instead of sending everyone down the same road.
It's a demo project, not a production service.

## How to run

You need Python 3.10 or newer installed.

### Easy way (one click)

- Windows: double-click `run.bat`
- macOS / Linux: run `./run.sh` in a terminal

It creates everything by itself and opens the app in your browser. The first
run takes a few minutes (it downloads the libraries and the Astana map). After
that it starts in a few seconds.

### Manual way

```
git clone https://github.com/tatrat14/redbull-smart-navigator.git
cd redbull-smart-navigator

python -m venv .venv
# Windows:        .\.venv\Scripts\Activate.ps1
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

Notes:
- On Windows, if `python` doesn't work (it opens the Microsoft Store), use `py`
  instead: `py -m venv .venv`.
- The first run needs internet to download the road network. After that it
  works offline.

## What it does

- Downloads the real road network of Astana (OpenStreetMap) and turns
  intersections and roads into a graph.
- Predicts how busy each road is with a RandomForest model.
- Distributes cars across the city so the traffic spreads out.
- Three vehicle types take different routes:
  - Emergency: fastest path, ignores jams and accidents.
  - Heavy (trucks): stays on wide main roads.
  - Regular: avoids jams and accidents.
- Guesses the traffic level from the time of day, the weekday and Kazakhstan
  holidays (Nauryz, Capital Day, and so on).
- Detects unusual jams and asks you for the reason (accident, roadworks, event).
  Your answer updates the map and trains the model.
- Shows alerts only to people near them: you set your location and a radius, and
  you only see the alerts inside it.
- Lets you route from your location to any street, even with typos
  ("туран 26", "манггылык ел"). Street search works offline and understands
  house numbers.
- Uses the current time in Astana, and has a Live mode that keeps updating the
  traffic on its own.
- Colors the map by congestion (green / yellow / red) and shows charts over time.

## Built with

Python, OSMnx, NetworkX, scikit-learn, Streamlit, Folium, Plotly, pandas, NumPy.

## Files

- `app.py` — the dashboard (start here)
- `graph_builder.py` — road graph and street search
- `ml_model.py` — congestion prediction model
- `demand.py` — traffic prediction from time and holidays
- `traffic_simulator.py` — simulation and routing
- `alert_system.py` — alerts and feedback
- `visualizer.py` — the map
- `config.py` — settings
