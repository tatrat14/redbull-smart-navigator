"""
Anomaly detection and the alert / feedback loop.

Detection rule (per spec): an edge is anomalous when
    actual_load > predicted_load * ANOMALY_RATIO   (default 1.4)
and the predicted load is above a small floor (so near-empty roads don't fire).

When the operator responds to an alert (Accident / Road works / Public event /
Unknown):
  * the labelled data point is fed back into the ML model, and
  * the edge weight / capacity is updated in real time (incident penalty),
    which immediately changes routing and the map.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config

EdgeKey = Tuple[int, int, int]

# Operator response labels and their real-time impact on the edge.
#   capacity_mult   -> multiply the edge's capacity (lower = worse road)
#   incident_factor -> multiply routing cost (higher = stronger avoidance)
#   congested_label -> label fed back to the ML model (1 = congested)
RESPONSE_IMPACT = {
    "Accident": {"capacity_mult": 0.30, "incident_factor": 6.0, "congested_label": 1},
    "Road works": {"capacity_mult": 0.55, "incident_factor": 3.0, "congested_label": 1},
    "Public event": {"capacity_mult": 0.65, "incident_factor": 2.2, "congested_label": 1},
    "Unknown": {"capacity_mult": 0.85, "incident_factor": 1.4, "congested_label": 1},
}
RESPONSE_OPTIONS = list(RESPONSE_IMPACT.keys())


@dataclass
class Alert:
    id: int
    u: int
    v: int
    key: int
    lat: float
    lon: float
    road_name: str
    road_type: str
    actual_load: float
    predicted_load: float
    ratio: float
    time_of_day: int
    day_of_week: int
    created_at: float = field(default_factory=time.time)
    status: str = "active"          # "active" | "resolved"
    label: Optional[str] = None

    @property
    def edge(self) -> EdgeKey:
        return (self.u, self.v, self.key)


class AlertSystem:
    def __init__(self):
        self.alerts: List[Alert] = []
        self._next_id = 1
        # edges that already have an active alert (avoid duplicates)
        self._active_edges: set = set()

    # ------------------------------------------------------------------ #
    def _edge_midpoint(self, G, u, v) -> Tuple[float, float]:
        lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2.0
        lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2.0
        return lat, lon

    def _road_name(self, data: dict) -> str:
        name = data.get("name")
        if isinstance(name, (list, tuple)):
            name = name[0] if name else None
        return str(name) if name else "(unnamed road)"

    def detect(self, G, time_of_day: int, day_of_week: int,
               ratio_threshold: float = config.ANOMALY_RATIO,
               max_new: int = config.MAX_NEW_ALERTS_PER_STEP) -> List[Alert]:
        """Scan the graph and create alerts for the most anomalous edges.

        Many edges can exceed the threshold during a peak; we keep only the
        ``max_new`` worst (by load/predicted ratio) so the operator panel stays
        focused on the severe anomalies.
        """
        # 1. Collect candidates (edge + ratio) not already alerted.
        candidates = []
        for u, v, k, data in G.edges(keys=True, data=True):
            predicted = data.get("predicted_load", 0.0)
            actual = data.get("load", 0.0)
            if predicted < config.ANOMALY_MIN_PREDICTED:
                continue
            if actual <= predicted * ratio_threshold:
                continue
            if (u, v, k) in self._active_edges:
                continue
            candidates.append((actual / max(predicted, 1e-6), u, v, k, actual, predicted))

        # 2. Keep the worst ``max_new`` by ratio.
        candidates.sort(key=lambda c: c[0], reverse=True)
        candidates = candidates[:max_new]

        # 3. Materialise alerts.
        new_alerts: List[Alert] = []
        for ratio, u, v, k, actual, predicted in candidates:
            data = G.edges[u, v, k]
            lat, lon = self._edge_midpoint(G, u, v)
            alert = Alert(
                id=self._next_id,
                u=u, v=v, key=k,
                lat=lat, lon=lon,
                road_name=self._road_name(data),
                road_type=data.get("road_type", "unknown"),
                actual_load=float(actual),
                predicted_load=float(predicted),
                ratio=float(ratio),
                time_of_day=time_of_day,
                day_of_week=day_of_week,
            )
            self.alerts.append(alert)
            self._active_edges.add((u, v, k))
            new_alerts.append(alert)
            self._next_id += 1
        return new_alerts

    # ------------------------------------------------------------------ #
    def active_alerts(self) -> List[Alert]:
        return [a for a in self.alerts if a.status == "active"]

    def get(self, alert_id: int) -> Optional[Alert]:
        for a in self.alerts:
            if a.id == alert_id:
                return a
        return None

    def respond(self, alert_id: int, label: str, G, model) -> Optional[Alert]:
        """
        Apply an operator response to an alert:
          * mark it resolved + store the label,
          * update the edge capacity / incident penalty in real time,
          * feed the labelled data point back into the ML model.
        """
        alert = self.get(alert_id)
        if alert is None or alert.status != "active":
            return None
        impact = RESPONSE_IMPACT.get(label, RESPONSE_IMPACT["Unknown"])

        alert.label = label
        alert.status = "resolved"
        self._active_edges.discard(alert.edge)

        # --- real-time edge update ------------------------------------- #
        if G.has_edge(alert.u, alert.v, alert.key):
            data = G.edges[alert.u, alert.v, alert.key]
            base_cap = data.get("base_capacity", data.get("capacity", 0.3))
            data["capacity"] = float(base_cap * impact["capacity_mult"])
            data["incident"] = label
            data["incident_factor"] = float(impact["incident_factor"])
            # Reflect the disruption in the live load too.
            data["load"] = float(min(1.0, max(data.get("load", 0.0), 0.85)))

            # --- feedback to the ML model ------------------------------ #
            feature_row = {
                "edge_capacity": base_cap,
                "road_type_encoded": data.get(
                    "road_type_encoded", config.DEFAULT_ROAD_TYPE_CODE
                ),
                "time_of_day": alert.time_of_day,
                "day_of_week": alert.day_of_week,
                "current_load": data.get("load", 0.9),
            }
            if model is not None:
                model.add_feedback(feature_row, impact["congested_label"])

        return alert

    def feedback_count(self) -> int:
        return sum(1 for a in self.alerts if a.label is not None)

    def clear(self):
        self.alerts.clear()
        self._active_edges.clear()
        self._next_id = 1
