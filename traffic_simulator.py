"""
Traffic-flow simulation and vehicle routing.

Core ideas
----------
* Every edge carries a runtime ``load`` (0..1 fraction of capacity), a
  ``predicted_load`` baseline (time-of-day expectation) and a model-derived
  ``congestion_prob``. These live directly on the graph edge attributes so the
  router, visualiser and alert system all read the same state.

* A simulation *step* (a) sets the baseline expected load for the chosen time of
  day, (b) routes many synthetic vehicle trips using a **congestion-aware,
  incremental assignment** so that successive vehicles spread across the city
  instead of piling onto one corridor, (c) injects a few anomalies so the alert
  system has something to detect, and (d) asks the ML model for a congestion
  probability per edge.

* Three vehicle categories route differently:
    - Emergency: fastest free-flow path, ignores congestion (priority).
    - Heavy: arterial roads only (no residential / service), avoids incidents.
    - Regular: congestion-aware path that load-balances across the network.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

import config
from ml_model import CongestionModel, time_of_day_factor

EdgeKey = Tuple[int, int, int]


@dataclass
class RouteResult:
    vehicle_type: str
    origin: int
    destination: int
    nodes: List[int] = field(default_factory=list)
    edges: List[EdgeKey] = field(default_factory=list)
    total_time: float = 0.0
    total_length: float = 0.0
    found: bool = False
    restricted_fallback: bool = False
    message: str = ""


class TrafficSimulator:
    def __init__(self, G: nx.MultiDiGraph, model: CongestionModel, seed: int = 7):
        self.G = G
        self.model = model
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.step_index = 0
        self.history: List[dict] = []
        self._nodes = list(G.nodes())

    def _baseline_load(self, data: dict, tod: int, dow: int) -> float:
        tod_f = float(time_of_day_factor(tod))
        importance = data.get("road_type_encoded", 0) / float(config.MAX_ROAD_TYPE_CODE)
        weekend = 1.0 if dow >= 5 else 0.0
        base = (
            0.10
            + 0.55 * tod_f * (0.35 + 0.65 * importance)
            - 0.12 * weekend
        )
        return float(min(0.92, max(0.03, base)))

    def _single_edge_cost(self, data: dict, vehicle_type: str) -> Optional[float]:
        """Cost of traversing one edge for a given vehicle type, or None if
        the edge is not usable by that vehicle.

        The three classes weight the network *structurally* differently, so they
        pick genuinely different roads:
          * Emergency  -> pure fastest free-flow path (ignores load & class).
          * Heavy      -> widest/highest-capacity arterials only.
          * Regular    -> congestion-aware, detours around the loaded corridor.
        """
        t0 = data.get("free_flow_time", 1.0)
        load = data.get("load", 0.0)
        cong = data.get("congestion_prob", 0.0)
        cap = data.get("capacity", 0.3)
        incident_factor = data.get("incident_factor", 1.0)
        bpr = 1.0 + 0.9 * (load ** 2)

        if vehicle_type == config.VEHICLE_EMERGENCY:
            return t0

        if vehicle_type == config.VEHICLE_HEAVY:
            road_type = data.get("road_type", "residential")
            lanes = data.get("lanes_num", 1)
            penalty = 1.0
            if road_type not in config.HEAVY_ALLOWED_ROAD_TYPES:
                penalty *= 6.0
            if lanes < config.HEAVY_MIN_LANES:
                penalty *= 2.5
            class_pref = 1.0 + 1.3 * (1.0 - cap)
            return t0 * penalty * class_pref * (1.0 + 0.4 * load) * incident_factor

        return t0 * bpr * (1.0 + 1.6 * cong) * incident_factor

    def _weight_function(self, vehicle_type: str) -> Callable:
        """Return a networkx-compatible weight callable for a MultiDiGraph."""

        def weight(u, v, multi_data):
            best = None
            for data in multi_data.values():
                c = self._single_edge_cost(data, vehicle_type)
                if c is None:
                    continue
                if best is None or c < best:
                    best = c
            return best

        return weight

    def _path_edges(self, nodes: List[int], vehicle_type: str) -> Tuple[List[EdgeKey], float, float]:
        """Resolve the concrete (u, v, key) edges and totals for a node path."""
        edges: List[EdgeKey] = []
        total_time = 0.0
        total_length = 0.0
        for u, v in zip(nodes[:-1], nodes[1:]):
            best_key, best_cost, best_data = None, float("inf"), None
            for k, data in self.G[u][v].items():
                c = self._single_edge_cost(data, vehicle_type)
                if c is None:
                    continue
                if c < best_cost:
                    best_cost, best_key, best_data = c, k, data
            if best_key is None:
                k = next(iter(self.G[u][v]))
                best_key, best_data = k, self.G[u][v][k]
                best_cost = best_data.get("free_flow_time", 1.0)
            edges.append((u, v, best_key))
            total_time += best_cost
            total_length += best_data.get("length_m", 0.0)
        return edges, total_time, total_length

    def route(
        self,
        origin: int,
        destination: int,
        vehicle_type: str = config.VEHICLE_REGULAR,
    ) -> RouteResult:
        result = RouteResult(
            vehicle_type=vehicle_type, origin=origin, destination=destination
        )
        if origin == destination:
            result.message = "Origin and destination are the same."
            return result

        weight = self._weight_function(vehicle_type)
        try:
            nodes = nx.shortest_path(
                self.G, origin, destination, weight=weight, method="dijkstra"
            )
            result.nodes = nodes
            result.edges, result.total_time, result.total_length = self._path_edges(
                nodes, vehicle_type
            )
            result.found = True
        except nx.NetworkXNoPath:
            if vehicle_type == config.VEHICLE_HEAVY:
                fallback = self.route(origin, destination, config.VEHICLE_REGULAR)
                fallback.vehicle_type = vehicle_type
                fallback.restricted_fallback = True
                fallback.message = (
                    "No arterial-only path available; showing a regular route."
                )
                return fallback
            result.message = "No path found between the selected points."
        except nx.NodeNotFound:
            result.message = "Origin or destination node not in graph."
        return result

    def _regular_cost(self, data: dict) -> float:
        """Scalar congestion-aware cost of an edge for regular vehicles. Stored
        on the edge as ``reg_cost`` so trip assignment can use fast string-keyed
        Dijkstra instead of a slow python callable on the full city graph."""
        t0 = data.get("free_flow_time", 1.0)
        load = data.get("load", 0.0)
        cong = data.get("congestion_prob", 0.0)
        incident_factor = data.get("incident_factor", 1.0)
        return t0 * (1.0 + 0.9 * (load ** 2)) * (1.0 + 1.6 * cong) * incident_factor

    def _assign_trips(self, n_trips: int, district_node_ids: List[int],
                      event_focus: float = 0.0, event_nodes: Optional[List[int]] = None):
        """
        Route many synthetic trips with incremental, congestion-aware
        assignment. Each routed trip raises the load on its edges and refreshes
        their cost, so later trips are nudged onto alternative roads -> traffic
        spreads out across the city.

        ``event_focus`` (0..1) pulls a share of destinations toward
        ``event_nodes`` (e.g. the city centre on a holiday), concentrating load
        there the way a public celebration would.
        """
        anchors = district_node_ids or self._nodes
        event_nodes = event_nodes or []

        for _, _, data in self.G.edges(data=True):
            data["reg_cost"] = self._regular_cost(data)

        for _ in range(n_trips):
            if anchors and self.rng.random() < 0.6:
                origin = self.rng.choice(anchors)
            else:
                origin = self.rng.choice(self._nodes)
            if event_nodes and self.rng.random() < event_focus:
                dest = self.rng.choice(event_nodes)
            elif anchors and self.rng.random() < 0.6:
                dest = self.rng.choice(anchors)
            else:
                dest = self.rng.choice(self._nodes)
            if origin == dest:
                continue
            try:
                nodes = nx.shortest_path(
                    self.G, origin, dest, weight="reg_cost", method="dijkstra"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            for u, v in zip(nodes[:-1], nodes[1:]):
                best_key, best_cost = None, float("inf")
                for k, data in self.G[u][v].items():
                    c = data.get("reg_cost", float("inf"))
                    if c < best_cost:
                        best_cost, best_key = c, k
                if best_key is None:
                    continue
                data = self.G[u][v][best_key]
                inc = 1.0 / max(data.get("veh_capacity", 30.0), 5.0)
                data["load"] = float(min(1.0, data.get("load", 0.0) + inc))
                data["reg_cost"] = self._regular_cost(data)

    def _inject_anomalies(self, n_edges: int) -> List[EdgeKey]:
        """Spike a few capable edges above their predicted load to create
        detectable anomalies (accidents / events the system hasn't 'planned')."""
        candidates = [
            (u, v, k)
            for u, v, k, d in self.G.edges(keys=True, data=True)
            if d.get("capacity", 0) > 0.4
            and d.get("predicted_load", 0) > config.ANOMALY_MIN_PREDICTED
            and d.get("incident") is None
        ]
        if not candidates:
            return []
        chosen = self.rng.sample(candidates, min(n_edges, len(candidates)))
        spiked = []
        for u, v, k in chosen:
            data = self.G.edges[u, v, k]
            spike = self.rng.uniform(1.55, 2.4)
            spiked_load = data.get("predicted_load", 0.2) * spike
            data["load"] = float(min(1.0, max(data.get("load", 0.0), spiked_load)))
            spiked.append((u, v, k))
        return spiked

    def simulate_step(
        self,
        time_of_day: int,
        day_of_week: int,
        n_trips: int = config.DEFAULT_TRIPS_PER_STEP,
        anomaly_edges: int = config.ANOMALY_EDGES_PER_STEP,
        district_node_ids: Optional[List[int]] = None,
        event_focus: float = 0.0,
        event_node_ids: Optional[List[int]] = None,
    ) -> dict:
        """Advance the simulation by one time-step and return a summary dict."""
        self.step_index += 1

        for u, v, k, data in self.G.edges(keys=True, data=True):
            baseline = self._baseline_load(data, time_of_day, day_of_week)
            data["predicted_load"] = baseline
            noise = float(self.np_rng.normal(0, 0.04))
            data["load"] = float(min(1.0, max(0.0, baseline + noise)))

        self._assign_trips(n_trips, district_node_ids or [],
                           event_focus=event_focus, event_nodes=event_node_ids)

        spiked = self._inject_anomalies(anomaly_edges)

        self.model.predict_for_graph(self.G, time_of_day, day_of_week)

        stats = self.current_stats(time_of_day, day_of_week)
        stats["anomaly_edges"] = spiked
        self.history.append(
            {
                "step": self.step_index,
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "avg_load": stats["avg_load"],
                "pct_congested": stats["pct_congested"],
                "pct_moderate": stats["pct_moderate"],
                "pct_free": stats["pct_free"],
            }
        )
        return stats

    def current_stats(self, time_of_day: int, day_of_week: int) -> dict:
        loads = np.array(
            [d.get("load", 0.0) for _, _, d in self.G.edges(data=True)]
        )
        total = len(loads)
        if total == 0:
            return {
                "total_edges": 0,
                "avg_load": 0.0,
                "pct_congested": 0.0,
                "pct_moderate": 0.0,
                "pct_free": 0.0,
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
            }
        congested = int(np.sum(loads > config.LOAD_MODERATE_MAX))
        moderate = int(
            np.sum((loads > config.LOAD_FREE_MAX) & (loads <= config.LOAD_MODERATE_MAX))
        )
        free = total - congested - moderate
        return {
            "total_edges": total,
            "avg_load": float(np.mean(loads)),
            "pct_congested": 100.0 * congested / total,
            "pct_moderate": 100.0 * moderate / total,
            "pct_free": 100.0 * free / total,
            "n_congested": congested,
            "n_moderate": moderate,
            "n_free": free,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
        }

    def reset(self):
        """Clear all runtime load state and history."""
        self.step_index = 0
        self.history.clear()
        for _, _, data in self.G.edges(data=True):
            data["load"] = 0.0
            data["predicted_load"] = 0.0
            data["congestion_prob"] = 0.0
            data["incident"] = None
            data["incident_factor"] = 1.0
            data["capacity"] = data.get("base_capacity", data.get("capacity", 0.3))
