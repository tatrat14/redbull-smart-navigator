"""
Congestion-prediction model.

The model maps per-edge features
    [edge_capacity, road_type_encoded, time_of_day, day_of_week, current_load]
to a congestion probability in [0, 1].

Because there is no real-time feed, we generate synthetic training data from a
latent congestion function that encodes realistic time-of-day behaviour
(morning + evening peaks, lighter weekends) and the intuition that low-capacity,
heavily-loaded roads congest first. A RandomForestClassifier is trained on
Bernoulli-sampled labels; ``predict_proba`` then gives a smooth probability.

User feedback from the alert system (labelled incidents) can be folded back in
via :meth:`CongestionModel.retrain_with_feedback`.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

import config

try:
    import joblib
except Exception:
    joblib = None


def time_of_day_factor(t):
    """
    Smooth 0..1 demand factor across the day with morning (~08:00) and evening
    (~18:00) peaks plus a small midday bump. Accepts scalars or numpy arrays.
    """
    t = np.asarray(t, dtype=float)
    morning = np.exp(-((t - 8.0) ** 2) / (2 * 1.4 ** 2))
    evening = np.exp(-((t - 18.0) ** 2) / (2 * 1.6 ** 2))
    midday = 0.45 * np.exp(-((t - 13.0) ** 2) / (2 * 2.5 ** 2))
    return np.clip(morning + evening + midday, 0.0, 1.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def latent_congestion_probability(capacity, road_type_encoded, tod, dow, load):
    """
    Vectorised latent probability of congestion. This is the "true" function the
    ML model learns to approximate from noisy Bernoulli samples.
    """
    capacity = np.asarray(capacity, dtype=float)
    road_type_encoded = np.asarray(road_type_encoded, dtype=float)
    load = np.asarray(load, dtype=float)
    tod_f = time_of_day_factor(tod)
    weekend = (np.asarray(dow, dtype=float) >= 5).astype(float)
    importance = road_type_encoded / float(config.MAX_ROAD_TYPE_CODE)

    z = (
        -1.30
        + 2.30 * load
        + 1.35 * (1.0 - capacity)
        + 1.45 * tod_f
        + 0.80 * load * tod_f
        - 0.55 * weekend
        - 0.35 * importance
    )
    return _sigmoid(z)


def generate_synthetic_training_data(n_samples: int = 9000, seed: int = 42) -> pd.DataFrame:
    """Generate a labelled training set covering all times of day / week."""
    rng = np.random.default_rng(seed)

    capacity = rng.uniform(config.CAPACITY_MIN, config.CAPACITY_MAX, n_samples)
    road_type_encoded = rng.integers(0, config.MAX_ROAD_TYPE_CODE + 1, n_samples)
    time_of_day = rng.integers(0, 24, n_samples)
    day_of_week = rng.integers(0, 7, n_samples)
    current_load = rng.beta(2.0, 2.5, n_samples)

    p = latent_congestion_probability(
        capacity, road_type_encoded, time_of_day, day_of_week, current_load
    )
    congested = (rng.uniform(0, 1, n_samples) < p).astype(int)

    return pd.DataFrame(
        {
            "edge_capacity": capacity,
            "road_type_encoded": road_type_encoded,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "current_load": current_load,
            "congested": congested,
        }
    )


class CongestionModel:
    """Thin wrapper around a scikit-learn classifier with save/load helpers."""

    def __init__(self, model=None, feedback: Optional[pd.DataFrame] = None):
        self.model = model
        self.feature_columns = list(config.FEATURE_COLUMNS)
        self.feedback = feedback if feedback is not None else pd.DataFrame(
            columns=self.feature_columns + ["congested"]
        )
        self.train_metrics: dict = {}

    def fit(self, df: Optional[pd.DataFrame] = None, n_samples: int = 9000):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        if df is None:
            df = generate_synthetic_training_data(n_samples=n_samples)

        X = df[self.feature_columns].to_numpy(dtype=float)
        y = df["congested"].to_numpy(dtype=int)

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        clf = RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)
        self.model = clf
        self.train_metrics = {
            "train_accuracy": float(clf.score(X_tr, y_tr)),
            "test_accuracy": float(clf.score(X_te, y_te)),
            "n_samples": int(len(df)),
            "positive_rate": float(y.mean()),
        }
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return P(congested) for each row of a feature DataFrame/array."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        X = (
            features[self.feature_columns].to_numpy(dtype=float)
            if isinstance(features, pd.DataFrame)
            else np.asarray(features, dtype=float)
        )
        if X.size == 0:
            return np.zeros(0)
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.full(len(X), float(classes[0]))

    def predict_for_graph(self, G, time_of_day: int, day_of_week: int) -> dict:
        """
        Predict congestion probability for every edge given the current load
        already stored on the graph. Returns {(u, v, k): prob} and also writes
        ``congestion_prob`` back onto each edge.
        """
        rows = []
        keys = []
        for u, v, k, data in G.edges(keys=True, data=True):
            rows.append(
                (
                    data.get("capacity", 0.3),
                    data.get("road_type_encoded", config.DEFAULT_ROAD_TYPE_CODE),
                    time_of_day,
                    day_of_week,
                    data.get("load", 0.0),
                )
            )
            keys.append((u, v, k))
        if not rows:
            return {}
        feats = pd.DataFrame(rows, columns=self.feature_columns)
        probs = self.predict_proba(feats)
        result = {}
        for (u, v, k), p in zip(keys, probs):
            result[(u, v, k)] = float(p)
            G.edges[u, v, k]["congestion_prob"] = float(p)
        return result

    def add_feedback(self, feature_row: dict, congested: int):
        """Append one user-labelled data point (from an alert response)."""
        row = {col: feature_row.get(col) for col in self.feature_columns}
        row["congested"] = int(congested)
        new_row = pd.DataFrame([row])
        self.feedback = (
            new_row
            if self.feedback.empty
            else pd.concat([self.feedback, new_row], ignore_index=True)
        )

    def retrain_with_feedback(self, n_samples: int = 9000):
        """Retrain on synthetic data augmented with collected feedback rows."""
        base = generate_synthetic_training_data(n_samples=n_samples)
        if not self.feedback.empty:
            weighted = pd.concat([self.feedback] * 25, ignore_index=True)
            combined = pd.concat([base, weighted], ignore_index=True)
        else:
            combined = base
        return self.fit(combined)

    def save(self, path: str = config.MODEL_CACHE_PATH):
        if joblib is None:
            return
        joblib.dump(
            {
                "model": self.model,
                "feedback": self.feedback,
                "metrics": self.train_metrics,
            },
            path,
        )

    @classmethod
    def load(cls, path: str = config.MODEL_CACHE_PATH) -> Optional["CongestionModel"]:
        if joblib is None or not os.path.exists(path):
            return None
        try:
            blob = joblib.load(path)
            obj = cls(model=blob.get("model"), feedback=blob.get("feedback"))
            obj.train_metrics = blob.get("metrics", {})
            return obj
        except Exception as exc:
            print(f"[ml_model] Could not load model ({exc}).")
            return None


def load_or_train_model(path: str = config.MODEL_CACHE_PATH) -> CongestionModel:
    """Load a cached model or train (and cache) a fresh one."""
    model = CongestionModel.load(path)
    if model is not None and model.model is not None:
        return model
    model = CongestionModel().fit()
    model.save(path)
    return model
