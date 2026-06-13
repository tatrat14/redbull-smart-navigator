"""
Predictive traffic-demand model.

Instead of asking the operator for a vehicle count, we *predict* how busy the
city is from context — time of day, day of week and Kazakhstan public holidays
(plus pre-holiday "getaway" evenings). The result drives the simulation, and a
short, human-readable explanation tells the user *why* that level was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional

from ml_model import time_of_day_factor

KZ_HOLIDAYS = {
    (1, 1): "New Year",
    (1, 2): "New Year holiday",
    (1, 7): "Orthodox Christmas",
    (3, 8): "International Women's Day",
    (3, 21): "Nauryz Meyramy",
    (3, 22): "Nauryz Meyramy",
    (3, 23): "Nauryz Meyramy",
    (5, 1): "Unity Day",
    (5, 7): "Defender's Day",
    (5, 9): "Victory Day",
    (7, 6): "Capital Day (Astana)",
    (8, 30): "Constitution Day",
    (10, 25): "Republic Day",
    (12, 16): "Independence Day",
}
BIG_EVENT_HOLIDAYS = {"Nauryz Meyramy", "Capital Day (Astana)"}

BASE_TRIPS = 170
MIN_TRIPS, MAX_TRIPS = 30, 420


def get_holiday(d: date) -> Optional[str]:
    return KZ_HOLIDAYS.get((d.month, d.day))


@dataclass
class DemandPrediction:
    trips: int
    factor: float
    reasons: List[str] = field(default_factory=list)
    holiday: Optional[str] = None
    big_event: bool = False
    event_focus: float = 0.0

    @property
    def level(self) -> str:
        if self.factor >= 1.15:
            return "Very high"
        if self.factor >= 0.85:
            return "High"
        if self.factor >= 0.55:
            return "Moderate"
        return "Low"


def predict_demand(d: date, hour: int, base: int = BASE_TRIPS) -> DemandPrediction:
    """Estimate traffic demand for a given date + hour with an explanation."""
    tod = float(time_of_day_factor(hour))
    weekend = d.weekday() >= 5
    holiday = get_holiday(d)
    reasons: List[str] = []

    factor = 0.45 + 0.95 * tod
    if 7 <= hour <= 9:
        reasons.append("morning rush hour")
    elif 17 <= hour <= 19:
        reasons.append("evening rush hour")
    elif hour <= 5:
        reasons.append("overnight — very light")

    if weekend and not holiday:
        factor *= 0.72
        reasons.append("weekend — lighter commute")

    event_focus = 0.0
    big_event = False
    if holiday:
        factor *= 0.75
        reasons.append(f"public holiday: {holiday}")
        if 11 <= hour <= 22:
            factor *= 1.5
            reasons.append("holiday daytime/evening activity")
        event_focus = 0.35
        if holiday in BIG_EVENT_HOLIDAYS:
            factor *= 1.35
            big_event = True
            event_focus = 0.7
            reasons.append("major city-centre celebrations")
    else:
        tomorrow = d + timedelta(days=1)
        if get_holiday(tomorrow) and hour >= 16:
            factor *= 1.25
            reasons.append("pre-holiday getaway traffic")

    if not reasons:
        reasons.append("typical daytime traffic")

    trips = int(max(MIN_TRIPS, min(MAX_TRIPS, round(base * factor))))
    return DemandPrediction(
        trips=trips,
        factor=factor,
        reasons=reasons,
        holiday=holiday,
        big_event=big_event,
        event_focus=event_focus,
    )
