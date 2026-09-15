"""
IRANav simulated Antarctic environment.

All icebergs, sea ice, weather, and drift values are generated for this
prototype. They are not live ice-chart, satellite, or weather data.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

RANDOM_SEED = 42

# Simulated operating box near the Antarctic Peninsula / Southern Ocean.
LAT_MIN = -68.4
LAT_MAX = -62.2
LON_MIN = -67.5
LON_MAX = -54.5

GRID_ROWS = 32
GRID_COLS = 36

DEFAULT_SHIP = {
    "name": "RV Polar Pioneer",
    "lat": -63.15,
    "lon": -65.40,
}

DEFAULT_DESTINATION = {
    "name": "Research station approach (simulated)",
    "lat": -67.05,
    "lon": -55.80,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    radius_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(min(1.0, a)))


def latlon_to_cell(lat: float, lon: float) -> Tuple[int, int]:
    row_f = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (GRID_ROWS - 1)
    col_f = (lon - LON_MIN) / (LON_MAX - LON_MIN) * (GRID_COLS - 1)
    row = int(round(min(max(row_f, 0), GRID_ROWS - 1)))
    col = int(round(min(max(col_f, 0), GRID_COLS - 1)))
    return row, col


def cell_to_latlon(row: int, col: int) -> Tuple[float, float]:
    lat = LAT_MAX - (row / (GRID_ROWS - 1)) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (col / (GRID_COLS - 1)) * (LON_MAX - LON_MIN)
    return lat, lon


def destination_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2 (0 = north)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def offset_latlon(lat: float, lon: float, distance_km: float, bearing_deg: float) -> Tuple[float, float]:
    """Move a point by a distance and compass bearing (simple sphere)."""
    radius_km = 6371.0
    br = math.radians(bearing_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    p2 = math.asin(
        math.sin(p1) * math.cos(distance_km / radius_km)
        + math.cos(p1) * math.sin(distance_km / radius_km) * math.cos(br)
    )
    l2 = l1 + math.atan2(
        math.sin(br) * math.sin(distance_km / radius_km) * math.cos(p1),
        math.cos(distance_km / radius_km) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), math.degrees(l2)


def base_weather() -> Dict:
    """Baseline simulated weather for the operating area."""
    return {
        "wind_kt": 18.0,
        "wind_dir_deg": 245.0,
        "current_kt": 0.55,
        "current_dir_deg": 95.0,
        "temperature_c": -7.5,
        "visibility_km": 8.0,
    }


def generate_icebergs() -> List[Dict]:
    """Stable set of simulated icebergs."""
    return [
        {"id": 1, "name": "IB-Alpha", "lat": -64.05, "lon": -62.80, "radius_km": 22.0, "hazard": 92},
        {"id": 2, "name": "IB-Bravo", "lat": -65.55, "lon": -60.90, "radius_km": 28.0, "hazard": 95},
        {"id": 3, "name": "IB-Charlie", "lat": -63.55, "lon": -58.20, "radius_km": 18.0, "hazard": 88},
        {"id": 4, "name": "IB-Delta", "lat": -66.85, "lon": -63.70, "radius_km": 20.0, "hazard": 90},
        {"id": 5, "name": "IB-Echo", "lat": -66.35, "lon": -57.40, "radius_km": 24.0, "hazard": 93},
        {"id": 6, "name": "IB-Foxtrot", "lat": -64.70, "lon": -56.10, "radius_km": 16.0, "hazard": 86},
    ]


def generate_sea_ice_zones() -> List[Dict]:
    """Simulated sea-ice fields with concentration in percent (0–100)."""
    return [
        {"id": 1, "name": "Packed ice field A", "lat": -64.30, "lon": -61.80, "radius_km": 95.0, "concentration": 48},
        {"id": 2, "name": "Packed ice field B", "lat": -66.00, "lon": -59.10, "radius_km": 110.0, "concentration": 62},
        {"id": 3, "name": "Drift ice belt", "lat": -63.10, "lon": -60.40, "radius_km": 85.0, "concentration": 40},
        {"id": 4, "name": "Coastal fast ice", "lat": -67.30, "lon": -62.20, "radius_km": 90.0, "concentration": 70},
    ]


def ice_pressure_field() -> np.ndarray:
    """Simulated ice-pressure / convergence field (0–100). Not a real ice model."""
    rng = np.random.default_rng(RANDOM_SEED)
    field = rng.normal(28.0, 9.0, size=(GRID_ROWS, GRID_COLS))
    for row in range(GRID_ROWS):
        field[row, :] += 12.0 * max(0.0, 1.0 - row / 8.0)
    return np.clip(field, 5.0, 85.0)


def create_base_sim() -> Dict:
    """Fresh simulated mission state used by Reset Simulation."""
    return {
        "ship": deepcopy(DEFAULT_SHIP),
        "destination": deepcopy(DEFAULT_DESTINATION),
        "icebergs": generate_icebergs(),
        "zones": generate_sea_ice_zones(),
        "weather": base_weather(),
        "pressure": ice_pressure_field(),
        "storm": None,
        "next_iceberg_id": 100,
        "event_log": [],
    }


def iceberg_along_route(path_latlon: List[Tuple[float, float]], berg_id: int) -> Dict:
    """Place a new simulated iceberg near the middle of the current route."""
    if path_latlon and len(path_latlon) >= 4:
        idx = max(1, min(len(path_latlon) - 2, int(len(path_latlon) * 0.48)))
        lat, lon = path_latlon[idx]
        # Sit on the track with a tiny offset so A* can still slip around the core.
        lat += 0.03
        lon += 0.06
    else:
        lat, lon = -65.20, -60.40
    return {
        "id": berg_id,
        "name": f"IB-NEW-{berg_id}",
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "radius_km": 34.0,
        "hazard": 98,
        "is_new": True,
    }


def make_storm(path_latlon: Optional[List[Tuple[float, float]]] = None) -> Dict:
    """Simulated storm / high-wind region, biased toward the current route."""
    if path_latlon and len(path_latlon) >= 3:
        idx = int(len(path_latlon) * 0.55)
        lat, lon = path_latlon[idx]
    else:
        lat, lon = -65.40, -59.80
    return {
        "name": "Simulated polar storm",
        "lat": lat,
        "lon": lon,
        "radius_km": 130.0,
        "wind_kt": 46.0,
        "visibility_km": 1.8,
        "extra_risk": 28.0,
    }


def predict_iceberg_track(
    berg: Dict,
    weather: Dict,
    hours: int,
    storm: Optional[Dict] = None,
) -> Dict:
    """
    Prototype iceberg trajectory prediction.

    Drift is a simple mix of simulated current and a small wind component.
    This is NOT a scientifically validated iceberg model.
    """
    wind_kt = float(weather["wind_kt"])
    wind_dir = float(weather["wind_dir_deg"])
    current_kt = float(weather["current_kt"])
    current_dir = float(weather["current_dir_deg"])

    if storm is not None:
        dist = haversine_km(berg["lat"], berg["lon"], storm["lat"], storm["lon"])
        if dist < storm["radius_km"]:
            wind_kt = max(wind_kt, float(storm["wind_kt"]))

    # Prototype only: current dominates; wind adds a small extra push.
    speed_kmh = current_kt * 1.852 * 0.75 + wind_kt * 1.852 * 0.02
    # Blend directions (current 70%, wind 30%) in a lightweight way.
    bearing = (0.70 * current_dir + 0.30 * wind_dir) % 360.0
    distance = speed_kmh * hours
    pred_lat, pred_lon = offset_latlon(berg["lat"], berg["lon"], distance, bearing)
    return {
        "hours": hours,
        "lat": pred_lat,
        "lon": pred_lon,
        "bearing_deg": bearing,
        "speed_kmh": speed_kmh,
    }
